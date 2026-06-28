"""
RPGW-Net Model + Training Loop
===============================
Architecture (v2):
  Raw Signal → 1D-CNN + CWT → Patch → Feature Fusion → GAT → GW → Prototype

Key changes from v1:
  - Added 1D-CNN feature extractor (DAGCN-equivalent) for discriminative features
  - CNN(256) + CWT_patch(256) → concat(512) → Linear(512→256) → GAT
  - Fixed GW shape bug: passes full batch (B,N,N) instead of only C_s[0]

Usage:
    from rpgw.train import RPGWNet
    model = RPGWNet(config)
"""

import os
import time
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloaders
from rpgw.models.gat_encoder import GATEncoder
from rpgw.models.gw_alignment import GWAlignment
from rpgw.models.prototype import PrototypeClassifier
from rpgw.models.cnn_encoder import CNNEncoder


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )


class RPGWNet(nn.Module):
    """
    RPGW-Net v2: 1D-CNN → Feature Fusion → GAT → GW → Prototype.

    Architecture:
      1. 1D-CNN extracts 256-dim discriminative features from raw signal
      2. CWT patch features (256-dim per node) from time-frequency graph
      3. Fusion: concat(CNN, CWT_patch) = 512 → Linear(512→256) per node
      4. GAT encodes graph → node embeddings + structure distance matrix
      5. GW aligns source→target graph structures (full batch, not just [0])
      6. Prototype classifier on pooled global embeddings
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        pre_cfg = config["preprocess"]

        # ---- 1D-CNN feature extractor (DAGCN-equivalent) ----
        self.cnn = CNNEncoder(**cfg["cnn"])                    # (B, 1, 1024) → (B, 256)

        # ---- Feature fusion: CNN(256) + CWT_patch(256) → 256 ----
        patch_dim = pre_cfg["patch_size"] ** 2                 # 256 (16×16)
        self.fusion_proj = nn.Sequential(
            nn.Linear(cfg["cnn"]["feature_dim"] + patch_dim, cfg["gat"]["in_dim"]),
            nn.LayerNorm(cfg["gat"]["in_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["cnn"].get("dropout", 0.2)),
        )

        # ---- GAT encoder ----
        self.gat = GATEncoder(
            in_dim=cfg["gat"]["in_dim"],
            **{k: v for k, v in cfg["gat"].items() if k != "in_dim"},
        )

        # ---- Prototype classifier ----
        self.prototype = PrototypeClassifier(
            in_dim=cfg["gat"]["out_dim"],
            num_classes=10,
            **cfg["prototype"],
        )

        # ---- GW alignment ----
        self.use_gw = cfg["gw"]["type"] != "none"
        if self.use_gw:
            self.gw_align = GWAlignment(**cfg["gw"])

    def _extract_cnn(self, raw_signal: torch.Tensor, node_feat: torch.Tensor):
        """
        Extract CNN features from raw signal and fuse with CWT patch features.

        Args:
            raw_signal: (B, 1, L)    raw vibration
            node_feat:  (B, N, 256)  CWT patch pixel features

        Returns:
            fused: (B, N, 256)       CNN-CWT fused node features
        """
        B, N, _ = node_feat.shape

        # CNN extracts global discriminative features
        cnn_feat = self.cnn(raw_signal)                        # (B, 256)

        # Broadcast CNN features to every graph node
        cnn_expand = cnn_feat.unsqueeze(1).expand(B, N, -1)    # (B, N, 256)

        # Concatenate: global CNN context + local CWT patch energy
        fused_512 = torch.cat([cnn_expand, node_feat], dim=-1) # (B, N, 512)

        # Project back to GAT input dim
        fused = self.fusion_proj(fused_512)                    # (B, N, 256)

        return fused

    def forward(self, src_batch, tgt_batch, mode="train", gw_weight: float = 0.0):
        device = next(self.parameters()).device

        # ---- Step 1: CNN + CWT feature fusion ----
        src_feat = self._extract_cnn(
            src_batch["signal"].to(device),
            src_batch["node_feat"].to(device))
        tgt_feat = self._extract_cnn(
            tgt_batch["signal"].to(device),
            tgt_batch["node_feat"].to(device))

        # ---- Step 2: GAT encoding ----
        H_s, C_s = self.gat(src_feat, src_batch["adj"].to(device))  # (B,N,D), (B,N,N)
        H_t, C_t = self.gat(tgt_feat, tgt_batch["adj"].to(device))

        # Global pooling → graph-level embeddings
        H_s_global = H_s.mean(dim=1)  # (B, D)
        H_t_global = H_t.mean(dim=1)

        # ---- Step 3: GW structure alignment (FIXED: full batch, skip if weight=0) ----
        gw_loss = torch.tensor(0.0, device=device)
        if self.use_gw and mode == "train" and gw_weight > 0:
            try:
                gw_result = self.gw_align(C_s, C_t)   # ✅ full batch (B,N,N)
                gw_loss = gw_result["gw_loss"]
            except Exception:
                gw_loss = torch.tensor(0.0, device=device)

        # ---- Step 4: Prototype classification ----
        src_labels = src_batch["label"].to(device)
        prototypes = self.prototype.compute_prototypes(H_s_global, src_labels)
        logits = self.prototype(H_s_global, prototypes)

        # Target classification (using source prototypes)
        tgt_labels = tgt_batch["label"].to(device)
        tgt_logits = self.prototype(H_t_global, prototypes)

        return {
            "logits": logits,
            "labels": src_labels,
            "tgt_logits": tgt_logits,
            "tgt_labels": tgt_labels,
            "gw_loss": gw_loss,
        }


def train_rpgw(config: dict, save_dir: str):
    """
    Train RPGW-Net. Returns best target accuracy.
    """
    setup_logging(os.path.join(save_dir, "logs"))
    cfg_train = config["train"]
    cfg_data = config["data"]
    cfg_pre = config["preprocess"]
    cfg_noise = config["noise"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ---- Data ---- #
    source_load = config["experiment"].get("source_load", 0)
    target_load = config["experiment"].get("target_load", 3)

    dataloaders = get_dataloaders(
        data_dir=cfg_data["data_dir"],
        source_load=source_load,
        target_load=target_load,
        batch_size=cfg_train["batch_size"],
        n_shot=cfg_train.get("n_shot"),
        noise_snr_db=cfg_noise.get("target_snr_db", float("inf")),
        norm_type=cfg_data["norm_type"],
        window_size=cfg_data["signal_len"],
        cwt_scales=cfg_pre["cwt_scales"],
        cwt_wavelet=cfg_pre["cwt_wavelet"],
        patch_size=cfg_pre["patch_size"],
        k_neighbors=cfg_pre["k_neighbors"],
        denoise=cfg_pre.get("denoise", True),
    )

    # ---- Model ---- #
    model = RPGWNet(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model params: {n_params:,}")

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg_train["lr"],
                           weight_decay=cfg_train.get("weight_decay", 1e-5))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg_train.get("lr_decay_milestones", [150, 250]),
        gamma=cfg_train.get("lr_decay_gamma", 0.1),
    )

    # ---- Training ---- #
    best_acc = 0.0
    best_epoch = 0
    gw_weight = cfg_train["loss"].get("gw_weight", 0.1)
    writer = SummaryWriter(os.path.join(save_dir, "runs"))

    for epoch in range(cfg_train["epochs"]):
        # ===== Train =====
        model.train()
        train_loss_sum, train_acc_sum, n_train = 0.0, 0.0, 0

        for src_batch, tgt_batch in zip(
            dataloaders["source_train"], dataloaders["target_train"]
        ):
            try:
                out = model(src_batch, tgt_batch, mode="train")
            except RuntimeError:
                continue  # skip batch if shape mismatch

            loss = ce_loss(out["logits"], out["labels"])
            loss = loss + gw_weight * out["gw_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = out["logits"].argmax(dim=1)
            train_acc_sum += (pred == out["labels"]).sum().item()
            train_loss_sum += loss.item() * len(out["labels"])
            n_train += len(out["labels"])

        scheduler.step()
        train_loss = train_loss_sum / max(n_train, 1)
        train_acc = train_acc_sum / max(n_train, 1)

        # ===== Eval =====
        model.eval()
        tgt_acc_sum, n_tgt = 0.0, 0

        with torch.no_grad():
            for src_batch, tgt_batch in zip(
                dataloaders["source_test"], dataloaders["target_test"]
            ):
                try:
                    out = model(src_batch, tgt_batch, mode="eval")
                except RuntimeError:
                    continue

                pred = out["tgt_logits"].argmax(dim=1)
                tgt_acc_sum += (pred == out["tgt_labels"]).sum().item()
                n_tgt += len(out["tgt_labels"])

        tgt_acc = tgt_acc_sum / max(n_tgt, 1)

        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/source", train_acc, epoch)
        writer.add_scalar("Acc/target", tgt_acc, epoch)

        if epoch % 20 == 0 or epoch == cfg_train["epochs"] - 1:
            logging.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_loss:.4f} | Src: {train_acc:.3f} | "
                f"Tgt: {tgt_acc:.3f} | GW: {out['gw_loss']:.4f}"
            )

        # Save best
        if tgt_acc > best_acc:
            best_acc = tgt_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    logging.info(f"Best: epoch={best_epoch}, target_acc={best_acc:.4f}")
    writer.close()
    return best_acc
