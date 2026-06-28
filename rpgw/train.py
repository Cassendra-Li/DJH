"""
RPGW-Net Training Loop
=======================
Handles the full training pipeline:
  1. Load CWRU data (source + target domains)
  2. Forward: CNN → GraphBuilder → GAT → GW Alignment → Prototype Classifier
  3. Loss: cross-entropy (source) + GW alignment + optional MMD
  4. Evaluate on target domain test set
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import CWRUDataset, get_dataloaders
from rpgw.models.cnn_encoder import CNNEncoder
from rpgw.models.graph_builder import GraphBuilder
from rpgw.models.gat_encoder import GATEncoder
from rpgw.models.gw_alignment import GWAlignment
from rpgw.models.prototype import PrototypeClassifier
from rpgw.losses.gw_loss import CombinedLoss, mmd_loss

# ============================================================
# Logging Setup
# ============================================================

def setup_logging(log_dir: str):
    """Set up file + console logging."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )


# ============================================================
# Model Factory
# ============================================================

class RPGWNet(nn.Module):
    """
    Complete RPGW-Net model.
    Combines all sub-modules: CNN → Graph → GAT → GW → Prototype.
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        pre_cfg = config["preprocess"]

        # CNN feature extractor
        self.cnn = CNNEncoder(**cfg["cnn"])

        # Graph builder (precomputed mode by default)
        self.graph = GraphBuilder(
            patch_size=pre_cfg["patch_size"],
            k_neighbors=pre_cfg["k_neighbors"],
            mode="precomputed",
        )

        # GAT encoder
        self.gat = GATEncoder(
            in_dim=pre_cfg["patch_size"] ** 2,
            **{k: v for k, v in cfg["gat"].items() if k != "in_dim"},
        )

        # GW alignment
        self.gw_align = GWAlignment(**cfg["gw"])

        # Prototype classifier
        self.prototype = PrototypeClassifier(
            in_dim=cfg["gat"]["out_dim"],
            num_classes=10,
            **cfg["prototype"],
        )

        self.gat_out_dim = cfg["gat"]["out_dim"]

    def forward(
        self,
        source_batch: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Full RPGW-Net forward pass.

        Args:
            source_batch: Source domain data dict
            target_batch: Target domain data dict
            mode:         'train' (with GW alignment) | 'eval' (no GW)

        Returns:
            Dict with logits, loss components, etc.
        """
        device = next(self.parameters()).device

        # ---- 1. Extract CNN features ---- #
        src_cnn = self.cnn(source_batch["signal"].to(device))      # (B_s, 256)
        tgt_cnn = self.cnn(target_batch["signal"].to(device))      # (B_t, 256)

        # ---- 2. Build graphs ---- #
        src_node, src_adj, src_cost = self.graph(
            None, source_batch["node_feat"].to(device),
            source_batch["adj"].to(device), source_batch["cost"].to(device),
        )
        tgt_node, tgt_adj, tgt_cost = self.graph(
            None, target_batch["node_feat"].to(device),
            target_batch["adj"].to(device), target_batch["cost"].to(device),
        )

        # ---- 3. GAT encode ---- #
        H_s, C_s = self.gat(src_node, src_adj)      # (B_s, N, D), (B_s, N, N)
        H_t, C_t = self.gat(tgt_node, tgt_adj)      # (B_t, N, D), (B_t, N, N)

        # Global graph pooling (mean) → graph-level embeddings
        H_s_global = H_s.mean(dim=1)                 # (B_s, D)
        H_t_global = H_t.mean(dim=1)                 # (B_t, D)

        # ---- 4. GW Alignment (mean across batch of pairwise dist matrices) ---- #
        gw_loss = torch.tensor(0.0, device=device)
        aligned_H_t = H_t_global
        transport_quality = None

        if mode == "train":
            # Average distance matrices across batch for GW
            C_s_mean = C_s.mean(dim=0)              # (N, N)
            C_t_mean = C_t.mean(dim=0)              # (N, N)
            H_s_mean = H_s.mean(dim=0)              # (N, D)
            H_t_mean = H_t.mean(dim=0)              # (N, D)

            gw_result = self.gw_align(
                C_s_mean.unsqueeze(0), C_t_mean.unsqueeze(0),
                H_s_mean.unsqueeze(0), H_t_mean.unsqueeze(0),
            )
            gw_loss = gw_result["gw_loss"]
            transport = gw_result["transport"]       # (1, N, N)

            # Compute per-node alignment quality (sum of transport mass per node)
            transport_quality = transport[0].sum(dim=1)  # (N,)

        # ---- 5. Prototype classification ---- #
        # Source: use labels to build prototypes
        prototypes = self.prototype.compute_prototypes(
            H_s_global, source_batch["label"].to(device),
            weights=None,  # source is clean
        )

        # Target: classify by distance to prototypes
        logits = self.prototype(H_t_global, prototypes)

        # ---- 6. Compute MMD loss (for ablation) ---- #
        mmd = mmd_loss(H_s_global, H_t_global)

        return {
            "logits":     logits,
            "labels":     target_batch["label"].to(device) if mode == "eval"
                          else source_batch["label"].to(device),
            "gw_loss":    gw_loss,
            "mmd_loss":   mmd,
            "H_s":        H_s_global,
            "H_t":        H_t_global,
        }


# ============================================================
# Training Loop
# ============================================================

def train_rpgw(config: dict, save_dir: str):
    """
    Main training function for RPGW-Net.

    Args:
        config:   Config dict (loaded from YAML)
        save_dir: Directory for checkpoints and logs
    """
    setup_logging(os.path.join(save_dir, "logs"))
    cfg_train = config["train"]
    cfg_data = config["data"]
    cfg_pre = config["preprocess"]
    cfg_noise = config["noise"]

    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

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
    loss_fn = CombinedLoss(**cfg_train["loss"])
    logging.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ---- Optimizer ---- #
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg_train["lr"],
        weight_decay=cfg_train.get("weight_decay", 1e-5),
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg_train.get("lr_decay_milestones", [150, 250]),
        gamma=cfg_train.get("lr_decay_gamma", 0.1),
    )

    # ---- TensorBoard ---- #
    writer = SummaryWriter(os.path.join(save_dir, "runs"))

    # ---- Training ---- #
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(cfg_train["epochs"]):
        # ===== Train =====
        model.train()
        train_loss, train_acc, n_train = 0.0, 0.0, 0

        for src_batch, tgt_batch in zip(
            dataloaders["source_train"], dataloaders["target_train"]
        ):
            # Only use source labels for classification
            outputs = model(src_batch, tgt_batch, mode="train")
            src_labels = src_batch["label"].to(device)

            # Recompute logits on source for labeled loss
            H_s = outputs["H_s"]
            prototypes = model.prototype.compute_prototypes(H_s, src_labels)
            logits = model.prototype(H_s, prototypes)

            loss_dict = loss_fn({
                "logits":   logits,
                "labels":   src_labels,
                "gw_loss":  outputs["gw_loss"],
                "mmd_loss": outputs["mmd_loss"],
            })

            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            train_acc += (pred == src_labels).sum().item()
            train_loss += loss_dict["total"].item() * src_labels.size(0)
            n_train += src_labels.size(0)

        scheduler.step()
        train_acc = train_acc / n_train
        train_loss = train_loss / n_train

        # ===== Eval =====
        model.eval()
        tgt_acc, n_tgt = 0.0, 0

        with torch.no_grad():
            for src_batch, tgt_batch in zip(
                dataloaders["source_test"], dataloaders["target_test"]
            ):
                outputs = model(src_batch, tgt_batch, mode="eval")

                # Build prototypes from source test (few-shot evaluation)
                H_s = outputs["H_s"]
                src_labels = src_batch["label"].to(device)
                prototypes = model.prototype.compute_prototypes(H_s, src_labels)

                # Classify target samples
                logits = model.prototype(outputs["H_t"], prototypes)
                pred = logits.argmax(dim=1)
                tgt_acc += (pred == tgt_batch["label"].to(device)).sum().item()
                n_tgt += tgt_batch["label"].size(0)

        tgt_acc = tgt_acc / n_tgt

        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/target", tgt_acc, epoch)

        if epoch % 10 == 0 or epoch == cfg_train["epochs"] - 1:
            logging.info(
                f"Epoch {epoch:3d}/{cfg_train['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Target Acc: {tgt_acc:.4f}"
            )

        # Save best
        if tgt_acc > best_acc:
            best_acc = tgt_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    logging.info(f"Best: epoch={best_epoch}, target_acc={best_acc:.4f}")
    writer.close()

    return best_acc
