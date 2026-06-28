"""
Quick Demo Test (5-minute pipeline verification)
=================================================
Runs a minimal training to verify everything works.

What it tests:
  1. CWT + graph construction on synthetic data
  2. GW alignment with all variants
  3. GAT encoder forward pass
  4. End-to-end prototype classification

Run: python scripts/demo_quick_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

print("=" * 60)
print("RPGW-Net Quick Demo Test")
print("=" * 60)


# ============================================================
# Test 1: CWT + Graph Construction
# ============================================================
print("\n[1/5] Testing CWT + Graph Construction...")

from data.preprocess import segment_signal, cwt_transform, build_graph, add_noise, adaptive_denoise

# Generate synthetic vibration signal (sinusoidal with fault impulses)
t = np.linspace(0, 1, 2048)
fault_freq = 100  # Hz (bearing fault frequency)
sig = np.sin(2 * np.pi * fault_freq * t)
sig += 0.1 * np.random.randn(2048)  # add noise

# Segment
windows = segment_signal(sig, window_size=1024)
assert windows.shape == (2, 1024), f"Expected (2, 1024), got {windows.shape}"

# CWT
tf_map = cwt_transform(windows[0], scales=64)
assert tf_map.shape == (64, 1024), f"Expected (64, 1024), got {tf_map.shape}"

# Denoise
tf_clean = adaptive_denoise(tf_map)
assert tf_clean.shape == tf_map.shape

# Graph
node_feat, adj, cost = build_graph(tf_map, patch_size=8, k_neighbors=8)
n_nodes = (64 // 8) * (1024 // 8)  # 8 * 128 = 1024??? Actually 8 * 128 = 1024
# Wait: 64/8=8, 1024/8=128 → 1024 nodes
print(f"  [OK] Graph: {n_nodes} nodes, adj density: {adj.mean():.3f}")
assert node_feat.shape[0] == n_nodes
print("  [OK] PASSED")


# ============================================================
# Test 2: GW Alignment (all variants)
# ============================================================
print("\n[2/5] Testing GW Alignment variants...")

from rpgw.models.gw_alignment import GWAlignment

# Create two synthetic graphs (source & target) with slightly different structures
N = 32  # nodes (use smaller for speed)
np.random.seed(42)
X_s = np.random.randn(N, 64)
X_t = np.random.randn(N, 64) * 0.8 + 0.2  # slightly different distribution

C_s_np = np.sqrt(((X_s[:, None] - X_s[None, :])**2).sum(-1))
C_t_np = np.sqrt(((X_t[:, None] - X_t[None, :])**2).sum(-1))

C_s = torch.from_numpy(C_s_np).float().unsqueeze(0)  # (1, N, N)
C_t = torch.from_numpy(C_t_np).float().unsqueeze(0)
H_s = torch.from_numpy(X_s).float().unsqueeze(0)
H_t = torch.from_numpy(X_t).float().unsqueeze(0)

for gw_type in ["vanilla", "entropic", "fused", "partial"]:
    gw = GWAlignment(type=gw_type, multi_init=2)
    result = gw(C_s, C_t, H_s, H_t)
    print(f"  {gw_type:10s}: GW loss = {result['gw_loss']:.4f}, "
          f"transport shape = {tuple(result['transport'].shape)}")
print("  [OK] PASSED")


# ============================================================
# Test 3: GAT Encoder
# ============================================================
print("\n[3/5] Testing GAT Encoder...")

from rpgw.models.gat_encoder import GATEncoder

gat = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256, heads=4)
node_feat_t = torch.from_numpy(node_feat).float().unsqueeze(0)  # (1, N, 64)
adj_t = torch.from_numpy(adj).float().unsqueeze(0)                # (1, N, N)

H, C_gat = gat(node_feat_t, adj_t)
print(f"  [OK] GAT output: H shape = {tuple(H.shape)}, C shape = {tuple(C_gat.shape)}")
print("  [OK] PASSED")


# ============================================================
# Test 4: Prototype Classifier
# ============================================================
print("\n[4/5] Testing Prototype Classifier...")

from rpgw.models.prototype import PrototypeClassifier

proto = PrototypeClassifier(in_dim=256, num_classes=4)

# Simulate few-shot: 4 classes, 3 support samples each
embeddings = torch.randn(12, 256)
labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

prototypes = proto.compute_prototypes(embeddings, labels)
assert prototypes.shape == (4, 256)

query = torch.randn(8, 256)
logits = proto(query, prototypes)
assert logits.shape == (8, 4)

pred = logits.argmax(dim=1)
print(f"  [OK] Prototypes: {prototypes.shape}, Logits: {logits.shape}")
print("  [OK] PASSED")


# ============================================================
# Test 5: End-to-End (toy data)
# ============================================================
print("\n[5/5] Testing End-to-End Pipeline (toy data)...")

from rpgw.train import RPGWNet

# Minimal config (matches current train.py interface)
config = {
    "model": {
        "gat": {"hidden_dim": 128, "out_dim": 256, "heads": 4, "dropout": 0.3},
        "gw": {"type": "partial", "epsilon": 0.8, "alpha": 0.7,
               "partial_mass": 0.85, "multi_init": 2},
        "prototype": {"distance": "euclidean", "use_weighting": True, "temperature": 1.0},
    },
    "preprocess": {"patch_size": 8, "k_neighbors": 8},
}

model = RPGWNet(config)

# Toy batches (B=4, N nodes varies per sample → use fixed N_pad)
B = 2
N_pts = 32
src_batch = {
    "signal": torch.randn(B, 1, 1024),
    "node_feat": torch.randn(B, N_pts, 64),
    "adj": (torch.rand(B, N_pts, N_pts) > 0.9).float(),
    "cost": torch.cdist(torch.randn(B, N_pts, 64), torch.randn(B, N_pts, 64)),
    "label": torch.randint(0, 10, (B,)),
}
tgt_batch = {
    "signal": torch.randn(B, 1, 1024),
    "node_feat": torch.randn(B, N_pts, 64),
    "adj": (torch.rand(B, N_pts, N_pts) > 0.9).float(),
    "cost": torch.cdist(torch.randn(B, N_pts, 64), torch.randn(B, N_pts, 64)),
    "label": torch.randint(0, 10, (B,)),
}

outputs = model(src_batch, tgt_batch, mode="train")
print(f"  [OK] E2E forward: logits = {outputs['logits'].shape}, "
      f"gw_loss = {outputs['gw_loss']:.4f}")

outputs_eval = model(src_batch, tgt_batch, mode="eval")
print(f"  [OK] E2E eval:   logits = {outputs_eval['logits'].shape}")

num_params = sum(p.numel() for p in model.parameters())
print(f"  [OK] Total parameters: {num_params:,}")
print("  [OK] PASSED")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED [OK]")
print("RPGW-Net pipeline is ready for training on CWRU data.")
print("=" * 60)
print("\nNext steps:")
print("  1. Download CWRU:  python data/download_cwru.py")
print("  2. Train RPGW-Net: python experiments/run_rpgw.py --source 0 --target 3")
