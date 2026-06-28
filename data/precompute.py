"""
Precompute CWT + Graphs for All CWRU Samples
=============================================
Run ONCE before training. Saves node features, adjacency matrices, and distance
matrices as .npy files for fast loading during training.

Usage:
    python data/precompute.py                     # all loads
    python data/precompute.py --load 0 --load 3   # specific loads

Output structure:
    data/processed/CWRU/load_{hp}/
        sample_{idx}_node.npy   # (N, D) node features
        sample_{idx}_adj.npy    # (N, N) adjacency matrix
        sample_{idx}_cost.npy   # (N, N) distance matrix
        sample_{idx}_label.npy  # scalar label
        metadata.pkl            # {num_samples, classes, nodes, ...}
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import CWRUDataset, NUM_CLASSES

DEFAULT_CONFIG = {
    "norm_type": "z-score",
    "window_size": 1024,
    "cwt_scales": 64,
    "cwt_wavelet": "morl",
    "patch_size": 16,
    "k_neighbors": 8,
    "denoise": True,
}


def precompute_load(
    data_dir: str,
    load_hp: int,
    save_dir: str,
    config: dict,
) -> int:
    """Precompute all samples for one motor load. Returns number of samples."""
    save_path = Path(save_dir) / f"load_{load_hp}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Check if already done
    meta_path = save_path / "metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"  Load {load_hp}: already precomputed ({meta['num_samples']} samples). Skipping.")
        return meta["num_samples"]

    print(f"  Load {load_hp}: loading raw signals + computing CWT + building graphs...")

    # Create dataset without noise/few-shot (we want all clean samples)
    dataset = CWRUDataset(
        data_dir=data_dir,
        load_hp=load_hp,
        norm_type=config["norm_type"],
        window_size=config["window_size"],
        cwt_scales=config["cwt_scales"],
        cwt_wavelet=config["cwt_wavelet"],
        patch_size=config["patch_size"],
        k_neighbors=config["k_neighbors"],
        denoise=config["denoise"],
        noise_snr_db=float("inf"),
        n_shot=None,
    )

    print(f"    Computing {len(dataset)} samples...")
    class_counts = {i: 0 for i in range(NUM_CLASSES)}

    for idx in tqdm(range(len(dataset)), desc=f"    Load {load_hp}"):
        sample = dataset[idx]
        label = int(sample["label"].item())

        np.save(save_path / f"sample_{idx}_node.npy", sample["node_feat"].numpy())
        np.save(save_path / f"sample_{idx}_adj.npy",  sample["adj"].numpy())
        np.save(save_path / f"sample_{idx}_cost.npy", sample["cost"].numpy())
        np.save(save_path / f"sample_{idx}_label.npy", np.array(label, dtype=np.int32))
        class_counts[label] += 1

    # Save metadata
    meta = {
        "num_samples": len(dataset),
        "classes": class_counts,
        "num_nodes": sample["node_feat"].shape[0],
        "node_dim":   sample["node_feat"].shape[1],
        "config":     config,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"    Done: {len(dataset)} samples, classes: {dict(sorted(class_counts.items()))}")
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description="Precompute CWRU graphs")
    parser.add_argument("--data_dir", type=str, default="data/raw/CWRU")
    parser.add_argument("--save_dir", type=str, default="data/processed/CWRU")
    parser.add_argument("--load", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    print("=" * 60)
    print("CWRU Data Precomputation")
    print(f"  Data: {args.data_dir}")
    print(f"  Save: {args.save_dir}")
    print(f"  Config: patch_size={DEFAULT_CONFIG['patch_size']}, "
          f"k={DEFAULT_CONFIG['k_neighbors']}, wavelet={DEFAULT_CONFIG['cwt_wavelet']}")
    print("=" * 60)

    total = 0
    for ld in args.load:
        n = precompute_load(args.data_dir, ld, args.save_dir, DEFAULT_CONFIG)
        total += n

    print(f"\nTotal: {total} samples precomputed across {len(args.load)} loads.")
    print(f"Ready for training!")
    print(f"\nNext: python experiments/run_rpgw.py --source 0 --target 3")


if __name__ == "__main__":
    main()
