"""
Cached CWRU Dataset — loads precomputed .npy graphs for fast training.

Run `python data/precompute.py` first to generate cache files.

For clean (no-noise) experiments: instant loading from .npy (~0.1ms per sample)
For noisy experiments: falls back to on-the-fly CWT + graph construction.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import cwt_transform, build_graph, add_noise, adaptive_denoise, normalize

NUM_CLASSES = 10


class CachedCWRUDataset(Dataset):
    """Loads precomputed graphs from data/processed/CWRU/load_{hp}/."""

    def __init__(
        self,
        cache_dir: str,
        load_hp: int,
        noise_snr_db: float = float("inf"),
        noise_type: str = "gaussian",
        n_shot: Optional[int] = None,
        seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir) / f"load_{load_hp}"
        self.noise_snr_db = noise_snr_db
        self.noise_type = noise_type
        self.n_shot = n_shot

        # Load metadata
        with open(self.cache_dir / "metadata.pkl", "rb") as f:
            self.meta = pickle.load(f)

        self.num_samples = self.meta["num_samples"]
        self.labels = []

        # Load all labels
        for idx in range(self.num_samples):
            lbl = int(np.load(self.cache_dir / f"sample_{idx}_label.npy"))
            self.labels.append(lbl)

        # Few-shot sampling
        self.indices = list(range(self.num_samples))
        if n_shot is not None:
            rng = random.Random(seed)
            class_indices = {l: [] for l in range(NUM_CLASSES)}
            for i, lbl in enumerate(self.labels):
                if lbl in class_indices:
                    class_indices[lbl].append(i)
            sampled = []
            for lbl, idxs in class_indices.items():
                if len(idxs) > n_shot:
                    sampled.extend(rng.sample(idxs, n_shot))
                else:
                    sampled.extend(idxs)
            self.indices = sampled
            rng.shuffle(self.indices)

        self._use_cache = (noise_snr_db == float("inf"))

        # Always load raw signals from .mat files (needed for CNN input)
        from .dataset import CWRUDataset
        raw_dir = Path(cache_dir).parent / "raw" / "CWRU"
        if raw_dir.exists():
            self._raw_dataset = CWRUDataset(
                data_dir=str(raw_dir),
                load_hp=load_hp,
                # Use minimal params — only _load_mat_files() runs in __init__
                # (CWT/graph computation only happens in __getitem__)
                cwt_scales=1, cwt_wavelet="morl",
                patch_size=8, k_neighbors=8,
            )
        else:
            self._raw_dataset = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        label = self.labels[real_idx]

        # Always get raw signal from source .mat files (needed for CNN)
        if self._raw_dataset is not None:
            signal_raw = self._raw_dataset.samples[real_idx][0].copy()
        else:
            signal_raw = np.zeros(1024, dtype=np.float32)

        if self._use_cache:
            # Fast path: precomputed graphs from cache + raw signal from .mat
            node_feat = np.load(self.cache_dir / f"sample_{real_idx}_node.npy")
            adj = np.load(self.cache_dir / f"sample_{real_idx}_adj.npy")
            cost = np.load(self.cache_dir / f"sample_{real_idx}_cost.npy")
            cwt_map = np.zeros((self.meta["config"]["cwt_scales"], 1024), dtype=np.float32)
        else:
            # Noisy path: add noise, recompute CWT + graph
            signal_raw = add_noise(signal_raw, self.noise_snr_db, self.noise_type)
            cwt_map = cwt_transform(signal_raw, self.meta["config"]["cwt_scales"],
                                     self.meta["config"]["cwt_wavelet"])
            if self.meta["config"].get("denoise", True):
                cwt_map = adaptive_denoise(cwt_map)
            node_feat, adj, cost = build_graph(
                cwt_map, self.meta["config"]["patch_size"],
                self.meta["config"]["k_neighbors"])

        return {
            "signal":    torch.from_numpy(signal_raw[None, :]).float(),
            "cwt":       torch.from_numpy(cwt_map).float(),
            "node_feat": torch.from_numpy(node_feat).float(),
            "adj":       torch.from_numpy(adj).float(),
            "cost":      torch.from_numpy(cost).float(),
            "label":     torch.tensor(label, dtype=torch.long),
        }


def get_cached_dataloaders(
    cache_dir: str = "data/processed/CWRU",
    source_load: int = 0,
    target_load: int = 3,
    batch_size: int = 8,
    n_shot: Optional[int] = None,
    noise_snr_db: float = float("inf"),
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Get dataloaders backed by precomputed cache.

    Returns:
        {'source_train': DL, 'source_test': DL,
         'target_train': DL, 'target_test': DL}
    """
    # Source: always clean
    src_full = CachedCWRUDataset(cache_dir, source_load, noise_snr_db=float("inf"))
    # Target: may have noise + few-shot
    tgt_full = CachedCWRUDataset(cache_dir, target_load,
                                  noise_snr_db=noise_snr_db, n_shot=n_shot)

    n_src = len(src_full)
    n_tgt = len(tgt_full)
    src_train, src_test = torch.utils.data.random_split(
        src_full, [int(0.8 * n_src), n_src - int(0.8 * n_src)]
    )
    tgt_train, tgt_test = torch.utils.data.random_split(
        tgt_full, [int(0.8 * n_tgt), n_tgt - int(0.8 * n_tgt)]
    )

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=False, drop_last=True)
    return {
        "source_train": DataLoader(src_train, shuffle=True, **kw),
        "source_test":  DataLoader(src_test, shuffle=False, **kw),
        "target_train": DataLoader(tgt_train, shuffle=True, **kw),
        "target_test":  DataLoader(tgt_test, shuffle=False, **kw),
    }
