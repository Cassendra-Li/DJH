"""
PyTorch Dataset & DataLoader for CWRU Bearing Fault Diagnosis.

Works with the GitHub mirror directory structure:
    data/raw/CWRU/op_{load}/fault_type_{size}/X.mat

Supports:
- On-the-fly CWT + graph construction per sample
- Noise injection at target SNR
- Few-shot per-class sampling
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import segment_signal, cwt_transform, build_graph, add_noise, normalize

# ========== Directory name → (fault_type, fault_size) ==========
_DIR_TO_FAULT = {
    "normal":    ("Normal", None),
    "inner_07":  ("IR", "007"),
    "inner_14":  ("IR", "014"),
    "inner_21":  ("IR", "021"),
    "outer_07":  ("OR", "007"),
    "outer_14":  ("OR", "014"),
    "outer_21":  ("OR", "021"),
    "ball_07":   ("B", "007"),
    "ball_14":   ("B", "014"),
    "ball_21":   ("B", "021"),
}

# ========== Label mapping ==========
_LABEL_MAP = {
    "Normal": 0,
    "IR007": 1, "IR014": 2, "IR021": 3,
    "OR007": 4, "OR014": 5, "OR021": 6,
    "B007": 7,  "B014": 8,  "B021": 9,
}

NUM_CLASSES = 10


def get_label(fault_type: str, fault_size: str = None) -> int:
    if fault_type == "Normal":
        return _LABEL_MAP["Normal"]
    return _LABEL_MAP[f"{fault_type}{fault_size}"]


class CWRUDataset(Dataset):
    """Loads .mat files, segments, applies CWT, builds graphs."""

    def __init__(
        self,
        data_dir: str,
        load_hp: int,
        norm_type: str = "z-score",
        window_size: int = 1024,
        cwt_scales: int = 64,
        cwt_wavelet: str = "morl",
        patch_size: int = 8,
        k_neighbors: int = 8,
        denoise: bool = True,
        noise_snr_db: float = float("inf"),
        noise_type: str = "gaussian",
        n_shot: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.load_hp = load_hp
        self.norm_type = norm_type
        self.window_size = window_size
        self.cwt_scales = cwt_scales
        self.cwt_wavelet = cwt_wavelet
        self.patch_size = patch_size
        self.k_neighbors = k_neighbors
        self.denoise = denoise
        self.noise_snr_db = noise_snr_db
        self.noise_type = noise_type
        self.n_shot = n_shot

        self.samples: list = []  # list of (signal_1d, label_int)
        self._load_mat_files()

        if n_shot is not None:
            self._few_shot_sample(n_shot)

    def _load_mat_files(self):
        """Walk op_{load}/ directory, load .mat files."""
        import scipy.io as sio

        op_dir = self.data_dir / f"op_{self.load_hp}"
        if not op_dir.exists():
            raise RuntimeError(
                f"Directory {op_dir} not found. "
                f"Run 'python data/download_cwru.py' first."
            )

        for fault_dir_name, (ft, fs) in _DIR_TO_FAULT.items():
            fault_path = op_dir / fault_dir_name
            if not fault_path.exists():
                continue

            for mat_file in sorted(fault_path.glob("*.mat")):
                try:
                    mat = sio.loadmat(mat_file)
                except Exception:
                    continue

                # Find DE (drive-end) vibration data key
                de_keys = [k for k in mat.keys()
                           if "DE" in k.upper() and not k.startswith("_")]
                if not de_keys:
                    continue

                raw_signal = mat[de_keys[0]].squeeze().astype(np.float32)

                # Segment into windows
                windows = segment_signal(raw_signal, self.window_size, overlap=0.0)
                windows = np.array([normalize(w, self.norm_type) for w in windows])

                label = get_label(ft, fs)
                for w in windows:
                    self.samples.append((w, label))

        # Report
        class_counts = {}
        for _, lbl in self.samples:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
        print(f"  [Load {self.load_hp}] Loaded {len(self.samples)} windows "
              f"from {len(class_counts)} classes: {dict(sorted(class_counts.items()))}")

        if not self.samples:
            raise RuntimeError(
                f"No data found for load_hp={self.load_hp} in {op_dir}."
            )

    def _few_shot_sample(self, n_shot: int):
        """Keep only n_shot samples per class."""
        class_samples = {lbl: [] for lbl in range(NUM_CLASSES)}
        for sig, lbl in self.samples:
            if lbl in class_samples:
                class_samples[lbl].append((sig, lbl))

        sampled = []
        for lbl, items in class_samples.items():
            if items and len(items) > n_shot:
                sampled.extend(random.sample(items, n_shot))
            elif items:
                sampled.extend(items)

        self.samples = sampled
        random.shuffle(self.samples)
        print(f"  Few-shot (n={n_shot}): {len(self.samples)} samples total")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        signal, label = self.samples[idx]

        # Noise injection
        if self.noise_snr_db != float("inf"):
            signal = add_noise(signal, self.noise_snr_db, self.noise_type)

        # CWT
        cwt_map = cwt_transform(signal, self.cwt_scales, self.cwt_wavelet)

        # Denoise
        if self.denoise:
            from .preprocess import adaptive_denoise
            cwt_map = adaptive_denoise(cwt_map)

        # Graph
        node_feat, adj, cost = build_graph(cwt_map, self.patch_size, self.k_neighbors)

        return {
            "signal":    torch.from_numpy(signal[None, :]).float(),
            "cwt":       torch.from_numpy(cwt_map).float(),
            "node_feat": torch.from_numpy(node_feat).float(),
            "adj":       torch.from_numpy(adj).float(),
            "cost":      torch.from_numpy(cost).float(),
            "label":     torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(
    data_dir: str,
    source_load: int,
    target_load: int,
    batch_size: int = 64,
    n_shot: Optional[int] = None,
    noise_snr_db: float = float("inf"),
    num_workers: int = 0,
    **dataset_kwargs,
) -> Dict[str, DataLoader]:
    """
    Get train/test dataloaders for source and target domains.

    Returns:
        {'source_train': DL, 'source_test': DL,
         'target_train': DL, 'target_test': DL}
    """
    # Source domain (no noise, full data)
    source_full = CWRUDataset(
        data_dir, source_load, noise_snr_db=float("inf"), **dataset_kwargs
    )
    # Target domain (with noise, optional few-shot)
    target_full = CWRUDataset(
        data_dir, target_load, noise_snr_db=noise_snr_db,
        n_shot=n_shot, **dataset_kwargs
    )

    # 80/20 split
    n_src, n_tgt = len(source_full), len(target_full)
    src_train, src_test = torch.utils.data.random_split(
        source_full, [int(0.8 * n_src), n_src - int(0.8 * n_src)]
    )
    tgt_train, tgt_test = torch.utils.data.random_split(
        target_full, [int(0.8 * n_tgt), n_tgt - int(0.8 * n_tgt)]
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True)

    return {
        "source_train": DataLoader(src_train, shuffle=True, **loader_kwargs),
        "source_test":  DataLoader(src_test, shuffle=False, **loader_kwargs),
        "target_train": DataLoader(tgt_train, shuffle=True, **loader_kwargs),
        "target_test":  DataLoader(tgt_test, shuffle=False, **loader_kwargs),
    }
