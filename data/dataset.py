"""
PyTorch Dataset & DataLoader for CWRU Bearing Fault Diagnosis.

Loads preprocessed .npy files (CWT + graph) or processes raw .mat on-the-fly.
"""

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import segment_signal, cwt_transform, build_graph, add_noise, normalize
from .download_cwru import CWRU_FILES, get_label, NUM_CLASSES


class CWRUDataset(Dataset):
    """
    CWRU Bearing Fault Diagnosis Dataset.

    Supports:
    - Raw .mat loading → on-the-fly CWT + graph construction
    - Preprocessed .npy caching (fast)
    - Noise injection
    - Few-shot sampling per class
    """

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
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir:     Directory containing .mat files
            load_hp:      Motor load (0, 1, 2, or 3 hp)
            norm_type:    Normalization type
            window_size:  Samples per sliding window
            cwt_scales:   CWT scale count
            cwt_wavelet:  Mother wavelet
            patch_size:   TF patch size for graph nodes
            k_neighbors:  k for kNN graph
            denoise:      Enable adaptive denoising
            noise_snr_db: Target SNR. inf = no noise
            noise_type:   Noise distribution
            n_shot:       Limit samples per class (few-shot). None = all.
            cache_dir:    Directory for preprocessed .npy cache
        """
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

        # ---- Load raw signals ---- #
        self.samples: list = []  # list of (signal_1d, label_int)
        self._load_mat_files()

        # Few-shot sampling
        if n_shot is not None:
            self._few_shot_sample(n_shot)

    def _load_mat_files(self):
        """Read .mat files, segment into windows, normalize, store."""
        import scipy.io as sio

        for (ft, fs, ld), filename in CWRU_FILES.items():
            if ld != self.load_hp:
                continue
            filepath = self.data_dir / filename
            if not filepath.exists():
                print(f"  [WARNING] {filename} not found, skipping {ft}@{ld}hp")
                continue

            mat = sio.loadmat(filepath)

            # CWRU .mat files have DE (drive-end) vibration data
            # Find the right key
            de_keys = [k for k in mat.keys() if "DE" in k.upper() and not k.startswith("_")]
            if not de_keys:
                continue
            raw_signal = mat[de_keys[0]].squeeze().astype(np.float32)

            # Segment into windows
            windows = segment_signal(raw_signal, self.window_size, overlap=0.0)

            # Normalize each window
            windows = np.array([normalize(w, self.norm_type) for w in windows])

            # Assign label
            label = get_label(ft, fs)

            for w in windows:
                self.samples.append((w, label))

        if not self.samples:
            raise RuntimeError(
                f"No data found for load_hp={self.load_hp} in {self.data_dir}. "
                f"Run 'python data/download_cwru.py' first."
            )

    def _few_shot_sample(self, n_shot: int):
        """Keep only n_shot samples per class."""
        class_samples = {lbl: [] for lbl in range(NUM_CLASSES)}
        for sig, lbl in self.samples:
            class_samples[lbl].append((sig, lbl))

        sampled = []
        for lbl, items in class_samples.items():
            if len(items) > n_shot:
                sampled.extend(random.sample(items, n_shot))
            else:
                sampled.extend(items)

        self.samples = sampled
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                'signal':      raw 1D signal (1, window_size)
                'cwt':         time-frequency map (cwt_scales, window_size)
                'node_feat':   graph node features (n_nodes, patch_size²)
                'adj':         adjacency matrix (n_nodes, n_nodes)
                'cost':        pairwise distance matrix (n_nodes, n_nodes)
                'label':       class label (int)
        """
        signal, label = self.samples[idx]

        # Noise injection
        if self.noise_snr_db != float("inf"):
            signal = add_noise(signal, self.noise_snr_db, self.noise_type)

        # CWT → time-frequency map
        cwt_map = cwt_transform(signal, self.cwt_scales, self.cwt_wavelet)

        # Optional denoising
        if self.denoise:
            from .preprocess import adaptive_denoise
            cwt_map = adaptive_denoise(cwt_map)

        # Graph construction
        node_feat, adj, cost = build_graph(
            cwt_map, self.patch_size, self.k_neighbors
        )

        return {
            "signal":    torch.from_numpy(signal[None, :]).float(),   # (1, 1024)
            "cwt":       torch.from_numpy(cwt_map).float(),            # (64, 1024)
            "node_feat": torch.from_numpy(node_feat).float(),          # (N_nodes, feat_dim)
            "adj":       torch.from_numpy(adj).float(),                # (N_nodes, N_nodes)
            "cost":      torch.from_numpy(cost).float(),               # (N_nodes, N_nodes)
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

    Args:
        data_dir:     Path to .mat files
        source_load:  Source domain motor load (e.g., 0)
        target_load:  Target domain motor load (e.g., 3)
        batch_size:   Batch size
        n_shot:       Few-shot samples per class in target train
        noise_snr_db: Noise level applied to TARGET domain
        num_workers:  DataLoader workers
        **dataset_kwargs: Passed to CWRUDataset

    Returns:
        {'source_train': DL, 'source_test': DL, 'target_train': DL, 'target_test': DL}
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

    # Split: 80% train, 20% test
    n_src = len(source_full)
    n_tgt = len(target_full)
    src_train, src_test = torch.utils.data.random_split(
        source_full, [int(0.8 * n_src), n_src - int(0.8 * n_src)]
    )
    tgt_train, tgt_test = torch.utils.data.random_split(
        target_full, [int(0.8 * n_tgt), n_tgt - int(0.8 * n_tgt)]
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=False)

    return {
        "source_train": DataLoader(src_train, shuffle=True, **loader_kwargs),
        "source_test":  DataLoader(src_test, shuffle=False, **loader_kwargs),
        "target_train": DataLoader(tgt_train, shuffle=True, **loader_kwargs),
        "target_test":  DataLoader(tgt_test, shuffle=False, **loader_kwargs),
    }
