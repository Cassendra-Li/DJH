"""
Signal Preprocessing & Graph Construction
==========================================
1. Sliding window segmentation
2. Continuous Wavelet Transform (CWT) → time-frequency map
3. Adaptive threshold denoising
4. Graph construction: patch → node, kNN → edge

Key design: all heavy preprocessing done ONCE, saved to .npy for fast loading.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import pywt

# ============================================================
# 1. Sliding Window Segmentation
# ============================================================

def segment_signal(
    raw_signal: np.ndarray,
    window_size: int = 1024,
    overlap: float = 0.0,
) -> np.ndarray:
    """
    Slice a long 1D vibration signal into fixed-size windows.

    Args:
        raw_signal:  1D array of vibration measurements
        window_size: Number of samples per window
        overlap:     Overlap ratio [0, 1)

    Returns:
        (n_windows, window_size) numpy array
    """
    n = len(raw_signal)
    stride = int(window_size * (1 - overlap))
    if stride <= 0:
        raise ValueError(f"stride={stride}, decrease window_size or overlap")

    windows = []
    for start in range(0, n - window_size + 1, stride):
        windows.append(raw_signal[start : start + window_size])
    return np.array(windows)


# ============================================================
# 2. Continuous Wavelet Transform (CWT)
# ============================================================

def cwt_transform(
    signal: np.ndarray,
    scales: int = 64,
    wavelet: str = "morl",
) -> np.ndarray:
    """
    Apply Continuous Wavelet Transform to a 1D signal.

    Args:
        signal:  1D vibration signal (window_size,)
        scales:  Number of frequency scales (output dim: scales × window_size)
        wavelet: Mother wavelet name. Common choices:
                 'morl' = Morlet (best for bearing vibration)
                 'cmor' = Complex Morlet
                 'mexh' = Mexican hat

    Returns:
        Time-frequency map of shape (scales, window_size)
        Values are frequency magnitudes (positive real numbers)

    >>> sig = np.random.randn(1024)
    >>> tf_map = cwt_transform(sig, scales=64)
    >>> tf_map.shape
    (64, 1024)
    """
    if signal.ndim != 1:
        signal = signal.squeeze()

    # pywt.cwt returns [coefficients] where coefficients.shape = (scales, len(signal))
    # We use the real part for Morlet
    coef, _ = pywt.cwt(signal, np.arange(1, scales + 1), wavelet)
    tf_map = np.abs(coef)  # Magnitude = frequency energy
    return tf_map.astype(np.float32)


# ============================================================
# 3. Adaptive Wavelet Threshold Denoising (Optional)
# ============================================================

def adaptive_denoise(
    tf_map: np.ndarray,
    threshold_factor: float = 1.0,
) -> np.ndarray:
    """
    Adaptive threshold denoising on time-frequency map.

    Uses the universal threshold: λ = σ · √(2·log(N))
    where σ = median absolute deviation (MAD) / 0.6745.

    The threshold is applied PER-FREQUENCY-SCALE (adaptive across scales).

    Args:
        tf_map:           CWT time-frequency map (scales, time)
        threshold_factor: Multiplier for threshold (higher = more denoising)

    Returns:
        Denoised time-frequency map
    """
    denoised = tf_map.copy()
    for scale in range(tf_map.shape[0]):
        row = tf_map[scale]
        # Robust noise std estimate via MAD
        median = np.median(row)
        mad = np.median(np.abs(row - median))
        sigma = mad / 0.6745
        threshold = threshold_factor * sigma * np.sqrt(2 * np.log(len(row)))

        # Soft thresholding
        denoised[scale] = np.sign(row) * np.maximum(np.abs(row) - threshold, 0)

    return denoised


# ============================================================
# 4. Time-Frequency Graph Construction
# ============================================================

def build_graph(
    tf_map: np.ndarray,
    patch_size: int = 8,
    k_neighbors: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time-frequency map into a graph.

    Steps:
      1. Split the TF map into non-overlapping patches (patch_size × patch_size)
      2. Each patch → a graph node (node feature = flattened patch)
      3. Compute k-nearest neighbors in Euclidean space → graph edges
      4. Build the pairwise distance matrix C (used as GW cost matrix)

    Args:
        tf_map:      Time-frequency map of shape (scales, time_steps)
        patch_size:  Size of each square patch (e.g., 8 → 8×8 patch)
        k_neighbors: Number of neighbors in kNN graph

    Returns:
        node_features: (n_nodes, patch_size * patch_size)     node features X
        adj_matrix:    (n_nodes, n_nodes)                     adjacency matrix
        cost_matrix:   (n_nodes, n_nodes)                     pairwise distance matrix C
    """
    n_scales, n_time = tf_map.shape
    n_patch_h = n_scales // patch_size      # patches along frequency axis
    n_patch_w = n_time // patch_size         # patches along time axis
    n_nodes = n_patch_h * n_patch_w

    if n_nodes <= 0:
        raise ValueError(
            f"TF map ({n_scales}×{n_time}) too small for patch_size={patch_size}. "
            f"Got {n_patch_h}×{n_patch_w} patches → {n_nodes} nodes."
        )

    # Step 1: Extract patch features
    node_features = np.zeros((n_nodes, patch_size * patch_size), dtype=np.float32)
    for i, (hi, wi) in enumerate(
        [(h, w) for h in range(n_patch_h) for w in range(n_patch_w)]
    ):
        patch = tf_map[
            hi * patch_size : (hi + 1) * patch_size,
            wi * patch_size : (wi + 1) * patch_size,
        ]
        node_features[i] = patch.flatten()

    # Step 2: kNN graph construction
    # Compute pairwise Euclidean distances between node features
    # Using efficient matrix ops: ||a-b||² = ||a||² + ||b||² - 2·a·b^T
    sq_norms = (node_features ** 2).sum(axis=1)
    dist_matrix = sq_norms[:, None] + sq_norms[None, :] - 2 * node_features @ node_features.T
    dist_matrix = np.maximum(dist_matrix, 0)  # numerical stability
    cost_matrix = np.sqrt(dist_matrix)         # Euclidean distance matrix C

    # kNN adjacency: connect each node to its k nearest neighbors
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        # Get indices of k+1 smallest distances (skip self)
        nn_indices = np.argpartition(cost_matrix[i], k_neighbors + 1)[: k_neighbors + 1]
        nn_indices = nn_indices[nn_indices != i]  # exclude self
        nn_indices = nn_indices[:k_neighbors]
        adj_matrix[i, nn_indices] = 1.0
        adj_matrix[nn_indices, i] = 1.0  # make symmetric

    return node_features, adj_matrix, cost_matrix


# ============================================================
# 5. Gaussian Noise Injection
# ============================================================

def add_noise(
    signal: np.ndarray,
    target_snr_db: float,
    noise_type: str = "gaussian",
) -> np.ndarray:
    """
    Add noise to a signal at a target SNR level.

    Args:
        signal:        Clean signal
        target_snr_db: Target SNR in dB. Lower = more noise.
                       Common values: inf (no noise), 10, 5, 0, -5
        noise_type:    'gaussian' | 'laplacian' | 'uniform'

    Returns:
        Noisy signal (same shape)

    SNR formula: SNR_dB = 10·log₁₀(σ²_signal / σ²_noise)

    Examples:
        SNR=10dB → noise power = 10% of signal power (moderate)
        SNR=5dB  → noise power ≈ 30% (significant)
        SNR=0dB  → noise power = signal power (severe)
        SNR=-5dB → noise power ≈ 3× signal power (extreme)
    """
    if target_snr_db == float("inf") or target_snr_db > 100:
        return signal.copy()

    signal_power = np.mean(signal ** 2)

    if noise_type == "gaussian":
        noise = np.random.randn(*signal.shape).astype(np.float32)
    elif noise_type == "laplacian":
        noise = np.random.laplace(0, 1, signal.shape).astype(np.float32)
    else:
        noise = np.random.uniform(-1, 1, signal.shape).astype(np.float32)

    noise_power = np.mean(noise ** 2)

    # Scale noise to achieve target SNR
    scaling = np.sqrt(signal_power / (noise_power * (10 ** (target_snr_db / 10))))
    noisy_signal = signal + noise / scaling

    return noisy_signal.astype(np.float32)


# ============================================================
# 6. Z-score Normalization
# ============================================================

def normalize(signal: np.ndarray, method: str = "z-score") -> np.ndarray:
    """
    Normalize a signal.

    Args:
        signal: Input signal
        method: 'z-score' → (x - μ) / σ  |  'min-max' → [0, 1]

    Returns:
        Normalized signal
    """
    if method == "z-score":
        mu, std = np.mean(signal), np.std(signal)
        if std < 1e-8:
            return signal - mu
        return (signal - mu) / std
    else:  # min-max
        lo, hi = np.min(signal), np.max(signal)
        if hi - lo < 1e-8:
            return signal - lo
        return (signal - lo) / (hi - lo)
