"""
Visualization Tools
===================
t-SNE feature visualization + GW distance heatmap + training curves.

Usage:
    python scripts/visualize.py --ckpt checkpoints/best_model.pth
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE


def plot_tsne(features: np.ndarray, labels: np.ndarray, domains: np.ndarray,
              title: str = "t-SNE Feature Visualization", save_path: str = None):
    """
    Plot t-SNE of features colored by class and domain.

    Args:
        features: (N, D) feature matrix
        labels:   (N,) class labels
        domains:  (N,) 0=source, 1=target
        title:    Plot title
        save_path: Path to save figure
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feat_2d = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: colored by class
    scatter1 = axes[0].scatter(feat_2d[:, 0], feat_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=15)
    axes[0].set_title("Colored by Fault Class")
    plt.colorbar(scatter1, ax=axes[0])

    # Right: colored by domain
    colors = ["blue" if d == 0 else "red" for d in domains]
    axes[1].scatter(feat_2d[:, 0], feat_2d[:, 1], c=colors, alpha=0.7, s=15)
    axes[1].set_title("Colored by Domain (Blue=Source, Red=Target)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_gw_heatmap(cost_matrix: np.ndarray, title: str = "GW Cost Matrix",
                    save_path: str = None):
    """Plot GW distance matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cost_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    print("Visualization tools ready.")
    print("Run with --ckpt to visualize trained model features.")
