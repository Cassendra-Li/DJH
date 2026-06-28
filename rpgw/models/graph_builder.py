"""
Time-Frequency Graph Builder
=============================
Converts CWT time-frequency maps into graph representations.

Core operation (done in data/preprocess.py for efficiency, NOT here):
  1. Split TF map into patches → each patch = a graph node
  2. kNN connects nodes based on Euclidean distance between patch features

This module provides the PyTorch wrapper for graph construction
during training (when you need differentiable graph building).
"""

import torch
import torch.nn as nn
from typing import Tuple


class GraphBuilder(nn.Module):
    """
    Graph construction from time-frequency maps.

    Two modes:
    - precomputed:  graphs already built offline → just pass through
    - online:       build graphs on-the-fly (slower but flexible)
    """

    def __init__(
        self,
        patch_size: int = 8,
        k_neighbors: int = 8,
        mode: str = "precomputed",
    ):
        """
        Args:
            patch_size:   Size of each TF patch
            k_neighbors:  Number of neighbors in kNN graph
            mode:         'precomputed' | 'online'
        """
        super().__init__()
        self.patch_size = patch_size
        self.k_neighbors = k_neighbors
        self.mode = mode

    def forward(
        self,
        cwt_map: torch.Tensor,
        node_feat: torch.Tensor = None,
        adj: torch.Tensor = None,
        cost: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cwt_map:    (B, scales, time_steps)  raw CWT maps
            node_feat:  (B, N, D)  precomputed node features (optional)
            adj:        (B, N, N)  precomputed adj matrix (optional)
            cost:       (B, N, N)  precomputed distance matrix (optional)

        Returns:
            node_feat:  (B, N, D)  node features
            adj:        (B, N, N)  adjacency matrix
            cost:       (B, N, N)  pairwise distance matrix
        """
        if self.mode == "precomputed":
            assert node_feat is not None, "node_feat required in precomputed mode"
            return node_feat, adj, cost

        # Online mode: build graph from cwt_map
        # (Simplified — full implementation mirrors preprocess.build_graph)
        raise NotImplementedError(
            "Online graph building not yet implemented. "
            "Use precomputed mode (graphs built in Dataset)."
        )
