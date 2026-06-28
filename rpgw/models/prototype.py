"""
Weighted Prototypical Few-shot Classifier
==========================================
Based on Prototypical Networks (Snell et al., NeurIPS 2017).

Key enhancement: prototypes are WEIGHTED by GW alignment quality.
Nodes with higher GW transport mass (better aligned) contribute more
to the prototype computation, reducing noise influence.

Why prototype learning for this task?
- Bearings have 10 fault classes → few-shot per class
- Prototypes naturally handle class imbalance
- Simple, interpretable, fast inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PrototypeClassifier(nn.Module):
    """
    Prototypical Network classifier with optional GW quality weighting.

    Training:
        1. Compute prototypes: c_k = mean(embeddings of support samples in class k)
        2. Classify queries: softmax(-distance(q, c_k))
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 10,
        distance: str = "euclidean",
        use_weighting: bool = True,
        temperature: float = 1.0,
    ):
        """
        Args:
            in_dim:        Embedding dimension
            num_classes:   Number of fault classes
            distance:      'euclidean' | 'cosine'
            use_weighting: Whether to weight prototypes by GW quality
            temperature:   Softmax temperature (lower = sharper)
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.distance = distance
        self.use_weighting = use_weighting
        self.temperature = temperature

    def compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute class prototypes from support embeddings.

        Args:
            embeddings: (N, D)  embedding vectors
            labels:     (N,)    class labels [0, num_classes-1]
            weights:    (N,)    GW alignment quality weights (optional)

        Returns:
            prototypes: (num_classes, D)
        """
        prototypes = torch.zeros(self.num_classes, self.in_dim, device=embeddings.device)
        counts = torch.zeros(self.num_classes, device=embeddings.device)

        for k in range(self.num_classes):
            mask = (labels == k)
            if mask.sum() == 0:
                # No support samples for this class → use random init
                prototypes[k] = torch.randn(self.in_dim, device=embeddings.device)
                counts[k] = 1
                continue

            if self.use_weighting and weights is not None:
                # Weighted mean: more aligned = more weight
                k_weights = weights[mask]
                k_weights = k_weights / (k_weights.sum() + 1e-8)
                prototypes[k] = (embeddings[mask] * k_weights.unsqueeze(1)).sum(dim=0)
            else:
                prototypes[k] = embeddings[mask].mean(dim=0)

            counts[k] = mask.sum().float()

        return prototypes

    def forward(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify query samples by distance to prototypes.

        Args:
            query_embeddings: (B, D)
            prototypes:       (num_classes, D)

        Returns:
            logits: (B, num_classes) — negative distance (higher = closer)
        """
        if self.distance == "euclidean":
            # ||q - c_k||²
            dists = torch.cdist(query_embeddings, prototypes, p=2) ** 2
        else:
            # 1 - cosine similarity
            q_norm = F.normalize(query_embeddings, dim=1)
            p_norm = F.normalize(prototypes, dim=1)
            dists = 1.0 - q_norm @ p_norm.T

        # Negative distance → logits (higher = more similar)
        logits = -dists / self.temperature
        return logits
