"""
Graph Attention Network (GAT) Encoder
======================================
Encodes time-frequency graphs into node embeddings.

Key difference from DAGCN's ChebConv:
- GAT uses attention to weight neighbor importance
- Better for noisy settings: can learn to ignore noise nodes
- No need for torch_geometric — pure PyTorch implementation

Reference: Veličković et al., "Graph Attention Networks", ICLR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GraphAttentionLayer(nn.Module):
    """
    Single GAT layer: attends over neighbors of each node.

    Attention: e_{ij} = LeakyReLU(a^T · [W·h_i || W·h_j])
               α_{ij} = softmax(e_{ij}) over neighbors j
               h'_i = σ(∑_j α_{ij} · W·h_j)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3, alpha: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, N, in_dim)   node features
            adj: (B, N, N)        adjacency (can be weighted)

        Returns:
            out: (B, N, out_dim)  updated node features
        """
        B, N, _ = x.shape

        # Linear transformation
        Wh = self.W(x)  # (B, N, out_dim)

        # Attention coefficients
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, out_dim)
        Wh2 = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, out_dim)
        Wh_cat = torch.cat([Wh1, Wh2], dim=-1)         # (B, N, N, 2*out_dim)

        e = self.leaky_relu(Wh_cat @ self.a).squeeze(-1)  # (B, N, N)

        # Mask non-neighbors
        e = e.masked_fill(adj == 0, float("-inf"))

        # Softmax over neighbors
        attention = F.softmax(e, dim=-1)  # (B, N, N)
        attention = self.dropout_layer(attention)

        # Aggregate
        out = attention @ Wh  # (B, N, out_dim)
        return out


class MultiHeadGATLayer(nn.Module):
    """Multi-head GAT layer with concatenation."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_dim, out_dim, dropout) for _ in range(heads)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Concatenate outputs from all heads."""
        head_outs = [att(x, adj) for att in self.attentions]
        return torch.cat(head_outs, dim=-1)  # (B, N, heads * out_dim)


class GATEncoder(nn.Module):
    """
    GAT encoder for time-frequency graphs.

    Architecture: GAT → GAT (2 layers, multi-head)
    Outputs: node embeddings + pairwise distance matrix
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_dim:     Input node feature dimension (= patch_size², e.g., 64 for 8×8)
            hidden_dim: Hidden dimension per head
            out_dim:    Output node embedding dimension
            heads:      Number of attention heads
            dropout:    Dropout rate
        """
        super().__init__()
        self.out_dim = out_dim

        # Layer 1: multi-head GAT
        self.gat1 = MultiHeadGATLayer(in_dim, hidden_dim, heads, dropout)
        self.bn1 = nn.BatchNorm1d(heads * hidden_dim)

        # Layer 2: single-head GAT (output)
        self.gat2 = GraphAttentionLayer(heads * hidden_dim, out_dim, dropout)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feat: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feat: (B, N, D_in)  node features
            adj:       (B, N, N)     adjacency matrix

        Returns:
            H:     (B, N, out_dim)  node embeddings
            C:     (B, N, N)        pairwise distance matrix between nodes
        """
        # Layer 1
        x = self.gat1(node_feat, adj)          # (B, N, heads*hidden_dim)
        x = x.transpose(1, 2)                   # (B, heads*hidden_dim, N) for BatchNorm1d
        x = self.bn1(x)
        x = x.transpose(1, 2)                   # (B, N, heads*hidden_dim)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.gat2(x, adj)                    # (B, N, out_dim)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)
        H = F.elu(x)                             # node embeddings

        # Pairwise distance matrix C (Euclidean distance in embedding space)
        C = torch.cdist(H, H, p=2)               # (B, N, N)

        return H, C
