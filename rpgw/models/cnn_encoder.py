"""
1D-CNN Time-Frequency Feature Extractor
========================================
Adapted from DAGCN (Li et al., IEEE TIM 2021).

Takes raw vibration signal → extracts features → feeds to graph builder.
Architecture: 4 Conv1d layers + BatchNorm + ReLU + MaxPool
Output: feature vectors per time step (used as graph node initial features)
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """1D CNN for raw vibration signal feature extraction."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list = None,
        kernel_sizes: list = None,
        feature_dim: int = 256,
    ):
        """
        Args:
            in_channels:    Input channels (1 for raw vibration)
            conv_channels:  List of conv layer output channels
            kernel_sizes:   Kernel size per conv layer
            feature_dim:    Final feature dimension
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [16, 32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [15, 3, 3, 3]

        layers = []
        prev_ch = in_channels
        for i, (out_ch, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            # MaxPool every other layer
            if i % 2 == 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_ch = out_ch

        # Adaptive pooling → fixed output length
        layers.append(nn.AdaptiveMaxPool1d(4))
        self.conv = nn.Sequential(*layers)

        # Project to feature_dim
        self.project = nn.Sequential(
            nn.Linear(prev_ch * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L)  raw vibration signals

        Returns:
            features: (B, feature_dim)
        """
        x = self.conv(x)              # (B, C, 4)
        x = x.view(x.size(0), -1)     # (B, C*4)
        x = self.project(x)           # (B, feature_dim)
        return x
