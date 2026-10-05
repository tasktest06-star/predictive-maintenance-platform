from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 4) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(pool),
            nn.Dropout(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FaultCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        n_classes: int = 8,
        window_size: int = 2048,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.features = nn.Sequential(
            _ConvBlock(in_channels, 32, kernel=31),
            _ConvBlock(32, 64, kernel=15),
            _ConvBlock(64, 128, kernel=7),
        )
        # Global average pooling collapses time dimension
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)       # (B, 128, T')
        x = self.gap(x).squeeze(-1)  # (B, 128)
        return self.classifier(x)  # (B, n_classes)

    def class_count(self) -> int:
        return self.n_classes

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
