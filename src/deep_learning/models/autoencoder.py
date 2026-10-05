from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VibrationAutoencoder(nn.Module):
    def __init__(self, window_size: int = 2048, latent_dim: int = 64) -> None:
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        # Use single-channel input (RMS of xyz vibration)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(4),                                             # -> W/4
            nn.Conv1d(16, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),                                             # -> W/16
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),                                     # -> (B, 64, 1)
        )
        self.bottleneck = nn.Linear(64, latent_dim)
        self.expand = nn.Linear(latent_dim, 64 * (window_size // 16))

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=15, padding=7),
        )
        self._anomaly_threshold: float = float("inf")

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x).squeeze(-1)  # (B, 64)
        return self.bottleneck(h)        # (B, latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        h = self.expand(z)                                            # (B, 64 * T_reduced)
        h = h.view(z.size(0), 64, self.window_size // 16)            # (B, 64, T_reduced)
        out = self.decoder(h)                                         # (B, 1, T')
        return F.interpolate(out, size=self.window_size, mode="linear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    @torch.inference_mode()
    def reconstruction_loss(self, x: Tensor) -> float:
        recon = self.forward(x)
        return float(F.mse_loss(recon, x).item())

    def set_anomaly_threshold(self, scores: list[float], percentile: float = 95.0) -> None:
        import numpy as np
        self._anomaly_threshold = float(np.percentile(scores, percentile))

    def is_anomaly(self, x: Tensor) -> bool:
        return self.reconstruction_loss(x) > self._anomaly_threshold
