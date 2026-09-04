"""
Supervised 1D CNN models for bearing fault classification.

References:
    arXiv 2602.09699 - 1D CNN on CWRU dataset
    arXiv 1909.07801 - CNN+LSTM hybrid for bearing fault diagnosis
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Fault class registry (12 classes)
# ---------------------------------------------------------------------------
FAULT_CLASSES = [
    "Normal",
    "BPFO",          # Ball Pass Frequency Outer race
    "BPFI",          # Ball Pass Frequency Inner race
    "BSF",           # Ball Spin Frequency (ball defect)
    "FTF",           # Fundamental Train Frequency (cage)
    "Unbalance",
    "Misalignment",
    "Lubrication",
    "Looseness",
    "Cavitation",
    "Gearbox",
    "Electrical",
]
NUM_CLASSES = len(FAULT_CLASSES)  # 12


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class Conv1DBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU + MaxPool + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pool_size: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # "same"-style padding
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# BearingFaultCNN1D
# ---------------------------------------------------------------------------

class BearingFaultCNN1D(nn.Module):
    """
    1D CNN for bearing fault classification operating on FFT magnitude spectra.

    Architecture (arXiv 2602.09699):
        4 x Conv1DBlock → Global Average Pooling → FC(256→128→num_classes) → Softmax

    Args:
        num_classes: Number of fault classes (default 12).
        input_length: Length of the 1D input spectrum (default 2048).
        dropout_fc: Dropout probability before the final FC layer.

    Input:
        x: Tensor of shape (batch, 1, 2048) — raw FFT magnitude spectrum.

    Output:
        Tensor of shape (batch, num_classes) — softmax class probabilities.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_length: int = 2048,
        dropout_fc: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_length = input_length

        # Feature extractor — 4 Conv1D blocks
        self.feature_extractor = nn.Sequential(
            # Block 1: 1 → 64, kernel 64, stride 2
            Conv1DBlock(1, 64, kernel_size=64, stride=2, pool_size=2, dropout=0.25),
            # Block 2: 64 → 128, kernel 32
            Conv1DBlock(64, 128, kernel_size=32, stride=1, pool_size=2, dropout=0.25),
            # Block 3: 128 → 256, kernel 16
            Conv1DBlock(128, 256, kernel_size=16, stride=1, pool_size=2, dropout=0.25),
            # Block 4: 256 → 256, kernel 8
            Conv1DBlock(256, 256, kernel_size=8, stride=1, pool_size=2, dropout=0.25),
        )

        # Global Average Pooling (reduces spatial dimension to 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 1, 2048) FFT magnitude spectrum.

        Returns:
            (batch, num_classes) softmax probabilities.
        """
        features = self.feature_extractor(x)   # (B, 256, L')
        pooled = self.gap(features)             # (B, 256, 1)
        logits = self.classifier(pooled)        # (B, num_classes)
        return F.softmax(logits, dim=1)

    def predict_class(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convenience wrapper returning (class_indices, confidences).

        Args:
            x: (batch, 1, 2048)

        Returns:
            class_idx: (batch,) int tensor of predicted class indices.
            confidence: (batch,) float tensor of max softmax probability.
        """
        probs = self.forward(x)
        confidence, class_idx = probs.max(dim=1)
        return class_idx, confidence

    @property
    def fault_class_names(self) -> list[str]:
        return FAULT_CLASSES[: self.num_classes]


# ---------------------------------------------------------------------------
# BearingFaultCNNLSTM  (arXiv 1909.07801)
# ---------------------------------------------------------------------------

class BearingFaultCNNLSTM(nn.Module):
    """
    CNN+LSTM hybrid for bearing fault diagnosis with temporal context.

    The CNN feature extractor encodes each window independently; the LSTM
    then models temporal dependencies across a sequence of consecutive windows.

    Reference: arXiv 1909.07801

    Args:
        num_classes:     Number of fault classes (default 12).
        sequence_length: Number of consecutive FFT windows per sample (default 10).
        lstm_hidden:     LSTM hidden state size (default 128).
        lstm_layers:     Number of stacked LSTM layers (default 2).
        dropout_lstm:    Dropout between LSTM layers.
        dropout_fc:      Dropout before final classifier.

    Input:
        x: Tensor of shape (batch, sequence_length, 1, 2048).

    Output:
        Tensor of shape (batch, num_classes) — softmax class probabilities derived
        from the last LSTM timestep.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        sequence_length: int = 10,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout_lstm: float = 0.3,
        dropout_fc: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.lstm_hidden = lstm_hidden

        # Shared CNN feature extractor (same blocks as BearingFaultCNN1D)
        self.feature_extractor = nn.Sequential(
            Conv1DBlock(1, 64, kernel_size=64, stride=2, pool_size=2, dropout=0.25),
            Conv1DBlock(64, 128, kernel_size=32, stride=1, pool_size=2, dropout=0.25),
            Conv1DBlock(128, 256, kernel_size=16, stride=1, pool_size=2, dropout=0.25),
            Conv1DBlock(256, 256, kernel_size=8, stride=1, pool_size=2, dropout=0.25),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # → (B*T, 256, 1)

        # LSTM operating over the sequence of CNN feature vectors
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Classifier head applied to the last LSTM output
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, sequence_length, 1, 2048) — sequence of FFT windows.

        Returns:
            (batch, num_classes) softmax probabilities from last timestep.
        """
        B, T, C, L = x.shape
        # Merge batch and time dims for shared CNN encoding
        x_flat = x.view(B * T, C, L)                    # (B*T, 1, 2048)
        feats = self.feature_extractor(x_flat)            # (B*T, 256, L')
        feats = self.gap(feats).squeeze(-1)               # (B*T, 256)
        feats_seq = feats.view(B, T, 256)                 # (B, T, 256)

        # LSTM over time dimension
        lstm_out, _ = self.lstm(feats_seq)                # (B, T, lstm_hidden)
        last_hidden = lstm_out[:, -1, :]                  # (B, lstm_hidden)

        logits = self.classifier(last_hidden)             # (B, num_classes)
        return F.softmax(logits, dim=1)

    def predict_class(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convenience wrapper returning (class_indices, confidences).

        Args:
            x: (batch, sequence_length, 1, 2048)

        Returns:
            class_idx: (batch,) int tensor.
            confidence: (batch,) float tensor.
        """
        probs = self.forward(x)
        confidence, class_idx = probs.max(dim=1)
        return class_idx, confidence

    @property
    def fault_class_names(self) -> list[str]:
        return FAULT_CLASSES[: self.num_classes]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test BearingFaultCNN1D
    cnn = BearingFaultCNN1D(num_classes=NUM_CLASSES).to(device)
    dummy = torch.randn(4, 1, 2048, device=device)
    out = cnn(dummy)
    assert out.shape == (4, NUM_CLASSES), f"Unexpected shape: {out.shape}"
    assert abs(out.sum(dim=1).mean().item() - 1.0) < 1e-5, "Softmax does not sum to 1"
    print(f"BearingFaultCNN1D OK — output shape: {out.shape}")

    # Test BearingFaultCNNLSTM
    clstm = BearingFaultCNNLSTM(num_classes=NUM_CLASSES, sequence_length=5).to(device)
    seq = torch.randn(4, 5, 1, 2048, device=device)
    out2 = clstm(seq)
    assert out2.shape == (4, NUM_CLASSES), f"Unexpected shape: {out2.shape}"
    print(f"BearingFaultCNNLSTM OK — output shape: {out2.shape}")

    total_cnn = sum(p.numel() for p in cnn.parameters())
    total_lstm = sum(p.numel() for p in clstm.parameters())
    print(f"BearingFaultCNN1D parameters: {total_cnn:,}")
    print(f"BearingFaultCNNLSTM parameters: {total_lstm:,}")
