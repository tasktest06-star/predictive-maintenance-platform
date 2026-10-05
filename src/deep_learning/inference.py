from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.common.models import FaultDiagnosis, FaultType, Severity, SensorReading
from .models.cnn1d import FaultCNN
from .models.autoencoder import VibrationAutoencoder


_FAULT_TYPES = list(FaultType)
_WINDOW_SIZE = 2048


def _severity_from(confidence: float, anomaly_score: float, threshold: float) -> Severity:
    normalized_anomaly = min(anomaly_score / max(threshold, 1e-8), 3.0)
    if confidence > 0.9 and normalized_anomaly < 1.0:
        return Severity.NORMAL if _FAULT_TYPES[0] == FaultType.NORMAL else Severity.WARNING
    if confidence > 0.7 or normalized_anomaly > 2.0:
        return Severity.CRITICAL
    if confidence > 0.5 or normalized_anomaly > 1.5:
        return Severity.ALERT
    return Severity.WARNING


def _reading_to_tensor(reading: SensorReading, window_size: int = _WINDOW_SIZE) -> Tensor:
    def crop_or_pad(arr: np.ndarray) -> np.ndarray:
        if len(arr) >= window_size:
            return arr[:window_size]
        return np.pad(arr, (0, window_size - len(arr)))

    vx = crop_or_pad(reading.vibration_x).astype(np.float32)
    vy = crop_or_pad(reading.vibration_y).astype(np.float32)
    vz = crop_or_pad(reading.vibration_z).astype(np.float32)
    temp = np.full(window_size, reading.temperature, dtype=np.float32)

    channels = np.stack([vx, vy, vz, temp], axis=0)
    mean = channels.mean(axis=1, keepdims=True)
    std = channels.std(axis=1, keepdims=True) + 1e-8
    channels = (channels - mean) / std

    return torch.from_numpy(channels).unsqueeze(0)  # (1, 4, W)


def _rms_channel(reading: SensorReading, window_size: int = _WINDOW_SIZE) -> Tensor:
    def crop_or_pad(arr: np.ndarray) -> np.ndarray:
        if len(arr) >= window_size:
            return arr[:window_size]
        return np.pad(arr, (0, window_size - len(arr)))

    vx = crop_or_pad(reading.vibration_x).astype(np.float32)
    vy = crop_or_pad(reading.vibration_y).astype(np.float32)
    vz = crop_or_pad(reading.vibration_z).astype(np.float32)
    rms = np.sqrt((vx**2 + vy**2 + vz**2) / 3)
    rms = (rms - rms.mean()) / (rms.std() + 1e-8)
    return torch.from_numpy(rms).unsqueeze(0).unsqueeze(0)  # (1, 1, W)


class FaultDetector:
    def __init__(
        self,
        classifier_path: str | Path,
        autoencoder_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=True)
        self.classifier = FaultCNN(
            in_channels=checkpoint.get("in_channels", 4),
            n_classes=checkpoint.get("n_classes", 8),
            window_size=checkpoint.get("window_size", _WINDOW_SIZE),
        )
        self.classifier.load_state_dict(checkpoint["model_state"])
        self.classifier.to(self.device).eval()

        self.autoencoder: VibrationAutoencoder | None = None
        self._ae_threshold = float("inf")
        if autoencoder_path is not None:
            ae_ckpt = torch.load(autoencoder_path, map_location=self.device, weights_only=True)
            self.autoencoder = VibrationAutoencoder(
                window_size=ae_ckpt.get("window_size", _WINDOW_SIZE),
                latent_dim=ae_ckpt.get("latent_dim", 64),
            )
            self.autoencoder.load_state_dict(ae_ckpt["model_state"])
            self.autoencoder.to(self.device).eval()
            self._ae_threshold = ae_ckpt.get("anomaly_threshold", float("inf"))

    @torch.inference_mode()
    def predict(self, reading: SensorReading) -> FaultDiagnosis:
        x = _reading_to_tensor(reading, _WINDOW_SIZE).to(self.device)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_idx = int(probs.argmax().item())
        confidence = float(probs[predicted_idx].item())
        fault_type = _FAULT_TYPES[predicted_idx]

        anomaly_score = 0.0
        if self.autoencoder is not None:
            ae_input = _rms_channel(reading, _WINDOW_SIZE).to(self.device)
            recon = self.autoencoder(ae_input)
            anomaly_score = float(torch.nn.functional.mse_loss(recon, ae_input).item())

        # Reclassify as WARNING if autoencoder flags anomaly but classifier says normal
        if fault_type == FaultType.NORMAL and anomaly_score > self._ae_threshold:
            fault_type = FaultType.BEARING_WEAR  # conservative fallback
            confidence = min(confidence, 0.5)

        severity = _severity_from(confidence, anomaly_score, self._ae_threshold)
        # Normal reading should always map to NORMAL severity
        if fault_type == FaultType.NORMAL and anomaly_score <= self._ae_threshold:
            severity = Severity.NORMAL

        return FaultDiagnosis(
            asset_id=reading.asset_id,
            timestamp=reading.timestamp,
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            features={
                "anomaly_score": anomaly_score,
                "top_probs": {_FAULT_TYPES[i].value: float(probs[i]) for i in range(len(_FAULT_TYPES))},
            },
        )

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, Any]) -> None:
        torch.save(
            {
                "model_state": self.classifier.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "in_channels": 4,
                "n_classes": self.classifier.n_classes,
                "window_size": _WINDOW_SIZE,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> "FaultDetector":
        return cls(classifier_path=path, device=device)

    def warmup(self, n_samples: int = 10) -> None:
        dummy = torch.zeros(1, 4, _WINDOW_SIZE, device=self.device)
        for _ in range(n_samples):
            _ = self.classifier(dummy)
