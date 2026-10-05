from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.common.models import FaultType, SensorReading


_FAULT_TYPES = list(FaultType)
_FAULT_INDEX: dict[FaultType, int] = {ft: i for i, ft in enumerate(_FAULT_TYPES)}


class VibrationDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        readings: Sequence[SensorReading],
        labels: Sequence[FaultType],
        window_size: int = 2048,
    ) -> None:
        assert len(readings) == len(labels)
        self.window_size = window_size
        self._tensors: list[Tensor] = []
        self._labels: list[int] = []

        for reading, label in zip(readings, labels):
            tensor = self._to_tensor(reading)
            self._tensors.append(tensor)
            self._labels.append(_FAULT_INDEX[label])

    def _to_tensor(self, reading: SensorReading) -> Tensor:
        def crop_or_pad(arr: np.ndarray) -> np.ndarray:
            if len(arr) >= self.window_size:
                return arr[: self.window_size]
            return np.pad(arr, (0, self.window_size - len(arr)))

        vx = crop_or_pad(reading.vibration_x).astype(np.float32)
        vy = crop_or_pad(reading.vibration_y).astype(np.float32)
        vz = crop_or_pad(reading.vibration_z).astype(np.float32)
        # Temperature broadcast to match window_size
        temp = np.full(self.window_size, reading.temperature, dtype=np.float32)

        channels = np.stack([vx, vy, vz, temp], axis=0)  # (4, window_size)

        # Normalize each channel independently
        mean = channels.mean(axis=1, keepdims=True)
        std = channels.std(axis=1, keepdims=True) + 1e-8
        channels = (channels - mean) / std

        return torch.from_numpy(channels)

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self._tensors[idx], self._labels[idx]


def class_weights(labels: Sequence[FaultType]) -> Tensor:
    counts = np.zeros(len(FaultType), dtype=np.float32)
    for label in labels:
        counts[_FAULT_INDEX[label]] += 1
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (len(FaultType) * counts)
    return torch.from_numpy(weights)
