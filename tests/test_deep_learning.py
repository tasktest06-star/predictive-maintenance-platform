from __future__ import annotations

import sys
import os
import time
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.data_generator import SensorDataGenerator
from src.common.models import FaultType, SensorReading
from src.deep_learning.dataset import VibrationDataset, class_weights
from src.deep_learning.models.cnn1d import FaultCNN
from src.deep_learning.models.autoencoder import VibrationAutoencoder
from src.deep_learning.trainer import Trainer


_WINDOW = 256  # smaller window for fast tests


def _make_reading(fault: FaultType = FaultType.NORMAL) -> SensorReading:
    gen = SensorDataGenerator(sample_rate=_WINDOW, duration_seconds=1.0, rng=np.random.default_rng(0))
    return gen.generate("TEST-01", fault_type=fault)


def _small_dataset() -> tuple[VibrationDataset, list[FaultType]]:
    gen = SensorDataGenerator(sample_rate=_WINDOW, duration_seconds=1.0, rng=np.random.default_rng(42))
    readings, labels = gen.generate_dataset(n_per_class=4, asset_ids=["A"])
    return VibrationDataset(readings, labels, window_size=_WINDOW), labels


def test_dataset_shape() -> None:
    ds, _ = _small_dataset()
    x, y = ds[0]
    assert x.shape == (4, _WINDOW), f"Expected (4, {_WINDOW}), got {x.shape}"
    assert isinstance(y, int)
    assert 0 <= y < len(FaultType)


def test_class_weights_shape() -> None:
    gen = SensorDataGenerator(sample_rate=_WINDOW, duration_seconds=1.0, rng=np.random.default_rng(1))
    _, labels = gen.generate_dataset(n_per_class=3, asset_ids=["A"])
    weights = class_weights(labels)
    assert weights.shape == (len(FaultType),)
    assert (weights > 0).all()


def test_cnn_forward_pass() -> None:
    model = FaultCNN(in_channels=4, n_classes=8, window_size=_WINDOW)
    x = torch.randn(4, 4, _WINDOW)
    logits = model(x)
    assert logits.shape == (4, 8)
    assert model.parameter_count() > 0
    assert model.class_count() == 8


def test_autoencoder_reconstruction() -> None:
    ae = VibrationAutoencoder(window_size=_WINDOW, latent_dim=16)
    x = torch.randn(2, 1, _WINDOW)
    out = ae(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_autoencoder_reconstruction_loss() -> None:
    ae = VibrationAutoencoder(window_size=_WINDOW, latent_dim=16)
    x = torch.randn(1, 1, _WINDOW)
    loss = ae.reconstruction_loss(x)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_trainer_one_epoch() -> None:
    ds, labels = _small_dataset()
    weights = class_weights(labels)
    split = max(1, len(ds) // 5)
    train_ds, val_ds = ds, ds  # tiny dataset; use same for speed
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = FaultCNN(in_channels=4, n_classes=8, window_size=_WINDOW)
    trainer = Trainer(model, train_loader, val_loader, device="cpu", lr=1e-3, weight=weights)

    train_metrics = trainer.train_epoch()
    assert "loss" in train_metrics and "accuracy" in train_metrics
    assert train_metrics["loss"] >= 0.0

    val_metrics = trainer.val_epoch()
    assert "loss" in val_metrics and "accuracy" in val_metrics


def test_trainer_fit_early_stop() -> None:
    ds, labels = _small_dataset()
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = FaultCNN(in_channels=4, n_classes=8, window_size=_WINDOW)
    trainer = Trainer(model, loader, loader, device="cpu", lr=1e-3)
    history = trainer.fit(epochs=3, early_stopping_patience=2)
    assert len(history["train_loss"]) <= 3


def test_inference_predict() -> None:
    """Test FaultDetector.predict by saving and loading a checkpoint."""
    model = FaultCNN(in_channels=4, n_classes=8, window_size=_WINDOW)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": 0,
            "metrics": {},
            "in_channels": 4,
            "n_classes": 8,
            "window_size": _WINDOW,
        },
        ckpt_path,
    )

    from src.deep_learning.inference import FaultDetector

    detector = FaultDetector(classifier_path=ckpt_path, device="cpu")
    reading = _make_reading(FaultType.BEARING_WEAR)
    diagnosis = detector.predict(reading)

    assert diagnosis.asset_id == reading.asset_id
    assert isinstance(diagnosis.fault_type, FaultType)
    assert 0.0 <= diagnosis.confidence <= 1.0

    os.unlink(ckpt_path)
