from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

from src.classical_ml.feature_extraction import FeatureExtractor
from src.classical_ml.model import PredictiveMaintenanceClassifier
from src.common.data_generator import SensorDataGenerator
from src.common.models import FaultType


@pytest.fixture(scope="module")
def generator() -> SensorDataGenerator:
    return SensorDataGenerator(rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


def test_feature_extraction_shape(generator: SensorDataGenerator, extractor: FeatureExtractor) -> None:
    r1 = generator.generate("PUMP-001", FaultType.NORMAL)
    r2 = generator.generate("PUMP-001", FaultType.BEARING_WEAR)
    f1 = extractor.extract(r1)
    f2 = extractor.extract(r2)
    assert f1.shape == f2.shape
    assert f1.ndim == 1
    assert len(f1) == len(extractor.feature_names)


def test_feature_extraction_fault_separability(
    generator: SensorDataGenerator, extractor: FeatureExtractor
) -> None:
    normal_readings = [generator.generate("A", FaultType.NORMAL) for _ in range(20)]
    fault_readings = [generator.generate("A", FaultType.BEARING_WEAR) for _ in range(20)]

    normal_features = np.stack([extractor.extract(r) for r in normal_readings])
    fault_features = np.stack([extractor.extract(r) for r in fault_readings])

    # RMS (index 0) should be higher for bearing wear on average
    normal_rms_mean = float(np.mean(normal_features[:, 0]))
    fault_rms_mean = float(np.mean(fault_features[:, 0]))
    assert fault_rms_mean > normal_rms_mean, (
        f"Bearing wear RMS {fault_rms_mean:.4f} not greater than normal {normal_rms_mean:.4f}"
    )


def test_classifier_train_predict(generator: SensorDataGenerator) -> None:
    readings, labels = generator.generate_dataset(n_per_class=50)
    # Sequential split would leave NORMAL entirely in test — use stratified split
    train_r, test_r, train_l, test_l = train_test_split(
        readings, labels, test_size=0.2, stratify=labels, random_state=42
    )

    clf = PredictiveMaintenanceClassifier()
    clf.fit(train_r, train_l)

    diagnoses = clf.predict_batch(test_r)
    assert len(diagnoses) == len(test_r)

    correct = sum(d.fault_type == l for d, l in zip(diagnoses, test_l))
    accuracy = correct / len(test_r)
    assert accuracy > 0.6, f"Accuracy {accuracy:.3f} below threshold 0.6"


def test_classifier_save_load(generator: SensorDataGenerator) -> None:
    readings, labels = generator.generate_dataset(n_per_class=30)
    clf = PredictiveMaintenanceClassifier()
    clf.fit(readings, labels)

    test_reading = generator.generate("TEST-001", FaultType.NORMAL)
    pred_before = clf.predict(test_reading)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name

    try:
        clf.save(path)
        clf2 = PredictiveMaintenanceClassifier()
        clf2.load(path)
        pred_after = clf2.predict(test_reading)
        assert pred_before.fault_type == pred_after.fault_type
        assert abs(pred_before.confidence - pred_after.confidence) < 1e-6
    finally:
        os.unlink(path)


def test_api_health() -> None:
    from src.classical_ml.api import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_api_predict(generator: SensorDataGenerator) -> None:
    # Pre-fit a model and inject it into the API module
    readings, labels = generator.generate_dataset(n_per_class=30)

    from src.classical_ml import api as api_module
    clf = PredictiveMaintenanceClassifier()
    clf.fit(readings, labels)
    api_module._classifier = clf

    from src.classical_ml.api import app
    client = TestClient(app)

    reading = generator.generate("MOTOR-001", FaultType.BEARING_WEAR)
    body = {
        "asset_id": reading.asset_id,
        "vibration_x": reading.vibration_x.tolist(),
        "vibration_y": reading.vibration_y.tolist(),
        "vibration_z": reading.vibration_z.tolist(),
        "temperature": reading.temperature,
        "rpm": reading.rpm,
        "sample_rate": reading.sample_rate,
        "timestamp": reading.timestamp,
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert "fault_type" in data
    assert "severity" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["fault_type"] in [f.value for f in FaultType]
