from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.common.data_generator import SensorDataGenerator
from src.common.models import FaultType, SensorReading
from .signal_processor import SignalProcessor

logger = logging.getLogger(__name__)


class QuickFaultClassifier:
    def __init__(self, n_estimators: int = 100, n_train_per_class: int = 150) -> None:
        self._clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        self._le = LabelEncoder()
        self._processor = SignalProcessor()
        self._n_train = n_train_per_class
        self._ready = False

    def fit(self, readings: list[SensorReading], labels: list[FaultType]) -> None:
        X = np.stack([self._processor.extract_features(r) for r in readings])
        y_str = [lbl.value for lbl in labels]
        y = self._le.fit_transform(y_str)
        self._clf.fit(X, y)
        self._ready = True
        logger.info("QuickFaultClassifier trained on %d samples", len(readings))

    def _ensure_trained(self) -> None:
        if self._ready:
            return
        logger.info("Auto-training classifier on synthetic data (%d per class)…", self._n_train)
        gen = SensorDataGenerator(rng=np.random.default_rng(0))
        readings, labels = gen.generate_dataset(n_per_class=self._n_train)
        self.fit(readings, labels)

    def predict(self, features: np.ndarray) -> tuple[FaultType, float]:
        self._ensure_trained()
        x = features.reshape(1, -1)
        label_idx = int(self._clf.predict(x)[0])
        proba = float(self._clf.predict_proba(x)[0][label_idx])
        fault_str = self._le.inverse_transform([label_idx])[0]
        return FaultType(fault_str), proba

    def predict_reading(self, reading: SensorReading) -> tuple[FaultType, float]:
        features = self._processor.extract_features(reading)
        return self.predict(features)
