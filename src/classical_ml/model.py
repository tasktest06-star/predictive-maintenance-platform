from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.common.models import FaultDiagnosis, FaultType, Severity, SensorReading
from .feature_extraction import FeatureExtractor


def _confidence_to_severity(fault: FaultType, confidence: float) -> Severity:
    if fault == FaultType.NORMAL:
        return Severity.NORMAL
    if confidence > 0.85:
        return Severity.CRITICAL
    if confidence > 0.70:
        return Severity.ALERT
    if confidence > 0.50:
        return Severity.WARNING
    return Severity.NORMAL


class PredictiveMaintenanceClassifier:
    def __init__(self) -> None:
        self._extractor = FeatureExtractor()
        self._label_encoder = LabelEncoder()
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted = False
        self._training_info: dict = {}

    def _build_pipeline(self) -> Pipeline:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=4,
            n_jobs=-1, random_state=42
        )
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        voting = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
            n_jobs=-1,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=40)),
            ("classifier", voting),
        ])

    def _extract_matrix(self, readings: list[SensorReading]) -> np.ndarray:
        return np.stack([self._extractor.extract(r) for r in readings])

    def fit(self, readings: list[SensorReading], labels: list[FaultType]) -> None:
        X = self._extract_matrix(readings)
        y_str = [l.value for l in labels]
        y = self._label_encoder.fit_transform(y_str)
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        self._is_fitted = True
        self._training_info = {
            "n_samples": len(readings),
            "n_features": X.shape[1],
            "classes": list(self._label_encoder.classes_),
            "trained_at": time.time(),
        }

    def predict(self, reading: SensorReading) -> FaultDiagnosis:
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x = self._extractor.extract(reading).reshape(1, -1)
        proba = self._pipeline.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        fault_type = FaultType(self._label_encoder.classes_[pred_idx])
        severity = _confidence_to_severity(fault_type, confidence)

        # Include top-3 class probabilities and RF feature importances
        top3 = sorted(enumerate(proba), key=lambda t: t[1], reverse=True)[:3]
        features: dict = {
            "class_probabilities": {
                self._label_encoder.classes_[i]: round(float(p), 4)
                for i, p in top3
            }
        }
        # Extract RF importances from the selected features
        try:
            selector = self._pipeline.named_steps["selector"]
            rf = self._pipeline.named_steps["classifier"].estimators_[0]
            selected_names = [
                self._extractor.feature_names[i]
                for i in selector.get_support(indices=True)
            ]
            importances = rf.feature_importances_
            top_imp = sorted(zip(selected_names, importances), key=lambda t: t[1], reverse=True)[:5]
            features["top_features"] = {name: round(float(imp), 4) for name, imp in top_imp}
        except Exception:
            pass

        return FaultDiagnosis(
            asset_id=reading.asset_id,
            timestamp=reading.timestamp,
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            features=features,
        )

    def predict_batch(self, readings: list[SensorReading]) -> list[FaultDiagnosis]:
        return [self.predict(r) for r in readings]

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Nothing to save: model not fitted.")
        payload = {
            "pipeline": self._pipeline,
            "label_encoder": self._label_encoder,
            "training_info": self._training_info,
        }
        joblib.dump(payload, path, compress=3)

    def load(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        self._pipeline = payload["pipeline"]
        self._label_encoder = payload["label_encoder"]
        self._training_info = payload.get("training_info", {})
        self._is_fitted = True

    @property
    def training_info(self) -> dict:
        return self._training_info

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
