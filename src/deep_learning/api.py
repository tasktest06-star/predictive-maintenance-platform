from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.common.models import FaultDiagnosis, SensorReading

logger = logging.getLogger(__name__)

_detector: Any = None  # FaultDetector, loaded at startup


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _detector
    classifier_path = os.environ.get("CLASSIFIER_PATH")
    autoencoder_path = os.environ.get("AUTOENCODER_PATH")

    if classifier_path:
        try:
            from .inference import FaultDetector
            _detector = FaultDetector(
                classifier_path=classifier_path,
                autoencoder_path=autoencoder_path,
                device=os.environ.get("DEVICE", "cpu"),
            )
            _detector.warmup()
            logger.info("FaultDetector loaded from %s", classifier_path)
        except Exception as exc:
            logger.error("Failed to load FaultDetector: %s", exc)
    else:
        logger.warning("CLASSIFIER_PATH not set — /predict endpoints will return 503")

    yield


app = FastAPI(title="Predictive Maintenance — Deep Learning", lifespan=lifespan)


class SensorPayload(BaseModel):
    asset_id: str
    timestamp: float = Field(default_factory=time.time)
    vibration_x: list[float]
    vibration_y: list[float]
    vibration_z: list[float]
    temperature: float
    rpm: float
    sample_rate: int = 12800

    @field_validator("vibration_x", "vibration_y", "vibration_z")
    @classmethod
    def non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("vibration channel must not be empty")
        return v

    def to_reading(self) -> SensorReading:
        return SensorReading(
            asset_id=self.asset_id,
            timestamp=self.timestamp,
            vibration_x=np.array(self.vibration_x, dtype=np.float32),
            vibration_y=np.array(self.vibration_y, dtype=np.float32),
            vibration_z=np.array(self.vibration_z, dtype=np.float32),
            temperature=self.temperature,
            rpm=self.rpm,
            sample_rate=self.sample_rate,
        )


def _require_detector() -> Any:
    if _detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Set CLASSIFIER_PATH env var and restart.",
        )
    return _detector


@app.post("/predict")
def predict(payload: SensorPayload) -> dict[str, Any]:
    detector = _require_detector()
    diagnosis: FaultDiagnosis = detector.predict(payload.to_reading())
    return diagnosis.to_dict()


@app.post("/predict/batch")
def predict_batch(payloads: list[SensorPayload]) -> list[dict[str, Any]]:
    if not payloads:
        return []
    detector = _require_detector()
    return [detector.predict(p.to_reading()).to_dict() for p in payloads]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": os.environ.get("DEVICE", "cpu"),
        "model_loaded": _detector is not None,
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    detector = _require_detector()
    return {
        "parameter_count": detector.classifier.parameter_count(),
        "n_classes": detector.classifier.class_count(),
        "autoencoder_loaded": detector.autoencoder is not None,
        "anomaly_threshold": detector._ae_threshold,
    }
