from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common.models import FaultDiagnosis, SensorReading
from .model import PredictiveMaintenanceClassifier

app = FastAPI(title="Predictive Maintenance API — Classical ML", version="1.0.0")

_classifier = PredictiveMaintenanceClassifier()
_model_load_error: Optional[str] = None


@app.on_event("startup")
async def _load_model() -> None:
    global _model_load_error
    model_path = os.environ.get("MODEL_PATH", "model.joblib")
    try:
        _classifier.load(model_path)
    except FileNotFoundError:
        _model_load_error = f"Model file not found: {model_path}"
    except Exception as exc:
        _model_load_error = str(exc)


class SensorReadingRequest(BaseModel):
    asset_id: str
    vibration_x: list[float]
    vibration_y: list[float]
    vibration_z: list[float]
    temperature: float
    rpm: float = Field(gt=0)
    sample_rate: int = Field(default=12800, gt=0)
    timestamp: Optional[float] = None

    def to_sensor_reading(self) -> SensorReading:
        return SensorReading(
            asset_id=self.asset_id,
            timestamp=self.timestamp or time.time(),
            vibration_x=np.array(self.vibration_x, dtype=np.float32),
            vibration_y=np.array(self.vibration_y, dtype=np.float32),
            vibration_z=np.array(self.vibration_z, dtype=np.float32),
            temperature=self.temperature,
            rpm=self.rpm,
            sample_rate=self.sample_rate,
        )


class DiagnosisResponse(BaseModel):
    asset_id: str
    timestamp: float
    fault_type: str
    severity: str
    confidence: float
    features: dict
    explanation: Optional[str] = None


def _require_model() -> None:
    if not _classifier.is_fitted:
        detail = _model_load_error or "Model not loaded"
        raise HTTPException(status_code=503, detail=detail)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _classifier.is_fitted}


@app.get("/model/info")
async def model_info() -> dict:
    _require_model()
    info = _classifier.training_info.copy()
    info["model_type"] = "VotingClassifier(RandomForest + GradientBoosting)"
    return info


@app.post("/predict", response_model=DiagnosisResponse)
async def predict(body: SensorReadingRequest) -> DiagnosisResponse:
    _require_model()
    try:
        reading = body.to_sensor_reading()
        diagnosis = _classifier.predict(reading)
        return DiagnosisResponse(**diagnosis.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[DiagnosisResponse])
async def predict_batch(bodies: list[SensorReadingRequest]) -> list[DiagnosisResponse]:
    _require_model()
    if not bodies:
        raise HTTPException(status_code=422, detail="Empty batch")
    try:
        readings = [b.to_sensor_reading() for b in bodies]
        diagnoses = _classifier.predict_batch(readings)
        return [DiagnosisResponse(**d.to_dict()) for d in diagnoses]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
