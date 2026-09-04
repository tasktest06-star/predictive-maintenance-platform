"""
FastAPI inference server for BearingFaultCNN1D.

Endpoints:
    POST /predict           — single 2048-point FFT array, returns fault class + confidence
    POST /predict_batch     — batch of FFT arrays
    GET  /health            — liveness / readiness probe

Model is loaded from an MLflow model URI (e.g. runs:/<run_id>/model or
models:/BearingFaultCNN1D/Production) specified via the MLFLOW_MODEL_URI
environment variable, or from a local checkpoint path via CHECKPOINT_PATH.

Usage::

    # From MLflow model registry
    MLFLOW_MODEL_URI="models:/BearingFaultCNN1D/Production" uvicorn serve:app --host 0.0.0.0 --port 8080

    # From local checkpoint
    CHECKPOINT_PATH=./runs/best_model.pt uvicorn serve:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from model import FAULT_CLASSES, NUM_CLASSES, BearingFaultCNN1D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_MODEL: BearingFaultCNN1D | None = None
_DEVICE: torch.device = torch.device("cpu")
_MODEL_URI: str = ""
_LOAD_TIME: float = 0.0


def _load_model_from_mlflow(uri: str) -> BearingFaultCNN1D:
    """Load model from MLflow model URI."""
    import mlflow.pytorch
    logger.info("Loading model from MLflow URI: %s", uri)
    model = mlflow.pytorch.load_model(uri, map_location=_DEVICE)
    model.eval()
    return model


def _load_model_from_checkpoint(ckpt_path: str) -> BearingFaultCNN1D:
    """Load model from a local .pt checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    model = BearingFaultCNN1D(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint: %s (epoch %d, val_f1 %.4f)",
                ckpt_path, ckpt.get("epoch", -1), ckpt.get("val_f1", float("nan")))
    return model


# ---------------------------------------------------------------------------
# Lifespan: load model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MODEL, _DEVICE, _MODEL_URI, _LOAD_TIME

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", _DEVICE)

    t0 = time.time()
    mlflow_uri = os.environ.get("MLFLOW_MODEL_URI", "")
    ckpt_path = os.environ.get("CHECKPOINT_PATH", "")

    if mlflow_uri:
        _MODEL = _load_model_from_mlflow(mlflow_uri)
        _MODEL_URI = mlflow_uri
    elif ckpt_path:
        _MODEL = _load_model_from_checkpoint(ckpt_path)
        _MODEL_URI = f"file://{Path(ckpt_path).resolve()}"
    else:
        logger.warning(
            "Neither MLFLOW_MODEL_URI nor CHECKPOINT_PATH set — "
            "server will start but /predict will return 503."
        )
        _MODEL_URI = "not_loaded"

    if _MODEL is not None:
        _MODEL.to(_DEVICE)
        _LOAD_TIME = time.time() - t0
        logger.info("Model loaded in %.2fs", _LOAD_TIME)

    yield  # Application runs here

    logger.info("Shutting down inference server.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BearingFaultCNN1D Inference Server",
    description=(
        "Supervised 1D CNN for industrial bearing fault classification. "
        "Classifies 12 fault types from raw FFT magnitude spectra. "
        "Reference: arXiv 2602.09699."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-sample prediction request."""

    fft_spectrum: list[float] = Field(
        ...,
        min_length=2048,
        max_length=2048,
        description="2048-point FFT magnitude spectrum (one-sided, normalised).",
        examples=[[0.0] * 2048],
    )

    @field_validator("fft_spectrum")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) != 2048:
            raise ValueError(f"fft_spectrum must have exactly 2048 values, got {len(v)}")
        return v


class FaultPrediction(BaseModel):
    """Single-sample prediction result."""

    fault_class: str = Field(..., description="Predicted fault class label.")
    fault_class_id: int = Field(..., description="Predicted fault class index (0-11).")
    confidence: float = Field(..., description="Model confidence (softmax probability) for the predicted class.")
    class_probabilities: dict[str, float] = Field(
        ..., description="Softmax probability for each of the 12 fault classes."
    )
    inference_time_ms: float = Field(..., description="Server-side inference time in milliseconds.")


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    fft_spectra: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=512,
        description="List of 2048-point FFT spectra (up to 512 per call).",
    )

    @field_validator("fft_spectra")
    @classmethod
    def check_spectra(cls, v: list[list[float]]) -> list[list[float]]:
        for i, s in enumerate(v):
            if len(s) != 2048:
                raise ValueError(f"Spectrum at index {i} has length {len(s)}, expected 2048")
        return v


class BatchPredictResponse(BaseModel):
    predictions: list[FaultPrediction]
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str
    device: str
    num_classes: int
    fault_classes: list[str]
    model_load_time_s: float


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_inference(spectra: np.ndarray) -> tuple[list[int], list[float], np.ndarray, float]:
    """
    Run model inference on a batch of FFT spectra.

    Args:
        spectra: (N, 2048) float32 numpy array.

    Returns:
        class_ids:     List of predicted class indices.
        confidences:   List of max softmax probabilities.
        all_probs:     (N, num_classes) softmax probability matrix.
        elapsed_ms:    Total inference time in milliseconds.
    """
    if _MODEL is None:
        raise RuntimeError("Model not loaded")

    x = torch.from_numpy(spectra[:, np.newaxis, :]).to(_DEVICE)  # (N, 1, 2048)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = _MODEL(x).cpu().numpy()  # (N, num_classes)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    class_ids = probs.argmax(axis=1).tolist()
    confidences = probs.max(axis=1).tolist()
    return class_ids, confidences, probs, elapsed_ms


def _format_prediction(
    class_id: int,
    confidence: float,
    all_probs: np.ndarray,
    inference_time_ms: float,
) -> FaultPrediction:
    return FaultPrediction(
        fault_class=FAULT_CLASSES[class_id],
        fault_class_id=class_id,
        confidence=round(float(confidence), 6),
        class_probabilities={
            FAULT_CLASSES[i]: round(float(all_probs[i]), 6)
            for i in range(len(FAULT_CLASSES))
        },
        inference_time_ms=round(inference_time_ms, 3),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns model status, loaded URI, device, and fault class registry.
    """
    return HealthResponse(
        status="ok" if _MODEL is not None else "model_not_loaded",
        model_loaded=_MODEL is not None,
        model_uri=_MODEL_URI,
        device=str(_DEVICE),
        num_classes=NUM_CLASSES,
        fault_classes=FAULT_CLASSES,
        model_load_time_s=round(_LOAD_TIME, 3),
    )


@app.post("/predict", response_model=FaultPrediction, tags=["Inference"])
def predict(request: PredictRequest) -> FaultPrediction:
    """
    Classify a single vibration window.

    Accepts a 2048-point FFT magnitude spectrum (one-sided, normalised).
    Returns the predicted fault class, confidence score, and full softmax
    distribution over all 12 fault classes.
    """
    if _MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Set MLFLOW_MODEL_URI or CHECKPOINT_PATH.",
        )

    try:
        spectrum = np.array(request.fft_spectrum, dtype=np.float32).reshape(1, 2048)
        class_ids, confidences, all_probs, elapsed_ms = _run_inference(spectrum)
        return _format_prediction(class_ids[0], confidences[0], all_probs[0], elapsed_ms)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Classify a batch of vibration windows.

    Accepts up to 512 FFT spectra in a single request.
    Returns a prediction for each input plus total batch inference time.
    """
    if _MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Set MLFLOW_MODEL_URI or CHECKPOINT_PATH.",
        )

    try:
        spectra = np.array(request.fft_spectra, dtype=np.float32)  # (N, 2048)
        class_ids, confidences, all_probs, total_ms = _run_inference(spectra)
        per_sample_ms = total_ms / len(class_ids)

        predictions = [
            _format_prediction(class_ids[i], confidences[i], all_probs[i], per_sample_ms)
            for i in range(len(class_ids))
        ]
        return BatchPredictResponse(
            predictions=predictions,
            total_inference_time_ms=round(total_ms, 3),
        )
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        reload=False,
        workers=1,  # Single worker to share GPU memory
        log_level="info",
    )
