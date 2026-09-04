"""
FaultFormer FastAPI Inference Server
=====================================
Serves FaultFormerClassifier and FaultFormerRUL models loaded from MLflow
model registry. Provides three endpoints:

  POST /embed     → 256-dim CLS embedding for a vibration window
                    (for similarity search, clustering, visualization)

  POST /classify  → fault class name + confidence per class

  POST /rul       → RUL estimate as fraction [0,1] + estimated days

  POST /similar   → top-5 most similar historical events (cosine similarity
                    over CLS embeddings) — nearest-neighbor explainability

Similarity search enables explainability: when a fault is detected, the UI
can show "This looks like 3 previous cases of inner-race defect at bearing B-47".

Usage:
  # Development
  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

  # Production
  uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4

Environment variables:
  MLFLOW_TRACKING_URI     MLflow server (default: ./mlruns)
  MODEL_NAME              MLflow model name (default: faultformer_classifier)
  MODEL_STAGE             MLflow stage (default: Production)
  CLASSIFIER_PATH         Direct path to classifier .pt file (overrides MLflow)
  RUL_MODEL_PATH          Direct path to RUL .pt file
  EMBEDDING_STORE_PATH    Path to historical embeddings .npz file
  TOTAL_LIFE_DAYS         Total bearing life in days for RUL conversion (default 365)
  DEVICE                  torch device (default: auto-detect cuda/cpu)
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from model import (
    FaultFormer,
    FaultFormerClassifier,
    FaultFormerPretraining,
    FaultFormerRUL,
    SIGNAL_LENGTH,
    NUM_FAULT_CLASSES,
)


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FaultFormer Inference API",
    description=(
        "Transformer foundation model for vibration-based fault diagnosis. "
        "Pretrained via masked patch reconstruction (arXiv:2312.02380). "
        "Endpoints: /embed, /classify, /rul, /similar"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model Registry (loaded once on startup)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Holds loaded model instances. Loaded lazily on first request."""

    def __init__(self):
        self.classifier: Optional[FaultFormerClassifier] = None
        self.rul_model: Optional[FaultFormerRUL] = None
        self.device = self._resolve_device()
        self.embedding_store: Optional["EmbeddingStore"] = None

    @staticmethod
    def _resolve_device() -> torch.device:
        device_str = os.getenv("DEVICE", "auto")
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def load_classifier(self) -> FaultFormerClassifier:
        """Load classifier from MLflow registry or direct path."""
        if self.classifier is not None:
            return self.classifier

        # Option 1: Direct path (for development / testing)
        direct_path = os.getenv("CLASSIFIER_PATH")
        if direct_path and Path(direct_path).exists():
            print(f"Loading classifier from: {direct_path}")
            checkpoint = torch.load(direct_path, map_location="cpu")
            backbone = FaultFormerPretraining.load_backbone(
                os.getenv("BACKBONE_PATH", "models/faultformer_pretrained.pt")
            )
            model = FaultFormerClassifier(backbone=backbone, num_classes=NUM_FAULT_CLASSES)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.classifier = model.to(self.device)
            return self.classifier

        # Option 2: MLflow model registry
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        model_name = os.getenv("MODEL_NAME", "faultformer_classifier")
        stage = os.getenv("MODEL_STAGE", "Production")

        mlflow.set_tracking_uri(mlflow_uri)
        try:
            model_uri = f"models:/{model_name}/{stage}"
            print(f"Loading classifier from MLflow: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            model.eval()
            self.classifier = model.to(self.device)
            return self.classifier
        except Exception as e:
            print(f"WARNING: Could not load from MLflow ({e}). Loading default stub.")
            # Fallback: untrained model for API development testing
            backbone = FaultFormer()
            model = FaultFormerClassifier(backbone=backbone, num_classes=NUM_FAULT_CLASSES)
            model.eval()
            self.classifier = model.to(self.device)
            return self.classifier

    def load_rul_model(self) -> FaultFormerRUL:
        """Load RUL model (shares backbone with classifier)."""
        if self.rul_model is not None:
            return self.rul_model

        rul_path = os.getenv("RUL_MODEL_PATH")
        if rul_path and Path(rul_path).exists():
            print(f"Loading RUL model from: {rul_path}")
            # Share backbone with classifier for efficiency
            classifier = self.load_classifier()
            rul = FaultFormerRUL(backbone=classifier.backbone)
            checkpoint = torch.load(rul_path, map_location="cpu")
            rul.regression_head.load_state_dict(checkpoint["regression_head_state_dict"])
            rul.eval()
            self.rul_model = rul.to(self.device)
        else:
            # Use classifier backbone without trained RUL head (development mode)
            print("WARNING: No RUL model path set. Using untrained RUL head.")
            classifier = self.load_classifier()
            self.rul_model = FaultFormerRUL(backbone=classifier.backbone).to(self.device)

        return self.rul_model


class EmbeddingStore:
    """
    In-memory store of historical CLS embeddings for nearest-neighbor search.

    Populated from a .npz file:
      embeddings: (N, 256) float32 — CLS embeddings
      labels:     (N,) int — fault class indices
      timestamps: (N,) str — ISO timestamps
      asset_ids:  (N,) str — asset/bearing identifiers
      event_ids:  (N,) str — unique event IDs for linking to CMMS records
    """

    def __init__(self, store_path: Optional[str] = None):
        self.embeddings: Optional[np.ndarray] = None  # (N, 256)
        self.labels: Optional[np.ndarray] = None
        self.timestamps: Optional[np.ndarray] = None
        self.asset_ids: Optional[np.ndarray] = None
        self.event_ids: Optional[np.ndarray] = None
        self._loaded = False

        if store_path and Path(store_path).exists():
            self.load(store_path)

    def load(self, path: str):
        print(f"Loading embedding store from {path}")
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = data["labels"]
        self.timestamps = data.get("timestamps", np.array([""]*len(self.labels)))
        self.asset_ids  = data.get("asset_ids",  np.array([""]*len(self.labels)))
        self.event_ids  = data.get("event_ids",  np.array([""]*len(self.labels)))
        self._loaded = True
        print(f"  Loaded {len(self.embeddings):,} historical embeddings")

    def add(
        self,
        embedding: np.ndarray,
        label: int,
        asset_id: str = "",
        event_id: str = "",
        timestamp: str = "",
    ):
        """Add a new embedding to the store."""
        emb = embedding.reshape(1, -1)
        if self.embeddings is None:
            self.embeddings = emb
            self.labels = np.array([label])
            self.asset_ids = np.array([asset_id])
            self.event_ids = np.array([event_id])
            self.timestamps = np.array([timestamp])
        else:
            self.embeddings = np.concatenate([self.embeddings, emb], axis=0)
            self.labels = np.append(self.labels, label)
            self.asset_ids = np.append(self.asset_ids, asset_id)
            self.event_ids = np.append(self.event_ids, event_id)
            self.timestamps = np.append(self.timestamps, timestamp)

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find top-k most similar historical events using cosine similarity.

        Args:
            query_embedding: (256,) float32 — L2-normalized CLS embedding
            top_k:           number of neighbors to return

        Returns:
            list of dicts with similarity, label, asset_id, event_id, timestamp
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Normalize stored embeddings (if not already)
        store_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        normalized_store = self.embeddings / store_norms

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = normalized_store @ query_norm  # (N,)

        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        from model import FaultFormerClassifier
        fault_names = FaultFormerClassifier.FAULT_CLASSES

        results = []
        for idx in top_indices:
            label = int(self.labels[idx])
            results.append({
                "rank": len(results) + 1,
                "similarity": float(similarities[idx]),
                "fault_class": fault_names[label] if label < len(fault_names) else f"class_{label}",
                "fault_class_idx": label,
                "asset_id": str(self.asset_ids[idx]),
                "event_id": str(self.event_ids[idx]),
                "timestamp": str(self.timestamps[idx]),
            })

        return results


# Global model registry
registry = ModelRegistry()
embedding_store = EmbeddingStore(os.getenv("EMBEDDING_STORE_PATH"))


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class VibrationWindow(BaseModel):
    """Raw vibration signal window for inference."""
    signal: List[float] = Field(
        ...,
        description=f"Vibration time-series window. Must be exactly {SIGNAL_LENGTH} samples.",
        min_items=SIGNAL_LENGTH,
        max_items=SIGNAL_LENGTH,
    )
    asset_id: str = Field(default="", description="Optional asset/bearing identifier for logging")
    sample_rate_hz: float = Field(default=12000.0, description="Sampling rate in Hz (metadata only)")

    @validator("signal")
    def validate_signal_length(cls, v):
        if len(v) != SIGNAL_LENGTH:
            raise ValueError(
                f"Signal must be exactly {SIGNAL_LENGTH} samples, got {len(v)}"
            )
        return v

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.signal, dtype=torch.float32).unsqueeze(0).to(device)


class EmbedResponse(BaseModel):
    embedding: List[float] = Field(
        description=f"256-dimensional CLS token embedding (L2-normalized). "
                    f"Use for cosine similarity search and visualization."
    )
    asset_id: str
    inference_time_ms: float


class ClassifyResponse(BaseModel):
    predicted_class: str = Field(description="Predicted fault class name")
    predicted_class_idx: int
    confidence: float = Field(description="Probability of predicted class (0-1)")
    class_probabilities: Dict[str, float] = Field(
        description="Softmax probability for each of the 12 fault classes"
    )
    asset_id: str
    inference_time_ms: float


class RULResponse(BaseModel):
    rul_fraction: float = Field(
        description="Estimated remaining useful life as fraction [0, 1]. "
                    "1.0 = new/healthy, 0.0 = end of life."
    )
    rul_days: float = Field(
        description="Estimated remaining days based on total_life_days parameter."
    )
    total_life_days: float
    asset_id: str
    inference_time_ms: float


class SimilarRequest(BaseModel):
    """Request for nearest-neighbor similarity search."""
    signal: List[float] = Field(..., min_items=SIGNAL_LENGTH, max_items=SIGNAL_LENGTH)
    top_k: int = Field(default=5, ge=1, le=20)
    asset_id: str = Field(default="")

    @validator("signal")
    def validate_length(cls, v):
        if len(v) != SIGNAL_LENGTH:
            raise ValueError(f"Signal must be {SIGNAL_LENGTH} samples, got {len(v)}")
        return v


class SimilarResponse(BaseModel):
    query_asset_id: str
    similar_events: List[Dict]
    embedding: List[float] = Field(description="Query CLS embedding for reference")
    inference_time_ms: float


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(registry.device),
        "classifier_loaded": registry.classifier is not None,
        "rul_loaded": registry.rul_model is not None,
        "embedding_store_size": (
            len(embedding_store.embeddings)
            if embedding_store.embeddings is not None
            else 0
        ),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(window: VibrationWindow):
    """
    Compute 256-dim CLS embedding for a vibration window.

    The CLS embedding captures the global signal character learned during
    transformer pretraining. Use for:
      - Cosine similarity search over historical events
      - t-SNE / UMAP visualization of fleet health
      - Anomaly detection via distance from normal cluster

    Returns L2-normalized embedding.
    """
    t0 = time.perf_counter()

    model = registry.load_classifier()
    x = window.to_tensor(registry.device)

    with torch.no_grad():
        embedding = model.get_cls_embedding(x)  # (1, 256), already L2-normalized

    embedding_np = embedding.squeeze(0).cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return EmbedResponse(
        embedding=embedding_np.tolist(),
        asset_id=window.asset_id,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(window: VibrationWindow):
    """
    Classify a vibration window into one of 12 fault classes.

    Returns:
      - predicted_class: name of the most likely fault class
      - confidence: probability of the predicted class
      - class_probabilities: softmax probability for all 12 classes

    The 12 classes are:
      Normal, Outer-race defect, Inner-race defect, Ball defect, Cage defect,
      Unbalance, Misalignment, Lubrication defect, Looseness, Cavitation,
      Gearbox fault, Electrical fault
    """
    t0 = time.perf_counter()

    model = registry.load_classifier()
    x = window.to_tensor(registry.device)

    with torch.no_grad():
        probs = model.predict_proba(x).squeeze(0).cpu().numpy()  # (12,)

    predicted_idx = int(probs.argmax())
    predicted_class = FaultFormerClassifier.FAULT_CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx])

    class_probs = {
        FaultFormerClassifier.FAULT_CLASSES[i]: float(probs[i])
        for i in range(len(probs))
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ClassifyResponse(
        predicted_class=predicted_class,
        predicted_class_idx=predicted_idx,
        confidence=confidence,
        class_probabilities=class_probs,
        asset_id=window.asset_id,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/rul", response_model=RULResponse)
def estimate_rul(window: VibrationWindow, total_life_days: float = 365.0):
    """
    Estimate Remaining Useful Life (RUL) from a vibration window.

    Returns:
      - rul_fraction: [0, 1] — fraction of remaining life (1.0 = new, 0.0 = EOL)
      - rul_days: estimated days remaining = rul_fraction × total_life_days

    The total_life_days parameter should be set to the expected service life
    of the specific bearing type. Default is 365 days.

    Note: For more accurate RUL estimation, use a time-series of degradation
    trend features (GluonTS DeepAR model) rather than a single window.
    This endpoint provides a single-window estimate for quick assessment.
    """
    t0 = time.perf_counter()

    model = registry.load_rul_model()
    x = window.to_tensor(registry.device)

    with torch.no_grad():
        rul_days, rul_fraction = model.predict_rul(x, total_life_days=total_life_days)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return RULResponse(
        rul_fraction=float(rul_fraction.squeeze()),
        rul_days=float(rul_days.squeeze()),
        total_life_days=total_life_days,
        asset_id=window.asset_id,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/similar", response_model=SimilarResponse)
def find_similar(request: SimilarRequest):
    """
    Find top-k most similar historical fault events using CLS embedding cosine similarity.

    This enables nearest-neighbor explainability:
    "This signal looks like 3 previous confirmed outer-race defect events
    on bearing B-47 from 2024-03-12."

    The embedding store is populated from confirmed historical fault events
    stored in the CMMS / event database.

    Requires EMBEDDING_STORE_PATH to be set and the .npz file to be present.
    Returns empty list if no embedding store is loaded.
    """
    t0 = time.perf_counter()

    model = registry.load_classifier()
    x = torch.tensor(request.signal, dtype=torch.float32).unsqueeze(0).to(registry.device)

    with torch.no_grad():
        embedding = model.get_cls_embedding(x).squeeze(0).cpu().numpy()  # (256,)

    similar_events = embedding_store.find_similar(embedding, top_k=request.top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return SimilarResponse(
        query_asset_id=request.asset_id,
        similar_events=similar_events,
        embedding=embedding.tolist(),
        inference_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Pre-load models on startup to avoid cold-start latency on first request."""
    print("FaultFormer API startup: pre-loading models...")
    registry.load_classifier()
    registry.load_rul_model()
    print(f"Models loaded on device: {registry.device}")

    store_path = os.getenv("EMBEDDING_STORE_PATH")
    if store_path:
        embedding_store.load(store_path)


# ---------------------------------------------------------------------------
# Development Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
