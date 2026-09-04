"""
Inference module for VibrationVAE anomaly detection.

Provides:
  - AnomalyDetector: loads a saved checkpoint, scores feature vectors
  - FastAPI app (when run directly or imported by ml-server container)

Designed to be consumed either:
  1. Via direct import from a Flink stream processor (Python operator)
  2. Via REST API (POST /score) when running in FastAPI mode

Usage (direct import)
---------------------
    from infer import AnomalyDetector

    detector = AnomalyDetector("models/vae_checkpoint.pt")
    result = detector.score(feature_vector_np_array)
    # result = {"anomaly_score": 0.032, "is_anomalous": True, "percentile": 97.4}

Usage (gRPC integration)
------------------------
The detector.score() method is synchronous and thread-safe after construction.
Wrap it in a gRPC servicer as needed; the feature_vector is a plain np.ndarray
so serialisation is trivial (e.g. NumPy bytes via protobuf).

Usage (REST server)
-------------------
    uvicorn infer:app --host 0.0.0.0 --port 8080

POST /score
    Content-Type: application/json
    Body: {"features": [0.12, 0.44, ...], "machine_id": 7}   # machine_id optional

Response:
    {"anomaly_score": 0.032, "is_anomalous": true, "percentile": 97.4,
     "threshold": 0.028, "model_version": "vae_v1"}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model import VibrationVAE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Loads a trained VibrationVAE checkpoint and scores feature vectors.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the .pt checkpoint produced by train.py.
    device : str | None
        Torch device string ("cpu", "cuda", etc.).  If None, auto-detected.

    Thread safety
    -------------
    After __init__, score() is stateless and thread-safe (model.eval() is
    set permanently; reparameterize falls back to mu in eval mode).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        log.info("Loading VAE checkpoint from %s", self.checkpoint_path)
        ckpt = torch.load(
            str(self.checkpoint_path), map_location=self.device, weights_only=False
        )

        self.input_dim: int = ckpt["input_dim"]
        self.latent_dim: int = ckpt["latent_dim"]
        self.beta: float = ckpt["beta"]
        self.threshold: float = ckpt["threshold_99pct"]
        self.feature_cols: list[str] = ckpt.get("feature_cols", [])
        self.val_scores_mean: float = ckpt.get("val_scores_mean", 0.0)
        self.val_scores_std: float = ckpt.get("val_scores_std", 1.0)
        self.val_scores_percentiles: dict[str, float] = ckpt.get(
            "val_scores_percentiles", {}
        )

        # Reconstruct scaler arrays for normalisation
        self._scaler_min = np.asarray(ckpt["scaler_min"], dtype=np.float32)
        self._scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        # Avoid division by zero for constant features
        self._scaler_scale = np.where(
            self._scaler_scale == 0, 1.0, self._scaler_scale
        )

        self.model = VibrationVAE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            beta=self.beta,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        log.info(
            "Checkpoint loaded.  input_dim=%d  threshold(99pct)=%.6f",
            self.input_dim,
            self.threshold,
        )

    def _preprocess(self, x: np.ndarray) -> torch.Tensor:
        """MinMax-scale x using training scaler, pad/truncate to input_dim."""
        x = x.astype(np.float32).copy()
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = x.shape[1]
        if n < self.input_dim:
            pad = np.zeros((x.shape[0], self.input_dim - n), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif n > self.input_dim:
            x = x[:, : self.input_dim]

        # Apply MinMax scaling: X_scaled = (X - min) / scale
        scaler_min = self._scaler_min
        scaler_scale = self._scaler_scale
        # Extend if needed (e.g. padding added)
        if scaler_min.shape[0] < self.input_dim:
            pad_len = self.input_dim - scaler_min.shape[0]
            scaler_min = np.concatenate([scaler_min, np.zeros(pad_len)])
            scaler_scale = np.concatenate([scaler_scale, np.ones(pad_len)])

        x_scaled = (x - scaler_min[: self.input_dim]) / scaler_scale[: self.input_dim]
        x_scaled = np.clip(x_scaled, 0.0, 1.0)
        return torch.from_numpy(x_scaled).to(self.device)

    def _compute_percentile(self, score: float) -> float:
        """Estimate what percentile `score` falls at vs. the val distribution.

        Uses the stored percentile landmarks with linear interpolation.
        Returns a value in [0, 100].
        """
        if not self.val_scores_percentiles:
            # Fallback: use z-score approximation
            z = (score - self.val_scores_mean) / max(self.val_scores_std, 1e-9)
            from scipy.special import ndtr  # type: ignore
            return float(min(ndtr(z) * 100, 100.0))

        pts = sorted(
            (float(k), v) for k, v in self.val_scores_percentiles.items()
        )
        pcts = [p for p, _ in pts]
        vals = [v for _, v in pts]

        if score <= vals[0]:
            return pcts[0]
        if score >= vals[-1]:
            return 100.0

        for i in range(len(vals) - 1):
            if vals[i] <= score <= vals[i + 1]:
                span = vals[i + 1] - vals[i]
                frac = (score - vals[i]) / span if span > 0 else 0.0
                return pcts[i] + frac * (pcts[i + 1] - pcts[i])
        return 100.0

    def score(self, feature_vector: np.ndarray) -> dict:
        """Compute anomaly score for a single feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray, shape (D,) or (1, D)
            Raw (un-normalised) feature values in the same order as training.

        Returns
        -------
        dict with keys:
            anomaly_score  : float  — MSE reconstruction error
            is_anomalous   : bool   — True if score > threshold (99th pct)
            percentile     : float  — estimated percentile vs. training distribution
        """
        x_tensor = self._preprocess(feature_vector)
        with torch.no_grad():
            raw_score = self.model.anomaly_score(x_tensor)

        score_value = float(raw_score.item() if raw_score.ndim == 0 else raw_score[0])
        pct = self._compute_percentile(score_value)
        return {
            "anomaly_score": score_value,
            "is_anomalous": score_value > self.threshold,
            "percentile": round(pct, 2),
        }

    def score_batch(self, feature_matrix: np.ndarray) -> list[dict]:
        """Score a batch of feature vectors.

        Parameters
        ----------
        feature_matrix : np.ndarray, shape (N, D)

        Returns
        -------
        list of N dicts (same structure as score())
        """
        x_tensor = self._preprocess(feature_matrix)
        with torch.no_grad():
            scores = self.model.anomaly_score(x_tensor)
        scores_np = scores.cpu().numpy()
        return [
            {
                "anomaly_score": float(s),
                "is_anomalous": float(s) > self.threshold,
                "percentile": round(self._compute_percentile(float(s)), 2),
            }
            for s in scores_np
        ]


# ---------------------------------------------------------------------------
# FastAPI app (REST serving)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(
        title="VibrationVAE Anomaly Detector",
        description="Unsupervised anomaly detection for industrial machinery",
        version="1.0.0",
    )

    # Lazily initialised on first request (or at startup event)
    _detector: Optional[AnomalyDetector] = None

    CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT", "models/vae_checkpoint.pt")

    @app.on_event("startup")
    def load_model() -> None:
        global _detector
        _detector = AnomalyDetector(CHECKPOINT_PATH)

    class ScoreRequest(BaseModel):
        features: list[float]
        machine_id: Optional[int] = None  # reserved for MachineIDConditionedVAE

    class ScoreResponse(BaseModel):
        anomaly_score: float
        is_anomalous: bool
        percentile: float
        threshold: float
        model_version: str = "vae_v1"

    @app.post("/score", response_model=ScoreResponse)
    def score_endpoint(request: ScoreRequest) -> ScoreResponse:
        if _detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        x = np.array(request.features, dtype=np.float32)
        result = _detector.score(x)
        return ScoreResponse(
            anomaly_score=result["anomaly_score"],
            is_anomalous=result["is_anomalous"],
            percentile=result["percentile"],
            threshold=_detector.threshold,
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": _detector is not None,
            "threshold": _detector.threshold if _detector else None,
        }

except ImportError:
    # FastAPI not installed — REST serving unavailable, direct import still works
    app = None  # type: ignore


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="models/vae_checkpoint.pt",
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI REST server instead of running example",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.serve:
        if app is None:
            print("FastAPI not installed. Run: pip install fastapi uvicorn")
            raise SystemExit(1)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # --- Direct-import example ---
        print(f"Loading detector from: {args.checkpoint}")
        detector = AnomalyDetector(args.checkpoint)

        # Simulate a normal sample (random near-zero noise)
        rng = np.random.default_rng(0)
        normal_sample = rng.normal(loc=0.3, scale=0.05, size=detector.input_dim).astype(
            np.float32
        )
        normal_sample = np.clip(normal_sample, 0, 1)

        # Simulate an anomalous sample (large deviation)
        anomalous_sample = rng.normal(loc=0.3, scale=0.05, size=detector.input_dim).astype(
            np.float32
        )
        anomalous_sample[:50] += 0.6  # inject spike in first 50 features
        anomalous_sample = np.clip(anomalous_sample, 0, 1)

        normal_result = detector.score(normal_sample)
        anomalous_result = detector.score(anomalous_sample)

        print("\n--- Normal sample ---")
        print(json.dumps(normal_result, indent=2))
        print("\n--- Anomalous sample ---")
        print(json.dumps(anomalous_result, indent=2))
        print(f"\nThreshold (99th pct of training): {detector.threshold:.6f}")
