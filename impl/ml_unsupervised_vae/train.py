"""
Training script for VibrationVAE (unsupervised anomaly detection).

Usage
-----
    python train.py \
        --data data/cwru_features.csv \
        --epochs 100 \
        --batch-size 256 \
        --latent-dim 64 \
        --beta 1.0 \
        --output models/vae_checkpoint.pt

Dataset format (CSV)
--------------------
Required columns:
    fault_label  — string: "Normal" for healthy samples (others are ignored)
    rpm          — float: shaft speed in RPM
    BPFO_mag     — float: Ball Pass Frequency Outer-race magnitude
    BPFI_mag     — float: Ball Pass Frequency Inner-race magnitude
    BSF_mag      — float: Ball Spin Frequency magnitude
    FTF_mag      — float: Fundamental Train Frequency magnitude
    RMS          — float: vibration RMS
    kurtosis     — float: signal kurtosis
    crest_factor — float: crest factor
    skewness     — float: signal skewness
    peak_to_peak — float: peak-to-peak amplitude
    ... (additional FFT bins can be appended)

Only rows with fault_label == "Normal" are used for training.

MLflow
------
Set MLFLOW_TRACKING_URI env var or pass --mlflow-uri.  Default: ./mlruns
Logs: train_loss, val_loss, val_recon_loss, val_kl_loss per epoch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.pytorch

from model import VibrationVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns (non-FFT, always present)
# ---------------------------------------------------------------------------

BASE_FEATURE_COLS = [
    "rpm",
    "BPFO_mag",
    "BPFI_mag",
    "BSF_mag",
    "FTF_mag",
    "RMS",
    "kurtosis",
    "crest_factor",
    "skewness",
    "peak_to_peak",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_normal_data(csv_path: str) -> Tuple[np.ndarray, list[str]]:
    """Load CSV, filter to normal-only samples, return feature matrix + col names.

    All numeric columns except 'fault_label' and any string columns are used
    as features.  FFT bin columns (if present) are detected automatically as
    columns matching the pattern 'fft_*' or 'bin_*'.

    Returns
    -------
    X : np.ndarray, shape (N, D)
    feature_cols : list[str]
    """
    log.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("  Total rows loaded: %d", len(df))

    if "fault_label" not in df.columns:
        raise ValueError("CSV must contain a 'fault_label' column.")

    normal_df = df[df["fault_label"].str.strip().str.lower() == "normal"].copy()
    log.info("  Normal-only rows : %d", len(normal_df))
    if len(normal_df) == 0:
        raise ValueError("No normal samples found (fault_label == 'Normal').")

    # Build feature column list: base features + any fft_* / bin_* columns
    present_base = [c for c in BASE_FEATURE_COLS if c in normal_df.columns]
    fft_cols = sorted(
        c
        for c in normal_df.columns
        if (c.startswith("fft_") or c.startswith("bin_"))
    )
    feature_cols = present_base + fft_cols

    if not feature_cols:
        # Fallback: use all numeric columns
        feature_cols = [
            c
            for c in normal_df.select_dtypes(include=[np.number]).columns
            if c != "fault_label"
        ]

    log.info("  Feature columns  : %d  (%s ... %s)", len(feature_cols),
             feature_cols[0], feature_cols[-1])

    X = normal_df[feature_cols].values.astype(np.float32)
    # Replace NaN with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    return X, feature_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # -- reproducibility --
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    log.info("Using device: %s", device)

    # -- data --
    X_raw, feature_cols = load_normal_data(args.data)
    n_features = X_raw.shape[1]

    # Pad or truncate to exactly args.input_dim
    if n_features < args.input_dim:
        pad = np.zeros((X_raw.shape[0], args.input_dim - n_features), dtype=np.float32)
        X_raw = np.concatenate([X_raw, pad], axis=1)
        log.info("  Padded features %d → %d", n_features, args.input_dim)
    elif n_features > args.input_dim:
        X_raw = X_raw[:, : args.input_dim]
        log.info("  Truncated features %d → %d", n_features, args.input_dim)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    tensor_X = torch.from_numpy(X_scaled)
    dataset = TensorDataset(tensor_X)

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info("  Train samples: %d  |  Val samples: %d", train_size, val_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # -- model --
    model = VibrationVAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        dropout=args.dropout,
    ).to(device)
    log.info("Model:\n%s", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # -- MLflow --
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    run = mlflow.start_run(run_name=args.run_name)
    mlflow.log_params(vars(args))

    # -- training loop --
    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_losses = []
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            x_hat, mu, log_var = model(batch_x)
            total, recon, kl = model.loss(batch_x, x_hat, mu, log_var)
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(total.item())

        # --- validate ---
        model.eval()
        val_total_list, val_recon_list, val_kl_list = [], [], []
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                x_hat, mu, log_var = model(batch_x)
                total, recon, kl = model.loss(batch_x, x_hat, mu, log_var)
                val_total_list.append(total.item())
                val_recon_list.append(recon.item())
                val_kl_list.append(kl.item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_total_list))
        mean_val_recon = float(np.mean(val_recon_list))
        mean_val_kl = float(np.mean(val_kl_list))

        scheduler.step(mean_val)

        mlflow.log_metrics(
            {
                "train_loss": mean_train,
                "val_loss": mean_val,
                "val_recon_loss": mean_val_recon,
                "val_kl_loss": mean_val_kl,
            },
            step=epoch,
        )

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  train=%.6f  val=%.6f  (recon=%.6f  kl=%.6f)",
                epoch, args.epochs, mean_train, mean_val, mean_val_recon, mean_val_kl,
            )

        # --- early stopping ---
        if mean_val < best_val_loss - 1e-6:
            best_val_loss = mean_val
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(
                    "Early stopping at epoch %d (best val_loss=%.6f)",
                    epoch, best_val_loss,
                )
                break

    # -- restore best weights --
    if best_state is not None:
        model.load_state_dict(best_state)

    # -- compute anomaly threshold from 99th percentile of val set --
    model.eval()
    all_val_scores = []
    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            scores = model.anomaly_score(batch_x)
            all_val_scores.append(scores.cpu().numpy())
    val_scores_np = np.concatenate(all_val_scores)
    threshold_99 = float(np.percentile(val_scores_np, 99))
    log.info("Anomaly score threshold (99th pct of val set): %.6f", threshold_99)
    mlflow.log_metric("anomaly_threshold_99pct", threshold_99)

    # -- save checkpoint --
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_scale": scaler.data_range_.tolist(),
        "feature_cols": feature_cols,
        "input_dim": args.input_dim,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "threshold_99pct": threshold_99,
        "val_scores_mean": float(val_scores_np.mean()),
        "val_scores_std": float(val_scores_np.std()),
        "val_scores_percentiles": {
            str(p): float(np.percentile(val_scores_np, p))
            for p in [50, 75, 90, 95, 99, 99.9]
        },
    }
    torch.save(checkpoint, str(output_path))
    log.info("Checkpoint saved to %s", output_path)

    # also log as MLflow artefact
    mlflow.log_artifact(str(output_path), artifact_path="checkpoints")
    mlflow.end_run()
    log.info("Training complete.  Best val_loss=%.6f", best_val_loss)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VibrationVAE on normal-only data")
    p.add_argument("--data", required=True, help="Path to features CSV")
    p.add_argument("--output", default="models/vae_checkpoint.pt",
                   help="Where to save model checkpoint")
    p.add_argument("--input-dim", type=int, default=1024,
                   help="Input feature dimension (pad/truncate to match)")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=1.0, help="KL weight")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if GPU available")
    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    p.add_argument("--experiment", default="vae-anomaly-detection")
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


# Needed for type hint in load_normal_data
from typing import Tuple  # noqa: E402 (placed after imports for clarity)

if __name__ == "__main__":
    args = parse_args()
    train(args)
