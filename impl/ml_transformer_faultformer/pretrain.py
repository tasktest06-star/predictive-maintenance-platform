"""
FaultFormer Self-Supervised Pretraining Script
==============================================
Pretrain the FaultFormer backbone on unlabeled vibration time-series using
masked patch reconstruction (BERT-style, arXiv:2312.02380).

Data sources (all treated as unlabeled — no class labels used):
  - CWRU (Case Western Reserve University) bearing dataset
  - MFPT (Machinery Failure Prevention Technology) bearing dataset
  - MIMII (Malfunctioning Industrial Machine Investigation and Inspection)
    — acoustic data; same patch tokenization applies

Strategy:
  - 75% masking ratio validated in FaultFormer paper
  - Combine ALL available data (normal + fault) as unlabeled corpus
  - Adam optimizer with cosine LR schedule + linear warmup
  - Log reconstruction loss to MLflow
  - Save pretrained backbone to models/faultformer_pretrained.pt

Usage:
  python pretrain.py \
    --cwru_dir /data/cwru \
    --mfpt_dir /data/mfpt \
    --mimii_dir /data/mimii \
    --output_dir models \
    --epochs 200 \
    --batch_size 128 \
    --lr 1e-3

  # Or with defaults (expects data in standard locations):
  python pretrain.py
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split

from model import (
    FaultFormerPretraining,
    SIGNAL_LENGTH,
    MASK_RATIO,
)


# ---------------------------------------------------------------------------
# Dataset Helpers
# ---------------------------------------------------------------------------

class VibrationWindowDataset(Dataset):
    """
    Loads a numpy array of vibration signals and returns fixed-length windows.

    Signals longer than window_length are split into non-overlapping windows.
    Shorter signals are zero-padded.

    Args:
        signals:       list of 1D numpy arrays (raw vibration recordings)
        window_length: samples per window (default 2048, matches model input)
        normalize:     if True, z-score normalize each window independently
    """

    def __init__(
        self,
        signals: List[np.ndarray],
        window_length: int = SIGNAL_LENGTH,
        normalize: bool = True,
    ):
        self.windows = []
        self.normalize = normalize

        for signal in signals:
            # Split long signal into windows
            n = len(signal)
            for start in range(0, n - window_length + 1, window_length):
                window = signal[start : start + window_length].astype(np.float32)
                self.windows.append(window)

        if len(self.windows) == 0:
            raise ValueError("No windows extracted. Check signal lengths vs window_length.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.windows[idx].copy()
        if self.normalize:
            mean = window.mean()
            std = window.std() + 1e-8
            window = (window - mean) / std
        return torch.from_numpy(window)


def load_cwru_dataset(data_dir: str, window_length: int = SIGNAL_LENGTH) -> Optional[Dataset]:
    """
    Load CWRU bearing dataset from .npy files.

    Expected directory structure:
      data_dir/
        normal_*.npy
        inner_*.npy
        outer_*.npy
        ball_*.npy

    All files loaded as unlabeled — labels ignored for pretraining.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"  CWRU directory not found: {data_dir} — skipping")
        return None

    signals = []
    for npy_file in sorted(data_dir.glob("*.npy")):
        try:
            arr = np.load(npy_file)
            if arr.ndim == 2:
                # Multiple channels — use first (drive-end accelerometer)
                arr = arr[:, 0]
            signals.append(arr)
        except Exception as e:
            print(f"  Warning: could not load {npy_file}: {e}")

    if not signals:
        print(f"  No .npy files found in {data_dir}")
        return None

    dataset = VibrationWindowDataset(signals, window_length=window_length)
    print(f"  CWRU: loaded {len(signals)} files → {len(dataset):,} windows")
    return dataset


def load_mfpt_dataset(data_dir: str, window_length: int = SIGNAL_LENGTH) -> Optional[Dataset]:
    """
    Load MFPT bearing dataset.

    Expected directory structure:
      data_dir/
        *.mat  (MATLAB format with 'gs' field for accelerometer)

    Falls back to .npy if scipy not available for .mat loading.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"  MFPT directory not found: {data_dir} — skipping")
        return None

    signals = []

    # Try .mat files first
    mat_files = sorted(data_dir.glob("*.mat"))
    if mat_files:
        try:
            from scipy.io import loadmat
            for mat_file in mat_files:
                try:
                    mat = loadmat(str(mat_file))
                    # MFPT dataset stores data in 'gs' (accelerometer) field
                    if "gs" in mat:
                        sig = mat["gs"].ravel().astype(np.float32)
                    elif "bearing" in mat:
                        # Some MFPT files use 'bearing' struct
                        bearing = mat["bearing"]
                        sig = bearing["gs"][0, 0].ravel().astype(np.float32)
                    else:
                        # Take first large numeric array
                        candidates = [
                            v.ravel() for v in mat.values()
                            if isinstance(v, np.ndarray) and v.size > window_length
                        ]
                        if not candidates:
                            continue
                        sig = candidates[0].astype(np.float32)
                    signals.append(sig)
                except Exception as e:
                    print(f"  Warning: could not load {mat_file}: {e}")
        except ImportError:
            print("  scipy not available; falling back to .npy for MFPT")

    # Also try .npy files
    for npy_file in sorted(data_dir.glob("*.npy")):
        try:
            arr = np.load(npy_file)
            if arr.ndim == 2:
                arr = arr[:, 0]
            signals.append(arr.astype(np.float32))
        except Exception as e:
            print(f"  Warning: could not load {npy_file}: {e}")

    if not signals:
        print(f"  No usable files found in {data_dir}")
        return None

    dataset = VibrationWindowDataset(signals, window_length=window_length)
    print(f"  MFPT: loaded {len(signals)} files → {len(dataset):,} windows")
    return dataset


def load_mimii_dataset(data_dir: str, window_length: int = SIGNAL_LENGTH) -> Optional[Dataset]:
    """
    Load MIMII acoustic dataset (fan, pump, slider, valve).

    Expected directory structure:
      data_dir/
        fan/normal/*.wav
        fan/abnormal/*.wav
        pump/normal/*.wav
        ...

    Acoustic signals are processed identically to vibration (same tokenization).
    Stereo files are converted to mono by averaging channels.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"  MIMII directory not found: {data_dir} — skipping")
        return None

    signals = []

    try:
        import scipy.io.wavfile as wav
    except ImportError:
        print("  scipy not available for MIMII .wav loading — skipping")
        return None

    for wav_file in sorted(data_dir.rglob("*.wav")):
        try:
            sample_rate, data = wav.read(str(wav_file))
            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            if data.ndim == 2:
                data = data.mean(axis=1)
            signals.append(data)
        except Exception as e:
            print(f"  Warning: could not load {wav_file}: {e}")

    if not signals:
        print(f"  No .wav files found in {data_dir}")
        return None

    dataset = VibrationWindowDataset(signals, window_length=window_length)
    print(f"  MIMII: loaded {len(signals)} files → {len(dataset):,} windows")
    return dataset


def build_combined_dataset(
    cwru_dir: Optional[str],
    mfpt_dir: Optional[str],
    mimii_dir: Optional[str],
    window_length: int = SIGNAL_LENGTH,
) -> Dataset:
    """Combine all available datasets into a single unlabeled pretraining corpus."""
    datasets = []

    if cwru_dir:
        ds = load_cwru_dataset(cwru_dir, window_length)
        if ds:
            datasets.append(ds)

    if mfpt_dir:
        ds = load_mfpt_dataset(mfpt_dir, window_length)
        if ds:
            datasets.append(ds)

    if mimii_dir:
        ds = load_mimii_dataset(mimii_dir, window_length)
        if ds:
            datasets.append(ds)

    if not datasets:
        print("WARNING: No real data found. Using synthetic random signals for demonstration.")
        # Fallback: synthetic data for testing the training loop
        synthetic = torch.randn(10_000, window_length)
        datasets.append(torch.utils.data.TensorDataset(synthetic))

    if len(datasets) == 1:
        return datasets[0]

    combined = ConcatDataset(datasets)
    print(f"  Combined dataset: {len(combined):,} total windows")
    return combined


# ---------------------------------------------------------------------------
# Learning Rate Schedule: Cosine with Linear Warmup
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """
    Cosine LR schedule with linear warmup.
    - Linear ramp from 0 to peak LR over num_warmup_steps
    - Cosine decay from peak to min_lr_ratio * peak over remaining steps
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def pretrain(
    model: FaultFormerPretraining,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    output_dir: str,
    device: torch.device,
    mlflow_experiment: str = "faultformer_pretraining",
) -> None:
    """
    Main pretraining loop with MLflow logging.

    Args:
        model:              FaultFormerPretraining model
        train_loader:       DataLoader for training data
        val_loader:         DataLoader for validation data
        num_epochs:         total training epochs
        learning_rate:      peak learning rate
        output_dir:         directory to save checkpoints
        device:             training device (cuda/cpu)
        mlflow_experiment:  MLflow experiment name
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.10 * num_training_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="pretrain_masked_reconstruction"):
        mlflow.log_params({
            "model": "FaultFormerPretraining",
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "mask_ratio": MASK_RATIO,
            "batch_size": train_loader.batch_size,
            "signal_length": SIGNAL_LENGTH,
            "num_warmup_steps": num_warmup_steps,
            "optimizer": "Adam",
            "lr_schedule": "cosine_with_warmup",
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        })

        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            # ---- Training ----
            model.train()
            train_losses = []

            for batch in train_loader:
                # Handle TensorDataset (tuple) vs plain tensor
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)

                optimizer.zero_grad()
                loss = model.masked_reconstruction_loss(x)
                loss.backward()

                # Gradient clipping for stable training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                global_step += 1

                if global_step % 100 == 0:
                    mlflow.log_metric("train_loss_step", loss.item(), step=global_step)
                    mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=global_step)

            avg_train_loss = np.mean(train_losses)

            # ---- Validation ----
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.to(device)
                    loss = model.masked_reconstruction_loss(x)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }, step=epoch)

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(output_dir, "faultformer_pretrained.pt")
                model.save_backbone(best_path)
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                print(f"  * New best val_loss={best_val_loss:.4f} — checkpoint saved")

            # Also save periodic checkpoint every 50 epochs
            if epoch % 50 == 0:
                ckpt_path = os.path.join(output_dir, f"faultformer_epoch{epoch:03d}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                }, ckpt_path)

        print(f"\nPretraining complete. Best val_loss={best_val_loss:.4f}")
        print(f"Pretrained backbone saved to {os.path.join(output_dir, 'faultformer_pretrained.pt')}")

        # Log the pretrained model as an MLflow artifact
        mlflow.log_artifact(os.path.join(output_dir, "faultformer_pretrained.pt"))
        mlflow.log_metric("final_best_val_loss", best_val_loss)


# ---------------------------------------------------------------------------
# Reconstruction Quality Evaluation
# ---------------------------------------------------------------------------

def evaluate_reconstruction(
    model: FaultFormerPretraining,
    val_loader: DataLoader,
    device: torch.device,
    n_visualize: int = 8,
) -> dict:
    """
    Evaluate reconstruction quality on held-out normal signals.

    Computes:
      - MSE of masked patch reconstruction
      - Visual comparison: original vs reconstructed for n_visualize examples

    Returns:
        metrics dict with 'mse_masked', 'mse_all'
    """
    model.eval()
    model = model.to(device)

    all_mse_masked = []
    all_mse_all = []
    examples_collected = 0
    examples = {"original": [], "reconstructed": [], "mask": []}

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            reconstructed, original, mask = model.forward_pretraining(x)

            # MSE on masked patches
            mse_masked = nn.functional.mse_loss(
                reconstructed[mask], original[mask]
            ).item()
            # MSE on all patches
            mse_all = nn.functional.mse_loss(reconstructed, original).item()

            all_mse_masked.append(mse_masked)
            all_mse_all.append(mse_all)

            # Collect examples for visualization
            if examples_collected < n_visualize:
                b = min(n_visualize - examples_collected, x.shape[0])
                examples["original"].append(original[:b].cpu().numpy())
                examples["reconstructed"].append(reconstructed[:b].cpu().numpy())
                examples["mask"].append(mask[:b].cpu().numpy())
                examples_collected += b

    metrics = {
        "mse_masked": np.mean(all_mse_masked),
        "mse_all": np.mean(all_mse_all),
    }

    print("\nReconstruction Evaluation:")
    print(f"  MSE (masked patches only): {metrics['mse_masked']:.4f}")
    print(f"  MSE (all patches):         {metrics['mse_all']:.4f}")

    # Optionally save visual comparison
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        original_ex = np.concatenate(examples["original"], axis=0)  # (n, num_patches, patch_size)
        recon_ex = np.concatenate(examples["reconstructed"], axis=0)
        mask_ex = np.concatenate(examples["mask"], axis=0)

        fig, axes = plt.subplots(min(4, n_visualize), 1, figsize=(16, 12))
        if min(4, n_visualize) == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            orig_signal = original_ex[i].reshape(-1)
            recon_signal = recon_ex[i].reshape(-1)
            mask_signal = mask_ex[i].repeat(64)  # expand to sample level

            ax.plot(orig_signal, label="Original", alpha=0.8, linewidth=0.8)
            ax.plot(recon_signal, label="Reconstructed", alpha=0.8, linewidth=0.8, linestyle="--")

            # Shade masked regions
            for j, is_masked in enumerate(mask_ex[i]):
                if is_masked:
                    ax.axvspan(j * 64, (j + 1) * 64, alpha=0.15, color="red")

            ax.set_title(f"Sample {i} — masked regions shaded in red")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("Amplitude")

        axes[-1].set_xlabel("Sample index")
        plt.tight_layout()
        fig.savefig("reconstruction_examples.png", dpi=100)
        plt.close(fig)
        print("  Visual comparison saved to reconstruction_examples.png")

    except Exception as e:
        print(f"  Could not generate visualization: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaultFormer self-supervised pretraining"
    )
    parser.add_argument("--cwru_dir",  type=str, default=None, help="CWRU dataset directory")
    parser.add_argument("--mfpt_dir",  type=str, default=None, help="MFPT dataset directory")
    parser.add_argument("--mimii_dir", type=str, default=None, help="MIMII dataset directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--epochs",     type=int, default=200,  help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128,  help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--val_split",  type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mlflow_uri",  type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    # Build dataset
    print("\nLoading datasets (all treated as unlabeled):")
    full_dataset = build_combined_dataset(
        cwru_dir=args.cwru_dir,
        mfpt_dir=args.mfpt_dir,
        mimii_dir=args.mimii_dir,
        window_length=SIGNAL_LENGTH,
    )

    # Train/val split
    n_total = len(full_dataset)
    n_val = max(1, int(args.val_split * n_total))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"\nSplit: train={n_train:,} windows | val={n_val:,} windows")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Build model
    model = FaultFormerPretraining(mask_ratio=MASK_RATIO)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFaultFormerPretraining: {total_params:,} trainable parameters")

    # Pretrain
    print(f"\nStarting pretraining: {args.epochs} epochs, LR={args.lr}, "
          f"mask_ratio={MASK_RATIO}, batch_size={args.batch_size}")
    pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=device,
    )

    # Evaluate reconstruction on validation set
    print("\nEvaluating reconstruction quality on validation set...")
    metrics = evaluate_reconstruction(model, val_loader, device)

    with mlflow.start_run(run_name="eval_reconstruction", nested=True):
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
