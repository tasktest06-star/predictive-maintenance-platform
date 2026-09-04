"""
Training script for BearingFaultCNN1D on CWRU + MFPT combined dataset.

Key features:
    - Bearing-wise stratified split (no data leakage, arXiv 2509.22267)
    - Class-weighted cross-entropy loss for imbalanced fault classes
    - Adam optimizer + cosine annealing LR schedule
    - Per-epoch accuracy, per-class F1, confusion matrix
    - Early stopping on validation macro F1
    - MLflow experiment tracking (parameters, metrics, artifacts)
    - Best model checkpoint saved

Usage::

    python train.py --cwru-root /data/cwru --mfpt-root /data/mfpt \\
        --epochs 100 --batch-size 256 --lr 1e-3 --output-dir ./runs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import DatasetFactory
from model import FAULT_CLASSES, NUM_CLASSES, BearingFaultCNN1D

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BearingFaultCNN1D")
    parser.add_argument("--cwru-root", required=True, type=Path, help="CWRU dataset root")
    parser.add_argument("--mfpt-root", required=True, type=Path, help="MFPT dataset root")
    parser.add_argument("--output-dir", default="./runs", type=Path, help="Checkpoint & artifact output dir")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=15, type=int, help="Early stopping patience (val F1 epochs)")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test-fraction", default=0.2, type=float)
    parser.add_argument("--experiment-name", default="bearing_fault_cnn1d", type=str)
    parser.add_argument("--mlflow-tracking-uri", default="mlruns", type=str)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    plt.savefig(str(save_path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# One-epoch train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        probs = model(x)
        loss = criterion(torch.log(probs + 1e-8), y)   # NLLLoss on log-softmax
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = probs.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Returns (avg_loss, accuracy, macro_f1, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        probs = model(x)
        loss = criterion(torch.log(probs + 1e-8), y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    accuracy = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return total_loss / total, accuracy, macro_f1, preds, labels


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, test_loader, class_weights = DatasetFactory.combined_cwru_mfpt(
        cwru_root=args.cwru_root,
        mfpt_root=args.mfpt_root,
        test_fraction=args.test_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser
    # ------------------------------------------------------------------
    model = BearingFaultCNN1D(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    # Use NLLLoss (model outputs softmax; we take log inside the loop)
    criterion = nn.NLLLoss(weight=weights_tensor)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": "BearingFaultCNN1D",
            "num_classes": NUM_CLASSES,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "device": str(device),
            "n_params": n_params,
        })

        # ------------------------------------------------------------------
        # Training loop with early stopping
        # ------------------------------------------------------------------
        best_val_f1 = -1.0
        patience_counter = 0
        best_ckpt_path = args.output_dir / "best_model.pt"
        history: list[dict] = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            # Per-class F1
            per_class_f1 = f1_score(val_labels, val_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

            logger.info(
                "Epoch %03d/%03d | loss %.4f/%.4f | acc %.4f/%.4f | macro-F1 %.4f | %.1fs",
                epoch, args.epochs, train_loss, val_loss, train_acc, val_acc, val_f1, elapsed,
            )

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)
            for i, cls_f1 in enumerate(per_class_f1):
                mlflow.log_metric(f"val_f1_{FAULT_CLASSES[i]}", cls_f1, step=epoch)

            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_acc": val_acc, "val_macro_f1": val_f1,
            })

            # Checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": val_f1,
                        "val_acc": val_acc,
                        "fault_classes": FAULT_CLASSES,
                        "num_classes": NUM_CLASSES,
                    },
                    best_ckpt_path,
                )
                logger.info("  -> Saved best checkpoint (val macro-F1 = %.4f)", val_f1)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                    break

        # ------------------------------------------------------------------
        # Final evaluation
        # ------------------------------------------------------------------
        logger.info("Loading best checkpoint from %s ...", best_ckpt_path)
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        _, final_acc, final_f1, final_preds, final_labels = evaluate(
            model, test_loader, criterion, device
        )

        report = classification_report(
            final_labels, final_preds,
            target_names=FAULT_CLASSES,
            zero_division=0,
        )
        logger.info("Classification report:\n%s", report)

        # Confusion matrix artifact
        cm = confusion_matrix(final_labels, final_preds, labels=list(range(NUM_CLASSES)))
        cm_path = args.output_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, FAULT_CLASSES, cm_path)

        # Save report and history
        report_path = args.output_dir / "classification_report.txt"
        report_path.write_text(report)
        history_path = args.output_dir / "training_history.json"
        history_path.write_text(json.dumps(history, indent=2))

        # Log final metrics and artifacts to MLflow
        mlflow.log_metrics({
            "final_test_acc": final_acc,
            "final_test_macro_f1": final_f1,
            "best_val_macro_f1": best_val_f1,
        })
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(history_path), artifact_path="reports")
        mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
        mlflow.pytorch.log_model(model, artifact_path="model")

        logger.info("Run ID: %s", run.info.run_id)
        logger.info("Final test accuracy: %.4f | macro F1: %.4f", final_acc, final_f1)
        logger.info("Best checkpoint: %s", best_ckpt_path)


if __name__ == "__main__":
    main()
