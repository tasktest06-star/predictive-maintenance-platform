from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: torch.device | str = "cpu",
        lr: float = 1e-3,
        weight: torch.Tensor | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def _setup_scheduler(self, total_epochs: int) -> None:
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6
        )

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), torch.tensor(y, dtype=torch.long).to(self.device) if not isinstance(y, torch.Tensor) else y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        return {"loss": total_loss / total, "accuracy": correct / total}

    @torch.inference_mode()
    def val_epoch(self) -> dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), torch.tensor(y, dtype=torch.long).to(self.device) if not isinstance(y, torch.Tensor) else y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        return {"loss": total_loss / total, "accuracy": correct / total}

    def fit(self, epochs: int = 30, early_stopping_patience: int = 5) -> dict[str, list[float]]:
        self._setup_scheduler(epochs)
        history: dict[str, list[float]] = {
            "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []
        }
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict[str, Any] = {}

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()
            if self._scheduler is not None:
                self._scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            logger.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
                epoch, epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return history
