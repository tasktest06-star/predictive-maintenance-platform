from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.common.data_generator import SensorDataGenerator
from src.common.models import FaultType
from .feature_extraction import FeatureExtractor
from .model import PredictiveMaintenanceClassifier


def train_and_evaluate(
    n_per_class: int = 300,
    test_size: float = 0.2,
    model_path: str = "model.joblib",
    random_seed: int = 42,
) -> dict:
    print(f"Generating {n_per_class} samples per class...")
    gen = SensorDataGenerator(rng=np.random.default_rng(random_seed))
    readings, labels = gen.generate_dataset(n_per_class=n_per_class)

    train_r, test_r, train_l, test_l = train_test_split(
        readings, labels, test_size=test_size, stratify=labels, random_state=random_seed
    )

    print(f"Training on {len(train_r)} samples, evaluating on {len(test_r)} samples...")
    clf = PredictiveMaintenanceClassifier()
    clf.fit(train_r, train_l)

    diagnoses = clf.predict_batch(test_r)
    y_true = [l.value for l in test_l]
    y_pred = [d.fault_type.value for d in diagnoses]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred))
    print(f"{'='*60}\n")

    clf.save(model_path)
    print(f"Model saved to: {model_path}")

    return {
        "accuracy": acc,
        "per_class_f1": {k: v["f1-score"] for k, v in report.items() if k in [f.value for f in FaultType]},
        "confusion_matrix": cm,
        "n_train": len(train_r),
        "n_test": len(test_r),
        "model_path": model_path,
    }


if __name__ == "__main__":
    train_and_evaluate()
