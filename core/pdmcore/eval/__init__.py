"""Evaluation: leakage-resistant splits and operationally meaningful metrics.

See :mod:`pdmcore.eval.splits` for why a random split is never acceptable here,
and :mod:`pdmcore.eval.metrics` for why accuracy is not the metric.
"""

from __future__ import annotations

from .metrics import (
    DetectionSummary,
    alarms_per_day,
    auc,
    average_precision,
    best_f1,
    brier_score,
    classification_report,
    confusion_matrix,
    detection_delay,
    expected_calibration_error,
    expected_cost,
    partial_auc,
    precision_recall,
    roc_curve,
    summarise_detector,
)
from .splits import Split, group_kfold, group_split, leakage_report, stratified_group_split, time_split

__all__ = [
    "Split", "group_split", "stratified_group_split", "time_split", "group_kfold", "leakage_report",
    "roc_curve", "auc", "partial_auc", "precision_recall", "average_precision", "best_f1",
    "confusion_matrix", "classification_report", "alarms_per_day", "detection_delay",
    "expected_cost", "brier_score", "expected_calibration_error",
    "DetectionSummary", "summarise_detector",
]
