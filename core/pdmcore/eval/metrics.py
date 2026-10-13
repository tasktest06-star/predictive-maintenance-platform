"""Metrics that mean something for industrial condition monitoring.

Headline accuracy is close to useless here. A plant with 500 monitored assets
and one failure per asset per three years has a prevalence of about 1 in 13 000
hourly acquisitions. A model that always says "healthy" scores 99.99 % accuracy
and earns nothing. Worse, at that prevalence a detector with a 1 % false-positive
rate generates ~5 false alarms per hour across the fleet, the maintenance team
stops reading them within a week, and the deployment is dead -- regardless of
how good the AUC looked.

So the metrics that drive decisions here are:

:func:`partial_auc`        AUC restricted to the usable false-positive region.
                           The full AUC averages over FPRs no plant would ever
                           accept, which is why two models with the same AUC can
                           have 10x different nuisance-alarm rates.
:func:`alarms_per_day`     Nuisance alarm rate in the unit the customer uses.
:func:`detection_delay`    How long after fault onset the alarm fires, measured
                           on a run-to-failure series. This is the number that
                           decides whether a work order can be scheduled or the
                           machine fails first.
:func:`expected_cost`      Explicit cost model. Puts a currency figure on the
                           false-positive / false-negative trade-off so the
                           threshold is a business decision, not a default.
:func:`expected_calibration_error`
                           Whether a reported 80 % confidence means 80 %.
                           Uncalibrated confidence is what makes an operator
                           ignore the next alarm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "roc_curve",
    "auc",
    "partial_auc",
    "precision_recall",
    "average_precision",
    "best_f1",
    "confusion_matrix",
    "classification_report",
    "alarms_per_day",
    "detection_delay",
    "expected_cost",
    "brier_score",
    "expected_calibration_error",
    "DetectionSummary",
    "summarise_detector",
]


def _check_binary(y_true: Sequence) -> np.ndarray:
    y = np.asarray(y_true)
    if y.dtype.kind in "biu":
        y = y.astype(int)
    else:
        y = (np.asarray(y, dtype=float) > 0.5).astype(int)
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, (0, 1))):
        raise ValueError(f"labels must be binary, got {uniq[:5]}")
    return y


def roc_curve(y_true: Sequence, scores: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fpr, tpr, thresholds)`` with ties handled correctly.

    Ties matter: a quantised int8 score has at most 256 levels, so hundreds of
    samples share a value. Treating them one-by-one produces a staircase that
    overstates AUC.
    """
    y = _check_binary(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if y.size != s.size:
        raise ValueError("y_true and scores must have the same length")
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both classes present")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    distinct = np.flatnonzero(np.diff(s_sorted)) + 1
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y_sorted)[idx]
    fp = np.cumsum(1 - y_sorted)[idx]
    return (
        np.r_[0.0, fp / n_neg],
        np.r_[0.0, tp / n_pos],
        np.r_[np.inf, s_sorted[idx]],
    )


def auc(y_true: Sequence, scores: Sequence) -> float:
    """Area under the ROC curve (trapezoidal)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.trapz(tpr, fpr))


def partial_auc(y_true: Sequence, scores: Sequence, max_fpr: float = 0.1, *, standardise: bool = True) -> float:
    """AUC over ``fpr in [0, max_fpr]``.

    ``standardise=True`` applies McClish's correction, rescaling so that 0.5
    still means random and 1.0 still means perfect inside the restricted
    region. Report this at ``max_fpr`` equal to the alarm budget the customer
    has agreed to; the DCASE machine-condition-monitoring task uses 0.1 for the
    same reason.
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    fpr, tpr, _ = roc_curve(y_true, scores)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    f = np.r_[fpr[:stop], max_fpr]
    t = np.r_[tpr[:stop], np.interp(max_fpr, fpr, tpr)]
    area = float(np.trapz(t, f))
    if not standardise:
        return area
    min_area = 0.5 * max_fpr * max_fpr
    max_area = max_fpr
    return float(0.5 * (1.0 + (area - min_area) / (max_area - min_area)))


def precision_recall(y_true: Sequence, scores: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(precision, recall, thresholds)``."""
    y = _check_binary(y_true)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    distinct = np.flatnonzero(np.diff(s_sorted)) + 1
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y_sorted)[idx]
    fp = np.cumsum(1 - y_sorted)[idx]
    n_pos = int(np.count_nonzero(y == 1))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    return precision, recall, s_sorted[idx]


def average_precision(y_true: Sequence, scores: Sequence) -> float:
    """Area under the precision-recall curve.

    Preferred over ROC AUC when positives are rare, because PR-AUC degrades
    visibly with prevalence while ROC AUC barely moves.
    """
    precision, recall, _ = precision_recall(y_true, scores)
    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0] if precision.size else 1.0, precision]
    return float(np.sum(np.diff(recall) * precision[1:]))


def best_f1(y_true: Sequence, scores: Sequence) -> Dict[str, float]:
    """Best achievable F1 and the threshold that achieves it."""
    precision, recall, thresholds = precision_recall(y_true, scores)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return {
        "f1": float(f1[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "threshold": float(thresholds[i]),
    }


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Optional[Sequence] = None) -> Tuple[np.ndarray, list]:
    """Multi-class confusion matrix and the label order used."""
    yt = np.asarray([str(v) for v in y_true])
    yp = np.asarray([str(v) for v in y_pred])
    classes = list(labels) if labels is not None else sorted(set(yt) | set(yp))
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(yt, yp):
        if a in index and b in index:
            cm[index[a], index[b]] += 1
    return cm, classes


def classification_report(y_true: Sequence, y_pred: Sequence, labels: Optional[Sequence] = None) -> Dict[str, Dict[str, float]]:
    """Per-class precision / recall / F1 / support plus macro and weighted means."""
    cm, classes = confusion_matrix(y_true, y_pred, labels)
    out: Dict[str, Dict[str, float]] = {}
    support = cm.sum(axis=1)
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(support[i])}
    total = int(support.sum())
    out["macro"] = {
        k: float(np.mean([out[c][k] for c in classes])) for k in ("precision", "recall", "f1")
    }
    out["macro"]["support"] = total
    out["weighted"] = {
        k: float(np.sum([out[c][k] * out[c]["support"] for c in classes]) / max(total, 1))
        for k in ("precision", "recall", "f1")
    }
    out["weighted"]["support"] = total
    out["accuracy"] = {"value": float(np.trace(cm) / max(total, 1)), "support": total}
    return out


def alarms_per_day(
    healthy_scores: Sequence[float],
    threshold: float,
    *,
    acquisitions_per_day: float = 24.0,
    n_assets: int = 1,
    consecutive_required: int = 1,
) -> float:
    """Expected nuisance alarms per day across a fleet.

    ``consecutive_required`` models the standard de-bouncing rule ("alarm only
    after N consecutive exceedances"), which reduces the nuisance rate by
    roughly ``p**N`` while costing ``N-1`` acquisition intervals of detection
    delay. Quantifying both sides of that trade is the point.
    """
    s = np.asarray(healthy_scores, dtype=np.float64)
    if s.size == 0:
        raise ValueError("need healthy scores")
    p = float(np.mean(s > threshold))
    effective = p**max(1, consecutive_required)
    return float(effective * acquisitions_per_day * max(1, n_assets))


def detection_delay(
    scores: Sequence[float],
    threshold: float,
    *,
    onset_index: int,
    sample_interval_h: float = 1.0,
    consecutive_required: int = 2,
) -> Optional[float]:
    """Hours from fault onset to a confirmed alarm. ``None`` if never detected.

    Scores must be in chronological order. ``onset_index`` is the first sample
    with a real fault present.
    """
    s = np.asarray(scores, dtype=np.float64)
    if not 0 <= onset_index < s.size:
        raise ValueError("onset_index out of range")
    run = 0
    for i in range(onset_index, s.size):
        run = run + 1 if s[i] > threshold else 0
        if run >= consecutive_required:
            return float((i - onset_index) * sample_interval_h)
    return None


def expected_cost(
    y_true: Sequence,
    scores: Sequence,
    *,
    cost_fp: float = 400.0,
    cost_fn: float = 25000.0,
    prevalence: Optional[float] = None,
) -> Dict[str, float]:
    """Minimum-cost threshold under an explicit cost model.

    Defaults are deliberately concrete rather than abstract: ``cost_fp`` is one
    technician inspecting a healthy machine (half a day plus travel);
    ``cost_fn`` is an unplanned stop on a mid-size production line (a few hours
    of lost output plus expedited parts). The 60x asymmetry is why the
    cost-optimal operating point sits at a much higher recall than the
    F1-optimal one -- and why reporting F1 alone quietly recommends the wrong
    threshold.

    ``prevalence`` re-weights the test set to the real base rate, which is
    almost never the base rate of a curated dataset.
    """
    y = _check_binary(y_true)
    s = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(y, s)
    pi = float(np.mean(y == 1)) if prevalence is None else float(prevalence)
    if not 0.0 < pi < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    cost = pi * (1.0 - tpr) * cost_fn + (1.0 - pi) * fpr * cost_fp
    i = int(np.argmin(cost))
    baseline_never = pi * cost_fn
    baseline_always = (1.0 - pi) * cost_fp
    return {
        "threshold": float(thresholds[i]),
        "cost_per_acquisition": float(cost[i]),
        "tpr": float(tpr[i]),
        "fpr": float(fpr[i]),
        "prevalence": pi,
        "cost_never_alarm": float(baseline_never),
        "cost_always_alarm": float(baseline_always),
        "savings_vs_best_baseline": float(min(baseline_never, baseline_always) - cost[i]),
    }


def brier_score(y_true: Sequence, probabilities: Sequence[float]) -> float:
    """Mean squared error of predicted probabilities."""
    y = _check_binary(y_true)
    p = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: Sequence, probabilities: Sequence[float], *, n_bins: int = 10
) -> Dict[str, float]:
    """ECE and the worst single-bin gap.

    Reported alongside every probability the platform shows a user. A model that
    says "85 % confident" and is right 50 % of the time destroys trust faster
    than one that says "60 %" and is right 60 % of the time.
    """
    y = _check_binary(y_true)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    worst = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        gap = abs(conf - acc)
        ece += gap * float(np.mean(m))
        worst = max(worst, gap)
    return {"ece": float(ece), "max_gap": float(worst), "n_bins": float(n_bins)}


@dataclass
class DetectionSummary:
    """Everything needed to decide whether a detector is deployable."""

    auc: float
    pauc_10: float
    average_precision: float
    best_f1: float
    threshold_f1: float
    threshold_fpr1pct: float
    tpr_at_fpr1pct: float
    alarms_per_day_per_100_assets: float
    n_healthy: int
    n_faulty: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "auc": self.auc,
            "pauc_10": self.pauc_10,
            "average_precision": self.average_precision,
            "best_f1": self.best_f1,
            "threshold_f1": self.threshold_f1,
            "threshold_fpr1pct": self.threshold_fpr1pct,
            "tpr_at_fpr1pct": self.tpr_at_fpr1pct,
            "alarms_per_day_per_100_assets": self.alarms_per_day_per_100_assets,
            "n_healthy": float(self.n_healthy),
            "n_faulty": float(self.n_faulty),
        }


def summarise_detector(
    y_true: Sequence,
    scores: Sequence,
    *,
    acquisitions_per_day: float = 24.0,
) -> DetectionSummary:
    """One-call evaluation of a binary detector at operationally meaningful points."""
    y = _check_binary(y_true)
    s = np.asarray(scores, dtype=np.float64)
    healthy = s[y == 0]
    thr_1pct = float(np.quantile(healthy, 0.99))
    tpr_1pct = float(np.mean(s[y == 1] > thr_1pct))
    f1 = best_f1(y, s)
    return DetectionSummary(
        auc=auc(y, s),
        pauc_10=partial_auc(y, s, 0.1),
        average_precision=average_precision(y, s),
        best_f1=f1["f1"],
        threshold_f1=f1["threshold"],
        threshold_fpr1pct=thr_1pct,
        tpr_at_fpr1pct=tpr_1pct,
        alarms_per_day_per_100_assets=alarms_per_day(
            healthy, thr_1pct, acquisitions_per_day=acquisitions_per_day, n_assets=100
        ),
        n_healthy=int(np.count_nonzero(y == 0)),
        n_faulty=int(np.count_nonzero(y == 1)),
    )
