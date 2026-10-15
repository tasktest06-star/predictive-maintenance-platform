"""The comparison: engineered features vs foundation-model representations.

Everything about *how* the comparison is run is here, and it was written before
any number came out of it (``research/02_protocol.md`` is the same protocol in
prose, committed in the same change). The reason for that order is in
``docs/08_branch_catalogue.md``: the defects in this repository that survived
longest were checks that could not fail, and the easiest way to build one is to
look at the scores first and then decide what counts as winning.

Structure of a fold
-------------------
Three disjoint pools, separated two different ways, because there are two
different leaks to close:

``train``   healthy windows. Fits the scaler and the one-class model.
``calib``   healthy windows from the **same recordings** as ``train`` but
            strictly **later in the record**, with a gap
            (``pdmcore.eval.splits.time_split``). This is what
            ``AnomalyDetector.calibrate`` consumes. It must not be the training
            data -- that is the 15.4x error already recorded in
            ``pdmcore/models/anomaly.py`` -- and it must not be the test data,
            or the false-alarm rate is equal to its target by construction,
            which is the Jetson branch's ``train_onsite.py`` bug.
``test``    healthy windows from a recording that appears in **neither** pool,
            plus faulty windows from physically different bearings. Audited by
            ``pdmcore.eval.splits.leakage_report`` with ``groups=recording``.

For CWRU the fold axis is **motor load**: train and calibrate on the normal
recordings at three loads, test on the fourth load. That is harder than the
usual CWRU protocol and it is the realistic one -- a commissioning window never
covers every duty point, and the first time the plant runs the machine at a
load the detector has not seen is the first time it produces nuisance alarms.
The cost of that choice is visible in the results as a realised false-positive
rate well above the 1 % the threshold was calibrated for, for *both* arms.

For MFPT the fold axis is the baseline recording (3 folds), with the faults
held at the same 270 lb radial load and the same 97 656 Hz rate, so fault is
not confounded with operating point.

A deliberately leaky control is also computed: a random window-level split that
ignores recordings entirely. It exists to put a number on the inflation, the
way ``slides/figures/make_core_figures.py`` does for the simulated fleet.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if os.path.join(_REPO, "core") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "core"))

from pdmcore.eval.metrics import (alarms_per_day, auc, average_precision,  # noqa: E402
                                  detection_delay, partial_auc)
from pdmcore.eval.splits import Split, leakage_report, time_split  # noqa: E402
from pdmcore.models.anomaly import (MahalanobisDetector,  # noqa: E402
                                    PCAResidualDetector, RobustThresholdDetector)

__all__ = [
    "Fold", "ArmResult", "HEADS", "MIN_TRAIN_WINDOWS", "TARGET_FPR",
    "build_folds", "leaky_fold", "score_arm", "aggregate", "decide",
    "paired_bootstrap_ci", "GATE_MARGIN_PAUC", "GATE_FPR_TOLERANCE",
    "GATE_LATENCY_MULTIPLE",
]

#: The alarm budget every threshold in this branch is calibrated to. 1 % of
#: hourly acquisitions is 0.24 nuisance alarms per asset per day, i.e. about
#: one per asset per four days -- already at the top of what a maintenance team
#: tolerates. ``pdmcore.eval.metrics`` reports alarms/day per 100 assets so the
#: fleet-level consequence is visible.
TARGET_FPR = 0.01

#: Fewer healthy training windows than this and no detector is fitted at all.
#: 3 samples per feature is ``MahalanobisDetector.MIN_SAMPLES_PER_FEATURE``;
#: below it the class forces shrinkage upward, which is the right behaviour but
#: makes a *comparison* between a 63-dimensional and a 512-dimensional
#: representation a comparison of two different amounts of shrinkage.
MIN_TRAIN_WINDOWS = 60

# -- the decision rule, fixed in advance ------------------------------------
#: A foundation-model arm must beat the baseline's partial AUC by at least this
#: much to be worth the dependency. 0.02 is roughly 1 standard error of a pAUC
#: estimated from ~200 healthy and ~500 faulty windows, so a smaller margin
#: could not be distinguished from fold noise.
GATE_MARGIN_PAUC = 0.02
#: ...and must not multiply the realised false-alarm rate by more than this.
GATE_FPR_TOLERANCE = 2.0
#: ...and must not cost more than this multiple of the 63-feature extractor's
#: measured per-window time on the host that ran the benchmark.
GATE_LATENCY_MULTIPLE = 10.0


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    """One train / calibrate / test partition, with its audit attached."""

    name: str
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    group_audit: Dict[str, object]
    time_audit: Dict[str, object]
    oneclass_audit: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a, b, what in ((self.train, self.calib, "train/calib"),
                           (self.train, self.test, "train/test"),
                           (self.calib, self.test, "calib/test")):
            shared = np.intersect1d(a, b)
            if shared.size:
                raise AssertionError(f"fold {self.name}: {what} share "
                                     f"{shared.size} window indices")

    @property
    def ok(self) -> bool:
        return bool(self.group_audit.get("ok", False)
                    and self.oneclass_audit.get("ok", False))

    def sizes(self) -> Dict[str, int]:
        return {"train": int(self.train.size), "calib": int(self.calib.size),
                "test": int(self.test.size)}


def _healthy_pool_split(
    idx_healthy: np.ndarray, t_index: np.ndarray, group: np.ndarray,
    *, calib_fraction: float = 0.3, gap_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split healthy windows into train / calibrate *within each recording*.

    Per recording rather than pooled, so every training recording contributes
    to both pools. Pooling first and then time-splitting would put whole
    recordings into the calibration set whenever their window indices happened
    to sort late, and the calibration set would then be measuring a recording
    the model never trained on -- which is a different (and much harder)
    quantity than the one ``calibrate()`` is documented to estimate.
    """
    train_parts, calib_parts = [], []
    for g in np.unique(group[idx_healthy]):
        sel = idx_healthy[group[idx_healthy] == g]
        order = sel[np.argsort(t_index[sel])]
        if order.size < 8:
            train_parts.append(order)
            continue
        sp = time_split(np.arange(order.size, dtype=float),
                        test_fraction=calib_fraction, gap_fraction=gap_fraction)
        train_parts.append(order[sp.train])
        calib_parts.append(order[sp.test])
    train = np.sort(np.concatenate(train_parts)) if train_parts else np.empty(0, int)
    calib = np.sort(np.concatenate(calib_parts)) if calib_parts else np.empty(0, int)
    return train, calib


def build_folds(meta: Dict[str, np.ndarray], scheme: str) -> List[Fold]:
    """Folds for one dataset.

    ``scheme``:

    ``"cwru_load"``    leave-one-motor-load-out (4 folds).
    ``"mfpt_record"``  leave-one-baseline-recording-out (3 folds).
    ``"sim_unit"``     leave-one-group-of-units-out (5 folds).
    """
    y, group, cond = meta["y"], meta["group"], meta["cond"]
    t_index = meta["t_index"]
    n = y.size
    all_idx = np.arange(n)
    folds: List[Fold] = []

    if scheme in ("cwru_load", "mfpt_record"):
        if scheme == "cwru_load":
            axis = cond
            keys = sorted(set(axis.tolist()))
        else:
            axis = group
            keys = sorted(g for g in set(group.tolist()) if np.all(y[group == g] == 0))
        for key in keys:
            held = axis == key
            healthy_train_pool = all_idx[(y == 0) & ~held]
            if healthy_train_pool.size < MIN_TRAIN_WINDOWS:
                continue
            train, calib = _healthy_pool_split(healthy_train_pool, t_index, group)
            if scheme == "cwru_load":
                test = all_idx[held]
            else:
                # Held-out baseline recording as the healthy test set, plus
                # every faulty recording (all at the same load and rate).
                test = all_idx[held | (y == 1)]
            folds.append(_make_fold(f"{scheme}={key}", train, calib, test, meta))
        return folds

    if scheme == "sim_unit":
        from pdmcore.eval.splits import group_kfold

        for sp in group_kfold(group, n_splits=5, seed=0):
            healthy_train_pool = sp.train[y[sp.train] == 0]
            if healthy_train_pool.size < MIN_TRAIN_WINDOWS:
                continue
            train, calib = _healthy_pool_split(healthy_train_pool, t_index, group)
            folds.append(_make_fold(sp.name, train, calib, sp.test, meta))
        return folds

    raise ValueError(f"unknown fold scheme {scheme!r}")


def _make_fold(name: str, train: np.ndarray, calib: np.ndarray,
               test: np.ndarray, meta: Dict[str, np.ndarray]) -> Fold:
    fitted = np.sort(np.concatenate([train, calib]))
    # `labels=` is deliberately NOT passed. leakage_report's label rule is
    # "every class must appear in the train fold", which is right for a
    # classifier and is violated *by construction* by any one-class protocol:
    # the training fold is healthy on purpose. Passing labels here makes the
    # audit fail on all four CWRU folds for the one reason that is not a leak,
    # which would push someone into disabling the audit. The three replacement
    # conditions in `oneclass_audit` below are the ones that can actually catch
    # a one-class leak, and they are hard failures too.
    group_audit = leakage_report(Split(fitted, np.sort(test), name=name),
                                 groups=meta["group"])
    y = meta["y"].astype(int)
    lab = meta["label"]
    train_clean = bool(np.all(y[train] == 0))
    calib_clean = bool(calib.size == 0 or np.all(y[calib] == 0))
    test_both = bool(np.any(y[test] == 0) and np.any(y[test] == 1))
    oc_violations: List[str] = []
    if not train_clean:
        oc_violations.append(
            f"{int(np.count_nonzero(y[train] == 1))} faulty window(s) in the "
            f"training fold: a one-class model fitted on faults learns that "
            f"the fault is normal")
    if not calib_clean:
        oc_violations.append(
            f"{int(np.count_nonzero(y[calib] == 1))} faulty window(s) in the "
            f"calibration fold: the threshold would be set above the faults")
    if not test_both:
        oc_violations.append("test fold does not contain both classes")
    oneclass_audit: Dict[str, object] = {
        "ok": not oc_violations,
        "violations": oc_violations,
        "train_all_healthy": train_clean,
        "calib_all_healthy": calib_clean,
        "test_has_both_classes": test_both,
        "test_fault_labels": sorted(set(lab[test][y[test] == 1].tolist())),
        "test_healthy_recordings": sorted(set(meta["group"][test][y[test] == 0].tolist())),
        "note": "leakage_report was called without labels= on purpose; see the "
                "comment in _make_fold. These three conditions replace its "
                "label rule for a one-class protocol.",
    }
    # The train/calib separation is temporal, not grouped -- by design, see
    # _healthy_pool_split. Audit it as such: every calibration window must come
    # strictly after the last training window of its own recording.
    violations: List[str] = []
    for g in np.unique(meta["group"][calib]):
        tr = train[meta["group"][train] == g]
        ca = calib[meta["group"][calib] == g]
        if tr.size and ca.size and meta["t_index"][ca].min() <= meta["t_index"][tr].max():
            violations.append(f"recording {g}: calibration window at "
                              f"t={int(meta['t_index'][ca].min())} is not after "
                              f"training end t={int(meta['t_index'][tr].max())}")
    time_audit = {"ok": not violations, "violations": violations,
                  "n_calib_recordings": int(np.unique(meta["group"][calib]).size)}
    return Fold(name, np.sort(train), np.sort(calib), np.sort(test),
                group_audit, time_audit, oneclass_audit)


def leaky_fold(meta: Dict[str, np.ndarray], *, seed: int = 0) -> Fold:
    """The control: a random window-level split that ignores recordings.

    Not a baseline anyone should use -- it is here to quantify how much the
    published CWRU numbers that use it are inflated by. Its group audit
    deliberately reports ``ok=False``; a run that printed ``ok=True`` here
    would mean the audit had stopped working.
    """
    y = meta["y"]
    rng = np.random.default_rng(seed)
    healthy = np.flatnonzero(y == 0)
    faulty = np.flatnonzero(y == 1)
    rng.shuffle(healthy)
    n_tr = int(0.5 * healthy.size)
    n_ca = int(0.2 * healthy.size)
    train = np.sort(healthy[:n_tr])
    calib = np.sort(healthy[n_tr:n_tr + n_ca])
    test = np.sort(np.concatenate([healthy[n_tr + n_ca:], faulty]))
    group_audit = leakage_report(
        Split(np.sort(np.concatenate([train, calib])), test, name="leaky"),
        groups=meta["group"])
    return Fold("leaky_random_windows", train, calib, test, group_audit,
                {"ok": False, "violations": ["random split: no time ordering"],
                 "n_calib_recordings": -1},
                {"ok": True, "violations": [],
                 "note": "the one-class conditions hold (training windows are "
                         "still healthy); it is the GROUP audit that fails, "
                         "which is the whole point of this control."})


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------
def _pca_project(train: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Project onto the top-k principal directions of the *training* matrix."""
    mu = train.mean(axis=0)
    _u, _s, vt = np.linalg.svd(train - mu, full_matrices=False)
    comps = vt[:k]
    return (X - mu) @ comps.T


def _head_mahalanobis(Ztr, Zca, Zte, names):
    det = MahalanobisDetector(feature_names=names)
    det.fit(Ztr)
    thr = det.calibrate(Zca, fpr=TARGET_FPR)
    return det.score(Zte), thr, {"shrinkage": det.shrinkage_used_,
                                 "samples_per_feature": det.samples_per_feature_}


def _head_pca_residual(Ztr, Zca, Zte, names):
    det = PCAResidualDetector(feature_names=names)
    det.fit(Ztr)
    thr = det.calibrate(Zca, fpr=TARGET_FPR)
    return det.score(Zte), thr, {"n_components": det.n_components_}


def _head_robust(Ztr, Zca, Zte, names):
    det = RobustThresholdDetector(feature_names=names)
    det.fit(Ztr)
    thr = det.calibrate(Zca, fpr=TARGET_FPR)
    return det.score(Zte), thr, {}


def _head_pca_then_mahalanobis(Ztr, Zca, Zte, names):
    """For 512-dimensional embeddings: reduce, then the same Gaussian head.

    A commissioning window has a few hundred frames. Mahalanobis on 512
    dimensions with 300 samples is 0.6 samples per feature, which
    ``MahalanobisDetector`` correctly answers by shrinking almost to a diagonal
    -- at which point the comparison is no longer between representations but
    between regularisation strengths. Reducing the embedding to the baseline's
    63 dimensions first makes the two arms dimensionally comparable, and the
    projection is fitted on healthy training windows only.
    """
    k = int(min(63, Ztr.shape[0] - 1, Ztr.shape[1]))
    if k < 2:
        raise ValueError("not enough training windows to project")
    Ptr = _pca_project(Ztr, Ztr, k)
    return _head_mahalanobis(Ptr, _pca_project(Ztr, Zca, k),
                             _pca_project(Ztr, Zte, k),
                             tuple(f"pc{i:02d}" for i in range(k)))


HEADS: Dict[str, Callable] = {
    "mahalanobis": _head_mahalanobis,
    "pca_residual": _head_pca_residual,
    "robust_threshold": _head_robust,
    "pca63_mahalanobis": _head_pca_then_mahalanobis,
}


# ---------------------------------------------------------------------------
# scoring one arm on one fold
# ---------------------------------------------------------------------------
@dataclass
class ArmResult:
    """Metrics for one (representation, head) on one fold."""

    arm: str
    head: str
    fold: str
    auc: float
    pauc_10: float
    pauc_01: float
    average_precision: float
    threshold: float
    tpr_at_threshold: float
    fpr_at_threshold: float
    alarms_per_day_per_100: float
    n_test_healthy: int
    n_test_faulty: int
    n_train: int
    n_calib: int
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["extra"] = {k: float(v) for k, v in self.extra.items()}
        return d


def _metrics(arm: str, head: str, fold: Fold, y_test: np.ndarray,
             scores: np.ndarray, threshold: float, extra: Dict[str, float]) -> ArmResult:
    healthy = scores[y_test == 0]
    faulty = scores[y_test == 1]
    return ArmResult(
        arm=arm, head=head, fold=fold.name,
        auc=auc(y_test, scores),
        pauc_10=partial_auc(y_test, scores, 0.1),
        pauc_01=partial_auc(y_test, scores, 0.01),
        average_precision=average_precision(y_test, scores),
        threshold=float(threshold),
        tpr_at_threshold=float(np.mean(faulty > threshold)),
        fpr_at_threshold=float(np.mean(healthy > threshold)),
        alarms_per_day_per_100=alarms_per_day(healthy, threshold,
                                              acquisitions_per_day=24.0, n_assets=100),
        n_test_healthy=int(healthy.size), n_test_faulty=int(faulty.size),
        n_train=int(fold.train.size), n_calib=int(fold.calib.size),
        extra=extra,
    )


def score_arm(
    arm: str, Z: np.ndarray, y: np.ndarray, folds: Sequence[Fold],
    *, heads: Sequence[str] = ("mahalanobis",),
    feature_names: Optional[Sequence[str]] = None,
) -> List[ArmResult]:
    """Fit and score one representation across every fold and head.

    ``Z`` is ``(n_windows, d)`` for the whole dataset; the folds index into it.
    Nothing is fitted on anything outside ``fold.train``.
    """
    out: List[ArmResult] = []
    names = tuple(feature_names or [f"f{i:03d}" for i in range(Z.shape[1])])
    for fold in folds:
        if fold.train.size < MIN_TRAIN_WINDOWS:
            continue
        Ztr, Zca, Zte = Z[fold.train], Z[fold.calib], Z[fold.test]
        y_te = y[fold.test]
        if np.unique(y_te).size < 2:
            continue
        for head in heads:
            try:
                scores, thr, extra = HEADS[head](Ztr, Zca, Zte, names)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                out.append(ArmResult(arm=arm, head=head, fold=fold.name,
                                     auc=float("nan"), pauc_10=float("nan"),
                                     pauc_01=float("nan"), average_precision=float("nan"),
                                     threshold=float("nan"), tpr_at_threshold=float("nan"),
                                     fpr_at_threshold=float("nan"),
                                     alarms_per_day_per_100=float("nan"),
                                     n_test_healthy=int(np.count_nonzero(y_te == 0)),
                                     n_test_faulty=int(np.count_nonzero(y_te == 1)),
                                     n_train=int(fold.train.size),
                                     n_calib=int(fold.calib.size),
                                     extra={"error": 1.0}))
                print(f"    {arm}/{head}/{fold.name}: {type(exc).__name__}: {exc}")
                continue
            out.append(_metrics(arm, head, fold, y_te, scores, thr, extra))
    return out


def score_native(arm: str, raw_scores: np.ndarray, y: np.ndarray,
                 folds: Sequence[Fold]) -> List[ArmResult]:
    """A zero-shot score that needs no fitting (MOMENT reconstruction, Chronos
    forecast residual).

    The threshold still has to come from somewhere, and it comes from the same
    held-out healthy calibration pool at the same 1 % target, so the operating
    point is comparable. "Zero-shot" in the literature almost always means
    "zero-shot representation, thresholded on healthy data"; a score with no
    threshold is not a detector.
    """
    out: List[ArmResult] = []
    for fold in folds:
        s_ca = raw_scores[fold.calib]
        s_te = raw_scores[fold.test]
        y_te = y[fold.test]
        if s_ca.size < 10 or np.unique(y_te).size < 2:
            continue
        thr = float(np.quantile(s_ca, 1.0 - TARGET_FPR))
        out.append(_metrics(arm, "native_threshold", fold, y_te, s_te, thr, {}))
    return out


# ---------------------------------------------------------------------------
# aggregation and the decision
# ---------------------------------------------------------------------------
def aggregate(results: Sequence[ArmResult]) -> Dict[str, Dict[str, object]]:
    """Mean/std per (arm, head) over folds, keeping the per-fold values."""
    out: Dict[str, Dict[str, object]] = {}
    keys = sorted({(r.arm, r.head) for r in results})
    for arm, head in keys:
        rows = [r for r in results if r.arm == arm and r.head == head]
        d: Dict[str, object] = {"arm": arm, "head": head, "n_folds": len(rows),
                                "folds": [r.fold for r in rows]}
        for metric in ("auc", "pauc_10", "pauc_01", "average_precision",
                       "tpr_at_threshold", "fpr_at_threshold",
                       "alarms_per_day_per_100"):
            vals = np.array([getattr(r, metric) for r in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            d[metric] = {
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite)) if finite.size else float("nan"),
                "per_fold": [None if not np.isfinite(v) else float(v) for v in vals],
            }
        out[f"{arm}|{head}"] = d
    return out


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], *,
                        n_boot: int = 10000, seed: int = 0,
                        alpha: float = 0.05) -> Dict[str, float]:
    """Bootstrap CI for ``mean(a) - mean(b)``, resampling folds in pairs.

    Paired because the folds are the same for both arms, and ``n_folds`` is 3-5
    -- small enough that an unpaired comparison is dominated by between-fold
    variance rather than between-arm difference. With 4 folds a bootstrap CI is
    wide, and saying so is the point: it is what stops a 0.01 pAUC difference
    from being reported as a result.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_folds": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return {"diff": float(np.mean(a - b)),
            "lo": float(np.quantile(diffs, alpha / 2)),
            "hi": float(np.quantile(diffs, 1.0 - alpha / 2)),
            "n_folds": int(a.size)}


def decide(
    challenger: Dict[str, object],
    baseline: Dict[str, object],
    *,
    challenger_ms_per_window: float,
    baseline_ms_per_window: float,
    challenger_delay_h: Optional[float] = None,
    baseline_delay_h: Optional[float] = None,
) -> Dict[str, object]:
    """Apply the four gates and return the verdict.

    All four must pass for ``adopt`` to be true. They are stated in
    ``research/02_protocol.md`` and in ``GATE_*`` above, and the unit tests
    include a case in which a synthetic challenger *does* pass all four --
    without that case this function would be a check that cannot fire.
    """
    ci = paired_bootstrap_ci(challenger["pauc_10"]["per_fold"],
                             baseline["pauc_10"]["per_fold"])
    g1 = bool(ci["diff"] >= GATE_MARGIN_PAUC and ci["lo"] > 0.0)

    fpr_c = float(challenger["fpr_at_threshold"]["mean"])
    fpr_b = float(baseline["fpr_at_threshold"]["mean"])
    # A challenger with a *lower* realised FPR always passes; the tolerance
    # only bounds how much worse it may be. The max() floor stops a baseline
    # that happened to realise 0 % from making the gate unpassable.
    g2 = bool(fpr_c <= GATE_FPR_TOLERANCE * max(fpr_b, TARGET_FPR))

    if challenger_delay_h is None or baseline_delay_h is None:
        g3, g3_note = False, ("no detection-delay measurement for one of the "
                              "arms; the gate cannot pass without it")
    else:
        g3 = bool(challenger_delay_h <= baseline_delay_h)
        g3_note = ""

    g4 = bool(challenger_ms_per_window
              <= GATE_LATENCY_MULTIPLE * baseline_ms_per_window)

    return {
        "adopt": bool(g1 and g2 and g3 and g4),
        "gates": {
            "pauc_margin": {"pass": g1, "diff": ci["diff"], "ci_lo": ci["lo"],
                            "ci_hi": ci["hi"], "required_diff": GATE_MARGIN_PAUC,
                            "n_folds": ci["n_folds"]},
            "false_alarm_rate": {"pass": g2, "challenger_fpr": fpr_c,
                                 "baseline_fpr": fpr_b,
                                 "tolerance_multiple": GATE_FPR_TOLERANCE},
            "detection_delay": {"pass": g3, "challenger_h": challenger_delay_h,
                                "baseline_h": baseline_delay_h, "note": g3_note},
            "latency": {"pass": g4,
                        "challenger_ms_per_window": challenger_ms_per_window,
                        "baseline_ms_per_window": baseline_ms_per_window,
                        "allowed_multiple": GATE_LATENCY_MULTIPLE},
        },
        "challenger": f"{challenger['arm']}|{challenger['head']}",
        "baseline": f"{baseline['arm']}|{baseline['head']}",
    }


def delay_on_series(scores: np.ndarray, calib_scores: np.ndarray,
                    severity: np.ndarray, *, days: float,
                    consecutive_required: int = 2) -> Dict[str, object]:
    """Detection delay in days on a run-to-failure series.

    ``onset_index`` is the first point whose true severity exceeds 0.05 --
    below that the simulator's "fault" is smaller than its own nuisance
    variation and an alarm there is not a detection, it is a false positive
    that happens to be on the right side of the label.
    """
    onset = int(np.argmax(severity > 0.05))
    thr = float(np.quantile(calib_scores, 1.0 - TARGET_FPR))
    interval_h = 24.0 * days / max(1, scores.size)
    d = detection_delay(scores, thr, onset_index=onset,
                        sample_interval_h=interval_h,
                        consecutive_required=consecutive_required)
    return {"delay_h": d, "delay_days": None if d is None else d / 24.0,
            "onset_index": onset, "threshold": thr,
            "sample_interval_h": interval_h,
            "pre_onset_alarm_fraction": float(np.mean(scores[:onset] > thr))
            if onset > 0 else 0.0}
