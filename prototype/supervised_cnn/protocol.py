"""The evaluation protocol: splits, audits and metrics. Written before results.

Everything here delegates to ``pdmcore.eval``. The splits are
``pdmcore.eval.splits``, the audit is ``leakage_report``, the operating-point
metrics are ``pdmcore.eval.metrics``. Nothing is reimplemented, including the
things it would have been faster to reimplement.

The split ladder, in increasing strictness
------------------------------------------
``random_window``
    A random partition of windows. This is the literature's default and it is
    implemented as ``group_split(groups=window_id)`` -- i.e. every window is its
    own group, which is exactly a random split -- so that the *same* function
    produces the leaky and the clean split and no difference can hide in the
    plumbing. ``leakage_report(..., groups=recording)`` is then run on it and
    reports the violation, which is the point.
``segment_grouped``
    Each recording is cut into 6 contiguous segments; whole segments are held
    out. Removes the 50 %-overlap near-duplicate leak while leaving the recording
    on both sides. Isolates flaw 1 from flaw 2.
``recording_grouped``
    Whole recordings held out. For CWRU a recording is one (bearing, load), so
    this holds out operating points but **not bearings**.
``cross_load:<hp>``
    Train on every load except one, test on that one. The strongest
    generalisation test CWRU admits.
``unit_grouped``
    Whole physical bearings held out. For CWRU this is *impossible*: the unit key
    and the 10-class label are the same string, and in the 4- and 2-class label
    spaces there is exactly one healthy bearing. ``stratified_group_split``
    raises, which is the designed behaviour of
    ``pdmcore.eval.splits.stratified_group_split`` and the correct answer.
``cross_dataset``
    Train on all of one dataset, test on all of another. No split function
    involved; the audit still runs, with the dataset name folded into the group
    key so that a mistake in dataset assembly shows up as a group violation.

What "generalises" means numerically, fixed in advance
------------------------------------------------------
A model generalises, for the purposes of this branch, if under the strictest
split the dataset admits it reaches

  * 10-class (or 3-class) **macro F1 >= 0.70**, and
  * binary healthy-vs-faulty **TPR >= 0.90 at 1 % FPR**, and
  * **pAUC(max_fpr=0.1) >= 0.90**,

using ``pdmcore.eval.metrics`` for all three. Macro F1 rather than accuracy
because CWRU's healthy class carries 28 % of the windows (the normal records are
2-4x longer) and accuracy would reward predicting it. TPR at 1 % FPR and pAUC at
10 % FPR because those are the operating points
``docs/03_model_cards.md`` already reports every detector at, and because the
full AUC averages over false-positive rates no plant would accept.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pdmcore.eval.metrics import classification_report, summarise_detector
from pdmcore.eval.splits import Split, group_split, leakage_report, stratified_group_split

from cwru import WindowSet

__all__ = [
    "SPLIT_KINDS", "GENERALISATION_GATE", "build_split", "audit", "evaluate",
    "subset_classes", "merge",
]

SPLIT_KINDS = (
    "random_window", "segment_grouped", "recording_grouped", "unit_grouped",
    "cross_load:0", "cross_load:1", "cross_load:2", "cross_load:3",
)

#: The preregistered pass marks. Read by ``run_matrix.py`` and by
#: ``production/supervised_cnn/gates.py``; written here once so the deck, the
#: docs and the deployment gate cannot quote different numbers.
GENERALISATION_GATE = {"macro_f1": 0.70, "tpr_at_fpr1pct": 0.90, "pauc_10": 0.90}


def purge_overlap(ws: WindowSet, split: Split) -> Tuple[Split, int]:
    """Drop test windows that share samples with any training window.

    A contiguous-segment split still leaks at the segment boundaries: with a
    2048-sample window and a 1024-sample step, the two windows either side of
    each boundary overlap across it. Measured on CWRU with 240 segments, 8.4 % of
    test windows shared samples with training windows before purging. That is
    small, but it is the same mechanism as the random-split leak, and leaving it
    in would mean the "clean" arm of this comparison was 8.4 % dirty.

    Purging the test side rather than the train side keeps the training set
    identical between the purged and unpurged variants, so the difference is
    attributable to the evaluation and not to the training data. Returns the
    purged split and the number of windows removed.
    """
    tr, te = split.train, split.test
    if te.size == 0 or tr.size == 0:
        return split, 0
    win_s = ws.X.shape[1] / ws.fs_hz
    same_rec = ws.recording[tr][:, None] == ws.recording[te][None, :]
    dt = np.abs(ws.t_start_s[tr][:, None] - ws.t_start_s[te][None, :])
    bad = np.any(same_rec & (dt < win_s), axis=0)
    if not np.any(bad):
        return split, 0
    kept = te[~bad]
    if kept.size == 0:
        raise ValueError(f"purging overlap emptied the test fold of {split.name}")
    return Split(train=tr, test=kept, name=f"{split.name}+purged"), int(np.count_nonzero(bad))


def build_split(
    ws: WindowSet,
    kind: str,
    *,
    n_classes: int = 10,
    seed: int = 0,
    test_fraction: float = 0.3,
) -> Split:
    """Produce one split. Raises for a split the data genuinely cannot support.

    ``random_window`` is deliberately NOT purged: its leak is the thing being
    measured. Every grouped split is purged, so the comparison is leaky-vs-clean
    rather than leaky-vs-slightly-less-leaky.
    """
    y = ws.labels(n_classes)
    if kind == "random_window":
        return group_split(ws.window_id, test_fraction=test_fraction, seed=seed, name="random_window")
    if kind in ("segment_grouped", "recording_grouped", "unit_grouped"):
        key = kind.split("_")[0]
        sp = stratified_group_split(ws.groups(key), y, test_fraction=test_fraction, seed=seed)
        return purge_overlap(ws, sp)[0]
    if kind.startswith("cross_load:"):
        held = int(kind.split(":", 1)[1])
        test = np.flatnonzero(ws.load_hp == held)
        train = np.flatnonzero(ws.load_hp != held)
        if test.size == 0 or train.size == 0:
            raise ValueError(f"load {held} is absent or is the only load present")
        missing = set(np.unique(y[train])) ^ set(np.unique(y[test]))
        if missing:
            raise ValueError(f"cross_load:{held} does not share a label space: {sorted(missing)}")
        return Split(train=train, test=test, name=f"cross_load(held_out={held}hp)")
    raise ValueError(f"unknown split kind {kind!r}; expected one of {SPLIT_KINDS}")


def audit(ws: WindowSet, split: Split, *, n_classes: int = 10,
          group_key: str = "recording") -> Dict[str, object]:
    """``pdmcore.eval.splits.leakage_report``, against the *physical* group key.

    ``group_key`` defaults to ``recording`` rather than to whatever key the split
    was built on, because auditing a split against its own key cannot fail. That
    is the whole class of defect ``docs/08_branch_catalogue.md`` tabulates: a
    grouped split audited by the grouping that produced it is a tautology.
    """
    rep = leakage_report(
        split,
        groups=ws.groups(group_key),
        times=ws.t_start_s,
        labels=ws.labels(n_classes),
    )
    rep["group_key_audited"] = group_key
    # Near-duplicate accounting: with 50 % overlap, a train window and a test
    # window from the same recording whose start times differ by less than one
    # window length share samples outright.
    tr, te = split.train, split.test
    same_rec = ws.recording[tr][:, None] == ws.recording[te][None, :]
    dt = np.abs(ws.t_start_s[tr][:, None] - ws.t_start_s[te][None, :])
    win_s = ws.X.shape[1] / ws.fs_hz
    overlapping = same_rec & (dt < win_s)
    rep["n_test_windows_sharing_samples_with_train"] = int(np.count_nonzero(np.any(overlapping, axis=0)))
    rep["frac_test_windows_sharing_samples_with_train"] = float(
        np.mean(np.any(overlapping, axis=0))) if te.size else 0.0
    return rep


def evaluate(y_true: Sequence, y_pred: Sequence, scores: Optional[Sequence[float]] = None,
             *, healthy: str = "healthy") -> Dict[str, object]:
    """Classification metrics plus, when a healthy class exists, detector metrics."""
    yt = np.asarray([str(v) for v in y_true])
    yp = np.asarray([str(v) for v in y_pred])
    classes = sorted(set(yt) | set(yp))
    rep = classification_report(yt, yp, labels=classes)
    out: Dict[str, object] = {
        "accuracy": rep["accuracy"]["value"],
        "macro_f1": rep["macro"]["f1"],
        "weighted_f1": rep["weighted"]["f1"],
        "per_class_f1": {c: rep[c]["f1"] for c in classes},
        "per_class_support": {c: rep[c]["support"] for c in classes},
        "n_test": int(yt.size),
    }
    if scores is not None:
        y_bin = (yt != healthy).astype(int)
        if 0 < int(y_bin.sum()) < y_bin.size:
            out["detector"] = summarise_detector(y_bin, np.asarray(scores, dtype=float)).as_dict()
        else:
            out["detector"] = None
            out["detector_note"] = "only one class present in the binary view; ROC undefined"
    return out


def passes_gate(metrics: Dict[str, object]) -> Dict[str, object]:
    """Apply :data:`GENERALISATION_GATE`. Reports every component, not just the verdict."""
    det = metrics.get("detector") or {}
    got = {
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "tpr_at_fpr1pct": float(det.get("tpr_at_fpr1pct", 0.0)),
        "pauc_10": float(det.get("pauc_10", 0.0)),
    }
    per = {k: bool(got[k] >= v) for k, v in GENERALISATION_GATE.items()}
    return {"required": dict(GENERALISATION_GATE), "achieved": got,
            "per_criterion": per, "passes": all(per.values())}


# ---------------------------------------------------------------------------
# Window-set surgery for the cross-dataset arm
# ---------------------------------------------------------------------------
def subset_classes(ws: WindowSet, keep: Sequence[str], *, n_classes: int = 4) -> WindowSet:
    """Keep only windows whose ``n_classes`` label is in ``keep``."""
    y = ws.labels(n_classes)
    idx = np.flatnonzero(np.isin(y, list(keep)))
    if idx.size == 0:
        raise ValueError(f"no windows left after keeping {list(keep)}")
    return ws.take(idx)


def merge(a: WindowSet, b: WindowSet) -> WindowSet:
    """Concatenate two window sets, prefixing group keys with the source name.

    The prefix is why a cross-dataset split can be audited at all: without it,
    two recordings called ``baseline_1`` from different assemblies would collide
    in the group key and ``leakage_report`` would see one group where there are
    two.
    """
    if a.X.shape[1] != b.X.shape[1]:
        raise ValueError(f"window length differs: {a.X.shape[1]} vs {b.X.shape[1]}")
    if abs(a.fs_hz - b.fs_hz) > 1e-9:
        raise ValueError(f"sample rate differs: {a.fs_hz} vs {b.fs_hz}")

    def tag(ws: WindowSet, field: str) -> np.ndarray:
        return np.asarray([f"{ws.source}/{v}" for v in getattr(ws, field)])

    return WindowSet(
        X=np.vstack([a.X, b.X]),
        label10=np.concatenate([a.label10, b.label10]),
        label4=np.concatenate([a.label4, b.label4]),
        label2=np.concatenate([a.label2, b.label2]),
        unit=np.concatenate([tag(a, "unit"), tag(b, "unit")]),
        recording=np.concatenate([tag(a, "recording"), tag(b, "recording")]),
        segment=np.concatenate([tag(a, "segment"), tag(b, "segment")]),
        window_id=np.concatenate([tag(a, "window_id"), tag(b, "window_id")]),
        load_hp=np.concatenate([a.load_hp, b.load_hp]),
        shaft_hz=np.concatenate([a.shaft_hz, b.shaft_hz]),
        t_start_s=np.concatenate([a.t_start_s, b.t_start_s]),
        fs_hz=a.fs_hz,
        source=f"{a.source}+{b.source}",
    )


def cross_dataset_split(src: WindowSet, tgt: WindowSet) -> Tuple[WindowSet, Split]:
    """``(merged, split)`` with the whole source in train and the whole target in test."""
    both = merge(src, tgt)
    n = len(src)
    return both, Split(train=np.arange(n), test=np.arange(n, len(both)),
                       name=f"cross_dataset({src.source}->{tgt.source})")


def describe(ws: WindowSet, n_classes: int = 10) -> Dict[str, object]:
    y = ws.labels(n_classes)
    return {
        "source": ws.source,
        "n_windows": len(ws),
        "window": int(ws.X.shape[1]),
        "fs_hz": ws.fs_hz,
        "n_units": int(np.unique(ws.unit).size),
        "n_recordings": int(np.unique(ws.recording).size),
        "n_segments": int(np.unique(ws.segment).size),
        "class_counts": {str(c): int(np.count_nonzero(y == c)) for c in np.unique(y)},
        "loads": sorted(int(v) for v in np.unique(ws.load_hp)),
        "shaft_hz_range": [float(ws.shaft_hz.min()), float(ws.shaft_hz.max())],
        "rms_g_range": [float(np.sqrt(np.mean(ws.X**2, axis=1)).min()),
                        float(np.sqrt(np.mean(ws.X**2, axis=1)).max())],
    }


def unit_groups_are_degenerate(ws: WindowSet, n_classes: int) -> Dict[str, object]:
    """Why a unit-grouped split is impossible, as data rather than as prose.

    Counts physical units per class. A unit-grouped split needs >= 2 units in
    every class; this reports the minimum and which classes hit it.
    """
    y = ws.labels(n_classes)
    per_class = {str(c): int(np.unique(ws.unit[y == c]).size) for c in np.unique(y)}
    worst = min(per_class.values())
    return {
        "units_per_class": per_class,
        "min_units_per_class": worst,
        "unit_grouped_split_possible": worst >= 2,
        "singleton_classes": sorted(c for c, n in per_class.items() if n < 2),
    }
