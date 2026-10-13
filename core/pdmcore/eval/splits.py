"""Leakage-resistant data splits.

The reported accuracy of bearing-fault classifiers on public datasets is
systematically inflated, and the mechanism is always the same: consecutive
windows cut from one continuous recording of one physical bearing end up in both
train and test. The model then recognises *that recording* -- its exact shaft
speed, its exact resonance map, its exact noise floor -- and reports 99 %.
Partition by physical unit instead and the same model reports 70-85 %.

This is not a subtle statistical nicety. A model validated with a random split
will be deployed, will see a bearing it has never met, and will be wrong, and
the customer will stop trusting every alarm the platform ever raises.

Rules enforced here
-------------------
1. :func:`group_split` -- never let one physical unit (bearing, machine, site)
   appear on both sides.
2. :func:`time_split` -- for anything that trends or predicts, test must be
   strictly *later* than train. A random split lets the model interpolate
   degradation it has already seen the end of.
3. :func:`stratified_group_split` -- keep the class balance while still grouping,
   because fault classes are rarely balanced and a group split can otherwise
   put every single outer-race example in the test fold.

:func:`leakage_report` checks a split that someone else produced and is wired
into CI for every model card in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Split",
    "group_split",
    "stratified_group_split",
    "time_split",
    "group_kfold",
    "leakage_report",
]


@dataclass(frozen=True)
class Split:
    """Index arrays for one train/test partition."""

    train: np.ndarray
    test: np.ndarray
    name: str = "split"

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.test)
        if overlap.size:
            raise ValueError(f"train and test share {overlap.size} indices")

    @property
    def sizes(self) -> Tuple[int, int]:
        return int(self.train.size), int(self.test.size)


def _as_groups(groups: Sequence) -> np.ndarray:
    return np.asarray([str(g) for g in groups])


def group_split(
    groups: Sequence,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
    name: str = "group",
) -> Split:
    """Hold out whole groups.

    Groups are shuffled and assigned to the test side until ``test_fraction`` of
    *samples* is reached, so an imbalanced group size does not produce a 5 %
    test set.
    """
    g = _as_groups(groups)
    uniq, counts = np.unique(g, return_counts=True)
    if uniq.size < 2:
        raise ValueError(f"need >= 2 distinct groups to split, got {uniq.size}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq.size)
    target = test_fraction * g.size
    chosen: List[str] = []
    running = 0
    for i in order:
        if running >= target and chosen:
            break
        chosen.append(str(uniq[i]))
        running += int(counts[i])
    test_mask = np.isin(g, chosen)
    if not np.any(~test_mask):
        raise ValueError("test_fraction too large: no groups left for training")
    return Split(
        train=np.flatnonzero(~test_mask),
        test=np.flatnonzero(test_mask),
        name=f"{name}(held_out={','.join(sorted(chosen))})",
    )


def stratified_group_split(
    groups: Sequence,
    labels: Sequence,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
    max_tries: int = 200,
) -> Split:
    """Group split that also keeps every label present on both sides.

    Retries random group assignments and keeps the one whose class distribution
    best matches the overall distribution. Raises if no attempt manages to put
    every class on both sides -- which is a genuine signal that the dataset
    cannot support a grouped evaluation for that class, and the honest response
    is to collect more units, not to fall back to a random split.
    """
    g = _as_groups(groups)
    y = np.asarray([str(v) for v in labels])
    if g.size != y.size:
        raise ValueError("groups and labels must have the same length")
    classes = np.unique(y)
    overall = np.array([np.mean(y == c) for c in classes])
    best: Optional[Split] = None
    best_cost = np.inf
    for attempt in range(max_tries):
        try:
            cand = group_split(g, test_fraction=test_fraction, seed=seed + attempt)
        except ValueError:
            continue
        ytr, yte = y[cand.train], y[cand.test]
        if not all((np.any(ytr == c) and np.any(yte == c)) for c in classes):
            continue
        dist_te = np.array([np.mean(yte == c) for c in classes])
        dist_tr = np.array([np.mean(ytr == c) for c in classes])
        cost = float(np.sum(np.abs(dist_te - overall)) + np.sum(np.abs(dist_tr - overall)))
        cost += 2.0 * abs(cand.test.size / g.size - test_fraction)
        if cost < best_cost:
            best_cost, best = cost, Split(cand.train, cand.test, name=f"stratified_group({attempt})")
    if best is None:
        raise ValueError(
            "no grouped split keeps every class on both sides; collect more units "
            "per class rather than falling back to a random split"
        )
    return best


def time_split(
    times: Sequence[float],
    *,
    test_fraction: float = 0.3,
    gap_fraction: float = 0.05,
) -> Split:
    """Chronological split with a gap.

    The gap matters whenever features are computed on overlapping windows or a
    trend filter (EWMA) carries state: without it the last training sample and
    the first test sample share information and the split leaks in miniature.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size < 4:
        raise ValueError("need >= 4 samples for a time split")
    order = np.argsort(t)
    n = t.size
    n_test = max(1, int(round(test_fraction * n)))
    n_gap = int(round(gap_fraction * n))
    train_end = n - n_test - n_gap
    if train_end < 1:
        raise ValueError("test_fraction + gap_fraction leaves no training data")
    return Split(
        train=np.sort(order[:train_end]),
        test=np.sort(order[n - n_test :]),
        name=f"time(gap={n_gap})",
    )


def group_kfold(groups: Sequence, *, n_splits: int = 5, seed: int = 0) -> Iterator[Split]:
    """Yield ``n_splits`` grouped folds, each group in exactly one test fold."""
    g = _as_groups(groups)
    uniq = np.unique(g)
    if uniq.size < n_splits:
        raise ValueError(f"need >= {n_splits} groups for {n_splits}-fold, got {uniq.size}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(uniq)
    folds = np.array_split(shuffled, n_splits)
    for k, fold in enumerate(folds):
        mask = np.isin(g, fold)
        yield Split(
            train=np.flatnonzero(~mask),
            test=np.flatnonzero(mask),
            name=f"group_kfold[{k + 1}/{n_splits}]",
        )


def leakage_report(
    split: Split,
    *,
    groups: Optional[Sequence] = None,
    times: Optional[Sequence[float]] = None,
    labels: Optional[Sequence] = None,
) -> Dict[str, object]:
    """Audit a split. ``ok`` is False if any hard rule is violated.

    Wired into CI: a model card whose split fails this audit cannot be
    published. See ``docs/06_test_strategy.md``.
    """
    out: Dict[str, object] = {"ok": True, "violations": [], "warnings": []}
    violations: List[str] = []
    warnings: List[str] = []

    if np.intersect1d(split.train, split.test).size:
        violations.append("train and test share sample indices")

    if groups is not None:
        g = _as_groups(groups)
        shared = np.intersect1d(np.unique(g[split.train]), np.unique(g[split.test]))
        if shared.size:
            violations.append(
                f"{shared.size} group(s) appear in both folds: {', '.join(map(str, shared[:5]))}"
            )
        out["n_train_groups"] = int(np.unique(g[split.train]).size)
        out["n_test_groups"] = int(np.unique(g[split.test]).size)

    if times is not None:
        t = np.asarray(times, dtype=np.float64)
        if t[split.test].min() < t[split.train].max():
            warnings.append(
                f"test starts at t={t[split.test].min():.3f} before train ends at "
                f"t={t[split.train].max():.3f}; acceptable only for non-temporal models"
            )

    if labels is not None:
        y = np.asarray([str(v) for v in labels])
        for c in np.unique(y):
            if not np.any(y[split.test] == c):
                warnings.append(f"class {c!r} absent from test fold")
            if not np.any(y[split.train] == c):
                violations.append(f"class {c!r} absent from train fold")
        out["test_class_counts"] = {
            str(c): int(np.count_nonzero(y[split.test] == c)) for c in np.unique(y)
        }

    out["violations"] = violations
    out["warnings"] = warnings
    out["ok"] = not violations
    out["sizes"] = split.sizes
    return out
