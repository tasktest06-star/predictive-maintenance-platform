#!/usr/bin/env python3
"""Current versus vibration, per fault mode, at a matched false-alarm rate.

    python3 mcsa/prototype/head_to_head.py --out mcsa/results/head_to_head.json

The experiment this branch exists for. One simulated machine, three units, both
channels from the same acquisition, both pipelines calibrated to 1 % false
alarms on held-out healthy data, grouped 3-fold by unit so no unit appears in
both the fit and the test side.

Reported per fault mode:

* ``detection_severity`` -- the lowest severity at which >= 80 % of records
  alarm. The number a sales conversation actually turns on.
* ``auc`` and ``pauc_10`` -- healthy versus that mode, with the partial AUC at
  the 10 % false-positive region because the full AUC averages over operating
  points no plant would accept.
* ``tpr_at_1pct`` -- recall at the shared operating point.
* ``detection_delay_h`` -- on a 90-day run-to-failure series, from the
  ``pdmcore.eval.metrics.detection_delay`` definition with two consecutive
  exceedances required.

Run twice: ``equal_cost`` (1 s vibration, 10 s current) and ``equal_time``
(1 s both).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    CURRENT_S_LONG,
    CURRENT_S_SHORT,
    FAULT_SEVERITIES,
    MODES,
    SPEED_REL_ERROR,
    Record,
    build_dataset,
    current_n_fft,
    detection_severity,
    extract_all,
    machine_spec,
    score_fold,
)

from pdmcore.eval.metrics import auc, detection_delay, partial_auc  # noqa: E402
from pdmcore.eval.splits import group_kfold, leakage_report          # noqa: E402
from pdmcore.mcsa import assess_supply, mcsa_admissible              # noqa: E402
from pdmcore.mcsa.supply import estimate_slip_from_sidebands         # noqa: E402
from pdmcore.sim import degradation_series                           # noqa: E402


def _per_mode_metrics(records: List[Record], fold, mode: str) -> Dict[str, object]:
    """AUC / pAUC / TPR for one mode against the healthy records of this fold."""
    lab, sc = [], []
    for i, j in enumerate(fold.test_index):
        r = records[j]
        if r.severity <= 0:
            lab.append(0)
            sc.append(fold.scores[i])
        elif r.mode == mode:
            lab.append(1)
            sc.append(fold.scores[i])
    lab = np.asarray(lab)
    sc = np.asarray(sc)
    if lab.sum() == 0 or (lab == 0).sum() == 0:      # pragma: no cover
        return {}
    pos = sc[lab == 1]
    return {
        "auc": float(auc(lab, sc)),
        "pauc_10": float(partial_auc(lab, sc, 0.1)),
        "tpr_at_1pct": float(np.mean(pos > fold.threshold)),
        "fpr_on_test_healthy": float(np.mean(sc[lab == 0] > fold.threshold)),
        "n_pos": int(lab.sum()),
        "n_neg": int((lab == 0).sum()),
    }


def run_variant(variant: str, current_s: float, *, n_healthy: int, n_reps: int,
                seed0: int) -> Dict[str, object]:
    t0 = time.time()
    records = build_dataset(n_healthy=n_healthy, n_reps=n_reps,
                            current_s=current_s, seed0=seed0)
    feats = extract_all(records, current_s=current_s, estimate_slip=True)
    groups = [f"u{r.unit}" for r in records]
    labels = [f"{r.mode}@{r.severity:g}" for r in records]

    out: Dict[str, object] = {
        "variant": variant,
        "current_s": current_s,
        "vib_s": 1.0,
        "current_n_fft": current_n_fft(current_s),
        "n_records": len(records),
        "n_healthy": int(sum(1 for r in records if r.severity <= 0)),
        "slip_error_pct": {
            "median": float(100.0 * np.median(feats["slip_err"])),
            "p90": float(100.0 * np.quantile(feats["slip_err"], 0.9)),
            "max": float(100.0 * np.max(feats["slip_err"])),
        },
        "slip_estimate_valid_frac": float(
            np.mean([g["valid"] for g in feats["gates"]])),
        "folds": [],
        "per_mode": {},
    }

    acc: Dict[str, Dict[str, List[float]]] = {}
    for fold_i, split in enumerate(group_kfold(groups, n_splits=3, seed=0)):
        audit = leakage_report(split, groups=groups, labels=labels)
        train_units = sorted({records[i].unit for i in split.train})
        test_units = sorted({records[i].unit for i in split.test})
        fold_row: Dict[str, object] = {
            "fold": fold_i,
            "train_units": train_units,
            "test_units": test_units,
            "leakage_ok": bool(audit["ok"]),
            "leakage_violations": list(audit["violations"]),
        }
        for tag, X in (("vib", feats["X_vib"]), ("cur", feats["X_cur"])):
            names = feats["names_vib"] if tag == "vib" else feats["names_cur"]
            fs = score_fold(X, records, train_units, test_units, fpr=0.01,
                            names=names, seed=fold_i)
            fold_row[tag] = {
                "threshold": fs.threshold,
                "n_fit": fs.n_fit,
                "shrinkage": fs.shrinkage,
                "samples_per_feature": fs.samples_per_feature,
            }
            for mode in MODES:
                m = _per_mode_metrics(records, fs, mode)
                ds = detection_severity(records, fs, mode, FAULT_SEVERITIES)
                key = f"{tag}:{mode}"
                a = acc.setdefault(key, {"auc": [], "pauc_10": [], "tpr_at_1pct": [],
                                         "detection_severity": [], "fpr": []})
                a["auc"].append(m["auc"])
                a["pauc_10"].append(m["pauc_10"])
                a["tpr_at_1pct"].append(m["tpr_at_1pct"])
                a["fpr"].append(m["fpr_on_test_healthy"])
                a["detection_severity"].append(
                    ds["detection_severity"] if ds["detection_severity"] is not None
                    else float("nan"))
                if fold_i == 0:
                    out["per_mode"].setdefault(mode, {})[f"{tag}_per_severity"] = \
                        ds["per_severity"]
        out["folds"].append(fold_row)

    for mode in MODES:
        row = out["per_mode"].setdefault(mode, {})
        for tag in ("vib", "cur"):
            a = acc[f"{tag}:{mode}"]
            det = [v for v in a["detection_severity"] if v == v]
            row[tag] = {
                "auc": float(np.mean(a["auc"])),
                "auc_sd": float(np.std(a["auc"])),
                "pauc_10": float(np.mean(a["pauc_10"])),
                "tpr_at_1pct": float(np.mean(a["tpr_at_1pct"])),
                "fpr_on_test_healthy": float(np.mean(a["fpr"])),
                "detection_severity": float(np.mean(det)) if det else None,
                "detection_severity_folds": [
                    None if v != v else float(v) for v in a["detection_severity"]],
            }
        v, c = row["vib"], row["cur"]
        row["verdict"] = _verdict(v, c)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _verdict(vib: Dict[str, object], cur: Dict[str, object]) -> str:
    """The decision rule from ``mcsa/research/03_evaluation_protocol.md``.

    Fixed before the results existed: current is *sufficient* for a mode when
    it detects at a severity no worse than 1.5x the vibration severity and
    reaches pAUC(10 %) >= 0.9; *complementary* when it detects at all but
    later; *blind* when it never reaches the 80 % hit rate at severity 1.0.
    """
    dv, dc = vib["detection_severity"], cur["detection_severity"]
    if dc is None:
        return "current_blind"
    if dv is None:
        return "current_only"
    if dc <= 1.5 * dv and float(cur["pauc_10"]) >= 0.9:
        return "current_sufficient"
    return "current_complementary"


def run_delay(mode: str, *, n_points: int = 90, days: float = 90.0,
              current_s: float = CURRENT_S_LONG) -> Dict[str, object]:
    """Detection delay on a run-to-failure series, both modalities.

    The series is a single unit, so the detector is fitted on the *early* part
    of its own history -- which is what a commissioning window actually is --
    and the split is chronological with a gap, not grouped.
    """
    from common import cur_extractor, vib_extractor

    spec = machine_spec(0)
    series = degradation_series(spec, mode=mode, n_points=n_points, duration_s=1.0,
                               days=days, seed=7000, with_current=True,
                               speed_from_load=True, current_duration_s=current_s)
    vx = vib_extractor()
    cx = cur_extractor(current_n_fft(current_s))
    xv, xc, sev, t_days = [], [], [], []
    rng = np.random.default_rng(31337)
    for r in series:
        believed = float(r.spec.shaft_hz * (1.0 + SPEED_REL_ERROR * rng.standard_normal()))
        xv.append(vx.extract(r.accel_g, believed, fs_hz=r.fs_hz).vector)
        est = estimate_slip_from_sidebands(r.current_abc_a[0], r.current_fs_hz,
                                           line_hz=60.0, pole_pairs=2,
                                           n_fft=current_n_fft(current_s))
        slip = est.slip if est.valid and est.slip > 0 else r.electrical_truth["slip"]
        xc.append(cx.extract(r.current_abc_a, slip=max(slip, 1e-5)).vector)
        sev.append(r.truth["severity"])
        t_days.append(r.truth["t_days"])
    Xv, Xc = np.vstack(xv), np.vstack(xc)
    sev = np.asarray(sev)
    onset = int(np.flatnonzero(sev > 0.02)[0])
    # Commissioning: the first 45 % of the series, split into fit and calibrate.
    # Both parts are before the knee, so the window is healthy-by-construction
    # in the same way a real commissioning window is: the machine was running
    # and nobody had reported a problem.
    n_fit = max(12, int(0.30 * len(series)))
    n_cal = max(12, int(0.15 * len(series)))
    if n_fit + n_cal > onset:
        raise ValueError(
            f"commissioning window ({n_fit}+{n_cal}) reaches past fault onset "
            f"at {onset}: raise n_points")
    interval_h = days * 24.0 / (n_points - 1)
    row: Dict[str, object] = {
        "mode": mode, "n_points": n_points, "days": days,
        "interval_h": interval_h, "onset_index": onset,
        "onset_day": float(t_days[onset]),
        "severity_at_onset": float(sev[onset]),
    }
    from pdmcore.models.anomaly import MahalanobisDetector
    for tag, X, names in (("vib", Xv, vx.names), ("cur", Xc, cx.names)):
        det = MahalanobisDetector(feature_names=list(names))
        det.fit(X[:n_fit])
        thr = det.calibrate(X[n_fit:n_fit + n_cal], fpr=0.01)
        scores = det.score(X)
        delay = detection_delay(scores, thr, onset_index=onset,
                                sample_interval_h=interval_h, consecutive_required=2)
        idx = None
        run = 0
        for i in range(onset, len(scores)):
            run = run + 1 if scores[i] > thr else 0
            if run >= 2:
                idx = i
                break
        row[tag] = {
            "threshold": float(thr),
            "delay_h": None if delay is None else float(delay),
            "delay_days": None if delay is None else float(delay / 24.0),
            "severity_at_detection": None if idx is None else float(sev[idx]),
            "day_at_detection": None if idx is None else float(t_days[idx]),
            "false_alarms_before_onset": int(np.sum(scores[:onset] > thr)),
        }
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="mcsa/results/head_to_head.json")
    ap.add_argument("--healthy", type=int, default=80)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="small dataset, for checking the plumbing only")
    args = ap.parse_args(argv)

    n_healthy = 20 if args.quick else args.healthy
    n_reps = 1 if args.quick else args.reps

    result: Dict[str, object] = {
        "machine": "motor_pump_37kw (37 kW, 4 pole, 60 Hz, 1780 rpm, SKF6205)",
        "speed_rel_error_vib": SPEED_REL_ERROR,
        "slip_source_cur": "estimate_slip_from_sidebands (rotation family)",
        "detector": "MahalanobisDetector, Ledoit-Wolf shrinkage, "
                    "calibrated at 1 % FPR on held-out healthy",
        "split": "group_kfold(3) by unit, leakage audited",
        "severities": list(FAULT_SEVERITIES),
        "variants": {},
        "delay": {},
        "quick": bool(args.quick),
    }
    def _write() -> None:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)

    for variant, cs in (("equal_cost", CURRENT_S_LONG), ("equal_time", CURRENT_S_SHORT)):
        print(f"== {variant}: current {cs:g} s", flush=True)
        result["variants"][variant] = run_variant(
            variant, cs, n_healthy=n_healthy, n_reps=n_reps, seed0=20000)
        _write()

    for mode in ("bearing_outer", "rotor_bar", "eccentricity_mixed"):
        print(f"== delay: {mode}", flush=True)
        result["delay"][mode] = run_delay(mode, n_points=60 if args.quick else 120)
        _write()
    print("wrote", args.out)
    for variant, v in result["variants"].items():
        print(f"\n{variant}: slip error median {v['slip_error_pct']['median']:.1f} %")
        print(f"  {'mode':22s} {'vib sev':>8s} {'cur sev':>8s} {'vib pAUC':>9s} "
              f"{'cur pAUC':>9s}  verdict")
        for mode, row in v["per_mode"].items():
            dv = row["vib"]["detection_severity"]
            dc = row["cur"]["detection_severity"]
            print(f"  {mode:22s} {('-' if dv is None else f'{dv:.2f}'):>8s} "
                  f"{('-' if dc is None else f'{dc:.2f}'):>8s} "
                  f"{row['vib']['pauc_10']:9.3f} {row['cur']['pauc_10']:9.3f}  "
                  f"{row['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
