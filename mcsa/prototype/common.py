"""Shared rig for the MCSA experiments: one machine, two pipelines, one split.

Both pipelines are the platform's own. The vibration side is
``pdmcore.features.FeatureExtractor`` -- the 63-feature contract the firmware
emits -- and the current side is ``pdmcore.mcsa.CurrentFeatureExtractor``.
Both are scored by ``pdmcore.models.anomaly.MahalanobisDetector``, fitted on
healthy records from the training units and **calibrated on held-out healthy
records**, per the warning in ``AnomalyDetector.calibrate``.

The two things that decide whether the comparison is fair
---------------------------------------------------------
1. **Both devices get the speed input they would really have.** A battery
   vibration node has no tachometer on most installations, so it gets the
   nameplate shaft speed with a 0.5 % error -- which is what
   ``order_bands(rel_width=0.06)`` is already sized for. A panel current device
   has no speed input at all, so it estimates its own slip from the rotation
   sidebands (``estimate_slip_from_sidebands``), which is a capability the
   modality genuinely has and vibration genuinely does not. Giving both the
   true slip would flatter current, because current's fault frequencies depend
   on slip and vibration's do not.
2. **Both are calibrated to the same false-alarm rate on the same held-out
   healthy data.** Detection severity is then comparable because the operating
   point is.

Acquisition lengths differ, deliberately, and both variants are reported:

=================  =====================  ======================================
variant            vibration              current
=================  =====================  ======================================
``equal_cost``     1 s at 25.6 kHz        10 s at 10 kHz
``equal_time``     1 s at 25.6 kHz        1 s at 10 kHz
=================  =====================  ======================================

``equal_cost`` is the deployment comparison: both devices cost roughly the
same, and the battery node cannot afford a 10 s acquisition because the
acquisition window is 63-85 % of its energy budget (``tools/power_budget.py``),
while a panel device is mains-powered and can afford 30 s. ``equal_time`` is
the modality comparison, and it exists because the difference between the two
is itself one of this branch's results.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_REPO, "core"), os.path.join(_REPO, "edge"),
           os.path.join(_REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pdmcore.dsp.bearing import BEARING_CATALOG                       # noqa: E402
from pdmcore.features import FeatureConfig, FeatureExtractor          # noqa: E402
from pdmcore.mcsa import (                                            # noqa: E402
    CurrentFeatureConfig,
    CurrentFeatureExtractor,
    assess_supply,
    estimate_slip_from_sidebands,
    mcsa_admissible,
)
from pdmcore.models.anomaly import MahalanobisDetector                # noqa: E402
from pdmcore.sim import MACHINE_LIBRARY, SupplySpec, simulate         # noqa: E402

#: The reference asset for every experiment on this branch: a 37 kW 4-pole
#: motor-pump on a 60 Hz supply, nameplate 1780 rpm (slip 1.11 %), SKF 6205
#: bearings. Chosen because it is the platform's default machine, so the
#: vibration numbers here are comparable with every other branch's.
MACHINE = "motor_pump_37kw"

#: Current channel: 10 kHz is the lowest rate that still sees a 4 kHz inverter
#: carrier (so the VFD gate can fire) and the rotor-slot family up to ~4 kHz.
CURRENT_FS_HZ = 10000.0
CURRENT_S_LONG = 10.0
CURRENT_S_SHORT = 1.0
VIB_S = 1.0

#: Speed error a device without a tachometer actually has: a nameplate rpm is
#: quoted at rated load and the machine is rarely at rated load.
SPEED_REL_ERROR = 0.005

FAULT_SEVERITIES = (0.1, 0.2, 0.35, 0.55, 0.8, 1.0)

#: The modes compared head to head, and which modality the textbook assigns
#: them to. The point of the experiment is to check the assignment.
MODES: Tuple[str, ...] = (
    "rotor_bar",
    "eccentricity_mixed",
    "stator_asymmetry",
    "bearing_outer",
    "bearing_inner",
    "unbalance",
    "misalignment",
)


def machine_spec(unit: int = 0, *, rpm_jitter_pct: float = 3.0,
                 supply: Optional[SupplySpec] = None, coupling=None):
    """One unit of the reference asset.

    ``rpm_jitter_pct`` per unit is the simulated analogue of the data-leakage
    problem on CWRU: without it a model can memorise an exact shaft frequency
    and the grouped split stops meaning anything.
    """
    base = MACHINE_LIBRARY[MACHINE]
    rng = np.random.default_rng(9000 + unit)
    jitter = 1.0 + rpm_jitter_pct / 100.0 * rng.standard_normal()
    sup = supply or SupplySpec(current_fs_hz=CURRENT_FS_HZ)
    spec = replace(base, rpm=float(base.rpm * jitter), name=f"{MACHINE}_u{unit}",
                   supply=sup, rotor_slots=28)
    if coupling is not None:
        spec = replace(spec, coupling=coupling)
    return spec


@dataclass
class Record:
    """One acquisition of both channels, with ground truth and both speed inputs."""

    unit: int
    mode: str
    severity: float
    accel_g: np.ndarray
    fs_hz: float
    current_abc_a: np.ndarray
    voltage_abc_v: np.ndarray
    current_fs_hz: float
    #: Shaft speed as the vibration node believes it: nameplate + 0.5 % error.
    shaft_hz_believed: float
    truth: Dict[str, float]
    electrical_truth: Dict[str, float]

    @property
    def label(self) -> int:
        return 0 if self.severity <= 0 else 1


def make_record(unit: int, mode: Optional[str], severity: float, seed: int, *,
                current_s: float = CURRENT_S_LONG, spec=None) -> Record:
    spec = spec if spec is not None else machine_spec(unit)
    faults = {mode: severity} if mode and severity > 0 else None
    res = simulate(spec, faults, duration_s=VIB_S, seed=seed, with_current=True,
                   speed_from_load=True, current_duration_s=current_s)
    rng = np.random.default_rng(seed ^ 0x5EED)
    believed = float(spec.shaft_hz * (1.0 + SPEED_REL_ERROR * rng.standard_normal()))
    return Record(
        unit=unit,
        mode=mode or "healthy",
        severity=float(severity),
        accel_g=res.accel_g,
        fs_hz=res.fs_hz,
        current_abc_a=res.current_abc_a,
        voltage_abc_v=res.voltage_abc_v,
        current_fs_hz=float(res.current_fs_hz),
        shaft_hz_believed=believed,
        truth=dict(res.truth),
        electrical_truth=dict(res.electrical_truth),
    )


def build_dataset(
    *,
    units: Sequence[int] = (0, 1, 2),
    modes: Sequence[str] = MODES,
    severities: Sequence[float] = FAULT_SEVERITIES,
    n_healthy: int = 80,
    n_reps: int = 3,
    current_s: float = CURRENT_S_LONG,
    seed0: int = 20000,
    spec_for=None,
) -> List[Record]:
    """Healthy + fault records for every unit. Deterministic in ``seed0``."""
    out: List[Record] = []
    counter = 0
    for unit in units:
        spec = spec_for(unit) if spec_for else machine_spec(unit)
        for _ in range(n_healthy):
            counter += 1
            out.append(make_record(unit, None, 0.0, seed0 + counter,
                                   current_s=current_s, spec=spec))
        for mode in modes:
            for sev in severities:
                for _ in range(n_reps):
                    counter += 1
                    out.append(make_record(unit, mode, float(sev), seed0 + counter,
                                           current_s=current_s, spec=spec))
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def vib_extractor(fs_hz: float = 25600.0) -> FeatureExtractor:
    return FeatureExtractor(FeatureConfig(fs_hz=fs_hz, n_fft=4096,
                                          bearing=BEARING_CATALOG["SKF6205"]))


def cur_extractor(n_fft: int, line_hz: float = 60.0) -> CurrentFeatureExtractor:
    return CurrentFeatureExtractor(CurrentFeatureConfig(
        fs_hz=CURRENT_FS_HZ, n_fft=n_fft, line_hz=line_hz, pole_pairs=2,
        bearing=BEARING_CATALOG["SKF6205"], speed_rel_error=SPEED_REL_ERROR,
    ))


def current_n_fft(current_s: float) -> int:
    """Largest power-of-two block that fits the record, capped at 2^17."""
    n = int(current_s * CURRENT_FS_HZ)
    return int(min(1 << 17, 1 << int(np.floor(np.log2(max(2, n))))))


def extract_all(records: Sequence[Record], *, current_s: float = CURRENT_S_LONG,
                estimate_slip: bool = True) -> Dict[str, object]:
    """Both feature matrices, the slip estimates and the gate verdicts.

    Returns ``X_vib``, ``X_cur``, ``names_vib``, ``names_cur``, ``slip_est``,
    ``slip_err`` (actual, against the simulator's truth) and ``gate`` rows.
    """
    vx = vib_extractor()
    cx = cur_extractor(current_n_fft(current_s))
    xv, xc = [], []
    slips, slip_err, gates = [], [], []
    for r in records:
        xv.append(vx.extract(r.accel_g, r.shaft_hz_believed, fs_hz=r.fs_hz).vector)
        truth_slip = r.electrical_truth["slip"]
        if estimate_slip:
            est = estimate_slip_from_sidebands(
                r.current_abc_a[0], r.current_fs_hz, line_hz=60.0, pole_pairs=2,
                n_fft=current_n_fft(current_s))
            slip = est.slip if est.valid and est.slip > 0 else truth_slip
            slips.append(est.slip)
            slip_err.append(abs(est.slip - truth_slip) / truth_slip)
            gates.append({"valid": bool(est.valid), "rel_error": float(est.rel_error),
                          "snr": float(est.snr)})
        else:
            # The slip a device with no speed input would assume: nameplate.
            slip = float(1.0 - r.shaft_hz_believed * 2.0 / 60.0)
            slips.append(slip)
            slip_err.append(abs(slip - truth_slip) / truth_slip)
            gates.append({"valid": True, "rel_error": float("nan"), "snr": float("nan")})
        xc.append(cx.extract(r.current_abc_a, slip=max(slip, 1e-5)).vector)
    return {
        "X_vib": np.vstack(xv),
        "X_cur": np.vstack(xc),
        "names_vib": list(vx.names),
        "names_cur": list(cx.names),
        "slip_est": slips,
        "slip_err": slip_err,
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class FoldScores:
    """One fold's scores for one modality."""

    threshold: float
    scores: np.ndarray
    test_index: np.ndarray
    n_fit: int
    shrinkage: float
    samples_per_feature: float


def score_fold(X: np.ndarray, records: Sequence[Record], train_units: Sequence[int],
               test_units: Sequence[int], *, fpr: float = 0.01,
               calib_fraction: float = 0.35, names: Optional[Sequence[str]] = None,
               seed: int = 0) -> FoldScores:
    """Fit on healthy from the training units, calibrate on held-out healthy.

    The healthy records of the training units are split into a fit part and a
    calibration part. The threshold comes from the calibration part only --
    ``threshold_at_fpr`` on the fitted scores is 15.4x too low for a detector
    with any capacity, which is one of the six "checks that cannot fail" in
    ``docs/08_branch_catalogue.md``.
    """
    units = np.array([r.unit for r in records])
    healthy = np.array([r.severity <= 0 for r in records])
    tr = np.isin(units, list(train_units)) & healthy
    idx = np.flatnonzero(tr)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_cal = max(10, int(round(calib_fraction * idx.size)))
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    det = MahalanobisDetector(feature_names=names)
    det.fit(X[fit_idx])
    thr = det.calibrate(X[cal_idx], fpr=fpr)
    test_index = np.flatnonzero(np.isin(units, list(test_units)))
    return FoldScores(
        threshold=float(thr),
        scores=det.score(X[test_index]),
        test_index=test_index,
        n_fit=int(fit_idx.size),
        shrinkage=float(det.shrinkage_used_),
        samples_per_feature=float(det.samples_per_feature_),
    )


def detection_severity(records: Sequence[Record], fold: FoldScores, mode: str,
                       severities: Sequence[float], *, hit_fraction: float = 0.8
                       ) -> Dict[str, object]:
    """Lowest severity at which at least ``hit_fraction`` of records alarm.

    Reported with the per-severity hit rate, because a single crossing severity
    hides a non-monotonic detector -- and the one thing worse than a late
    detection is a detector that fires at severity 0.2 and stops at 0.55.
    """
    rows = []
    out_sev = None
    for sev in severities:
        sel = [i for i, j in enumerate(fold.test_index)
               if records[j].mode == mode and abs(records[j].severity - sev) < 1e-9]
        if not sel:
            rows.append({"severity": float(sev), "hit_rate": float("nan"), "n": 0})
            continue
        hits = float(np.mean(fold.scores[sel] > fold.threshold))
        rows.append({"severity": float(sev), "hit_rate": hits, "n": len(sel),
                     "median_score": float(np.median(fold.scores[sel]))})
        if out_sev is None and hits >= hit_fraction:
            out_sev = float(sev)
    return {"mode": mode, "detection_severity": out_sev, "per_severity": rows}
