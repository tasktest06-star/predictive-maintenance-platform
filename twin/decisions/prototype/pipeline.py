#!/usr/bin/env python3
"""Run the real pipeline on simulated run-to-failure lives, and keep the result.

This is the only stage on this branch that executes the DSP. Everything
downstream -- the value-of-information study, the scheduler study and the
multi-year fleet economics -- consumes what this stage measured, so that those
stages can be swept over hundreds of parameter combinations without re-running
25 kHz signal processing 40 000 times.

The chain, with nothing substituted:

    pdmcore.sim.degradation_series   acceleration with a two-stage severity
                                     profile and ground-truth remaining life
    pdmcore.features.FeatureExtractor      the 63-feature contract
    pdmcore.models.anomaly.MahalanobisDetector
                                     one-class detection, fitted on healthy
                                     frames and calibrated on a HELD-OUT
                                     healthy split
    pdmcore.health.HealthIndex       operating-point baseline, EWMA trend,
                                     hysteretic alarm state machine
    pdmcore.health.ScenarioRUL       scenario-conditional RUL over the
                                     particle filter

What is deliberately **not** simulated acquisition-by-acquisition: the healthy
stage of each life beyond the commissioning window. ``pdmcore.sim`` produces a
stationary signal for a healthy machine, so simulating three hundred flat days
costs three hundred DSP runs and adds no information. Each life therefore
consists of a commissioning window plus the degradation stage from onset to
failure, and the healthy duration is drawn as a number rather than as a signal.
That is a modelling shortcut and it is recorded in the output.

What the load history is, and what it is not: each life is given a declared
constant duty, recorded alongside every health observation so the
scenario-conditional estimator has a reference load. ``pdmcore.sim`` has **no
load-dependent damage model** -- its ``load_factor`` is a per-acquisition
amplitude nuisance, not a damage rate -- so nothing here validates the load-life
exponent. What is exercised is the conditional arithmetic, its reduction to the
unconditional answer at the observed duty, and every abstention gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if os.path.join(_ROOT, "core") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "core"))

from pdmcore.features import FeatureConfig, FeatureExtractor  # noqa: E402
from pdmcore.health import AlarmConfig, AlarmState, HealthIndex  # noqa: E402
from pdmcore.health.rul import ParticleFilterRUL  # noqa: E402
from pdmcore.health.scenario_rul import LoadScenario, ScenarioRUL  # noqa: E402
from pdmcore.models.anomaly import MahalanobisDetector  # noqa: E402
from pdmcore.sim import MACHINE_LIBRARY, degradation_series, simulate  # noqa: E402

RESULTS = os.path.join(_HERE, "results")

#: Machine presets used for the fleet. The slow agitator is included because it
#: is the preset that breaks a default measurement band, and a decision study
#: that only ever sees well-behaved assets is a study of a well-behaved plant.
PRESETS = ("motor_pump_37kw", "fan_7kw_belt", "gearbox_132kw", "vfd_compressor_75kw")

#: Bearing-family degradation modes. The decision layer does not care which, but
#: detection lead time does, and the spread across modes is part of the result.
MODES = ("bearing_outer", "bearing_inner", "bearing_ball", "looseness")

#: Same prognostic indicator and thresholds the ingest service already uses:
#: high-frequency band RMS normalised to its healthy median, because the capped
#: health index saturates and an exponential fit to a saturating signal
#: concludes the crossing has already happened.
RUL_FAILURE_THRESHOLD = 20.0
RUL_MIN_HEALTH = 2.0
RUL_INDICATOR = "band.10_rms"

#: Commissioning frames per life. Matches cloud/ingest/app.py's
#: AssetState.COMMISSION_FRAMES, plus the held-out split the detector needs.
N_COMMISSION = 60
#: Detector calibration target. Matches the platform's 1 % operating point.
DETECTOR_FPR = 0.01


def _extractor(spec) -> FeatureExtractor:
    return FeatureExtractor(FeatureConfig(
        fs_hz=spec.fs_hz,
        demod_band_hz=(2500.0, min(6000.0, spec.fs_hz / 2 * 0.45)),
        velocity_band_hz=(2.0, 1000.0) if spec.shaft_hz < 10.0 else (10.0, 1000.0),
    ))


@dataclass
class Life:
    """One asset life, from commissioning to failure, as the cloud would see it."""

    life_id: str
    preset: str
    mode: str
    duty: float
    healthy_days: float
    degradation_days: float
    #: Days from the start of the degradation stage.
    t_days: List[float] = field(default_factory=list)
    #: Health index in alarm units (1.0 == the detector's 1 % FPR threshold).
    index: List[float] = field(default_factory=list)
    #: Non-saturating prognostic indicator, normalised to its healthy median.
    indicator: List[float] = field(default_factory=list)
    alarm_state: List[str] = field(default_factory=list)
    #: True remaining life at each observation, from the simulator's ground truth.
    true_rul_days: List[float] = field(default_factory=list)
    #: Nuisance alarms raised during this life's commissioning window.
    commissioning_alarms: int = 0
    healthy_scores_above_threshold: int = 0
    n_healthy_scored: int = 0

    @property
    def total_days(self) -> float:
        return self.healthy_days + self.degradation_days

    def as_dict(self) -> Dict[str, object]:
        def r(xs, n=4):
            return [round(float(x), n) for x in xs]

        return {
            "life_id": self.life_id,
            "preset": self.preset,
            "mode": self.mode,
            "duty": round(self.duty, 4),
            "healthy_days": round(self.healthy_days, 2),
            "degradation_days": round(self.degradation_days, 2),
            "t_days": r(self.t_days, 3),
            "index": r(self.index, 4),
            "indicator": r(self.indicator, 4),
            "alarm_state": list(self.alarm_state),
            "true_rul_days": r(self.true_rul_days, 2),
            "commissioning_alarms": self.commissioning_alarms,
            "healthy_scores_above_threshold": self.healthy_scores_above_threshold,
            "n_healthy_scored": self.n_healthy_scored,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Life":
        life = cls(
            life_id=str(d["life_id"]), preset=str(d["preset"]), mode=str(d["mode"]),
            duty=float(d["duty"]), healthy_days=float(d["healthy_days"]),
            degradation_days=float(d["degradation_days"]),
        )
        life.t_days = [float(x) for x in d["t_days"]]
        life.index = [float(x) for x in d["index"]]
        life.indicator = [float(x) for x in d["indicator"]]
        life.alarm_state = [str(x) for x in d["alarm_state"]]
        life.true_rul_days = [float(x) for x in d["true_rul_days"]]
        life.commissioning_alarms = int(d["commissioning_alarms"])
        life.healthy_scores_above_threshold = int(d["healthy_scores_above_threshold"])
        life.n_healthy_scored = int(d["n_healthy_scored"])
        return life


# ---------------------------------------------------------------------------
# detector fitting, per preset
# ---------------------------------------------------------------------------
def fit_detector(preset: str, *, n: int = 180, seed: int = 0) -> Dict[str, object]:
    """Fit and calibrate a one-class detector on healthy frames of one preset.

    The split is the point. ``threshold_at_fpr`` on the scores the detector was
    fitted to reads 15.4x too low -- that is one of the six "checks that cannot
    fail" in ``docs/08_branch_catalogue.md`` -- so the threshold is calibrated on
    a held-out third and the two numbers are both reported, so the gap stays
    visible in the committed output rather than being a remembered fact.
    """
    spec = MACHINE_LIBRARY[preset]
    fx = _extractor(spec)
    X = np.vstack([
        fx.extract(simulate(spec, None, duration_s=0.32, seed=seed * 10_000 + i).accel_g,
                   shaft_hz=spec.shaft_hz).vector
        for i in range(n)
    ])
    cut = int(n * 2 / 3)
    det = MahalanobisDetector(feature_names=fx.names).fit(X[:cut])
    thr_heldout = det.calibrate(X[cut:], fpr=DETECTOR_FPR)
    thr_insample = det.threshold_at_fpr(DETECTOR_FPR)
    return {
        "preset": preset,
        "detector": det,
        "extractor": fx,
        "healthy_vectors": X,
        "threshold": float(thr_heldout),
        "threshold_heldout": float(thr_heldout),
        "threshold_in_sample": float(thr_insample),
        "calibration_ratio": float(thr_heldout / max(thr_insample, 1e-9)),
        "n_fit": cut,
        "n_calibration": n - cut,
    }


# ---------------------------------------------------------------------------
# one life through the whole chain
# ---------------------------------------------------------------------------
def run_life(
    life_id: str,
    preset: str,
    mode: str,
    *,
    duty: float,
    healthy_days: float,
    degradation_days: float,
    detector_bundle: Dict[str, object],
    seed: int,
) -> Life:
    spec = MACHINE_LIBRARY[preset]
    fx: FeatureExtractor = detector_bundle["extractor"]  # type: ignore[assignment]
    det: MahalanobisDetector = detector_bundle["detector"]  # type: ignore[assignment]
    threshold = float(detector_bundle["threshold"])
    idx_indicator = fx.names.index(RUL_INDICATOR)

    health = HealthIndex(life_id, fx.names, alarm_config=AlarmConfig(),
                         adapt_baseline=False)
    life = Life(life_id=life_id, preset=preset, mode=mode, duty=duty,
                healthy_days=healthy_days, degradation_days=degradation_days)

    # Commissioning: real healthy acquisitions, scored, alarm machine live, so
    # nuisance alarms during commissioning are counted rather than assumed away.
    rng = np.random.default_rng(seed)
    comm: List[np.ndarray] = []
    for i in range(N_COMMISSION):
        res = simulate(spec, None, duration_s=0.32, seed=seed * 1000 + i)
        v = fx.extract(res.accel_g, shaft_hz=spec.shaft_hz).vector
        comm.append(v)
        score = float(det.score(v.reshape(1, -1))[0])
        life.n_healthy_scored += 1
        if score > threshold:
            life.healthy_scores_above_threshold += 1
    comm_X = np.vstack(comm)
    hf_baseline = float(np.median(comm_X[:, idx_indicator]))

    # Alarm machine over the commissioning window, to count nuisance alarms the
    # way an operator would see them (debounced, hysteretic, dwell-limited).
    for i in range(N_COMMISSION):
        score = float(det.score(comm_X[i].reshape(1, -1))[0])
        out = health.update(float(i) * 86400.0, comm_X[i], speed_hz=spec.shaft_hz,
                            load_frac=duty, external_score=score,
                            external_reference=threshold)
        ev = out["event"]
        if ev is not None and ev.to_state in (AlarmState.ALARM, AlarmState.CRITICAL):
            life.commissioning_alarms += 1

    # Degradation stage: daily acquisitions from onset to failure.
    n_points = max(24, int(round(degradation_days)))
    series = degradation_series(
        spec, mode=mode, n_points=n_points, duration_s=0.32,
        days=float(degradation_days), knee_fraction=float(rng.uniform(0.25, 0.55)),
        final_severity=float(rng.uniform(0.80, 0.98)), seed=seed * 7 + 3,
    )
    t0 = N_COMMISSION * 86400.0
    for res in series:
        v = fx.extract(res.accel_g, shaft_hz=spec.shaft_hz).vector
        score = float(det.score(v.reshape(1, -1))[0])
        t_days = float(res.truth["t_days"])
        out = health.update(t0 + t_days * 86400.0, v, speed_hz=spec.shaft_hz,
                            load_frac=duty, external_score=score,
                            external_reference=threshold)
        life.t_days.append(t_days)
        life.index.append(float(out["smoothed"]))
        life.indicator.append(float(v[idx_indicator]) / max(hf_baseline, 1e-12))
        life.alarm_state.append(str(out["state"]))
        life.true_rul_days.append(float(res.truth["rul_days"]))
    return life


# ---------------------------------------------------------------------------
# the RUL replay, which the fleet simulation also uses
# ---------------------------------------------------------------------------
def replay_rul(
    life: Life, *, seed: int = 3, from_state: str = "watch"
) -> List[Dict[str, object]]:
    """Feed one life's indicator series into the scenario-conditional estimator.

    Observations are fed from the first acquisition at or above ``from_state``,
    which is what an operational system does: the prognostic filter is attached
    to an asset once it leaves the normal band, so its trend window is not
    dominated by a year of flat healthy history. ``pdmcore.health.rul`` names
    that effect in its own ``trend_window`` comment.

    Returns one record per day with the unconditional and scenario-conditional
    estimates and the true remaining life, so RUL accuracy can be scored
    against ground truth rather than against another model.
    """
    order = [s.value for s in AlarmState]
    start_rank = order.index(from_state)
    sr = ScenarioRUL(
        rul=ParticleFilterRUL(failure_threshold=RUL_FAILURE_THRESHOLD,
                              min_health=RUL_MIN_HEALTH, seed=seed),
    )
    at_duty = LoadScenario.constant(life.duty, name="as_operated")
    out: List[Dict[str, object]] = []
    attached = False
    for t, ind, state, true_rul in zip(life.t_days, life.indicator, life.alarm_state,
                                       life.true_rul_days):
        if not attached:
            try:
                rank = order.index(state)
            except ValueError:  # pragma: no cover - states come from the enum
                rank = 0
            if rank < start_rank:
                continue
            attached = True
        sr.update(float(t), float(ind), load_frac=life.duty)
        est = sr.estimate(at_duty)
        out.append({
            "t_days": float(t),
            "true_rul_days": float(true_rul),
            "alarm_state": state,
            "status": est.status,
            "median_days": est.median_days,
            "p10_days": est.p10_days,
            "p90_days": est.p90_days,
            "interval_width_days": est.interval_width_days,
        })
    return out


def build_scenario_rul(life: Life, *, seed: int = 3,
                       from_state: str = "watch") -> Tuple[ScenarioRUL, int]:
    """A :class:`ScenarioRUL` fed the whole life, plus the index it attached at."""
    order = [s.value for s in AlarmState]
    start_rank = order.index(from_state)
    sr = ScenarioRUL(
        rul=ParticleFilterRUL(failure_threshold=RUL_FAILURE_THRESHOLD,
                              min_health=RUL_MIN_HEALTH, seed=seed),
    )
    attach = -1
    for i, (t, ind, state) in enumerate(zip(life.t_days, life.indicator, life.alarm_state)):
        if attach < 0:
            if order.index(state) < start_rank:
                continue
            attach = i
        sr.update(float(t), float(ind), load_frac=life.duty)
    return sr, attach


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------
def build_pool(n_lives: int, *, seed: int = 0, verbose: bool = True) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    detectors = {p: fit_detector(p, seed=i + 1) for i, p in enumerate(PRESETS)}
    lives: List[Life] = []
    t_start = time.time()
    for k in range(n_lives):
        preset = PRESETS[k % len(PRESETS)]
        mode = MODES[(k // len(PRESETS)) % len(MODES)]
        # Healthy duration: the long flat stage, drawn rather than simulated.
        healthy = float(np.clip(rng.lognormal(math.log(420.0), 0.55), 60.0, 2200.0))
        # Onset-to-failure: weeks to months, which is the window the whole
        # decision problem lives in.
        degradation = float(np.clip(rng.lognormal(math.log(80.0), 0.45), 24.0, 260.0))
        duty = float(np.clip(rng.uniform(0.70, 1.0), 0.5, 1.0))
        life = run_life(
            f"life_{k:04d}", preset, mode, duty=duty, healthy_days=healthy,
            degradation_days=degradation, detector_bundle=detectors[preset],
            seed=1000 + k,
        )
        lives.append(life)
        if verbose and (k + 1) % 5 == 0:
            el = time.time() - t_start
            print(f"  {k + 1}/{n_lives} lives, {el:.0f} s elapsed, "
                  f"{el / (k + 1) * (n_lives - k - 1):.0f} s remaining", flush=True)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_lives": len(lives),
        "seed": seed,
        "presets": list(PRESETS),
        "modes": list(MODES),
        "acquisition_cadence_days": 1.0,
        "n_commission_frames": N_COMMISSION,
        "detector_fpr_target": DETECTOR_FPR,
        "rul_indicator": RUL_INDICATOR,
        "rul_failure_threshold": RUL_FAILURE_THRESHOLD,
        "rul_min_health": RUL_MIN_HEALTH,
        "detectors": {
            p: {k: v for k, v in b.items()
                if k not in ("detector", "extractor", "healthy_vectors")}
            for p, b in detectors.items()
        },
        "caveats": [
            "The healthy stage of each life beyond the commissioning window is not "
            "simulated acquisition-by-acquisition: pdmcore.sim's healthy output is "
            "stationary, so it would cost DSP runs and add no information. The healthy "
            "duration is drawn as a number.",
            "pdmcore.sim has no load-dependent damage model. The declared duty per life "
            "gives the scenario-conditional estimator a reference load, and the "
            "conditional arithmetic is exercised, but the ISO 281 load-life exponent is "
            "NOT validated by anything here.",
            "The detector is fitted and calibrated on simulated healthy data from the "
            "same generator that produced the faults, so some of the detection "
            "performance is self-fulfilling -- the platform-wide caveat in "
            "docs/07_production_readiness.md applies unchanged.",
        ],
        "lives": [life.as_dict() for life in lives],
        "wall_clock_s": round(time.time() - t_start, 1),
    }


def load_pool(path: Optional[str] = None) -> Tuple[Dict[str, object], List[Life]]:
    path = path or os.path.join(RESULTS, "life_pool.json")
    with open(path) as fh:
        blob = json.load(fh)
    return blob, [Life.from_dict(d) for d in blob["lives"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lives", type=int, default=96,
                    help="number of run-to-failure lives to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"generating {args.lives} lives through the real pipeline "
          f"(~{args.lives * 145 * 0.027:.0f} s of DSP)", flush=True)
    pool = build_pool(args.lives, seed=args.seed)
    path = os.path.join(args.out, "life_pool.json")
    with open(path, "w") as fh:
        json.dump(pool, fh, indent=1)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB, "
          f"{pool['wall_clock_s']:.0f} s)")
    for p, d in pool["detectors"].items():
        print(f"  {p:24s} threshold held-out {d['threshold_heldout']:8.2f}  "
              f"in-sample {d['threshold_in_sample']:8.2f}  "
              f"ratio {d['calibration_ratio']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
