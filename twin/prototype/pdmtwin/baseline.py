"""The thing the twin has to beat, wrapped in the twin's own harness.

:class:`pdmcore.models.anomaly.MahalanobisDetector` on the 63 calibrated
features, fitted on a commissioning window and calibrated on held-out healthy
data, is the platform's shipped detector. It is cheap (17 us per frame, measured
on this host by ``hardware/rpi5_hailo/tools/benchmark.py``), explainable through
``explain()``, and it scores 0.95-0.99 grouped AUC on simulated data.

Three things about this wrapper are deliberate and all three make the baseline
*stronger*, because a comparison against a weak baseline is worthless:

* It gets ``calibrate()`` on a held-out window, never ``threshold_at_fpr`` on
  training scores. ``anomaly.py``'s own docstring records that the latter was
  15.4x too low.
* It gets the **same sequential machinery** as the twin. If a CUSUM on the
  residual beats a per-frame threshold on the Mahalanobis score, that is a
  result about sequential testing, not about twins -- so the baseline is run
  both ways and both numbers are reported.
* It gets the same ARL0 target, measured the same way, on the same healthy
  trajectories.

The one asymmetry left is the information set: the baseline sees all 63
features, the twin sees the 9 channels a physics model can predict. That
asymmetry favours the baseline, so a restricted ``feature_subset`` arm is run as
well -- otherwise "the twin lost" and "the twin was shown less" are the same
number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from pdmcore.models.anomaly import AnomalyDetector, build_detector

from .data import Dataset, ObsSpec
from .detect import (
    CusumTest,
    ThresholdTest,
    measure_arl0,
    poisson_allowance,
    robust_centre_scale,
    threshold_for_arl0,
)

__all__ = ["BaselineArm"]


@dataclass
class BaselineArm:
    """One-class detector plus the shared sequential harness.

    ``sequential=False`` is the shipped behaviour: a per-frame threshold with an
    N-consecutive confirmation. ``sequential=True`` puts the identical CUSUM the
    twin uses on top of the detector's own score.
    """

    kind: str = "mahalanobis"
    label: str = "baseline63"
    feature_subset: Optional[ObsSpec] = None
    sequential: bool = False
    cusum_drift: float = 0.5
    consecutive_required: int = 2
    detector_kwargs: Optional[Dict] = None

    def __post_init__(self) -> None:
        self.detector: Optional[AnomalyDetector] = None
        self.q_centre = 0.0
        self.q_scale = 1.0
        self.test = None
        self.reports: Dict[str, Dict] = {}

    # -- helpers -----------------------------------------------------------
    def _X(self, ds: Dataset) -> np.ndarray:
        if self.feature_subset is None:
            return ds.X
        return ds.obs(self.feature_subset)

    def _names(self, ds: Dataset) -> Sequence[str]:
        if self.feature_subset is None:
            return ds.feature_names
        return self.feature_subset.names

    def q(self, ds: Dataset) -> np.ndarray:
        assert self.detector is not None
        s = self.detector.score(self._X(ds))
        return (s - self.q_centre) / self.q_scale

    def _by_trajectory(self, ds: Dataset) -> List[np.ndarray]:
        out = []
        for t in np.unique(ds.traj):
            idx = np.flatnonzero(ds.traj == t)
            idx = idx[np.argsort(ds.t_days[idx], kind="mergesort")]
            out.append(self.q(ds.take(idx)))
        return out

    # -- fit / calibrate / verify ------------------------------------------
    def fit(self, ds_fit: Dataset, ds_cal: Dataset, *, target_arl0_frames: float = 720.0) -> "BaselineArm":
        self.detector = build_detector(self.kind, self._names(ds_fit), **(self.detector_kwargs or {}))
        self.detector.fit(self._X(ds_fit))
        s_cal = self.detector.score(self._X(ds_cal))
        self.q_centre, self.q_scale = robust_centre_scale(s_cal)
        self.q_cal = self._by_trajectory(ds_cal)
        self._cal_meta = {
            "q_centre": self.q_centre, "q_scale": self.q_scale,
            "n_fit_frames": int(ds_fit.n), "n_cal_frames": int(ds_cal.n),
            "n_features": int(self._X(ds_fit).shape[1]),
            "sequential": self.sequential,
        }
        self.set_target(target_arl0_frames)
        extra: Dict[str, float] = {}
        for attr in ("shrinkage_used_", "samples_per_feature_", "n_components_"):
            if hasattr(self.detector, attr):
                extra[attr.rstrip("_")] = float(getattr(self.detector, attr))
        self.reports["detector"] = {"kind": self.kind, **extra}
        return self

    def set_target(self, target_arl0_frames: float) -> "BaselineArm":
        """Threshold for a target ARL0 on the cached calibration stream.

        Same separation as :meth:`pdmtwin.twin.HybridTwin.set_target`, and for
        the same reason: the detector score is a property of the window, the
        threshold is the operating point.
        """
        import dataclasses
        base = (CusumTest(drift=self.cusum_drift, consecutive_required=self.consecutive_required)
                if self.sequential else
                ThresholdTest(consecutive_required=self.consecutive_required))
        h, rep = threshold_for_arl0(self.q_cal, base, target_arl0_frames)
        self.test = dataclasses.replace(base, threshold=float(h))
        self.reports["calibration"] = {**rep, **self._cal_meta}
        return self

    def verify(self, ds_ver: Optional[Dataset] = None,
               q_ver: Optional[Sequence[np.ndarray]] = None) -> Dict[str, float]:
        if q_ver is None:
            if ds_ver is None:
                raise ValueError("verify needs ds_ver or q_ver")
            q_ver = self._by_trajectory(ds_ver)
        meas = measure_arl0(q_ver, self.test)
        expected = meas["n_frames"] / max(1e-9, self.reports["calibration"]["arl0_frames"])
        allowed = poisson_allowance(expected)
        out = {
            **meas, "expected_alarms": float(expected),
            "allowed_alarms_3sigma_poisson": float(allowed),
            "passed": bool(meas["n_alarms"] <= allowed),
        }
        self.reports["verification"] = out
        return out
