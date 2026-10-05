from __future__ import annotations

import time
from typing import Optional

from ..common.models import FaultType, Severity
from .window import WindowFeatures

# Temperature deltas above baseline that separate fault classes
_TEMP_BEARING_THRESH = 7.0   # bearing or lubrication if temp delta >= this
_TEMP_LUBRICATION_THRESH = 11.0  # lubrication (higher temp) vs bearing
_TEMP_COOLING_THRESH = 15.0  # pure thermal fault


class OnlineAnomalyDetector:
    """
    Per-asset EWMA baseline with adaptive k-sigma anomaly detection.
    No labeled data required — learns normal behaviour during warmup.
    """

    def __init__(self, asset_id: str, warmup_samples: int = 50, alpha: float = 0.05) -> None:
        self.asset_id = asset_id
        self.warmup_samples = warmup_samples
        self.alpha = alpha  # EWMA smoothing factor

        self._n = 0
        self._rms_mean: float = 0.0
        self._rms_var: float = 0.0
        self._temp_mean: float = 0.0
        self._temp_var: float = 0.0

    @property
    def is_warmed_up(self) -> bool:
        return self._n >= self.warmup_samples

    @property
    def _k(self) -> float:
        # Threshold tightens from 4 to 3 sigma after warmup
        return 3.0 if self.is_warmed_up else 4.0

    def _update_ewma(self, mean: float, var: float, value: float) -> tuple[float, float]:
        diff = value - mean
        new_mean = mean + self.alpha * diff
        new_var = (1.0 - self.alpha) * (var + self.alpha * diff ** 2)
        return new_mean, new_var

    def _anomaly_score(self, value: float, mean: float, var: float) -> float:
        std = var ** 0.5
        if std == 0:
            return 0.0
        z = abs(value - mean) / std
        # Sigmoid-like normalisation: score=0 at z=k, score→1 as z grows
        excess = max(0.0, z - self._k)
        return float(1.0 - 1.0 / (1.0 + excess))

    def update(self, features: WindowFeatures) -> tuple[bool, float]:
        """Returns (is_anomaly, anomaly_score ∈ [0,1])."""
        if self._n == 0:
            self._rms_mean = features.rms
            self._temp_mean = features.temp_mean
            self._rms_var = 1e-6
            self._temp_var = 1e-6
        else:
            self._rms_mean, self._rms_var = self._update_ewma(
                self._rms_mean, self._rms_var, features.rms
            )
            self._temp_mean, self._temp_var = self._update_ewma(
                self._temp_mean, self._temp_var, features.temp_mean
            )

        self._n += 1

        rms_score = self._anomaly_score(features.rms, self._rms_mean, self._rms_var)
        temp_score = self._anomaly_score(features.temp_mean, self._temp_mean, self._temp_var)
        rpm_score = min(1.0, features.rpm_cv * 10.0)  # RPM CV > 0.1 → anomalous

        # Weighted combination; vibration is most reliable signal
        combined = 0.6 * rms_score + 0.25 * temp_score + 0.15 * rpm_score

        # During warmup suppress mild anomalies to build a reliable baseline
        if not self.is_warmed_up:
            combined *= 0.3

        is_anomaly = combined > 0.5
        return is_anomaly, float(min(combined, 1.0))


class FaultTypeEstimator:
    """
    Deterministic rule engine mapping window features to fault type + severity.
    Rules derived from domain knowledge (TRACTIAN fault signatures).
    """

    def estimate(
        self,
        features: WindowFeatures,
        anomaly_score: float,
        baseline_temp: Optional[float] = None,
    ) -> tuple[FaultType, Severity]:
        if anomaly_score < 0.15:
            return FaultType.NORMAL, Severity.NORMAL

        severity = self._severity(anomaly_score)
        temp_delta = features.temp_mean - (baseline_temp or 65.0)

        # Pure thermal fault
        if temp_delta >= _TEMP_COOLING_THRESH and features.rms < 0.3:
            return FaultType.COOLING_ANOMALY, severity

        # RPM instability + vibration → gearbox
        if features.rpm_cv > 0.05 and features.rms > 0.2:
            return FaultType.GEARBOX, severity

        # High vibration + high temperature → bearing or lubrication
        if features.rms > 0.2 and temp_delta >= _TEMP_BEARING_THRESH:
            if temp_delta >= _TEMP_LUBRICATION_THRESH:
                return FaultType.LUBRICATION, severity
            return FaultType.BEARING_WEAR, severity

        # High crest factor → impulsive bearing fault
        if features.peak_factor > 4.5 and features.rms > 0.1:
            return FaultType.BEARING_WEAR, severity

        # Multiple harmonics evident (spike count proxy)
        if features.recent_spike_count >= 2:
            return FaultType.MISALIGNMENT, severity

        # Clean high vibration, stable RPM → unbalance
        if features.rms > 0.15 and features.rpm_cv < 0.02:
            return FaultType.UNBALANCE, severity

        # Motor electrical — high vibration with moderate temp
        if features.rms > 0.2:
            return FaultType.MOTOR_ELECTRICAL, severity

        return FaultType.NORMAL, Severity.NORMAL

    @staticmethod
    def _severity(score: float) -> Severity:
        if score < 0.35:
            return Severity.WARNING
        if score < 0.65:
            return Severity.ALERT
        return Severity.CRITICAL
