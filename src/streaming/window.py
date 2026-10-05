from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..common.models import SensorReading


@dataclass
class WindowFeatures:
    asset_id: str
    rms: float
    temp_mean: float
    temp_trend: float        # degrees/reading (positive = rising)
    rpm_cv: float            # coefficient of variation (lower = more stable)
    peak_factor: float       # crest factor of vibration
    recent_spike_count: int  # readings exceeding 2x rms in window


class SlidingWindow:
    def __init__(self, asset_id: str, window_size: int = 10, step_size: int = 1) -> None:
        self.asset_id = asset_id
        self.window_size = window_size
        self.step_size = step_size
        self._buffer: deque[SensorReading] = deque(maxlen=window_size)
        self._steps_since_trigger = 0

    def push(self, reading: SensorReading) -> bool:
        self._buffer.append(reading)
        self._steps_since_trigger += 1
        if len(self._buffer) == self.window_size and self._steps_since_trigger >= self.step_size:
            self._steps_since_trigger = 0
            return True
        return False

    def get_window(self) -> list[SensorReading]:
        return list(self._buffer)

    def rms_vibration(self) -> float:
        if not self._buffer:
            return 0.0
        rms_vals = []
        for r in self._buffer:
            combined = np.concatenate([r.vibration_x, r.vibration_y, r.vibration_z])
            rms_vals.append(float(np.sqrt(np.mean(combined ** 2))))
        return float(np.mean(rms_vals))

    def mean_temperature(self) -> float:
        if not self._buffer:
            return 0.0
        return float(np.mean([r.temperature for r in self._buffer]))

    def rpm_stability(self) -> float:
        """Coefficient of variation of RPM; lower means more stable."""
        if len(self._buffer) < 2:
            return 0.0
        rpms = np.array([r.rpm for r in self._buffer], dtype=float)
        mean_rpm = float(np.mean(rpms))
        if mean_rpm == 0:
            return 0.0
        return float(np.std(rpms) / mean_rpm)

    def extract_features(self) -> Optional[WindowFeatures]:
        if len(self._buffer) < self.window_size:
            return None

        readings = list(self._buffer)
        rms = self.rms_vibration()

        temps = [r.temperature for r in readings]
        temp_mean = float(np.mean(temps))
        # linear trend via least squares slope
        x = np.arange(len(temps), dtype=float)
        temp_trend = float(np.polyfit(x, temps, 1)[0]) if len(temps) > 1 else 0.0

        rpm_cv = self.rpm_stability()

        # Peak factor (crest factor) from last reading
        last = readings[-1]
        combined = np.concatenate([last.vibration_x, last.vibration_y, last.vibration_z])
        rms_last = float(np.sqrt(np.mean(combined ** 2)))
        peak = float(np.max(np.abs(combined)))
        peak_factor = (peak / rms_last) if rms_last > 0 else 0.0

        # Count readings where per-reading RMS exceeds 2x window-mean RMS
        spike_count = 0
        for r in readings:
            c = np.concatenate([r.vibration_x, r.vibration_y, r.vibration_z])
            r_rms = float(np.sqrt(np.mean(c ** 2)))
            if r_rms > 2.0 * rms and rms > 0:
                spike_count += 1

        return WindowFeatures(
            asset_id=self.asset_id,
            rms=rms,
            temp_mean=temp_mean,
            temp_trend=temp_trend,
            rpm_cv=rpm_cv,
            peak_factor=peak_factor,
            recent_spike_count=spike_count,
        )
