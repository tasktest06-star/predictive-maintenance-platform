"""Synthetic sensor data generator for predictive maintenance."""
from __future__ import annotations

import time
from typing import Generator, Optional
import numpy as np

from .models import FaultType, SensorReading


# Fault signature parameters: (frequency multiplier of RPM, amplitude relative to baseline)
FAULT_SIGNATURES: dict[FaultType, list[tuple[float, float]]] = {
    FaultType.NORMAL: [],
    FaultType.UNBALANCE: [(1.0, 2.5)],
    FaultType.MISALIGNMENT: [(1.0, 1.8), (2.0, 1.2)],
    FaultType.BEARING_WEAR: [(6.7, 1.5), (13.4, 1.0), (3.5, 0.8)],
    FaultType.GEARBOX: [(12.0, 2.0), (24.0, 1.5)],
    FaultType.LUBRICATION: [(1.0, 0.5), (2.0, 0.5), (3.0, 0.5)],
    FaultType.MOTOR_ELECTRICAL: [(2.0, 3.0), (4.0, 1.5)],
    FaultType.COOLING_ANOMALY: [(1.0, 0.3)],
}

FAULT_TEMP_DELTA: dict[FaultType, float] = {
    FaultType.NORMAL: 0.0,
    FaultType.UNBALANCE: 2.0,
    FaultType.MISALIGNMENT: 3.5,
    FaultType.BEARING_WEAR: 8.0,
    FaultType.GEARBOX: 5.0,
    FaultType.LUBRICATION: 12.0,
    FaultType.MOTOR_ELECTRICAL: 6.0,
    FaultType.COOLING_ANOMALY: 18.0,
}


class SensorDataGenerator:
    def __init__(
        self,
        sample_rate: int = 12800,
        duration_seconds: float = 1.0,
        noise_level: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_samples = int(sample_rate * duration_seconds)
        self.noise_level = noise_level
        self.rng = rng or np.random.default_rng()

    def _time_axis(self) -> np.ndarray:
        return np.linspace(0, self.n_samples / self.sample_rate, self.n_samples)

    def _baseline_vibration(self, rpm: float) -> np.ndarray:
        t = self._time_axis()
        freq = rpm / 60.0
        signal = 0.1 * np.sin(2 * np.pi * freq * t)
        signal += self.noise_level * self.rng.standard_normal(self.n_samples)
        return signal.astype(np.float32)

    def _inject_fault(self, signal: np.ndarray, fault: FaultType, rpm: float) -> np.ndarray:
        t = self._time_axis()
        base_freq = rpm / 60.0
        for freq_mult, amplitude in FAULT_SIGNATURES[fault]:
            fault_freq = base_freq * freq_mult
            signal = signal + amplitude * 0.1 * np.sin(2 * np.pi * fault_freq * t)
        return signal.astype(np.float32)

    def generate(
        self,
        asset_id: str,
        fault_type: FaultType = FaultType.NORMAL,
        rpm: float = 1500.0,
        base_temp: float = 65.0,
        timestamp: Optional[float] = None,
    ) -> SensorReading:
        vib = self._baseline_vibration(rpm)
        vib = self._inject_fault(vib, fault_type, rpm)

        # Cross-axis coupling with slight asymmetry
        vib_x = vib + self.noise_level * 0.5 * self.rng.standard_normal(self.n_samples)
        vib_y = 0.7 * vib + self.noise_level * 0.5 * self.rng.standard_normal(self.n_samples)
        vib_z = 0.4 * vib + self.noise_level * 0.5 * self.rng.standard_normal(self.n_samples)

        temp = base_temp + FAULT_TEMP_DELTA[fault_type] + self.rng.normal(0, 0.5)

        return SensorReading(
            asset_id=asset_id,
            timestamp=timestamp or time.time(),
            vibration_x=vib_x.astype(np.float32),
            vibration_y=vib_y.astype(np.float32),
            vibration_z=vib_z.astype(np.float32),
            temperature=float(temp),
            rpm=rpm + float(self.rng.normal(0, 5)),
            sample_rate=self.sample_rate,
        )

    def generate_dataset(
        self,
        n_per_class: int = 200,
        asset_ids: Optional[list[str]] = None,
        rpm_range: tuple[float, float] = (900.0, 3600.0),
    ) -> tuple[list[SensorReading], list[FaultType]]:
        assets = asset_ids or ["PUMP-001", "MOTOR-002", "COMPRESSOR-003"]
        readings: list[SensorReading] = []
        labels: list[FaultType] = []

        for fault_type in FaultType:
            for i in range(n_per_class):
                asset = assets[i % len(assets)]
                rpm = float(self.rng.uniform(*rpm_range))
                reading = self.generate(asset, fault_type, rpm)
                readings.append(reading)
                labels.append(fault_type)

        return readings, labels

    def stream(
        self,
        asset_id: str,
        fault_type: FaultType = FaultType.NORMAL,
        rpm: float = 1500.0,
        interval_seconds: float = 1.0,
    ) -> Generator[SensorReading, None, None]:
        while True:
            yield self.generate(asset_id, fault_type, rpm, timestamp=time.time())
            time.sleep(interval_seconds)
