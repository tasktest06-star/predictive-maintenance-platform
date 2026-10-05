from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, Callable, Optional

import numpy as np

from ..common.data_generator import SensorDataGenerator
from ..common.models import FaultDiagnosis, FaultType, SensorReading
from .alert_manager import Alert, AlertManager
from .online_detector import FaultTypeEstimator, OnlineAnomalyDetector
from .window import SlidingWindow, WindowFeatures


class _AssetState:
    def __init__(self, asset_id: str) -> None:
        self.window = SlidingWindow(asset_id, window_size=10, step_size=1)
        self.detector = OnlineAnomalyDetector(asset_id)
        self.estimator = FaultTypeEstimator()
        self.last_features: Optional[WindowFeatures] = None
        self.readings_processed = 0


class StreamingPipeline:
    def __init__(self, alert_callback: Optional[Callable[[Alert], None]] = None) -> None:
        self._alert_callback = alert_callback
        self._assets: dict[str, _AssetState] = {}
        self._alert_manager = AlertManager()
        self._total_processed = 0
        self._total_diagnoses = 0
        self._start_time: float = time.time()

    def _get_or_create(self, asset_id: str) -> _AssetState:
        if asset_id not in self._assets:
            self._assets[asset_id] = _AssetState(asset_id)
        return self._assets[asset_id]

    async def process(self, reading: SensorReading) -> Optional[FaultDiagnosis]:
        state = self._get_or_create(reading.asset_id)
        state.readings_processed += 1
        self._total_processed += 1

        triggered = state.window.push(reading)
        if not triggered:
            return None

        features = state.window.extract_features()
        if features is None:
            return None

        state.last_features = features
        is_anomaly, score = state.detector.update(features)

        fault_type, severity = state.estimator.estimate(
            features, score if is_anomaly else 0.0
        )

        diagnosis = FaultDiagnosis(
            asset_id=reading.asset_id,
            timestamp=reading.timestamp,
            fault_type=fault_type,
            severity=severity,
            confidence=score,
            features={
                "rms": features.rms,
                "temp_mean": features.temp_mean,
                "temp_trend": features.temp_trend,
                "rpm_cv": features.rpm_cv,
                "peak_factor": features.peak_factor,
                "recent_spike_count": features.recent_spike_count,
            },
        )
        self._total_diagnoses += 1

        alert = self._alert_manager.process(diagnosis)
        if alert is not None and self._alert_callback is not None:
            self._alert_callback(alert)

        return diagnosis

    async def run_simulation(
        self,
        n_assets: int = 3,
        fault_scenario: Optional[dict[str, FaultType]] = None,
        duration_seconds: float = 30.0,
    ) -> AsyncGenerator[FaultDiagnosis, None]:
        gen = SensorDataGenerator(rng=np.random.default_rng(42))
        asset_ids = [f"ASSET-{i:03d}" for i in range(n_assets)]
        fault_scenario = fault_scenario or {}

        end_time = time.time() + duration_seconds
        reading_count = 0

        while time.time() < end_time:
            for asset_id in asset_ids:
                fault = fault_scenario.get(asset_id, FaultType.NORMAL)
                reading = gen.generate(asset_id=asset_id, fault_type=fault)
                diagnosis = await self.process(reading)
                if diagnosis is not None:
                    yield diagnosis
            reading_count += 1
            await asyncio.sleep(0)  # yield control without real delay in simulation

    def stats(self) -> dict:
        elapsed = max(time.time() - self._start_time, 1e-9)
        return {
            "active_assets": len(self._assets),
            "total_readings_processed": self._total_processed,
            "total_diagnoses_emitted": self._total_diagnoses,
            "throughput_readings_per_sec": self._total_processed / elapsed,
            "active_alerts": len(self._alert_manager.active_alerts()),
            "uptime_seconds": elapsed,
        }
