"""Tests for the streaming pipeline components."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

# Patch sys.path so imports resolve without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.data_generator import SensorDataGenerator
from src.common.models import FaultDiagnosis, FaultType, Severity
from src.streaming.alert_manager import Alert, AlertManager
from src.streaming.online_detector import FaultTypeEstimator, OnlineAnomalyDetector
from src.streaming.pipeline import StreamingPipeline
from src.streaming.window import SlidingWindow, WindowFeatures


GEN = SensorDataGenerator(rng=np.random.default_rng(0))


def _make_features(**kwargs) -> WindowFeatures:
    defaults = dict(
        asset_id="TEST",
        rms=0.1,
        temp_mean=65.0,
        temp_trend=0.0,
        rpm_cv=0.01,
        peak_factor=2.5,
        recent_spike_count=0,
    )
    defaults.update(kwargs)
    return WindowFeatures(**defaults)


# --- SlidingWindow ---

def test_sliding_window_triggers():
    win = SlidingWindow("A", window_size=5, step_size=1)
    reading = GEN.generate("A", FaultType.NORMAL)
    results = [win.push(reading) for _ in range(5)]
    # First 4 should not trigger (window not full), 5th should
    assert results[:4] == [False, False, False, False]
    assert results[4] is True


def test_sliding_window_step_size():
    win = SlidingWindow("A", window_size=3, step_size=2)
    reading = GEN.generate("A", FaultType.NORMAL)
    results = [win.push(reading) for _ in range(5)]
    # Trigger at index 2 (full) and not again until step_size readings later
    assert results[2] is True
    assert results[3] is False
    assert results[4] is True


def test_sliding_window_get_window():
    win = SlidingWindow("A", window_size=3, step_size=1)
    readings = [GEN.generate("A", FaultType.NORMAL) for _ in range(3)]
    for r in readings:
        win.push(r)
    assert len(win.get_window()) == 3


# --- OnlineAnomalyDetector ---

def test_online_detector_warmup():
    detector = OnlineAnomalyDetector("A", warmup_samples=50)
    normal_feat = _make_features(rms=0.1, temp_mean=65.0)
    # Feed warmup_samples - 1 readings; should not flag anomaly despite slight variance
    for _ in range(49):
        is_anom, score = detector.update(normal_feat)
    assert not detector.is_warmed_up
    # Score is suppressed during warmup
    is_anom, score = detector.update(normal_feat)
    assert score < 0.5, "Normal signal during warmup should not be anomalous"


def test_online_detector_anomaly():
    detector = OnlineAnomalyDetector("A", warmup_samples=20)
    normal_feat = _make_features(rms=0.1, temp_mean=65.0)
    # Build baseline
    for _ in range(30):
        detector.update(normal_feat)
    assert detector.is_warmed_up
    # Inject obvious fault signal
    fault_feat = _make_features(rms=1.5, temp_mean=80.0, rpm_cv=0.1)
    is_anom, score = detector.update(fault_feat)
    assert score > 0.5, f"Expected anomaly score > 0.5, got {score}"
    assert is_anom


# --- FaultTypeEstimator ---

def test_estimator_normal():
    est = FaultTypeEstimator()
    ft, sev = est.estimate(_make_features(), anomaly_score=0.05)
    assert ft == FaultType.NORMAL
    assert sev == Severity.NORMAL


def test_estimator_unbalance():
    est = FaultTypeEstimator()
    ft, sev = est.estimate(_make_features(rms=0.4, rpm_cv=0.005), anomaly_score=0.6)
    assert ft == FaultType.UNBALANCE


def test_estimator_cooling():
    est = FaultTypeEstimator()
    feat = _make_features(rms=0.05, temp_mean=65.0 + 16.0)
    ft, sev = est.estimate(feat, anomaly_score=0.7, baseline_temp=65.0)
    assert ft == FaultType.COOLING_ANOMALY


# --- AlertManager ---

def test_alert_deduplication():
    mgr = AlertManager(dedup_window_seconds=300.0)
    diag = FaultDiagnosis(
        asset_id="A", timestamp=time.time(), fault_type=FaultType.UNBALANCE,
        severity=Severity.WARNING, confidence=0.6
    )
    first = mgr.process(diag)
    assert first is not None
    second = mgr.process(diag)
    assert second is None, "Duplicate within dedup window should not emit new alert"


def test_alert_escalation():
    mgr = AlertManager(dedup_window_seconds=300.0)
    diag_warn = FaultDiagnosis(
        asset_id="A", timestamp=time.time(), fault_type=FaultType.UNBALANCE,
        severity=Severity.WARNING, confidence=0.4
    )
    diag_crit = FaultDiagnosis(
        asset_id="A", timestamp=time.time(), fault_type=FaultType.UNBALANCE,
        severity=Severity.CRITICAL, confidence=0.9
    )
    assert mgr.process(diag_warn) is not None
    escalated = mgr.process(diag_crit)
    assert escalated is not None, "Severity escalation must emit alert"
    assert escalated.severity == Severity.CRITICAL


def test_alert_clear_resolved():
    mgr = AlertManager()
    diag = FaultDiagnosis(
        asset_id="A", timestamp=time.time(), fault_type=FaultType.BEARING_WEAR,
        severity=Severity.ALERT, confidence=0.7
    )
    mgr.process(diag)
    assert len(mgr.active_alerts()) == 1
    mgr.clear_resolved([FaultType.BEARING_WEAR])
    assert len(mgr.active_alerts()) == 0


# --- StreamingPipeline ---

@pytest.mark.asyncio
async def test_pipeline_process():
    pipeline = StreamingPipeline()
    reading = GEN.generate("P001", FaultType.NORMAL)
    # First few readings should return None (window not yet full)
    result = await pipeline.process(reading)
    assert result is None or result.asset_id == "P001"


@pytest.mark.asyncio
async def test_pipeline_fills_window():
    pipeline = StreamingPipeline()
    results = []
    for _ in range(12):
        r = GEN.generate("P002", FaultType.NORMAL)
        d = await pipeline.process(r)
        if d is not None:
            results.append(d)
    assert len(results) > 0, "Pipeline should emit diagnoses once window is filled"
    assert all(d.asset_id == "P002" for d in results)


@pytest.mark.asyncio
async def test_pipeline_stats():
    pipeline = StreamingPipeline()
    for _ in range(5):
        await pipeline.process(GEN.generate("X", FaultType.NORMAL))
    s = pipeline.stats()
    assert s["total_readings_processed"] == 5
    assert "active_assets" in s


# --- API ---

@pytest.mark.asyncio
async def test_api_ingest():
    import httpx
    from fastapi.testclient import TestClient
    from src.streaming.api import app

    with TestClient(app) as client:
        reading = GEN.generate("API-TEST", FaultType.NORMAL)
        payload = {
            "asset_id": "API-TEST",
            "vibration_x": reading.vibration_x.tolist(),
            "vibration_y": reading.vibration_y.tolist(),
            "vibration_z": reading.vibration_z.tolist(),
            "temperature": reading.temperature,
            "rpm": reading.rpm,
            "sample_rate": reading.sample_rate,
        }
        resp = client.post("/ingest", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_health():
    from fastapi.testclient import TestClient
    from src.streaming.api import app

    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
