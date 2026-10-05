from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.common.data_generator import SensorDataGenerator
from src.common.models import FaultType, SensorReading
from src.llm_augmented.diagnosis_agent import DiagnosisAgent
from src.llm_augmented.fault_classifier import QuickFaultClassifier
from src.llm_augmented.signal_processor import SignalProcessor
from src.llm_augmented.work_order import WorkOrder, WorkOrderPriority


def _make_reading(fault: FaultType = FaultType.BEARING_WEAR) -> SensorReading:
    gen = SensorDataGenerator(rng=np.random.default_rng(42))
    return gen.generate("TEST-001", fault)


# --- SignalProcessor ---

def test_signal_processor_describe() -> None:
    processor = SignalProcessor()
    reading = _make_reading(FaultType.NORMAL)
    desc = processor.describe(reading)

    required_keys = {"rms_g", "peak_g", "dominant_freq_hz", "temperature_c", "rpm",
                     "harmonic_powers", "spectral_entropy", "kurtosis"}
    assert required_keys == set(desc.keys())
    assert desc["rms_g"] > 0
    assert desc["peak_g"] >= desc["rms_g"]
    assert set(desc["harmonic_powers"].keys()) == {"1x", "2x", "3x", "4x"}
    assert desc["temperature_c"] > 0
    assert desc["rpm"] > 0


def test_signal_processor_extract_features_shape() -> None:
    processor = SignalProcessor()
    reading = _make_reading()
    features = processor.extract_features(reading)
    assert features.shape == (11,)
    assert features.dtype == np.float32


# --- QuickFaultClassifier ---

def test_quick_classifier() -> None:
    clf = QuickFaultClassifier(n_estimators=10, n_train_per_class=20)
    gen = SensorDataGenerator(rng=np.random.default_rng(1))
    readings, labels = gen.generate_dataset(n_per_class=20)
    clf.fit(readings, labels)

    reading = _make_reading(FaultType.BEARING_WEAR)
    fault_type, confidence = clf.predict_reading(reading)

    assert isinstance(fault_type, FaultType)
    assert 0.0 <= confidence <= 1.0


def test_quick_classifier_auto_trains() -> None:
    clf = QuickFaultClassifier(n_estimators=5, n_train_per_class=10)
    assert not clf._ready
    reading = _make_reading()
    fault_type, confidence = clf.predict_reading(reading)
    assert clf._ready
    assert isinstance(fault_type, FaultType)


# --- WorkOrder model ---

def test_work_order_model() -> None:
    wo = WorkOrder(
        asset_id="PUMP-001",
        priority=WorkOrderPriority.URGENT,
        fault_type="bearing_wear",
        severity="alert",
        title="Bearing wear on PUMP-001",
        description="Elevated kurtosis indicates bearing defect.",
        root_cause="Insufficient lubrication leading to surface wear.",
        recommended_actions=["Inspect bearing", "Replace if worn"],
        estimated_labor_hours=3.0,
        parts_required=["SKF bearing 6205"],
        safety_precautions=["Apply LOTO", "Wear PPE"],
        signal_evidence={"rms_g": 0.35},
    )
    assert wo.asset_id == "PUMP-001"
    assert wo.priority == WorkOrderPriority.URGENT
    assert len(wo.work_order_id) == 36  # UUID format
    assert isinstance(wo.created_at, datetime)


# --- DiagnosisAgent ---

def test_diagnosis_agent_prompt() -> None:
    agent = DiagnosisAgent(api_key=None)
    reading = _make_reading(FaultType.BEARING_WEAR)
    desc = SignalProcessor().describe(reading)
    prompt = agent._build_prompt(
        reading.asset_id, FaultType.BEARING_WEAR, 0.85,
        __import__("src.common.models", fromlist=["Severity"]).Severity.ALERT,
        desc, None,
    )
    assert "bearing_wear" in prompt
    assert "TEST-001" in prompt
    assert "JSON" in prompt


def test_diagnosis_agent_prompt_with_context() -> None:
    agent = DiagnosisAgent(api_key=None)
    reading = _make_reading(FaultType.UNBALANCE)
    desc = SignalProcessor().describe(reading)
    context = {"last_maintenance": "2026-08-01", "operating_hours": 4500}
    from src.common.models import Severity
    prompt = agent._build_prompt(
        reading.asset_id, FaultType.UNBALANCE, 0.72, Severity.WARNING, desc, context
    )
    assert "last_maintenance" in prompt
    assert "4500" in prompt


def test_diagnosis_agent_mock_llm() -> None:
    fake_response = {
        "explanation": "Bearing shows early wear.",
        "root_cause": "Lack of lubrication.",
        "recommended_actions": ["Check lubrication", "Replace bearing"],
        "parts_required": ["Bearing 6205"],
        "safety_precautions": ["LOTO required"],
        "estimated_labor_hours": 2.5,
        "priority": "urgent",
    }

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=json.dumps(fake_response))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    agent = DiagnosisAgent(api_key=None)
    agent._client = mock_client

    reading = _make_reading(FaultType.BEARING_WEAR)
    diagnosis, work_order = asyncio.run(agent.diagnose(reading))

    assert isinstance(diagnosis.fault_type, FaultType)
    assert work_order.priority == WorkOrderPriority.URGENT
    assert "Bearing" in work_order.description
    assert diagnosis.explanation == "Bearing shows early wear."


def test_diagnosis_agent_fallback_on_llm_error() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network error")

    agent = DiagnosisAgent(api_key=None)
    agent._client = mock_client

    reading = _make_reading(FaultType.NORMAL)
    diagnosis, work_order = asyncio.run(agent.diagnose(reading))

    assert isinstance(diagnosis.fault_type, FaultType)
    assert "LLM unavailable" in work_order.root_cause


# --- FastAPI ---

def test_api_health() -> None:
    from src.llm_augmented.api import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "llm_available" in data
    assert "classifier_ready" in data


def test_api_diagnose_no_llm() -> None:
    gen = SensorDataGenerator(rng=np.random.default_rng(7))
    reading = gen.generate("MOTOR-007", FaultType.MISALIGNMENT)

    payload = {
        "asset_id": reading.asset_id,
        "timestamp": reading.timestamp,
        "vibration_x": reading.vibration_x.tolist(),
        "vibration_y": reading.vibration_y.tolist(),
        "vibration_z": reading.vibration_z.tolist(),
        "temperature": reading.temperature,
        "rpm": reading.rpm,
        "sample_rate": reading.sample_rate,
    }

    # Patch the agent to have no LLM client
    from src.llm_augmented import api as api_module
    original_client = api_module._agent._client
    api_module._agent._client = None

    try:
        from src.llm_augmented.api import app
        client = TestClient(app)
        resp = client.post("/diagnose", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "diagnosis" in data
        assert "work_order" in data
        assert data["work_order"]["asset_id"] == "MOTOR-007"
    finally:
        api_module._agent._client = original_client
