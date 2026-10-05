from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class FaultType(str, Enum):
    UNBALANCE = "mechanical_unbalance"
    MISALIGNMENT = "misalignment"
    BEARING_WEAR = "bearing_wear"
    GEARBOX = "gearbox_fault"
    LUBRICATION = "lubrication_failure"
    MOTOR_ELECTRICAL = "motor_electrical_fault"
    COOLING_ANOMALY = "cooling_anomaly"
    NORMAL = "normal"


class Severity(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class SensorReading:
    asset_id: str
    timestamp: float
    vibration_x: np.ndarray
    vibration_y: np.ndarray
    vibration_z: np.ndarray
    temperature: float
    rpm: float
    sample_rate: int = 12800


@dataclass
class FaultDiagnosis:
    asset_id: str
    timestamp: float
    fault_type: FaultType
    severity: Severity
    confidence: float
    features: dict = field(default_factory=dict)
    explanation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "timestamp": self.timestamp,
            "fault_type": self.fault_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "features": self.features,
            "explanation": self.explanation,
        }
