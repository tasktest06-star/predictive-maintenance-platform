from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..common.models import FaultDiagnosis, FaultType, Severity

_SEVERITY_RANK = {
    Severity.NORMAL: 0,
    Severity.WARNING: 1,
    Severity.ALERT: 2,
    Severity.CRITICAL: 3,
}


@dataclass
class Alert:
    asset_id: str
    fault_type: FaultType
    severity: Severity
    first_seen: float
    last_seen: float
    occurrence_count: int
    anomaly_score: float
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "fault_type": self.fault_type.value,
            "severity": self.severity.value,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "occurrence_count": self.occurrence_count,
            "anomaly_score": self.anomaly_score,
            "features": self.features,
        }


class AlertManager:
    def __init__(self, dedup_window_seconds: float = 300.0) -> None:
        self.dedup_window_seconds = dedup_window_seconds
        # key: (asset_id, fault_type)
        self._active: dict[tuple[str, FaultType], Alert] = {}

    def process(self, diagnosis: FaultDiagnosis) -> Optional[Alert]:
        if diagnosis.fault_type == FaultType.NORMAL:
            return None

        now = time.time()
        key = (diagnosis.asset_id, diagnosis.fault_type)
        existing = self._active.get(key)

        if existing is None:
            alert = Alert(
                asset_id=diagnosis.asset_id,
                fault_type=diagnosis.fault_type,
                severity=diagnosis.severity,
                first_seen=now,
                last_seen=now,
                occurrence_count=1,
                anomaly_score=diagnosis.confidence,
                features=dict(diagnosis.features),
            )
            self._active[key] = alert
            return alert

        # Within dedup window: update in place
        within_window = (now - existing.last_seen) <= self.dedup_window_seconds
        severity_increased = (
            _SEVERITY_RANK[diagnosis.severity] > _SEVERITY_RANK[existing.severity]
        )

        existing.last_seen = now
        existing.occurrence_count += 1
        existing.anomaly_score = max(existing.anomaly_score, diagnosis.confidence)

        if not within_window or severity_increased:
            existing.severity = diagnosis.severity
            return existing

        return None

    def active_alerts(self) -> list[Alert]:
        return list(self._active.values())

    def clear_resolved(self, resolved_fault_types: list[FaultType]) -> None:
        keys_to_remove = [
            k for k, v in self._active.items() if v.fault_type in resolved_fault_types
        ]
        for k in keys_to_remove:
            del self._active[k]
