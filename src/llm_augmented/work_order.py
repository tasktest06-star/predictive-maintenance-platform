from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class WorkOrderPriority(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class WorkOrder(BaseModel):
    work_order_id: str = Field(default_factory=lambda: str(uuid4()))
    asset_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: WorkOrderPriority
    fault_type: str
    severity: str
    title: str
    description: str
    root_cause: str
    recommended_actions: list[str]
    estimated_labor_hours: float
    parts_required: list[str]
    safety_precautions: list[str]
    signal_evidence: dict


def fallback_work_order(asset_id: str, fault_type: str, severity: str, signal_evidence: dict) -> WorkOrder:
    """Minimal work order when LLM is unavailable."""
    priority_map = {"critical": WorkOrderPriority.IMMEDIATE, "alert": WorkOrderPriority.URGENT}
    priority = priority_map.get(severity, WorkOrderPriority.ROUTINE)
    return WorkOrder(
        asset_id=asset_id,
        priority=priority,
        fault_type=fault_type,
        severity=severity,
        title=f"{fault_type.replace('_', ' ').title()} detected on {asset_id}",
        description=f"Automated detection: {fault_type}. Manual review required.",
        root_cause="LLM unavailable — root cause analysis not performed.",
        recommended_actions=["Inspect asset manually", "Review sensor trends", "Escalate to maintenance team"],
        estimated_labor_hours=2.0,
        parts_required=[],
        safety_precautions=["Follow LOTO procedures", "Wear appropriate PPE"],
        signal_evidence=signal_evidence,
    )
