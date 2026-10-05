from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.common.models import SensorReading
from .diagnosis_agent import DiagnosisAgent
from .work_order import WorkOrder

logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Maintenance — LLM-Augmented", version="1.0.0")

_agent: DiagnosisAgent = DiagnosisAgent()
_work_order_store: deque[WorkOrder] = deque(maxlen=100)
_store_lock = asyncio.Lock()


class DiagnoseRequest(BaseModel):
    asset_id: str
    timestamp: float
    vibration_x: list[float]
    vibration_y: list[float]
    vibration_z: list[float]
    temperature: float
    rpm: float
    sample_rate: int = 12800
    asset_context: Optional[dict[str, Any]] = None


class DiagnoseResponse(BaseModel):
    diagnosis: dict
    work_order: dict


class BatchDiagnoseRequest(BaseModel):
    readings: list[DiagnoseRequest]


def _to_sensor_reading(req: DiagnoseRequest) -> SensorReading:
    return SensorReading(
        asset_id=req.asset_id,
        timestamp=req.timestamp,
        vibration_x=np.array(req.vibration_x, dtype=np.float32),
        vibration_y=np.array(req.vibration_y, dtype=np.float32),
        vibration_z=np.array(req.vibration_z, dtype=np.float32),
        temperature=req.temperature,
        rpm=req.rpm,
        sample_rate=req.sample_rate,
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    reading = _to_sensor_reading(req)
    diagnosis, work_order = await _agent.diagnose(reading, req.asset_context)
    async with _store_lock:
        _work_order_store.append(work_order)
    return DiagnoseResponse(diagnosis=diagnosis.to_dict(), work_order=work_order.model_dump(mode="json"))


@app.post("/diagnose/batch")
async def diagnose_batch(req: BatchDiagnoseRequest) -> dict:
    results = []
    for item in req.readings:
        reading = _to_sensor_reading(item)
        diagnosis, work_order = await _agent.diagnose(reading, item.asset_context)
        async with _store_lock:
            _work_order_store.append(work_order)
        results.append({"diagnosis": diagnosis.to_dict(), "work_order": work_order.model_dump(mode="json")})
        # Brief yield to avoid blocking the event loop between heavy items
        await asyncio.sleep(0)
    return {"results": results, "count": len(results)}


@app.get("/work-orders")
async def list_work_orders() -> dict:
    async with _store_lock:
        orders = list(_work_order_store)
    return {"work_orders": [o.model_dump(mode="json") for o in orders], "count": len(orders)}


@app.get("/work-orders/{work_order_id}")
async def get_work_order(work_order_id: str) -> dict:
    async with _store_lock:
        for order in _work_order_store:
            if order.work_order_id == work_order_id:
                return order.model_dump(mode="json")
    raise HTTPException(status_code=404, detail=f"Work order {work_order_id} not found")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "llm_available": _agent.llm_available,
        "classifier_ready": _agent.classifier_ready,
    }
