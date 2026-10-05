from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..common.data_generator import SensorDataGenerator
from ..common.models import FaultType, SensorReading
from .alert_manager import Alert
from .pipeline import StreamingPipeline

app = FastAPI(title="Predictive Maintenance — Streaming Pipeline", version="1.0.0")

_pipeline = StreamingPipeline()
# Per-asset SSE queues: asset_id → list of asyncio.Queue
_sse_queues: dict[str, list[asyncio.Queue]] = {}


def _on_alert(alert: Alert) -> None:
    queues = _sse_queues.get(alert.asset_id, [])
    for q in queues:
        try:
            q.put_nowait(alert.to_dict())
        except asyncio.QueueFull:
            pass


_pipeline._alert_callback = _on_alert


# --- Pydantic request/response models ---

class IngestRequest(BaseModel):
    asset_id: str
    timestamp: Optional[float] = None
    vibration_x: list[float] = Field(min_length=64)
    vibration_y: list[float] = Field(min_length=64)
    vibration_z: list[float] = Field(min_length=64)
    temperature: float
    rpm: float
    sample_rate: int = 12800


class DiagnosisResponse(BaseModel):
    asset_id: str
    timestamp: float
    fault_type: str
    severity: str
    confidence: float
    features: dict


class AssetStatus(BaseModel):
    asset_id: str
    readings_processed: int
    is_warmed_up: bool
    last_rms: Optional[float]
    last_temp: Optional[float]


# --- Endpoints ---

@app.post("/ingest", response_model=Optional[DiagnosisResponse])
async def ingest(req: IngestRequest):
    import numpy as np

    reading = SensorReading(
        asset_id=req.asset_id,
        timestamp=req.timestamp or time.time(),
        vibration_x=np.array(req.vibration_x, dtype=np.float32),
        vibration_y=np.array(req.vibration_y, dtype=np.float32),
        vibration_z=np.array(req.vibration_z, dtype=np.float32),
        temperature=req.temperature,
        rpm=req.rpm,
        sample_rate=req.sample_rate,
    )
    diagnosis = await _pipeline.process(reading)
    if diagnosis is None:
        return None
    return DiagnosisResponse(**diagnosis.to_dict())


@app.get("/stream/{asset_id}")
async def stream_asset(asset_id: str, request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_queues.setdefault(asset_id, []).append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            queues = _sse_queues.get(asset_id, [])
            if queue in queues:
                queues.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/alerts")
async def get_alerts():
    return [a.to_dict() for a in _pipeline._alert_manager.active_alerts()]


@app.get("/assets")
async def get_assets():
    result = []
    for asset_id, state in _pipeline._assets.items():
        features = state.last_features
        result.append(
            AssetStatus(
                asset_id=asset_id,
                readings_processed=state.readings_processed,
                is_warmed_up=state.detector.is_warmed_up,
                last_rms=features.rms if features else None,
                last_temp=features.temp_mean if features else None,
            )
        )
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "stats": _pipeline.stats()}
