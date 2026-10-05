from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

from src.common.models import FaultDiagnosis, FaultType, Severity, SensorReading
from .fault_classifier import QuickFaultClassifier
from .signal_processor import SignalProcessor
from .work_order import WorkOrder, WorkOrderPriority, fallback_work_order

logger = logging.getLogger(__name__)


def _classify_severity(fault: FaultType, confidence: float, rms: float) -> Severity:
    if fault == FaultType.NORMAL:
        return Severity.NORMAL
    if confidence < 0.5 or rms < 0.15:
        return Severity.WARNING
    if fault in (FaultType.BEARING_WEAR, FaultType.LUBRICATION, FaultType.MOTOR_ELECTRICAL) and confidence > 0.8:
        return Severity.CRITICAL
    if confidence > 0.7:
        return Severity.ALERT
    return Severity.WARNING


class DiagnosisAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self._processor = SignalProcessor()
        self._classifier = QuickFaultClassifier()

        if self._api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
                logger.info("Anthropic client initialized (model=%s)", model)
            except ImportError:
                logger.warning("anthropic package not installed; LLM disabled")

    @property
    def llm_available(self) -> bool:
        return self._client is not None

    @property
    def classifier_ready(self) -> bool:
        return self._classifier._ready

    def _build_prompt(
        self,
        asset_id: str,
        fault_type: FaultType,
        confidence: float,
        severity: Severity,
        signal_desc: dict,
        asset_context: Optional[dict],
    ) -> str:
        h = signal_desc["harmonic_powers"]
        ctx_section = ""
        if asset_context:
            ctx_lines = "\n".join(f"- {k}: {v}" for k, v in asset_context.items())
            ctx_section = f"\nAsset history:\n{ctx_lines}\n"

        return f"""You are an industrial equipment diagnostic AI. Analyze this sensor data and provide maintenance guidance.

Asset: {asset_id}
Detected fault: {fault_type.value} (confidence: {confidence:.1%})
Severity: {severity.value}

Signal measurements:
- Vibration RMS: {signal_desc["rms_g"]:.3f} g
- Peak vibration: {signal_desc["peak_g"]:.3f} g
- Kurtosis: {signal_desc["kurtosis"]:.2f}
- Dominant frequency: {signal_desc["dominant_freq_hz"]:.1f} Hz (running at {signal_desc["rpm"]:.0f} RPM)
- Temperature: {signal_desc["temperature_c"]:.1f}°C
- Harmonic powers (1x/2x/3x/4x): {h["1x"]:.4f}/{h["2x"]:.4f}/{h["3x"]:.4f}/{h["4x"]:.4f} g²
{ctx_section}
Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{{
  "explanation": "plain-language description of the fault and its significance",
  "root_cause": "likely root cause based on signal evidence",
  "recommended_actions": ["step-by-step action 1", "action 2"],
  "parts_required": ["part name if applicable"],
  "safety_precautions": ["safety step 1"],
  "estimated_labor_hours": 2.5,
  "priority": "routine|urgent|immediate"
}}"""

    def _parse_llm_response(self, text: str) -> dict:
        text = text.strip()
        # Extract JSON block if wrapped in markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Last-resort: extract first {...} block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in LLM response: {text[:200]}")
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in LLM response: {exc}") from exc

        # Normalise priority
        raw_priority = str(data.get("priority", "routine")).lower()
        valid_priorities = {p.value for p in WorkOrderPriority}
        if raw_priority not in valid_priorities:
            raw_priority = "routine"
        data["priority"] = raw_priority
        return data

    def _call_llm_sync(self, prompt: str) -> str:
        if self._client is None:
            raise RuntimeError("LLM client not initialized")
        import anthropic
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def diagnose(
        self,
        reading: SensorReading,
        asset_context: Optional[dict] = None,
    ) -> tuple[FaultDiagnosis, WorkOrder]:
        features = self._processor.extract_features(reading)
        signal_desc = self._processor.describe(reading)
        fault_type, confidence = self._classifier.predict(features)
        severity = _classify_severity(fault_type, confidence, signal_desc["rms_g"])

        explanation: Optional[str] = None
        work_order: WorkOrder

        if self._client is not None:
            try:
                prompt = self._build_prompt(
                    reading.asset_id, fault_type, confidence, severity, signal_desc, asset_context
                )
                raw = await asyncio.to_thread(self._call_llm_sync, prompt)
                llm_data = self._parse_llm_response(raw)
                explanation = llm_data.get("explanation")
                work_order = WorkOrder(
                    asset_id=reading.asset_id,
                    priority=WorkOrderPriority(llm_data["priority"]),
                    fault_type=fault_type.value,
                    severity=severity.value,
                    title=f"{fault_type.value.replace('_', ' ').title()} on {reading.asset_id}",
                    description=llm_data.get("explanation", ""),
                    root_cause=llm_data.get("root_cause", ""),
                    recommended_actions=llm_data.get("recommended_actions", []),
                    estimated_labor_hours=float(llm_data.get("estimated_labor_hours", 2.0)),
                    parts_required=llm_data.get("parts_required", []),
                    safety_precautions=llm_data.get("safety_precautions", []),
                    signal_evidence=signal_desc,
                )
            except Exception as exc:
                logger.warning("LLM call failed (%s); falling back to ML-only", exc)
                work_order = fallback_work_order(
                    reading.asset_id, fault_type.value, severity.value, signal_desc
                )
        else:
            work_order = fallback_work_order(
                reading.asset_id, fault_type.value, severity.value, signal_desc
            )

        diagnosis = FaultDiagnosis(
            asset_id=reading.asset_id,
            timestamp=reading.timestamp,
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            features={k: v for k, v in signal_desc.items() if k != "harmonic_powers"},
            explanation=explanation,
        )
        return diagnosis, work_order
