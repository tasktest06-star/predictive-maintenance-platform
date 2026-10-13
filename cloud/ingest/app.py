#!/usr/bin/env python3
"""Telemetry ingest: decode -> validate -> score -> alarm -> work order.

Runs three ways, which is deliberate:

* ``--mqtt`` subscribes to the broker. Production.
* ``--http`` serves a small REST endpoint for gateways that cannot do MQTT
  (and for ``curl`` during commissioning).
* ``--replay`` drives the whole chain from the simulator with no broker, no
  database and no Docker. This is how the per-kit POCs run in CI, and how a
  reviewer sees the system work on a laptop in one command.

Design positions
----------------
**Validation before scoring, and the rejections are loud.** A frame whose CRC
fails, whose schema version is unknown, or whose flags say the ADC clipped is
*not* scored. Scoring a clipped acquisition produces a confident anomaly on an
artefact, which is worse than having no data because it teaches the operator to
ignore alarms. Every rejection is counted and exported as a metric.

**Per-asset state lives in one object.** :class:`AssetState` holds the health
index, the alarm state machine, the operating baseline and the RUL filter. One
asset's state is independent of every other's, so the service shards trivially
by device ID and a bad asset cannot corrupt a good one.

**Idempotency on (device_id, sequence).** Store-and-forward gateways replay
frames after a network outage -- that is the feature working correctly. Without
deduplication the replay would re-drive the alarm state machine and the EWMA
trend, producing a burst of spurious state changes hours after the fact.

**Work orders are created once per escalation, not once per alarming frame.**
The alarm state machine already debounces; the work-order step additionally
checks for an open order on the same asset and fault, because the thing that
destroys trust in a CMMS integration is twelve identical tickets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from collections import deque

LOG = logging.getLogger("ingest")

# pdmcore is the shared core; it must be importable. In the container it is
# pip-installed from core/; in --replay from a checkout it is on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(_HERE, "..", "..", "core"),
    os.path.join(_HERE, "..", "core"),
):
    if os.path.isdir(os.path.join(_candidate, "pdmcore")):
        sys.path.insert(0, os.path.abspath(_candidate))
        break

from pdmcore import FEATURE_SPEC_VERSION  # noqa: E402
from pdmcore.features import FeatureConfig, FeatureExtractor  # noqa: E402
from pdmcore.health import (  # noqa: E402
    AlarmConfig,
    AlarmState,
    HealthIndex,
    ParticleFilterRUL,
    baseline_from_vectors,
    diagnose,
)
from pdmcore.io.payload import PayloadError, TelemetryFrame, decode_frame  # noqa: E402

DEFAULT_TOPIC = "pdm/+/telemetry"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class Metrics:
    """Prometheus-text metrics. No client library: the exposition format is 20
    lines of string formatting, and a dependency that pulls in a web framework
    for that is not worth the supply-chain surface on an ingest hot path."""

    def __init__(self) -> None:
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[Tuple[str, str], float] = {}
        self.started = time.time()

    def inc(self, name: str, n: float = 1.0) -> None:
        self.counters[name] += n

    def set_gauge(self, name: str, asset: str, value: float) -> None:
        self.gauges[(name, asset)] = value

    def render(self) -> str:
        out = [
            "# HELP pdm_ingest_uptime_seconds Seconds since the service started",
            "# TYPE pdm_ingest_uptime_seconds gauge",
            f"pdm_ingest_uptime_seconds {time.time() - self.started:.1f}",
        ]
        for name, value in sorted(self.counters.items()):
            out.append(f"# TYPE pdm_{name} counter")
            out.append(f"pdm_{name} {value:g}")
        for (name, asset), value in sorted(self.gauges.items()):
            out.append(f"# TYPE pdm_{name} gauge")
            out.append(f'pdm_{name}{{asset="{asset}"}} {value:g}')
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Per-asset state
# ---------------------------------------------------------------------------
@dataclass
class WorkOrder:
    asset_id: str
    created_s: float
    fault: str
    severity: str
    confidence: float
    evidence: Dict[str, float]
    action: str
    alarm_state: str
    rul_days: Optional[float] = None
    status: str = "open"

    def as_dict(self) -> Dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_s)),
            "fault": self.fault,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "alarm_state": self.alarm_state,
            "rul_days": None if self.rul_days is None else round(self.rul_days, 1),
            "action": self.action,
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
            "status": self.status,
        }


@dataclass
class AssetState:
    """Everything the service knows about one asset."""

    asset_id: str
    shaft_hz: float
    extractor: FeatureExtractor
    health: HealthIndex
    rul: ParticleFilterRUL
    commissioning: List = field(default_factory=list)
    baseline: Dict[str, float] = field(default_factory=dict)
    hf_baseline: Optional[float] = None
    seen_sequences: "OrderedDict[int, bool]" = field(default_factory=OrderedDict)
    work_orders: List[WorkOrder] = field(default_factory=list)
    last_frame_s: Optional[float] = None
    n_frames: int = 0
    gaps: int = 0
    last_sequence: Optional[int] = None

    #: Commissioning window length. Below this the asset is "unknown", not
    #: "healthy" -- the dashboard says so and no alarm can fire.
    COMMISSION_FRAMES = 40
    #: Sequence numbers remembered for deduplication. 4096 at one frame per
    #: hour is ~170 days of history, far more than any store-and-forward queue.
    SEQ_MEMORY = 4096

    def is_duplicate(self, sequence: int) -> bool:
        if sequence in self.seen_sequences:
            return True
        self.seen_sequences[sequence] = True
        while len(self.seen_sequences) > self.SEQ_MEMORY:
            self.seen_sequences.popitem(last=False)
        return False

    def note_sequence(self, sequence: int) -> int:
        """Track gaps. Returns the number of frames apparently missed."""
        missed = 0
        if self.last_sequence is not None:
            delta = (sequence - self.last_sequence) & 0xFFFF
            if 1 < delta < 1000:  # a huge delta is a device reboot, not a gap
                missed = delta - 1
                self.gaps += 1
        self.last_sequence = sequence
        return missed

    @property
    def commissioned(self) -> bool:
        return len(self.commissioning) >= self.COMMISSION_FRAMES


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class IngestService:
    def __init__(
        self,
        *,
        fs_hz: float = 25600.0,
        default_shaft_hz: float = 29.67,
        iso_class: str = "group2_rigid",
        demod_band_hz: Tuple[float, float] = (2500.0, 6000.0),
        alarm_config: Optional[AlarmConfig] = None,
        vm_url: Optional[str] = None,
        asset_map: Optional[Dict[str, Dict]] = None,
    ):
        self.fs_hz = fs_hz
        self.default_shaft_hz = default_shaft_hz
        self.iso_class = iso_class
        self.demod_band_hz = demod_band_hz
        self.alarm_config = alarm_config or AlarmConfig()
        self.vm_url = vm_url
        self.asset_map = asset_map or {}
        self.assets: Dict[str, AssetState] = {}
        self.metrics = Metrics()
        self.events: Deque[Dict] = deque(maxlen=500)
        self.work_orders: List[WorkOrder] = []

    # -- asset lifecycle ---------------------------------------------------
    def _asset(self, device_id: str, shaft_hz: float) -> AssetState:
        st = self.assets.get(device_id)
        if st is not None:
            return st
        meta = self.asset_map.get(device_id, {})
        cfg = FeatureConfig(
            fs_hz=float(meta.get("fs_hz", self.fs_hz)),
            iso_class=meta.get("iso_class", self.iso_class),
            demod_band_hz=tuple(meta.get("demod_band_hz", self.demod_band_hz)),
            velocity_band_hz=(2.0, 1000.0) if shaft_hz < 10.0 else (10.0, 1000.0),
        )
        fx = FeatureExtractor(cfg)
        st = AssetState(
            asset_id=meta.get("name", device_id),
            shaft_hz=shaft_hz,
            extractor=fx,
            health=HealthIndex(device_id, fx.names, alarm_config=self.alarm_config),
            rul=ParticleFilterRUL(failure_threshold=20.0, min_health=2.0),
        )
        self.assets[device_id] = st
        LOG.info("new asset %s (%s), shaft %.2f Hz", device_id, st.asset_id, shaft_hz)
        self.metrics.inc("assets_discovered_total")
        return st

    # -- the hot path ------------------------------------------------------
    def handle_raw(self, raw: bytes) -> Optional[Dict]:
        """Decode and process one wire frame. Returns a result dict or None."""
        self.metrics.inc("frames_received_total")
        try:
            frame = decode_frame(raw, expect_feature_spec=FEATURE_SPEC_VERSION)
        except PayloadError as exc:
            # Categorise, because the three causes need different responses: a
            # CRC storm is a hardware or radio problem, a version mismatch is a
            # fleet-management problem, a length mismatch is a firmware bug.
            msg = str(exc)
            if "CRC" in msg:
                self.metrics.inc("frames_rejected_crc_total")
            elif "feature spec" in msg or "schema" in msg:
                self.metrics.inc("frames_rejected_version_total")
            else:
                self.metrics.inc("frames_rejected_malformed_total")
            LOG.warning("rejected frame: %s", exc)
            return None
        return self.handle_frame(frame)

    def handle_frame(self, frame: TelemetryFrame) -> Optional[Dict]:
        device_id = f"{frame.device_id:016x}"
        now_s = frame.timestamp_ms / 1000.0

        if not frame.scoreable:
            # Clipped, FIFO overrun or failed self-test. Recorded as a device
            # health event, never scored as a machine health event -- these are
            # different problems with different owners.
            self.metrics.inc("frames_rejected_flags_total")
            self.events.append({
                "ts": now_s, "asset": device_id, "kind": "device_fault",
                "flags": frame.as_dict()["flag_names"],
            })
            LOG.warning("device %s unscoreable: %s", device_id, frame.as_dict()["flag_names"])
            return None

        shaft_hz = frame.shaft_hz if frame.shaft_hz > 0 else self.default_shaft_hz
        st = self._asset(device_id, shaft_hz)

        if st.is_duplicate(frame.sequence):
            # Store-and-forward replay. Not an error -- the feature working.
            self.metrics.inc("frames_duplicate_total")
            return None
        missed = st.note_sequence(frame.sequence)
        if missed:
            self.metrics.inc("frames_missed_total", missed)

        if frame.n_features != len(st.extractor.names):
            self.metrics.inc("frames_rejected_width_total")
            LOG.error("device %s sent %d features, extractor expects %d",
                      device_id, frame.n_features, len(st.extractor.names))
            return None

        vector = frame.features
        st.n_frames += 1
        st.last_frame_s = now_s

        # Commissioning: accumulate a healthy reference before alarming. An
        # asset inside its commissioning window is "unknown", not "healthy".
        if not st.commissioned:
            st.commissioning.append(vector)
            if st.commissioned:
                import numpy as np

                X = np.vstack(st.commissioning)
                st.health.commission(X, [shaft_hz] * X.shape[0])
                st.baseline = baseline_from_vectors(X, st.extractor.names)
                idx = st.extractor.names.index("band.10_rms")
                st.hf_baseline = float(np.median(X[:, idx]))
                LOG.info("asset %s commissioned on %d frames", device_id, X.shape[0])
                self.metrics.inc("assets_commissioned_total")
            self.metrics.set_gauge("commissioning_frames", device_id, len(st.commissioning))
            return {"device_id": device_id, "state": "commissioning",
                    "frames": len(st.commissioning), "needed": st.COMMISSION_FRAMES}

        out = st.health.update(now_s, vector, speed_hz=shaft_hz)

        # Prognostics consume a physical, non-saturating degradation indicator
        # (high-frequency band RMS normalised to its healthy median), not the
        # capped health index. An exponential model fitted to a saturating
        # signal concludes the crossing has already happened.
        if st.hf_baseline and st.hf_baseline > 0:
            idx = st.extractor.names.index("band.10_rms")
            st.rul.update(now_s / 86400.0, float(vector[idx]) / st.hf_baseline)
        rul = st.rul.estimate()

        frame_obj = _FrameView(vector, st.extractor.names, shaft_hz)
        diagnoses = diagnose(frame_obj, baseline=st.baseline)

        self.metrics.set_gauge("health_index", device_id, out["index"])
        self.metrics.set_gauge("health_index_smoothed", device_id, out["smoothed"])
        self.metrics.set_gauge("health_slope_per_day", device_id, out["slope_per_day"])
        self.metrics.set_gauge("battery_mv", device_id, frame.battery_mv)
        self.metrics.set_gauge("rssi_dbm", device_id, frame.rssi_dbm)
        self.metrics.set_gauge("temperature_c", device_id, frame.temperature_c)
        self.metrics.inc("frames_scored_total")

        event = out["event"]
        if event is not None:
            self.events.append({
                "ts": now_s, "asset": device_id, "kind": "alarm",
                "from": event.from_state.value, "to": event.to_state.value,
                "reason": event.reason, "index": event.index,
                "evidence": event.evidence,
            })
            LOG.info("asset %s %s -> %s (%s)", device_id,
                     event.from_state.value, event.to_state.value, event.reason)
            self.metrics.inc(f"alarm_transitions_total_{event.to_state.value}")
            if event.is_escalation and event.to_state in (AlarmState.ALARM, AlarmState.CRITICAL):
                self._raise_work_order(st, event, diagnoses, rul)

        self._push_timeseries(device_id, now_s, out, frame, vector, st)

        return {
            "device_id": device_id,
            "asset_id": st.asset_id,
            "state": out["state"],
            "index": round(out["index"], 3),
            "smoothed": round(out["smoothed"], 3),
            "slope_per_day": round(out["slope_per_day"], 4),
            "top_contributors": {k: round(v, 2) for k, v in out["top_contributors"].items()},
            "diagnoses": [d.as_dict() for d in diagnoses],
            "rul": rul.as_dict(),
            "sequence": frame.sequence,
            "missed": missed,
        }

    # -- side effects ------------------------------------------------------
    def _raise_work_order(self, st: AssetState, event, diagnoses, rul) -> None:
        fault = diagnoses[0].fault if diagnoses else "elevated_vibration_unclassified"
        # One order per (asset, fault) while one is open. The alarm machine
        # debounces in time; this debounces in identity.
        for wo in st.work_orders:
            if wo.status == "open" and wo.fault == fault:
                LOG.info("work order already open for %s / %s", st.asset_id, fault)
                self.metrics.inc("work_orders_suppressed_total")
                return
        top = diagnoses[0] if diagnoses else None
        wo = WorkOrder(
            asset_id=st.asset_id,
            created_s=event.timestamp_s,
            fault=fault,
            severity=top.severity if top else "unknown",
            confidence=top.confidence if top else 0.0,
            evidence=dict(top.evidence) if top else dict(event.evidence),
            action=top.action if top else "Collect a full-resolution waveform and review.",
            alarm_state=event.to_state.value,
            rul_days=rul.median_days if rul.status == "ok" else None,
        )
        st.work_orders.append(wo)
        self.work_orders.append(wo)
        self.metrics.inc("work_orders_created_total")
        LOG.warning("WORK ORDER %s: %s (%s, confidence %.2f)",
                    st.asset_id, fault, wo.severity, wo.confidence)

    def _push_timeseries(self, device_id, now_s, out, frame, vector, st) -> None:
        """Write to VictoriaMetrics via the Prometheus import endpoint.

        Fire-and-forget with a short timeout: the time-series write is not
        allowed to block or fail the ingest path. Losing a trend point is
        recoverable; losing an alarm is not.
        """
        if not self.vm_url:
            return
        lines = [
            f'pdm_health_index{{asset="{device_id}"}} {out["index"]:g} {int(now_s * 1000)}',
            f'pdm_health_smoothed{{asset="{device_id}"}} {out["smoothed"]:g} {int(now_s * 1000)}',
            f'pdm_battery_mv{{asset="{device_id}"}} {frame.battery_mv:g} {int(now_s * 1000)}',
        ]
        for name in ("time.rms", "time.kurtosis", "time.crest", "iso.velocity_rms_mm_s"):
            if name in st.extractor.names:
                v = float(vector[st.extractor.names.index(name)])
                metric = "pdm_" + name.replace(".", "_")
                lines.append(f'{metric}{{asset="{device_id}"}} {v:g} {int(now_s * 1000)}')
        try:
            import urllib.request

            req = urllib.request.Request(
                self.vm_url.rstrip("/") + "/api/v1/import/prometheus",
                data=("\n".join(lines) + "\n").encode(),
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2.0).close()
            self.metrics.inc("timeseries_writes_total")
        except Exception as exc:  # noqa: BLE001 - never fail ingest on a metrics write
            self.metrics.inc("timeseries_write_errors_total")
            LOG.debug("timeseries write failed: %s", exc)

    # -- staleness ---------------------------------------------------------
    def check_stale(self, now_s: Optional[float] = None) -> List[Dict]:
        """Raise STALE alarms for silent devices.

        A silent sensor is indistinguishable from a healthy one on a dashboard,
        which is how a dead battery becomes an unmonitored machine for six
        months. Staleness is a first-class alarm.
        """
        now_s = now_s if now_s is not None else time.time()
        raised = []
        for device_id, st in self.assets.items():
            ev = st.health.alarms.check_stale(now_s)
            if ev is not None:
                raised.append({"asset": device_id, "reason": ev.reason})
                self.metrics.inc("alarm_transitions_total_stale")
                LOG.warning("asset %s stale: %s", device_id, ev.reason)
        return raised


class _FrameView:
    """Adapts a bare feature vector to what pdmcore.health.diagnose expects."""

    def __init__(self, vector, names: Sequence[str], shaft_hz: float):
        self.vector = vector
        self.names = tuple(names)
        self.shaft_hz = shaft_hz

    def as_dict(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.vector)}

    def get(self, name: str) -> float:
        return float(self.vector[self.names.index(name)])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_replay(args: argparse.Namespace) -> int:
    """End-to-end POC with no broker, no database and no Docker.

    Simulates a degrading machine, encodes real wire frames on the device side,
    pushes them through the actual ingest path, and prints the alarm and
    work-order history. This is the command every per-kit README points at.
    """
    import numpy as np

    from pdmcore.io.payload import FrameFlags, encode_frame
    from pdmcore.sim import MACHINE_LIBRARY, degradation_series, simulate

    spec = MACHINE_LIBRARY[args.machine]
    svc = IngestService(
        fs_hz=spec.fs_hz,
        default_shaft_hz=spec.shaft_hz,
        demod_band_hz=(2500.0, 6000.0),
        alarm_config=AlarmConfig(min_dwell_s=0.0),
        vm_url=args.vm_url,
    )
    cfg = FeatureConfig(fs_hz=spec.fs_hz, bearing=spec.bearing, demod_band_hz=(2500.0, 6000.0))
    device_fx = FeatureExtractor(cfg)
    device_id = 0x00A1B2C3D4E5F600

    print(f"machine        {spec.name}, {spec.rpm:.0f} rpm, bearing {spec.bearing.name}")
    print(f"commissioning  {AssetState.COMMISSION_FRAMES} healthy frames")
    print(f"degradation    {args.points} frames over {args.days:.0f} days, fault={args.fault}")
    print(f"wire format    {len(device_fx.names)} features\n")

    seq = 0
    t0 = 1_760_000_000.0

    def send(accel, t_s, *, flags=FrameFlags.NONE):
        nonlocal seq
        seq = (seq + 1) & 0xFFFF
        frame = TelemetryFrame(
            device_id=device_id,
            timestamp_ms=int(t_s * 1000),
            features=device_fx.extract(accel, spec.shaft_hz).vector,
            shaft_hz=spec.shaft_hz,
            temperature_c=48.0,
            battery_mv=3600 - seq // 20,
            rssi_dbm=-95,
            uptime_s=int(t_s - t0),
            sequence=seq,
            flags=flags,
        )
        return svc.handle_raw(encode_frame(frame))

    # Commissioning
    for i in range(AssetState.COMMISSION_FRAMES):
        send(simulate(spec, None, duration_s=0.4, seed=900 + i).accel_g, t0 + i * 3600.0)

    # Degradation
    series = degradation_series(spec, mode=args.fault, n_points=args.points,
                                duration_s=0.4, days=args.days, seed=5)
    print(f"{'day':>6} {'sev':>5} {'HI':>5} {'trend':>6} {'state':>9}  {'diagnosis':<30} RUL")
    for i, rec in enumerate(series):
        t = t0 + (AssetState.COMMISSION_FRAMES * 3600.0) + rec.truth["t_days"] * 86400.0
        res = send(rec.accel_g, t)
        if res is None or res.get("state") == "commissioning":
            continue
        escalated = any(e["ts"] == t and e["kind"] == "alarm" for e in svc.events)
        if i % max(1, args.points // 12) == 0 or escalated:
            dx = res["diagnoses"]
            label = f"{dx[0]['fault']} ({dx[0]['confidence']:.2f})" if dx else "-"
            r = res["rul"]
            rul = (f"{r['median_days']:.0f} d [{r['p10_days']:.0f},{r['p90_days']:.0f}]"
                   if r["status"] == "ok" else r["status"])
            print(f"{rec.truth['t_days']:6.1f} {rec.truth['severity']:5.2f} {res['index']:5.2f} "
                  f"{res['smoothed']:6.2f} {res['state']:>9}  {label:<30} {rul}")

    # Negative controls: the rejection paths must actually reject.
    print("\nvalidation checks")
    bad = bytearray(encode_frame(TelemetryFrame(
        device_id=device_id, timestamp_ms=int(t0 * 1000),
        features=device_fx.extract(series[-1].accel_g, spec.shaft_hz).vector,
        shaft_hz=spec.shaft_hz, sequence=60000)))
    bad[80] ^= 0x01
    print(f"  corrupted CRC      -> {'rejected' if svc.handle_raw(bytes(bad)) is None else 'ACCEPTED (BUG)'}")
    clipped = encode_frame(TelemetryFrame(
        device_id=device_id, timestamp_ms=int(t0 * 1000),
        features=device_fx.extract(series[-1].accel_g, spec.shaft_hz).vector,
        shaft_hz=spec.shaft_hz, sequence=60001, flags=FrameFlags.CLIPPED))
    print(f"  CLIPPED flag       -> {'rejected' if svc.handle_raw(clipped) is None else 'ACCEPTED (BUG)'}")
    dup = encode_frame(TelemetryFrame(
        device_id=device_id, timestamp_ms=int(t0 * 1000),
        features=device_fx.extract(series[-1].accel_g, spec.shaft_hz).vector,
        shaft_hz=spec.shaft_hz, sequence=seq))
    print(f"  replayed sequence  -> {'deduplicated' if svc.handle_raw(dup) is None else 'ACCEPTED (BUG)'}")
    stale = svc.check_stale(now_s=t0 + 1e7)
    print(f"  silent for 115 d   -> {'stale alarm raised' if stale else 'NO ALARM (BUG)'}")

    print("\nalarm history")
    for e in (ev for ev in svc.events if ev["kind"] == "alarm"):
        day = (e["ts"] - t0) / 86400.0
        top = ", ".join(f"{k}={v:.1f}" for k, v in list(e["evidence"].items())[:3])
        print(f"  day {day:6.1f}  {e['from']} -> {e['to']:9s} ({e['reason']}) [{top}]")

    print("\nwork orders")
    if not svc.work_orders:
        print("  none")
    for wo in svc.work_orders:
        print(json.dumps(wo.as_dict(), indent=2))

    print("\nmetrics")
    for line in svc.metrics.render().splitlines():
        if line.startswith("pdm_") and "{" not in line:
            print("  " + line)
    return 0


def run_mqtt(args: argparse.Namespace) -> int:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        LOG.error("paho-mqtt is not installed: pip install 'pdmcore[field]'")
        return 2

    svc = IngestService(vm_url=args.vm_url)
    stop = {"flag": False}

    def on_connect(client, _ud, _flags, rc):
        LOG.info("connected to broker rc=%s, subscribing to %s", rc, args.topic)
        client.subscribe(args.topic, qos=1)

    def on_message(_client, _ud, msg):
        result = svc.handle_raw(msg.payload)
        if result and result.get("state") not in (None, "commissioning"):
            LOG.debug("%s %s index=%.2f", result["device_id"], result["state"], result["index"])

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    if args.mqtt_user:
        client.username_pw_set(args.mqtt_user, os.environ.get("MQTT_PASSWORD", ""))
    if args.mqtt_tls:
        client.tls_set()
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)

    def handle_signal(_signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    client.loop_start()
    last_stale = time.time()
    try:
        while not stop["flag"]:
            time.sleep(1.0)
            if time.time() - last_stale > 300.0:
                svc.check_stale()
                last_stale = time.time()
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


def run_http(args: argparse.Namespace) -> int:
    """Minimal REST endpoint using only the standard library.

    POST /telemetry  (application/octet-stream)  -> JSON result
    GET  /metrics                                -> Prometheus text
    GET  /healthz                                -> 200
    GET  /assets                                 -> JSON asset summary
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    svc = IngestService(vm_url=args.vm_url)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/telemetry":
                self._send(404, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                self._send(400, b'{"error":"bad length"}')
                return
            result = svc.handle_raw(self.rfile.read(length))
            if result is None:
                self._send(422, b'{"error":"frame rejected; see service metrics"}')
                return
            self._send(200, json.dumps(result).encode())

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metrics":
                self._send(200, svc.metrics.render().encode(), "text/plain; version=0.0.4")
            elif self.path == "/healthz":
                self._send(200, b'{"status":"ok"}')
            elif self.path == "/assets":
                body = json.dumps({
                    did: {
                        "asset_id": st.asset_id,
                        "state": st.health.alarms.state.value,
                        "frames": st.n_frames,
                        "commissioned": st.commissioned,
                        "index": st.health.last_index,
                        "gaps": st.gaps,
                        "open_work_orders": sum(1 for w in st.work_orders if w.status == "open"),
                    }
                    for did, st in svc.assets.items()
                }).encode()
                self._send(200, body)
            else:
                self._send(404, b'{"error":"not found"}')

        def log_message(self, fmt: str, *a) -> None:
            LOG.debug(fmt, *a)

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    LOG.info("listening on http://%s:%d", args.bind, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", action="store_true", help="end-to-end POC, no infrastructure")
    mode.add_argument("--mqtt", action="store_true", help="subscribe to the broker")
    mode.add_argument("--http", action="store_true", help="serve the REST endpoint")
    ap.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "localhost"))
    ap.add_argument("--mqtt-port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    ap.add_argument("--mqtt-user", default=os.environ.get("MQTT_USER"))
    ap.add_argument("--mqtt-tls", action="store_true", default=bool(os.environ.get("MQTT_TLS")))
    ap.add_argument("--topic", default=os.environ.get("MQTT_TOPIC", DEFAULT_TOPIC))
    ap.add_argument("--vm-url", default=os.environ.get("VM_URL"))
    ap.add_argument("--bind", default="0.0.0.0")  # noqa: S104 - containerised
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--machine", default="motor_pump_37kw")
    ap.add_argument("--fault", default="bearing_outer")
    ap.add_argument("--points", type=int, default=90)
    ap.add_argument("--days", type=float, default=90.0)
    ap.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    if args.replay:
        return run_replay(args)
    if args.mqtt:
        return run_mqtt(args)
    return run_http(args)


if __name__ == "__main__":
    raise SystemExit(main())
