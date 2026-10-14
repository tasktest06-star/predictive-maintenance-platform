"""The agent loop. Deliberately the same shape as `cdsp/src/pdm_app.c`.

    health gate -> acquire -> features -> encode -> publish or queue
                -> drain the backlog while the link is up -> sleep

Same order, and for the same reasons:

- **The gate comes first.** On an MCU it is the battery: a node at 3.25 V can
  still beacon "my battery is flat" but cannot complete an acquisition and a
  transmission. On a Linux node the equivalent gates are thermal throttling and
  disk space -- a gateway with a full queue partition cannot store-and-forward,
  and discovering that *after* acquiring wastes the acquisition.
- **Drain after a successful publish.** The link is demonstrably up at that
  moment, which is the cheapest possible time to flush a backlog. Draining on a
  timer instead means paying a connection attempt to discover the link is still
  down.
- **Sequence advances only once a frame is built**, so a failed acquisition does
  not create a gap the cloud reports as lost data.

What differs from the firmware, and why:

- **Per channel, not per device.** A gateway aggregates several sensors; the
  channel index rides in the frame flags exactly as it does for the RA8D1's
  three axes, so the cloud needs no new concept.
- **Features come from `pdmcore`**, the same code the cloud trains on, rather
  than from a C reimplementation. On this tier there is no reason to have two.
- **No tiering.** The MCU computes a cheap tier first and escalates to save
  energy. A mains-powered node has nothing to save, so it always computes the
  full vector -- which also means its frames are never partially populated, and
  the cloud can tell a Linux node's "no bearing fault" from an MCU node's
  "bearing block not computed" by the tier field alone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pdmcore import FEATURE_SPEC_VERSION
from pdmcore.dsp.bearing import BEARING_CATALOG
from pdmcore.features import FeatureConfig, FeatureExtractor
from pdmcore.io.payload import FrameFlags, TelemetryFrame, encode_frame

from .calibration import Calibration, CalibrationError
from .config import AgentConfig, ChannelConfig
from .queue import FrameQueue
from .sink import Sink, SinkError
from .source import BlockMeta, Overrun, Source, SourceError

log = logging.getLogger("pdmedge")

#: ISO 20816 evaluation zones, indexed by `iso.zone_index`. Reported as a letter
#: because that is what the standard and every maintenance planner use; the
#: feature vector carries the index so the models see a number.
_ISO_ZONES = ("A", "B", "C", "D")

#: Frame flag bits shared with the firmware (`cdsp/include/pdm_pipeline.h`).
#: Imported from pdmcore rather than redefined, with the channel helper here
#: because the Python enum exposes the two bits separately.
def channel_flag(index: int) -> FrameFlags:
    """Frame flags carrying a channel index in bits 8-9."""
    if not 0 <= index <= 3:
        raise ValueError(f"channel index {index} does not fit the 2-bit field")
    return FrameFlags(index << 8)


@dataclass
class CycleResult:
    """What one acquisition cycle did, per channel."""

    channel: str
    ok: bool
    published: bool = False
    queued: bool = False
    drained: int = 0
    flags: FrameFlags = FrameFlags.NONE
    sequence: int = 0
    score: float = float("nan")
    velocity_rms_mm_s: float = float("nan")
    iso_zone: str = ""
    error: str = ""


@dataclass
class AgentStats:
    cycles: int = 0
    frames_published: int = 0
    frames_queued: int = 0
    frames_drained: int = 0
    acquire_failures: int = 0
    overruns: int = 0
    clipped: int = 0
    uncalibrated: int = 0
    gate_blocks: int = 0
    per_channel: Dict[str, int] = field(default_factory=dict)


class HealthGate:
    """The Linux equivalent of the firmware's battery check.

    Two conditions, both of which make a cycle pointless rather than merely
    degraded:

    - **No room to queue.** If publication fails and the queue partition is
      full, the frame is lost. Acquiring first and discovering that afterwards
      wastes the acquisition and, on a thermally limited board, the thermal
      budget too.
    - **Thermal throttling.** A Pi 5 or a Jetson under throttle produces
      correct numbers slowly, which is usually fine -- so this is a *warning*
      flag on the frame, not a block. It is on the frame because an operator
      debugging missing data needs to know the gateway was hot, and that fact
      is invisible from the data itself.
    """

    #: Below this much free space the queue cannot be trusted to accept a frame.
    MIN_FREE_MB = 64.0

    def __init__(self, queue_path: str, thermal_path: str = "",
                 throttle_c: float = 80.0) -> None:
        self.queue_path = queue_path
        self.thermal_path = thermal_path or "/sys/class/thermal/thermal_zone0/temp"
        self.throttle_c = float(throttle_c)

    def free_mb(self) -> float:
        import os
        import shutil

        path = self.queue_path
        while path and not os.path.isdir(path):
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        try:
            return shutil.disk_usage(path or ".").free / 1e6
        except OSError:  # pragma: no cover
            return float("inf")

    def temperature_c(self) -> float:
        try:
            with open(self.thermal_path, encoding="ascii") as fh:
                raw = fh.read().strip()
        except (OSError, ValueError):
            return float("nan")
        try:
            milli = float(raw)
        except ValueError:  # pragma: no cover
            return float("nan")
        # Linux thermal zones report millidegrees; a few report degrees.
        return milli / 1000.0 if milli > 200.0 else milli

    def check(self) -> Tuple[bool, FrameFlags, str]:
        """(may_acquire, flags_to_set, reason)."""
        free = self.free_mb()
        if free < self.MIN_FREE_MB:
            return False, FrameFlags.NONE, (
                f"only {free:.0f} MB free at {self.queue_path}: a failed publish "
                f"could not be queued, so this cycle would lose its data"
            )
        temp = self.temperature_c()
        if temp == temp and temp > self.throttle_c:
            return True, FrameFlags.TEMP_OUT_OF_RANGE, f"thermal: {temp:.0f} C"
        return True, FrameFlags.NONE, ""


class EdgeAgent:
    """One device's acquisition loop."""

    def __init__(
        self,
        config: AgentConfig,
        source: Source,
        sink: Sink,
        queue: Optional[FrameQueue] = None,
        calibration: Optional[Calibration] = None,
        gate: Optional[HealthGate] = None,
    ) -> None:
        self.cfg = config
        self.source = source
        self.sink = sink
        self.queue = queue if queue is not None else FrameQueue(config.queue_path,
                                                                config.queue_max_frames)
        self.calibration = calibration
        self.gate = gate if gate is not None else HealthGate(config.queue_path)
        self.stats = AgentStats()
        self._sequence = 0
        self._t0 = time.time()
        self._extractors: Dict[str, FeatureExtractor] = {}
        self._pending_flags = FrameFlags.NONE
        for ch in config.channels:
            self._extractors[ch.name] = FeatureExtractor(self._feature_config(ch))

    # -- setup ------------------------------------------------------------
    def _feature_config(self, ch: ChannelConfig) -> FeatureConfig:
        bearing = BEARING_CATALOG.get(ch.bearing)
        if bearing is None:
            raise ValueError(
                f"channel {ch.name!r}: unknown bearing {ch.bearing!r}; have "
                f"{sorted(BEARING_CATALOG)}"
            )
        return FeatureConfig(
            fs_hz=self.cfg.fs_hz,
            n_fft=self.cfg.n_fft,
            bearing=bearing,
            demod_band_hz=ch.demod_band_hz,
            envelope_n_fft=self.cfg.envelope_n_fft,
        )

    def _to_g(self, ch: ChannelConfig, raw: np.ndarray) -> np.ndarray:
        """Engineering units, or an explicit refusal.

        A channel that is neither declared to be in g nor calibrated raises:
        shipping volts labelled as g produces a confident anomaly score against
        a baseline in a different quantity.
        """
        if ch.units_are_g:
            return np.asarray(raw, dtype=np.float64)
        if self.calibration is None:
            raise CalibrationError(
                f"channel {ch.name!r} is not declared units_are_g and no "
                f"calibration was loaded. Set calibration_path, or set "
                f"units_are_g for a source that already produces g."
            )
        cal = self.calibration[ch.source_channel or ch.name]
        return cal.to_g(raw)

    # -- one cycle --------------------------------------------------------
    def run_cycle(self, now_ms: Optional[int] = None) -> List[CycleResult]:
        self.stats.cycles += 1
        results: List[CycleResult] = []

        may, gate_flags, reason = self.gate.check()
        if not may:
            self.stats.gate_blocks += 1
            log.warning("cycle skipped: %s", reason)
            return [CycleResult(channel="", ok=False, error=reason)]
        self._pending_flags |= gate_flags
        if gate_flags:
            log.warning("%s", reason)

        for index, ch in enumerate(self.cfg.channels):
            results.append(self._run_channel(index, ch, now_ms))
        return results

    def _run_channel(self, index: int, ch: ChannelConfig,
                     now_ms: Optional[int]) -> CycleResult:
        res = CycleResult(channel=ch.name, ok=False)
        try:
            raw, meta = self.source.read_block(self.cfg.block_samples,
                                               ch.source_channel or ch.name)
        except Overrun as exc:
            self.stats.overruns += 1
            self.stats.acquire_failures += 1
            res.error = str(exc)
            # A diagnostic frame, as the firmware does: a failed acquisition is
            # otherwise indistinguishable from a dead sensor, and the flag must
            # travel on THIS cycle's frame rather than the next good one.
            self._emit(res, index, ch,
                       np.zeros(len(self._extractors[ch.name].names), dtype=np.float64),
                       FrameFlags.FIFO_OVERRUN, now_ms, meta=None)
            return res
        except SourceError as exc:
            self.stats.acquire_failures += 1
            res.error = str(exc)
            log.error("channel %s: acquire failed: %s", ch.name, exc)
            return res

        flags = FrameFlags.NONE
        if meta.clipped:
            flags |= FrameFlags.CLIPPED
            self.stats.clipped += 1
        if not meta.fs_measured:
            # Not an error, but the cloud must know the rate is nominal: an IIO
            # divider or a MEMS oscillator can be percent-wrong, and every
            # bearing frequency scales with it. Its own flag rather than
            # borrowing FALLBACK_CONFIG, which means "running a degraded config
            # after a failed update" -- a different fact with a different fix.
            flags |= FrameFlags.RATE_NOMINAL

        try:
            x = self._to_g(ch, raw)
        except CalibrationError as exc:
            self.stats.uncalibrated += 1
            res.error = str(exc)
            log.error("channel %s: %s", ch.name, exc)
            return res

        extractor = self._extractors[ch.name]
        frame = extractor.extract(x, ch.shaft_hz, fs_hz=meta.fs_hz)
        res.velocity_rms_mm_s = frame.get("iso.velocity_rms_mm_s")
        res.iso_zone = _ISO_ZONES[min(int(frame.get("iso.zone_index")), len(_ISO_ZONES) - 1)]

        self._emit(res, index, ch, np.asarray(frame.vector, dtype=np.float64),
                   flags, now_ms, meta=meta)
        res.ok = True
        return res

    def _emit(self, res: CycleResult, index: int, ch: ChannelConfig,
              values: np.ndarray, flags: FrameFlags,
              now_ms: Optional[int], meta: Optional[BlockMeta]) -> None:
        ts = int(now_ms if now_ms is not None
                 else (meta.t0_ms if meta is not None else time.time() * 1000.0))
        tf = TelemetryFrame(
            device_id=self.cfg.device_id,
            timestamp_ms=ts,
            features=values,
            shaft_hz=ch.shaft_hz,
            temperature_c=self.gate.temperature_c(),
            battery_mv=0,                       # mains: 0 is "not applicable"
            rssi_dbm=0,
            uptime_s=int(time.time() - self._t0),
            sequence=self._sequence & 0xFFFF,
            flags=flags | self._pending_flags | channel_flag(index),
            feature_spec_version=FEATURE_SPEC_VERSION,
        )
        payload = encode_frame(tf)
        res.flags = tf.flags
        res.sequence = tf.sequence
        # Sequence advances once the frame exists, so a failed acquisition does
        # not look like lost data to the cloud's gap detector.
        self._sequence = (self._sequence + 1) & 0xFFFF
        self._pending_flags = FrameFlags.NONE

        try:
            self.sink.publish(payload)
        except SinkError as exc:
            log.info("publish failed (%s); queueing", exc)
            self.queue.push(payload, ts_ms=ts)
            self.stats.frames_queued += 1
            res.queued = True
            return

        self.stats.frames_published += 1
        res.published = True
        self.stats.per_channel[ch.name] = self.stats.per_channel.get(ch.name, 0) + 1
        res.drained = self.drain()

    # -- backlog ----------------------------------------------------------
    def drain(self, limit: Optional[int] = None) -> int:
        """Flush queued frames while the link is demonstrably up.

        Marks each replayed frame STORED_AND_FORWARDED and recomputes the CRC,
        because the flag byte changed. Forgetting the recompute turns every
        replayed frame into a CRC rejection -- i.e. store-and-forward silently
        discards everything it saved. (The firmware made exactly that mistake.)
        """
        budget = int(limit if limit is not None else self.cfg.max_drain)
        sent = 0
        for frame_id, payload in self.queue.peek_batch(budget):
            marked = _mark_stored_and_forwarded(payload)
            try:
                self.sink.publish(marked)
            except SinkError:
                break           # link went down mid-drain; leave the rest queued
            self.queue.ack(frame_id)
            sent += 1
        self.stats.frames_drained += sent
        return sent

    def run(self, cycles: Optional[int] = None, sleep: bool = True) -> AgentStats:
        n = 0
        while cycles is None or n < cycles:
            started = time.monotonic()
            self.run_cycle()
            n += 1
            if cycles is not None and n >= cycles:
                break
            if sleep:
                remaining = self.cfg.interval_s - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        return self.stats


def _mark_stored_and_forwarded(payload: bytes) -> bytes:
    """Set bit 7 of the flags field and fix the CRC.

    The offsets are the wire format's, from `pdmcore.io.payload`: flags is a
    little-endian uint16 at byte 40, and the CRC-32 is the last four bytes.
    """
    import zlib

    if len(payload) < 48:
        raise ValueError(f"payload is {len(payload)} bytes, too short to be a frame")
    buf = bytearray(payload)
    flags = int.from_bytes(buf[40:42], "little") | int(FrameFlags.STORED_AND_FORWARDED)
    buf[40:42] = (flags & 0xFFFF).to_bytes(2, "little")
    crc = zlib.crc32(bytes(buf[:-4])) & 0xFFFFFFFF
    buf[-4:] = crc.to_bytes(4, "little")
    return bytes(buf)
