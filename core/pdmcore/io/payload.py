"""The wire format, shared byte-for-byte with firmware.

A telemetry frame is a packed little-endian structure followed by ``n_features``
float32 values and a CRC-32. It is deliberately not JSON: at 63 features a JSON
frame is ~1.6 kB against 300 B packed, and on LTE-M that difference is most of
the energy budget of an acquisition.

Layout (``PAYLOAD_SCHEMA_VERSION = 3``)::

    offset size  field
    0      4     magic           b"PDM3"
    4      1     schema_version  uint8
    5      1     feature_spec    uint8   -- FEATURE_SPEC_VERSION
    6      2     n_features      uint16
    8      8     device_id       uint64  -- MCU unique ID, truncated
    16     8     timestamp_ms    uint64  -- Unix ms, UTC
    24     4     uptime_s        uint32
    28     2     battery_mv      uint16
    30     2     rssi_dbm        int16
    32     4     shaft_hz        float32
    36     4     temperature_c   float32
    40     2     flags           uint16  -- see FrameFlags
    42     2     sequence        uint16  -- wraps; used for gap detection
    44     4*N   features        float32[n_features]
    44+4N  4     crc32           uint32  -- over bytes [0, 44+4N)

Why each unusual choice:

* **CRC-32 over the whole frame, even under TLS.** The CRC catches corruption
  that happens *before* transmission -- a brown-out during the DMA copy out of
  the accelerometer FIFO, or a flash-wear bit flip in the store-and-forward
  queue. TLS protects the link, not the device.
* **``sequence`` wraps at 16 bits** and is used to detect gaps. A sensor that
  silently drops 30 % of its frames looks healthy on a dashboard; the gap count
  is what makes that visible.
* **``flags``** carries the conditions the cloud cannot infer: clipping, FIFO
  overrun, a failed self-test, a fallback configuration. A clipped acquisition
  must never be scored, and only the device knows it clipped.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Optional, Sequence, Tuple

import numpy as np

from .. import FEATURE_SPEC_VERSION, PAYLOAD_SCHEMA_VERSION

__all__ = [
    "MAGIC",
    "HEADER_SIZE",
    "FrameFlags",
    "TelemetryFrame",
    "encode_frame",
    "decode_frame",
    "PayloadError",
    "frame_size",
]

MAGIC = b"PDM3"
_HEADER_FMT = "<4sBBHQQIHhffHH"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert HEADER_SIZE == 44, f"header layout changed: {HEADER_SIZE} bytes"


class PayloadError(ValueError):
    """Raised for any frame that must not be scored."""


class FrameFlags(IntFlag):
    """Device-side conditions the cloud cannot detect from the features alone."""

    NONE = 0
    CLIPPED = 1 << 0          # ADC hit full scale: amplitudes are invalid
    FIFO_OVERRUN = 1 << 1     # samples were lost: spectrum is corrupted
    SELFTEST_FAIL = 1 << 2    # sensor self-test failed this cycle
    LOW_BATTERY = 1 << 3      # below the configured cut-off
    FALLBACK_CONFIG = 1 << 4  # running a degraded config after a failed update
    SPEED_ESTIMATED = 1 << 5  # shaft_hz is tacholess, not measured
    TEMP_OUT_OF_RANGE = 1 << 6
    STORED_AND_FORWARDED = 1 << 7  # replayed from the queue, not live
    # Channel index, bits 8-9: zero on every single-axis kit, so those frames
    # are byte-identical to before this field existed. A multi-axis node ships
    # one frame per axis per acquisition and these bits are the only thing
    # distinguishing them -- read them with `channel`, below.
    CHANNEL_BIT0 = 1 << 8
    CHANNEL_BIT1 = 1 << 9
    ORDER_TRACKED = 1 << 10  # features are per-revolution orders, not hertz
    # A tachometer measured material speed drift across the record. Features are
    # still in hertz, so high-order content is smeared: widen the harmonic
    # tolerance and discount orders above ~10x rather than scoring them as-is.
    SPEED_UNSTEADY = 1 << 11

    @property
    def channel(self) -> int:
        """Sensor channel (axis) this frame came from, 0-3."""
        return (int(self) >> 8) & 0x3

    @property
    def blocks_scoring(self) -> bool:
        """Conditions under which the frame must be rejected, not merely noted.

        Scoring a clipped or FIFO-overrun acquisition produces a confident
        anomaly on an artefact. That is worse than having no data, because it
        trains the operator to ignore alarms.
        """
        return bool(self & (FrameFlags.CLIPPED | FrameFlags.FIFO_OVERRUN | FrameFlags.SELFTEST_FAIL))


@dataclass
class TelemetryFrame:
    """One decoded telemetry frame."""

    device_id: int
    timestamp_ms: int
    features: np.ndarray
    shaft_hz: float = 0.0
    temperature_c: float = float("nan")
    battery_mv: int = 0
    rssi_dbm: int = 0
    uptime_s: int = 0
    sequence: int = 0
    flags: FrameFlags = FrameFlags.NONE
    schema_version: int = PAYLOAD_SCHEMA_VERSION
    feature_spec_version: int = FEATURE_SPEC_VERSION
    extras: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(np.asarray(self.features).size)

    @property
    def scoreable(self) -> bool:
        return not self.flags.blocks_scoring

    def as_dict(self) -> dict:
        return {
            "device_id": f"{self.device_id:016x}",
            "timestamp_ms": self.timestamp_ms,
            "schema_version": self.schema_version,
            "feature_spec_version": self.feature_spec_version,
            "n_features": self.n_features,
            "shaft_hz": self.shaft_hz,
            "temperature_c": self.temperature_c,
            "battery_mv": self.battery_mv,
            "rssi_dbm": self.rssi_dbm,
            "uptime_s": self.uptime_s,
            "sequence": self.sequence,
            "flags": int(self.flags),
            "flag_names": [f.name for f in FrameFlags if f != FrameFlags.NONE and f in self.flags],
            "scoreable": self.scoreable,
        }


def frame_size(n_features: int) -> int:
    """Encoded size in bytes, for radio airtime and energy budgeting."""
    return HEADER_SIZE + 4 * int(n_features) + 4


def encode_frame(frame: TelemetryFrame) -> bytes:
    """Pack a frame. float32 is deliberate: float64 would double the payload
    for precision no condition indicator has."""
    feats = np.asarray(frame.features, dtype=np.float32).ravel()
    if feats.size > 0xFFFF:
        raise PayloadError(f"too many features: {feats.size}")
    if not np.all(np.isfinite(feats)):
        raise PayloadError("refusing to encode non-finite features")
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        frame.schema_version & 0xFF,
        frame.feature_spec_version & 0xFF,
        feats.size,
        frame.device_id & 0xFFFFFFFFFFFFFFFF,
        frame.timestamp_ms & 0xFFFFFFFFFFFFFFFF,
        frame.uptime_s & 0xFFFFFFFF,
        max(0, min(0xFFFF, int(frame.battery_mv))),
        max(-32768, min(32767, int(frame.rssi_dbm))),
        float(frame.shaft_hz),
        float(frame.temperature_c),
        int(frame.flags) & 0xFFFF,
        frame.sequence & 0xFFFF,
    )
    body = header + feats.tobytes()
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_frame(raw: bytes, *, expect_feature_spec: Optional[int] = FEATURE_SPEC_VERSION) -> TelemetryFrame:
    """Unpack and validate a frame.

    Raises :class:`PayloadError` on magic mismatch, truncation, CRC failure, or
    an unexpected feature-spec version. The version check is not pedantry:
    scoring a v1 vector with a v2 model silently produces confident, wrong
    diagnoses -- the features are in different positions, so the model is
    reading band RMS as kurtosis. Pass ``expect_feature_spec=None`` to accept
    any version (archival tooling only).
    """
    if len(raw) < HEADER_SIZE + 4:
        raise PayloadError(f"frame too short: {len(raw)} bytes")
    body, crc_bytes = raw[:-4], raw[-4:]
    (crc,) = struct.unpack("<I", crc_bytes)
    actual = zlib.crc32(body) & 0xFFFFFFFF
    if crc != actual:
        raise PayloadError(f"CRC mismatch: frame says {crc:08x}, computed {actual:08x}")
    (magic, schema, fspec, n_features, device_id, ts_ms, uptime,
     battery, rssi, shaft, temp, flags, seq) = struct.unpack(_HEADER_FMT, body[:HEADER_SIZE])
    if magic != MAGIC:
        raise PayloadError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if schema != PAYLOAD_SCHEMA_VERSION:
        raise PayloadError(
            f"schema version {schema} != {PAYLOAD_SCHEMA_VERSION}; upgrade the ingest service "
            "or the device fleet, do not guess the layout"
        )
    expected_len = HEADER_SIZE + 4 * n_features
    if len(body) != expected_len:
        raise PayloadError(f"length mismatch: header says {n_features} features "
                           f"({expected_len} B), got {len(body)} B")
    if expect_feature_spec is not None and fspec != expect_feature_spec:
        raise PayloadError(
            f"feature spec v{fspec} but this service expects v{expect_feature_spec}; "
            "refusing to score a frame whose feature layout is unknown"
        )
    feats = np.frombuffer(body[HEADER_SIZE:], dtype=np.float32).astype(np.float64)
    return TelemetryFrame(
        device_id=device_id,
        timestamp_ms=ts_ms,
        features=feats,
        shaft_hz=float(shaft),
        temperature_c=float(temp),
        battery_mv=int(battery),
        rssi_dbm=int(rssi),
        uptime_s=int(uptime),
        sequence=int(seq),
        flags=FrameFlags(flags),
        schema_version=int(schema),
        feature_spec_version=int(fspec),
    )


def sequence_gaps(sequences: Sequence[int]) -> Tuple[int, int]:
    """``(n_gaps, n_missing)`` from a list of received sequence numbers.

    Handles the 16-bit wrap. A sensor dropping frames looks identical to a
    healthy one on a trend chart, so this is computed per device per day and
    alarmed on separately from the vibration itself.
    """
    seqs = [int(s) & 0xFFFF for s in sequences]
    n_gaps = 0
    n_missing = 0
    for prev, cur in zip(seqs, seqs[1:]):
        delta = (cur - prev) & 0xFFFF
        if delta > 1:
            n_gaps += 1
            n_missing += delta - 1
    return n_gaps, n_missing
