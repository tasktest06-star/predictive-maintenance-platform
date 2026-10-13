"""Transport and storage.

:mod:`pdmcore.io.payload` is the byte-for-byte contract with firmware; the C
twin lives in ``cdsp/include/pdm_payload.h`` on every hardware branch.
"""

from __future__ import annotations

from .payload import (
    HEADER_SIZE,
    MAGIC,
    FrameFlags,
    PayloadError,
    TelemetryFrame,
    decode_frame,
    encode_frame,
    frame_size,
    sequence_gaps,
)

__all__ = [
    "MAGIC", "HEADER_SIZE", "FrameFlags", "TelemetryFrame",
    "encode_frame", "decode_frame", "PayloadError", "frame_size", "sequence_gaps",
]
