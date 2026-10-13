"""Wire-format tests. The firmware must agree with these byte for byte."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from pdmcore import FEATURE_SPEC_VERSION, PAYLOAD_SCHEMA_VERSION
from pdmcore.io.payload import (
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


def _frame(n=63, **kw):
    rng = np.random.default_rng(0)
    base = dict(
        device_id=0x0123456789ABCDEF,
        timestamp_ms=1_760_000_000_000,
        features=rng.standard_normal(n),
        shaft_hz=29.67,
        temperature_c=48.5,
        battery_mv=3612,
        rssi_dbm=-97,
        uptime_s=864_000,
        sequence=4242,
    )
    base.update(kw)
    return TelemetryFrame(**base)


def test_header_layout_is_frozen():
    """The header is 44 bytes. Changing it breaks every deployed device, so the
    size is asserted here as well as in the module."""
    assert HEADER_SIZE == 44
    assert MAGIC == b"PDM3"
    assert frame_size(63) == 44 + 63 * 4 + 4 == 300


def test_round_trip_preserves_everything():
    f = _frame()
    back = decode_frame(encode_frame(f))
    assert back.device_id == f.device_id
    assert back.timestamp_ms == f.timestamp_ms
    assert back.sequence == f.sequence
    assert back.battery_mv == f.battery_mv
    assert back.rssi_dbm == f.rssi_dbm
    assert back.uptime_s == f.uptime_s
    assert back.shaft_hz == pytest.approx(f.shaft_hz, rel=1e-6)
    assert back.temperature_c == pytest.approx(f.temperature_c, rel=1e-6)
    # float32 on the wire: 1e-6 relative is the expected precision, and no
    # condition indicator needs better.
    assert np.allclose(back.features, f.features, rtol=1e-6, atol=1e-7)


def test_crc_detects_single_bit_flip():
    """Corruption before transmission -- a brown-out mid-DMA, a flash bit flip
    in the store-and-forward queue -- is what the CRC exists for. TLS would not
    catch any of those."""
    raw = bytearray(encode_frame(_frame()))
    raw[HEADER_SIZE + 7] ^= 0x01
    with pytest.raises(PayloadError, match="CRC mismatch"):
        decode_frame(bytes(raw))


def test_truncated_frame_rejected():
    raw = encode_frame(_frame())
    with pytest.raises(PayloadError):
        decode_frame(raw[:20])
    with pytest.raises(PayloadError, match="length mismatch|CRC"):
        decode_frame(raw[:-8] + raw[-4:])


def test_bad_magic_rejected():
    raw = bytearray(encode_frame(_frame()))
    raw[0:4] = b"XXXX"
    import zlib
    raw[-4:] = struct.pack("<I", zlib.crc32(bytes(raw[:-4])) & 0xFFFFFFFF)
    with pytest.raises(PayloadError, match="bad magic"):
        decode_frame(bytes(raw))


def test_feature_spec_mismatch_is_refused():
    """A v1 vector scored by a v2 model reads band RMS as kurtosis and produces
    confident nonsense. Refusing is the only safe behaviour."""
    f = _frame()
    f.feature_spec_version = FEATURE_SPEC_VERSION + 1
    raw = encode_frame(f)
    with pytest.raises(PayloadError, match="feature spec"):
        decode_frame(raw)
    # Archival tooling may opt out of the check explicitly.
    assert decode_frame(raw, expect_feature_spec=None).feature_spec_version == FEATURE_SPEC_VERSION + 1


def test_non_finite_features_refused_at_encode():
    f = _frame()
    f.features[3] = np.nan
    with pytest.raises(PayloadError, match="non-finite"):
        encode_frame(f)


@pytest.mark.parametrize("flag", [FrameFlags.CLIPPED, FrameFlags.FIFO_OVERRUN, FrameFlags.SELFTEST_FAIL])
def test_blocking_flags_mark_frame_unscoreable(flag):
    assert not _frame(flags=flag).scoreable
    assert flag.blocks_scoring


@pytest.mark.parametrize("flag", [FrameFlags.LOW_BATTERY, FrameFlags.SPEED_ESTIMATED,
                                  FrameFlags.STORED_AND_FORWARDED, FrameFlags.FALLBACK_CONFIG])
def test_informational_flags_do_not_block_scoring(flag):
    assert _frame(flags=flag).scoreable


def test_flags_survive_round_trip_and_combine():
    f = _frame(flags=FrameFlags.LOW_BATTERY | FrameFlags.SPEED_ESTIMATED)
    back = decode_frame(encode_frame(f))
    assert back.flags == f.flags
    assert FrameFlags.LOW_BATTERY in back.flags
    assert "SPEED_ESTIMATED" in back.as_dict()["flag_names"]


def test_sequence_gaps_handles_16bit_wrap():
    assert sequence_gaps([1, 2, 3, 4]) == (0, 0)
    assert sequence_gaps([1, 2, 5, 6]) == (1, 2)
    # Wrap from 65535 to 0 is contiguous, not a 65535-frame gap.
    assert sequence_gaps([65534, 65535, 0, 1]) == (0, 0)
    assert sequence_gaps([65534, 2]) == (1, 3)


def test_schema_version_is_what_the_module_claims():
    assert decode_frame(encode_frame(_frame())).schema_version == PAYLOAD_SCHEMA_VERSION
