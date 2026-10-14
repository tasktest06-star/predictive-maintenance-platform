"""Tests for the Linux-tier agent.

The agent is the Linux equivalent of `cdsp/src/pdm_app.c`, so these mirror
`core/tests/test_virtual_device.py`: the same end-to-end properties, checked on
the same wire format, including the same negative controls. A node is a node
whichever tier it is on, and the cloud must not be able to tell.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "core")))

from pdmcore.io.payload import FrameFlags, decode_frame  # noqa: E402

from pdmedge import (  # noqa: E402
    AgentConfig,
    Calibration,
    CalibrationError,
    ChannelCalibration,
    ChannelConfig,
    FrameQueue,
    SimSource,
    StdoutSink,
)
from pdmedge.agent import EdgeAgent, HealthGate, _mark_stored_and_forwarded  # noqa: E402
from pdmedge.config import ConfigError, load_config  # noqa: E402
from pdmedge.sink import FailingSink  # noqa: E402

FS = 25600.0
BLOCK = 8192


def _cfg(tmp_path, channels=None, **kw):
    return AgentConfig(
        device_id=kw.pop("device_id", 0x00A1B2C3D4E5F600),
        fs_hz=FS,
        block_samples=BLOCK,
        interval_s=kw.pop("interval_s", 3600.0),
        n_fft=4096,
        envelope_n_fft=2048,
        queue_path=str(tmp_path / "q.sqlite"),
        channels=channels or [ChannelConfig(name="accel", shaft_hz=29.6667, units_are_g=True)],
        **kw,
    )


def _frames(blob: bytes):
    """Split the length-prefixed stream, the same framing the firmware's
    virtual device writes -- so one parser serves both tiers."""
    out, off = [], 0
    while off + 2 <= len(blob):
        (n,) = (int.from_bytes(blob[off:off + 2], "little"),)
        off += 2
        if off + n > len(blob):
            break
        out.append(blob[off:off + n])
        off += n
    return out


def _agent(tmp_path, sink=None, cfg=None, **kw):
    cfg = cfg or _cfg(tmp_path)
    return EdgeAgent(cfg, SimSource(fs_hz=FS), sink or StdoutSink(io.BytesIO()),
                     queue=FrameQueue(cfg.queue_path, 1000), **kw)


# ----------------------------------------------------------------- the contract
def test_frames_decode_in_the_cloud(tmp_path):
    """The contract: a Linux node's frames are indistinguishable from an MCU
    node's. Same 300 bytes, same schema, same feature-spec version."""
    buf = io.BytesIO()
    agent = _agent(tmp_path, StdoutSink(buf))
    agent.run_cycle()
    agent.run_cycle()

    frames = _frames(buf.getvalue())
    assert len(frames) == 2
    for raw in frames:
        assert len(raw) == 300
        f = decode_frame(raw)          # raises on CRC, magic or version mismatch
        assert f.n_features == 63
        assert f.device_id == 0x00A1B2C3D4E5F600
        assert f.shaft_hz == pytest.approx(29.6667, rel=1e-4)
        assert np.all(np.isfinite(f.features))
        assert f.scoreable


def test_sequence_numbers_are_contiguous(tmp_path):
    buf = io.BytesIO()
    agent = _agent(tmp_path, StdoutSink(buf))
    for _ in range(5):
        agent.run_cycle()
    seqs = [decode_frame(r).sequence for r in _frames(buf.getvalue())]
    assert seqs == list(range(5))


def test_nominal_sample_rate_is_flagged(tmp_path):
    """A configured rate is not a measured rate, and the cloud has to know.

    The simulator reports `fs_measured=False`, so every frame must carry
    RATE_NOMINAL. Without it a 2 % rate error -- entirely normal for an IIO
    divider or a MEMS oscillator -- moves a 106 Hz BPFO line by 2.1 Hz, the
    whole harmonic-match tolerance, with nothing in the data to show why.
    """
    buf = io.BytesIO()
    agent = _agent(tmp_path, StdoutSink(buf))
    agent.run_cycle()
    f = decode_frame(_frames(buf.getvalue())[0])
    assert f.flags & FrameFlags.RATE_NOMINAL
    # And it must not be confused with FALLBACK_CONFIG, which means something
    # else entirely: a degraded config after a failed update.
    assert not (f.flags & FrameFlags.FALLBACK_CONFIG)


def test_multi_channel_frames_are_tagged(tmp_path):
    """A gateway's channels are distinguished exactly as the RA8D1's axes are."""
    chans = [
        ChannelConfig(name="pump_de", shaft_hz=29.6667, units_are_g=True),
        ChannelConfig(name="pump_nde", shaft_hz=29.6667, units_are_g=True),
        ChannelConfig(name="motor_de", shaft_hz=29.6667, units_are_g=True),
    ]
    buf = io.BytesIO()
    cfg = _cfg(tmp_path, channels=chans, tier="gateway")
    agent = _agent(tmp_path, StdoutSink(buf), cfg=cfg)
    agent.run_cycle()

    frames = [decode_frame(r) for r in _frames(buf.getvalue())]
    assert len(frames) == 3
    assert [FrameFlags(f.flags).channel for f in frames] == [0, 1, 2]
    # Sequence is per FRAME: ingest deduplicates on (device, sequence), so
    # three channels sharing one would have two silently discarded.
    assert [f.sequence for f in frames] == [0, 1, 2]


def test_more_channels_than_the_field_can_hold_is_refused(tmp_path):
    """Five channels do not fit a 2-bit field, and must fail loudly."""
    chans = [ChannelConfig(name=f"c{i}", shaft_hz=25.0, units_are_g=True) for i in range(5)]
    cfg = _cfg(tmp_path, channels=chans, tier="gateway")
    agent = _agent(tmp_path, StdoutSink(io.BytesIO()), cfg=cfg)
    with pytest.raises(ValueError, match="2-bit"):
        agent.run_cycle()


# ------------------------------------------------------------ store and forward
def test_store_and_forward_survives_an_outage(tmp_path):
    """Frames queued during an outage are replayed, CRC-clean, flagged.

    The CRC recompute is the part worth testing: setting the replay flag
    changes a byte, and forgetting to recompute turns every replayed frame into
    a rejection -- store-and-forward silently discarding everything it saved.
    The firmware made exactly that mistake.
    """
    sink = FailingSink(fail_until=3)
    cfg = _cfg(tmp_path)
    agent = EdgeAgent(cfg, SimSource(fs_hz=FS), sink, queue=FrameQueue(cfg.queue_path, 1000))

    for _ in range(3):
        agent.run_cycle()
    assert agent.queue.depth == 3, "frames should be queued while the link is down"
    assert agent.stats.frames_published == 0

    results = agent.run_cycle()           # link recovers
    assert results[0].published
    assert results[0].drained == 3
    assert agent.queue.depth == 0

    replayed = [decode_frame(p) for p in sink.frames]
    assert len(replayed) == 4
    stored = [f for f in replayed if f.flags & FrameFlags.STORED_AND_FORWARDED]
    assert len(stored) == 3
    for f in stored:
        assert f.scoreable, "a replayed frame must still be scoreable"


def test_drain_is_bounded(tmp_path):
    """A gateway flushing a huge backlog must not starve live data."""
    sink = FailingSink(fail_until=10)
    cfg = _cfg(tmp_path, max_drain=4)
    agent = EdgeAgent(cfg, SimSource(fs_hz=FS), sink, queue=FrameQueue(cfg.queue_path, 1000))
    for _ in range(10):
        agent.run_cycle()
    assert agent.queue.depth == 10
    results = agent.run_cycle()
    assert results[0].drained == 4, "drain must respect max_drain"
    assert agent.queue.depth == 6


def test_mark_stored_and_forwarded_fixes_the_crc(tmp_path):
    buf = io.BytesIO()
    agent = _agent(tmp_path, StdoutSink(buf))
    agent.run_cycle()
    original = _frames(buf.getvalue())[0]
    marked = _mark_stored_and_forwarded(original)
    assert len(marked) == len(original)
    f = decode_frame(marked)             # raises if the CRC was not recomputed
    assert f.flags & FrameFlags.STORED_AND_FORWARDED
    # And nothing else moved.
    assert np.allclose(f.features, decode_frame(original).features)


# --------------------------------------------------------------- calibration --
def test_uncalibrated_channel_is_refused_not_guessed(tmp_path):
    """An uncalibrated channel reports a different quantity, not a worse one."""
    cfg = _cfg(tmp_path, channels=[ChannelConfig(name="accel", shaft_hz=29.6667)])
    agent = _agent(tmp_path, StdoutSink(io.BytesIO()), cfg=cfg)
    results = agent.run_cycle()
    assert not results[0].ok
    assert "units_are_g" in results[0].error
    assert agent.stats.uncalibrated == 1


def test_calibration_converts_counts_to_g():
    """The whole chain reduces to counts_per_g, so check it against hand arithmetic."""
    cal = ChannelCalibration(
        name="piezo", sensitivity_mv_per_g=100.0, adc_full_scale_v=4.096,
        adc_bits=24, gain=1.0, certificate_date="2026-01-15",
    )
    # 100 mV/g into a 4.096 V full-scale 24-bit bipolar converter:
    #   codes/volt = 2^23 / 4.096 = 2 048 000
    #   codes/g    = 0.1 V/g x 2 048 000 = 204 800
    assert cal.counts_per_g == pytest.approx(204800.0, rel=1e-9)
    assert cal.full_scale_g == pytest.approx(40.96, rel=1e-9)
    assert cal.resolution_g == pytest.approx(1.0 / 204800.0, rel=1e-9)
    assert np.allclose(cal.to_g([204800, -102400, 0]), [1.0, -0.5, 0.0])


def test_calibration_rejects_an_impossible_sensitivity():
    with pytest.raises(CalibrationError):
        ChannelCalibration(name="x", sensitivity_mv_per_g=0.0,
                           adc_full_scale_v=4.096, adc_bits=24)
    with pytest.raises(CalibrationError):
        ChannelCalibration(name="x", sensitivity_mv_per_g=100.0,
                           adc_full_scale_v=4.096, adc_bits=4)


def test_frequency_response_correction_holds_its_ends():
    """Outside the measured band the correction holds, never extrapolates.

    A piezo's response rises towards its mounted resonance. An extrapolation
    that guessed the slope would be confidently wrong in exactly the region
    where the correction matters -- so it holds the end value instead.
    """
    cal = ChannelCalibration(
        name="p", sensitivity_mv_per_g=100.0, adc_full_scale_v=4.096, adc_bits=24,
        response=[(10.0, 0.95), (159.2, 1.00), (5000.0, 1.08), (10000.0, 1.35)],
    )
    g = cal.response_at([1.0, 10.0, 159.2, 5000.0, 10000.0, 40000.0])
    assert g[0] == pytest.approx(0.95)        # held below the band
    assert g[2] == pytest.approx(1.00)
    assert g[5] == pytest.approx(1.35)        # held above, not extrapolated
    # And the correction divides the response out.
    corrected = cal.correct_spectrum([10000.0], [1.35])
    assert corrected[0] == pytest.approx(1.0, rel=1e-9)


def test_certificate_status_tracks_age():
    import datetime as dt

    today = dt.date(2026, 10, 14)
    fresh = ChannelCalibration(name="a", sensitivity_mv_per_g=100.0,
                               adc_full_scale_v=4.096, adc_bits=24,
                               certificate_date="2026-06-01")
    due = ChannelCalibration(name="b", sensitivity_mv_per_g=100.0,
                             adc_full_scale_v=4.096, adc_bits=24,
                             certificate_date="2025-06-01")
    stale = ChannelCalibration(name="c", sensitivity_mv_per_g=100.0,
                               adc_full_scale_v=4.096, adc_bits=24,
                               certificate_date="2022-01-01")
    none = ChannelCalibration(name="d", sensitivity_mv_per_g=100.0,
                              adc_full_scale_v=4.096, adc_bits=24)
    assert fresh.certificate_status(today) == "current"
    assert due.certificate_status(today) == "due"
    assert stale.certificate_status(today) == "stale"
    assert none.certificate_status(today) == "none"


def test_calibration_file_round_trip(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"channels": {
        "voltage0": {
            "sensitivity_mv_per_g": 100.3, "adc_full_scale_v": 4.096,
            "adc_bits": 24, "serial": "TE8911-1234",
            "certificate_date": "2026-02-10", "noise_density_g": 7e-6,
            "response": [[10.0, 0.98], [159.2, 1.0], [10000.0, 1.2]],
        }
    }}), encoding="utf-8")
    cal = Calibration.load(str(path))
    assert len(cal) == 1
    assert "voltage0" in cal
    assert cal["voltage0"].serial == "TE8911-1234"
    assert "sensor noise" in cal.report()
    with pytest.raises(CalibrationError, match="no calibration for channel"):
        _ = cal["voltage1"]


def test_missing_calibration_file_is_an_error(tmp_path):
    with pytest.raises(CalibrationError, match="does not exist"):
        Calibration.load(str(tmp_path / "nope.json"))


# ------------------------------------------------------------------- the gate --
def test_gate_blocks_when_the_queue_partition_is_full(tmp_path, monkeypatch):
    """Discovering there is no room AFTER acquiring wastes the acquisition."""
    cfg = _cfg(tmp_path)
    gate = HealthGate(cfg.queue_path)
    monkeypatch.setattr(gate, "free_mb", lambda: 1.0)
    agent = _agent(tmp_path, StdoutSink(io.BytesIO()), cfg=cfg, gate=gate)
    results = agent.run_cycle()
    assert len(results) == 1 and not results[0].ok
    assert "MB free" in results[0].error
    assert agent.stats.gate_blocks == 1
    assert agent.stats.frames_published == 0


def test_thermal_throttle_flags_but_does_not_block(tmp_path, monkeypatch):
    """A hot gateway produces correct numbers slowly, which is usually fine --
    but an operator debugging missing data needs to know it was hot."""
    cfg = _cfg(tmp_path)
    gate = HealthGate(cfg.queue_path, throttle_c=50.0)
    monkeypatch.setattr(gate, "temperature_c", lambda: 85.0)
    buf = io.BytesIO()
    agent = _agent(tmp_path, StdoutSink(buf), cfg=cfg, gate=gate)
    results = agent.run_cycle()
    assert results[0].ok and results[0].published
    assert decode_frame(_frames(buf.getvalue())[0]).flags & FrameFlags.TEMP_OUT_OF_RANGE


def test_temperature_parses_millidegrees_and_degrees(tmp_path):
    p = tmp_path / "temp"
    p.write_text("57230", encoding="ascii")
    assert HealthGate(str(tmp_path), str(p)).temperature_c() == pytest.approx(57.23)
    p.write_text("57", encoding="ascii")
    assert HealthGate(str(tmp_path), str(p)).temperature_c() == pytest.approx(57.0)
    assert np.isnan(HealthGate(str(tmp_path), str(tmp_path / "absent")).temperature_c())


# ------------------------------------------------------------------- config ---
def test_config_refuses_a_block_shorter_than_its_envelope_fft(tmp_path):
    """Decimation cannot create resolution, only record length can.

    The firmware's version of this mistake silently stayed at tier 1 and
    reported every frame as "no bearing fault". Here it is a startup error.
    """
    with pytest.raises(ConfigError, match="envelope"):
        AgentConfig(device_id=1, fs_hz=FS, block_samples=4096, n_fft=4096,
                    envelope_n_fft=8192,
                    channels=[ChannelConfig(name="a", shaft_hz=25.0)])


def test_config_refuses_a_zero_shaft_speed():
    with pytest.raises(ConfigError, match="shaft_hz"):
        ChannelConfig(name="a", shaft_hz=0.0)


def test_config_refuses_a_zero_device_id(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"fs_hz": FS, "block_samples": BLOCK,
                                "channels": [{"name": "a", "shaft_hz": 25.0}]}),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="device_id"):
        load_config(str(path))


def test_config_refuses_a_duty_cycle_above_one():
    """A 0.32 s block every 0.1 s is a schedule the agent cannot keep."""
    with pytest.raises(ConfigError, match="duty cycle"):
        AgentConfig(device_id=1, fs_hz=FS, block_samples=BLOCK, interval_s=0.1,
                    channels=[ChannelConfig(name="a", shaft_hz=25.0)])


def test_config_loads_from_json(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "device_id": "a1b2c3d4e5f60001",
        "fs_hz": FS, "block_samples": BLOCK, "interval_s": 600,
        "tier": "gateway",
        "queue_path": str(tmp_path / "q.sqlite"),
        "channels": [
            {"name": "pump_de", "shaft_hz": 29.6667, "demod_band_hz": [2500, 6000]},
            {"name": "fan_de", "shaft_hz": 24.5, "bearing": "SKF6205"},
        ],
    }), encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.device_id == 0xA1B2C3D4E5F60001
    assert cfg.tier == "gateway"
    assert len(cfg.channels) == 2
    assert cfg.channels[0].demod_band_hz == (2500.0, 6000.0)


def test_noise_budget_compares_over_a_bandwidth():
    """Sensor noise is a density and ADC noise is a step: the comparison needs
    a bandwidth, and the first version of it did not have one."""
    cal = ChannelCalibration(
        name="piezo", sensitivity_mv_per_g=100.0, adc_full_scale_v=4.096,
        adc_bits=24, noise_density_g=7e-6,
    )
    b = cal.noise_budget(16000.0)
    # 7 ug/sqrt(Hz) over 16 kHz = 885 ug RMS; LSB 4.88 ug / sqrt(12) = 1.41 ug.
    assert float(b["sensor_rms_g"]) == pytest.approx(7e-6 * 16000.0 ** 0.5, rel=1e-9)
    assert float(b["adc_rms_g"]) == pytest.approx(cal.resolution_g / 12.0 ** 0.5, rel=1e-9)
    assert b["limited_by"] == "sensor"
    assert float(b["margin_db"]) == pytest.approx(56.0, abs=0.5)

    # A deliberately coarse converter flips the verdict, which is the point of
    # having the number at all.
    coarse = ChannelCalibration(name="p", sensitivity_mv_per_g=100.0,
                                adc_full_scale_v=4.096, adc_bits=12,
                                noise_density_g=7e-6)
    assert coarse.noise_budget(100.0)["limited_by"] == "adc"

    with pytest.raises(CalibrationError):
        cal.noise_budget(0.0)


def test_report_refuses_the_dimensionless_comparison():
    """Without a bandwidth the report says so instead of comparing anyway."""
    cal = Calibration(channels={"a": ChannelCalibration(
        name="a", sensitivity_mv_per_g=100.0, adc_full_scale_v=4.096,
        adc_bits=24, noise_density_g=7e-6)})
    assert "not the same quantity" in cal.report()
    assert "sensor-limited by" in cal.report(bandwidth_hz=16000.0)
