"""Firmware to cloud, end to end, with no hardware.

Drives the **real firmware** -- `pdm_app.c`, `pdm_pipeline.c`, `pdm_payload.c`
compiled for the host against a file-backed board -- and feeds the 300-byte
frames it emits into the actual ingest service. If the firmware's feature
extraction, wire encoding, sequence handling, flag reporting or
store-and-forward logic is wrong, the cloud side notices here.

This is the strongest statement available without a board on the bench, and it
is the test every per-kit README points at. What it still does not cover is
stated plainly in `docs/07_production_readiness.md`: the sensor driver, DMA,
clock setup, radio stack, sleep modes and power are per-kit `board_*.c` files
and they need hardware.

Skipped, not failed, without a C compiler.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from dataclasses import replace
from typing import List

import numpy as np
import pytest

from pdmcore.io.payload import FrameFlags, PayloadError, decode_frame
from pdmcore.sim import MACHINE_LIBRARY, degradation_series, simulate

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CDSP = os.path.join(_REPO, "cdsp")
_BIN = os.path.join(_CDSP, "tests", "virtual_device")

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("cc") is None or not os.path.isdir(_CDSP),
    reason="no host toolchain or cdsp/ not present",
)

#: The host reference config samples at 26.667 kHz in 8192-sample blocks.
FS_HZ = 26667.0
ACQ = 8192


@pytest.fixture(scope="module")
def device_binary():
    proc = subprocess.run(
        ["make", "-C", _CDSP, "virtual-device"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PY": sys.executable},
    )
    if proc.returncode != 0 or not os.path.isfile(_BIN):
        pytest.fail(f"virtual_device failed to build:\n{proc.stdout}\n{proc.stderr}")
    return _BIN


def _write_samples(path: str, records: List[np.ndarray]) -> int:
    with open(path, "w", encoding="ascii") as fh:
        for rec in records:
            np.savetxt(fh, rec[:ACQ], fmt="%.7e")
    return len(records) * ACQ


def _run_device(binary: str, samples_path: str, cycles: int, **flags) -> tuple:
    argv = [binary, "--samples", samples_path, "--cycles", str(cycles)]
    for key, value in flags.items():
        opt = "--" + key.replace("_", "-")
        if value is True:
            argv.append(opt)
        elif value is not False and value is not None:
            argv += [opt, str(value)]
    proc = subprocess.run(argv, capture_output=True, timeout=600)
    assert proc.returncode == 0, f"device exited {proc.returncode}: {proc.stderr.decode()}"
    return _parse_frames(proc.stdout), proc.stderr.decode()


def _parse_frames(blob: bytes) -> List[bytes]:
    """Split the length-prefixed stream the host board writes."""
    frames, off = [], 0
    while off + 2 <= len(blob):
        (length,) = struct.unpack_from("<H", blob, off)
        off += 2
        if off + length > len(blob):
            break
        frames.append(blob[off : off + length])
        off += length
    return frames


# ---------------------------------------------------------------------------
def test_firmware_frames_decode_in_the_cloud(device_binary, tmp_path):
    """Every frame the firmware emits must decode, CRC-clean, with the right
    schema and feature-spec version. This is the contract."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=700 + i).accel_g for i in range(4)]
    path = str(tmp_path / "healthy.txt")
    _write_samples(path, records)

    frames, _log = _run_device(device_binary, path, 4, threshold=0.0)
    assert len(frames) == 4, f"expected 4 frames, got {len(frames)}"

    for i, raw in enumerate(frames):
        assert len(raw) == 300, f"frame {i} is {len(raw)} bytes, expected 300"
        decoded = decode_frame(raw)       # raises on CRC, magic or version mismatch
        assert decoded.n_features == 63
        assert decoded.device_id == 0x00A1B2C3D4E5F600
        assert decoded.shaft_hz == pytest.approx(29.6667, rel=1e-4)
        assert np.all(np.isfinite(decoded.features))
        assert decoded.scoreable


def test_sequence_numbers_are_contiguous(device_binary, tmp_path):
    """Gaps in the sequence are how the cloud detects dropped frames, so the
    firmware must only advance the counter when it actually built a frame."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=800 + i).accel_g for i in range(3)]
    path = str(tmp_path / "seq.txt")
    _write_samples(path, records)

    frames, _ = _run_device(device_binary, path, 8, threshold=0.0)
    seqs = [decode_frame(f).sequence for f in frames]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs))), f"non-contiguous: {seqs}"


def test_firmware_features_match_the_cloud_extractor(device_binary, tmp_path):
    """The headline check: the firmware's feature vector equals the one the
    cloud computes from the same samples.

    Only the non-bearing block is compared -- the device uses
    rectify-and-low-pass for the envelope where the cloud uses the analytic
    signal, a divergence documented in `docs/02_feature_spec.md`. The bearing
    block is verified by property in `cdsp/tests/test_pipeline.c`.
    """
    from pdmcore.dsp.bearing import BEARING_CATALOG
    from pdmcore.features import FeatureConfig, FeatureExtractor

    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    rec = simulate(spec, {"bearing_outer": 0.45}, duration_s=0.35, seed=909).accel_g[:ACQ]
    path = str(tmp_path / "one.txt")
    _write_samples(path, [rec])

    frames, _ = _run_device(device_binary, path, 1, threshold=0.0)
    device = decode_frame(frames[0]).features

    fx = FeatureExtractor(FeatureConfig(
        fs_hz=FS_HZ, n_fft=4096, bearing=BEARING_CATALOG["SKF6205"],
        demod_band_hz=(2500.0, 6000.0), envelope_n_fft=1024,
        velocity_band_hz=(10.0, 1000.0)))
    cloud = fx.extract(rec, 29.6667).vector

    bearing_idx = fx.names.index("bearing.outer_race.ratio")
    # float32 on the wire plus float32 accumulation on the device; 1e-3
    # relative is the float32 noise floor for these aggregates.
    for i in range(bearing_idx):
        assert device[i] == pytest.approx(cloud[i], rel=2e-3, abs=1e-9), (
            f"{fx.names[i]}: device {device[i]:.6g} vs cloud {cloud[i]:.6g}"
        )


def test_store_and_forward_survives_an_outage(device_binary, tmp_path):
    """An outage then a recovery: nothing may be lost, and replays must be
    marked and still decode.

    Modelled as "the first 5 transmissions fail, then the link is back", which
    is what store-and-forward exists for. A *periodic* failure pattern would be
    unsatisfiable by construction -- the drain's own transmission is the next
    attempt, so it fails too and the queue can never empty. That was the first
    version of this test, and it was the test that was wrong, not the firmware.
    """
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=900 + i).accel_g for i in range(3)]
    path = str(tmp_path / "flaky.txt")
    _write_samples(path, records)

    frames, log = _run_device(device_binary, path, 10, threshold=0.0, tx_fail_until=5)
    decoded = [decode_frame(f) for f in frames]
    # 10 cycles, 5 failed transmissions: 5 frames queued, then the recovery
    # cycle sends its own frame and drains the backlog.
    assert len(frames) == 10, f"frames lost across the outage; log:\n{log}"
    replayed = [d for d in decoded if FrameFlags.STORED_AND_FORWARDED in d.flags]
    assert len(replayed) == 5, (
        f"expected 5 replayed frames, got {len(replayed)}; log:\n{log}"
    )
    # Every frame still decodes -- the CRC is recomputed after the replay flag
    # is set. Forgetting that turns the whole queue into CRC rejections, i.e.
    # store-and-forward silently discards everything it saved.
    seqs = sorted(d.sequence for d in decoded)
    assert seqs == list(range(seqs[0], seqs[0] + 10)), f"gaps or duplicates: {seqs}"


def test_fifo_overrun_is_flagged_on_its_own_frame_only(device_binary, tmp_path):
    """An overrun block is a concatenation of two different times: its spectrum
    has a step discontinuity that looks exactly like an impact, so it must not
    be scored.

    The firmware emits a *diagnostic* frame instead of going silent -- a failed
    acquisition is otherwise invisible to the cloud except as a missing hour,
    indistinguishable from a dead sensor. The flag must land on that cycle's
    frame and no other: the first version made it sticky, which carried it onto
    the next, perfectly good acquisition and made the cloud refuse to score it.
    """
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=950 + i).accel_g for i in range(2)]
    path = str(tmp_path / "overrun.txt")
    _write_samples(path, records)

    frames, log = _run_device(device_binary, path, 6, threshold=0.0, overrun_every=2)
    assert "acquire-fail" in log, f"overruns were not reported:\n{log}"
    decoded = [decode_frame(f) for f in frames]
    overrun = [d for d in decoded if FrameFlags.FIFO_OVERRUN in d.flags]
    clean = [d for d in decoded if FrameFlags.FIFO_OVERRUN not in d.flags]
    assert overrun, "no frame carried the overrun flag"
    assert clean, "every frame was flagged: the flag is leaking between cycles"
    for d in overrun:
        assert not d.scoreable, "an overrun frame must not be scoreable"
        # Diagnostic frame: features are zeroed rather than being a corrupt
        # spectrum that happens to look plausible.
        assert float(np.max(np.abs(d.features))) == 0.0
    for d in clean:
        assert d.scoreable, "a good acquisition was marked unscoreable"


def test_critical_battery_stops_acquiring(device_binary, tmp_path):
    """Below the critical threshold the node must stop acquiring. A node at
    3.25 V can still beacon "my battery is flat"; it cannot reliably complete a
    307 ms acquisition plus a transmission, and dying mid-frame leaves a corrupt
    entry at the head of the flash queue."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=960).accel_g]
    path = str(tmp_path / "batt.txt")
    _write_samples(path, records)

    frames, log = _run_device(device_binary, path, 10, threshold=0.0,
                              battery_mv=3400, battery_decay_mv=30)
    assert "battery-critical" in log, f"never reached the critical path:\n{log}"
    assert len(frames) < 10, "kept acquiring below the critical battery level"


def test_selftest_failure_marks_the_first_frame(device_binary, tmp_path):
    """A node that cannot verify its own sensor must say so rather than stream
    plausible noise for a year. The cloud refuses to score the frame."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=970).accel_g]
    path = str(tmp_path / "st.txt")
    _write_samples(path, records)

    frames, _ = _run_device(device_binary, path, 2, threshold=0.0, selftest_fail=True)
    assert frames, "no frames emitted"
    first = decode_frame(frames[0])
    assert FrameFlags.SELFTEST_FAIL in first.flags
    assert not first.scoreable, "a failed-self-test frame must not be scoreable"


def test_estimated_speed_is_flagged(device_binary, tmp_path):
    """A tacholess speed estimate carries a few tenths of a percent of error
    that widens every order band. If a diagnosis is later disputed, whether the
    speed was measured or inferred is the first question."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=980).accel_g]
    path = str(tmp_path / "est.txt")
    _write_samples(path, records)

    frames, _ = _run_device(device_binary, path, 1, threshold=0.0, speed_estimated=True)
    frame = decode_frame(frames[0])
    assert FrameFlags.SPEED_ESTIMATED in frame.flags
    # Informational, not blocking.
    assert frame.scoreable


def test_ingest_scores_firmware_frames(device_binary, tmp_path):
    """The whole loop: firmware frames through the real ingest service, which
    commissions the asset and then escalates as the simulated bearing degrades.
    """
    sys.path.insert(0, os.path.join(_REPO, "cloud", "ingest"))
    import importlib

    app = importlib.import_module("app")

    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    healthy = [simulate(spec, None, duration_s=0.35, seed=1000 + i).accel_g
               for i in range(app.AssetState.COMMISSION_FRAMES + 4)]
    degrading = [r.accel_g for r in degradation_series(
        spec, mode="bearing_outer", n_points=18, duration_s=0.35, days=90.0, seed=5)]
    path = str(tmp_path / "run.txt")
    _write_samples(path, healthy + degrading)

    n_cycles = len(healthy) + len(degrading)
    frames, _ = _run_device(device_binary, path, n_cycles, threshold=0.0, quiet=True)
    assert len(frames) == n_cycles

    svc = app.IngestService(fs_hz=FS_HZ, default_shaft_hz=29.6667,
                            demod_band_hz=(2500.0, 6000.0),
                            alarm_config=app.AlarmConfig(min_dwell_s=0.0))
    states, indices = [], []
    for raw in frames:
        result = svc.handle_raw(raw)
        if result and result.get("state") != "commissioning":
            states.append(result["state"])
            indices.append(result["index"])

    assert svc.metrics.counters["frames_rejected_crc_total"] == 0
    assert svc.metrics.counters["frames_rejected_version_total"] == 0
    assert svc.metrics.counters["assets_commissioned_total"] == 1
    assert states, "no frame was scored after commissioning"
    # The health index must rise over the degradation, and the asset must leave
    # the normal state. Not asserting a specific escalation day: that depends on
    # the simulated severity profile and is covered by the Python-side tests.
    assert indices[-1] > indices[0], f"index did not rise: {indices[0]:.2f} -> {indices[-1]:.2f}"
    assert set(states) != {"normal"}, f"never escalated; states were {sorted(set(states))}"


def test_corrupting_a_firmware_frame_is_caught(device_binary, tmp_path):
    """Negative control on the whole chain: a single bit flip in a real
    firmware frame must be rejected by the cloud."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS_HZ)
    records = [simulate(spec, None, duration_s=0.35, seed=1100).accel_g]
    path = str(tmp_path / "corrupt.txt")
    _write_samples(path, records)

    frames, _ = _run_device(device_binary, path, 1, threshold=0.0)
    raw = bytearray(frames[0])
    raw[120] ^= 0x01
    with pytest.raises(PayloadError, match="CRC mismatch"):
        decode_frame(bytes(raw))
