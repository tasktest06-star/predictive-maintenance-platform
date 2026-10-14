"""Tests for the Jetson branch.

The kit's content is the workload study (an adversarial argument against its own
hardware) and the on-site training pipeline (the one justification that
survived). Both are testable; the GPU is not present.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (os.path.join(_REPO, "core"), os.path.join(_REPO, "edge"),
          os.path.join(_HERE, "..", "edge"), os.path.join(_HERE, "..", "tools")):
    sys.path.insert(0, p)

from pdmcore import FEATURE_SPEC_VERSION  # noqa: E402
from pdmcore.io.payload import FrameFlags, TelemetryFrame, encode_frame  # noqa: E402
from pdmedge.queue import FrameQueue  # noqa: E402

from train_onsite import (  # noqa: E402
    CALIBRATE_FRACTION,
    FIT_FRACTION,
    FPR_COUNT_FLOOR,
    FPR_SIGMA,
    MIN_FRAMES,
    MIN_SPAN_DAYS,
    TrainingSet,
    load_from_queue,
    train_one,
)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def _vec(rng, scale=1.0):
    """A plausible 63-feature vector: a stable mean with modest variation."""
    base = np.linspace(0.05, 1.2, 63)
    return base * (1.0 + scale * rng.normal(0.0, 0.03, 63))


def _fill_queue(path, n, *, device_id=0xD1, channel=0, step_ms=HOUR_MS,
                flags=FrameFlags.NONE, t0=DAY_MS, seed=1, drift=0.0):
    q = FrameQueue(path, max_frames=100_000)
    rng = np.random.default_rng(seed)
    for i in range(n):
        v = _vec(rng) * (1.0 + drift * i / max(1, n - 1))
        payload = encode_frame(TelemetryFrame(
            device_id=device_id, timestamp_ms=t0 + i * step_ms,
            features=v, shaft_hz=29.6667, sequence=i & 0xFFFF,
            flags=flags | FrameFlags(channel << 8)))
        q.push(payload, ts_ms=t0 + i * step_ms)
    q.close()
    return path


# --------------------------------------------------------------- queue intake
def test_loads_and_groups_by_device_and_channel(tmp_path):
    path = str(tmp_path / "q.sqlite")
    _fill_queue(path, 50, device_id=0xAA, channel=0)
    _fill_queue(path, 50, device_id=0xAA, channel=1, seed=2)
    _fill_queue(path, 50, device_id=0xBB, channel=0, seed=3)
    groups = load_from_queue(path)
    assert set(groups) == {(0xAA, 0), (0xAA, 1), (0xBB, 0)}
    assert all(g.n == 50 for g in groups.values())


def test_training_does_not_consume_the_queue(tmp_path):
    """These frames still have to go upstream."""
    path = str(tmp_path / "q.sqlite")
    _fill_queue(path, 30)
    load_from_queue(path)
    q = FrameQueue(path, max_frames=100_000)
    assert q.depth == 30


def test_unscoreable_frames_are_excluded(tmp_path):
    """A detector trained on clipped acquisitions learns the artefacts, then
    fails to alarm on the real thing because the real thing is less extreme."""
    path = str(tmp_path / "q.sqlite")
    _fill_queue(path, 20, flags=FrameFlags.NONE)
    _fill_queue(path, 20, flags=FrameFlags.CLIPPED, seed=9, t0=DAY_MS * 5)
    _fill_queue(path, 20, flags=FrameFlags.FIFO_OVERRUN, seed=10, t0=DAY_MS * 9)
    groups = load_from_queue(path)
    assert sum(g.n for g in groups.values()) == 20
    rejected = next(iter(groups.values())).rejected
    assert rejected.get("unscoreable (clipped/overrun/selftest)") == 40


def test_unsteady_speed_is_excluded_by_default(tmp_path):
    """High-order content is smeared on an unsteady record, so training on it
    teaches the detector that smearing is normal."""
    path = str(tmp_path / "q.sqlite")
    _fill_queue(path, 15, flags=FrameFlags.SPEED_UNSTEADY)
    assert load_from_queue(path) == {} or sum(
        g.n for g in load_from_queue(path).values()) == 0
    assert sum(g.n for g in load_from_queue(path, allow_unsteady=True).values()) == 15


def test_a_wrong_feature_spec_is_excluded(tmp_path):
    path = str(tmp_path / "q.sqlite")
    q = FrameQueue(path, max_frames=100)
    rng = np.random.default_rng(4)
    q.push(encode_frame(TelemetryFrame(
        device_id=1, timestamp_ms=DAY_MS, features=_vec(rng), shaft_hz=29.0,
        feature_spec_version=FEATURE_SPEC_VERSION + 1)), ts_ms=DAY_MS)
    q.close()
    groups = load_from_queue(path)
    assert sum(g.n for g in groups.values()) == 0


# ---------------------------------------------------------------- the refusals
def test_refuses_too_few_frames(tmp_path):
    path = str(tmp_path / "q.sqlite")
    _fill_queue(path, MIN_FRAMES - 100, step_ms=HOUR_MS * 3)
    ts = next(iter(load_from_queue(path).values()))
    res, det = train_one(ts)
    assert not res.accepted and det is None
    assert "usable frames" in res.reason
    assert str(MIN_FRAMES) in res.reason
    assert "covariance" in res.reason


def test_refuses_too_short_a_span(tmp_path):
    """200 frames from one afternoon describe one operating point.

    This is the refusal that matters most in practice: a commissioning engineer
    with a day on site has exactly enough frames and nowhere near enough span.
    """
    path = str(tmp_path / "q.sqlite")
    # 600 frames one minute apart: plenty of frames, 10 hours of span.
    _fill_queue(path, 600, step_ms=60_000)
    ts = next(iter(load_from_queue(path).values()))
    res, det = train_one(ts)
    assert not res.accepted and det is None
    assert "span" in res.reason
    assert "Monday" in res.reason, "the reason should say why span matters"


def test_accepts_a_proper_training_set(tmp_path):
    path = str(tmp_path / "q.sqlite")
    _fill_queue(path, 600, step_ms=HOUR_MS)          # 600 h = 25 days
    ts = next(iter(load_from_queue(path).values()))
    assert ts.span_days > MIN_SPAN_DAYS
    res, det = train_one(ts, target_fpr=0.01)
    assert res.accepted, res.reason
    assert det is not None
    assert res.n_train == int(600 * FIT_FRACTION)
    assert res.n_calibrate == int(600 * CALIBRATE_FRACTION)
    assert res.n_verify == 600 - res.n_train - res.n_calibrate
    assert np.isfinite(res.threshold) and res.threshold > 0


def test_refuses_a_population_that_moved(tmp_path):
    """The three-way time split exists to catch exactly this.

    A strong monotone drift means the calibration window and the verification
    window describe different machines. A random split would average the drift
    into both and produce a threshold that looks fine and fails next week.

    This test is also what found the bug it now guards: with a TWO-way split --
    calibrate on the holdout, then measure the false-alarm rate on that same
    holdout -- the rate equals the target by construction, the check could
    never fail, and this 4x drift sailed through it.
    """
    path = str(tmp_path / "q.sqlite")
    _fill_queue(path, 600, step_ms=HOUR_MS, drift=3.0)
    ts = next(iter(load_from_queue(path).values()))
    res, det = train_one(ts, target_fpr=0.01)
    assert not res.accepted, (
        f"a 4x drift across the span should be refused; got threshold "
        f"{res.threshold} with verify FPR {res.verify_fpr}"
    )
    assert det is None
    assert "population moved" in res.reason
    assert res.verify_alarms > res.verify_allowed_alarms
    # And the circular measurement is still reported, so the contrast is
    # visible: it sits at the target while the real one is far above it.
    assert res.calibrate_fpr <= 0.01 * 1.5


def test_the_verification_gate_tolerates_poisson_noise():
    """A fixed multiple of a 1 % target is the wrong gate on a small window.

    On 150 verification frames the expected false-alarm count is 1.5, and
    Poisson noise alone spans 0-6 at 3 sigma. A "4x the target" gate refuses
    stationary data roughly a third of the time -- which in a plant means the
    commissioning engineer re-runs training until it passes, selecting for
    whichever window happened to look quiet. That is worse than no gate.
    """
    for n, target in ((150, 0.01), (100, 0.01), (1000, 0.01)):
        lam = n * target
        allowed = lam + FPR_SIGMA * (lam ** 0.5) + FPR_COUNT_FLOOR
        # The band must admit ordinary noise...
        assert allowed > lam * 2.0, f"n={n}: band {allowed:.1f} is too tight"
        # ...and still reject a population that genuinely moved. A drifted
        # window alarms on a large fraction of its frames, not a few percent.
        assert allowed < n * 0.25, f"n={n}: band {allowed:.1f} is too loose"


def test_the_holdout_is_later_in_time_not_random(tmp_path):
    """Calibrating on data the fit has seen puts the threshold far too low --
    tools/export_model.py made exactly that mistake, 15x too low."""
    rng = np.random.default_rng(5)
    n = 600
    vectors = np.asarray([_vec(rng) for _ in range(n)])
    # Shuffled timestamps: train_one must sort by time, so the holdout is the
    # LATEST quarter regardless of row order.
    ts_ms = np.arange(n) * HOUR_MS + DAY_MS
    perm = rng.permutation(n)
    tset = TrainingSet(device_id=1, channel=0, vectors=vectors[perm],
                       timestamps_ms=ts_ms[perm])
    res, det = train_one(tset)
    assert res.accepted, res.reason
    assert res.n_train == int(n * FIT_FRACTION)
    assert res.n_verify == n - res.n_train - int(n * CALIBRATE_FRACTION)


# -------------------------------------------------------- the workload study
def test_workload_study_results_are_committed_and_negative():
    """The kit's own study concludes against its own hardware. If that ever
    flips, this test should fail and somebody should look at why."""
    path = os.path.join(_HERE, "..", "results", "workload_study.json")
    if not os.path.isfile(path):
        pytest.skip("workload_study.json not generated on this checkout")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    assert "Nothing in this platform needs a Jetson" in data["conclusion"]
    # Candidate 1: re-scoring a site-year is CPU work.
    c1 = data["candidate_1_rescoring"]
    assert c1["verdict"] == "CPU is sufficient"
    assert float(c1["full_history_s"]) < 600.0
    # Candidate 3: raw-waveform inference at hourly cadence is nowhere near a
    # CPU core, and the study records the cadence at which that changes.
    c3 = data["candidate_3_waveform"]
    assert c3["verdict"].startswith("CPU is sufficient")
    assert float(c3["gmac_per_s_required"]) < float(c3["cpu_gmac_per_s_realistic"]) / 100
    assert float(c3["cadence_that_would_need_a_gpu_hz"]) > 100.0
    # And the energy cost of being wrong is recorded.
    boards = {b["board"]: b for b in data["energy"]["boards"]}
    jetson = [b for n, b in boards.items() if "Super" in n][0]
    pi = [b for n, b in boards.items() if "Pi 5" in n][0]
    assert float(jetson["eur_per_year"]) > 3.0 * float(pi["eur_per_year"])


def test_training_fit_times_are_milliseconds_not_minutes():
    """Candidate 2's numbers decide whether training needs a GPU at all. The
    classical detectors are closed-form; only gradient descent would change
    the answer, and the study says so rather than guessing."""
    path = os.path.join(_HERE, "..", "results", "workload_study.json")
    if not os.path.isfile(path):
        pytest.skip("workload_study.json not generated on this checkout")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    c2 = data["candidate_2_training"]
    biggest = max(c2["fits"], key=lambda r: r["n_frames"])
    assert float(biggest["mahalanobis_ms"]) < 5000.0
    assert "torch" in c2["note"] and "absent rather than guessed" in c2["note"]
    assert c2["torch_available"] is False, (
        "torch became available: candidate 2 should now measure a Deep SVDD fit, "
        "which is the only training workload a GPU would change"
    )
