"""Tests for the gateway tier.

The gateway's job is fan-in, forwarding, local autonomy and knowing which of
its nodes have gone quiet. The last one is the part that cannot be tested
anywhere else, because it is the only view that survives an uplink outage.
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
          os.path.join(_HERE, "..", "gateway"), os.path.join(_HERE, "..", "tools")):
    sys.path.insert(0, p)

from pdmcore.io.payload import FrameFlags, TelemetryFrame, encode_frame  # noqa: E402
from pdmedge.queue import FrameQueue  # noqa: E402
from pdmedge.sink import FailingSink, SinkError, StdoutSink  # noqa: E402

from aggregator import Gateway, NodeState  # noqa: E402
from hailo_backend import (  # noqa: E402
    CpuWaveformScorer,
    HailoScorer,
    HailoUnavailable,
    build_scorer,
)

HOUR_MS = 3_600_000


def _frame(device_id: int, sequence: int, ts_ms: int, *,
           features: np.ndarray = None, flags: FrameFlags = FrameFlags.NONE) -> bytes:
    f = features if features is not None else np.linspace(0.0, 1.0, 63)
    return encode_frame(TelemetryFrame(
        device_id=device_id, timestamp_ms=ts_ms,
        features=np.asarray(f, dtype=np.float64), shaft_hz=29.6667,
        sequence=sequence & 0xFFFF, flags=flags))


def _gw(tmp_path, sink=None, **kw):
    q = FrameQueue(str(tmp_path / "gq.sqlite"), max_frames=1000)
    return Gateway(q, sink or StdoutSink(open(os.devnull, "wb")), **kw)


# ------------------------------------------------------------------- fan-in --
def test_accepts_and_forwards(tmp_path):
    sink = FailingSink()
    gw = _gw(tmp_path, sink)
    out = gw.handle(_frame(0xAA01, 0, HOUR_MS))
    assert out["accepted"] and out["forwarded"]
    assert gw.stats.nodes == 1
    assert sink.published == 1


def test_a_corrupt_frame_is_not_forwarded(tmp_path):
    """Passing a corrupt frame upstream converts a local radio problem into a
    fleet-wide data-quality problem, and the cloud knows less than the gateway
    about which link it came from."""
    sink = FailingSink()
    gw = _gw(tmp_path, sink)
    payload = bytearray(_frame(0xAA02, 0, HOUR_MS))
    payload[60] ^= 0xFF                       # break a feature byte, not the CRC field
    out = gw.handle(bytes(payload))
    assert not out["accepted"]
    assert not out["forwarded"]
    assert sink.published == 0
    assert gw.stats.rejected == 1


def test_an_unknown_schema_is_rejected(tmp_path):
    gw = _gw(tmp_path, FailingSink(), accept_schema=(99,))
    out = gw.handle(_frame(0xAA03, 0, HOUR_MS))
    assert not out["accepted"]
    assert "schema" in str(out["error"])


def test_gaps_are_counted_wrap_safely(tmp_path):
    gw = _gw(tmp_path, FailingSink())
    gw.handle(_frame(0xAA04, 65534, HOUR_MS))
    # 65534 -> 65535 -> 0 -> 1, so 65535 and 0 are both missing: TWO frames.
    # The first version of this test asserted 1 and was wrong, which is a
    # useful reminder that wrap arithmetic is easier to get wrong in the test
    # than in the code.
    out = gw.handle(_frame(0xAA04, 1, 2 * HOUR_MS))
    assert out["gap"] == 2, f"wrap-around mis-counted: {out['gap']}"
    assert gw.stats.gaps == 2


def test_contiguous_multichannel_frames_are_not_gaps(tmp_path):
    """A three-axis node advances the sequence once per FRAME, so N, N+1, N+2
    is contiguous -- not two gaps. Getting this wrong would report every
    multi-axis node as losing two thirds of its data."""
    gw = _gw(tmp_path, FailingSink())
    for ch in range(3):
        out = gw.handle(_frame(0xAA05, ch, HOUR_MS,
                               flags=FrameFlags(ch << 8)))
        assert out["gap"] == 0
        assert out["channel"] == ch
    assert gw.stats.gaps == 0


# -------------------------------------------------------- store and forward --
def test_queues_when_the_uplink_is_down_then_drains(tmp_path):
    sink = FailingSink(fail_until=3)
    gw = _gw(tmp_path, sink)
    for i in range(3):
        out = gw.handle(_frame(0xAA06, i, HOUR_MS * (i + 1)))
        assert out["queued"], f"frame {i} should have been queued"
    assert gw.queue.depth == 3
    out = gw.handle(_frame(0xAA06, 3, HOUR_MS * 4))
    assert out["forwarded"]
    assert out["drained"] == 3
    assert gw.queue.depth == 0


def test_drain_is_bounded(tmp_path):
    sink = FailingSink(fail_until=20)
    gw = _gw(tmp_path, sink, max_drain=5)
    for i in range(20):
        gw.handle(_frame(0xAA07, i, HOUR_MS * (i + 1)))
    out = gw.handle(_frame(0xAA07, 20, HOUR_MS * 21))
    assert out["drained"] == 5
    assert gw.queue.depth == 15


# ------------------------------------------------------------ local autonomy --
class _Scorer:
    def __init__(self, value):
        self.value = value

    def score(self, X):
        return np.full(len(np.atleast_2d(X)), self.value, dtype=np.float64)


def test_local_alarm_does_not_need_the_uplink(tmp_path):
    """The outage that takes the uplink down is often the event that takes the
    machine down, so an alarm must not require a round trip."""
    sink = FailingSink(fail_until=99)          # uplink permanently down
    gw = _gw(tmp_path, sink, scorer=_Scorer(50.0), threshold=10.0)
    out = gw.handle(_frame(0xAA08, 0, HOUR_MS))
    assert out["queued"], "the frame should be queued, the uplink is down"
    assert out["alarm"], "but the alarm must still fire"
    assert gw.stats.local_alarms == 1


def test_a_failing_scorer_does_not_take_the_gateway_down(tmp_path):
    class _Broken:
        def score(self, X):
            raise RuntimeError("model file corrupt")

    gw = _gw(tmp_path, FailingSink(), scorer=_Broken(), threshold=1.0)
    out = gw.handle(_frame(0xAA09, 0, HOUR_MS))
    assert out["accepted"] and out["forwarded"], "forwarding must survive a bad model"
    assert np.isnan(out["score"])


def test_unscoreable_frames_are_forwarded_but_not_scored(tmp_path):
    """A clipped or overrun frame must not be scored -- a confident anomaly on
    an artefact trains the operator to ignore alarms -- but it must still reach
    the cloud, which needs to know the acquisition failed."""
    gw = _gw(tmp_path, FailingSink(), scorer=_Scorer(1e6), threshold=1.0)
    out = gw.handle(_frame(0xAA0A, 0, HOUR_MS, flags=FrameFlags.FIFO_OVERRUN))
    assert out["forwarded"]
    assert "score" not in out
    assert not out["alarm"]
    assert gw.stats.local_alarms == 0


# ------------------------------------------------------- the gateway's view --
def test_silent_nodes_are_detected_from_the_learned_interval(tmp_path):
    """The commonest real event in a deployed fleet, and invisible downstream:
    a node that stops reporting looks exactly like a healthy quiet machine."""
    gw = _gw(tmp_path, FailingSink())
    # Two nodes, both on an hourly cadence, observed for a few cycles.
    # now_ms is passed explicitly: silence is measured from ARRIVAL time, not
    # from the frame's own timestamp, so that a node with a wrong clock cannot
    # look permanently fresh.
    for i in range(5):
        t = HOUR_MS * (i + 1)
        gw.handle(_frame(0xBB01, i, t), now_ms=t)
        gw.handle(_frame(0xBB02, i, t), now_ms=t)
    now = HOUR_MS * 6
    assert gw.silent_nodes(now) == [], "nothing should be silent yet"

    # BB01 keeps reporting right up to "now"; BB02 stops at hour 5.
    for i in range(5, 13):
        t = HOUR_MS * (i + 1)
        gw.handle(_frame(0xBB01, i, t), now_ms=t)
    now = HOUR_MS * 14
    silent = gw.silent_nodes(now)
    assert len(silent) == 1, f"expected one silent node, got {silent}"
    assert silent[0][0] == 0xBB02
    assert silent[0][1] == pytest.approx(9.0, abs=0.1)
    # And the node that is still reporting is not flagged, even though it is
    # an hour quiet: one interval of silence is normal.
    assert 0xBB01 not in [d for d, _ in silent]
    assert "SILENT NODES" in gw.report(now)


def test_one_missed_cycle_is_not_silence(tmp_path):
    """An alarm that cries wolf on every missed frame trains an operator to
    ignore it. Three intervals, not one."""
    gw = _gw(tmp_path, FailingSink())
    for i in range(5):
        t = HOUR_MS * (i + 1)
        gw.handle(_frame(0xBB03, i, t), now_ms=t)
    assert gw.silent_nodes(HOUR_MS * 7) == [], "two hours quiet is not silence"
    assert gw.silent_nodes(HOUR_MS * 10) != [], "five hours quiet is"


def test_interval_learning_is_slow_to_believe_a_change(tmp_path):
    """One early frame must not reset the silence threshold and mask an outage."""
    st = NodeState(device_id=1, first_seen_ms=0, last_seen_ms=0)
    for _ in range(10):
        st.observe_interval(HOUR_MS)
    assert st.interval_ms == pytest.approx(HOUR_MS, rel=1e-6)
    st.observe_interval(60_000.0)                 # one frame a minute later
    assert st.interval_ms > 0.75 * HOUR_MS, (
        "a single early frame moved the learned interval too far"
    )


# ------------------------------------------------------------------- Hailo ---
def test_hailo_refuses_rather_than_stubbing():
    """A stub returning plausible probabilities would be the most misleading
    artefact in the repository."""
    with pytest.raises(HailoUnavailable, match="not implemented"):
        HailoScorer("/nonexistent/model.hef")


def test_waveform_scorer_holds_the_interface_without_producing_numbers():
    with pytest.raises(NotImplementedError, match="no raw-waveform model"):
        CpuWaveformScorer().score(np.zeros((1, 8192)))


def test_the_default_scorer_is_the_one_that_works():
    det = build_scorer()
    assert hasattr(det, "score")
    assert type(det).__name__ == "MahalanobisDetector"
    with pytest.raises(ValueError, match="unknown scorer kind"):
        build_scorer("tpu")


# -------------------------------------------------------------- the results --
def test_benchmark_results_are_committed_and_honest():
    path = os.path.join(_HERE, "..", "results", "benchmark.json")
    if not os.path.isfile(path):
        pytest.skip("benchmark.json not generated on this checkout")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["host"]["machine"], "a benchmark without its host is unquotable"
    assert "NOT a Raspberry Pi 5" in data["caveat"]
    # The finding: scoring is microseconds, so the NPU has no job on this path.
    assert float(data["scoring"]["single"]["median_ms"]) < 1.0
    # And compute is not the fan-out constraint.
    hourly = [f for f in data["fan_out"] if f["interval_s"] == 3600.0]
    assert hourly and hourly[0]["nodes"] > 10_000


def test_holdover_results_are_committed():
    path = os.path.join(_HERE, "..", "results", "holdover.json")
    if not os.path.isfile(path):
        pytest.skip("holdover.json not generated on this checkout")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["recommended_holdover_s"] == 60.0
    # 60 s at 7.5 W is well inside supercapacitor territory, which is the
    # recommendation's whole point: no battery chemistry in the enclosure.
    assert data["recommended_wh"] < 0.5
