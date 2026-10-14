"""Tests for the durable store-and-forward queue.

The eviction policy is the part worth testing: it is non-obvious, it is a real
trade, and the obvious alternatives are both wrong for condition monitoring.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from pdmedge.queue import FrameQueue  # noqa: E402


def _q(tmp_path, max_frames=64, durable=True):
    return FrameQueue(str(tmp_path / "q.sqlite"), max_frames=max_frames, durable=durable)


def test_push_pop_ack(tmp_path):
    q = _q(tmp_path)
    q.push(b"one", ts_ms=1000)
    q.push(b"two", ts_ms=2000)
    assert q.depth == 2
    got = q.pop()
    assert got is not None and got[1] == b"one"
    assert q.depth == 2, "pop must not delete: the sink has not accepted it yet"
    assert q.ack(got[0]) == 1
    assert q.depth == 1


def test_pop_is_two_phase(tmp_path):
    """A frame must not leave the queue because a publish was *attempted*.

    The firmware pushes a frame back when the link drops mid-drain. The Python
    equivalent is to not delete until the sink has accepted, which is what the
    separate `ack` is for.
    """
    q = _q(tmp_path)
    q.push(b"payload")
    for _ in range(3):
        assert q.pop()[1] == b"payload"
    assert q.depth == 1


def test_fifo_order_is_preserved(tmp_path):
    q = _q(tmp_path)
    for i in range(10):
        q.push(f"f{i}".encode(), ts_ms=1000 + i)
    batch = q.peek_batch(10)
    assert [p for _, p in batch] == [f"f{i}".encode() for i in range(10)]


def test_empty_payload_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _q(tmp_path).push(b"")


def test_survives_reopen(tmp_path):
    """The whole reason this is SQLite and not a list."""
    path = str(tmp_path / "q.sqlite")
    q = FrameQueue(path, max_frames=64)
    q.push(b"durable", ts_ms=42)
    q.close()
    q2 = FrameQueue(path, max_frames=64)
    assert q2.depth == 1
    assert q2.pop()[1] == b"durable"
    assert q2.stats().pushed == 1, "counters must survive too"


def test_eviction_keeps_both_ends(tmp_path):
    """The policy: degrade resolution, not extent.

    Dropping the newest loses the hours before a failure -- the only data
    anybody wants afterwards. Dropping the oldest loses the baseline, and the
    health index's operating-point baseline needs weeks. So the middle is
    decimated and both ends survive.
    """
    q = _q(tmp_path, max_frames=32)
    for i in range(80):
        q.push(f"{i:03d}".encode(), ts_ms=1000 + i)

    assert q.depth <= 32
    kept = [int(p) for _, p in q.peek_batch(100)]
    assert kept == sorted(kept), "order must survive eviction"
    # The oldest and newest frames are both still there.
    assert kept[0] <= 4, f"oldest frames were dropped: queue starts at {kept[0]}"
    assert kept[-1] >= 75, f"newest frames were dropped: queue ends at {kept[-1]}"
    # And the coverage is wide, not a contiguous recent window.
    assert kept[-1] - kept[0] >= 60


def test_eviction_is_counted(tmp_path):
    q = _q(tmp_path, max_frames=32)
    for i in range(80):
        q.push(f"{i:03d}".encode(), ts_ms=1000 + i)
    s = q.stats()
    assert s.pushed == 80
    assert s.evicted > 0
    assert s.depth + s.evicted + s.popped == s.pushed


def test_snapshots_are_dropped_before_frames(tmp_path):
    """A 51 kB raw snapshot's value is almost entirely in being recent, so an
    old one is the cheapest thing in the queue to lose."""
    q = _q(tmp_path, max_frames=32)
    for i in range(20):
        q.push(f"snap{i}".encode(), ts_ms=1000 + i, kind=FrameQueue.SNAPSHOT_KIND)
    for i in range(20):
        q.push(f"frame{i}".encode(), ts_ms=2000 + i)

    kept = [p for _, p in q.peek_batch(100)]
    snaps = [p for p in kept if p.startswith(b"snap")]
    frames = [p for p in kept if p.startswith(b"frame")]
    assert len(frames) == 20, "no frame should have been evicted while snapshots remained"
    assert len(snaps) < 20


def test_queue_never_exceeds_its_bound(tmp_path):
    """Including the pathological case where the middle is already sparse."""
    q = _q(tmp_path, max_frames=16)
    for i in range(500):
        q.push(f"{i:04d}".encode(), ts_ms=1000 + i)
        assert q.depth <= 16, f"depth {q.depth} exceeded the bound at push {i}"


def test_tiny_bound_is_refused(tmp_path):
    with pytest.raises(ValueError, match="decimate"):
        FrameQueue(str(tmp_path / "q.sqlite"), max_frames=4)


def test_stats_span(tmp_path):
    q = _q(tmp_path)
    assert q.stats().span_hours == 0.0
    q.push(b"a", ts_ms=0)
    q.push(b"b", ts_ms=7_200_000)
    assert q.stats().span_hours == pytest.approx(2.0)
