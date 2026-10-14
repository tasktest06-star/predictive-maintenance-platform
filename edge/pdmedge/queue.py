"""Durable store-and-forward, and a non-obvious eviction policy.

An MCU queues frames into internal flash with no filesystem in the way. A Linux
gateway holding a site's data in a page cache loses it on a power cut, and a
gateway is exactly where an outage is most expensive: it may be holding
hundreds of nodes' data.

So: SQLite, with `synchronous=FULL` and WAL. The cost is one fsync per frame --
about 1 ms on an SD card, against a 300-byte write. At one frame per node per
hour that is irrelevant, and it is the difference between a queue and a cache.

THE EVICTION POLICY IS THE INTERESTING PART

A bounded queue has to drop something. The two obvious policies are both wrong
for condition monitoring:

- **Drop the newest** (refuse to enqueue). The node keeps its oldest history and
  loses the hours leading up to a failure -- which is the only data anybody will
  want afterwards.
- **Drop the oldest** (ring buffer). The node keeps recency and loses the
  baseline. A trend with its first half missing cannot establish what "normal"
  was, and the health index's operating-point baseline needs weeks.

Both ends matter and the middle does not, so this queue **decimates the
middle**: when it is full it deletes every other frame from the central
two-thirds, keeping the oldest and newest intact. The result is a queue whose
time coverage degrades in *resolution* rather than in *extent* -- an hourly
trend becomes two-hourly, then four-hourly, and the alarm-relevant recent
frames and the baseline-relevant old ones both survive.

That is a real trade and it is not free: a decimated region cannot be
un-decimated, and a fault whose signature appears in a single frame may be the
frame that was dropped. For a 300-byte feature frame at hourly cadence the
queue holds months before it ever decimates, which is why the policy is
acceptable; for raw snapshots it is not, and those are dropped oldest-first and
counted separately.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class QueueStats:
    depth: int
    bytes_used: int
    oldest_ms: Optional[int]
    newest_ms: Optional[int]
    pushed: int
    popped: int
    evicted: int

    @property
    def span_hours(self) -> float:
        if self.oldest_ms is None or self.newest_ms is None:
            return 0.0
        return (self.newest_ms - self.oldest_ms) / 3_600_000.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms       INTEGER NOT NULL,
    kind        TEXT    NOT NULL DEFAULT 'frame',
    payload     BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_ts ON frames(ts_ms);
CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
"""


class FrameQueue:
    """A durable, bounded FIFO of encoded frames.

    `max_frames` bounds the queue. On overflow, `kind='frame'` entries are
    decimated out of the middle (see the module docstring) and `kind='snapshot'`
    entries are dropped oldest-first, because a snapshot is 51 kB of raw
    waveform whose value is almost entirely in being recent.
    """

    SNAPSHOT_KIND = "snapshot"

    def __init__(self, path: str, max_frames: int = 100_000, durable: bool = True) -> None:
        self.path = path
        self.max_frames = int(max_frames)
        if self.max_frames < 16:
            raise ValueError("max_frames below 16 leaves no room to decimate")
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None, timeout=30.0)
        self._db.execute("PRAGMA journal_mode=WAL")
        # FULL, not NORMAL: with NORMAL, WAL frames are not fsynced on commit,
        # so a power cut can lose committed transactions -- which is precisely
        # the event this queue exists for. The cost is one fsync per push.
        self._db.execute(f"PRAGMA synchronous={'FULL' if durable else 'OFF'}")
        self._db.executescript(_SCHEMA)

    # -- counters ---------------------------------------------------------
    def _bump(self, name: str, by: int = 1) -> None:
        self._db.execute(
            "INSERT INTO counters(name, value) VALUES(?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = value + ?",
            (name, by, by),
        )

    def _counter(self, name: str) -> int:
        row = self._db.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()
        return int(row[0]) if row else 0

    # -- api --------------------------------------------------------------
    def push(self, payload: bytes, ts_ms: Optional[int] = None, kind: str = "frame") -> None:
        if not payload:
            raise ValueError("refusing to queue an empty payload")
        ts = int(ts_ms if ts_ms is not None else time.time() * 1000.0)
        self._db.execute(
            "INSERT INTO frames(ts_ms, kind, payload) VALUES(?, ?, ?)", (ts, kind, payload)
        )
        self._bump("pushed")
        if self.depth > self.max_frames:
            self._evict()

    def pop(self) -> Optional[Tuple[int, bytes]]:
        """Oldest frame as (id, payload), or None. Does not delete it.

        Two-phase on purpose: the firmware's equivalent pushes a frame back when
        the link drops mid-drain, and losing data because a transmission was
        *reported* sent is the failure this avoids. Call `ack` once the sink has
        accepted it.
        """
        row = self._db.execute(
            "SELECT id, payload FROM frames ORDER BY id LIMIT 1"
        ).fetchone()
        return (int(row[0]), bytes(row[1])) if row else None

    def peek_batch(self, limit: int) -> List[Tuple[int, bytes]]:
        rows = self._db.execute(
            "SELECT id, payload FROM frames ORDER BY id LIMIT ?", (int(limit),)
        ).fetchall()
        return [(int(r[0]), bytes(r[1])) for r in rows]

    def ack(self, ids: object) -> int:
        """Delete frames the sink has accepted. Accepts an id or an iterable."""
        if isinstance(ids, int):
            ids = [ids]
        id_list = [int(i) for i in ids]  # type: ignore[union-attr]
        if not id_list:
            return 0
        marks = ",".join("?" * len(id_list))
        cur = self._db.execute(f"DELETE FROM frames WHERE id IN ({marks})", id_list)
        n = int(cur.rowcount or 0)
        self._bump("popped", n)
        return n

    @property
    def depth(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM frames").fetchone()[0])

    def stats(self) -> QueueStats:
        row = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)),0), MIN(ts_ms), MAX(ts_ms) FROM frames"
        ).fetchone()
        return QueueStats(
            depth=int(row[0]),
            bytes_used=int(row[1]),
            oldest_ms=int(row[2]) if row[2] is not None else None,
            newest_ms=int(row[3]) if row[3] is not None else None,
            pushed=self._counter("pushed"),
            popped=self._counter("popped"),
            evicted=self._counter("evicted"),
        )

    def _evict(self) -> int:
        """Make room. Snapshots oldest-first, then decimate the middle."""
        removed = 0
        over = self.depth - self.max_frames
        if over <= 0:
            return 0

        # Snapshots first: 51 kB of raw waveform whose value is almost all in
        # being recent, so an old one is the cheapest thing to lose.
        cur = self._db.execute(
            "DELETE FROM frames WHERE id IN ("
            "  SELECT id FROM frames WHERE kind=? ORDER BY id LIMIT ?)",
            (self.SNAPSHOT_KIND, over),
        )
        removed += int(cur.rowcount or 0)
        over -= int(cur.rowcount or 0)
        if over <= 0:
            self._bump("evicted", removed)
            return removed

        # Then decimate the central two-thirds, keeping both ends. Deleting
        # every other row there frees ~1/3 of the queue in one pass, so this
        # runs rarely rather than on every push.
        row = self._db.execute("SELECT MIN(id), MAX(id) FROM frames").fetchone()
        if row is None or row[0] is None:
            return removed
        lo, hi = int(row[0]), int(row[1])
        span = hi - lo
        if span < 8:
            # Too small to have a middle; fall back to oldest-first so the
            # queue cannot wedge above its bound.
            cur = self._db.execute(
                "DELETE FROM frames WHERE id IN (SELECT id FROM frames ORDER BY id LIMIT ?)",
                (over,),
            )
            removed += int(cur.rowcount or 0)
            self._bump("evicted", removed)
            return removed

        keep_edge = max(1, span // 6)
        inner_lo, inner_hi = lo + keep_edge, hi - keep_edge
        cur = self._db.execute(
            "DELETE FROM frames WHERE id > ? AND id < ? AND (id - ?) % 2 = 1",
            (inner_lo, inner_hi, inner_lo),
        )
        removed += int(cur.rowcount or 0)
        self._bump("evicted", removed)

        # If one decimation pass was not enough -- which happens once the middle
        # is already sparse -- drop oldest-first for the remainder. Bounded, and
        # it preserves the recent end, which is the one the alarm logic needs.
        still_over = self.depth - self.max_frames
        if still_over > 0:
            cur = self._db.execute(
                "DELETE FROM frames WHERE id IN (SELECT id FROM frames ORDER BY id LIMIT ?)",
                (still_over,),
            )
            n = int(cur.rowcount or 0)
            removed += n
            self._bump("evicted", n)
        return removed

    def vacuum(self) -> None:
        """Reclaim file space. Not automatic: VACUUM rewrites the whole file,
        which on an SD card is both slow and a write-endurance event."""
        self._db.execute("VACUUM")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "FrameQueue":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
