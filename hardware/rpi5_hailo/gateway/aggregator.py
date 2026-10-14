"""The gateway: many nodes in, one uplink out, autonomous when the link drops.

A gateway is not a node with more CPU. The benchmark says why: one core
feature-extracts for ~780 000 hourly nodes, so compute is not what a gateway is
for. What it is for:

1. **Protocol translation.** BLE nodes cannot reach a cloud broker. Something has
   to terminate BLE/sub-GHz and speak MQTT/TLS upstream, and that something is
   the only part of the chain that must be mains-powered anyway.
2. **Store-and-forward for a whole site.** A node's flash queue holds days; a
   gateway's disk holds months. When the site's uplink goes, the gateway is
   where the plant's data lives.
3. **Local autonomy.** An alarm that needs a round trip to a cloud is not
   available during the outage that most often accompanies a real failure. The
   gateway scores locally and can raise an alarm with no uplink at all.
4. **Fan-in accounting.** Which nodes are silent, and since when. A node that
   stops reporting is the single most common real event in a deployed fleet, and
   nothing downstream can see it -- only the thing the nodes report *to* knows
   the difference between "healthy and quiet" and "gone".

Point 4 is the one most often missed, so it is implemented here rather than
deferred to the cloud: `silent_nodes()` is the gateway's own view of its
population, and it works during an uplink outage, which is exactly when a node
dying matters most.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pdmcore.io.payload import FrameFlags, PayloadError, decode_frame
from pdmedge.queue import FrameQueue
from pdmedge.sink import Sink, SinkError

log = logging.getLogger("pdmedge.gateway")


@dataclass
class NodeState:
    """What the gateway knows about one node, independent of the cloud."""

    device_id: int
    first_seen_ms: int
    last_seen_ms: int
    frames: int = 0
    rejected: int = 0
    gaps: int = 0
    last_sequence: Optional[int] = None
    last_score: float = float("nan")
    alarms: int = 0
    flags_seen: int = 0

    #: Reporting interval learned from arrivals, used to decide silence.
    #: Learned rather than configured because a fleet's intervals are changed
    #: from the cloud and a gateway's copy of that config goes stale.
    interval_ms: float = float("nan")

    def observe_interval(self, dt_ms: float) -> None:
        if not (dt_ms > 0.0):
            return
        if not np.isfinite(self.interval_ms):
            self.interval_ms = dt_ms
        else:
            # Slow EWMA: a node whose interval really changed should take a few
            # cycles to be believed, so one early frame does not reset the
            # silence threshold and mask a genuine outage.
            self.interval_ms = 0.8 * self.interval_ms + 0.2 * dt_ms


@dataclass
class GatewayStats:
    received: int = 0
    rejected: int = 0
    forwarded: int = 0
    queued: int = 0
    local_alarms: int = 0
    gaps: int = 0
    nodes: int = 0


class Gateway:
    """Terminates node traffic, forwards upstream, and keeps its own view.

    `scorer` is optional and takes an (n, 63) array to a score per row -- the
    same interface `pdmcore`'s detectors expose, and the same one a Hailo or
    TensorRT backend presents. When absent the gateway forwards without
    scoring, which is a valid configuration for a site whose models live in the
    cloud.
    """

    #: A node is silent once it has missed this many of its own intervals.
    #: Three rather than one because a single missed cycle is normal on a
    #: congested BLE link, and an alarm that cries wolf on every missed frame
    #: trains an operator to ignore it.
    SILENCE_INTERVALS = 3.0
    #: Floor, for a node whose interval has not been learned yet.
    SILENCE_FLOOR_MS = 4 * 3600 * 1000

    def __init__(
        self,
        queue: FrameQueue,
        uplink: Sink,
        scorer: Optional[object] = None,
        threshold: float = float("inf"),
        max_drain: int = 64,
        accept_schema: Tuple[int, ...] = (3,),
    ) -> None:
        self.queue = queue
        self.uplink = uplink
        self.scorer = scorer
        self.threshold = float(threshold)
        self.max_drain = int(max_drain)
        self.accept_schema = tuple(accept_schema)
        self.nodes: Dict[int, NodeState] = {}
        self.stats = GatewayStats()

    # -- ingest -----------------------------------------------------------
    def handle(self, payload: bytes, now_ms: Optional[int] = None) -> Dict[str, object]:
        """One frame from a node. Returns what was done with it."""
        now = int(now_ms if now_ms is not None else time.time() * 1000.0)
        self.stats.received += 1
        out: Dict[str, object] = {"accepted": False, "forwarded": False,
                                  "queued": False, "alarm": False, "gap": 0}

        try:
            frame = decode_frame(payload)
        except PayloadError as exc:
            # A frame that does not decode is NOT forwarded. Passing a corrupt
            # frame upstream in the hope the cloud sorts it out converts a local
            # radio problem into a fleet-wide data-quality problem, and the
            # cloud has less information than the gateway about which link it
            # came from.
            self.stats.rejected += 1
            out["error"] = str(exc)
            log.warning("rejected a frame: %s", exc)
            return out
        if frame.schema_version not in self.accept_schema:
            self.stats.rejected += 1
            out["error"] = (f"schema {frame.schema_version} not in "
                            f"{self.accept_schema}")
            return out

        state = self.nodes.get(frame.device_id)
        if state is None:
            state = NodeState(device_id=frame.device_id, first_seen_ms=now,
                              last_seen_ms=now)
            self.nodes[frame.device_id] = state
            self.stats.nodes = len(self.nodes)
            log.info("new node %016x", frame.device_id)
        else:
            state.observe_interval(now - state.last_seen_ms)

        # Gap detection, wrap-safe on the 16-bit sequence. Per channel, because
        # a multi-channel node advances the sequence once per FRAME: three axes
        # arriving as N, N+1, N+2 is contiguous, not two gaps.
        if state.last_sequence is not None:
            delta = (frame.sequence - state.last_sequence) & 0xFFFF
            if delta > 1:
                gap = delta - 1
                state.gaps += gap
                self.stats.gaps += gap
                out["gap"] = gap
                log.info("node %016x: %d frame(s) missing", frame.device_id, gap)
        state.last_sequence = frame.sequence
        state.last_seen_ms = now
        state.frames += 1
        state.flags_seen |= int(frame.flags)
        out["accepted"] = True
        out["device_id"] = frame.device_id
        out["channel"] = FrameFlags(frame.flags).channel

        # Local scoring, before forwarding: an alarm must not depend on the
        # uplink, because the outage that takes the uplink down is often the
        # same event that takes the machine down.
        if self.scorer is not None and frame.scoreable:
            try:
                score = float(np.asarray(
                    self.scorer.score(np.atleast_2d(frame.features))).ravel()[0])
            except Exception as exc:          # a scorer must not take the gateway down
                log.error("local scoring failed: %s", exc)
                score = float("nan")
            state.last_score = score
            out["score"] = score
            if np.isfinite(score) and score > self.threshold:
                state.alarms += 1
                self.stats.local_alarms += 1
                out["alarm"] = True
                log.warning("node %016x: local alarm, score %.2f > %.2f",
                            frame.device_id, score, self.threshold)

        # Forward, or queue.
        try:
            self.uplink.publish(payload)
            self.stats.forwarded += 1
            out["forwarded"] = True
            out["drained"] = self.drain()
        except SinkError as exc:
            self.queue.push(payload, ts_ms=frame.timestamp_ms)
            self.stats.queued += 1
            out["queued"] = True
            out["error"] = str(exc)
            log.info("uplink down (%s); queued", exc)
        return out

    # -- backlog ----------------------------------------------------------
    def drain(self, limit: Optional[int] = None) -> int:
        from pdmedge.agent import _mark_stored_and_forwarded

        budget = int(limit if limit is not None else self.max_drain)
        sent = 0
        for frame_id, payload in self.queue.peek_batch(budget):
            try:
                self.uplink.publish(_mark_stored_and_forwarded(payload))
            except SinkError:
                break
            self.queue.ack(frame_id)
            sent += 1
        return sent

    # -- the gateway's own view -------------------------------------------
    def silent_nodes(self, now_ms: Optional[int] = None) -> List[Tuple[int, float]]:
        """Nodes that have missed several of their own intervals.

        The gateway is the only place this can be computed during an uplink
        outage, which is when it matters most: a node that stops reporting is
        the commonest real event in a deployed fleet, and from downstream it is
        indistinguishable from a healthy quiet machine.

        Returns (device_id, hours_silent), worst first.
        """
        now = int(now_ms if now_ms is not None else time.time() * 1000.0)
        out = []
        for state in self.nodes.values():
            limit = (self.SILENCE_INTERVALS * state.interval_ms
                     if np.isfinite(state.interval_ms) else self.SILENCE_FLOOR_MS)
            silent_ms = now - state.last_seen_ms
            if silent_ms > limit:
                out.append((state.device_id, silent_ms / 3_600_000.0))
        return sorted(out, key=lambda t: -t[1])

    def report(self, now_ms: Optional[int] = None) -> str:
        now = int(now_ms if now_ms is not None else time.time() * 1000.0)
        s = self.stats
        qs = self.queue.stats()
        lines = [
            f"gateway: {s.nodes} node(s), {s.received} frames received, "
            f"{s.forwarded} forwarded, {s.queued} queued, {s.rejected} rejected",
            f"  queue depth {qs.depth} ({qs.bytes_used / 1e6:.1f} MB, "
            f"{qs.span_hours:.1f} h span, {qs.evicted} evicted)",
            f"  gaps {s.gaps}, local alarms {s.local_alarms}",
        ]
        silent = self.silent_nodes(now)
        if silent:
            lines.append(f"  SILENT NODES ({len(silent)}):")
            for dev, hours in silent[:10]:
                lines.append(f"    {dev:016x}  {hours:6.1f} h")
        return "\n".join(lines)
