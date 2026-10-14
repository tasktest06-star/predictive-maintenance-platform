#!/usr/bin/env python3
"""UPS holdover sizing: the one power requirement here that is a correctness
requirement.

    PYTHONPATH=core:edge python3 hardware/rpi5_hailo/tools/holdover.py

Every other kit's power analysis is about battery life -- how long before
someone has to visit. This one is different: a gateway that loses power loses
data that exists nowhere else, and the question is not "how long can it run?"
but "how long does it need to survive in order to lose nothing?"

Three distinct losses on a power cut, and they need different holdover:

1. **The cycle in progress.** Whatever the gateway was mid-write. SQLite with
   synchronous=FULL means committed frames survive; the one in flight does not.
   Holdover needed: milliseconds. Solved by the filesystem, not by a UPS.

2. **The nodes' in-flight data.** A BLE node that cannot reach its gateway
   queues to its own flash -- days of capacity -- so this is covered, PROVIDED
   the node's queue is deep enough for the outage. That is a node
   configuration, checked by each kit's test_kit.py.

3. **The gateway's own backlog.** This is the real one. If the site uplink has
   been down and the gateway holds months of frames, a power cut with no
   holdover does not lose them (they are on disk) -- but a power cut during the
   *flush*, or a disk that does not come back, does. And the flush itself takes
   time that has to be bounded.

So the number that matters is: **how long does it take to flush the backlog,
and can the UPS cover that?** Computed below from the measured frame size and
realistic uplink rates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))

#: 44-byte header + 63 float32 + CRC.
FRAME_BYTES = 300
#: Raw snapshot, for the escalation path.
SNAPSHOT_BYTES = 51_200

#: Realistic site uplinks, effective application throughput in bytes/s after
#: TLS, MQTT framing and typical contention. Deliberately pessimistic: a "4G"
#: link at a plant is not a lab 4G link.
UPLINKS = {
    "LTE Cat-M1 (shared)": 8_000,
    "4G LTE (plant, contended)": 250_000,
    "Wi-Fi to site LAN": 2_000_000,
    "Gigabit Ethernet": 40_000_000,
}

#: Pi 5 + AI HAT, measured-class figures from published Pi 5 power work.
PI5_IDLE_W = 2.7
PI5_GATEWAY_W = 7.5          # the compose stack plus the agent, sustained
PI5_PEAK_W = 12.0            # boot and burst, which sizes the supply not the UPS


def flush_time_s(frames: int, bytes_per_s: float,
                 frame_bytes: int = FRAME_BYTES) -> float:
    return frames * frame_bytes / bytes_per_s if bytes_per_s > 0 else float("inf")


def ups_energy_wh(seconds: float, watts: float) -> float:
    return watts * seconds / 3600.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=50)
    ap.add_argument("--interval-s", type=float, default=3600.0)
    ap.add_argument("--outage-days", type=float, default=14.0)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    args = ap.parse_args(argv)

    frames = int(args.nodes * args.outage_days * 86400.0 / args.interval_s)
    backlog_mb = frames * FRAME_BYTES / 1e6
    print(f"gateway holdover sizing")
    print(f"  {args.nodes} nodes at {args.interval_s / 60:.0f} min over a "
          f"{args.outage_days:.0f}-day uplink outage")
    print(f"  = {frames} frames = {backlog_mb:.1f} MB of feature frames")
    print(f"  (+ snapshots: each is {SNAPSHOT_BYTES / 1024:.0f} kB, so 100 of them "
          f"is another {100 * SNAPSHOT_BYTES / 1e6:.1f} MB)")
    print()
    print("  flush time when the uplink returns, and the UPS energy to cover it:")
    rows: List[Dict[str, object]] = []
    for name, bps in UPLINKS.items():
        t = flush_time_s(frames, bps)
        wh = ups_energy_wh(t, PI5_GATEWAY_W)
        rows.append({"uplink": name, "bytes_per_s": bps, "flush_s": t, "ups_wh": wh})
        unit = f"{t:7.1f} s" if t < 600 else f"{t / 60:7.1f} min"
        print(f"    {name:28s} {unit}   {wh:7.2f} Wh at {PI5_GATEWAY_W:.1f} W")

    # The practical recommendation. Note what sizes it: not the flush, which is
    # fast on any real uplink, but the graceful-shutdown window plus margin for
    # a brownout that recovers.
    print()
    print("  WHAT ACTUALLY SIZES THE UPS")
    print("  Not the flush. On anything above Cat-M1 the backlog clears in")
    print("  minutes, and the flush does not have to happen on battery at all --")
    print("  it happens when mains and uplink are both back.")
    print()
    print("  The UPS exists for two other things:")
    print("   1. A graceful shutdown: close SQLite cleanly, so the WAL is")
    print("      checkpointed and the next boot does not have to recover. 30 s")
    print("      is generous.")
    print("   2. Riding through brownouts and short cuts without a reboot. A Pi")
    print("      that reboots every time the plant's compressors start loses the")
    print("      cycle in progress each time, and SD-card corruption is a real")
    print("      risk on unclean power-down.")
    for seconds in (30.0, 60.0, 300.0, 900.0):
        wh = ups_energy_wh(seconds, PI5_GATEWAY_W)
        # A supercapacitor bank is realistic to ~60 s; beyond that it is a
        # LiFePO4 pack, which brings its own thermal and lifetime problems in a
        # plant enclosure.
        tech = "supercapacitor" if seconds <= 60 else "LiFePO4 pack"
        print(f"    {seconds:6.0f} s holdover  {wh:6.2f} Wh at "
              f"{PI5_GATEWAY_W:.1f} W   ({tech})")

    print()
    print("  RECOMMENDATION: 60 s of supercapacitor holdover plus a clean")
    print("  shutdown handler. Enough for (1) and most of (2), no battery")
    print("  chemistry in the enclosure, and no maintenance item. Longer")
    print("  holdover buys uptime, not data integrity -- the data is already on")
    print("  disk with synchronous=FULL.")
    print()
    print("  AND THE THING A UPS DOES NOT FIX: if the SD card dies on an unclean")
    print("  power-down, the backlog is gone regardless. An eMMC or SSD carrier")
    print("  is a bigger data-integrity win than any UPS, and it is cheaper.")

    results = {
        "nodes": args.nodes,
        "interval_s": args.interval_s,
        "outage_days": args.outage_days,
        "frames": frames,
        "backlog_mb": backlog_mb,
        "uplinks": rows,
        "gateway_w": PI5_GATEWAY_W,
        "recommended_holdover_s": 60.0,
        "recommended_wh": ups_energy_wh(60.0, PI5_GATEWAY_W),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "holdover.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
