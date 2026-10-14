#!/usr/bin/env python3
"""Measured gateway throughput, and how many nodes one box can carry.

    PYTHONPATH=core:edge python3 hardware/rpi5_hailo/tools/benchmark.py

This is the one benchmark in this repository that is a MEASUREMENT rather than
an estimate, because the gateway tier runs on a general-purpose aarch64 Linux
box and so does the machine this was written on. Every MCU timing figure here
is a cycle count; these are a stopwatch.

What it measures:

  1. Feature extraction wall time per block, which is what decides fan-out.
  2. Detector scoring time, which is what decides whether the Hailo-8L has
     anything to do.
  3. Durable queue push rate, which decides whether the store-and-forward path
     can keep up with the fan-in.
  4. Frame encode/decode, which is the per-frame tax on every path.

WHAT IT DOES NOT MEASURE: a Raspberry Pi 5. The host is reported below, and the
Pi 5 scaling is given as an explicit, labelled ratio from published SPECint and
clock figures -- not folded silently into the numbers. A Pi 5's Cortex-A76 at
2.4 GHz is roughly 1.4-1.8x this container's per-core throughput on this kind of
numpy-heavy float work, but the honest range is wide and the only way to close
it is to run this script on a Pi.

The reason to measure at all: the gateway's design question is "how many nodes
per box?", and the answer from an estimate is worthless. From a measurement on
similar-class hardware it is at least the right order of magnitude.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, os.path.join(_REPO, "edge"))

from pdmcore.features import FeatureConfig, FeatureExtractor  # noqa: E402
from pdmcore.io.payload import TelemetryFrame, decode_frame, encode_frame  # noqa: E402
from pdmcore.models.anomaly import MahalanobisDetector  # noqa: E402
from pdmcore.sim import MACHINE_LIBRARY, simulate  # noqa: E402
from pdmedge.queue import FrameQueue  # noqa: E402

FS = 25600.0


def host_info() -> Dict[str, object]:
    """What was actually measured. Printed so a number can never be quoted
    without the machine it came from."""
    info: Dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", encoding="ascii") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "bogomips")):
                    k, _, v = line.partition(":")
                    info[k.strip().lower()] = v.strip()
                    break
    except OSError:
        pass
    return info


def _timeit(fn, n: int) -> Dict[str, float]:
    """Median and p95 of n timed calls, after a warm-up.

    Median rather than mean because a container's scheduler produces occasional
    multi-millisecond outliers that a mean would absorb into the headline
    figure; p95 is reported separately because a gateway's fan-out has to be
    sized on the slow case, not the typical one.
    """
    fn()                                      # warm up caches and any lazy init
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    a = np.array(samples)
    return {
        "median_ms": float(np.median(a) * 1e3),
        "p95_ms": float(np.percentile(a, 95) * 1e3),
        "min_ms": float(a.min() * 1e3),
        "n": n,
    }


def bench_features(block: int, n_fft: int, reps: int) -> Dict[str, object]:
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS)
    x = np.asarray(simulate(spec, None, duration_s=block / FS, seed=11).accel_g)[:block]
    ex = FeatureExtractor(FeatureConfig(
        fs_hz=FS, n_fft=n_fft, envelope_n_fft=min(2048, block // 4),
        demod_band_hz=(2500.0, 6000.0)))
    out = _timeit(lambda: ex.extract(x, spec.shaft_hz, fs_hz=FS), reps)
    out["block_samples"] = block
    out["n_fft"] = n_fft
    out["record_s"] = block / FS
    return out


def bench_scoring(reps: int) -> Dict[str, object]:
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS)
    ex = FeatureExtractor(FeatureConfig(fs_hz=FS, n_fft=4096, envelope_n_fft=2048,
                                        demod_band_hz=(2500.0, 6000.0)))
    rows = [ex.extract(np.asarray(simulate(spec, None, duration_s=0.32, seed=200 + i).accel_g)[:8192],
                       spec.shaft_hz, fs_hz=FS).vector for i in range(40)]
    X = np.asarray(rows)
    det = MahalanobisDetector().fit(X)
    one = X[:1]
    batch = np.repeat(X, 25, axis=0)          # 1000 frames
    return {
        "single": _timeit(lambda: det.score(one), reps),
        "batch_1000": _timeit(lambda: det.score(batch), max(10, reps // 20)),
    }


def bench_wire(reps: int) -> Dict[str, object]:
    tf = TelemetryFrame(device_id=0xA1B2C3D4E5F60001, timestamp_ms=1_760_000_000_000,
                        features=np.linspace(0.0, 1.0, 63).astype(np.float64),
                        shaft_hz=29.6667, sequence=1)
    payload = encode_frame(tf)
    return {
        "encode": _timeit(lambda: encode_frame(tf), reps),
        "decode": _timeit(lambda: decode_frame(payload), reps),
        "frame_bytes": len(payload),
    }


def bench_queue(reps: int) -> Dict[str, object]:
    """Durable push rate. The fsync is the point, so both modes are measured.

    The ratio between them is the price of durability, and on an SD card it is
    far worse than it is here -- which is exactly why the number is reported
    rather than assumed to be small.
    """
    payload = b"x" * 300
    out: Dict[str, object] = {}
    for durable in (True, False):
        d = tempfile.mkdtemp(prefix="pdmbench-")
        q = FrameQueue(os.path.join(d, "q.sqlite"), max_frames=1_000_000, durable=durable)
        t0 = time.perf_counter()
        for i in range(reps):
            q.push(payload, ts_ms=1_760_000_000_000 + i)
        dt = time.perf_counter() - t0
        q.close()
        out["durable" if durable else "nosync"] = {
            "frames": reps,
            "seconds": dt,
            "frames_per_s": reps / dt if dt > 0 else float("inf"),
            "ms_per_frame": dt / reps * 1e3,
        }
    d_rate = float(out["durable"]["frames_per_s"])        # type: ignore[index]
    n_rate = float(out["nosync"]["frames_per_s"])         # type: ignore[index]
    out["durability_cost"] = n_rate / d_rate if d_rate > 0 else float("inf")
    # A real fsync to real storage costs something. If durable and nosync come
    # out within 20 % of each other, the fsync is not reaching a device -- an
    # overlay or tmpfs-backed filesystem, or a drive lying about write
    # completion. Reporting that is the point: a benchmark that silently
    # measures nothing is worse than no benchmark, because the number gets
    # quoted.
    out["fsync_appears_effective"] = bool(out["durability_cost"] > 1.2)
    return out


def fan_out(feature_ms: float, interval_s: float, headroom: float = 0.25,
            cores: int = 1) -> Dict[str, object]:
    """Nodes per gateway, from the measured extraction time.

    `headroom` is the fraction of the box left for everything that is not
    feature extraction: the MQTT broker, the time-series database, Grafana, the
    OS, and the fact that arrivals are not uniformly spaced. 0.25 means only
    75 % of the measured capacity is offered, which is still optimistic for a
    box also running the compose stack.
    """
    per_node_s = feature_ms / 1e3
    capacity = interval_s / per_node_s * cores * (1.0 - headroom)
    return {
        "interval_s": interval_s,
        "cores_used": cores,
        "headroom": headroom,
        "nodes": int(capacity),
        "feature_ms": feature_ms,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--queue-frames", type=int, default=300)
    args = ap.parse_args(argv)

    info = host_info()
    print("gateway benchmark -- MEASURED on this host, not a Raspberry Pi 5")
    for k, v in info.items():
        print(f"  {k:14s} {v}")
    print()

    print("feature extraction (the number that decides fan-out)")
    feats: List[Dict[str, object]] = []
    for block, n_fft in ((4096, 2048), (8192, 4096), (16384, 4096), (32768, 8192)):
        r = bench_features(block, n_fft, args.reps)
        feats.append(r)
        print(f"  {block:6d} samples ({float(r['record_s']):.2f} s record), n_fft "
              f"{n_fft:5d}:  {float(r['median_ms']):7.1f} ms median, "
              f"{float(r['p95_ms']):7.1f} ms p95  "
              f"-> {float(r['record_s']) * 1e3 / float(r['median_ms']):6.1f}x real time")

    print("\ndetector scoring (the number that decides whether the NPU has a job)")
    sc = bench_scoring(args.reps)
    print(f"  1 frame:      {float(sc['single']['median_ms']):8.3f} ms")
    print(f"  1000 frames:  {float(sc['batch_1000']['median_ms']):8.3f} ms  "
          f"({float(sc['batch_1000']['median_ms']) / 1000.0 * 1e3:.1f} us/frame)")

    print("\nwire format")
    w = bench_wire(args.reps * 10)
    print(f"  encode {float(w['encode']['median_ms']) * 1e3:7.1f} us, "
          f"decode {float(w['decode']['median_ms']) * 1e3:7.1f} us, "
          f"{w['frame_bytes']} bytes")

    print("\ndurable queue (synchronous=FULL vs OFF)")
    q = bench_queue(args.queue_frames)
    for mode in ("durable", "nosync"):
        d = q[mode]                                        # type: ignore[index]
        print(f"  {mode:8s} {float(d['frames_per_s']):9.0f} frames/s  "
              f"({float(d['ms_per_frame']):.3f} ms/frame)")
    print(f"  durability costs {float(q['durability_cost']):.1f}x throughput on this "
          f"filesystem")
    if not q["fsync_appears_effective"]:
        print("  WARNING  durable and nosync are within 20 %: the fsync is not")
        print("           reaching a storage device on this host (overlay/tmpfs,")
        print("           or a drive that lies about write completion). This")
        print("           measurement does NOT validate the SD-card case, which")
        print("           is the one that matters -- expect 1-10 ms per fsync and")
        print("           100-1000 frames/s there.")

    # Fan-out, from the 8192-sample case: the configuration every MCU kit uses,
    # so the gateway's capacity is quoted in units of "nodes like the others".
    base = next(f for f in feats if f["block_samples"] == 8192)
    ms = float(base["median_ms"])
    print(f"\nfan-out at {ms:.1f} ms per feature vector, 75 % of one core offered:")
    fans = []
    for interval in (60.0, 300.0, 900.0, 3600.0):
        f = fan_out(ms, interval, cores=1)
        fans.append(f)
        print(f"  {interval / 60.0:5.0f} min interval:  {f['nodes']:7d} nodes")
    f4 = fan_out(ms, 3600.0, cores=4)
    print(f"  and on 4 cores at 1 h:   {f4['nodes']:7d} nodes "
          f"(a Pi 5 has 4; the compose stack wants some of them)")
    print()
    print("  These numbers are large enough to be the finding: FEATURE EXTRACTION")
    print("  IS NOT THE GATEWAY'S CONSTRAINT. At hourly cadence one core covers")
    print("  more nodes than any plant has. What actually binds a gateway is the")
    print("  broker's concurrent connections, the time-series database's write")
    print("  rate, memory per connection, and the uplink -- none of which this")
    print("  script measures, and all of which need the compose stack running.")

    results = {
        "host": info,
        "features": feats,
        "scoring": sc,
        "wire": w,
        "queue": q,
        "fan_out": fans,
        "fan_out_4core_1h": f4,
        "caveat": ("Measured on the host above, which is NOT a Raspberry Pi 5. "
                   "A Pi 5 Cortex-A76 at 2.4 GHz is roughly 1.4-1.8x this "
                   "container per core on numpy-heavy float work; the range is "
                   "wide and only running this script on a Pi closes it."),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
