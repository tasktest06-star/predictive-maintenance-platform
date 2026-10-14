"""`python -m pdmedge` -- run the agent from a config file.

    python -m pdmedge --config /etc/pdm/agent.json
    python -m pdmedge --config /etc/pdm/agent.json --cycles 3 --stdout
    python -m pdmedge --self-test

`--self-test` runs the whole loop against the simulator with a temporary queue
and prints what each stage produced. It is the first thing to run on a new
board: it proves the Python stack, the DSP, the wire format and the queue work
before any sensor is attached, so a later failure is unambiguously the sensor.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "core")))

from .agent import EdgeAgent, HealthGate            # noqa: E402
from .calibration import Calibration                # noqa: E402
from .config import AgentConfig, ChannelConfig, load_config  # noqa: E402
from .queue import FrameQueue                       # noqa: E402
from .sink import MqttSink, Sink, StdoutSink        # noqa: E402
from .source import FileSource, IIOSource, SimSource, Source  # noqa: E402

log = logging.getLogger("pdmedge")


def build_source(cfg: AgentConfig, spec: dict) -> Source:
    kind = str(spec.get("kind", "sim"))
    if kind == "sim":
        return SimSource(machine=str(spec.get("machine", "motor_pump_37kw")),
                         fs_hz=cfg.fs_hz, seed=int(spec.get("seed", 0)))
    if kind == "file":
        return FileSource(path=str(spec["path"]), fs_hz=cfg.fs_hz)
    if kind == "iio":
        return IIOSource(uri=str(spec.get("uri", "ip:analog.local")),
                         device=str(spec.get("device", "ad7768-1")),
                         channels=tuple(spec.get("channels", ("voltage0",))),
                         fs_hz=cfg.fs_hz)
    raise SystemExit(f"unknown source kind {kind!r}: use sim, file or iio")


def build_sink(spec: dict, force_stdout: bool = False) -> Sink:
    if force_stdout or str(spec.get("kind", "stdout")) == "stdout":
        return StdoutSink()
    kind = str(spec.get("kind"))
    if kind == "mqtt":
        return MqttSink(
            host=str(spec.get("host", "localhost")),
            port=int(spec.get("port", 8883)),
            topic=str(spec.get("topic", "pdm/v1/raw")),
            client_id=str(spec.get("client_id", "pdm-edge")),
            tls=bool(spec.get("tls", True)),
            ca_cert=spec.get("ca_cert"),
            certfile=spec.get("certfile"),
            keyfile=spec.get("keyfile"),
            username=spec.get("username"),
            password=spec.get("password"),
        )
    raise SystemExit(f"unknown sink kind {kind!r}: use stdout or mqtt")


def self_test() -> int:
    """Prove the stack without a sensor."""
    import io

    tmp = tempfile.mkdtemp(prefix="pdmedge-selftest-")
    cfg = AgentConfig(
        device_id=0x00A1B2C3D4E5F600, fs_hz=25600.0, block_samples=8192,
        interval_s=3600.0, queue_path=os.path.join(tmp, "q.sqlite"),
        queue_max_frames=1000,
        channels=[ChannelConfig(name="accel", shaft_hz=29.6667, units_are_g=True)],
    )
    buf = io.BytesIO()
    agent = EdgeAgent(cfg, SimSource(fs_hz=cfg.fs_hz), StdoutSink(buf),
                      queue=FrameQueue(cfg.queue_path, cfg.queue_max_frames))
    print("pdmedge self-test")
    gate_ok, flags, reason = agent.gate.check()
    print(f"  gate        {'ok' if gate_ok else 'BLOCKED'}  "
          f"free {agent.gate.free_mb():.0f} MB  "
          f"temp {agent.gate.temperature_c():.1f} C  {reason}")
    results = agent.run_cycle()
    for r in results:
        print(f"  channel     {r.channel or '-'}  ok={r.ok}  published={r.published}  "
              f"seq={r.sequence}  flags={r.flags!r}")
        if r.ok:
            print(f"  features    velocity {r.velocity_rms_mm_s:.3f} mm/s, ISO zone {r.iso_zone}")
        elif r.error:
            print(f"  error       {r.error}")
    payload = buf.getvalue()
    print(f"  wire        {len(payload)} bytes written "
          f"({len(payload) - 2} frame + 2 length prefix)")

    from pdmcore.io.payload import decode_frame

    frame = decode_frame(payload[2:])
    print(f"  decoded     {frame.n_features} features, schema {frame.schema_version}, "
          f"spec {frame.feature_spec_version}, scoreable={frame.scoreable}")
    print(f"  queue       {agent.queue.stats()}")
    ok = bool(results and results[0].ok and frame.n_features == 63)
    print(f"\n{'PASS' if ok else 'FAIL'}  the Python stack, DSP, wire format and queue work.")
    print("      Nothing here involves a sensor: a later failure is the sensor,")
    print("      its wiring, or its calibration -- not this.")
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON or TOML agent config")
    ap.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    ap.add_argument("--stdout", action="store_true",
                    help="override the configured sink with length-prefixed stdout")
    ap.add_argument("--no-sleep", action="store_true", help="do not wait between cycles")
    ap.add_argument("--self-test", action="store_true", help="run without a sensor and exit")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
                        stream=sys.stderr)

    if args.self_test:
        return self_test()
    if not args.config:
        ap.error("--config is required unless --self-test is given")

    cfg = load_config(args.config)
    with open(args.config, "rb") as fh:
        raw = json.loads(fh.read().decode("utf-8")) if args.config.endswith(".json") else {}
    source = build_source(cfg, dict(raw.get("source", {})))
    sink = build_sink(cfg.sink, force_stdout=args.stdout)

    cal = Calibration.load(cfg.calibration_path) if cfg.calibration_path else None
    if cal is not None:
        print(cal.report(bandwidth_hz=cfg.fs_hz / 2.0), file=sys.stderr)

    queue = FrameQueue(cfg.queue_path, cfg.queue_max_frames)
    agent = EdgeAgent(cfg, source, sink, queue=queue, calibration=cal,
                      gate=HealthGate(cfg.queue_path))

    stop = {"now": False}

    def _handle(signum: int, _frame: object) -> None:
        # Graceful: finish the cycle in progress so a frame is not half-written
        # and the queue is not left mid-transaction.
        log.info("signal %d: stopping after this cycle", signum)
        stop["now"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    source.open()
    sink.connect()
    try:
        n = 0
        import time as _time

        while not stop["now"] and (args.cycles is None or n < args.cycles):
            started = _time.monotonic()
            agent.run_cycle()
            n += 1
            if stop["now"] or (args.cycles is not None and n >= args.cycles):
                break
            if not args.no_sleep:
                left = cfg.interval_s - (_time.monotonic() - started)
                while left > 0 and not stop["now"]:
                    _time.sleep(min(1.0, left))
                    left -= 1.0
    finally:
        sink.close()
        source.close()
        queue.close()
        log.info("stats: %s", agent.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
