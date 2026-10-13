"""``pdm`` command-line interface.

Exists so that the whole pipeline can be demonstrated and debugged without
writing Python, and so each hardware branch's README can show a copy-pasteable
command that actually runs.

    pdm simulate --fault bearing_inner --severity 0.4 --out run.npz
    pdm features run.npz
    pdm diagnose run.npz
    pdm demo --machine gearbox_132kw
    pdm payload --n-features 63
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

import numpy as np

from . import FEATURE_SPEC_VERSION, PAYLOAD_SCHEMA_VERSION, __version__


def _load(path: str):
    data = np.load(path, allow_pickle=False)
    return data["accel_g"], float(data["fs_hz"]), float(data["shaft_hz"])


def _machine(name: str):
    from .sim import MACHINE_LIBRARY

    if name not in MACHINE_LIBRARY:
        raise SystemExit(f"unknown machine {name!r}; choose from {', '.join(MACHINE_LIBRARY)}")
    return MACHINE_LIBRARY[name]


def _extractor(spec, demod: Optional[Sequence[float]] = None):
    from .features import FeatureConfig, FeatureExtractor

    return FeatureExtractor(
        FeatureConfig(
            fs_hz=spec.fs_hz,
            bearing=spec.bearing,
            demod_band_hz=tuple(demod) if demod else (2500.0, 6000.0),
            velocity_band_hz=(2.0, 1000.0) if spec.shaft_hz < 10 else (10.0, 1000.0),
        )
    )


# ---------------------------------------------------------------------------
def cmd_simulate(args: argparse.Namespace) -> int:
    from .sim import FAULT_MODES, simulate

    spec = _machine(args.machine)
    if args.fault and args.fault not in FAULT_MODES:
        raise SystemExit(f"unknown fault {args.fault!r}; choose from {', '.join(FAULT_MODES)}")
    faults = {args.fault: args.severity} if args.fault and args.fault != "healthy" else None
    res = simulate(spec, faults, duration_s=args.duration, seed=args.seed, with_current=args.current)
    print(f"machine        {spec.name}  {spec.rpm:.0f} rpm ({spec.shaft_hz:.2f} Hz)")
    print(f"fault          {res.label} severity={res.severity:.2f}")
    print(f"acquisition    {res.accel_g.size} samples @ {res.fs_hz:.0f} Hz = {res.duration_s:.2f} s")
    print(f"amplitude      rms={np.sqrt(np.mean(res.accel_g**2)):.4f} g  "
          f"peak={np.max(np.abs(res.accel_g)):.4f} g  temp={res.temperature_c:.1f} C")
    print("truth          " + "  ".join(f"{k}={v:.2f}" for k, v in res.truth.items()
                                        if k.endswith("_hz")))
    if args.out:
        payload = {
            "accel_g": res.accel_g,
            "fs_hz": np.array(res.fs_hz),
            "shaft_hz": np.array(spec.shaft_hz),
        }
        if res.current_a is not None:
            payload["current_a"] = res.current_a
        np.savez_compressed(args.out, **payload)
        print(f"wrote          {args.out}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    x, fs, shaft = _load(args.path)
    spec = _machine(args.machine)
    fx = _extractor(spec, args.demod)
    frame = fx.extract(x, shaft, fs_hz=fs)
    if args.json:
        print(json.dumps(frame.as_dict(), indent=2))
        return 0
    print(f"{len(frame.names)} features (spec v{frame.spec_version}), shaft {frame.shaft_hz:.2f} Hz")
    width = max(len(n) for n in frame.names)
    for name, value in frame.as_dict().items():
        print(f"  {name:<{width}}  {value: 14.6g}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .health import baseline_from_vectors, diagnose
    from .sim import simulate

    x, fs, shaft = _load(args.path)
    spec = _machine(args.machine)
    fx = _extractor(spec, args.demod)
    # A diagnosis needs a healthy reference. With no commissioning data on disk,
    # synthesise one from the machine model and say so loudly -- a ratio-based
    # rule set compared against the wrong baseline is worse than no diagnosis.
    print(f"baseline       {args.baseline_n} simulated healthy acquisitions "
          f"({'SYNTHETIC - not the real asset' if not args.baseline else args.baseline})")
    if args.baseline:
        X = np.load(args.baseline, allow_pickle=False)["vectors"]
    else:
        X = np.vstack([
            fx.extract(simulate(spec, None, duration_s=0.4, seed=9000 + i).accel_g, spec.shaft_hz).vector
            for i in range(args.baseline_n)
        ])
    base = baseline_from_vectors(X, fx.names)
    frame = fx.extract(x, shaft, fs_hz=fs)
    results = diagnose(frame, baseline=base)
    print(f"velocity       {frame.get('iso.velocity_rms_mm_s'):.3f} mm/s RMS  "
          f"ISO zone {'ABCD'[int(frame.get('iso.zone_index'))]}")
    print(f"time domain    rms={frame.get('time.rms'):.4f} g  kurtosis={frame.get('time.kurtosis'):.2f}  "
          f"crest={frame.get('time.crest'):.2f}")
    if not results:
        print("\ndiagnosis      no fault signature matched")
        return 0
    print()
    for d in results:
        print(f"  {d.fault}  confidence={d.confidence:.2f}  severity={d.severity}")
        print(f"    {d.explanation}")
        print(f"    action: {d.action}")
        print("    evidence: " + ", ".join(f"{k}={v:.3g}" for k, v in d.evidence.items()))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Full walk-through: commissioning, degradation, alarm escalation, RUL."""
    from .health import AlarmConfig, HealthIndex, ParticleFilterRUL, baseline_from_vectors, diagnose
    from .sim import degradation_series, simulate

    spec = _machine(args.machine)
    fx = _extractor(spec, args.demod)
    print(f"== {spec.name}: {spec.rpm:.0f} rpm, {spec.power_kw:.0f} kW, bearing {spec.bearing.name}")
    print(f"   commissioning on {args.commission} healthy acquisitions ...")
    comm = [simulate(spec, None, duration_s=0.4, seed=900 + i) for i in range(args.commission)]
    Xc = np.vstack([fx.extract(r.accel_g, spec.shaft_hz).vector for r in comm])
    base = baseline_from_vectors(Xc, fx.names)
    hi = HealthIndex(args.machine, fx.names, alarm_config=AlarmConfig(min_dwell_s=0.0))
    print("   " + json.dumps({k: v for k, v in hi.commission(Xc, [spec.shaft_hz] * len(comm)).items()
                              if k != "coverage"}))

    series = degradation_series(spec, mode=args.fault, n_points=args.points,
                               duration_s=0.4, days=args.days, seed=5)
    pf = ParticleFilterRUL(failure_threshold=20.0, min_health=2.0)
    hf_base = float(np.median([fx.extract(r.accel_g, spec.shaft_hz).get("band.10_rms") for r in comm]))
    print(f"\n   {'day':>6} {'sev':>5} {'HI':>5} {'trend':>6} {'state':>9}  {'diagnosis':<28} RUL")
    for i, rec in enumerate(series):
        frame = fx.extract(rec.accel_g, spec.shaft_hz)
        out = hi.update(rec.truth["t_days"] * 86400.0, frame.vector, speed_hz=spec.shaft_hz)
        pf.update(rec.truth["t_days"], frame.get("band.10_rms") / max(hf_base, 1e-12))
        escalated = out["event"] is not None and out["event"].is_escalation
        if i % max(1, args.points // 10) == 0 or escalated:
            dx = diagnose(frame, baseline=base)
            est = pf.estimate()
            rul = (f"{est.median_days:.0f} d [{est.p10_days:.0f}, {est.p90_days:.0f}]"
                   if est.status == "ok" else est.status)
            label = f"{dx[0].fault} ({dx[0].confidence:.2f})" if dx else "-"
            print(f"   {rec.truth['t_days']:6.1f} {rec.truth['severity']:5.2f} {out['index']:5.2f} "
                  f"{out['smoothed']:6.2f} {out['state']:>9}  {label:<28} {rul}")
    print("\n   alarm history:")
    for ev in hi.alarms.events:
        top = ", ".join(f"{k}={v:.1f}" for k, v in list(ev.evidence.items())[:3])
        print(f"     day {ev.timestamp_s / 86400:6.1f}  {ev.from_state.value} -> {ev.to_state.value:9s} "
              f"({ev.reason}) [{top}]")
    return 0


def cmd_payload(args: argparse.Namespace) -> int:
    from .io.payload import FrameFlags, TelemetryFrame, decode_frame, encode_frame, frame_size

    n = args.n_features
    rng = np.random.default_rng(0)
    frame = TelemetryFrame(
        device_id=0x0123456789ABCDEF,
        timestamp_ms=1_760_000_000_000,
        features=rng.standard_normal(n),
        shaft_hz=29.67,
        temperature_c=48.5,
        battery_mv=3612,
        rssi_dbm=-97,
        uptime_s=864_000,
        sequence=4242,
        flags=FrameFlags.SPEED_ESTIMATED,
    )
    raw = encode_frame(frame)
    back = decode_frame(raw)
    print(f"schema v{PAYLOAD_SCHEMA_VERSION}, feature spec v{FEATURE_SPEC_VERSION}")
    print(f"{n} features -> {len(raw)} bytes on the wire (predicted {frame_size(n)})")
    print(f"round-trip exact: {np.allclose(back.features, frame.features, atol=1e-6)}")
    print(f"as JSON instead:  {len(json.dumps(frame.as_dict()).encode()) + n * 20} bytes "
          f"(~{(len(json.dumps(frame.as_dict()).encode()) + n * 20) / len(raw):.1f}x larger)")
    for rate, label in ((20_000, "LTE-M 20 kbit/s"), (50_000, "BLE 50 kbit/s"), (6_250, "sub-GHz 6.25 kbit/s")):
        print(f"  airtime on {label:<24} {len(raw) * 8 / rate * 1000:7.1f} ms")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pdm", description="Predictive maintenance core toolkit")
    ap.add_argument("--version", action="version", version=f"pdmcore {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--machine", default="motor_pump_37kw", help="machine preset")
        p.add_argument("--demod", nargs=2, type=float, metavar=("LO", "HI"),
                       help="fixed demodulation band in Hz")

    p = sub.add_parser("simulate", help="generate a synthetic acquisition")
    add_common(p)
    p.add_argument("--fault", default=None, help="fault mode (default healthy)")
    p.add_argument("--severity", type=float, default=0.4)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--current", action="store_true", help="also simulate stator current")
    p.add_argument("--out", default=None, help="write .npz")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("features", help="extract the platform feature vector")
    add_common(p)
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("diagnose", help="run the rule-based diagnosis")
    add_common(p)
    p.add_argument("path")
    p.add_argument("--baseline", default=None, help=".npz with a 'vectors' array")
    p.add_argument("--baseline-n", type=int, default=40)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("demo", help="commissioning -> degradation -> alarm -> RUL")
    add_common(p)
    p.add_argument("--fault", default="bearing_outer")
    p.add_argument("--points", type=int, default=80)
    p.add_argument("--days", type=float, default=90.0)
    p.add_argument("--commission", type=int, default=60)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("payload", help="show the wire format and radio airtime")
    p.add_argument("--n-features", type=int, default=63)
    p.set_defaults(func=cmd_payload)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
