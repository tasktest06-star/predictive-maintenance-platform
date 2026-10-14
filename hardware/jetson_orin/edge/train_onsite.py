"""On-site training: the one justification for this board that survives.

    PYTHONPATH=core:edge python3 hardware/jetson_orin/edge/train_onsite.py \
        --queue /var/lib/pdm/gateway.sqlite --out /var/lib/pdm/models --dry-run

The workload study in `tools/workload_study.py` tried to find a job for a
67-TOPS GPU in this platform and failed: re-scoring a 1000-node site's entire
year takes 44 seconds of CPU, the classical detectors fit in milliseconds, and
raw-waveform inference at hourly cadence is four orders of magnitude below one
core. Nothing here needs a Jetson.

Two justifications survived, and neither is about speed. This file implements
the first one.

**DATA SOVEREIGNTY.** A plant that will not send vibration data off site cannot
have its models trained in a cloud. Pharmaceutical, defence, and increasingly
any site whose legal team has read a data-processing agreement: the data stays
inside the fence. That makes the *site* the place where training happens, and a
site that trains needs more than a gateway.

So this trains from what the site already has -- the gateway's own
store-and-forward queue -- and produces a deployable bundle without any data
leaving the building.

WHAT IT REFUSES TO DO, which is most of the value:

A model trained on whatever happened to be in a queue is a model trained on an
unknown population. This tool enforces the gates that `tools/export_model.py`
enforces for the int8 MCU path, plus the ones specific to training on field
data:

  - a minimum number of frames, and a minimum SPAN of time, because 500 frames
    from one afternoon describe one operating point;
  - no unscoreable frames (clipped, overrun, self-test failed) in the training
    set -- a detector trained on artefacts learns the artefacts;
  - no frames flagged SPEED_UNSTEADY or RATE_NOMINAL unless explicitly allowed,
    because their high-order content is smeared or mis-scaled;
  - THREE windows in time order -- fit, calibrate, verify -- so the threshold
    is calibrated on data the fit has not seen AND verified on data the
    calibration has not seen;
  - a refusal if the VERIFICATION window's false-alarm count exceeds a 3-sigma
    Poisson band around the target, which is what a non-stationary population
    looks like -- and NOT a fixed multiple of the target, which at these sample
    sizes refuses stationary data a third of the time.

The three-way split is not fastidiousness. The first version of this file used
two windows: fit on the first 75 %, calibrate the threshold on the last 25 %,
then measure the false-alarm rate on that same 25 %. That equals the target BY
CONSTRUCTION -- it is what calibrating to an FPR means -- so the check always
passed, and a deliberate 4x monotone drift across the span sailed through it.
A check that cannot fail is worse than no check.

Every refusal prints the number that caused it. A training tool that silently
produces a model from insufficient data is how a plant ends up with a detector
that alarms on Monday mornings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_REPO, "core"), os.path.join(_REPO, "edge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pdmcore import FEATURE_SPEC_VERSION  # noqa: E402
from pdmcore.io.payload import FrameFlags, PayloadError, decode_frame  # noqa: E402
from pdmcore.models.anomaly import MahalanobisDetector  # noqa: E402

log = logging.getLogger("pdmedge.train")

#: Minimum usable frames. The span is split 50/25/25 (fit/calibrate/verify),
#: so the fit window gets half: 400 frames gives it 200, against the ~189 a
#: 63-feature covariance estimate needs
#: (MahalanobisDetector.MIN_SAMPLES_PER_FEATURE is 3).
MIN_FRAMES = 400
#: Minimum SPAN of the training set. A week of hourly frames covers a weekly
#: duty cycle; 200 frames from one afternoon do not, and a detector that has
#: never seen a Monday start-up will alarm on the first one.
MIN_SPAN_DAYS = 7.0
#: THREE windows, in time order: fit, calibrate, verify.
#:
#: Two would not be enough, and the first version of this file used two and was
#: wrong in a way worth recording. It fitted on the first 75 %, calibrated the
#: threshold on the last 25 %, and then measured the false-alarm rate on that
#: same last 25 % -- which equals the target BY CONSTRUCTION, because that is
#: what calibrating to an FPR does. The check always passed. A deliberate 4x
#: monotone drift across the span sailed through it.
#:
#: With three windows the threshold is calibrated on the middle one and
#: verified on a later one it has never seen, so a population that moved shows
#: up as a verification FPR far above target. That is the whole point of
#: splitting by time rather than at random.
FIT_FRACTION = 0.50
CALIBRATE_FRACTION = 0.25
#: The verification gate is a POISSON band, not a fixed multiple of the target.
#:
#: A fixed multiple is the wrong test at these sample sizes and was the second
#: bug in this file. On a 150-frame verification window with a 1 % target, the
#: expected number of false alarms is 1.5 -- and Poisson noise alone spans 0 to
#: 6 at 3 sigma. A "4x the target" gate therefore refuses perfectly stationary
#: data roughly a third of the time, which in a plant means a commissioning
#: engineer re-running training until it passes: the worst possible outcome,
#: because it selects for whichever window happened to look quiet.
#:
#: So: allowed count = lambda + Z*sqrt(lambda) + FLOOR, with lambda = n * target.
#: The floor keeps the band from collapsing when lambda is tiny, and Z = 3 is
#: the same 3-sigma convention the health index uses, so the numbers are
#: comparable across the platform.
FPR_SIGMA = 3.0
FPR_COUNT_FLOOR = 3.0


@dataclass
class TrainingSet:
    """Frames for one device, with the provenance that justifies them."""

    device_id: int
    channel: int
    vectors: np.ndarray
    timestamps_ms: np.ndarray
    rejected: Dict[str, int] = field(default_factory=dict)

    @property
    def span_days(self) -> float:
        if self.timestamps_ms.size < 2:
            return 0.0
        return float(self.timestamps_ms.max() - self.timestamps_ms.min()) / 86_400_000.0

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])


def load_from_queue(path: str, *, allow_unsteady: bool = False,
                    allow_nominal_rate: bool = True) -> Dict[Tuple[int, int], TrainingSet]:
    """Read a gateway queue and group decodable, trainable frames by (device, channel).

    Reads directly rather than through FrameQueue because training must not
    consume the queue: these frames still have to go upstream.
    """
    import sqlite3

    if not os.path.isfile(path):
        raise SystemExit(f"no queue at {path}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = db.execute("SELECT ts_ms, payload FROM frames ORDER BY ts_ms").fetchall()
    db.close()

    groups: Dict[Tuple[int, int], Dict[str, list]] = {}
    rejected: Dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for ts_ms, payload in rows:
        try:
            frame = decode_frame(bytes(payload))
        except PayloadError:
            _reject("undecodable")
            continue
        if frame.feature_spec_version != FEATURE_SPEC_VERSION:
            _reject(f"feature_spec {frame.feature_spec_version}")
            continue
        flags = FrameFlags(frame.flags)
        if flags.blocks_scoring:
            # A detector trained on clipped or overrun acquisitions learns the
            # artefacts, then fails to alarm on the real thing because the real
            # thing is less extreme than its training data.
            _reject("unscoreable (clipped/overrun/selftest)")
            continue
        if not allow_unsteady and (flags & FrameFlags.SPEED_UNSTEADY):
            _reject("speed unsteady")
            continue
        if not allow_nominal_rate and (flags & FrameFlags.RATE_NOMINAL):
            _reject("nominal sample rate")
            continue
        if not np.all(np.isfinite(frame.features)):
            _reject("non-finite features")
            continue
        key = (frame.device_id, flags.channel)
        g = groups.setdefault(key, {"v": [], "t": []})
        g["v"].append(np.asarray(frame.features, dtype=np.float64))
        g["t"].append(int(ts_ms))

    out: Dict[Tuple[int, int], TrainingSet] = {}
    for (dev, ch), g in groups.items():
        out[(dev, ch)] = TrainingSet(
            device_id=dev, channel=ch,
            vectors=np.asarray(g["v"]),
            timestamps_ms=np.asarray(g["t"], dtype=np.int64),
            rejected=dict(rejected),
        )
    return out


@dataclass
class TrainingResult:
    device_id: int
    channel: int
    accepted: bool
    reason: str = ""
    n_train: int = 0
    n_calibrate: int = 0
    n_verify: int = 0
    span_days: float = 0.0
    threshold: float = float("nan")
    calibrate_fpr: float = float("nan")
    verify_fpr: float = float("nan")
    verify_alarms: int = 0
    verify_allowed_alarms: float = float("nan")
    target_fpr: float = float("nan")
    fit_ms: float = float("nan")


def train_one(ts: TrainingSet, target_fpr: float = 0.01) -> Tuple[TrainingResult,
                                                                 Optional[object]]:
    """Fit and calibrate one device-channel, or refuse with the number."""
    res = TrainingResult(device_id=ts.device_id, channel=ts.channel, accepted=False,
                         span_days=ts.span_days, target_fpr=target_fpr)

    if ts.n < MIN_FRAMES:
        res.reason = (f"only {ts.n} usable frames; {MIN_FRAMES} needed so the "
                      f"50/25/25 fit/calibrate/verify split leaves a fit window "
                      f"big enough for a 63-feature covariance estimate")
        return res, None
    if ts.span_days < MIN_SPAN_DAYS:
        res.reason = (f"frames span only {ts.span_days:.1f} days; {MIN_SPAN_DAYS:.0f} "
                      f"needed to cover a weekly duty cycle. A detector that has "
                      f"never seen a Monday start-up will alarm on the first one")
        return res, None

    # THREE windows in TIME order. Not random: a random split from a drifting
    # population leaks the drift into every window, and the threshold then looks
    # fine and fails next week.
    order = np.argsort(ts.timestamps_ms)
    v = ts.vectors[order]
    n = len(v)
    i_fit = int(n * FIT_FRACTION)
    i_cal = i_fit + int(n * CALIBRATE_FRACTION)
    train, calib, verify = v[:i_fit], v[i_fit:i_cal], v[i_cal:]
    res.n_train, res.n_calibrate, res.n_verify = len(train), len(calib), len(verify)
    if res.n_calibrate < 10 or res.n_verify < 10:
        res.reason = (f"calibrate/verify windows are {res.n_calibrate}/"
                      f"{res.n_verify} frames; 10 each is the minimum")
        return res, None

    t0 = time.perf_counter()
    det = MahalanobisDetector().fit(train)
    res.fit_ms = (time.perf_counter() - t0) * 1e3
    res.threshold = float(det.calibrate(calib, fpr=target_fpr))
    # By construction this is ~the target: calibrating to an FPR makes it so.
    # Recorded for the record, never used as a check -- treating it as one is
    # the bug this three-way split exists to fix.
    res.calibrate_fpr = float(np.mean(det.score(calib) > res.threshold))

    # The real check, on a LATER window the threshold has never seen, against a
    # Poisson band rather than a fixed multiple (see FPR_SIGMA above).
    alarms = int(np.sum(det.score(verify) > res.threshold))
    res.verify_fpr = alarms / res.n_verify
    lam = res.n_verify * target_fpr
    allowed = lam + FPR_SIGMA * float(np.sqrt(lam)) + FPR_COUNT_FLOOR
    res.verify_allowed_alarms = allowed
    res.verify_alarms = alarms
    if alarms > allowed:
        res.reason = (f"{alarms} false alarms on the later verification window of "
                      f"{res.n_verify} frames, against {lam:.1f} expected and "
                      f"{allowed:.1f} allowed at {FPR_SIGMA:.0f} sigma: the "
                      f"population moved between the calibration and "
                      f"verification windows, so this threshold would not hold "
                      f"next week")
        return res, None

    res.accepted = True
    return res, det


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True, help="gateway queue SQLite file")
    ap.add_argument("--out", default="", help="directory for accepted model bundles")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--allow-unsteady", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be trained, write nothing")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(message)s", stream=sys.stderr)

    groups = load_from_queue(args.queue, allow_unsteady=args.allow_unsteady)
    if not groups:
        print("no trainable frames in the queue", file=sys.stderr)
        return 1

    any_rejected = next(iter(groups.values())).rejected
    if any_rejected:
        print("frames excluded before training:")
        for reason, count in sorted(any_rejected.items(), key=lambda kv: -kv[1]):
            print(f"  {count:6d}  {reason}")
        print()

    print(f"{len(groups)} device-channel(s) in the queue")
    accepted: List[TrainingResult] = []
    for key in sorted(groups):
        ts = groups[key]
        res, det = train_one(ts, target_fpr=args.fpr)
        tag = f"{res.device_id:016x}/ch{res.channel}"
        if res.accepted:
            accepted.append(res)
            print(f"  {tag}  ACCEPTED  {res.n_train}/{res.n_calibrate}/"
                  f"{res.n_verify} fit/cal/verify over {res.span_days:.1f} d, "
                  f"threshold {res.threshold:.3f}, verify "
                  f"{res.verify_alarms}/{res.n_verify} alarms "
                  f"(<= {res.verify_allowed_alarms:.1f} allowed), "
                  f"fit {res.fit_ms:.1f} ms")
            if args.out and not args.dry_run and det is not None:
                os.makedirs(args.out, exist_ok=True)
                card = {
                    "device_id": f"{res.device_id:016x}",
                    "channel": res.channel,
                    "kind": "mahalanobis",
                    "feature_spec_version": FEATURE_SPEC_VERSION,
                    "n_train": res.n_train,
                    "n_calibrate": res.n_calibrate,
                    "n_verify": res.n_verify,
                    "span_days": res.span_days,
                    "threshold": res.threshold,
                    "target_fpr": args.fpr,
                    "calibrate_fpr": res.calibrate_fpr,
                    "verify_fpr": res.verify_fpr,
                    "verify_alarms": res.verify_alarms,
                    "verify_allowed_alarms": res.verify_allowed_alarms,
                    "trained_at_ms": int(time.time() * 1000),
                    "trained_on": "site",
                    "data_left_site": False,
                    "allow_unsteady": bool(args.allow_unsteady),
                }
                path = os.path.join(args.out, f"{res.device_id:016x}_ch{res.channel}.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(card, fh, indent=1)
                print(f"      wrote {path}")
        else:
            print(f"  {tag}  REFUSED   {res.reason}")

    print(f"\n{len(accepted)} of {len(groups)} accepted"
          f"{' (dry run: nothing written)' if args.dry_run else ''}")
    print("No frame left the site. That is the point of training here rather")
    print("than in a cloud, and it is the only justification for this board")
    print("that survived hardware/jetson_orin/tools/workload_study.py.")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
