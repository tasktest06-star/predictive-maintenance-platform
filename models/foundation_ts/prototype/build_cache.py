#!/usr/bin/env python3
"""Build the per-dataset cache: 63 features, five presentations, ground truth.

    /workshop/predictive_maintaince/.venv/bin/python -I \
        models/foundation_ts/prototype/build_cache.py \
        --dataset cwru --out /workshop/predictive_maintaince/fm_work

Run with ``-I``. The CWRU and MFPT ``.npz``/``.json`` files are *downloaded
data*, so the interpreter must not pick code up from the directory they live
in or from the current directory; this script therefore puts the repository's
``core/`` on ``sys.path`` itself rather than relying on ``PYTHONPATH``, which
``-I`` ignores.

Why a cache at all: the foundation-model arm runs in a **different virtual
environment** (Python 3.13 + torch) from the rest of the repository (Python 3.9
+ numpy/scipy/sklearn). The cache is the interface between the two, and keeping
it a plain ``.npz`` of float32 arrays means the torch process never imports
``pdmcore`` and the evaluation process never imports torch. That separation is
not incidental -- it is what lets the production variant's dependency set stay
at numpy+scipy while the research arm needs 2.5 GB of PyTorch.

Nothing in the cache is committed. It is derived from data that is itself not
committed (CWRU: research-use; MFPT: CC BY-NC-SA 4.0). What is committed is
this script, the manifest digests the loader records, and the results JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_REPO, "core"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pdmcore import FEATURE_SPEC_VERSION                               # noqa: E402
from pdmcore.features import FeatureConfig, FeatureExtractor           # noqa: E402

from datasets import (DEFAULT_DATA_ROOT, WindowSet, load_cwru,         # noqa: E402
                      load_mfpt, load_sim_degradation, load_sim_fleet)
from presentations import DEMOD_FRACTION, PRESENTATIONS, SEQ_LEN, present_all  # noqa: E402

DEFAULT_WORK = "/workshop/predictive_maintaince/fm_work"


def feature_config(fs_hz: float, window_samples: int, geometry) -> FeatureConfig:
    """A ``FeatureConfig`` legal for this window length and sample rate.

    Two constraints that are easy to violate and produce a silently degraded
    baseline, which would make this branch's headline a strawman:

    * ``n_fft`` must not exceed the window. A 4096-point config on a
      2048-sample window makes ``pdmcore.dsp.spectrum`` fall back to the
      largest power of two that fits, so the config and the data disagree
      about the resolution.
    * the envelope decimation target must leave enough envelope samples for
      ``envelope_n_fft``. At 97 656 Hz a 2 500 Hz target decimates by 39 and a
      2048-sample window becomes 52 envelope samples, which is 32 usable FFT
      points -- a resolution of 38 Hz, coarser than the 81 Hz BPFO it is
      supposed to resolve.

    The demodulation band is the same fixed fraction of Nyquist the
    presentations use, so the baseline and the foundation-model arm demodulate
    the same band. Fixed rather than kurtogram-selected for the reason given in
    ``presentations.py``.
    """
    n_fft = 1 << int(np.floor(np.log2(window_samples)))
    nyq = fs_hz / 2.0
    # Target ~8x the highest kinematic rate we need (BPFI at 5.5 orders of a
    # 30 Hz shaft is 165 Hz), but never so low that the envelope FFT has fewer
    # than 128 points.
    env_n_fft = 256
    min_env_samples = 2 * env_n_fft
    max_factor = max(1, window_samples // min_env_samples)
    decimate_to = fs_hz / max_factor if max_factor > 1 else None
    return FeatureConfig(
        fs_hz=fs_hz,
        n_fft=n_fft,
        bearing=geometry,
        demod_band_hz=(DEMOD_FRACTION[0] * nyq, DEMOD_FRACTION[1] * nyq),
        envelope_n_fft=env_n_fft,
        envelope_decimate_to_hz=decimate_to,
    )


def extract_features(ws: WindowSet) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, object]]:
    """The 63-feature baseline matrix for every window.

    One extractor per ``(bearing, fs)`` combination, because ``FeatureConfig``
    pins both and a single extractor reused across sample rates would compute
    band edges for the wrong Nyquist.
    """
    keys = sorted({(str(b), float(f)) for b, f in zip(ws.bearing_key, ws.fs_hz)})
    extractors = {
        k: FeatureExtractor(feature_config(k[1], ws.window_samples, ws.geometries[k[0]]))
        for k in keys
    }
    names = next(iter(extractors.values())).names
    for ex in extractors.values():
        if ex.names != names:
            raise AssertionError("feature layout differs between extractors")
    X = np.empty((len(ws), len(names)), dtype=np.float64)
    n_nonfinite = 0
    for i in range(len(ws)):
        ex = extractors[(str(ws.bearing_key[i]), float(ws.fs_hz[i]))]
        frame = ex.extract(ws.x[i], float(ws.shaft_hz[i]), fs_hz=float(ws.fs_hz[i]))
        X[i] = frame.vector
        n_nonfinite += int(frame.extras.get("n_nonfinite", 0))
    cfgs = {f"{k[0]}@{k[1]:.0f}Hz": extractors[k].config.as_dict() for k in keys}
    return X, names, {"feature_configs": cfgs, "n_nonfinite_values": n_nonfinite,
                      "feature_spec_version": FEATURE_SPEC_VERSION}


def build(name: str, ws: WindowSet, out_dir: str,
          extra: Optional[Dict[str, np.ndarray]] = None) -> str:
    t0 = time.time()
    X, names, feat_meta = extract_features(ws)
    t_feat = time.time() - t0
    t0 = time.time()
    seqs, levels = present_all(ws.x, ws.fs_hz)
    t_pres = time.time() - t0

    arrays: Dict[str, np.ndarray] = {
        "features": X.astype(np.float32),
        "levels_db": levels.astype(np.float32),
        "y": ws.y.astype(np.int8),
        "label": ws.label.astype("U24"),
        "group": ws.group.astype("U48"),
        "cond": ws.cond.astype("U16"),
        "bearing_key": ws.bearing_key.astype("U32"),
        "t_index": ws.t_index.astype(np.int32),
        "fs_hz": ws.fs_hz.astype(np.float64),
        "shaft_hz": ws.shaft_hz.astype(np.float64),
    }
    for k, v in seqs.items():
        arrays[f"seq_{k}"] = v
    if extra:
        arrays.update(extra)

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"cache_{name}.npz")
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "name": name,
        "summary": ws.summary(),
        "provenance": ws.provenance,
        "feature_names": list(names),
        "presentations": list(PRESENTATIONS),
        "seq_len": SEQ_LEN,
        "demod_fraction_of_nyquist": list(DEMOD_FRACTION),
        "geometries": {k: {"n_balls": g.n_balls, "ball_d": g.ball_d,
                           "pitch_d": g.pitch_d,
                           "contact_angle_deg": g.contact_angle_deg}
                       for k, g in ws.geometries.items()},
        "timing_s": {"feature_extraction": round(t_feat, 2),
                     "presentations": round(t_pres, 2)},
        "per_window_ms": {"feature_extraction": round(1e3 * t_feat / len(ws), 3),
                          "presentations_all5": round(1e3 * t_pres / len(ws), 3)},
    }
    meta.update(feat_meta)
    with open(os.path.join(out_dir, f"cache_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print(f"{name}: {len(ws)} windows, {X.shape[1]} features, "
          f"features {t_feat:.1f}s, presentations {t_pres:.1f}s -> {npz_path}")
    return npz_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["cwru", "cwru_fe", "mfpt", "mfpt_vload",
                             "sim_fleet", "sim_degradation", "all"])
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=DEFAULT_WORK)
    ap.add_argument("--window", type=int, default=2048)
    args = ap.parse_args(argv)

    wanted = (["cwru", "cwru_fe", "mfpt", "mfpt_vload", "sim_fleet", "sim_degradation"]
              if args.dataset == "all" else [args.dataset])
    for name in wanted:
        if name == "cwru":
            build("cwru_de", load_cwru(args.data_root, channel="DE",
                                       window_samples=args.window), args.out)
        elif name == "cwru_fe":
            build("cwru_fe", load_cwru(args.data_root, channel="FE",
                                       window_samples=args.window), args.out)
        elif name == "mfpt":
            build("mfpt_97k", load_mfpt(args.data_root, window_samples=args.window,
                                        fs_hz_filter=97656.0), args.out)
        elif name == "mfpt_vload":
            build("mfpt_49k", load_mfpt(args.data_root, window_samples=args.window,
                                        fs_hz_filter=48828.0), args.out)
        elif name == "sim_fleet":
            build("sim_fleet", load_sim_fleet(window_samples=args.window), args.out)
        elif name == "sim_degradation":
            ws, t_days, sev = load_sim_degradation(window_samples=args.window)
            build("sim_degradation", ws, args.out,
                  extra={"t_days": t_days.astype(np.float64),
                         "severity": sev.astype(np.float64)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
