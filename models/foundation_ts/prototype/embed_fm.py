#!/usr/bin/env python3
"""Run the real pretrained foundation models over the cached presentations.

This is the foundation-model arm, and it is **path (a)**: genuine pretrained
checkpoints, downloaded from the Hugging Face Hub and run here. It is not an
architecture re-implementation standing in for one. The three checkpoints:

==========================  ==========  ========================================
checkpoint                  parameters  what it is
==========================  ==========  ========================================
``amazon/chronos-bolt-small``   47.7 M  Patch-encoder/decoder T5, direct
                                        multi-step quantile forecaster.
                                        Pretrained on ~100 B observations
                                        (Chronos corpus + synthetic). This is
                                        the "patch transformer pretrained on a
                                        big corpus" the brief names.
``amazon/chronos-t5-small``     46.2 M  Chronos (arXiv 2403.07815): scale the
                                        series, quantise into 4096 bins, treat
                                        bins as tokens, train a T5 as a
                                        language model over them.
``AutonLab/MOMENT-1-small``     35.3 M  MOMENT (arXiv 2402.03885): T5 encoder
                                        over 8-point patches, pretrained by
                                        masked patch reconstruction on the
                                        Time Series Pile. Its reconstruction
                                        head is pretrained, which is what makes
                                        a genuinely *zero-shot* anomaly score
                                        possible.
==========================  ==========  ========================================

Two kinds of output, because "zero-shot anomaly detection" means two different
things in the literature and they deserve separate numbers:

**Embeddings** (``emb_<model>_<presentation>``, ``(n, d)``) -- the encoder's
mean-pooled hidden states. These still need a one-class head fitted on healthy
data, so this is *representation transfer*, not zero-shot detection. It is the
fairer comparison against the repository's baseline because both arms then use
the identical ``pdmcore`` detector and the identical calibration.

**Native zero-shot scores** (``score_<name>``, ``(n,)``) -- no fitting at all:

* ``score_moment_recon_<presentation>``: mean squared reconstruction error from
  MOMENT's pretrained reconstruction head. This is the mechanism MOMENT's own
  paper uses for anomaly detection.
* ``score_chronos_<model>_fcast_<presentation>``: give the model the first 448
  points, ask for the next 64, and score the median forecast's absolute error
  normalised by the context's mean absolute first difference (a per-window
  MASE). This is the mechanism every "forecast and threshold the residual"
  anomaly-detection pipeline uses.

Usage (must be the torch venv, which is Python 3.13; the rest of the
repository is Python 3.9 and has no torch)::

    HF_HOME=/workshop/predictive_maintaince/fm_models/hf_cache \\
    /workshop/predictive_maintaince/.venv-fm/bin/python -I \\
        models/foundation_ts/prototype/embed_fm.py \\
        --cache /workshop/predictive_maintaince/fm_work/cache_cwru_de.npz \\
        --out   /workshop/predictive_maintaince/fm_work

``-I`` because the cache is derived from downloaded data; this script imports
nothing from the repository, so it needs no ``sys.path`` surgery.

Determinism: inference only, ``torch.no_grad``, no dropout (``.eval()``), fixed
batch order, float32. Repeated runs agree bit-for-bit on this host; a different
BLAS or thread count can change the last couple of digits, which is why the
decision rule in ``research/02_protocol.md`` is stated with a margin rather
than as an exact inequality.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CHRONOS_MODELS = {
    "bolt_small": "amazon/chronos-bolt-small",
    "t5_small": "amazon/chronos-t5-small",
}
MOMENT_MODELS = {"moment_small": "AutonLab/MOMENT-1-small"}

#: Forecast split for the zero-shot residual score. 448 + 64 = 512, i.e. the
#: whole context, so the forecast arm and the embedding arm see the same window.
FCAST_CONTEXT, FCAST_HORIZON = 448, 64


def _host_info() -> Dict[str, object]:
    """What was actually measured. Printed so a latency can never be quoted
    without the machine it came from -- the convention
    ``hardware/rpi5_hailo/tools/benchmark.py`` set."""
    import torch

    info: Dict[str, object] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
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


def _batches(n: int, size: int):
    for i in range(0, n, size):
        yield i, min(n, i + size)


# ---------------------------------------------------------------------------
# Chronos
# ---------------------------------------------------------------------------
def chronos_arm(pipeline, seqs: np.ndarray, batch: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """``(embeddings (n,d), zero-shot forecast residual (n,), timings)``."""
    import torch

    n = seqs.shape[0]
    emb_out: List[np.ndarray] = []
    res_out = np.empty(n, dtype=np.float64)
    t_emb = t_fc = 0.0
    with torch.no_grad():
        for lo, hi in _batches(n, batch):
            ctx = torch.from_numpy(np.ascontiguousarray(seqs[lo:hi])).to(torch.float32)

            t0 = time.perf_counter()
            emb, _loc_scale = pipeline.embed(ctx)
            t_emb += time.perf_counter() - t0
            # Mean-pool over tokens. Chronos-T5 appends an EOS token and
            # Chronos-Bolt emits one token per patch; pooling over whatever the
            # encoder produced is the standard way these embeddings are used
            # downstream and keeps the two comparable at (n, 512).
            emb_out.append(emb.to(torch.float32).mean(dim=1).cpu().numpy())

            t0 = time.perf_counter()
            q, _mean = pipeline.predict_quantiles(
                ctx[:, :FCAST_CONTEXT], prediction_length=FCAST_HORIZON,
                quantile_levels=[0.5])
            t_fc += time.perf_counter() - t0
            median = q[:, :, 0].to(torch.float32).cpu().numpy()
            truth = seqs[lo:hi, FCAST_CONTEXT:FCAST_CONTEXT + FCAST_HORIZON]
            # Per-window MASE: absolute forecast error over the context's mean
            # absolute first difference. Scale-free, which matters because the
            # presentations differ in units (g, g^2, dB) and because every one
            # of these models normalises the level away anyway.
            naive = np.mean(np.abs(np.diff(seqs[lo:hi, :FCAST_CONTEXT], axis=1)), axis=1)
            res_out[lo:hi] = (np.mean(np.abs(median - truth), axis=1)
                              / np.maximum(naive, 1e-12))
    return (np.vstack(emb_out).astype(np.float32), res_out,
            {"embed_ms_per_window": 1e3 * t_emb / n,
             "forecast_ms_per_window": 1e3 * t_fc / n})


# ---------------------------------------------------------------------------
# MOMENT
# ---------------------------------------------------------------------------
def moment_arm(model_id: str, seqs: np.ndarray, batch: int
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """``(embeddings (n,d), zero-shot reconstruction error (n,), timings)``.

    Two separate pipelines because ``task_name`` is fixed at construction:
    ``embedding`` returns the pooled encoder state, ``reconstruction`` runs the
    pretrained masked-reconstruction head. Only the reconstruction head is
    pretrained (the library warns about it on load); classification and
    forecasting heads would need fine-tuning and are not used here.
    """
    import torch
    from momentfm import MOMENTPipeline

    n = seqs.shape[0]
    mask = torch.ones(1, seqs.shape[1], dtype=torch.float32)
    emb_out: List[np.ndarray] = []
    rec_out = np.empty(n, dtype=np.float64)
    t_emb = t_rec = 0.0

    emb_model = MOMENTPipeline.from_pretrained(model_id, model_kwargs={"task_name": "embedding"})
    emb_model.init()
    emb_model.eval()
    rec_model = MOMENTPipeline.from_pretrained(model_id, model_kwargs={"task_name": "reconstruction"})
    rec_model.init()
    rec_model.eval()

    with torch.no_grad():
        for lo, hi in _batches(n, batch):
            x = torch.from_numpy(np.ascontiguousarray(seqs[lo:hi])).to(torch.float32)
            x = x.unsqueeze(1)                       # (b, 1 channel, 512)
            m = mask.expand(x.shape[0], -1)

            t0 = time.perf_counter()
            out = emb_model(x_enc=x, input_mask=m)
            t_emb += time.perf_counter() - t0
            emb_out.append(out.embeddings.to(torch.float32).cpu().numpy())

            t0 = time.perf_counter()
            rec = rec_model(x_enc=x, input_mask=m).reconstruction
            t_rec += time.perf_counter() - t0
            err = (rec.to(torch.float32) - x).pow(2).mean(dim=(1, 2)).cpu().numpy()
            # Normalise by the window's own variance. MOMENT instance-normalises
            # internally, so a raw MSE would mostly report the window's
            # amplitude -- which would make this score a disguised RMS detector
            # rather than a measurement of the model.
            var = np.var(seqs[lo:hi], axis=1)
            rec_out[lo:hi] = err / np.maximum(var, 1e-12)
    return (np.vstack(emb_out).astype(np.float32), rec_out,
            {"embed_ms_per_window": 1e3 * t_emb / n,
             "reconstruct_ms_per_window": 1e3 * t_rec / n})


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="cache_<name>.npz from build_cache.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="bolt_small,t5_small,moment_small")
    ap.add_argument("--presentations", default="raw_crop,raw_decim,envelope,logspec,envelope_spec")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N windows")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(int(args.threads))
    torch.manual_seed(0)

    name = os.path.basename(args.cache)[len("cache_"):-len(".npz")]
    with np.load(args.cache, allow_pickle=False) as npz:
        seq_keys = [k for k in npz.files if k.startswith("seq_")]
        seqs = {k[4:]: np.asarray(npz[k], dtype=np.float32) for k in seq_keys}
    wanted_pres = [p for p in args.presentations.split(",") if p in seqs]
    missing = sorted(set(args.presentations.split(",")) - set(seqs))
    if missing:
        raise SystemExit(f"cache {args.cache} has no presentation(s) {missing}; "
                         f"it has {sorted(seqs)}")
    if args.limit:
        seqs = {k: v[: args.limit] for k, v in seqs.items()}

    out: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {
        "cache": os.path.abspath(args.cache),
        "dataset": name,
        "host": _host_info(),
        "models": {},
        "fcast_context": FCAST_CONTEXT,
        "fcast_horizon": FCAST_HORIZON,
        "n_windows": int(next(iter(seqs.values())).shape[0]),
        "batch": args.batch,
    }

    for key in [m for m in args.models.split(",") if m in CHRONOS_MODELS]:
        from chronos import BaseChronosPipeline

        model_id = CHRONOS_MODELS[key]
        t0 = time.time()
        pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu",
                                                   dtype=torch.float32)
        load_s = time.time() - t0
        n_params = int(sum(p.numel() for p in pipe.model.parameters()))
        timings: Dict[str, Dict[str, float]] = {}
        for pres in wanted_pres:
            emb, res, t = chronos_arm(pipe, seqs[pres], args.batch)
            out[f"emb_{key}_{pres}"] = emb
            out[f"score_{key}_fcast_{pres}"] = res.astype(np.float64)
            timings[pres] = {k: round(v, 3) for k, v in t.items()}
            print(f"  {key:12s} {pres:14s} emb {emb.shape} "
                  f"{t['embed_ms_per_window']:.1f} ms/win", flush=True)
        meta["models"][key] = {"hf_id": model_id, "parameters": n_params,
                               "load_s": round(load_s, 2),
                               "pipeline_class": type(pipe).__name__,
                               "timings_ms_per_window": timings,
                               "embedding_dim": int(out[f"emb_{key}_{wanted_pres[0]}"].shape[1])}
        del pipe

    for key in [m for m in args.models.split(",") if m in MOMENT_MODELS]:
        model_id = MOMENT_MODELS[key]
        timings = {}
        for pres in wanted_pres:
            t0 = time.time()
            emb, rec, t = moment_arm(model_id, seqs[pres], args.batch)
            out[f"emb_{key}_{pres}"] = emb
            out[f"score_{key}_recon_{pres}"] = rec.astype(np.float64)
            timings[pres] = {k: round(v, 3) for k, v in t.items()}
            print(f"  {key:12s} {pres:14s} emb {emb.shape} "
                  f"{t['embed_ms_per_window']:.1f} ms/win "
                  f"(incl. {time.time() - t0:.0f}s load)", flush=True)
        import torch as _t
        from momentfm import MOMENTPipeline
        probe = MOMENTPipeline.from_pretrained(model_id, model_kwargs={"task_name": "embedding"})
        probe.init()
        meta["models"][key] = {
            "hf_id": model_id,
            "parameters": int(sum(p.numel() for p in probe.parameters())),
            "timings_ms_per_window": timings,
            "embedding_dim": int(out[f"emb_{key}_{wanted_pres[0]}"].shape[1]),
            "note": "only the reconstruction head is pretrained; the library "
                    "warns on load that classification and forecasting heads "
                    "require fine-tuning.",
        }
        del probe, _t

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, f"fm_{name}.npz")
    np.savez_compressed(npz_path, **out)
    with open(os.path.join(args.out, f"fm_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print(f"{name}: wrote {len(out)} arrays -> {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
