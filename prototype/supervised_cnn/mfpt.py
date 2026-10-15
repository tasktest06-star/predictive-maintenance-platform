"""MFPT loader, and the resampling that makes a cross-dataset test meaningful.

MFPT (Society for Machinery Failure Prevention Technology, Eric Bechhoefer's
seeded-fault set) is the second dataset named in
``docs/07_production_readiness.md`` as the validation gate. It is a genuinely
different rig from CWRU:

    quantity          CWRU 12k DE        MFPT
    ---------------   ----------------   -----------------------------
    sample rate       12 000 Hz          97 656 Hz (270 lb set)
                                         48 828 Hz (varying-load sets)
    shaft rate        28.8-30.0 Hz       25 Hz
    bearing           SKF 6205, 9 balls  NICE, 8 rollers
    d/D               0.2034             0.2647
    BPFO / shaft      3.585x             3.058x
    BPFI / shaft      5.415x             4.942x
    load              0-3 hp motor load  0-300 lb radial
    record length     10 s / 20-40 s     3 s / 6 s
    classes           normal, IR, OR, B  baseline, IR, OR  (no ball defect)

The common label space is therefore three classes: healthy, inner, outer.

Two resampling conventions, because the difference between them IS a result
--------------------------------------------------------------------------
``rate``
    Resample to CWRU's 12 kHz. The physical frequency axis is then shared, but
    the shaft rate is not (25 Hz vs ~29.9 Hz), so every shaft order and every
    bearing-fault frequency lands in a different bin than the source model saw.
``order``
    Resample so that *samples per revolution* match CWRU's. A window then covers
    the same number of shaft revolutions and the kinematic families land at the
    same bins, which is what ``pdmcore.dsp.ordertrack`` exists to do and what any
    careful practitioner would do before attempting transfer. Bearing geometry
    still differs, so BPFO/BPFI orders still do not coincide (3.585x vs 3.058x).

Reporting cross-dataset accuracy under only one of these would be a choice that
decides the answer. Both are in the matrix.

A STRUCTURAL LIMIT, SAME AS CWRU'S
----------------------------------
MFPT's "three baseline conditions", "three outer-race fault conditions" and
"seven ... varying load" are operating conditions of, as far as the published
description goes, one bearing per condition class -- not independent bearings.
So MFPT, like CWRU, supports a recording-grouped split and does **not** support a
bearing-grouped one. Neither public dataset can answer "does this generalise to a
bearing it has never met". Transfer between the two datasets is the closest
available substitute, and it is what this branch measures.

Provenance
----------
``mfpt.org/fault-data-sets`` now 301-redirects to an ASNT landing page with no
data, and the Wayback Machine holds only the redirect. The files used here came
from two independent GitHub vendorings, and **all ten files common to both
mirrors are byte-identical**:

    https://codeload.github.com/827916600/MFPT_Fault_Data_Sets/tar.gz/refs/heads/main
    https://codeload.github.com/Wu-yanan/CWRU-and-MFPT-data/tar.gz/refs/heads/main

sha256 of every file actually used is recorded in
``results/data_verification.json`` by :func:`verify`.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from cwru import WindowSet

__all__ = ["MfptRecord", "MFPT_RECORDS", "SHAFT_HZ", "verify", "resample_to", "load_records"]

#: Shaft rate of every seeded-fault record, from the ``rate`` field (25 Hz). The
#: three real-world example records have other rates and are not classification
#: data; they are excluded.
SHAFT_HZ = 25.0


@dataclass(frozen=True)
class MfptRecord:
    rel_path: str
    location: str      # normal | inner | outer
    load_lb: float

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.rel_path))[0]

    @property
    def unit(self) -> str:
        return "healthy" if self.location == "normal" else f"{self.location}_mfpt"


_D1 = "1 - Three Baseline Conditions"
_D2 = "2 - Three Outer Race Fault Conditions"
_D3 = "3 - Seven More Outer Race Fault Conditions"
_D4 = "4 - Seven Inner Race Fault Conditions"

MFPT_RECORDS: Tuple[MfptRecord, ...] = tuple(
    [MfptRecord(f"{_D1}/baseline_{i}.mat", "normal", 270.0) for i in (1, 2, 3)]
    + [MfptRecord(f"{_D2}/OuterRaceFault_{i}.mat", "outer", 270.0) for i in (1, 2, 3)]
    + [MfptRecord(f"{_D3}/OuterRaceFault_vload_{i}.mat", "outer", lb)
       for i, lb in zip(range(1, 8), (25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0))]
    + [MfptRecord(f"{_D4}/InnerRaceFault_vload_{i}.mat", "inner", lb)
       for i, lb in zip(range(1, 8), (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0))]
)


def _read(root: str, rec: MfptRecord) -> Tuple[np.ndarray, float, float, float]:
    """``(gs, sr, rate, load)`` out of the single ``bearing`` struct."""
    path = os.path.join(root, rec.rel_path)
    mat = loadmat(path)
    if "bearing" not in mat:
        raise KeyError(f"{path}: no 'bearing' struct; present "
                       f"{sorted(k for k in mat if not k.startswith('__'))}")
    b = mat["bearing"]

    def fld(name: str) -> Optional[float]:
        try:
            return float(np.asarray(b[name][0, 0]).ravel()[0])
        except (KeyError, ValueError, IndexError):
            return None

    gs = np.asarray(b["gs"][0, 0], dtype=np.float64).ravel()
    sr = fld("sr")
    rate = fld("rate")
    load = fld("load")
    if sr is None:
        raise KeyError(f"{path}: no 'sr' field; the acquisition rate cannot be assumed")
    return gs, float(sr), float(rate if rate is not None else SHAFT_HZ), float(load if load is not None else rec.load_lb)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float, *,
                max_den: int = 5000) -> Tuple[np.ndarray, float]:
    """Polyphase resample with a rational ratio. Exact for the ratios used here.

    MFPT's rates are not integer multiples of 12 kHz (48 828 / 12 000 = 4069/1000,
    97 656 / 12 000 = 4069/500), so ``scipy.signal.decimate`` cannot be used and
    the ratio has to go through ``resample_poly``. The achieved ratio is returned
    to the caller's log via :func:`load_records` so a silent 0.4 % frequency-axis
    error cannot hide inside a cross-dataset result.
    """
    frac = Fraction(fs_out / fs_in).limit_denominator(max_den)
    return np.asarray(resample_poly(x, frac.numerator, frac.denominator), dtype=np.float64), float(frac)


def file_digests(root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rec in MFPT_RECORDS:
        h = hashlib.sha256()
        with open(os.path.join(root, rec.rel_path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[rec.stem] = h.hexdigest()
    return out


def verify(root: str) -> Dict[str, object]:
    """Read every record and check its declared metadata against the file.

    Can fail: the shaft rate must be 25 Hz on every seeded-fault record (it is
    the ``rate`` field, and the resampling in :func:`load_records` depends on it),
    the sample rate must be one of the two published values, and the record
    lengths must match the published durations.
    """
    rows: Dict[str, Dict[str, float]] = {}
    problems: List[str] = []
    for rec in MFPT_RECORDS:
        gs, sr, rate, load = _read(root, rec)
        dur = gs.size / sr
        rows[rec.stem] = {"n_samples": float(gs.size), "sr_hz": sr, "rate_hz": rate,
                          "load_lb": load, "duration_s": dur,
                          "rms_g": float(np.sqrt(np.mean(gs**2))),
                          "kurtosis": float(np.mean(((gs - gs.mean()) / gs.std()) ** 4))}
        if abs(rate - SHAFT_HZ) > 0.01:
            problems.append(f"{rec.stem}: rate {rate} Hz != {SHAFT_HZ} Hz")
        if sr not in (48828.0, 97656.0):
            problems.append(f"{rec.stem}: unexpected sr {sr} Hz")
        if not (2.9 < dur < 6.1):
            problems.append(f"{rec.stem}: duration {dur:.2f} s outside 3 s / 6 s")
        if abs(load - rec.load_lb) > 0.5:
            problems.append(f"{rec.stem}: load {load} lb != declared {rec.load_lb} lb")
    return {
        "per_record": rows,
        "sha256": file_digests(root),
        "problems": problems,
        "ok": not problems,
        "summary": {
            "n_records": len(rows),
            "n_problems": len(problems),
            "sample_rates_hz": sorted({r["sr_hz"] for r in rows.values()}),
            "classes": {loc: sum(1 for r in MFPT_RECORDS if r.location == loc)
                        for loc in ("normal", "inner", "outer")},
        },
    }


def load_records(
    root: str,
    *,
    window: int = 2048,
    overlap: float = 0.5,
    n_segments: int = 6,
    target_fs_hz: float = 12000.0,
    align: str = "rate",
    reference_shaft_hz: float = 29.95,
    max_windows_per_record: Optional[int] = None,
) -> WindowSet:
    """Window MFPT onto CWRU's axis under one of the two alignment conventions.

    ``align="rate"``  -> resample to ``target_fs_hz``.
    ``align="order"`` -> resample to ``SHAFT_HZ / reference_shaft_hz * target_fs_hz``
                         and then *call* it ``target_fs_hz``, so samples per
                         revolution match the source domain.
    """
    if align not in ("rate", "order"):
        raise ValueError("align must be 'rate' or 'order'")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(round(window * (1.0 - overlap))))

    Xs: List[np.ndarray] = []
    cols: Dict[str, List] = {k: [] for k in (
        "label10", "label4", "label2", "unit", "recording", "segment",
        "window_id", "load_hp", "shaft_hz", "t_start_s")}
    ratios: Dict[str, float] = {}

    for rec in MFPT_RECORDS:
        gs, sr, rate, load = _read(root, rec)
        fs_out = target_fs_hz if align == "rate" else target_fs_hz * rate / reference_shaft_hz
        x, achieved = resample_to(gs, sr, fs_out)
        ratios[rec.stem] = achieved
        # Effective shaft rate on the resampled axis. Under 'order' alignment the
        # window is relabelled to target_fs_hz, so the apparent shaft rate becomes
        # the reference one -- which is the whole point.
        shaft_apparent = rate if align == "rate" else reference_shaft_hz
        n_win = (x.size - window) // step + 1
        if n_win < n_segments:
            raise ValueError(f"{rec.stem}: only {n_win} windows for n_segments={n_segments}")
        starts = np.arange(n_win) * step
        seg_of = np.minimum((starts * n_segments) // (x.size - window + 1), n_segments - 1)
        keep = np.arange(n_win)
        if max_windows_per_record is not None and n_win > max_windows_per_record:
            per_seg = max(1, max_windows_per_record // n_segments)
            picks = []
            for s in range(n_segments):
                in_seg = np.flatnonzero(seg_of == s)
                take = np.linspace(0, in_seg.size - 1, num=min(per_seg, in_seg.size))
                picks.append(in_seg[np.unique(take.round().astype(int))])
            keep = np.unique(np.concatenate(picks))
        for i in keep:
            s0 = int(starts[i])
            Xs.append(x[s0 : s0 + window])
            cols["label10"].append(rec.unit)
            cols["label4"].append(rec.location)
            cols["label2"].append("healthy" if rec.location == "normal" else "faulty")
            cols["unit"].append(rec.unit)
            cols["recording"].append(rec.stem)
            cols["segment"].append(f"{rec.stem}#{int(seg_of[i])}")
            cols["window_id"].append(f"{rec.stem}@{s0}")
            # Radial load in lb has no hp equivalent; the column is kept for
            # shape compatibility and holds the lb value, which is why nothing
            # downstream compares a CWRU load_hp to an MFPT one.
            cols["load_hp"].append(int(round(load)))
            cols["shaft_hz"].append(shaft_apparent)
            cols["t_start_s"].append(s0 / target_fs_hz)

    ws = WindowSet(
        X=np.vstack(Xs),
        label10=np.asarray(cols["label10"]),
        label4=np.asarray(cols["label4"]),
        label2=np.asarray(cols["label2"]),
        unit=np.asarray(cols["unit"]),
        recording=np.asarray(cols["recording"]),
        segment=np.asarray(cols["segment"]),
        window_id=np.asarray(cols["window_id"]),
        load_hp=np.asarray(cols["load_hp"], dtype=int),
        shaft_hz=np.asarray(cols["shaft_hz"], dtype=np.float64),
        t_start_s=np.asarray(cols["t_start_s"], dtype=np.float64),
        fs_hz=target_fs_hz,
        source=f"mfpt[{align}]",
    )
    ws.resample_ratios = ratios  # type: ignore[attr-defined]
    return ws
