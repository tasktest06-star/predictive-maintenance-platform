"""CWRU 12 kHz drive-end loader, with the dataset's own traps handled explicitly.

The Case Western Reserve University bearing dataset is the benchmark on which
essentially the whole supervised bearing-CNN literature reports 99 %+ accuracy.
Loading it correctly is harder than it looks, and three of the traps below are
load-bearing for the numbers this branch reports.

TRAP 1 -- THE SAMPLE RATE OF THE NORMAL BASELINE, MEASURED NOT ASSUMED
----------------------------------------------------------------------
This loader was first written believing the widely-repeated claim that CWRU's
normal-baseline records (97/98/99/100) are acquired at 48 kHz while the
seeded-fault records are at 12 kHz. The sample counts support it: 98/99/100 hold
~485 000 samples, which is 10.1 s at 48 kHz -- matching the 10 s length of every
fault record -- and 40.4 s at 12 kHz.

:func:`verify_sample_rates` refused it. Under the 12 kHz hypothesis record 97 has
a clean shaft-order family at 30.03 / 60.06 / 89.90 Hz, at 7.4x / 11.9x / 64.7x
the 5-400 Hz noise floor, for a file-declared 1796 rpm (29.93 Hz). Under the
48 kHz hypothesis the same bands hold 2.6x / 3.7x / 3.1x, i.e. nothing. Record
100 behaves identically. **The files served from
``engineering.case.edu/sites/default/files/{97,98,99,100}.mat`` are 12 kHz**;
they are simply longer records. The belief was wrong and the check said so, which
is the entire reason the check exists and runs before any model.

The risk is still real for anyone who mixes CWRU's separately published 48 kHz
drive-end fault set with the 12 kHz set, so the *cost* of the mistake is measured
anyway: ``rate_confound=True`` resamples the healthy class by 4x in time, which
is exactly what a 48 kHz record read as 12 kHz looks like, and the matrix reports
the free accuracy that buys. It is labelled a synthetic confound everywhere it
appears, because it is not a property of the bytes on disk.

TRAP 2 -- 99.mat CONTAINS TWO RECORDINGS
----------------------------------------
``99.mat`` as published holds ``X098_DE_time`` *and* ``X099_DE_time`` (and an
``ans`` variable). A loader that takes "the first key ending in ``_DE_time``"
gets 98's data out of 99's file, silently, with the right shape and plausible
content. This loader addresses variables by the explicit name the record table
declares, and raises if it is absent.

TRAP 3 -- "ONE BEARING PER FILE" IS FALSE
-----------------------------------------
There are 40 records but only **10 physical bearings**: one healthy, and one for
each (fault location x defect diameter) pair. The four records per bearing are
the same bearing re-run at 0, 1, 2 and 3 hp motor load. So a split that holds out
whole *files* -- which is what every "bearing-wise split" in the surrounding code
on this repository's ``impl/*`` branches actually does -- still puts the same
physical bearing on both sides. See :data:`GROUP_KEYS` and
``research/supervised_cnn/01_protocol.md``.

The honest consequence: **CWRU cannot support a bearing-grouped evaluation at
all.** With one unit per class, holding out a unit holds out a class. The
strongest generalisation test the dataset admits is cross-load, which is strictly
weaker. That is a property of the benchmark, not of any model trained on it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate, resample_poly

__all__ = [
    "CwruRecord",
    "CWRU_RECORDS",
    "GROUP_KEYS",
    "WindowSet",
    "load_records",
    "verify_sample_rates",
    "verify_fault_labels",
    "file_digests",
]

#: Nominal shaft speed per motor load, from the dataset documentation. Used only
#: when the .mat file carries no RPM variable (98.mat does not).
NOMINAL_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

#: Every record in this file set, measured by :func:`verify_sample_rates`.
FS_FAULT_HZ = 12000.0
FS_NORMAL_HZ = 12000.0

#: Hypotheses :func:`verify_sample_rates` discriminates between. Deliberately a
#: literal and NOT ``(FS_FAULT_HZ, FS_NORMAL_HZ)``: once both of those became
#: 12 kHz, a candidate list built from them left one hypothesis, the two scores
#: were equal by construction, and the check reported 40/40 agreement without
#: being able to disagree. That version of this file existed for about ten
#: minutes and is the kind of thing ``docs/08_branch_catalogue.md`` tabulates.
CANDIDATE_RATES_HZ = (12000.0, 48000.0)


@dataclass(frozen=True)
class CwruRecord:
    """One published CWRU recording and everything needed to interpret it."""

    stem: str          # file name without .mat
    var: str           # the exact variable holding the drive-end channel
    location: str      # normal | inner | outer | ball
    size_in: float     # seeded defect diameter in inches, 0.0 for normal
    load_hp: int       # motor load
    fs_hz: float       # acquisition rate of THIS record

    @property
    def unit(self) -> str:
        """The physical bearing. Four records share each one."""
        return "healthy" if self.location == "normal" else f"{self.location}_{self.size_in:.3f}"

    @property
    def label10(self) -> str:
        """The 10-class label used by the WDCNN/TICNN literature."""
        return self.unit

    @property
    def label4(self) -> str:
        """Location-only label: the only label space shared with MFPT."""
        return self.location

    @property
    def label2(self) -> str:
        return "healthy" if self.location == "normal" else "faulty"


def _r(stems: Sequence[str], location: str, size_in: float) -> List[CwruRecord]:
    return [
        CwruRecord(s, f"X{int(s):03d}_DE_time", location, size_in, load, FS_FAULT_HZ)
        for load, s in enumerate(stems)
    ]


#: The canonical 12 kHz drive-end table. Record order within each group of four
#: is motor load 0, 1, 2, 3 hp. Verified against bearing kinematics by
#: :func:`verify_fault_labels` rather than trusted -- a transposed row here would
#: look like a model that cannot tell inner from outer race.
CWRU_RECORDS: Tuple[CwruRecord, ...] = tuple(
    [
        CwruRecord("97", "X097_DE_time", "normal", 0.0, 0, FS_NORMAL_HZ),
        CwruRecord("98", "X098_DE_time", "normal", 0.0, 1, FS_NORMAL_HZ),
        # 99.mat holds X098_DE_time as well; the explicit variable name is the
        # whole point of this column.
        CwruRecord("99", "X099_DE_time", "normal", 0.0, 2, FS_NORMAL_HZ),
        CwruRecord("100", "X100_DE_time", "normal", 0.0, 3, FS_NORMAL_HZ),
    ]
    + _r(("105", "106", "107", "108"), "inner", 0.007)
    + _r(("169", "170", "171", "172"), "inner", 0.014)
    + _r(("209", "210", "211", "212"), "inner", 0.021)
    + _r(("118", "119", "120", "121"), "ball", 0.007)
    + _r(("185", "186", "187", "188"), "ball", 0.014)
    + _r(("222", "223", "224", "225"), "ball", 0.021)
    + _r(("130", "131", "132", "133"), "outer", 0.007)
    + _r(("197", "198", "199", "200"), "outer", 0.014)
    + _r(("234", "235", "236", "237"), "outer", 0.021)
)

#: The grouping keys available, in increasing strictness. ``window`` is not a
#: grouping at all and is listed so the literature's default has a name.
GROUP_KEYS = ("window", "segment", "recording", "unit")


@dataclass
class WindowSet:
    """Windows plus every column a split or an audit might need.

    ``X`` is ``(n_windows, window)`` raw acceleration in g at a single, uniform
    sample rate. Everything else is a per-window column of the same length.
    """

    X: np.ndarray
    label10: np.ndarray
    label4: np.ndarray
    label2: np.ndarray
    unit: np.ndarray
    recording: np.ndarray
    segment: np.ndarray
    window_id: np.ndarray
    load_hp: np.ndarray
    shaft_hz: np.ndarray
    t_start_s: np.ndarray
    fs_hz: float
    source: str = "cwru"

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def groups(self, key: str) -> np.ndarray:
        if key == "window":
            return self.window_id
        if key == "segment":
            return self.segment
        if key == "recording":
            return self.recording
        if key == "unit":
            return self.unit
        raise ValueError(f"unknown group key {key!r}; expected one of {GROUP_KEYS}")

    def labels(self, n_classes: int) -> np.ndarray:
        return {10: self.label10, 4: self.label4, 2: self.label2}[n_classes]

    def take(self, idx: np.ndarray) -> "WindowSet":
        idx = np.asarray(idx, dtype=int)
        return WindowSet(
            X=self.X[idx],
            label10=self.label10[idx],
            label4=self.label4[idx],
            label2=self.label2[idx],
            unit=self.unit[idx],
            recording=self.recording[idx],
            segment=self.segment[idx],
            window_id=self.window_id[idx],
            load_hp=self.load_hp[idx],
            shaft_hz=self.shaft_hz[idx],
            t_start_s=self.t_start_s[idx],
            fs_hz=self.fs_hz,
            source=self.source,
        )


def _read_channel(root: str, rec: CwruRecord) -> Tuple[np.ndarray, Optional[float]]:
    path = os.path.join(root, f"{rec.stem}.mat")
    mat = loadmat(path)
    if rec.var not in mat:
        raise KeyError(
            f"{path} has no variable {rec.var!r}; present: "
            f"{sorted(k for k in mat if not k.startswith('__'))}. "
            "Addressing the drive-end channel by 'first key ending in _DE_time' "
            "is what makes 99.mat return 98's data."
        )
    x = np.asarray(mat[rec.var], dtype=np.float64).ravel()
    rpm = None
    for k in mat:
        if k.endswith("RPM"):
            rpm = float(np.asarray(mat[k]).ravel()[0])
    return x, rpm


def file_digests(root: str) -> Dict[str, str]:
    """sha256 of each record actually used, so a result can be tied to the bytes."""
    out: Dict[str, str] = {}
    for rec in CWRU_RECORDS:
        path = os.path.join(root, f"{rec.stem}.mat")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[rec.stem] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------
# Verification -- run before any model, and it can fail
# ---------------------------------------------------------------------------
def verify_sample_rates(root: str, n_fft: int = 1 << 16) -> Dict[str, Dict[str, float]]:
    """Measure which acquisition rate each record was taken at.

    Method: under a candidate rate, score the shaft-order family by the mean
    spectral magnitude in narrow bands at 1x, 2x and 3x of the *file's own* RPM,
    normalised by the local median magnitude. The correct rate puts real lines
    under those bands; the wrong rate puts them at 1/4 (or 4x) of the shaft
    orders, where a rotating machine has no reason to have a line.

    This is a check that can fail: if the 48 kHz hypothesis did not win on
    97/98/99/100 and lose on the fault records, the loader's ``fs_hz`` column
    would be wrong and the tests in ``prototype/tests`` would say so.
    """
    out: Dict[str, Dict[str, float]] = {}
    for rec in CWRU_RECORDS:
        x, rpm = _read_channel(root, rec)
        rpm = rpm if rpm else NOMINAL_RPM[rec.load_hp]
        n = min(n_fft, x.size)
        seg = x[:n] * np.hanning(n)
        mag = np.abs(np.fft.rfft(seg))
        scores = {}
        for fs in CANDIDATE_RATES_HZ:
            freqs = np.fft.rfftfreq(n, 1.0 / fs)
            shaft = rpm / 60.0
            # Local median over 0-400 Hz as the reference floor.
            floor = float(np.median(mag[(freqs > 5) & (freqs < 400)]))
            got = []
            for order in (1.0, 2.0, 3.0):
                f0 = order * shaft
                band = (freqs > f0 * 0.985) & (freqs < f0 * 1.015)
                got.append(float(np.max(mag[band])) / max(floor, 1e-30) if np.any(band) else 0.0)
            scores[fs] = float(np.exp(np.mean(np.log(np.maximum(got, 1e-9)))))
        best = max(scores, key=lambda f: scores[f])
        out[rec.stem] = {
            "rpm": rpm,
            "n_samples": float(x.size),
            "score_12k": scores[12000.0],
            "score_48k": scores[48000.0],
            "margin": scores[best] / max(min(scores.values()), 1e-12),
            "declared_fs_hz": rec.fs_hz,
            "measured_fs_hz": best,
        }
    return out


#: ``location`` -> the ``pdmcore.dsp.bearing`` evidence mode it should excite.
MODE_OF = {"outer": "outer_race", "inner": "inner_race", "ball": "rolling_element"}


def verify_fault_labels(root: str, *, bearing_name: str = "SKF6205") -> Dict[str, object]:
    """Check the record table against bearing kinematics, falsifiably.

    Uses ``pdmcore.dsp.envelope`` and ``pdmcore.dsp.bearing``: band-pass, rectify,
    envelope-spectrum, then the harmonic SNR at BPFO / BPFI / BSF.

    Per-record agreement is reported but is NOT the gate, because a per-record
    gate here could not pass: CWRU's rolling-element records famously carry no
    clean BSF line (the defect enters and leaves the load zone, and the
    modulation is weak), and the 0.014" outer-race records are a known anomaly in
    the set. A gate that must fail is as useless as one that cannot.

    The gate is a **permutation test on the table itself**: total kinematic
    evidence under the declared location assignment must beat both cyclic
    relabellings of (inner, outer, ball). A transposed row -- the realistic
    hand-editing error, and one that would make any model look unable to tell
    inner race from outer race -- moves evidence from the declared mode to
    another and fails this. Chance-level agreement does not pass it.
    """
    from pdmcore.dsp.bearing import BEARING_CATALOG, bearing_evidence
    from pdmcore.dsp.envelope import envelope_spectrum

    geom = BEARING_CATALOG[bearing_name]
    modes = ("outer_race", "inner_race", "rolling_element")
    per_record: Dict[str, Dict[str, float]] = {}
    for rec in CWRU_RECORDS:
        if rec.location == "normal":
            continue
        x, rpm = _read_channel(root, rec)
        rpm = rpm if rpm else NOMINAL_RPM[rec.load_hp]
        env = envelope_spectrum(
            x[: int(4 * rec.fs_hz)], rec.fs_hz,
            band_hz=(2000.0, 5000.0), n_fft=4096, decimate_to_hz=2000.0,
        )
        ev = bearing_evidence(env, geom, rpm / 60.0, n_harmonics=5)
        snr = {m: float(ev[m]["peak_snr"]) for m in modes}
        predicted = max(snr, key=lambda k: snr[k])
        per_record[rec.stem] = {
            "location": rec.location,
            "declared": MODE_OF[rec.location],
            "predicted": predicted,
            "agrees": float(predicted == MODE_OF[rec.location]),
            **{f"snr_{k}": v for k, v in snr.items()},
        }

    # Permutation test. Score = mean, over fault records, of the declared mode's
    # SNR divided by the mean SNR of the three modes on that record -- a
    # within-record normalisation, so a record with a loud envelope cannot
    # dominate.
    locations = ("inner", "outer", "ball")
    perms = {
        "declared": {"inner": "inner", "outer": "outer", "ball": "ball"},
        "cycle1": {"inner": "outer", "outer": "ball", "ball": "inner"},
        "cycle2": {"inner": "ball", "outer": "inner", "ball": "outer"},
    }
    scores: Dict[str, float] = {}
    for name, perm in perms.items():
        vals = []
        for stem, r in per_record.items():
            mode = MODE_OF[perm[str(r["location"])]]
            trio = [float(r[f"snr_{m}"]) for m in modes]
            vals.append(float(r[f"snr_{mode}"]) / max(float(np.mean(trio)), 1e-12))
        scores[name] = float(np.mean(vals))
    del locations
    best = max(scores, key=lambda k: scores[k])
    by_location = {
        loc: float(np.mean([r["agrees"] for r in per_record.values() if r["location"] == loc]))
        for loc in ("inner", "outer", "ball")
    }
    return {
        "per_record": per_record,
        "n_records": len(per_record),
        "n_agree": int(sum(int(r["agrees"]) for r in per_record.values())),
        "agreement_by_location": by_location,
        "permutation_scores": scores,
        "permutation_best": best,
        "ok": best == "declared",
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_records(
    root: str,
    *,
    window: int = 2048,
    overlap: float = 0.5,
    n_segments: int = 6,
    rate_confound: bool = False,
    max_windows_per_record: Optional[int] = None,
) -> WindowSet:
    """Window every record onto a single 12 kHz axis.

    Parameters
    ----------
    overlap
        Window overlap. 0.5 is the literature's default and is also the
        mechanism of the random-split leak: at 50 % overlap, adjacent windows
        share half their samples, so a random partition puts near-duplicates on
        both sides. Kept at the literature default so that the leak being
        measured is the literature's leak.
    n_segments
        Each record is cut into this many contiguous, equal segments. The
        ``segment`` group key holds whole segments out, which removes the
        overlap leak while leaving the recording on both sides -- that isolates
        flaw 1 from flaw 2.
    rate_confound
        SYNTHETIC. Resample the healthy records by 4x in time, which is what a
        48 kHz record read as 12 kHz looks like, to measure how much accuracy a
        sample-rate mismatch between classes hands a classifier for free. The
        files on disk do not have this problem -- see TRAP 1 -- but a pipeline
        that mixes CWRU's 12 kHz and 48 kHz published sets does.
    max_windows_per_record
        Uniformly thin the windows of every record to at most this many, keeping
        segment coverage. Used to bring the upstream 1.3 M-parameter model inside
        this host's CPU budget; stated in every result that uses it.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(round(window * (1.0 - overlap))))

    Xs: List[np.ndarray] = []
    cols: Dict[str, List] = {k: [] for k in (
        "label10", "label4", "label2", "unit", "recording", "segment",
        "window_id", "load_hp", "shaft_hz", "t_start_s")}

    for rec in CWRU_RECORDS:
        x, rpm = _read_channel(root, rec)
        rpm = rpm if rpm else NOMINAL_RPM[rec.load_hp]
        if rec.fs_hz != FS_FAULT_HZ:
            factor = int(round(rec.fs_hz / FS_FAULT_HZ))
            if abs(rec.fs_hz / FS_FAULT_HZ - factor) > 1e-6:
                raise ValueError(f"{rec.stem}: {rec.fs_hz} is not an integer multiple of 12 kHz")
            x = np.asarray(decimate(x, factor, ftype="fir", zero_phase=True), dtype=np.float64)
        if rate_confound and rec.location == "normal":
            # resample_poly(up=4, down=1): the whole spectrum compressed into the
            # bottom quarter of the band, i.e. a 48 kHz record read as 12 kHz.
            x = np.asarray(resample_poly(x, 4, 1), dtype=np.float64)
        n_win = (x.size - window) // step + 1
        if n_win < n_segments:
            raise ValueError(f"{rec.stem}: only {n_win} windows, need >= n_segments={n_segments}")
        starts = np.arange(n_win) * step
        seg_of = np.minimum((starts * n_segments) // (x.size - window + 1), n_segments - 1)
        keep = np.arange(n_win)
        if max_windows_per_record is not None and n_win > max_windows_per_record:
            # Thin uniformly inside each segment so every segment survives: a
            # random subsample could empty one and silently break the
            # segment-grouped split.
            per_segment = max(1, max_windows_per_record // n_segments)
            picks = []
            for s in range(n_segments):
                in_seg = np.flatnonzero(seg_of == s)
                take = np.linspace(0, in_seg.size - 1, num=min(per_segment, in_seg.size))
                picks.append(in_seg[np.unique(take.round().astype(int))])
            keep = np.unique(np.concatenate(picks))
        for i in keep:
            s0 = int(starts[i])
            Xs.append(x[s0 : s0 + window])
            cols["label10"].append(rec.label10)
            cols["label4"].append(rec.label4)
            cols["label2"].append(rec.label2)
            cols["unit"].append(rec.unit)
            cols["recording"].append(rec.stem)
            cols["segment"].append(f"{rec.stem}#{int(seg_of[i])}")
            cols["window_id"].append(f"{rec.stem}@{s0}")
            cols["load_hp"].append(rec.load_hp)
            cols["shaft_hz"].append(rpm / 60.0)
            cols["t_start_s"].append(s0 / FS_FAULT_HZ)
    return WindowSet(
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
        fs_hz=FS_FAULT_HZ,
        source="cwru",
    )
