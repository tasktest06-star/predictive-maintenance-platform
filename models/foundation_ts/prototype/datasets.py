"""Windowed loaders for the measured bearing datasets, and for the simulator.

This is the first module in the repository that reads **measured** data. Until
now every detection number here has been a number about
``pdmcore.sim.machine``; ``docs/07_production_readiness.md`` names CWRU and MFPT
as the gate and this is the loader that opens it.

Three rules the loaders enforce, because all three have burned published
bearing-fault results:

1. **A window never straddles two recordings.** Windows are cut inside one
   recording and carry that recording's id as their ``group``, so
   :func:`pdmcore.eval.splits.group_split` and
   :func:`pdmcore.eval.splits.leakage_report` can do their job.
2. **Nothing is resampled, filtered or normalised here.** The raw stored g
   values go straight into the windows. CWRU drive-end records are 12 kHz;
   MFPT is 97 656 Hz for the baseline/270 lb records and 48 828 Hz for the
   varying-load ones. The per-window sample rate travels with the window and
   every downstream consumer must honour it -- silently assuming one rate is how
   a bearing frequency ends up 2x wrong.
3. **Geometry comes from the dataset manifest, not from a catalogue.** The
   manifest's geometry was verified against the defect-frequency *orders* the
   dataset publishers print, which is a check the catalogue in
   ``pdmcore.dsp.bearing`` failed twice -- see ``models/foundation_ts/research``
   and ``core/tests/test_bearing_catalogue.py``.

Data location
-------------
The datasets are **not** committed: CWRU is 105 MB of ``.mat`` and MFPT is
CC BY-NC-SA 4.0. What is committed is the manifest digest and the loader, so a
reader can reproduce the download. Default root is
``/workshop/predictive_maintaince/data_raw``; override with ``--data-root``.
Every loader raises with the exact missing path rather than falling back to the
simulator, because a silent fallback to simulated data is precisely the failure
this branch exists to avoid.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if os.path.join(_REPO, "core") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "core"))

from pdmcore.dsp.bearing import BearingGeometry  # noqa: E402

__all__ = [
    "WindowSet",
    "DEFAULT_DATA_ROOT",
    "MissingDataset",
    "load_cwru",
    "load_mfpt",
    "load_sim_fleet",
    "load_sim_degradation",
]

#: Where the download agent put the converted datasets. Overridable everywhere.
DEFAULT_DATA_ROOT = "/workshop/predictive_maintaince/data_raw"


class MissingDataset(FileNotFoundError):
    """The measured dataset is not on this machine.

    Carries the path and the reproduction command. Deliberately *not* caught
    anywhere: an experiment that cannot find its measured data must fail, not
    quietly become a simulator experiment.
    """


@dataclass
class WindowSet:
    """A set of equal-length acceleration windows with their ground truth.

    Attributes
    ----------
    x
        ``(n, L)`` acceleration in g. Raw, as stored by the dataset.
    fs_hz, shaft_hz
        ``(n,)`` per-window sample rate and shaft frequency. Per-window because
        MFPT mixes two sample rates and CWRU's speed moves with load.
    y
        ``(n,)`` 1 = faulty, 0 = healthy. The one-class detectors never see
        this during fitting; it exists only for scoring.
    label
        ``(n,)`` the fault class string ("normal", "inner_race", ...).
    group
        ``(n,)`` recording id -- the unit of grouping for the split.
    cond
        ``(n,)`` operating condition (CWRU load in hp, MFPT radial load in lb).
    bearing_key
        ``(n,)`` key into :attr:`geometries`.
    t_index
        ``(n,)`` window position within its recording, so a time split inside a
        recording is possible.
    """

    x: np.ndarray
    fs_hz: np.ndarray
    shaft_hz: np.ndarray
    y: np.ndarray
    label: np.ndarray
    group: np.ndarray
    cond: np.ndarray
    bearing_key: np.ndarray
    t_index: np.ndarray
    geometries: Dict[str, BearingGeometry]
    name: str
    provenance: Dict[str, object]

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        for field_name in ("fs_hz", "shaft_hz", "y", "label", "group", "cond",
                           "bearing_key", "t_index"):
            got = getattr(self, field_name).shape[0]
            if got != n:
                raise ValueError(f"{field_name} has {got} rows, x has {n}")
        missing = sorted(set(self.bearing_key.tolist()) - set(self.geometries))
        if missing:
            raise ValueError(f"no geometry for bearing key(s) {missing}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def window_samples(self) -> int:
        return int(self.x.shape[1])

    def subset(self, idx: Sequence[int]) -> "WindowSet":
        i = np.asarray(idx, dtype=int)
        return WindowSet(
            x=self.x[i], fs_hz=self.fs_hz[i], shaft_hz=self.shaft_hz[i],
            y=self.y[i], label=self.label[i], group=self.group[i],
            cond=self.cond[i], bearing_key=self.bearing_key[i],
            t_index=self.t_index[i], geometries=self.geometries,
            name=f"{self.name}[{i.size}]", provenance=self.provenance,
        )

    def summary(self) -> Dict[str, object]:
        labels, counts = np.unique(self.label, return_counts=True)
        return {
            "name": self.name,
            "n_windows": len(self),
            "window_samples": self.window_samples,
            "n_groups": int(np.unique(self.group).size),
            "fs_hz": sorted(set(float(v) for v in self.fs_hz)),
            "conditions": sorted(set(str(v) for v in self.cond)),
            "label_counts": {str(k): int(v) for k, v in zip(labels, counts)},
            "n_healthy": int(np.count_nonzero(self.y == 0)),
            "n_faulty": int(np.count_nonzero(self.y == 1)),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require(path: str, what: str, how: str) -> str:
    if not os.path.exists(path):
        raise MissingDataset(
            f"{what} not found at {path}.\n"
            f"This experiment refuses to run on simulated data in place of "
            f"measured data.\n"
            f"To reproduce: {how}"
        )
    return path


def _geometry_from_manifest(rec: Dict, name: str) -> BearingGeometry:
    """Build the geometry from the manifest's verified numbers.

    The manifest records ``n_balls`` and the ball/pitch diameters alongside the
    publisher's defect-frequency orders, and the download step checked that the
    former reproduce the latter. Using it directly means this experiment does
    not depend on the repository's own bearing catalogue being right -- which,
    as it turned out, it was not for two of its seven entries.
    """
    g = rec["bearing_geometry"]
    return BearingGeometry(
        name=name,
        n_balls=int(g["n_balls"]),
        ball_d=float(g["ball_dia_in"]),
        pitch_d=float(g["pitch_dia_in"]),
        contact_angle_deg=float(g.get("contact_angle_deg", 0.0) or 0.0),
    )


def _cut_windows(sig: np.ndarray, length: int, hop: int,
                 max_windows: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Non-overlapping (or hop-spaced) windows plus their index in the record.

    ``hop == length`` is the default and the only configuration used for
    train/test data. Overlap is available for the healthy *calibration* pool
    only, and when it is used it is recorded in the results JSON, because
    overlapping windows are correlated and inflate the effective sample count
    a threshold is calibrated from.
    """
    n = sig.size
    starts = np.arange(0, max(0, n - length + 1), hop, dtype=int)
    if max_windows is not None and starts.size > max_windows:
        starts = starts[:max_windows]
    if starts.size == 0:
        return np.empty((0, length), dtype=np.float64), np.empty(0, dtype=int)
    out = np.empty((starts.size, length), dtype=np.float64)
    for i, s in enumerate(starts):
        out[i] = sig[s:s + length]
    return out, np.arange(starts.size, dtype=int)


# ---------------------------------------------------------------------------
# CWRU
# ---------------------------------------------------------------------------
#: Fault keys in the CWRU manifest map to these binary/diagnostic labels.
_CWRU_LABEL = {
    "normal": "normal",
    "inner_race": "inner_race",
    "ball": "ball",
    "outer_race": "outer_race",
}


def load_cwru(
    data_root: str = DEFAULT_DATA_ROOT,
    *,
    channel: str = "DE",
    window_samples: int = 2048,
    hop_samples: Optional[int] = None,
    loads_hp: Optional[Sequence[int]] = None,
    fault_sizes_in: Optional[Sequence[float]] = None,
    max_windows_per_record: Optional[int] = None,
) -> WindowSet:
    """CWRU 12 kHz bearing data, windowed.

    Parameters
    ----------
    channel
        ``"DE"`` (drive end, where the seeded fault is), ``"FE"`` (fan end) or
        ``"BA"`` (base plate). DE is the only channel used in the headline
        results; FE and BA exist in the npz and are a transfer test.
    window_samples
        2048 by default -- matching ``impl/ml-transformer-faultformer``'s patch
        geometry so the three arms see the same acquisition length. 2048 at
        12 kHz is 171 ms, i.e. ~5 shaft revolutions at 1750 rpm and ~18 BPFO
        impacts, enough for an envelope spectrum to resolve the impact rate.

    Notes
    -----
    The manifest flags that ``99.mat`` (Normal, 2 hp) ships with the 1 hp
    variables alongside its own; the converter keyed 2 hp off ``X099_*``. Taking
    the first ``*_DE_time`` variable out of that file gives 1 hp data labelled
    2 hp, which is a real, silent label error present in a good deal of
    published CWRU tooling.
    """
    base = os.path.join(data_root, "cwru", "converted")
    npz_path = _require(os.path.join(base, "cwru_12kDE.npz"), "CWRU npz",
                        "see models/foundation_ts/research/02_protocol.md "
                        "(section 'Obtaining the data')")
    man_path = _require(os.path.join(base, "cwru_12kDE.json"), "CWRU manifest",
                        "same")
    with open(man_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    hop = int(hop_samples if hop_samples is not None else window_samples)

    xs: List[np.ndarray] = []
    fs_l, sh_l, y_l, lab_l, grp_l, cond_l, bk_l, ti_l = ([] for _ in range(8))
    geoms: Dict[str, BearingGeometry] = {}
    used: List[str] = []

    with np.load(npz_path, allow_pickle=False) as npz:
        for key in sorted(npz.files):
            if not key.startswith(f"cwru_{channel}_"):
                continue
            rec = manifest["recordings"][key]
            if loads_hp is not None and rec.get("load_hp") not in set(loads_hp):
                continue
            size = rec.get("fault_size_in")
            if (fault_sizes_in is not None and rec["label"] != "normal"
                    and size not in set(fault_sizes_in)):
                continue
            sig = np.asarray(npz[key], dtype=np.float64).ravel()
            win, ti = _cut_windows(sig, window_samples, hop, max_windows_per_record)
            if win.shape[0] == 0:
                continue
            # Measured speed where the .mat carries it; nominal otherwise. The
            # manifest records which, and 98/99.mat carry no RPM variable.
            rpm = rec.get("rpm_from_mat") or rec["rpm"]
            bkey = rec["bearing_geometry"]["bearing"].split(" ")[0] + "_" + channel
            geoms.setdefault(bkey, _geometry_from_manifest(rec, bkey))
            n = win.shape[0]
            xs.append(win)
            fs_l.append(np.full(n, float(rec["fs_hz"])))
            sh_l.append(np.full(n, float(rpm) / 60.0))
            y_l.append(np.full(n, 0 if rec["label"] == "normal" else 1, dtype=int))
            lab_l.append(np.full(n, _CWRU_LABEL[rec["label"]], dtype=object))
            grp_l.append(np.full(n, key, dtype=object))
            cond_l.append(np.full(n, f"{rec['load_hp']}hp", dtype=object))
            bk_l.append(np.full(n, bkey, dtype=object))
            ti_l.append(ti)
            used.append(key)

    if not xs:
        raise MissingDataset(
            f"CWRU npz at {npz_path} contains no records for channel {channel!r} "
            f"under loads={loads_hp} sizes={fault_sizes_in}"
        )
    return WindowSet(
        x=np.vstack(xs),
        fs_hz=np.concatenate(fs_l), shaft_hz=np.concatenate(sh_l),
        y=np.concatenate(y_l),
        label=np.concatenate(lab_l).astype(str),
        group=np.concatenate(grp_l).astype(str),
        cond=np.concatenate(cond_l).astype(str),
        bearing_key=np.concatenate(bk_l).astype(str),
        t_index=np.concatenate(ti_l),
        geometries=geoms, name=f"cwru_{channel}",
        provenance={
            "dataset": "CWRU Bearing Data Center, 12 kHz drive-end subset",
            "source_base_url": manifest.get("source_base_url"),
            "download_utc": manifest.get("download_utc"),
            "records_used": used,
            "channel": channel,
            "window_samples": window_samples,
            "hop_samples": hop,
            "manifest_sha256_count": len(manifest.get("sha256", {})),
            "licence_note": "CWRU data is published for research use; not "
                            "redistributed in this repository.",
        },
    )


# ---------------------------------------------------------------------------
# MFPT
# ---------------------------------------------------------------------------
def load_mfpt(
    data_root: str = DEFAULT_DATA_ROOT,
    *,
    window_samples: int = 2048,
    hop_samples: Optional[int] = None,
    fs_hz_filter: Optional[float] = 97656.0,
    max_windows_per_record: Optional[int] = None,
) -> WindowSet:
    """MFPT bearing data, windowed.

    ``fs_hz_filter`` defaults to the 97 656 Hz subset (3 baseline + 3
    outer-race at 270 lb). That is the only MFPT subset where healthy and
    faulty records share both a sample rate **and** a radial load, so it is the
    only one that supports a clean one-class comparison; the 48 828 Hz
    varying-load records are loaded by passing ``fs_hz_filter=48828.0`` and are
    reported separately because they confound fault with load *and* with rate.

    At 97 656 Hz a 2048-sample window is 21 ms -- half a shaft revolution at
    1500 rpm, and ~1.7 BPFO impacts. That is short, and it is a real limitation
    of this comparison rather than a choice: holding the window length fixed
    across datasets is what makes the foundation-model context identical, and
    holding the *rotational* length fixed would change the context. Both are
    reported; see ``mfpt_longwin`` in the results JSON.
    """
    base = os.path.join(data_root, "mfpt", "converted")
    npz_path = _require(os.path.join(base, "mfpt.npz"), "MFPT npz",
                        "see models/foundation_ts/research/02_protocol.md")
    man_path = _require(os.path.join(base, "mfpt.json"), "MFPT manifest", "same")
    with open(man_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    hop = int(hop_samples if hop_samples is not None else window_samples)

    xs: List[np.ndarray] = []
    fs_l, sh_l, y_l, lab_l, grp_l, cond_l, bk_l, ti_l = ([] for _ in range(8))
    geoms: Dict[str, BearingGeometry] = {}
    used: List[str] = []

    with np.load(npz_path, allow_pickle=False) as npz:
        for key in sorted(npz.files):
            rec = manifest["recordings"][key]
            if fs_hz_filter is not None and float(rec["fs_hz"]) != float(fs_hz_filter):
                continue
            sig = np.asarray(npz[key], dtype=np.float64).ravel()
            win, ti = _cut_windows(sig, window_samples, hop, max_windows_per_record)
            if win.shape[0] == 0:
                continue
            bkey = "NICE_MFPT_verified"
            geoms.setdefault(bkey, _geometry_from_manifest(rec, bkey))
            n = win.shape[0]
            xs.append(win)
            fs_l.append(np.full(n, float(rec["fs_hz"])))
            sh_l.append(np.full(n, float(rec["shaft_rate_hz"])))
            y_l.append(np.full(n, 0 if rec["label"] == "normal" else 1, dtype=int))
            lab_l.append(np.full(n, rec["label"], dtype=object))
            grp_l.append(np.full(n, key, dtype=object))
            cond_l.append(np.full(n, f"{rec.get('radial_load_lbs')}lb", dtype=object))
            bk_l.append(np.full(n, bkey, dtype=object))
            ti_l.append(ti)
            used.append(key)

    if not xs:
        raise MissingDataset(
            f"MFPT npz at {npz_path} has no records at fs={fs_hz_filter}"
        )
    return WindowSet(
        x=np.vstack(xs),
        fs_hz=np.concatenate(fs_l), shaft_hz=np.concatenate(sh_l),
        y=np.concatenate(y_l),
        label=np.concatenate(lab_l).astype(str),
        group=np.concatenate(grp_l).astype(str),
        cond=np.concatenate(cond_l).astype(str),
        bearing_key=np.concatenate(bk_l).astype(str),
        t_index=np.concatenate(ti_l),
        geometries=geoms, name=f"mfpt_{int(fs_hz_filter or 0)}",
        provenance={
            "dataset": "MFPT bearing fault dataset, via the MathWorks "
                       "redistribution (CC BY-NC-SA 4.0, data owner Eric "
                       "Bechhoefer)",
            "source_base_url": manifest.get("source_base_url")
                               or manifest.get("source_urls"),
            "download_utc": manifest.get("download_utc"),
            "records_used": used,
            "window_samples": window_samples,
            "hop_samples": hop,
            "licence_note": "CC BY-NC-SA 4.0; not redistributed in this "
                            "repository.",
        },
    )


# ---------------------------------------------------------------------------
# Simulator, for the sim-to-real comparison only
# ---------------------------------------------------------------------------
def load_sim_fleet(
    *,
    window_samples: int = 2048,
    machines: Optional[Sequence[str]] = None,
    n_healthy_per_machine: int = 24,
    n_per_fault: int = 8,
    seed: int = 0,
) -> WindowSet:
    """A grouped simulator fleet, cut to the same window length as CWRU.

    Here so that the *same* code path scores simulated and measured data, which
    is the only way the sim-to-real gap is a measurement rather than a
    comparison of two different harnesses.
    """
    from pdmcore.sim import MACHINE_LIBRARY, fleet_dataset

    results = fleet_dataset(
        machines=machines, n_healthy_per_machine=n_healthy_per_machine,
        n_per_fault=n_per_fault, duration_s=0.5, seed=seed,
    )
    xs, fs_l, sh_l, y_l, lab_l, grp_l, cond_l, bk_l, ti_l = ([] for _ in range(9))
    geoms: Dict[str, BearingGeometry] = {}
    per_group: Dict[str, int] = {}
    for r in results:
        sig = np.asarray(r.accel_g, dtype=np.float64)
        if sig.size < window_samples:
            continue
        win = sig[:window_samples][None, :]
        unit = str(r.truth["unit_id"])
        bkey = r.spec.bearing.name
        geoms.setdefault(bkey, r.spec.bearing)
        idx = per_group.get(unit, 0)
        per_group[unit] = idx + 1
        xs.append(win)
        fs_l.append(np.array([float(r.fs_hz)]))
        sh_l.append(np.array([float(r.truth["shaft_hz"])]))
        y_l.append(np.array([0 if r.label == "healthy" else 1]))
        lab_l.append(np.array([r.label], dtype=object))
        grp_l.append(np.array([unit], dtype=object))
        cond_l.append(np.array([str(r.truth.get("machine_class", "sim"))], dtype=object))
        bk_l.append(np.array([bkey], dtype=object))
        ti_l.append(np.array([idx]))
    del MACHINE_LIBRARY
    return WindowSet(
        x=np.vstack(xs), fs_hz=np.concatenate(fs_l), shaft_hz=np.concatenate(sh_l),
        y=np.concatenate(y_l), label=np.concatenate(lab_l).astype(str),
        group=np.concatenate(grp_l).astype(str), cond=np.concatenate(cond_l).astype(str),
        bearing_key=np.concatenate(bk_l).astype(str), t_index=np.concatenate(ti_l),
        geometries=geoms, name="sim_fleet",
        provenance={"dataset": "pdmcore.sim.fleet_dataset", "seed": seed,
                    "window_samples": window_samples,
                    "warning": "SIMULATED. The simulator and the 63-feature "
                               "extractor were written by the same process, so "
                               "agreement here is partly self-fulfilling."},
    )


def load_sim_degradation(
    *,
    window_samples: int = 2048,
    mode: str = "bearing_outer",
    n_points: int = 120,
    days: float = 90.0,
    seed: int = 0,
) -> Tuple[WindowSet, np.ndarray, np.ndarray]:
    """Run-to-failure series. Returns ``(windows, t_days, severity)``.

    Detection delay needs a time-ordered degradation, and no public bearing
    dataset in this branch has one: CWRU and MFPT are both *seeded-fault
    snapshots*, not run-to-failure. So the delay number here is a simulator
    number and is labelled as such everywhere it appears.
    """
    from pdmcore.sim import degradation_series

    results = degradation_series(mode=mode, n_points=n_points, duration_s=0.5,
                                 days=days, seed=seed)
    xs, fs_l, sh_l, y_l, lab_l, grp_l, cond_l, bk_l, ti_l = ([] for _ in range(9))
    geoms: Dict[str, BearingGeometry] = {}
    t_days, sev = [], []
    for i, r in enumerate(results):
        sig = np.asarray(r.accel_g, dtype=np.float64)[:window_samples]
        bkey = r.spec.bearing.name
        geoms.setdefault(bkey, r.spec.bearing)
        xs.append(sig[None, :])
        fs_l.append(np.array([float(r.fs_hz)]))
        sh_l.append(np.array([float(r.truth["shaft_hz"])]))
        y_l.append(np.array([1 if r.truth["severity"] > 0.05 else 0]))
        lab_l.append(np.array([r.label], dtype=object))
        grp_l.append(np.array(["degradation_unit"], dtype=object))
        cond_l.append(np.array([mode], dtype=object))
        bk_l.append(np.array([bkey], dtype=object))
        ti_l.append(np.array([i]))
        t_days.append(float(r.truth["t_days"]))
        sev.append(float(r.truth["severity"]))
    ws = WindowSet(
        x=np.vstack(xs), fs_hz=np.concatenate(fs_l), shaft_hz=np.concatenate(sh_l),
        y=np.concatenate(y_l), label=np.concatenate(lab_l).astype(str),
        group=np.concatenate(grp_l).astype(str), cond=np.concatenate(cond_l).astype(str),
        bearing_key=np.concatenate(bk_l).astype(str), t_index=np.concatenate(ti_l),
        geometries=geoms, name=f"sim_degradation_{mode}",
        provenance={"dataset": "pdmcore.sim.degradation_series", "mode": mode,
                    "days": days, "seed": seed, "window_samples": window_samples,
                    "warning": "SIMULATED run-to-failure. No measured "
                               "run-to-failure data is available on this branch."},
    )
    return ws, np.asarray(t_days), np.asarray(sev)
