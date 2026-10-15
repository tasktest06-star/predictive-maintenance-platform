"""Datasets, and the mapping from the 63-feature contract to the twin's
observation vector.

Two decisions here are load-bearing for the whole branch.

**The twin observes features, not waveforms.** A node transmits 300 B of
features (``pdmcore.features``); the waveform stays on the device. So a twin
that runs in the cloud or on the gateway can only be coupled to the machine
through those 63 numbers. That is not a limitation of this implementation, it is
the platform's architecture, and it bounds what any twin here can do.

**Temperature is excluded from both arms.** The frame carries
``temperature_c``, and in :mod:`pdmcore.sim` the temperature model is

    T = ambient + 18 + 22 * min(1, total_severity) + N(0, 0.4)

so temperature alone separates healthy from faulty by tens of sigma. That is an
artefact of the simulator -- a real bearing's temperature rise is confounded by
ambient, duty, lubrication and the sensor's thermal path -- and any arm that
used it would win the head-to-head for a reason that would not survive contact
with a plant. ``run_identifiability.py`` measures the artefact so the exclusion
is a documented number rather than a preference.

Severity, load factor and mount/resonance gain are read from ``truth`` and used
only for *scoring*. No estimator in this package sees them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pdmcore.features import FeatureConfig, FeatureExtractor
from pdmcore.sim import MACHINE_LIBRARY, MachineSpec, degradation_series, simulate

__all__ = [
    "OBS_NAMES",
    "ObsSpec",
    "Dataset",
    "extractor_for",
    "observations",
    "healthy_pool",
    "degradation_runs",
    "cache_dir",
]

#: The twin's observation vector: the feature-contract channels a physics model
#: of a rotating machine can actually predict.
#:
#: Chosen before any result was computed, on one rule -- include a channel if
#: the forward model in :mod:`pdmtwin.physics` produces it from first principles
#: (a line amplitude, a band integral of the modelled spectrum, an integration
#: to velocity), exclude it otherwise. That excludes every envelope and
#: bearing-evidence feature, every time-domain shape statistic (crest, kurtosis,
#: impulse factor) and the spectral-shape block: a lumped-parameter model of
#: shaft lines plus a structural floor says nothing about the kurtosis of an
#: envelope spectrum.
#:
#: The exclusion is the twin's structural handicap, and it is large: 9 of 63
#: channels. ``run_headtohead.py`` therefore also scores the baseline detector
#: restricted to exactly these 9, so that "the twin lost" and "the twin was
#: given less information" are separable.
#: Declared with ``order.1x_g`` included even though the validity gate in
#: :func:`channel_validity` removes it on the default asset. That is
#: deliberate: the gate firing on the platform's own reference machine, with the
#: bin arithmetic in its message, is worth more than a quietly shortened list.
OBS_NAMES: Tuple[str, ...] = (
    "time.rms",
    "iso.velocity_rms_mm_s",
    "order.1x_g",
    "order.2x_g",
    "band.02_rms",
    "band.03_rms",
    "band.04_rms",
    "band.07_rms",
    "band.09_rms",
    "band.10_rms",
)


def channel_validity(
    cfg: FeatureConfig,
    shaft_hz_range: Tuple[float, float],
    names: Sequence[str] = OBS_NAMES,
) -> Dict[str, Dict[str, object]]:
    """How many FFT bins each candidate channel's band actually contains.

    This gate exists because of a defect it found on the platform's own default
    asset. ``order_bands`` makes each order channel a +-6 % band around
    ``k * shaft_hz``; at ``n_fft = 4096`` and ``fs = 25.6 kHz`` the bin spacing
    is 6.25 Hz, and the 1x band of a 1780 rpm machine is 3.56 Hz wide. It can
    therefore contain **no bin at all**, in which case
    ``Spectrum.rms_in_band`` returns exactly 0.0 and the channel is identically
    zero -- not noisy, not small: zero, on every frame, for that asset.

    Measured over ``pdmcore.sim.MACHINE_LIBRARY`` at ``n_fft = 4096``, bins in
    band at nominal speed:

        motor_pump_37kw        0.5x:0  1x:1  2x:2
        fan_7kw_belt           0.5x:0  1x:0  2x:0
        gearbox_132kw          0.5x:1  1x:1  2x:1
        slow_agitator_400rpm   0.5x:0  1x:1  2x:1
        vfd_compressor_75kw    0.5x:1  1x:1  2x:2

    with the 1x count dropping to 0 on ``motor_pump_37kw`` under the 3 % speed
    jitter a fleet has. The consequence is not confined to this branch: see
    ``twin/research/02_findings.md`` and
    ``twin/tests/test_order_band_resolution.py``, where a severity-0.9
    looseness on ``motor_pump_37kw`` is diagnosed as *cavitation at 0.73
    confidence* because ``order.subharmonic_ratio`` is 0/0 and the looseness
    rule that needs it can never fire.

    The gate's job here is narrower: keep a dead channel out of the twin's
    observation vector, with the bin arithmetic in the refusal message, so that
    a physics core is never asked to predict a number that is structurally
    zero.
    """
    f_lo, f_hi = float(shaft_hz_range[0]), float(shaft_hz_range[1])
    if not 0 < f_lo <= f_hi:
        raise ValueError("shaft_hz_range must be positive and ordered")
    df = cfg.fs_hz / cfg.n_fft
    bins = np.arange(int(cfg.n_fft // 2) + 1) * df
    bands = cfg.bands()
    out: Dict[str, Dict[str, object]] = {}
    for name in names:
        lo_hi: Optional[Tuple[float, float]] = None
        if name.startswith("band."):
            lo_hi = bands[int(name.split(".")[1].split("_")[0])]
            counts = [int(np.sum((bins >= lo_hi[0]) & (bins <= lo_hi[1])))]
        elif name.startswith("order."):
            order = float(name.split(".")[1].rstrip("x_g").rstrip("x"))
            counts = []
            for f in np.linspace(f_lo, f_hi, 25):
                lo, hi = order * f * 0.94, order * f * 1.06
                counts.append(int(np.sum((bins >= lo) & (bins <= hi))))
            lo_hi = (order * f_lo * 0.94, order * f_hi * 1.06)
        else:
            out[name] = {"ok": True, "min_bins": None, "reason": "not a band statistic"}
            continue
        mn = int(min(counts))
        ok = mn >= 1
        reason = (
            "ok" if mn >= 2 else
            f"marginal: only {mn} FFT bin in band at some speed in "
            f"[{f_lo:.2f}, {f_hi:.2f}] Hz, df = {df:.3f} Hz" if mn == 1 else
            f"REFUSED: band [{lo_hi[0]:.2f}, {lo_hi[1]:.2f}] Hz contains no FFT bin at "
            f"df = {df:.3f} Hz, so the channel is identically zero"
        )
        out[name] = {"ok": ok, "min_bins": mn, "reason": reason,
                     "band_hz": [float(lo_hi[0]), float(lo_hi[1])], "df_hz": float(df)}
    return out


@dataclass(frozen=True)
class ObsSpec:
    """Which feature indices make up the observation vector."""

    names: Tuple[str, ...]
    index: Tuple[int, ...]
    dropped: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def from_feature_names(cls, feature_names: Sequence[str], names: Sequence[str] = OBS_NAMES) -> "ObsSpec":
        missing = [n for n in names if n not in feature_names]
        if missing:
            raise ValueError(f"observation channels absent from the feature contract: {missing}")
        return cls(tuple(names), tuple(list(feature_names).index(n) for n in names))

    @classmethod
    def gated(
        cls,
        feature_names: Sequence[str],
        cfg: FeatureConfig,
        shaft_hz_range: Tuple[float, float],
        names: Sequence[str] = OBS_NAMES,
        *,
        healthy: Optional[np.ndarray] = None,
    ) -> "ObsSpec":
        """Build an observation spec with the dead channels removed.

        Two independent gates, both of which can fail:

        * **Resolution** (:func:`channel_validity`): a band with no FFT bin in
          it. Deterministic, available at configuration time, before any data.
        * **Observed spread**: zero median-absolute-deviation across the healthy
          commissioning window. Empirical, and catches the cases the first gate
          cannot know about (a saturated channel, a stuck sensor).

        The first gate is the one that matters, because it fires before a model
        is fitted. The second is kept because a gate that only works in theory
        is not a gate.
        """
        validity = channel_validity(cfg, shaft_hz_range, names)
        keep: List[str] = []
        dropped: List[Tuple[str, str]] = []
        for n in names:
            v = validity[n]
            if not v["ok"]:
                dropped.append((n, str(v["reason"])))
                continue
            if healthy is not None:
                col = np.asarray(healthy, dtype=np.float64)[:, list(feature_names).index(n)]
                mad = float(np.median(np.abs(col - np.median(col))))
                if mad <= 0.0:
                    dropped.append((n, "REFUSED: zero MAD across the healthy commissioning window"))
                    continue
            keep.append(n)
        if len(keep) < 4:
            raise ValueError(
                f"only {len(keep)} usable observation channels after gating "
                f"({dropped}); a twin on fewer than 4 channels with 1-2 states is "
                "not worth running"
            )
        return cls(tuple(keep), tuple(list(feature_names).index(n) for n in keep), tuple(dropped))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.index)


def extractor_for(spec: MachineSpec) -> FeatureExtractor:
    """The extractor configuration every arm of the comparison uses.

    Identical to the one ``slides/figures/make_core_figures.py`` uses, including
    the fixed demodulation band: a kurtogram band search chosen per acquisition
    moves the ``bearing.demod_*`` channels around, and the baseline detector
    would then spend part of its healthy covariance on the band search rather
    than on the machine.
    """
    return FeatureExtractor(FeatureConfig(
        fs_hz=spec.fs_hz,
        bearing=spec.bearing,
        demod_band_hz=(2500.0, 6000.0),
        velocity_band_hz=(2.0, 1000.0) if spec.shaft_hz < 10 else (10.0, 1000.0),
    ))


def observations(X: np.ndarray, obs: ObsSpec) -> np.ndarray:
    """``(n, 63) -> (n, len(obs))`` in the observation-channel order."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return X[:, list(obs.index)]


@dataclass
class Dataset:
    """Feature matrix plus the ground truth used only for scoring.

    ``t_days`` is wall-clock within a trajectory; ``traj`` groups rows that came
    from the same run-to-failure series so nothing can be split mid-trajectory.
    """

    X: np.ndarray                      # (n, 63)
    feature_names: Tuple[str, ...]
    severity: np.ndarray               # (n,) ground-truth fault severity
    t_days: np.ndarray                 # (n,)
    load_factor: np.ndarray            # (n,) latent nuisance -- scoring only
    resonance_gain: np.ndarray         # (n,) latent nuisance -- scoring only
    shaft_hz: np.ndarray               # (n,) transmitted in the frame
    temperature_c: np.ndarray          # (n,) transmitted, deliberately unused
    mode: List[str] = field(default_factory=list)
    traj: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    unit: List[str] = field(default_factory=list)
    machine: str = "motor_pump_37kw"

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def take(self, idx) -> "Dataset":
        idx = np.asarray(idx)
        return Dataset(
            X=self.X[idx],
            feature_names=self.feature_names,
            severity=self.severity[idx],
            t_days=self.t_days[idx],
            load_factor=self.load_factor[idx],
            resonance_gain=self.resonance_gain[idx],
            shaft_hz=self.shaft_hz[idx],
            temperature_c=self.temperature_c[idx],
            mode=[self.mode[i] for i in np.atleast_1d(idx)] if self.mode else [],
            traj=self.traj[idx] if self.traj.size else self.traj,
            unit=[self.unit[i] for i in np.atleast_1d(idx)] if self.unit else [],
            machine=self.machine,
        )

    def obs(self, obs_spec: ObsSpec) -> np.ndarray:
        return observations(self.X, obs_spec)

    # -- caching -----------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            X=self.X,
            feature_names=np.array(self.feature_names),
            severity=self.severity,
            t_days=self.t_days,
            load_factor=self.load_factor,
            resonance_gain=self.resonance_gain,
            shaft_hz=self.shaft_hz,
            temperature_c=self.temperature_c,
            mode=np.array(self.mode if self.mode else [""] * self.n),
            traj=self.traj if self.traj.size else np.zeros(self.n, dtype=int),
            unit=np.array(self.unit if self.unit else [""] * self.n),
            machine=np.array(self.machine),
        )

    @classmethod
    def load(cls, path: str) -> "Dataset":
        z = np.load(path, allow_pickle=False)
        return cls(
            X=z["X"],
            feature_names=tuple(str(s) for s in z["feature_names"]),
            severity=z["severity"],
            t_days=z["t_days"],
            load_factor=z["load_factor"],
            resonance_gain=z["resonance_gain"],
            shaft_hz=z["shaft_hz"],
            temperature_c=z["temperature_c"],
            mode=[str(s) for s in z["mode"]],
            traj=z["traj"],
            unit=[str(s) for s in z["unit"]],
            machine=str(z["machine"]),
        )


def cache_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.abspath(os.path.join(here, "..", "results", "cache"))
    os.makedirs(d, exist_ok=True)
    return d


def _cached(tag: str, params: Dict, build) -> Dataset:
    """Cache on a hash of the build parameters.

    Datasets here cost 44 ms per frame (simulate + extract, measured on this
    host), so a full experiment set is minutes of CPU. The cache key is the
    parameter dict, not a filename, so a changed parameter cannot silently
    re-use the previous dataset -- the failure mode that makes a sweep look
    flat.
    """
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir(), f"{tag}_{key}.npz")
    if os.path.exists(path):
        return Dataset.load(path)
    ds = build()
    ds.save(path)
    return ds


def _ar1(n: int, rho: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) path with the given lag-1 correlation and spread.

    Used to give the nuisance a *memory*, which the simulator does not. In
    ``pdmcore.sim`` ``load_factor`` and ``resonance_gain`` are drawn afresh in
    every call to :func:`simulate`, so the nuisance of a simulated asset is
    white noise and a recursive filter has nothing to borrow from neighbouring
    frames. A real pump holds its duty for hours; a real mount relaxes over
    weeks. ``run_autocorrelation.py`` sweeps ``rho`` to find where a filter
    starts to pay for itself, which is the only honest way to report a filter
    result obtained on white-noise nuisance.

    Applied by modulating ``base_1x_g`` and ``base_2x_g``, which are the
    amplitudes the simulator scales with load -- so the AR(1) path acts on
    exactly the content ``load_factor`` acts on (shaft lines and impact
    amplitude), and not on the broadband floor, which ``load_factor`` also does
    not touch.
    """
    if rho <= 0.0:
        return np.ones(n)
    a = np.zeros(n)
    innov = sd * np.sqrt(max(1e-9, 1.0 - rho**2))
    a[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        a[i] = rho * a[i - 1] + rng.normal(0.0, innov)
    return np.exp(a)


def healthy_pool(
    *,
    machine: str = "motor_pump_37kw",
    n_units: int = 6,
    n_per_unit: int = 300,
    duration_s: float = 0.5,
    interval_h: float = 6.0,
    rpm_jitter_pct: float = 3.0,
    nuisance_rho: float = 0.0,
    nuisance_sd: float = 0.12,
    seed: int = 1000,
    use_cache: bool = True,
) -> Dataset:
    """Healthy frames from several units of the same machine class.

    ``n_units`` units with jittered nameplate speed, exactly as
    :func:`pdmcore.sim.fleet_dataset` does it, so nothing downstream can cheat
    by memorising a shaft frequency. ``nuisance=True`` throughout: without it
    the healthy cloud is unrealistically tight and every threshold on this
    branch would be hypersensitive (see the ``nuisance`` docstring in
    ``pdmcore/sim/machine.py``).

    The frames of one unit are in time order at ``interval_h`` spacing, because
    the calibration discipline this branch needs -- fit, calibrate and verify on
    three *disjoint windows in time order* -- is only meaningful if the rows
    have an order.
    """
    params = dict(kind="healthy", machine=machine, n_units=n_units, n_per_unit=n_per_unit,
                  duration_s=duration_s, interval_h=interval_h, rpm_jitter_pct=rpm_jitter_pct,
                  nuisance_rho=nuisance_rho, nuisance_sd=nuisance_sd,
                  seed=seed, obs=list(OBS_NAMES))

    def build() -> Dataset:
        from dataclasses import replace as _replace
        base = MACHINE_LIBRARY[machine]
        ex = extractor_for(base)
        rng = np.random.default_rng(seed)
        rows: List[np.ndarray] = []
        sev, td, lf, rg, sh, tc = ([] for _ in range(6))
        units: List[str] = []
        traj: List[int] = []
        counter = 0
        for u in range(n_units):
            jitter = 1.0 + rpm_jitter_pct / 100.0 * rng.standard_normal()
            uspec = _replace(base, rpm=float(base.rpm * jitter), name=f"{machine}_u{u}")
            ar = _ar1(n_per_unit, nuisance_rho, nuisance_sd, rng)
            for k in range(n_per_unit):
                counter += 1
                fspec = _replace(uspec, base_1x_g=uspec.base_1x_g * ar[k],
                                 base_2x_g=uspec.base_2x_g * ar[k]) if nuisance_rho > 0 else uspec
                r = simulate(fspec, None, duration_s=duration_s, seed=seed + counter, nuisance=True)
                f = ex.extract(r.accel_g, r.truth["shaft_hz"], fs_hz=r.fs_hz)
                rows.append(f.vector)
                sev.append(0.0)
                td.append(k * interval_h / 24.0)
                lf.append(r.truth["load_factor"] * float(ar[k]))
                rg.append(r.truth["resonance_gain"])
                sh.append(r.truth["shaft_hz"])
                tc.append(r.temperature_c)
                units.append(f"{machine}_u{u}")
                traj.append(u)
        return Dataset(
            X=np.vstack(rows), feature_names=tuple(ex.names),
            severity=np.asarray(sev), t_days=np.asarray(td),
            load_factor=np.asarray(lf), resonance_gain=np.asarray(rg),
            shaft_hz=np.asarray(sh), temperature_c=np.asarray(tc),
            mode=["healthy"] * len(rows), traj=np.asarray(traj, dtype=int),
            unit=units, machine=machine,
        )

    return _cached("healthy", params, build) if use_cache else build()


def degradation_runs(
    *,
    machine: str = "motor_pump_37kw",
    modes: Sequence[str] = ("bearing_outer", "bearing_inner", "unbalance"),
    n_seeds: int = 6,
    n_points: int = 160,
    days: float = 40.0,
    duration_s: float = 0.5,
    rpm_jitter_pct: float = 3.0,
    nuisance_rho: float = 0.0,
    nuisance_sd: float = 0.12,
    seed: int = 7000,
    use_cache: bool = True,
) -> Dataset:
    """Run-to-failure trajectories, one row per acquisition, in time order.

    ``n_points`` over ``days`` gives a 6 h acquisition cadence at the defaults,
    which is the cadence a battery node on a 4-per-day schedule actually runs.
    The severity profile is :func:`pdmcore.sim.degradation_series`'s two-stage
    one: a long flat stage then an exponential knee, which is what makes
    detection delay a meaningful quantity at all.

    Three modes rather than one, chosen to span what a physics core can and
    cannot see: ``unbalance`` moves a line the twin models explicitly,
    ``bearing_outer`` moves high-frequency energy the twin models only as a
    lumped floor, and ``bearing_inner`` adds modulation the twin does not model
    at all.
    """
    params = dict(kind="degr", machine=machine, modes=list(modes), n_seeds=n_seeds,
                  n_points=n_points, days=days, duration_s=duration_s,
                  rpm_jitter_pct=rpm_jitter_pct, nuisance_rho=nuisance_rho,
                  nuisance_sd=nuisance_sd, seed=seed, obs=list(OBS_NAMES))

    def build() -> Dataset:
        from dataclasses import replace as _replace
        base = MACHINE_LIBRARY[machine]
        ex = extractor_for(base)
        rng = np.random.default_rng(seed)
        rows: List[np.ndarray] = []
        sev, td, lf, rg, sh, tc = ([] for _ in range(6))
        modes_out: List[str] = []
        units: List[str] = []
        traj: List[int] = []
        tid = 0
        for mode in modes:
            for s in range(n_seeds):
                jitter = 1.0 + rpm_jitter_pct / 100.0 * rng.standard_normal()
                uspec = _replace(base, rpm=float(base.rpm * jitter), name=f"{machine}_t{tid}")
                base_seed = seed + 101 * tid
                if nuisance_rho > 0:
                    # Harvest the severity profile from the real function at a
                    # throwaway record length, then re-simulate each point with
                    # the AR(1)-modulated spec and the same per-point seed
                    # degradation_series would have used. Replicating the
                    # two-stage severity formula here instead would be a second
                    # copy of it, and the second copy is always the one that
                    # goes stale.
                    probe = degradation_series(uspec, mode=mode, n_points=n_points,
                                               duration_s=0.012, days=days, seed=base_seed)
                    ar = _ar1(n_points, nuisance_rho, nuisance_sd, rng)
                    series = []
                    for i, pr in enumerate(probe):
                        sv = float(pr.truth["severity"])
                        fspec = _replace(uspec, base_1x_g=uspec.base_1x_g * ar[i],
                                         base_2x_g=uspec.base_2x_g * ar[i])
                        rr = simulate(fspec, {mode: sv} if sv > 0 else None,
                                      duration_s=duration_s, seed=base_seed + i)
                        rr.truth.update({k: pr.truth[k] for k in ("t_days", "severity", "rul_days")})
                        rr.truth["load_factor"] = rr.truth["load_factor"] * float(ar[i])
                        series.append(rr)
                else:
                    series = degradation_series(
                        uspec, mode=mode, n_points=n_points, duration_s=duration_s,
                        days=days, seed=base_seed,
                    )
                for r in series:
                    f = ex.extract(r.accel_g, r.truth["shaft_hz"], fs_hz=r.fs_hz)
                    rows.append(f.vector)
                    sev.append(r.truth["severity"])
                    td.append(r.truth["t_days"])
                    lf.append(r.truth["load_factor"])
                    rg.append(r.truth["resonance_gain"])
                    sh.append(r.truth["shaft_hz"])
                    tc.append(r.temperature_c)
                    modes_out.append(mode)
                    units.append(f"{machine}_t{tid}")
                    traj.append(tid)
                tid += 1
        return Dataset(
            X=np.vstack(rows), feature_names=tuple(ex.names),
            severity=np.asarray(sev), t_days=np.asarray(td),
            load_factor=np.asarray(lf), resonance_gain=np.asarray(rg),
            shaft_hz=np.asarray(sh), temperature_c=np.asarray(tc),
            mode=modes_out, traj=np.asarray(traj, dtype=int), unit=units, machine=machine,
        )

    return _cached("degr", params, build) if use_cache else build()


def time_ordered_windows(
    ds: Dataset, fractions: Sequence[float] = (0.45, 0.25, 0.30)
) -> List[np.ndarray]:
    """Split a healthy pool into disjoint windows **in time order per unit**.

    This exists because of a defect this repository has already found twice: a
    threshold calibrated on one window and verified on the *same* window gives
    the target false-alarm rate by construction, so the check can never fail
    (``docs/08_branch_catalogue.md``, the Jetson row). The three windows here
    are fit / calibrate / verify, in that order within each unit, so a drift
    between them shows up instead of being absorbed.

    Splitting per unit rather than globally also keeps every unit represented in
    every window; a global time split would put whole units in one window only
    and confound drift with unit-to-unit variation.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    out: List[List[int]] = [[] for _ in fractions]
    for u in np.unique(ds.traj):
        idx = np.flatnonzero(ds.traj == u)
        idx = idx[np.argsort(ds.t_days[idx], kind="mergesort")]
        n = idx.size
        start = 0
        for j, fr in enumerate(fractions):
            stop = start + int(round(fr * n)) if j < len(fractions) - 1 else n
            out[j].extend(idx[start:stop].tolist())
            start = stop
    return [np.asarray(sorted(o), dtype=int) for o in out]
