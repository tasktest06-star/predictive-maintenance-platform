"""The physics core: a forward model of the *observable features*, not of the
waveform.

Why the model has to include the feature estimator
--------------------------------------------------
A twin of a rotating machine normally predicts a signal. This platform does not
transmit signals -- it transmits 63 band RMS values, order amplitudes and
indicators computed on the device (``pdmcore.features``). So the twin's
measurement equation is

    continuous physics  ->  pdmcore's own feature estimator  ->  observation

and the second arrow is not a formality. ``order.1x_g`` is the RMS inside a
+-6 % band around the shaft line, which at ``n_fft = 4096`` and 25.6 kHz is one
or two FFT bins wide: how much of a 29.7 Hz line lands in it depends on window
leakage, and a model that ignores that mispredicts the channel by tens of
percent. Rather than reimplement the leakage algebra, this module **probes the
real extractor**: it synthesises each modelled component on its own at unit
level, runs ``FeatureExtractor`` on it, and reads off the gain into every
observation channel. Band RMS is quadrature-additive for incoherent components
(``Spectrum.rms_in_band`` divides out the ENBW precisely so that it is), so the
probe produces an exact power-superposition model

.. math:: y_c^2 \\;=\\; \\sum_j M_{cj}\\, L_j^2

with ``L_j`` the RMS level of component ``j`` in g. The probe is a measurement
*calibration*, done once per asset, and it is the reason this twin's predictions
are in the same units as the features without any fudge factor.

The components, and where each one comes from
---------------------------------------------
=================  ======================================================
``line_1x``        residual unbalance; force ~ m r omega^2, so level
                   scales with ``(f/f_nom)^2``
``line_2x``        coupling/alignment reaction, same speed scaling
``line_3x``        higher coupling harmonic
``line_elec``      2 x line frequency, always present on an induction motor;
                   fixed in frequency, not a shaft order
``floor``          structural broadband background, 1/f
``noise``          sensor electronic noise, **from the datasheet**
                   (``SensorSpec.noise_density_ug_rthz``), not identified
``impact``         rolling-element impacts ringing the structure's
                   resonances -- the damage channel
=================  ======================================================

Five coefficients (one per component except ``noise``, which is datasheet, and
``impact``, which is unidentifiable from healthy data) are fitted to the healthy
commissioning window. The sensor's mount roll-off and bandwidth are applied
analytically from :class:`pdmcore.sim.SensorSpec`, which is the datasheet; the
structural resonance map comes from ``MachineSpec.resonances``, which in a real
deployment is a bump test at commissioning.

What is circular here, stated plainly
-------------------------------------
The *structure* of this model -- which lines exist, a 1/f floor, impacts ringing
resonances -- was chosen knowing how ``pdmcore.sim`` generates its signals. On a
real asset that structure has to be discovered, and getting it wrong is the
normal case. So the physics core here is **flattered**: it is about as right as a
physics core ever gets. That makes the head-to-head generous to the twin, which
is the direction an adversarial test should err in.

What is *not* circular: this module never calls :func:`pdmcore.sim.simulate`,
never sees a severity, and its coefficients are fitted only to healthy frames.
The damage model is a deliberately generic ``level ~ d`` with a unit
coefficient, where the simulator's own impact law is
``0.12*sqrt(sev) + 4*sev**1.4``; the twin therefore has the *shape* wrong, which
is realistic and is reported in the virtual-sensing results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from pdmcore.dsp.bearing import fault_frequencies
from pdmcore.features import FeatureExtractor
from pdmcore.sim import MachineSpec

from .data import ObsSpec

__all__ = [
    "COMPONENTS",
    "MECHANICAL",
    "ObservationModel",
    "PhysicsCore",
    "identify",
]

#: Component order, fixed: the probe matrix columns and the level vector share it.
COMPONENTS: Tuple[str, ...] = (
    "line_1x", "line_2x", "line_3x", "line_elec", "floor", "noise", "impact",
)
#: Components that scale with the machine's overall mechanical level (the state
#: the filter tracks). ``noise`` does not -- it is the front end, not the
#: machine, and that asymmetry is what makes the observation model nonlinear in
#: the state and gives the UKF something to do.
MECHANICAL: Tuple[str, ...] = ("line_1x", "line_2x", "line_3x", "line_elec", "floor", "impact")

#: Identified coefficients, in the same order everywhere.
THETA_NAMES: Tuple[str, ...] = ("k_1x", "k_2x", "k_3x", "k_elec", "k_floor")

_G_TO_MM_S = 9806.65  # 1 g in mm/s^2


# ---------------------------------------------------------------------------
# Probe signals -- elementary components at unit RMS
# ---------------------------------------------------------------------------
def _sensor_tf(f_hz: np.ndarray, spec: MachineSpec) -> np.ndarray:
    """Magnitude of the mount + bandwidth chain, from the datasheet.

    A single-pole mount roll-off and a two-pole bandwidth limit, which is what
    ``SensorSpec`` documents. This is the one place the twin uses the sensor's
    datasheet rather than identifying something, and it is the right division:
    the front end is specified, the machine is not.
    """
    s = spec.sensor
    h = 1.0 / np.sqrt(1.0 + (f_hz / max(1e-9, s.mount_corner_hz)) ** 2)
    h *= 1.0 / np.sqrt(1.0 + (f_hz / max(1e-9, s.bandwidth_hz)) ** 4)
    return h


def _tone(n: int, fs: float, f_hz: float, spec: MachineSpec, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS sinusoid at ``f_hz`` through the sensor chain."""
    t = np.arange(n, dtype=np.float64) / fs
    gain = float(_sensor_tf(np.array([f_hz]), spec)[0])
    return np.sqrt(2.0) * gain * np.sin(2.0 * np.pi * f_hz * t + rng.uniform(0.0, 2 * np.pi))


def _pink(n: int, fs: float, spec: MachineSpec, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS 1/f noise through the sensor chain."""
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    shape = np.ones_like(f)
    shape[1:] = 1.0 / np.sqrt(f[1:])
    shape[0] = shape[1]
    x = np.fft.irfft(w * shape * _sensor_tf(f, spec), n=n)
    sd = float(np.std(x))
    return x / sd if sd > 0 else x


def _white(n: int, fs: float, spec: MachineSpec, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS white noise, *not* through the mount.

    Electronic noise is added after the mechanical path, so the mount roll-off
    does not apply to it; the anti-alias/bandwidth limit does.
    """
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    h = 1.0 / np.sqrt(1.0 + (f / max(1e-9, spec.sensor.bandwidth_hz)) ** 4)
    x = np.fft.irfft(w * h, n=n)
    sd = float(np.std(x))
    return x / sd if sd > 0 else x


def _impacts(n: int, fs: float, rate_hz: float, spec: MachineSpec, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS impact train ringing the structural resonances.

    The twin's damage channel. A periodic train with slip jitter convolved with
    each resonance's impulse response -- the standard lumped model of a spall
    passing through the load zone. The resonance frequencies and Q come from the
    structure (a bump test at commissioning), the rate from the bearing
    kinematics in :mod:`pdmcore.dsp.bearing`.
    """
    out = np.zeros(n, dtype=np.float64)
    if rate_hz <= 0:
        return out
    period = fs / rate_hz
    pos = rng.uniform(0.0, period)
    train = np.zeros(n)
    while pos < n - 1:
        i = int(np.floor(pos))
        frac = pos - i
        train[i] += 1.0 - frac
        train[i + 1] += frac
        pos += period * (1.0 + 0.015 * rng.standard_normal())
    for res in spec.resonances:
        if res.freq_hz >= fs / 2.0:
            continue
        zeta = 1.0 / (2.0 * max(1.0, res.q))
        wn = 2.0 * np.pi * res.freq_hz
        tau = 1.0 / (zeta * wn)
        m = max(8, int(min(fs * tau * 6.9, fs * 0.25)))
        t = np.arange(m, dtype=np.float64) / fs
        ir = res.gain * np.exp(-zeta * wn * t) * np.sin(wn * np.sqrt(max(1e-9, 1 - zeta**2)) * t)
        out += np.convolve(train, ir, mode="same")
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    out = np.fft.irfft(np.fft.rfft(out) * _sensor_tf(f, spec), n=n)
    sd = float(np.std(out))
    return out / sd if sd > 0 else out


# ---------------------------------------------------------------------------
# The probe-calibrated observation model
# ---------------------------------------------------------------------------
@dataclass
class ObservationModel:
    """Power gains from component RMS levels to observation channels.

    ``gain[g, c, j]`` is the squared gain of component ``j`` into channel ``c``
    at grid speed ``grid_hz[g]``. Speed enters because the narrow order bands
    move with the shaft, and 3 % of unit-to-unit speed jitter is enough to
    change how much of the 1x line falls inside ``order.1x_g``'s window.
    """

    grid_hz: np.ndarray                 # (n_grid,)
    gain: np.ndarray                    # (n_grid, n_channels, n_components)
    channels: Tuple[str, ...]
    components: Tuple[str, ...] = COMPONENTS
    probe_cv: float = 0.0               # measured probe repeatability

    def gains_at(self, shaft_hz: np.ndarray) -> np.ndarray:
        """Interpolate the gain matrix to per-frame shaft speed -> ``(n, C, J)``."""
        s = np.atleast_1d(np.asarray(shaft_hz, dtype=np.float64))
        lo, hi = float(self.grid_hz[0]), float(self.grid_hz[-1])
        s = np.clip(s, lo, hi)
        pos = np.interp(s, self.grid_hz, np.arange(self.grid_hz.size))
        i0 = np.clip(np.floor(pos).astype(int), 0, self.grid_hz.size - 2)
        w = (pos - i0)[:, None, None]
        return (1.0 - w) * self.gain[i0] + w * self.gain[i0 + 1]

    def predict(self, levels: np.ndarray, shaft_hz: np.ndarray) -> np.ndarray:
        """``levels`` ``(n, J)`` in g RMS -> observations ``(n, C)``."""
        levels = np.atleast_2d(np.asarray(levels, dtype=np.float64))
        G = self.gains_at(shaft_hz)
        power = np.einsum("ncj,nj->nc", G, levels**2)
        return np.sqrt(np.maximum(power, 0.0))


def _probe(
    spec: MachineSpec,
    extractor: FeatureExtractor,
    obs: ObsSpec,
    *,
    duration_s: float = 0.5,
    n_grid: int = 7,
    speed_span_pct: float = 5.0,
    n_realisations: int = 3,
    seed: int = 11,
) -> ObservationModel:
    """Calibrate the observation model by running the real extractor.

    One extractor call per (speed, component, realisation). The stochastic
    components are averaged over ``n_realisations`` because a single draw of a
    1/f background has a few percent of band-to-band variation, and that
    variation would otherwise be baked into the physics as a bias. The spread
    across realisations is kept as :attr:`ObservationModel.probe_cv` -- a
    floor on how well any physics core can predict these channels, worth
    knowing before blaming the model for the residual.
    """
    fs = float(spec.fs_hz)
    n = int(round(duration_s * fs))
    f_nom = spec.shaft_hz
    grid = f_nom * np.linspace(1.0 - speed_span_pct / 100.0, 1.0 + speed_span_pct / 100.0, n_grid)
    gains = np.zeros((n_grid, len(obs), len(COMPONENTS)), dtype=np.float64)
    cvs: List[float] = []
    rng = np.random.default_rng(seed)
    for gi, f0 in enumerate(grid):
        ff = fault_frequencies(spec.bearing, float(f0))
        for cj, comp in enumerate(COMPONENTS):
            reps = 1 if comp.startswith("line") else n_realisations
            acc = np.zeros((reps, len(obs)))
            for r in range(reps):
                if comp == "line_1x":
                    x = _tone(n, fs, float(f0), spec, rng)
                elif comp == "line_2x":
                    x = _tone(n, fs, 2.0 * float(f0), spec, rng)
                elif comp == "line_3x":
                    x = _tone(n, fs, 3.0 * float(f0), spec, rng)
                elif comp == "line_elec":
                    x = _tone(n, fs, 2.0 * spec.line_hz, spec, rng)
                elif comp == "floor":
                    x = _pink(n, fs, spec, rng)
                elif comp == "noise":
                    x = _white(n, fs, spec, rng)
                elif comp == "impact":
                    x = _impacts(n, fs, ff.bpfo_hz, spec, rng)
                else:  # pragma: no cover - COMPONENTS is closed
                    raise AssertionError(comp)
                v = extractor.extract(x, float(f0), fs_hz=fs).vector
                acc[r] = v[list(obs.index)]
            mean = np.mean(acc, axis=0)
            gains[gi, :, cj] = mean**2
            if reps > 1:
                with np.errstate(invalid="ignore", divide="ignore"):
                    cv = np.std(acc, axis=0) / np.maximum(mean, 1e-12)
                cvs.append(float(np.nanmedian(cv)))
    return ObservationModel(
        grid_hz=grid, gain=gains, channels=tuple(obs.names),
        probe_cv=float(np.mean(cvs)) if cvs else 0.0,
    )


# ---------------------------------------------------------------------------
# The physics core
# ---------------------------------------------------------------------------
#: The state vector, in order. ``n_state`` truncates it: a twin with
#: ``n_state=1`` tracks only the rotating-force level and holds the rest at the
#: commissioning value.
#:
#: The decomposition is *line level* vs *floor level* rather than the textbook
#: *load* vs *mount gain*, and that choice was forced by a measurement. In
#: ``pdmcore.sim`` the two nuisance factors multiply the same array
#: (``x *= load_factor * resonance_gain``) before the broadband background is
#: added, so (a) load and mount gain are exactly degenerate -- only their
#: product is identifiable, whatever the simulator's docstring says about the
#: mount gain acting on high-frequency content only -- and (b) the structural
#: floor and the sensor noise do not move with either of them. Healthy log-std
#: measured per channel: ~0.16 where a shaft line dominates, ~0.04 where the
#: floor does. A state that scaled everything would therefore be fighting the
#: data. ``run_identifiability.py`` reports both numbers.
STATE_NAMES: Tuple[str, ...] = ("u_line", "u_floor", "damage")


@dataclass
class PhysicsCore:
    """Physics prediction of the twin's observation vector.

    State
    -----
    ``u_line``  log level of the rotating-force content (1x, 2x, 3x and the
        electrical line), relative to the commissioning median. This is the
        channel load enters through.
    ``u_floor`` log level of the broadband structural floor.
    ``damage``  impact level, as a multiple of the floor coefficient. Present
        only in the *estimation* filter. **The detection filter must not carry
        it**: a state that can explain the fault will explain it, and then there
        is nothing left in the residual to detect. That is the state-space form
        of this branch's central trap, and ``run_capacity.py`` measures it on
        the same axis as the residual-capacity form.

    ``n_state`` selects how much of the state is free. Each extra free state
    buys nuisance rejection and costs fault signal, which is the trade this
    branch exists to quantify.
    """

    spec: MachineSpec
    obs: ObsSpec
    model: ObservationModel
    theta: np.ndarray = field(default_factory=lambda: np.ones(len(THETA_NAMES)))
    noise_rms_g: float = 0.0
    n_state: int = 2
    fit_report: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.n_state <= 3:
            raise ValueError("n_state must be 1 (line), 2 (line+floor) or 3 (+damage)")

    @property
    def state_names(self) -> Tuple[str, ...]:
        return STATE_NAMES[: self.n_state]

    def _expand(self, x: np.ndarray) -> np.ndarray:
        """Pad a truncated state to the full ``(n, 3)`` form."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != self.n_state:
            raise ValueError(f"expected {self.n_state} state components, got {x.shape[1]}")
        full = np.zeros((x.shape[0], 3), dtype=np.float64)
        full[:, : self.n_state] = x
        return full

    # -- levels ------------------------------------------------------------
    def levels(self, x: np.ndarray, shaft_hz: np.ndarray) -> np.ndarray:
        """Component RMS levels ``(n, J)`` from the state.

        Unbalance and coupling reactions scale with ``omega^2`` (a rotating
        force); the electrical line and the structural floor do not. On a DOL
        motor the speed moves by well under a percent, so the ``omega^2`` term
        is nearly inert here -- it is in the model because it is the physics,
        and it is what would matter on a VFD asset.

        The impact (damage) level is referenced to the floor coefficient and
        scaled by the rotating-force state, because contact force at a spall
        rises with load. Its coefficient is **1 by definition**: nothing in a
        healthy commissioning window identifies the scale of a defect that is
        not there, so ``damage`` is an ordinal quantity, not a calibrated one.
        """
        full = self._expand(x)
        f = np.atleast_1d(np.asarray(shaft_hz, dtype=np.float64))
        s_line = np.exp(full[:, 0])
        s_floor = np.exp(full[:, 1])
        d = np.maximum(full[:, 2], 0.0)
        r2 = (f / self.spec.shaft_hz) ** 2
        k1, k2, k3, ke, kb = self.theta
        out = np.zeros((full.shape[0], len(COMPONENTS)), dtype=np.float64)
        out[:, 0] = k1 * s_line * r2
        out[:, 1] = k2 * s_line * r2
        out[:, 2] = k3 * s_line * r2
        out[:, 3] = ke * s_line
        out[:, 4] = kb * s_floor
        out[:, 5] = self.noise_rms_g
        out[:, 6] = kb * s_line * d
        return out

    def predict(self, x: np.ndarray, shaft_hz: np.ndarray) -> np.ndarray:
        """Physics prediction of the observation vector, ``(n, C)`` in feature units."""
        return self.model.predict(self.levels(x, shaft_hz), shaft_hz)

    def predict_log(self, x: np.ndarray, shaft_hz: np.ndarray) -> np.ndarray:
        """Prediction in log space, which is where the residual lives.

        Every observation channel is a positive RMS whose healthy variation is
        multiplicative (load, mount gain, sensitivity error are all gains), so
        the natural error model is multiplicative and the natural residual is a
        log ratio. The same reasoning is why
        :class:`pdmcore.models.anomaly.StandardScaler` log-transforms its
        skewed positive features.
        """
        return np.log(np.maximum(self.predict(x, shaft_hz), 1e-12))

    def nominal_state(self, n: int = 1) -> np.ndarray:
        return np.zeros((n, self.n_state), dtype=np.float64)

    @property
    def n_components(self) -> int:
        return len(COMPONENTS)


def identify(
    spec: MachineSpec,
    extractor: FeatureExtractor,
    obs: ObsSpec,
    Y_healthy: np.ndarray,
    shaft_hz: np.ndarray,
    *,
    n_state: int = 2,
    model: Optional[ObservationModel] = None,
    probe_kwargs: Optional[Dict] = None,
) -> PhysicsCore:
    """Identify the five coefficients on a healthy commissioning window.

    The window's *median* frame defines ``u = 0``. That is not a convenience:
    the nuisance factors in this simulator are log-normal with zero log-mean, so
    the median healthy frame is the nominal-duty frame by construction, and a
    real commissioning window is defined the same way (the engineer records
    "normal duty" and that becomes the reference). Any residual mean offset is
    absorbed by the coefficients, which is what they are for.

    Fitted in log space on the channel medians, with the sensor noise level
    fixed from the datasheet. Five parameters, ten channels: over-determined, and
    the fit residual is reported rather than assumed small -- it is the first
    number that says whether a physics core of this structure is usable at all.
    """
    if model is None:
        model = _probe(spec, extractor, obs, **(probe_kwargs or {}))
    Y = np.atleast_2d(np.asarray(Y_healthy, dtype=np.float64))
    f_med = float(np.median(shaft_hz))
    y_med = np.median(Y, axis=0)
    log_target = np.log(np.maximum(y_med, 1e-12))
    noise_rms = float(spec.sensor.noise_rms_g(spec.fs_hz))

    core = PhysicsCore(spec=spec, obs=obs, model=model, noise_rms_g=noise_rms, n_state=n_state)

    # Scale-free starting point: put every mechanical component at the level
    # that would alone explain the median overall RMS.
    y_rms = float(y_med[list(obs.names).index("time.rms")]) if "time.rms" in obs.names else float(np.max(y_med))
    x0 = np.log(np.full(len(THETA_NAMES), max(1e-6, y_rms / np.sqrt(len(THETA_NAMES)))))

    def resid(log_theta: np.ndarray) -> np.ndarray:
        core.theta = np.exp(log_theta)
        pred = core.predict_log(core.nominal_state(1), np.array([f_med]))[0]
        return pred - log_target

    sol = least_squares(resid, x0, method="trf", xtol=1e-12, ftol=1e-12, max_nfev=4000)
    core.theta = np.exp(sol.x)
    r = resid(sol.x)
    # Rank of the parameter Jacobian. Five coefficients are not necessarily five
    # identifiable coefficients: the 3x line (89 Hz) and the electrical line
    # (120 Hz) both fall in band.04 and nowhere else on this channel set, so
    # only their quadrature sum is identifiable and the fit is free to send one
    # of them to zero. That costs nothing in prediction and everything in
    # interpretation, which is why the rank is reported rather than the
    # coefficient count.
    sv = np.linalg.svd(sol.jac, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-6)) if sv.size else 0
    core.fit_report = {
        "rms_log_residual": float(np.sqrt(np.mean(r**2))),
        "max_abs_log_residual": float(np.max(np.abs(r))),
        "worst_channel": obs.names[int(np.argmax(np.abs(r)))],
        "worst_channel_pct": float((np.exp(np.max(np.abs(r))) - 1.0) * 100.0),
        "probe_cv": float(model.probe_cv),
        "n_healthy_frames": int(Y.shape[0]),
        "n_channels": int(len(obs)),
        "theta_rank": rank,
        "theta_count": int(len(THETA_NAMES)),
        "jac_cond": float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf"),
        **{n: float(v) for n, v in zip(THETA_NAMES, core.theta)},
    }
    return core
