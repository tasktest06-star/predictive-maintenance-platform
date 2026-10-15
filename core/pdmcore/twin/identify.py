r"""Parameter identification in three tiers, and the width of what comes out.

The question this module answers is not "what are the parameters" but **"how
wide is the posterior, and which predictions survive that width"**. A twin
reported as a point estimate is a twin whose validity domain cannot be derived,
and a validity domain that cannot be derived is one that will not be enforced.

The three information tiers
---------------------------
``nameplate``
    What is stamped on the plate. Free on every motor in every plant. Four
    sub-levels, because the fields differ by manufacturer and each one can be
    priced separately:

    ======  =================================================================
    1a      power, voltage, current, speed, frequency, poles, IE class
    1b      1a + rated power factor
    1c      1b + a printed efficiency value (not just the class floor)
    1d      1c + a locked-rotor code letter
    ======  =================================================================

``commissioning``
    Tier 1 plus **one healthy vibration record and one temperature**, which is
    exactly what this platform collects: 63 features
    (:mod:`pdmcore.features`), a ``shaft_hz`` and a ``temperature_c`` in each
    300-byte frame (:mod:`pdmcore.io.payload`).

``sweep``
    Tier 2 plus a run-up or coast-down, or -- on a VFD asset -- 1x amplitude
    logged across the normal speed range. See
    :func:`coast_down_availability` for whether a plant would ever give you one
    *on this platform*, which is a different question from whether a plant could.

Method
------
Maximum a posteriori with **profile likelihood** intervals, not a Fisher
linearisation. The reason is structural: several of these parameters are exactly
unidentifiable, their Fisher eigenvalues are zero to machine precision, and a
linearised standard error on a zero eigenvalue is a finite number computed from
numerical noise. A profile likelihood that stays flat across the whole physical
range reports what is true -- unbounded -- and :class:`ParameterInterval`
carries ``identifiable=False`` for it.

Observations enter as Gaussian terms; nameplate *bounds* (the IE class floor,
the speed-rounding interval, the per-unit plausibility bands) enter as smooth
box priors that are flat inside and steeply quadratic outside. Flat inside
matters: a Gaussian stand-in for a uniform prior would put curvature into the
likelihood and would make an unidentifiable parameter look identified, which is
the same defect as a check that cannot fail, one level up.

The reparameterisation that makes this cheap
--------------------------------------------
The obvious formulation solves ``P_shaft(theta, s) = P_rated`` for ``s`` inside
the likelihood, which costs a bracketed root find (~200 circuit solves) per
evaluation. Instead the rated slip is carried **as a parameter** with the
plate's rounding interval as its box prior, and the power match becomes one more
Gaussian observation -- the plate's power is a rounded number too, so this is
also the more honest model. One circuit solve per evaluation, ~15 us, which is
what makes 50 000 posterior samples affordable on two cores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp as _exp
from math import isfinite as _isfinite
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .catalogue import PU_BANDS
from .electrical import locked_rotor_current, steady_state
from .params import (
    ElectricalParams,
    Nameplate,
    ParameterInterval,
    ie_efficiency_floor,
    thermal_prior_bounds,
)

__all__ = [
    "Observation",
    "BoxPrior",
    "IdentificationProblem",
    "IdentificationResult",
    "PARAM_NAMES",
    "nameplate_problem",
    "commissioning_problem",
    "identify",
    "profile_likelihood",
    "fisher_analysis",
    "posterior_sample",
    "load_from_speed",
    "coast_down_availability",
    "coast_down_ridge",
]

#: Free parameters, in the order the vector uses. All are identified in **log**
#: space: they are positive, they span orders of magnitude across machine sizes,
#: and log space makes every reported width a relative width.
PARAM_NAMES: Tuple[str, ...] = (
    "rs_ohm",
    "rr_ohm",
    "xls_ohm",
    "xlr_ohm",
    "xm_ohm",
    "r_core_ohm",
    "friction_loss_w",
    "rated_slip",
)

#: Extra parameters at the commissioning tier.
COMMISSIONING_PARAM_NAMES: Tuple[str, ...] = ("operating_slip", "r_th_k_per_w")


@dataclass(frozen=True)
class Observation:
    """One Gaussian likelihood term.

    ``sigma`` is an absolute standard deviation in the same unit as ``value``.
    Where the underlying uncertainty is a *rounding* interval of width ``w``, the
    standard deviation of a uniform distribution is ``w / sqrt(12)``, and that
    conversion is applied at the construction site so that the factor is visible
    there rather than buried here.
    """

    name: str
    value: float
    sigma: float
    unit: str = ""

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"observation {self.name!r} needs a positive sigma")


@dataclass(frozen=True)
class BoxPrior:
    """A flat-inside, steep-outside prior on a derived quantity.

    ``softness`` is the fraction of the interval width over which the penalty
    rises; 0.02 makes the wall steep enough to act as a bound and smooth enough
    for a gradient-free optimiser not to stall on it.

    Flat inside is the whole point. See the module docstring.
    """

    name: str
    lo: float
    hi: float
    softness: float = 0.02

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValueError(f"box prior {self.name!r} needs lo < hi")

    def penalty(self, x: float) -> float:
        w = (self.hi - self.lo) * self.softness
        if x < self.lo:
            return float(((self.lo - x) / w) ** 2)
        if x > self.hi:
            return float(((x - self.hi) / w) ** 2)
        return 0.0


@dataclass
class IdentificationProblem:
    """Everything needed to evaluate a posterior: plate, data, priors, tier.

    ``predict`` maps a parameter dict to the observable quantities named by the
    observations. It is built by :func:`nameplate_problem` and
    :func:`commissioning_problem` rather than passed in, so that the mapping
    between a parameter vector and the circuit is defined once.
    """

    plate: Nameplate
    tier: str
    sub_tier: str
    param_names: Tuple[str, ...]
    observations: Tuple[Observation, ...]
    box_priors: Tuple[BoxPrior, ...]
    start: Dict[str, float]
    derived: Callable[[Dict[str, float]], Dict[str, float]]

    # -- parameter vector <-> dict ----------------------------------------
    def to_vector(self, p: Dict[str, float]) -> np.ndarray:
        return np.array([np.log(p[k]) for k in self.param_names], dtype=np.float64)

    def to_params(self, theta: Sequence[float]) -> Dict[str, float]:
        # ``math.exp`` on Python floats, not ``np.exp`` on scalars: the numpy
        # scalar path costs ~15 us per call and this runs ~10^6 times during a
        # profile-likelihood sweep. Measured: 128 us -> 6 us for the whole dict.
        return {k: _exp(float(v)) for k, v in zip(self.param_names, theta)}

    # -- posterior ---------------------------------------------------------
    def neg_log_posterior(self, theta: Sequence[float]) -> float:
        """``-log p(theta | data)`` up to a constant. Lower is better."""
        try:
            p = self.to_params(theta)
            d = self.derived(p)
        except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
            # An unphysical parameter set (negative slip, a degenerate circuit,
            # a log-parameter the simplex has pushed to exp(800)) is rejected
            # rather than clipped: clipping creates a plateau that an optimiser
            # settles on and a sampler mixes across.
            return 1e12
        if not all(_isfinite(v) for v in d.values()):
            return 1e12
        chi2 = 0.0
        for obs in self.observations:
            if obs.name not in d:
                raise KeyError(
                    f"observation {obs.name!r} has no matching derived quantity; "
                    f"derived produces {sorted(d)}"
                )
            chi2 += ((d[obs.name] - obs.value) / obs.sigma) ** 2
        for bp in self.box_priors:
            if bp.name not in d and bp.name not in p:
                raise KeyError(f"box prior {bp.name!r} matches no parameter or derived value")
            chi2 += bp.penalty(d.get(bp.name, p.get(bp.name, 0.0)))
        return 0.5 * float(chi2)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def n_observations(self) -> int:
        return len(self.observations)


@dataclass
class IdentificationResult:
    """MAP point, intervals, Fisher structure and the provenance to serialise."""

    problem: IdentificationProblem
    params: Dict[str, float]
    neg_log_posterior: float
    intervals: Tuple[ParameterInterval, ...]
    fisher: Dict[str, object]
    success: bool
    message: str

    def electrical(self) -> ElectricalParams:
        p = self.params
        return ElectricalParams(
            rs_ohm=p["rs_ohm"],
            rr_ohm=p["rr_ohm"],
            xls_ohm=p["xls_ohm"],
            xlr_ohm=p["xlr_ohm"],
            xm_ohm=p["xm_ohm"],
            r_core_ohm=p["r_core_ohm"],
            friction_loss_w=p["friction_loss_w"],
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "tier": self.problem.tier,
            "sub_tier": self.problem.sub_tier,
            "plate": self.problem.plate.name,
            "n_params": self.problem.n_params,
            "n_observations": self.problem.n_observations,
            "params": {k: float(v) for k, v in self.params.items()},
            "neg_log_posterior": float(self.neg_log_posterior),
            "intervals": [i.as_dict() for i in self.intervals],
            "fisher": self.fisher,
            "success": bool(self.success),
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------
def _pu_box_priors(plate: Nameplate) -> List[BoxPrior]:
    """Per-unit plausibility priors, from the plate alone.

    ``Z_base = V_phase / I_rated`` uses only nameplate numbers, so these priors
    cost no extra information. They are genuine engineering knowledge -- the
    magnetising reactance of a machine with an air gap is 1.2-6 pu and nothing
    else -- and they are the reason an 8-parameter fit to 4 observations has a
    bounded answer at all. Where a reported interval is set by one of these
    rather than by the data, the result says so (``note="prior-dominated"``).
    """
    return [
        BoxPrior("rs_pu", *PU_BANDS["rs_pu"]),
        BoxPrior("rr_pu", *PU_BANDS["rr_pu"]),
        BoxPrior("x_leakage_pu", *PU_BANDS["x_leakage_pu"]),
        BoxPrior("xm_pu", *PU_BANDS["xm_pu"]),
        # Core loss and friction as fractions of rated output. Both are
        # well-bounded by machine-design practice and neither is visible on a
        # nameplate: core loss is 0.5-3 % of rated output, friction and windage
        # 0.2-2 %, and their sum is most of the gap between the copper-only and
        # the real efficiency.
        BoxPrior("core_loss_frac", 0.003, 0.030),
        BoxPrior("friction_frac", 0.002, 0.020),
        BoxPrior("rated_slip", *plate.slip_bounds()),
    ]


def _electrical_from(p: Dict[str, float]) -> ElectricalParams:
    return ElectricalParams(
        rs_ohm=p["rs_ohm"], rr_ohm=p["rr_ohm"], xls_ohm=p["xls_ohm"],
        xlr_ohm=p["xlr_ohm"], xm_ohm=p["xm_ohm"], r_core_ohm=p["r_core_ohm"],
        friction_loss_w=p["friction_loss_w"],
    )


def nameplate_problem(plate: Nameplate, *, sub_tier: str = "1a") -> IdentificationProblem:
    """Tier 1: identification from the nameplate alone.

    ``sub_tier`` is ``"1a"``, ``"1b"``, ``"1c"`` or ``"1d"`` -- see the module
    docstring. Each level adds observations and no parameters, so the
    identifiability can only improve, and by how much is the measurement.
    """
    if sub_tier not in ("1a", "1b", "1c", "1d"):
        raise ValueError("sub_tier must be one of 1a, 1b, 1c, 1d")
    z_base = plate.phase_voltage_v / plate.rated_current_a
    p_out = plate.rated_power_kw * 1e3

    def derived(p: Dict[str, float]) -> Dict[str, float]:
        el = _electrical_from(p)
        ss = steady_state(el, plate, p["rated_slip"])
        out = {
            "shaft_power_w": ss.shaft_power_w,
            "stator_current_a": ss.stator_current_a,
            "power_factor": ss.power_factor,
            "efficiency": ss.efficiency,
            "rs_pu": el.rs_ohm / z_base,
            "rr_pu": el.rr_ohm / z_base,
            "x_leakage_pu": el.x_leakage_sum_ohm / z_base,
            "xm_pu": el.xm_ohm / z_base,
            "core_loss_frac": ss.loss_breakdown_w["core"] / p_out,
            "friction_frac": ss.loss_breakdown_w["friction_windage"] / p_out,
            "rated_slip": p["rated_slip"],
        }
        if sub_tier == "1d":
            out["locked_rotor_pu"] = locked_rotor_current(el, plate) / plate.rated_current_a
        return out

    obs: List[Observation] = [
        # Rated power is a printed, rounded number: 37 kW, 7.5 kW. Treat its own
        # rounding (to the printed precision) as the uncertainty, uniform ->
        # sigma = width / sqrt(12).
        Observation("shaft_power_w", p_out, _printed_rounding(p_out) / np.sqrt(12.0), "W"),
        Observation(
            "stator_current_a",
            plate.rated_current_a,
            (plate.current_bounds()[1] - plate.current_bounds()[0]) / np.sqrt(12.0),
            "A",
        ),
    ]
    priors = _pu_box_priors(plate)

    if sub_tier in ("1b", "1c", "1d"):
        if plate.rated_power_factor is None:
            raise ValueError(
                f"sub_tier {sub_tier!r} uses the rated power factor but the nameplate "
                "does not carry one. Use sub_tier='1a', or supply the field -- do not "
                "substitute a typical value, which would be an assumption presented as "
                "a measurement"
            )
        # Printed to two decimals: a rounding interval of 0.01.
        obs.append(Observation("power_factor", plate.rated_power_factor, 0.01 / np.sqrt(12.0)))

    eta_lo, eta_hi = plate.efficiency_bounds()
    if sub_tier in ("1c", "1d") and plate.rated_efficiency is not None:
        # A printed efficiency value: IEC 60034-2-1 measurement tolerance plus
        # printing resolution, about +-0.5 of a point.
        obs.append(Observation("efficiency", float(plate.rated_efficiency), 0.005))
    else:
        # Only an IE class: a one-sided bound, implemented as a box. Using the
        # class floor as if it were a measurement would bias every loss-derived
        # parameter the same way.
        priors.append(BoxPrior("efficiency", eta_lo, eta_hi))

    if sub_tier == "1d":
        if plate.locked_rotor_current_pu is None:
            raise ValueError(
                "sub_tier '1d' uses the locked-rotor code letter, which this nameplate "
                "does not carry"
            )
        # A NEMA code letter is a band of locked-rotor kVA/hp, not a value; +-10 %
        # on the per-unit current is a fair representation of the band width.
        obs.append(
            Observation(
                "locked_rotor_pu",
                float(plate.locked_rotor_current_pu),
                0.10 * float(plate.locked_rotor_current_pu) / np.sqrt(3.0),
            )
        )

    s0 = float(np.clip(plate.rated_slip, *plate.slip_bounds()))
    start = {
        # Mid-band per-unit starting point, converted to ohms. Deliberately not
        # the published values: starting an identification at the answer is the
        # oldest way to make a badly conditioned problem look well conditioned.
        "rs_ohm": 0.03 * z_base,
        "rr_ohm": 0.03 * z_base,
        "xls_ohm": 0.075 * z_base,
        "xlr_ohm": 0.075 * z_base,
        "xm_ohm": 3.0 * z_base,
        "r_core_ohm": plate.phase_voltage_v**2 * 3.0 / (0.012 * p_out),
        "friction_loss_w": 0.008 * p_out,
        "rated_slip": s0,
    }
    return IdentificationProblem(
        plate=plate,
        tier="nameplate",
        sub_tier=sub_tier,
        param_names=PARAM_NAMES,
        observations=tuple(obs),
        box_priors=tuple(priors),
        start=start,
        derived=derived,
    )


def commissioning_problem(
    plate: Nameplate,
    *,
    measured_shaft_hz: float,
    shaft_hz_sigma: float,
    measured_temperature_c: float,
    ambient_c: float,
    temperature_sigma_k: float = 1.0,
    sub_tier: str = "1b",
) -> IdentificationProblem:
    """Tier 2: nameplate plus one healthy record and one temperature.

    The two added parameters are the **operating** slip (the machine is not at
    rated load, and nothing in the telemetry says what load it is at) and the
    thermal resistance. The two added observations are the measured shaft speed
    and the measured temperature.

    Two unknowns, two observations. The arithmetic already suggests the answer
    and the measurement confirms it: see
    ``twin/motor_physics/results/tier_comparison.json``. What tier 2 genuinely
    unlocks is the *thermal baseline* and the *path mode frequencies*, neither
    of which is an electrical parameter.

    ``shaft_hz_sigma`` must be the real one. The tacholess estimator on this
    platform's own 1 s / 25.6 kHz records has a measured bias of 0.19 Hz and a
    scatter of 0.15 Hz (``twin/motor_physics/results/speed_precision.json``);
    a tachometer gives ~0.01 Hz. Passing an optimistic value here is the single
    easiest way to make this tier look informative when it is not.
    """
    base = nameplate_problem(plate, sub_tier=sub_tier)
    names = tuple(base.param_names) + COMMISSIONING_PARAM_NAMES
    p_out = plate.rated_power_kw * 1e3
    eta_lo, _ = plate.efficiency_bounds()
    r_lo, r_hi = thermal_prior_bounds(plate, p_out * (1.0 / eta_lo - 1.0))

    def derived(p: Dict[str, float]) -> Dict[str, float]:
        d = base.derived(p)
        el = _electrical_from(p)
        ss_op = steady_state(el, plate, p["operating_slip"])
        d["operating_shaft_hz"] = ss_op.shaft_hz
        d["temperature_c"] = ambient_c + p["r_th_k_per_w"] * ss_op.loss_w
        d["operating_load_fraction"] = ss_op.shaft_torque_nm / plate.rated_torque_nm
        d["operating_loss_w"] = ss_op.loss_w
        d["r_th_k_per_w"] = p["r_th_k_per_w"]
        d["operating_slip"] = p["operating_slip"]
        return d

    obs = tuple(base.observations) + (
        Observation("operating_shaft_hz", float(measured_shaft_hz), float(shaft_hz_sigma), "Hz"),
        Observation("temperature_c", float(measured_temperature_c), float(temperature_sigma_k), "degC"),
    )
    priors = tuple(base.box_priors) + (
        BoxPrior("r_th_k_per_w", r_lo, r_hi),
        # The operating slip cannot exceed breakdown and cannot be negative; a
        # commissioning record is taken on a running, loaded machine, so 2x the
        # rated slip is a generous upper bound and 5 % of it a generous lower one.
        BoxPrior("operating_slip", 0.05 * plate.rated_slip, 2.0 * plate.rated_slip),
    )
    start = dict(base.start)
    start["operating_slip"] = float(plate.rated_slip)
    start["r_th_k_per_w"] = float(np.sqrt(r_lo * r_hi))
    return IdentificationProblem(
        plate=plate,
        tier="commissioning",
        sub_tier=sub_tier,
        param_names=names,
        observations=obs,
        box_priors=priors,
        start=start,
        derived=derived,
    )


def _printed_rounding(value: float, sig: int = 3) -> float:
    """Width of the rounding interval of ``value`` printed to ``sig`` figures."""
    if value <= 0:
        raise ValueError("value must be positive")
    exp = int(np.floor(np.log10(value))) - (sig - 1)
    return float(10.0**exp)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def identify(
    problem: IdentificationProblem,
    *,
    n_restarts: int = 6,
    seed: int = 0,
    profile: bool = True,
    n_profile: int = 21,
) -> IdentificationResult:
    """MAP fit with restarts, then profile-likelihood intervals.

    Nelder-Mead rather than a gradient method. The box priors have a
    discontinuous second derivative at the wall and the posterior has flat
    directions by construction; a quasi-Newton method spends its time
    rediscovering that the Hessian is singular. With 8-10 parameters and a 15 us
    objective, a few thousand simplex evaluations cost milliseconds.

    ``n_restarts`` perturbs the start point in log space by 30 %. Multiple
    restarts landing at different parameter values with the *same* posterior is
    not a failure to converge -- it is the flat direction showing up, and
    ``fisher["null_directions"]`` names it.
    """
    rng = np.random.default_rng(seed)
    theta0 = problem.to_vector(problem.start)
    best: Optional[Tuple[float, np.ndarray, str]] = None
    for k in range(max(1, n_restarts)):
        start = theta0 if k == 0 else theta0 + rng.normal(0.0, 0.30, theta0.size)
        res = minimize(
            problem.neg_log_posterior,
            start,
            method="Nelder-Mead",
            options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-8, "fatol": 1e-10},
        )
        if best is None or res.fun < best[0]:
            best = (float(res.fun), np.asarray(res.x, dtype=np.float64), str(res.message))
    assert best is not None
    f_map, theta_map, message = best
    params = problem.to_params(theta_map)

    fisher = fisher_analysis(problem, theta_map)
    intervals: Tuple[ParameterInterval, ...] = ()
    if profile:
        intervals = profile_likelihood(problem, theta_map, f_map, n=n_profile)

    return IdentificationResult(
        problem=problem,
        params=params,
        neg_log_posterior=f_map,
        intervals=intervals,
        fisher=fisher,
        success=f_map < 1e11,
        message=message,
    )


#: chi-squared 95 % quantile with one degree of freedom. The profile-likelihood
#: interval is where ``2*(nll - nll_min) <= 3.841``.
CHI2_95_1DF = 3.8414588206941236


def profile_likelihood(
    problem: IdentificationProblem,
    theta_map: np.ndarray,
    f_map: float,
    *,
    n: int = 21,
    span: float = 2.5,
    extra: Sequence[Tuple[str, Callable[[Dict[str, float]], float]]] = (),
) -> Tuple[ParameterInterval, ...]:
    """95 % profile-likelihood intervals for every parameter.

    For each parameter in turn, fix it on a log grid spanning ``exp(+-span)``
    around the MAP (a factor of 12 either way at the default), re-minimise over
    the others, and find where ``2 dNLL`` crosses :data:`CHI2_95_1DF`.

    A parameter whose profile never crosses the threshold inside the grid is
    reported with ``identifiable=False`` and an infinite bound. That is the
    result this branch exists to report, and it is why the grid is deliberately
    *wide*: a narrow grid turns "unbounded" into a finite-looking interval whose
    width is the width of the grid. The factor of 12 is far outside any physical
    range for these parameters, so a profile that is still flat at the edge is
    flat everywhere that matters.

    ``extra`` adds profiles for *derived* combinations -- the leakage sum and
    the leakage split are the important ones, because the individual leakages are
    unidentifiable while their sum is not, and reporting only the individuals
    would hide that.
    """
    out: List[ParameterInterval] = []
    for j, name in enumerate(problem.param_names):
        grid = theta_map[j] + np.linspace(-span, span, n)
        nll = np.empty(n)
        free = [i for i in range(theta_map.size) if i != j]
        for k, val in enumerate(grid):

            def obj(sub: np.ndarray, _val: float = float(val)) -> float:
                th = np.empty(theta_map.size)
                th[j] = _val
                th[free] = sub
                return problem.neg_log_posterior(th)

            r = minimize(
                obj, theta_map[free], method="Nelder-Mead",
                options={"maxiter": 4000, "maxfev": 4000, "xatol": 1e-7, "fatol": 1e-9},
            )
            nll[k] = float(r.fun)
        out.append(_interval_from_profile(name, np.exp(grid), nll, f_map, float(np.exp(theta_map[j]))))

    for name, fn in extra:
        # Derived combinations: profiled by constraining the combination with a
        # penalty rather than by substitution, which keeps the parameterisation
        # unchanged and so keeps the profiles comparable.
        v_map = fn(problem.to_params(theta_map))
        grid = np.exp(np.log(v_map) + np.linspace(-span, span, n)) if v_map > 0 else None
        if grid is None:
            continue
        nll = np.empty(n)
        for k, val in enumerate(grid):

            def obj(th: np.ndarray, _val: float = float(val)) -> float:
                p = problem.to_params(th)
                try:
                    pen = ((fn(p) - _val) / (1e-4 * _val)) ** 2
                except (ValueError, ZeroDivisionError):
                    return 1e12
                return problem.neg_log_posterior(th) + 0.5 * pen

            r = minimize(
                obj, theta_map, method="Nelder-Mead",
                options={"maxiter": 8000, "maxfev": 8000, "xatol": 1e-7, "fatol": 1e-9},
            )
            nll[k] = float(r.fun)
        out.append(_interval_from_profile(name, grid, nll, f_map, float(v_map)))
    return tuple(out)


def _interval_from_profile(
    name: str, grid: np.ndarray, nll: np.ndarray, f_map: float, value: float
) -> ParameterInterval:
    d = 2.0 * (nll - f_map)
    # Numerical floor: the re-minimised profile can dip a hair below the global
    # MAP when the global fit stopped early. Clamp at zero rather than letting a
    # -1e-9 widen the interval.
    d = np.maximum(d, 0.0)
    i_map = int(np.argmin(np.abs(grid - value)))
    lo, hi = grid[0], grid[-1]
    below = np.where((d >= CHI2_95_1DF) & (np.arange(d.size) < i_map))[0]
    above = np.where((d >= CHI2_95_1DF) & (np.arange(d.size) > i_map))[0]
    if below.size:
        k = int(below[-1])
        lo = float(np.interp(CHI2_95_1DF, [d[k], d[k + 1]][::-1], [grid[k], grid[k + 1]][::-1]))
    if above.size:
        k = int(above[0])
        hi = float(np.interp(CHI2_95_1DF, [d[k - 1], d[k]], [grid[k - 1], grid[k]]))
    identifiable = bool(below.size and above.size)
    note = "" if identifiable else "profile flat across a factor of 12: unbounded"
    return ParameterInterval(
        name=name,
        value=float(value),
        lo=float(lo) if below.size else 0.0,
        hi=float(hi) if above.size else float("inf"),
        identifiable=identifiable,
        note=note,
    )


def fisher_analysis(
    problem: IdentificationProblem, theta: np.ndarray, *, h: float = 1e-5
) -> Dict[str, object]:
    """Fisher information of the **Gaussian terms only**, in log-parameter space.

    The box priors are excluded deliberately: they are flat inside, so they
    contribute exactly zero curvature where the fit sits, and including them
    would report information that comes from the analyst's prior rather than
    from the plate. The resulting matrix is therefore the information **the data
    actually carries**, and its null space is the set of parameter combinations
    the data cannot see at all.

    ``null_directions`` lists, for every eigenvalue below ``1e-8`` times the
    largest, the parameters that dominate that eigenvector. Those are the
    combinations that are determined by the prior and nothing else.
    """
    names = problem.param_names
    obs = problem.observations

    def resid(th: np.ndarray) -> np.ndarray:
        p = problem.to_params(th)
        d = problem.derived(p)
        return np.array([(d[o.name] - o.value) / o.sigma for o in obs])

    m = len(obs)
    jac = np.empty((m, theta.size))
    for j in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        jac[:, j] = (resid(tp) - resid(tm)) / (2.0 * h)
    fim = jac.T @ jac
    evals, evecs = np.linalg.eigh(fim)
    tol = 1e-8 * max(evals.max(), 1e-300)
    nulls = []
    for i, lam in enumerate(evals):
        if lam <= tol:
            w = np.abs(evecs[:, i])
            dominant = [names[k] for k in np.argsort(-w)[:3] if w[k] > 0.25]
            nulls.append({"eigenvalue": float(lam), "parameters": dominant,
                          "weights": [float(x) for x in evecs[:, i]]})
    return {
        "param_names": list(names),
        "n_observations": m,
        "n_params": int(theta.size),
        "eigenvalues": [float(x) for x in evals],
        "cond": float(evals.max() / max(evals.min(), 1e-300)),
        "rank": int(np.sum(evals > tol)),
        "rank_deficiency": int(theta.size - np.sum(evals > tol)),
        "null_directions": nulls,
    }


def posterior_sample(
    problem: IdentificationProblem,
    theta_map: np.ndarray,
    *,
    n_samples: int = 20000,
    burn: int = 2000,
    step: float = 0.08,
    seed: int = 0,
    thin: int = 5,
) -> Dict[str, object]:
    """Random-walk Metropolis on the log posterior.

    A deliberately plain sampler. The posterior has exactly flat directions,
    which is a problem for anything that estimates a covariance from the
    gradient, and it has hard box walls, which is a problem for HMC. A
    random-walk with an isotropic step in log space mixes slowly along the flat
    directions -- which is correct behaviour, and the reported acceptance rate
    and per-parameter effective ranges make the slow mixing visible rather than
    hiding it inside a tuned proposal.

    Returned ``chain`` is in **log** space, shape ``(n_kept, n_params)``. Use
    ``np.exp`` for the parameters. ``accept_rate`` below ~0.1 or above ~0.6
    means ``step`` wants adjusting, and the value is reported for exactly that
    reason.
    """
    rng = np.random.default_rng(seed)
    th = np.asarray(theta_map, dtype=np.float64).copy()
    f = problem.neg_log_posterior(th)
    keep: List[np.ndarray] = []
    n_acc = 0
    total = burn + n_samples
    for i in range(total):
        prop = th + rng.normal(0.0, step, th.size)
        fp = problem.neg_log_posterior(prop)
        if fp < f or rng.random() < np.exp(f - fp):
            th, f = prop, fp
            n_acc += 1
        if i >= burn and (i - burn) % thin == 0:
            keep.append(th.copy())
    chain = np.asarray(keep)
    return {
        "chain": chain,
        "param_names": list(problem.param_names),
        "accept_rate": float(n_acc / total),
        "n_kept": int(chain.shape[0]),
        "step": float(step),
    }


# ---------------------------------------------------------------------------
# The conditioning of load inference, and the coast-down question
# ---------------------------------------------------------------------------
def load_from_speed(
    plate: Nameplate, shaft_hz: float, shaft_hz_sigma: float
) -> Dict[str, float]:
    r"""Load fraction inferred from shaft speed, and how badly it is conditioned.

    On the linear part of the torque-slip curve, load fraction
    :math:`L \approx s/s_{\text{rated}}`, so

    .. math::

        \frac{\partial L}{\partial f_{\text{shaft}}}
            = \frac{-1}{f_{\text{sync}}\,s_{\text{rated}}}

    which for a 4-pole 60 Hz motor with 1.1 % rated slip is **3.0 per hertz**.
    A 0.1 Hz speed error is a 30 % load error; a 0.33 Hz error spans the entire
    load range.

    This is the single most consequential conditioning number in the whole
    identification, because the wire format has no load channel
    (:mod:`pdmcore.io.payload`) and ``OperatingBaseline`` on the health side
    takes ``load_frac`` as an argument that something has to supply. Supplying
    it from a tacholess speed estimate does not work. Supplying it from a
    tachometer does.
    """
    if shaft_hz_sigma <= 0:
        raise ValueError("shaft_hz_sigma must be positive")
    s_rated = plate.rated_slip
    f_sync = plate.sync_hz
    s = (f_sync - shaft_hz) / f_sync
    sensitivity = 1.0 / (f_sync * s_rated)
    return {
        "shaft_hz": float(shaft_hz),
        "shaft_hz_sigma": float(shaft_hz_sigma),
        "slip": float(s),
        "load_fraction": float(s / s_rated),
        "d_load_per_hz": float(sensitivity),
        "load_sigma": float(sensitivity * shaft_hz_sigma),
        "load_sigma_pct_of_rated": float(100.0 * sensitivity * shaft_hz_sigma),
        "usable": bool(sensitivity * shaft_hz_sigma < 0.10),
    }


def coast_down_availability(
    *,
    coast_down_s: float,
    acquisition_s: float = 1.0,
    interval_s: float = 3600.0,
    stops_per_year: float = 12.0,
    has_downlink: bool = False,
) -> Dict[str, float]:
    """Probability that this platform ever captures a coast-down.

    Tier 3 is the only tier at which the rotor model is identifiable, and a
    coast-down is the cheapest way a plant could provide it: it happens anyway,
    at every shutdown, and needs no special test. So the honest question is not
    "would a plant give you one" -- it would, free -- but **"can this platform
    record one"**, and that is an architecture question with a numeric answer.

    The node takes ``acquisition_s`` of data every ``interval_s``
    (``edge/pdmedge/config.py`` defaults: 1 s every 3600 s) on a fixed schedule.
    Nothing in ``pdmcore.io.payload`` or ``cloud/ingest/app.py`` is a downlink:
    the wire format is device-to-cloud only and there is no command channel, so
    the node cannot be told "a shutdown is starting, sample continuously".
    Catching a coast-down is therefore a coincidence, with probability
    ``(coast_down_s + acquisition_s) / interval_s`` per stop.

    Set ``has_downlink=True`` to price the fix.
    """
    if coast_down_s <= 0 or acquisition_s <= 0 or interval_s <= 0:
        raise ValueError("all durations must be positive")
    if has_downlink:
        p_per_stop = 1.0
    else:
        p_per_stop = min(1.0, (coast_down_s + acquisition_s) / interval_s)
    p_year = 1.0 - (1.0 - p_per_stop) ** max(0.0, stops_per_year)
    return {
        "coast_down_s": float(coast_down_s),
        "acquisition_s": float(acquisition_s),
        "interval_s": float(interval_s),
        "stops_per_year": float(stops_per_year),
        "has_downlink": bool(has_downlink),
        "p_capture_per_stop": float(p_per_stop),
        "p_capture_per_year": float(p_year),
        "expected_years_to_first_capture": (
            float(1.0 / p_year) if p_year > 0 else float("inf")
        ),
    }


def coast_down_ridge(
    *,
    inertia_kgm2: float,
    drag_coeff_nm_per_hz: float,
    shaft_hz0: float,
    t_s: Sequence[float],
    noise_hz: float = 0.02,
    factors: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> Dict[str, object]:
    r"""Demonstrate that a coast-down identifies only ``drag / inertia``.

    With a linear drag torque :math:`T = b\,f` the coast-down obeys
    :math:`2\pi J \dot f = -b f`, i.e. :math:`f(t) = f_0 e^{-bt/(2\pi J)}`. Only
    the ratio :math:`b/J` appears. Scaling both by the same factor leaves the
    trace **bit-identical**, so the likelihood is exactly flat along that ridge
    and the inertia is not identified by any amount of coast-down data.

    This matters because inertia is what a run-up prediction, a starting-torque
    prediction and any torsional work needs, and because "we did a coast-down so
    we know the inertia" is a common and wrong claim. Separating them needs an
    independent torque or power measurement during the deceleration -- which
    means a current clamp, which this platform does not have on the vibration
    node.

    Returns the traces, the pairwise maximum absolute difference (which must be
    zero), and the chi-squared difference against the noise level.
    """
    t = np.asarray(t_s, dtype=np.float64)
    if t.size < 2 or np.any(t < 0):
        raise ValueError("t_s must be a non-negative array of length >= 2")
    if noise_hz <= 0:
        raise ValueError("noise_hz must be positive")
    base_rate = drag_coeff_nm_per_hz / (2.0 * np.pi * inertia_kgm2)
    traces = {}
    for fac in factors:
        rate = (drag_coeff_nm_per_hz * fac) / (2.0 * np.pi * (inertia_kgm2 * fac))
        traces[float(fac)] = shaft_hz0 * np.exp(-rate * t)
    ref = traces[1.0]
    max_diff = max(float(np.max(np.abs(v - ref))) for v in traces.values())
    return {
        "t_s": t,
        "traces": traces,
        "decay_rate_per_s": float(base_rate),
        "max_abs_diff_hz": max_diff,
        "max_chi2_over_ridge": float((max_diff / noise_hz) ** 2 * t.size),
        "ridge_is_exact": bool(max_diff == 0.0),
        "factors": [float(f) for f in factors],
    }
