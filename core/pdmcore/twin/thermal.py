r"""First-order thermal model, and what one temperature scalar can buy.

Temperature is the one quantity the platform's own node already reports:
``pdmcore.io.payload.TelemetryFrame.temperature_c``, one float32 per frame. So a
thermal model is the cheapest extra physics the twin can have, and the question
is what it is worth.

The model
---------
.. math::  C\frac{dT}{dt} = P_{\text{loss}} - \frac{T - T_a}{R_{th}}

Steady state: :math:`\Delta T = R_{th} P_{\text{loss}}`. Time constant
:math:`\tau = R_{th} C`.

Why one node and not two
------------------------
The textbook model for a motor is two nodes -- winding and frame -- because the
winding heats in seconds to minutes and the frame in tens of minutes, and a
measured heat run shows both. A two-node model has four parameters. The platform
transmits **one temperature scalar per frame, with no load label, at an interval
of 3600 s by default** (``edge/pdmedge/config.py``, ``interval_s = 3600.0``).
Four parameters against a sequence of unlabelled hourly scalars is not an
identification problem, it is a decoration problem.
:func:`two_node_reduction_error` measures what the one-node reduction costs
against a two-node reference so the choice is priced rather than asserted.

What one temperature reading identifies
---------------------------------------
Exactly one scalar: the product :math:`R_{th} P_{\text{loss}}`. Since
:math:`P_{\text{loss}}` is what the electrical identification wants to know and
:math:`R_{th}` is unknown, a single reading identifies neither -- it identifies
their product, which is already known, because it *is* the measurement.

The reading becomes informative only through a **prior on**
:math:`R_{th}`, and the width of that prior sets the floor on everything
downstream. :func:`loss_from_temperature` propagates it:
``pdmcore.twin.params.thermal_prior_bounds`` gives :math:`R_{th}` to about a
factor of 1.8 from the insulation class, so a temperature reading bounds total
loss to about a factor of 1.8, i.e. efficiency at rated load to roughly
+-2 points on a 37 kW machine. Compare that with the +-1.5 points the IE class
label already gives you for free, and the honest conclusion is:

    **A single commissioning temperature adds almost nothing to the electrical
    identification. What it does add is a thermal *baseline*: a reference
    :math:`\Delta T` at a known operating point, against which a later rise is
    a residual. That is a trending quantity, not an identification quantity.**

What would change that
----------------------
A temperature **time series across a load change** identifies :math:`\tau`
directly from the exponential, and :math:`\tau` plus the steady rise separates
:math:`R_{th}` from :math:`C`. That needs sampling at a fraction of
:math:`\tau` (motors: :math:`\tau` is 20-60 min, so 2-5 min sampling) and a
load label. The platform's hourly cadence is 1-3 time constants per sample and
there is no load field in the wire format, so this is **not available on this
platform as built**. :func:`heat_run_identifiability` quantifies the sampling
interval at which it becomes available.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .params import Nameplate, ThermalParams, thermal_prior_bounds

__all__ = [
    "steady_temperature_c",
    "temperature_trajectory_c",
    "loss_from_temperature",
    "two_node_reduction_error",
    "heat_run_identifiability",
]


def steady_temperature_c(thermal: ThermalParams, loss_w: float, ambient_c: Optional[float] = None) -> float:
    """Steady-state temperature for a constant loss."""
    if loss_w < 0:
        raise ValueError("loss_w must be non-negative")
    amb = thermal.ambient_c if ambient_c is None else float(ambient_c)
    return float(amb + thermal.r_th_k_per_w * loss_w)


def temperature_trajectory_c(
    thermal: ThermalParams,
    t_s: Sequence[float],
    loss_w: Sequence[float],
    *,
    t0_c: Optional[float] = None,
    ambient_c: Optional[float] = None,
) -> np.ndarray:
    """Integrate the one-node model over a piecewise-linear loss history.

    Uses the analytic solution per interval rather than a numerical integrator:
    the equation is linear with a linear forcing term over each interval, so the
    exact solution is cheap and unconditionally stable, and the identification
    can call this thousands of times without worrying about step size.
    """
    t = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(loss_w, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError("t_s and loss_w must have the same shape")
    if t.size < 1:
        raise ValueError("need at least one sample")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_s must be strictly increasing")
    amb = thermal.ambient_c if ambient_c is None else float(ambient_c)
    tau = thermal.tau_s
    out = np.empty_like(t)
    theta = (amb if t0_c is None else float(t0_c)) - amb   # excess temperature
    out[0] = amb + theta
    for k in range(1, t.size):
        dt = t[k] - t[k - 1]
        # Excess-temperature target, averaged over the interval (trapezoid on the
        # forcing): exact for constant loss, second-order for a ramp.
        target = thermal.r_th_k_per_w * 0.5 * (p[k] + p[k - 1])
        decay = float(np.exp(-dt / tau))
        theta = target + (theta - target) * decay
        out[k] = amb + theta
    return out


def loss_from_temperature(
    plate: Nameplate,
    measured_c: float,
    ambient_c: float,
    *,
    nominal_loss_w: Optional[float] = None,
) -> Dict[str, float]:
    """Interval on total loss implied by one temperature reading.

    The inversion is ``P = dT / R_th``, so the loss interval is the measured rise
    divided by the :math:`R_{th}` prior interval -- in the opposite order,
    because a *larger* thermal resistance explains the same rise with a
    *smaller* loss.

    ``nominal_loss_w`` seeds the :math:`R_{th}` prior (which is expressed as
    "the rise at rated load is 45-80 % of the insulation class limit", so it
    needs a loss to turn into a resistance). Pass the nameplate-implied loss:
    ``P_rated * (1/eta - 1)``. If omitted it is computed from the IE-class
    efficiency floor, which widens the result and is noted in the output.

    Returns the loss interval, the implied efficiency interval at rated load,
    and ``information_bits``: the log2 reduction in the width of the efficiency
    interval relative to what the IE class alone gives. **That number is the
    answer to "is a commissioning temperature worth collecting for
    identification?"** and it is committed in
    ``twin/motor_physics/results/thermal_identifiability.json``.
    """
    rise = float(measured_c) - float(ambient_c)
    if rise <= 0:
        raise ValueError(
            f"measured temperature {measured_c} C is at or below ambient {ambient_c} C; "
            "a motor under load is always above ambient, so this is a sensor or "
            "ambient-reference fault, not an operating point"
        )
    eta_lo, eta_hi = plate.efficiency_bounds()
    p_out = plate.rated_power_kw * 1e3
    if nominal_loss_w is None:
        nominal_loss_w = p_out * (1.0 / eta_lo - 1.0)
        seeded = False
    else:
        seeded = True
    r_lo, r_hi = thermal_prior_bounds(plate, float(nominal_loss_w))
    # Larger R_th explains the same rise with less loss.
    loss_lo, loss_hi = rise / r_hi, rise / r_lo
    # Efficiency from loss: eta = P_out / (P_out + P_loss).
    eta_from_lo = p_out / (p_out + loss_hi)
    eta_from_hi = p_out / (p_out + loss_lo)
    w_plate = eta_hi - eta_lo
    w_temp = eta_from_hi - eta_from_lo
    return {
        "rise_k": rise,
        "r_th_prior_lo_k_per_w": float(r_lo),
        "r_th_prior_hi_k_per_w": float(r_hi),
        "r_th_prior_factor": float(r_hi / r_lo),
        "loss_lo_w": float(loss_lo),
        "loss_hi_w": float(loss_hi),
        "loss_factor": float(loss_hi / loss_lo),
        "efficiency_from_temperature_lo": float(eta_from_lo),
        "efficiency_from_temperature_hi": float(eta_from_hi),
        "efficiency_from_plate_lo": float(eta_lo),
        "efficiency_from_plate_hi": float(eta_hi),
        "plate_interval_points": float(100.0 * w_plate),
        "temperature_interval_points": float(100.0 * w_temp),
        "information_bits": float(np.log2(w_plate / w_temp)) if w_temp > 0 else float("inf"),
        "nominal_loss_seeded": bool(seeded),
    }


def two_node_reduction_error(
    *,
    r_winding_frame_k_per_w: float,
    c_winding_j_per_k: float,
    r_frame_ambient_k_per_w: float,
    c_frame_j_per_k: float,
    loss_w: float,
    ambient_c: float = 25.0,
    t_end_s: float = 10800.0,
    n_eval: int = 600,
    sample_interval_s: float = 3600.0,
) -> Dict[str, float]:
    """Cost of collapsing a two-node motor thermal model to one node.

    The two-node reference is the standard winding/frame lumped model: loss is
    injected into the winding, conducts to the frame through
    ``r_winding_frame``, and the frame convects to ambient through
    ``r_frame_ambient``. The one-node reduction uses the *correct* steady-state
    resistance (the series sum) and the slow time constant, which is the best
    one-node fit by construction, so this measures the structural error and not
    a fitting error.

    Two numbers matter and they are very different:

    ``max_abs_err_k``
        Worst transient error, dominated by the first few minutes when the
        winding is heating and the frame has not moved.

    ``max_abs_err_at_sampling_k``
        The same error **evaluated only at the instants the platform actually
        samples**, at ``sample_interval_s``. On a platform that reports one
        temperature an hour, the two-node transient has already finished between
        consecutive samples, so the reduction error at the sampled instants is
        very much smaller than the worst-case transient error. That is the
        honest justification for one node: not that two nodes are wrong, but
        that the sampling cannot see the difference.
    """
    for v in (r_winding_frame_k_per_w, c_winding_j_per_k, r_frame_ambient_k_per_w, c_frame_j_per_k):
        if v <= 0:
            raise ValueError("all two-node parameters must be positive")

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        tw, tf = y
        q_wf = (tw - tf) / r_winding_frame_k_per_w
        q_fa = (tf - ambient_c) / r_frame_ambient_k_per_w
        return np.array([(loss_w - q_wf) / c_winding_j_per_k, (q_wf - q_fa) / c_frame_j_per_k])

    t_eval = np.linspace(0.0, t_end_s, n_eval)
    sol = solve_ivp(
        rhs, (0.0, t_end_s), np.array([ambient_c, ambient_c]), t_eval=t_eval, rtol=1e-9, atol=1e-11
    )
    if not sol.success:
        raise RuntimeError(f"two-node integration failed: {sol.message}")
    t_two = sol.y[0]   # the winding is what a motor's temperature sensor reads

    r_eq = r_winding_frame_k_per_w + r_frame_ambient_k_per_w
    # Slow eigenvalue of the two-node system, which is what a one-node fit
    # recovers from a long heat run.
    a = np.array(
        [
            [-1.0 / (r_winding_frame_k_per_w * c_winding_j_per_k), 1.0 / (r_winding_frame_k_per_w * c_winding_j_per_k)],
            [
                1.0 / (r_winding_frame_k_per_w * c_frame_j_per_k),
                -(1.0 / r_winding_frame_k_per_w + 1.0 / r_frame_ambient_k_per_w) / c_frame_j_per_k,
            ],
        ]
    )
    tau_slow = float(-1.0 / np.max(np.linalg.eigvals(a).real))
    one = ThermalParams(r_th_k_per_w=r_eq, c_th_j_per_k=tau_slow / r_eq, ambient_c=ambient_c)
    t_one = temperature_trajectory_c(one, t_eval, np.full(n_eval, loss_w), t0_c=ambient_c)

    err = t_two - t_one
    sample_t = np.arange(sample_interval_s, t_end_s + 1e-9, sample_interval_s)
    err_at_sample = np.interp(sample_t, t_eval, err) if sample_t.size else np.array([0.0])
    return {
        "r_eq_k_per_w": r_eq,
        "tau_slow_s": tau_slow,
        "tau_fast_s": float(-1.0 / np.min(np.linalg.eigvals(a).real)),
        "steady_rise_two_node_k": float(t_two[-1] - ambient_c),
        "steady_rise_one_node_k": float(t_one[-1] - ambient_c),
        "max_abs_err_k": float(np.max(np.abs(err))),
        "t_at_max_err_s": float(t_eval[int(np.argmax(np.abs(err)))]),
        "max_abs_err_at_sampling_k": float(np.max(np.abs(err_at_sample))),
        "sample_interval_s": float(sample_interval_s),
        "n_samples": int(sample_t.size),
    }


def heat_run_identifiability(
    thermal: ThermalParams,
    loss_w: float,
    *,
    sample_interval_s: float,
    duration_s: float = 10800.0,
    noise_k: float = 0.4,
) -> Dict[str, float]:
    """Can ``R_th`` and ``C`` be separated from a sampled heating curve?

    Fisher information for :math:`(\\log R_{th}, \\log C)` from samples of the
    step response :math:`T(t) = T_a + R P (1 - e^{-t/RC})` with additive
    Gaussian noise. ``noise_k`` defaults to 0.4 K, which is the scatter
    ``pdmcore.sim.machine`` puts on its reported temperature and a realistic
    figure for an on-board digital sensor.

    The point is the dependence on ``sample_interval_s``. Sampling much faster
    than :math:`\\tau` resolves the exponential and separates the two
    parameters; sampling at or slower than :math:`\\tau` sees only the plateau,
    where the curve depends on :math:`R_{th}` alone and :math:`C` has no
    influence at all. The Fisher matrix then has a zero eigenvalue and the
    reported ``sigma_log_c`` goes to infinity, which is the correct answer.
    """
    if sample_interval_s <= 0 or duration_s <= sample_interval_s:
        raise ValueError("need duration_s > sample_interval_s > 0")
    t = np.arange(sample_interval_s, duration_s + 1e-9, sample_interval_s)
    theta = np.log(np.array([thermal.r_th_k_per_w, thermal.c_th_j_per_k]))

    def model(th: np.ndarray) -> np.ndarray:
        r, c = np.exp(th)
        return r * loss_w * (1.0 - np.exp(-t / (r * c)))

    h = 1e-6
    jac = np.empty((t.size, 2))
    for j in range(2):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        jac[:, j] = (model(tp) - model(tm)) / (2.0 * h)
    fim = jac.T @ jac / noise_k**2
    evals = np.linalg.eigvalsh(fim)
    cov = np.linalg.pinv(fim, rcond=1e-12)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return {
        "sample_interval_s": float(sample_interval_s),
        "samples_per_tau": float(thermal.tau_s / sample_interval_s),
        "n_samples": int(t.size),
        "tau_s": float(thermal.tau_s),
        "cond": float(evals.max() / max(evals.min(), 1e-300)),
        "sigma_log_r_th": float(sd[0]),
        "sigma_log_c_th": float(sd[1]),
        "c_identifiable": bool(sd[1] < 0.5),
    }
