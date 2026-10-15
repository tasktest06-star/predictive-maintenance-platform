r"""Rotor-bearing dynamics: unbalance response, critical speed, and the
degeneracy that a fixed-speed machine cannot escape.

The model
--------
One lumped mass on an isotropic flexible support with viscous damping -- a
Jeffcott rotor -- excited by residual unbalance :math:`U = m_u e` [kg.m]:

.. math::

    m \ddot x + c \dot x + k x = U \omega^2 \cos\omega t

With :math:`\omega_n = \sqrt{k/m}`, :math:`\zeta = c/(2\sqrt{km})` and
:math:`r = \omega/\omega_n`, the shaft displacement amplitude is

.. math::

    X(\omega) = \frac{U}{m}\,\frac{r^2}{\omega_n^2}\,
                \frac{1}{\sqrt{(1-r^2)^2 + (2\zeta r)^2}}

and the force transmitted into the housing, which is what an accelerometer on
the housing actually responds to, is

.. math::

    F_t(\omega) = kX\sqrt{1 + (2\zeta r)^2}
                = U\omega_n^2\,\frac{r^2\sqrt{1+(2\zeta r)^2}}
                                     {\sqrt{(1-r^2)^2+(2\zeta r)^2}}

Note what happened to the rotor mass: it cancelled. Transmitted force depends on
:math:`U`, :math:`\omega_n`, :math:`\zeta` and :math:`r` only. The mass and the
stiffness are individually irrelevant to everything this twin predicts, which is
why :func:`transmitted_force_n` takes :math:`\omega_n` rather than :math:`k` and
:math:`m`, and why :class:`pdmcore.twin.params.RotorParams` carries mass and
stiffness only so that the critical speed has a physical derivation.

The degeneracy
--------------
The measured casing acceleration at 1x is :math:`a_{1x} = G\,F_t(\omega)` with
:math:`G` the path receptance in g/N. So

.. math::

    a_{1x} = \underbrace{G U \omega_n^2}_{A}\;\times\;
             \underbrace{\frac{r^2\sqrt{1+(2\zeta r)^2}}
                               {\sqrt{(1-r^2)^2+(2\zeta r)^2}}}_{\text{shape}(r)}

At a single speed there is **one** observation and **three** unknowns
(:math:`A`, :math:`\omega_n`, :math:`\zeta`), and :math:`\omega_n` appears in
both factors. The degeneracy is not approximate and not a small-sample problem:
:func:`equivalent_rotors` constructs, for any target 1x amplitude, a continuous
one-parameter family of rotors that produce it *exactly*. A direct-on-line motor
runs at one speed. Therefore:

    **On a fixed-speed machine the rotor model is unidentifiable, and the
    twin cannot distinguish "the unbalance grew" from "the support softened".**

Those two have opposite maintenance actions -- balance the rotor, or tighten the
foundation -- so this is not an academic distinction. What breaks it is
observing the *shape*: a VFD asset sweeping its normal speed range, or a single
run-up. :func:`identifiability_from_sweep` quantifies how much speed range is
needed.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .params import PathParams, RotorParams

__all__ = [
    "unbalance_response_m",
    "transmitted_force_n",
    "response_shape",
    "casing_accel_1x_g",
    "equivalent_rotors",
    "identifiability_from_sweep",
    "unbalance_grade_kgm",
]


def response_shape(r: float | np.ndarray, zeta: float) -> np.ndarray:
    r"""The dimensionless transmitted-force shape :math:`\mathrm{shape}(r)`.

    Rises as :math:`r^2` well below resonance, peaks near :math:`r = 1` at
    roughly :math:`1/(2\zeta)`, and tends to 1 well above. The low-speed
    :math:`r^2` is the ISO-20816 fact in disguise: unbalance *force* grows with
    speed squared, which is why a velocity limit (one integration, one power of
    :math:`\omega` removed) is a roughly speed-independent criterion and an
    acceleration limit is not.
    """
    r = np.asarray(r, dtype=np.float64)
    if zeta <= 0:
        raise ValueError("zeta must be positive")
    num = r**2 * np.sqrt(1.0 + (2.0 * zeta * r) ** 2)
    den = np.sqrt((1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2)
    return num / den


def unbalance_response_m(
    unbalance_kgm: float,
    rotor_mass_kg: float,
    critical_speed_hz: float,
    damping_ratio: float,
    shaft_hz: float | np.ndarray,
) -> np.ndarray:
    """Shaft displacement amplitude in metres.

    Included for completeness and for the proximity-probe case. A
    condition-monitoring deployment of the kind this platform builds measures
    casing acceleration, not shaft displacement, so this is **not** on the path
    to any prediction the twin makes -- see :func:`casing_accel_1x_g`. It is
    here because the ISO 21940 balance grades are specified in terms of it and
    :func:`unbalance_grade_kgm` needs the link.
    """
    if rotor_mass_kg <= 0 or critical_speed_hz <= 0:
        raise ValueError("rotor_mass_kg and critical_speed_hz must be positive")
    r = np.asarray(shaft_hz, dtype=np.float64) / critical_speed_hz
    den = np.sqrt((1.0 - r**2) ** 2 + (2.0 * damping_ratio * r) ** 2)
    # X = (U/m) r^2 / den. Tends to U/m well above resonance -- the shaft orbits
    # about its mass centre and the amplitude is the eccentricity itself, which
    # is the standard check that the algebra is right.
    return unbalance_kgm / rotor_mass_kg * r**2 / den


def transmitted_force_n(
    unbalance_kgm: float,
    critical_speed_hz: float,
    damping_ratio: float,
    shaft_hz: float | np.ndarray,
) -> np.ndarray:
    """Force amplitude transmitted into the housing, in newtons.

    Independent of rotor mass and support stiffness separately -- only their
    ratio, through ``critical_speed_hz``. See the module docstring.
    """
    wn = 2.0 * np.pi * critical_speed_hz
    r = np.asarray(shaft_hz, dtype=np.float64) / critical_speed_hz
    return unbalance_kgm * wn**2 * response_shape(r, damping_ratio)


def casing_accel_1x_g(
    rotor: RotorParams,
    path: PathParams,
    shaft_hz: float | np.ndarray,
) -> np.ndarray:
    """Predicted 1x casing acceleration amplitude in g.

    This is the twin's only absolute-amplitude vibration prediction, and
    :mod:`pdmcore.twin.validity` gates it behind ``path.impact_tested``. The
    reason is visible in the arithmetic: the result is linear in
    ``path.low_freq_gain_g_per_n``, which no passive measurement constrains.
    Without the impact test the twin can report the *ratio* of this quantity
    between two operating points -- which is gain-free, and is what trending
    needs -- but not its value.
    """
    return path.low_freq_gain_g_per_n * transmitted_force_n(
        rotor.unbalance_kgm, rotor.critical_speed_hz, rotor.damping_ratio, shaft_hz
    )


def equivalent_rotors(
    target_accel_g: float,
    shaft_hz: float,
    *,
    critical_speed_hz_grid: Sequence[float],
    damping_ratio: float = 0.05,
    path_gain_g_per_n: float = 1.0e-4,
) -> Dict[str, np.ndarray]:
    """Rotors that produce *exactly* ``target_accel_g`` at one speed.

    For each candidate critical speed, solve for the unbalance that reproduces
    the observed 1x amplitude. All of them fit the single observation perfectly;
    they differ by orders of magnitude in unbalance and they disagree completely
    about what happens at any other speed. That disagreement is the content of
    the identifiability claim, and ``spread_at_1_3x`` quantifies it: the ratio
    between the largest and smallest 1x amplitude these indistinguishable rotors
    predict at 1.3 times the observed speed.

    Returns ``critical_speed_hz``, ``unbalance_kgm``, ``accel_at_obs_g`` (which
    must be constant, and is asserted to be), ``accel_at_1_3x_g``, and the
    summary ratios.
    """
    if target_accel_g <= 0 or shaft_hz <= 0:
        raise ValueError("target_accel_g and shaft_hz must be positive")
    fn = np.asarray(critical_speed_hz_grid, dtype=np.float64)
    if np.any(fn <= 0):
        raise ValueError("critical speeds must be positive")
    shape_obs = response_shape(shaft_hz / fn, damping_ratio)
    wn = 2.0 * np.pi * fn
    u = target_accel_g / (path_gain_g_per_n * wn**2 * shape_obs)

    a_obs = path_gain_g_per_n * u * wn**2 * shape_obs
    a_hi = path_gain_g_per_n * u * wn**2 * response_shape(1.3 * shaft_hz / fn, damping_ratio)
    if not np.allclose(a_obs, target_accel_g, rtol=1e-10):
        raise AssertionError("equivalent-rotor construction failed to reproduce the observation")
    return {
        "critical_speed_hz": fn,
        "unbalance_kgm": u,
        "accel_at_obs_g": a_obs,
        "accel_at_1_3x_g": a_hi,
        "unbalance_spread": np.array([float(u.max() / u.min())]),
        "spread_at_1_3x": np.array([float(a_hi.max() / a_hi.min())]),
    }


def identifiability_from_sweep(
    shaft_hz: Sequence[float],
    *,
    critical_speed_hz: float,
    damping_ratio: float,
    rel_noise: float = 0.10,
) -> Dict[str, float]:
    r"""Fisher information for :math:`(\log A, \log \omega_n, \log \zeta)` from a
    speed sweep, with multiplicative noise.

    The model is :math:`\log a_k = \log A + \log\mathrm{shape}(\omega_k/\omega_n,
    \zeta) + \epsilon_k`, :math:`\epsilon \sim N(0, \sigma^2)` with
    :math:`\sigma = ` ``rel_noise``. Log parameters make the Fisher matrix
    dimensionless and the eigenvalues directly comparable; ``rel_noise`` is a
    *relative* amplitude scatter because that is what healthy-machine repeat
    measurements show (``pdmcore.sim.machine`` uses 10-12 % and explains why).

    Returned fields:

    ``cond``
        Condition number of the Fisher matrix. Above ~1e6 the sweep is
        effectively rank-deficient and the third parameter is not identified.
    ``sigma_log_*``
        Cramer-Rao standard deviations of the log parameters, i.e. relative
        standard errors. ``sigma_log_wn = 0.05`` means 5 % on the critical speed.
    ``r_range``
        The sweep's span in ``r = omega/omega_n``. The driver of everything: a
        sweep that never leaves ``r << 1`` sees only the :math:`r^2` asymptote,
        in which ``shape`` depends on :math:`\omega_n` and :math:`A` only through
        the product, so the matrix is singular by construction rather than by
        numerical accident.
    """
    f = np.asarray(shaft_hz, dtype=np.float64)
    if f.size < 3:
        raise ValueError("need at least 3 speeds to identify 3 parameters")
    if rel_noise <= 0:
        raise ValueError("rel_noise must be positive")
    theta = np.log(np.array([1.0, critical_speed_hz, damping_ratio]))

    def log_model(th: np.ndarray) -> np.ndarray:
        a, wn, z = np.exp(th)
        return np.log(a * response_shape(f / wn, z))

    # Central differences in log space; 1e-5 is comfortably inside the
    # well-conditioned window for a function this smooth.
    h = 1e-5
    jac = np.empty((f.size, 3))
    for j in range(3):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        jac[:, j] = (log_model(tp) - log_model(tm)) / (2.0 * h)
    fim = jac.T @ jac / rel_noise**2
    evals = np.linalg.eigvalsh(fim)
    cond = float(evals.max() / max(evals.min(), 1e-300))
    # Pseudo-inverse, because a genuinely singular matrix must not raise here --
    # reporting an infinite standard error is the correct answer, not an error.
    cov = np.linalg.pinv(fim, rcond=1e-12)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    r = f / critical_speed_hz
    return {
        "n_speeds": int(f.size),
        "r_min": float(r.min()),
        "r_max": float(r.max()),
        "r_range": float(r.max() - r.min()),
        "cond": cond,
        "eig_min": float(evals.min()),
        "eig_max": float(evals.max()),
        "sigma_log_amplitude": float(sd[0]),
        "sigma_log_wn": float(sd[1]),
        "sigma_log_zeta": float(sd[2]),
        "rank_deficient": bool(cond > 1e6),
    }


def unbalance_grade_kgm(
    rotor_mass_kg: float, shaft_hz: float, grade_mm_per_s: float = 2.5
) -> float:
    """ISO 21940-11 balance grade G expressed as an unbalance moment.

    The grade is defined as a permissible *specific* unbalance
    ``e_per = G / omega`` in units of ``mm/s / (rad/s) = micrometres``, so the
    permissible unbalance moment is ``m * e_per``. G2.5 is the usual acceptance
    grade for pumps, fans and electric motors above about 1 kW.

    Used to give the identification a physically honest prior: an installed
    machine that was balanced to G2.5 and has since degraded sits somewhere
    between 1x and ~10x that value, which is a far tighter prior than anything
    the vibration data provides, and it is the only reason the unbalance /
    path-gain product can be split at all.
    """
    if rotor_mass_kg <= 0 or shaft_hz <= 0:
        raise ValueError("rotor_mass_kg and shaft_hz must be positive")
    e_per_m = (grade_mm_per_s * 1e-3) / (2.0 * np.pi * shaft_hz)
    return float(rotor_mass_kg * e_per_m)
