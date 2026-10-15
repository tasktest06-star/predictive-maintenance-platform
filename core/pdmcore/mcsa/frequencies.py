"""Where the lines are, derived rather than quoted.

Everything here follows from one transformation and one perturbation argument.

The transformation: rotor to stator
-----------------------------------
A cage motor with ``p`` pole pairs on a supply at ``f1`` has a synchronous field
speed of ``f1/p`` revolutions per second. The rotor turns at ``fr = (1-s) f1/p``
where ``s`` is the slip, so the field passes the rotor at ``s f1`` -- which is
the frequency of the rotor currents.

A rotor MMF wave at frequency ``f'`` *in the rotor frame* is seen from the
stator at ``p fr +- f' = f1 (1-s) +- f'``. That single relation generates every
rotor-side family below; nothing else is needed.

The perturbation: air-gap permeance
-----------------------------------
Anything that changes the air gap changes the permeance seen by the stator MMF.
Write the gap as ``g(theta, t) = g0 (1 - eps cos(theta - omega t))``; to first
order in ``eps`` the flux acquires components at ``f1 +- omega/2pi``, each with
relative amplitude ``eps/2``. So a *geometric* perturbation at frequency ``fm``
puts current sidebands at ``f1 +- fm`` whose size is set by the displacement
divided by the air gap -- a number of order ``1e-3`` for a bearing defect and
``1e-1`` for a worn-out bearing in a motor that is about to fail.

The families
------------
=========================== ==========================================================
broken rotor bar / end ring ``f1 (1 +- 2ks)``, k = 1, 2, 3 ...
dynamic eccentricity        ``f1 +- k fr``
mixed eccentricity          ``f1 +- k fr`` (the low-order pair, strongly)
static eccentricity         no low-order sideband; slot harmonics and 2 f1 only
rotor slot harmonics        ``f1 [(kR +- nd)(1-s)/p +- 1]``
bearing (via the air gap)   ``|f1 +- m fchar|``
stator turn fault           negative-sequence fundamental; ``f1 (k +- m(1-s)/p)``
load torque oscillation     ``f1 +- k fosc``  -- indistinguishable by frequency
=========================== ==========================================================

Two of those rows are the reason this module exists. The broken-bar family and
the load-oscillation family are the same family when ``fosc = 2 s f1``, and the
eccentricity family and the load-oscillation family are the same family when
``fosc = fr``. Frequency alone cannot separate a fault from its confounder;
only the load dependence, the sideband symmetry or a second sensor can.

Derivation of the broken-bar family, since it is the one most often quoted
without one
-------------------------------------------------------------------------
A broken bar makes the rotor winding asymmetric, so the rotor current at ``s f1``
splits into a forward and a *backward* rotating component (the standard
decomposition of an unbalanced polyphase winding). Apply the transformation
above to the backward component, ``f' = -s f1``:

    stator-frame frequency = f1 (1-s) - s f1 = f1 (1 - 2s)

That is the lower sideband, and it is a *pure consequence of rotor asymmetry* --
it exists even with infinite rotor inertia. The resulting pulsating torque at
``2 s f1`` then produces a speed ripple at the same frequency, which modulates
the fundamental and generates ``f1 (1 + 2s)``; the upper sideband is therefore
*smaller than the lower one on a high-inertia drive and comparable on a
low-inertia one*, which is a diagnostic in itself and the reason both are
measured separately here. Iterating the interaction gives the ``k > 1`` members.

The slip problem
----------------
Slip is a difference of two nearly equal speeds, so its relative error is the
speed error amplified by ``1/s``:

    ds/s = -(dfr/fr) (1-s)/s  ~= -(dfr/fr)/s

At a rated slip of 1.1 %, a **0.5 % speed error is a 45 % slip error**, and the
2sf1 sideband offset moves by 0.6 Hz -- several resolution bandwidths at any
practical record length. :func:`sideband_position_error_hz` quantifies it and
:func:`min_record_duration_s` says how long a record has to be before the
sideband is resolvable at all. Both are used as gates, not as advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..dsp.windows import window

__all__ = [
    "MCSAFrequencies",
    "sync_speed_hz",
    "slip_from_shaft_hz",
    "shaft_hz_from_slip",
    "rotor_bar_sidebands",
    "eccentricity_sidebands",
    "rotor_slot_harmonics",
    "bearing_current_sidebands",
    "load_oscillation_sidebands",
    "stator_fault_frequencies",
    "sideband_position_error_hz",
    "slip_relative_error",
    "min_record_duration_s",
    "mcsa_frequencies",
    "family_collisions",
    "as_json_dict",
]


def sync_speed_hz(line_hz: float, pole_pairs: int) -> float:
    """Synchronous mechanical speed in Hz (revolutions per second)."""
    if line_hz <= 0:
        raise ValueError("line_hz must be positive")
    if pole_pairs < 1:
        raise ValueError("pole_pairs must be >= 1")
    return float(line_hz) / float(pole_pairs)


def slip_from_shaft_hz(line_hz: float, pole_pairs: int, shaft_hz: float) -> float:
    """Slip from a measured shaft speed.

    Not clamped to be positive. A negative result means the shaft is turning
    *faster* than synchronous, which for a motor means the speed input is wrong
    -- and silently clamping it to zero would hide exactly the input error that
    puts every predicted sideband on top of the fundamental. Callers that need a
    gate should use :func:`pdmcore.mcsa.supply.mcsa_admissible`.
    """
    ns = sync_speed_hz(line_hz, pole_pairs)
    return float((ns - float(shaft_hz)) / ns)


def shaft_hz_from_slip(line_hz: float, pole_pairs: int, slip: float) -> float:
    """Shaft speed implied by a slip. The inverse of :func:`slip_from_shaft_hz`."""
    return float(sync_speed_hz(line_hz, pole_pairs) * (1.0 - float(slip)))


def rotor_bar_sidebands(line_hz: float, slip: float, *, k_max: int = 3) -> List[float]:
    """``f1 (1 +- 2ks)`` for k = 1..k_max, ascending, negatives dropped.

    At ``s -> 0`` (no load) the whole family collapses onto the fundamental: the
    fault does not disappear, it becomes *unobservable in this coordinate*. That
    is the single most important practical fact about broken-bar detection and
    the reason :func:`min_record_duration_s` exists.
    """
    f1 = float(line_hz)
    out = []
    for k in range(1, int(k_max) + 1):
        for sign in (-1.0, 1.0):
            f = f1 * (1.0 + sign * 2.0 * k * float(slip))
            if f > 0.0:
                out.append(float(f))
    return sorted(out)


def eccentricity_sidebands(line_hz: float, shaft_hz: float, *, k_max: int = 2) -> List[float]:
    """``f1 +- k fr`` for k = 1..k_max.

    Produced by *dynamic* and *mixed* eccentricity. Static eccentricity alone
    does not: its gap perturbation is stationary in the stator frame, so it adds
    no new temporal frequency -- it shows up as a 2 f1 force (and hence 2 f1
    vibration) and in the slot-harmonic family. A detector that looks for
    ``f1 +- fr`` and finds nothing has therefore excluded dynamic eccentricity,
    not eccentricity.
    """
    f1, fr = float(line_hz), float(shaft_hz)
    out = []
    for k in range(1, int(k_max) + 1):
        for sign in (-1.0, 1.0):
            f = f1 + sign * k * fr
            if f > 0.0:
                out.append(float(f))
    return sorted(out)


def rotor_slot_harmonics(
    line_hz: float,
    slip: float,
    pole_pairs: int,
    rotor_slots: int,
    *,
    k_max: int = 1,
    ecc_orders: Sequence[int] = (0, 1, 2),
) -> List[float]:
    """``f1 [(kR +- nd)(1-s)/p +- 1]`` (Cameron/Thomson/Dow 1986).

    ``nd = 0`` gives the principal slot harmonics, present on a healthy machine;
    ``nd = 1, 2`` appear with eccentricity. The family lives at 0.5-3 kHz on a
    typical 4-pole machine with 28-40 rotor slots, so it needs a wider
    acquisition bandwidth than the fundamental-sideband families -- which is the
    reason the panel device samples at 10 kHz rather than at 1 kHz. It also
    needs ``R``, the rotor slot count, which is **not on the nameplate**; this
    is the main reason the slot-harmonic route is rarely used in the field.
    """
    f1, s = float(line_hz), float(slip)
    p = int(pole_pairs)
    r = int(rotor_slots)
    if r < 1:
        raise ValueError("rotor_slots must be >= 1")
    out = []
    for k in range(1, int(k_max) + 1):
        for nd in ecc_orders:
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    f = f1 * ((k * r + s1 * nd) * (1.0 - s) / p + s2)
                    if f > 0.0:
                        out.append(float(f))
    return sorted(set(out))


def bearing_current_sidebands(
    line_hz: float, characteristic_hz: float, *, m_max: int = 2
) -> List[float]:
    """``|f1 +- m fchar|`` for m = 1..m_max (Schoen et al. 1995).

    Reflected around zero rather than dropped, because bearing characteristic
    frequencies routinely exceed the supply frequency: BPFO on a 6205 at
    1780 rpm is 107 Hz against a 60 Hz supply, so the "lower" sideband is at
    ``|60 - 107| = 47 Hz`` and is a real line in a real spectrum.

    Nothing about this function says the line is *detectable*. See
    ``mcsa/research/01_method_note.md`` for the coupling budget: the air-gap
    path gives a sideband 55-75 dB below the fundamental for a defect that
    vibration sees at 20 dB above its own noise floor.
    """
    f1, fc = float(line_hz), float(characteristic_hz)
    out = []
    for m in range(1, int(m_max) + 1):
        for sign in (-1.0, 1.0):
            f = abs(f1 + sign * m * fc)
            if f > 0.0:
                out.append(float(f))
    return sorted(out)


def load_oscillation_sidebands(line_hz: float, osc_hz: float, *, k_max: int = 2) -> List[float]:
    """``f1 +- k fosc``. Identical in form to every family above.

    Present deliberately and named deliberately: a reciprocating compressor, a
    screw pump, a mixer with a baffle or a conveyor with a periodic load all
    oscillate the shaft torque, and the resulting current sidebands are
    *structurally indistinguishable* from a fault family at the same offset.
    The confounder is not noise, it is a signal with the same signature.
    """
    f1, fo = float(line_hz), float(osc_hz)
    out = []
    for k in range(1, int(k_max) + 1):
        for sign in (-1.0, 1.0):
            f = f1 + sign * k * fo
            if f > 0.0:
                out.append(float(f))
    return sorted(out)


def stator_fault_frequencies(
    line_hz: float, slip: float, pole_pairs: int, *, k_max: int = 3, m_max: int = 3
) -> List[float]:
    """``f1 [k +- m (1-s)/p]`` -- the turn-fault family of Penman et al. 1994.

    Reported for completeness. The *usable* indicator for a shorted turn is not
    in this list: it is the negative-sequence current (see
    :func:`pdmcore.mcsa.park.sequence_components`), because a shorted turn makes
    the stator winding itself asymmetric. The list is here because the
    negative-sequence indicator is confounded by supply unbalance and by
    instrument-channel gain mismatch, and these lines are the fallback when the
    supply is known to be dirty.
    """
    f1, s = float(line_hz), float(slip)
    out = []
    for k in range(1, int(k_max) + 1):
        for m in range(1, int(m_max) + 1):
            for sign in (-1.0, 1.0):
                f = f1 * (k + sign * m * (1.0 - s) / int(pole_pairs))
                if f > 0.0:
                    out.append(float(f))
    return sorted(set(out))


def slip_relative_error(slip: float, speed_rel_error: float) -> float:
    """Relative slip error from a relative shaft-speed error.

    ``ds/s = -(dfr/fr)(1-s)/s``. The ``1/s`` is the whole story: slip is a
    difference of two nearly equal numbers, so it inherits the speed error
    amplified by the reciprocal of itself.
    """
    s = float(slip)
    if s <= 0.0:
        raise ValueError("slip must be positive to have a relative error")
    return float(abs(speed_rel_error) * (1.0 - s) / s)


def sideband_position_error_hz(
    line_hz: float, slip: float, speed_rel_error: float, *, k: int = 1
) -> float:
    """How far the ``f1(1 +- 2ks)`` sideband moves for a given speed error.

    ``df = 2 k f1 |ds| = 2 k f1 s * (ds/s)``.
    """
    s = float(slip)
    return float(2.0 * int(k) * float(line_hz) * s * slip_relative_error(s, speed_rel_error))


def min_record_duration_s(
    line_hz: float,
    slip: float,
    *,
    k: int = 1,
    window_name: str = "blackman_harris",
    separation_enbw: float = 2.0,
    n_fft: int = 4096,
) -> float:
    """Shortest record in which the ``k``-th rotor-bar sideband is resolvable.

    The sideband sits ``2 k s f1`` from the fundamental. Two lines are resolved
    when their separation is at least ``separation_enbw`` times the equivalent
    noise bandwidth of the analysis, and the ENBW of a record of length ``T`` is
    ``enbw_bins / T``. So

        T >= separation_enbw * enbw_bins / (2 k s f1)

    The window matters twice over and in opposite directions: a
    Blackman-Harris window has 2.00 ENBW bins against Hann's 1.50, so it needs a
    33 % longer record -- and it is the only one of the two whose sidelobes are
    below the sideband being measured. ``separation_enbw = 2`` is the
    conventional "clearly separate lines" criterion.

    The practical consequence is a hard floor on what can be monitored: at 10 %
    load a 4-pole 60 Hz motor has ``s ~ 0.001``, giving ``2sf1 = 0.13 Hz`` and a
    required record of **31 s**. Supply-frequency wander over 31 s then smears
    both lines by more than their separation, so the answer is not "acquire for
    longer" -- it is "this measurement cannot be made at this load".
    """
    s = float(slip)
    if s <= 0.0:
        return float("inf")
    enbw = window(window_name, n_fft).enbw_bins
    return float(separation_enbw * enbw / (2.0 * int(k) * s * float(line_hz)))


@dataclass(frozen=True)
class MCSAFrequencies:
    """Every predicted line for one operating point, with its provenance.

    ``families`` maps a fault-mode name to its predicted frequencies. The
    operating point is carried alongside because a line list without the slip
    and speed it was computed from cannot be checked, and a current spectrum
    analysed against the wrong operating point produces confident nonsense at
    the right-looking frequencies.
    """

    line_hz: float
    pole_pairs: int
    shaft_hz: float
    slip: float
    families: Dict[str, List[float]] = field(default_factory=dict)
    #: Position uncertainty of the 2sf1 pair, Hz, from the speed uncertainty.
    sideband_position_error_hz: float = 0.0
    #: Shortest record length that resolves the 2sf1 pair, seconds.
    min_duration_s: float = float("inf")

    def all_lines(self) -> List[float]:
        out: List[float] = []
        for v in self.families.values():
            out.extend(v)
        return sorted(set(out))

    def as_dict(self) -> Dict[str, object]:
        return {
            "line_hz": self.line_hz,
            "pole_pairs": self.pole_pairs,
            "shaft_hz": self.shaft_hz,
            "slip": self.slip,
            "families": {k: list(v) for k, v in self.families.items()},
            "sideband_position_error_hz": self.sideband_position_error_hz,
            "min_duration_s": self.min_duration_s,
        }


def mcsa_frequencies(
    line_hz: float,
    pole_pairs: int,
    *,
    shaft_hz: Optional[float] = None,
    slip: Optional[float] = None,
    speed_rel_error: float = 0.005,
    bearing_hz: Optional[Dict[str, float]] = None,
    rotor_slots: Optional[int] = None,
    load_osc_hz: float = 0.0,
) -> MCSAFrequencies:
    """Assemble the predicted line list for one operating point.

    Exactly one of ``shaft_hz`` or ``slip`` must be given; the other is derived.
    Requiring one of them is the point -- there is no default. Every family here
    except the bearing one moves with slip, and a default slip would produce a
    line list that looks authoritative and points at the wrong bins.
    """
    if (shaft_hz is None) == (slip is None):
        raise ValueError(
            "give exactly one of shaft_hz or slip: the other is derived. MCSA "
            "without a speed or a slip is not a degraded measurement, it is a "
            "list of frequencies for a machine that may not be this one."
        )
    if slip is None:
        slip = slip_from_shaft_hz(line_hz, pole_pairs, float(shaft_hz))
    else:
        shaft_hz = shaft_hz_from_slip(line_hz, pole_pairs, float(slip))
    slip = float(slip)
    shaft_hz = float(shaft_hz)

    fam: Dict[str, List[float]] = {
        "rotor_bar": rotor_bar_sidebands(line_hz, max(slip, 0.0), k_max=3),
        "eccentricity": eccentricity_sidebands(line_hz, shaft_hz, k_max=2),
        "stator_turn": stator_fault_frequencies(line_hz, slip, pole_pairs, k_max=2, m_max=2),
        "supply_harmonic": [float(line_hz) * h for h in (3, 5, 7, 11, 13)],
    }
    if rotor_slots:
        fam["rotor_slot"] = rotor_slot_harmonics(line_hz, slip, pole_pairs, rotor_slots)
    if bearing_hz:
        for name, fc in bearing_hz.items():
            if fc and fc > 0:
                fam[f"bearing_{name}"] = bearing_current_sidebands(line_hz, fc, m_max=2)
    if load_osc_hz > 0:
        fam["load_oscillation"] = load_oscillation_sidebands(line_hz, load_osc_hz, k_max=2)

    err = (sideband_position_error_hz(line_hz, slip, speed_rel_error)
           if slip > 0 else float("inf"))
    return MCSAFrequencies(
        line_hz=float(line_hz),
        pole_pairs=int(pole_pairs),
        shaft_hz=shaft_hz,
        slip=slip,
        families=fam,
        sideband_position_error_hz=float(err),
        min_duration_s=min_record_duration_s(line_hz, slip),
    )


def family_collisions(
    freqs: MCSAFrequencies, *, tol_hz: float = 0.25
) -> List[Tuple[str, str, float, float]]:
    """Pairs of lines from *different* families that fall within ``tol_hz``.

    Returned as ``(family_a, family_b, f_a, f_b)``. This is not a diagnostic
    nicety: at the default operating point of the reference 37 kW motor the
    load-oscillation family at ``fosc = 2sf1`` lands exactly on the broken-bar
    pair, and the eccentricity pair at ``f1 +- fr`` lands within 0.3 Hz of a
    bearing cage sideband. A detector that reports a mode without checking this
    list is reporting the first mode in its iteration order.
    """
    items: List[Tuple[str, float]] = []
    for name, lines in freqs.families.items():
        for f in lines:
            items.append((name, f))
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            na, fa = items[i]
            nb, fb = items[j]
            if na != nb and abs(fa - fb) <= tol_hz:
                out.append((na, nb, fa, fb))
    return out


def as_json_dict(freqs: MCSAFrequencies) -> Dict[str, object]:
    """``MCSAFrequencies`` plus its collision list, for a results file."""
    d = dict(freqs.as_dict())
    d["collisions"] = [
        {"a": a, "b": b, "f_a_hz": fa, "f_b_hz": fb}
        for a, b, fa, fb in family_collisions(freqs)
    ]
    return d
