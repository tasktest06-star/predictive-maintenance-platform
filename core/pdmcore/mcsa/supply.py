"""The gates. What has to be true before a current spectrum means anything.

A vibration channel needs one external fact: the shaft speed. A current channel
needs four, and every one of them is a way for a confident wrong answer to get
out:

1. **The slip**, to a *relative* accuracy of order 20 %. Not the speed -- the
   slip, which inherits the speed error amplified by ``1/s`` (see
   :mod:`pdmcore.mcsa.frequencies`).
2. **The supply unbalance**, measured, because a 1 % voltage unbalance produces
   5-8 % negative-sequence current and that is larger than the signature of the
   stator fault the negative sequence is supposed to detect.
3. **Whether the machine is inverter-fed**, because an inverter moves the
   fundamental, regulates part of the fault signature away, and adds a
   switching-frequency family that overlaps the rotor-slot family.
4. **Whether the load is steady**, because a torque oscillation at ``2 s f1``
   produces exactly the broken-bar signature, at exactly the broken-bar
   frequency, with the right load dependence.

Nothing here estimates its way around a missing input. :func:`mcsa_admissible`
returns the list of blockers, and the production agent refuses the analysis
rather than publishing a score -- the same choice ``pdmedge/calibration.py``
makes for an unknown sensitivity, for the same reason.

The circularity that this module exists to prevent
--------------------------------------------------
Slip can be estimated from the current spectrum itself. If it is estimated from
the ``f1(1 +- 2ks)`` pair and the ``f1(1 +- 2ks)`` pair is then reported as
evidence of a broken bar, **the test cannot fail**: any pair of noise peaks
symmetric about the fundamental defines a slip at which they are the
broken-bar sidebands. This is the same defect class as the six in
``docs/08_branch_catalogue.md``, and it is the single easiest mistake to make in
MCSA because the result looks better the worse the data is.

:func:`estimate_slip_from_sidebands` therefore refuses ``family="rotor_bar"``
and uses the rotation family ``f1 +- k fr`` instead, which depends on the shaft
speed and not on the fault being tested for. Its own circularity -- it cannot be
used to diagnose eccentricity -- is recorded in the returned
``independent_of`` field, and :func:`mcsa_admissible` honours it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dsp.bearing import line_snr
from ..dsp.spectrum import Spectrum, interpolate_peak, spectrum
from .frequencies import slip_relative_error, sync_speed_hz
from .park import (
    clarke_transform,
    negative_sequence_impedance,
    sequence_components,
    unbalance_pct,
)

__all__ = [
    "SupplyAssessment",
    "SlipEstimate",
    "assess_supply",
    "estimate_slip_from_sidebands",
    "mcsa_admissible",
    "detect_vfd",
]

#: IEC 60034-26 / NEMA MG-1 treat 1 % voltage unbalance as the point at which
#: derating starts. It is used here as the gate for the negative-sequence
#: stator indicator because above it the supply's contribution dominates.
UNBALANCE_BLOCK_PCT = 1.0
#: A 3-wire supply has ia+ib+ic == 0 by Kirchhoff. Anything above this is an
#: instrument fault (a reversed clamp, a mismatched channel gain, one CT on the
#: wrong conductor), not a machine fault.
ZERO_SEQUENCE_BLOCK = 0.05
#: Largest relative slip error at which a 2sf1 sideband search is still
#: pointing at the right bins for a 1 s..30 s record. Above it the search
#: window is wider than the distance to the fundamental.
SLIP_REL_ERROR_BLOCK = 0.30


@dataclass
class SupplyAssessment:
    """What the supply is doing, measured rather than assumed."""

    f1_hz: float
    i_unbalance_pct: float
    v_unbalance_pct: float = float("nan")
    zero_sequence_ratio: float = 0.0
    thd_i_pct: float = float("nan")
    vfd: bool = False
    vfd_evidence_db: float = float("nan")
    carrier_hz: float = float("nan")
    z_neg_ratio: float = float("nan")
    z_valid: float = 0.0
    three_phase: bool = False
    voltage_present: bool = False
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "f1_hz": self.f1_hz,
            "i_unbalance_pct": self.i_unbalance_pct,
            "v_unbalance_pct": self.v_unbalance_pct,
            "zero_sequence_ratio": self.zero_sequence_ratio,
            "thd_i_pct": self.thd_i_pct,
            "vfd": bool(self.vfd),
            "vfd_evidence_db": self.vfd_evidence_db,
            "carrier_hz": self.carrier_hz,
            "z_neg_ratio": self.z_neg_ratio,
            "z_valid": self.z_valid,
            "three_phase": bool(self.three_phase),
            "voltage_present": bool(self.voltage_present),
            "notes": list(self.notes),
        }


@dataclass
class SlipEstimate:
    """A slip with an error bound and a statement of what it must not be used for."""

    slip: float
    rel_error: float
    method: str
    valid: bool
    #: Fault families whose diagnosis this estimate must NOT be used for,
    #: because the estimate was derived from that family's own lines.
    independent_of: Tuple[str, ...] = ()
    snr: float = 0.0

    @property
    def abs_error(self) -> float:
        return float(abs(self.slip) * self.rel_error)

    def as_dict(self) -> Dict[str, object]:
        return {
            "slip": self.slip,
            "rel_error": self.rel_error,
            "abs_error": self.abs_error,
            "method": self.method,
            "valid": bool(self.valid),
            "independent_of": list(self.independent_of),
            "snr": self.snr,
        }


def detect_vfd(
    phase: Sequence[float], fs_hz: float, *, line_hz: float = 50.0,
    n_fft: int = 8192, floor_from_hz: float = 1000.0,
) -> Tuple[bool, float, float]:
    """``(is_vfd, evidence_db, carrier_hz)`` from the high-frequency content.

    A direct-on-line motor's current above ~1 kHz is broadband and tiny. An
    inverter puts a carrier family at 2-16 kHz with discrete lines, so the test
    is the *prominence of the largest discrete line* above 1 kHz against the
    local floor there, not the total high-frequency energy -- a dirty supply or
    a noisy clamp raises the energy without producing a line.

    Returns ``evidence_db`` so the caller can see how marginal the call was.
    A sampling rate below ~2x the carrier makes this blind, in which case the
    carrier aliases down into the band being analysed, which is worse than not
    detecting it. ``fs_hz < 2 * floor_from_hz`` therefore returns
    ``(False, nan, nan)`` rather than a verdict.
    """
    x = np.asarray(phase, dtype=np.float64).ravel()
    if fs_hz < 2.0 * floor_from_hz:
        return False, float("nan"), float("nan")
    n = int(min(n_fft, 1 << int(np.floor(np.log2(max(2, x.size))))))
    sp = spectrum(x, fs_hz, n_fft=n, window_name="hann", scaling="rms")
    m = (sp.freq_hz >= floor_from_hz) & (sp.freq_hz <= 0.45 * fs_hz)
    if not np.any(m):
        return False, float("nan"), float("nan")
    v = np.asarray(sp.value)[m]
    f = sp.freq_hz[m]
    floor = float(np.median(v))
    i = int(np.argmax(v))
    ev = 20.0 * np.log10(v[i] / floor) if floor > 0 and v[i] > 0 else 0.0
    return bool(ev > 20.0), float(ev), float(f[i])


def assess_supply(
    current_abc: Sequence[Sequence[float]],
    fs_hz: float,
    *,
    line_hz: float = 50.0,
    voltage_abc: Optional[Sequence[Sequence[float]]] = None,
    n_fft: int = 16384,
) -> SupplyAssessment:
    """Measure the supply conditions that decide whether MCSA is admissible."""
    i = np.asarray(current_abc, dtype=np.float64)
    if i.ndim == 1:
        i = i[None, :]
    three = i.shape[0] == 3
    n = int(min(n_fft, 1 << int(np.floor(np.log2(max(2, i.shape[1]))))))
    sp = spectrum(i[0], fs_hz, n_fft=n, window_name="hann", scaling="rms")
    m = np.abs(sp.freq_hz - line_hz) <= 5.0
    if np.any(m):
        idx = int(np.flatnonzero(m)[int(np.argmax(sp.value[m]))])
        f1, a1 = interpolate_peak(sp, idx)
    else:  # pragma: no cover - only for a record with no supply in it
        f1, a1 = float(line_hz), 0.0

    notes: List[str] = []
    thd = float("nan")
    if a1 > 0:
        hs = []
        for h in range(2, 14):
            hs.append(line_snr(sp, f1 * h, tol_hz=max(1.5 * sp.df_hz, 0.5))["amplitude"])
        thd = float(100.0 * np.sqrt(float(np.sum(np.square(hs)))) / a1)

    unb_i = 0.0
    zratio = 0.0
    if three:
        seq_i = sequence_components(i[0], i[1], i[2], fs_hz, f1)
        unb_i = min(unbalance_pct(seq_i), 1000.0)
        zratio = clarke_transform(i[0], i[1], i[2], fs_hz).zero_sequence_ratio()
    else:
        notes.append("single phase: unbalance, EPVA and sequence indicators unavailable")

    unb_v = float("nan")
    z_ratio = float("nan")
    z_valid = 0.0
    have_v = voltage_abc is not None
    if have_v and three:
        v = np.asarray(voltage_abc, dtype=np.float64)
        if v.shape != i.shape:
            raise ValueError(f"voltage shape {v.shape} != current shape {i.shape}")
        seq_v = sequence_components(v[0], v[1], v[2], fs_hz, f1)
        unb_v = min(unbalance_pct(seq_v), 1000.0)
        z = negative_sequence_impedance(seq_i, seq_v)
        z_ratio = z["z_ratio"]
        z_valid = z["valid"]
        if not z_valid:
            notes.append(
                "negative-sequence impedance not usable: voltage unbalance below "
                "0.3 %, so Z- is a ratio of two instrument-noise numbers"
            )
    elif not have_v:
        notes.append(
            "no voltage channel: supply unbalance cannot be measured, so the "
            "negative-sequence stator indicator is inadmissible (a current "
            "unbalance has two causes and this instrument sees one of them)"
        )

    vfd, ev, carrier = detect_vfd(i[0], fs_hz, line_hz=line_hz)
    if np.isnan(ev):
        notes.append(
            f"VFD detection blind: fs = {fs_hz:.0f} Hz cannot see a switching "
            f"carrier, and an undetected carrier aliases into the analysis band"
        )
    elif vfd:
        notes.append(f"inverter carrier at {carrier:.0f} Hz, {ev:.0f} dB above the local floor")

    return SupplyAssessment(
        f1_hz=float(f1),
        i_unbalance_pct=float(unb_i),
        v_unbalance_pct=float(unb_v),
        zero_sequence_ratio=float(zratio),
        thd_i_pct=thd,
        vfd=bool(vfd),
        vfd_evidence_db=float(ev),
        carrier_hz=float(carrier),
        z_neg_ratio=float(z_ratio),
        z_valid=float(z_valid),
        three_phase=bool(three),
        voltage_present=bool(have_v),
        notes=notes,
    )


def estimate_slip_from_sidebands(
    phase: Sequence[float],
    fs_hz: float,
    *,
    line_hz: float,
    pole_pairs: int,
    family: str = "rotation",
    slip_bracket: Tuple[float, float] = (0.002, 0.06),
    n_fft: int = 32768,
    window_name: str = "blackman_harris",
    min_snr: float = 4.0,
) -> SlipEstimate:
    """Slip from the rotation sidebands ``f1 +- fr``, with its error bound.

    The search runs over the slip bracket, scores the *symmetric pair* (the
    geometric mean of the two sideband SNRs, so a single noise peak on one side
    cannot win), and takes the best. Returning the geometric mean rather than
    the sum is what makes the estimate robust: an asymmetric pair is evidence of
    something, but it is not evidence of a rotation rate.

    ``family="rotor_bar"`` is refused. See the module docstring: estimating slip
    from the family that is then used as evidence is a check that cannot fail.

    The error bound is the half-width of the slip interval that the frequency
    resolution cannot distinguish:
    ``d_slip = (df/2) * p / f1`` from ``fr = (1-s) f1/p``.
    """
    if family == "rotor_bar":
        raise ValueError(
            "refusing to estimate slip from the rotor-bar family: the estimate "
            "would place the sidebands wherever the strongest symmetric pair of "
            "peaks happens to be, and the subsequent 'detection' of that pair "
            "could not fail. Use family='rotation' (f1 +- fr, independent of "
            "rotor asymmetry), a tacho, or the VFD/PLC speed feedback."
        )
    if family != "rotation":
        raise ValueError(f"unknown family {family!r}; expected 'rotation'")

    x = np.asarray(phase, dtype=np.float64).ravel()
    n = int(min(n_fft, 1 << int(np.floor(np.log2(max(2, x.size))))))
    sp = spectrum(x, fs_hz, n_fft=n, window_name=window_name, scaling="rms")
    m = np.abs(sp.freq_hz - line_hz) <= 5.0
    f1 = float(interpolate_peak(sp, int(np.flatnonzero(m)[int(np.argmax(sp.value[m]))]))[0]) \
        if np.any(m) else float(line_hz)

    ns = sync_speed_hz(f1, pole_pairs)
    s_lo, s_hi = slip_bracket
    # One candidate per half-bin of fr, so the search cannot skip a line.
    step = 0.5 * sp.df_hz * pole_pairs / f1
    cands = np.arange(s_lo, s_hi + step, max(step, 1e-6))
    best = (0.0, 0.0)
    tol = 1.5 * sp.df_hz
    for s in cands:
        fr = ns * (1.0 - s)
        lo = line_snr(sp, f1 - fr, tol_hz=tol, floor_width_hz=10.0)["snr"]
        hi = line_snr(sp, f1 + fr, tol_hz=tol, floor_width_hz=10.0)["snr"]
        score = float(np.sqrt(max(lo, 0.0) * max(hi, 0.0)))
        if score > best[1]:
            best = (float(s), score)
    slip, snr = best
    d_slip = 0.5 * sp.df_hz * pole_pairs / f1
    rel = float(d_slip / slip) if slip > 0 else float("inf")
    return SlipEstimate(
        slip=slip,
        rel_error=rel,
        method=f"rotation_sidebands(df={sp.df_hz:.4f}Hz)",
        valid=bool(snr >= min_snr and slip > 0),
        independent_of=("rotor_bar", "bearing", "stator_turn"),
        snr=float(snr),
    )


def mcsa_admissible(
    assessment: SupplyAssessment,
    slip: SlipEstimate,
    *,
    mode: str = "rotor_bar",
    speed_rel_error: Optional[float] = None,
) -> Dict[str, object]:
    """``{"ok": bool, "blockers": [...], "warnings": [...]}`` for one fault mode.

    The gates differ by mode because the confounders do:

    * ``rotor_bar`` needs slip and a steady load; supply unbalance only degrades
      it, so unbalance is a warning here and a blocker for ``stator_turn``.
    * ``stator_turn`` needs voltage, because the negative-sequence current it
      looks at is produced more strongly by a 1 % supply unbalance than by the
      fault.
    * ``eccentricity`` must not be diagnosed from a slip that was itself
      estimated from the rotation sidebands -- ``slip.independent_of`` makes
      that explicit, and this function enforces it.
    * every mode is blocked on an inverter-fed machine, and on a non-zero
      zero-sequence current, which is an instrument fault.
    """
    blockers: List[str] = []
    warnings: List[str] = []

    if not slip.valid:
        blockers.append(f"slip estimate not valid (method {slip.method}, snr {slip.snr:.1f})")
    rel = slip.rel_error if speed_rel_error is None else slip_relative_error(
        max(slip.slip, 1e-9), speed_rel_error)
    if rel > SLIP_REL_ERROR_BLOCK:
        blockers.append(
            f"slip relative error {rel * 100:.0f} % exceeds {SLIP_REL_ERROR_BLOCK * 100:.0f} %: "
            f"the 2sf1 search window is wider than the offset being searched"
        )
    if mode in slip.independent_of:
        pass
    elif slip.method.startswith("rotation_sidebands") and mode == "eccentricity":
        blockers.append(
            "slip came from the rotation sidebands, which are the eccentricity "
            "signature: diagnosing eccentricity from it is circular"
        )

    if assessment.zero_sequence_ratio > ZERO_SEQUENCE_BLOCK:
        blockers.append(
            f"zero-sequence current is {assessment.zero_sequence_ratio * 100:.1f} % of the "
            f"space-vector modulus; on a 3-wire supply it must be ~0, so this is a "
            f"wiring or channel-gain fault, not a machine fault"
        )
    if assessment.vfd:
        blockers.append(
            f"inverter-fed (carrier {assessment.carrier_hz:.0f} Hz): the fundamental is a "
            f"commanded value, the current regulator attenuates part of the fault "
            f"signature, and the switching family overlaps the rotor-slot family"
        )
    if not assessment.three_phase and mode in ("stator_turn", "eccentricity_mixed"):
        blockers.append(f"{mode} needs three phases")

    if mode == "stator_turn":
        if not assessment.voltage_present:
            blockers.append(
                "stator-turn diagnosis needs voltage: negative-sequence current "
                "alone cannot separate a shorted turn from supply unbalance"
            )
        elif assessment.v_unbalance_pct > UNBALANCE_BLOCK_PCT:
            blockers.append(
                f"voltage unbalance {assessment.v_unbalance_pct:.2f} % exceeds "
                f"{UNBALANCE_BLOCK_PCT:.1f} %: the supply produces more "
                f"negative-sequence current than the fault does"
            )
        elif not assessment.z_valid:
            blockers.append(
                "voltage unbalance below 0.3 %: Z- is a ratio of two "
                "instrument-noise numbers and carries no information"
            )
    else:
        if assessment.voltage_present and assessment.v_unbalance_pct > UNBALANCE_BLOCK_PCT:
            warnings.append(
                f"voltage unbalance {assessment.v_unbalance_pct:.2f} % raises the "
                f"spectral floor near the fundamental; sideband levels are degraded"
            )
        elif not assessment.voltage_present:
            warnings.append("supply unbalance not measured (no voltage channel)")

    if not np.isnan(assessment.vfd_evidence_db) and 10.0 < assessment.vfd_evidence_db <= 20.0:
        warnings.append(
            f"marginal inverter evidence ({assessment.vfd_evidence_db:.0f} dB): "
            f"confirm from the asset record rather than from this test"
        )
    if np.isnan(assessment.vfd_evidence_db):
        warnings.append("sample rate too low to detect an inverter carrier")

    return {"ok": not blockers, "blockers": blockers, "warnings": warnings, "mode": mode}
