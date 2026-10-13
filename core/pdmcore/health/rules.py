"""Rule-based diagnosis: the part a millwright can argue with.

A learned classifier says "inner race, 0.82". A rule engine says "BPFI and four
harmonics stand 9x above the local floor in the squared envelope spectrum, with
1x sidebands present; 1x itself is normal, so this is not unbalance". The second
one gets the machine opened. That is the entire argument for keeping an explicit
rule layer alongside the models -- and the rules also provide the labels that
bootstrap the first supervised model, and the sanity check that catches a model
which has learned a dataset artefact.

Rules encode standard vibration-analysis practice (ISO 13373-3 / ISO 18436
diagnostic tables). Each returns a :class:`Diagnosis` with a severity, the
numeric evidence behind it and a recommended action, so the whole chain from
number to work order is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Diagnosis", "RuleContext", "RULES", "diagnose", "register_rule"]


@dataclass
class Diagnosis:
    """One candidate fault with its evidence."""

    fault: str
    confidence: float          # 0-1, heuristic, deliberately not called probability
    severity: str              # "incipient" | "developing" | "advanced" | "severe"
    evidence: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    action: str = ""
    iso_zone: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "fault": self.fault,
            "confidence": round(float(self.confidence), 3),
            "severity": self.severity,
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
            "explanation": self.explanation,
            "action": self.action,
            "iso_zone": self.iso_zone,
        }


@dataclass
class RuleContext:
    """Everything a rule may look at.

    Built from a :class:`pdmcore.features.FeatureFrame` plus optional baseline
    ratios. Rules read ``f(name)`` rather than indexing, so adding a feature
    never silently shifts a rule onto the wrong column.
    """

    values: Dict[str, float]
    shaft_hz: float
    iso_zone: Optional[str] = None
    baseline: Dict[str, float] = field(default_factory=dict)

    def f(self, name: str, default: float = 0.0) -> float:
        return float(self.values.get(name, default))

    def ratio_to_baseline(self, name: str, default: float = 1.0) -> float:
        """Current value divided by its healthy baseline.

        Absolute amplitudes are not comparable across installations, so most
        rules key off this rather than raw values. Returns ``default`` when no
        baseline exists yet, which makes rules degrade to "no opinion" during
        commissioning instead of firing on an un-normalised number.
        """
        base = self.baseline.get(name)
        if base is None or base <= 0:
            return default
        return self.f(name) / base

    @classmethod
    def from_frame(cls, frame, *, baseline: Optional[Dict[str, float]] = None) -> "RuleContext":
        values = frame.as_dict()
        zone_idx = int(values.get("iso.zone_index", 0))
        return cls(
            values=values,
            shaft_hz=float(frame.shaft_hz),
            iso_zone="ABCD"[min(3, max(0, zone_idx))],
            baseline=dict(baseline or {}),
        )


def _severity_from(score: float) -> str:
    if score < 1.5:
        return "incipient"
    if score < 3.0:
        return "developing"
    if score < 6.0:
        return "advanced"
    return "severe"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
RuleFn = Callable[[RuleContext], Optional[Diagnosis]]
RULES: List[RuleFn] = []


def register_rule(fn: RuleFn) -> RuleFn:
    RULES.append(fn)
    return fn


_BEARING_LABELS = {
    "outer_race": ("bearing_outer_race", "BPFO"),
    "inner_race": ("bearing_inner_race", "BPFI"),
    "rolling_element": ("bearing_rolling_element", "BSF"),
    "cage": ("bearing_cage", "FTF"),
}


@register_rule
def rule_bearing(ctx: RuleContext) -> Optional[Diagnosis]:
    """Bearing defect localisation from envelope-spectrum harmonic families.

    Decision logic:

    * ``ratio`` (mean per-harmonic SNR) >= 4 and at least 2 matched harmonics.
      A single matched harmonic is not a bearing fault; it is a coincidence,
      and acting on it is how a programme acquires a reputation for crying wolf.
    * The winning mode is the one with the highest ``ratio``, but inner-race and
      rolling-element claims are *upgraded* when their characteristic sidebands
      are present and *downgraded* when they are not, because the sideband
      pattern -- not the centre frequency -- is what distinguishes them.
    """
    best_mode = None
    best_ratio = 0.0
    for mode in _BEARING_LABELS:
        r = ctx.f(f"bearing.{mode}.ratio")
        if r > best_ratio:
            best_ratio, best_mode = r, mode
    if best_mode is None or best_ratio < 4.0:
        return None
    matched = ctx.f(f"bearing.{best_mode}.n_matched")
    if matched < 2:
        return None
    peak_snr = ctx.f(f"bearing.{best_mode}.peak_snr")
    sb = ctx.f(f"bearing.{best_mode}.sideband_ratio")

    confidence = min(1.0, 0.25 + 0.09 * best_ratio + 0.08 * matched)
    expects_sidebands = best_mode in ("inner_race", "rolling_element")
    if expects_sidebands:
        confidence = min(1.0, confidence + 0.12) if sb >= 0.25 else max(0.2, confidence - 0.18)
    elif sb >= 0.6:
        # Strong modulation on a supposed outer-race defect means the mode call
        # is probably wrong even though a bearing fault is present.
        confidence = max(0.25, confidence - 0.15)

    label, abbrev = _BEARING_LABELS[best_mode]
    env_kurt = ctx.f("bearing.env_kurtosis")
    sev_score = 0.5 * best_ratio / 4.0 + 0.3 * ctx.f("time.kurtosis") / 3.0 + 0.2 * env_kurt / 3.0
    return Diagnosis(
        fault=label,
        confidence=confidence,
        severity=_severity_from(sev_score * 2.0),
        evidence={
            f"{abbrev}_mean_harmonic_snr": best_ratio,
            f"{abbrev}_peak_snr": peak_snr,
            f"{abbrev}_matched_harmonics": matched,
            "sideband_ratio": sb,
            "time_kurtosis": ctx.f("time.kurtosis"),
            "envelope_kurtosis": env_kurt,
        },
        explanation=(
            f"{abbrev} harmonic family stands {best_ratio:.1f}x above the local envelope-spectrum "
            f"floor with {matched:.0f} matched harmonics"
            + (f" and sideband ratio {sb:.2f}" if expects_sidebands else "")
        ),
        action=(
            "Confirm with a second acquisition 24-48 h apart; if the trend is rising, "
            "schedule bearing replacement and inspect lubrication and alignment for root cause."
        ),
        iso_zone=ctx.iso_zone,
    )


@register_rule
def rule_unbalance(ctx: RuleContext) -> Optional[Diagnosis]:
    """Unbalance: 1x dominant, harmonics low, stable.

    The discriminator against misalignment is the 2x/1x ratio: unbalance keeps
    2x well below 1x, misalignment pushes it to 0.5x of 1x or higher.
    """
    one_x = ctx.f("order.1x_g")
    two_x = ctx.f("order.2x_g")
    if one_x <= 0:
        return None
    rest = sum(ctx.f(f"order.{o:g}x_g") for o in (2.0, 3.0, 4.0, 5.0))
    dominance = one_x / max(rest, 1e-12)
    ratio_2x = two_x / one_x
    growth = ctx.ratio_to_baseline("order.1x_g")
    if dominance < 1.5 or ratio_2x > 0.45:
        return None
    if growth < 2.0 and ctx.iso_zone in (None, "A"):
        return None
    confidence = min(1.0, 0.3 + 0.1 * dominance + 0.1 * min(growth, 5.0))
    return Diagnosis(
        fault="unbalance",
        confidence=confidence,
        severity=_severity_from(growth),
        evidence={
            "1x_g": one_x,
            "2x_over_1x": ratio_2x,
            "1x_dominance": dominance,
            "1x_growth_vs_baseline": growth,
            "velocity_rms_mm_s": ctx.f("iso.velocity_rms_mm_s"),
        },
        explanation=(
            f"1x dominates the shaft orders ({dominance:.1f}x the sum of 2x-5x) with "
            f"2x/1x = {ratio_2x:.2f} and 1x {growth:.1f}x the healthy baseline"
        ),
        action=(
            "Field-balance the rotor. Check for product build-up on impeller or fan blades, "
            "and for a missing or thrown balance weight."
        ),
        iso_zone=ctx.iso_zone,
    )


@register_rule
def rule_misalignment(ctx: RuleContext) -> Optional[Diagnosis]:
    """Misalignment / bent shaft: strong 2x, often 3x, axial energy."""
    one_x = ctx.f("order.1x_g")
    two_x = ctx.f("order.2x_g")
    three_x = ctx.f("order.3x_g")
    if one_x <= 0 or two_x <= 0:
        return None
    ratio_2x = two_x / one_x
    ratio_3x = three_x / one_x
    if ratio_2x < 0.45:
        return None
    growth = ctx.ratio_to_baseline("order.2x_g")
    # A machine whose 2x/1x ratio has *always* been 0.5 is not newly misaligned;
    # it has a stiff coupling or an inherently 2x-heavy structure. Requiring
    # growth over the asset's own baseline (or an already-elevated ISO zone) is
    # what stops this rule from firing on every healthy acquisition of such a
    # machine -- the single biggest source of nuisance alarms in a ratio-based
    # rule set.
    if growth < 1.8 and ctx.iso_zone in (None, "A", "B"):
        return None
    confidence = min(1.0, 0.3 + 0.5 * min(ratio_2x, 1.5) + 0.2 * min(ratio_3x, 1.0))
    return Diagnosis(
        fault="misalignment",
        confidence=confidence,
        severity=_severity_from(max(growth, 1.5 * ratio_2x)),
        evidence={
            "2x_over_1x": ratio_2x,
            "3x_over_1x": ratio_3x,
            "2x_growth_vs_baseline": growth,
            "velocity_rms_mm_s": ctx.f("iso.velocity_rms_mm_s"),
        },
        explanation=(
            f"2x/1x = {ratio_2x:.2f} (3x/1x = {ratio_3x:.2f}), the signature of coupling "
            "misalignment or a bent shaft rather than unbalance"
        ),
        action=(
            "Laser-align the coupling cold and re-check hot. Inspect coupling elements and "
            "check for soft foot and pipe strain before re-alignment."
        ),
        iso_zone=ctx.iso_zone,
    )


@register_rule
def rule_looseness(ctx: RuleContext) -> Optional[Diagnosis]:
    """Mechanical looseness: subharmonics plus a raised broadband floor."""
    sub = ctx.f("order.subharmonic_ratio")
    if sub < 0.2:
        return None
    floor_growth = ctx.ratio_to_baseline("band.08_rms")
    confidence = min(1.0, 0.25 + 1.2 * min(sub, 0.6) + 0.1 * min(floor_growth, 4.0))
    return Diagnosis(
        fault="mechanical_looseness",
        confidence=confidence,
        severity=_severity_from(3.0 * sub + 0.5 * floor_growth),
        evidence={
            "subharmonic_ratio_0p5x_over_1x": sub,
            "mid_band_growth": floor_growth,
            "time_crest": ctx.f("time.crest"),
        },
        explanation=(
            f"0.5x subharmonic at {sub:.2f} of 1x with mid-band energy {floor_growth:.1f}x "
            "baseline -- structural looseness or a worn fit"
        ),
        action="Torque-check hold-down bolts and baseplate grouting; inspect bearing housing fit.",
        iso_zone=ctx.iso_zone,
    )


@register_rule
def rule_lubrication(ctx: RuleContext) -> Optional[Diagnosis]:
    """Lubrication distress: high-frequency broadband rise with no line structure.

    This is the diagnosis that pays for the whole acoustic/high-frequency
    channel. Grease starvation raises the 5-20 kHz floor days to weeks before
    any kinematic frequency appears, and re-greasing at that point is a 20-minute
    job instead of a bearing change.
    """
    hi_growth = max(
        ctx.ratio_to_baseline("band.10_rms"),
        ctx.ratio_to_baseline("band.11_rms"),
    )
    centroid_shift = ctx.ratio_to_baseline("shape.centroid_hz")
    bearing_lines = max(ctx.f(f"bearing.{m}.ratio") for m in _BEARING_LABELS)
    if hi_growth < 2.5 or bearing_lines >= 4.0:
        return None
    confidence = min(0.85, 0.2 + 0.15 * hi_growth + 0.2 * max(0.0, centroid_shift - 1.0))
    return Diagnosis(
        fault="lubrication_distress",
        confidence=confidence,
        severity=_severity_from(hi_growth),
        evidence={
            "hf_band_growth": hi_growth,
            "spectral_centroid_shift": centroid_shift,
            "max_bearing_line_snr": bearing_lines,
            "envelope_kurtosis": ctx.f("bearing.env_kurtosis"),
        },
        explanation=(
            f"High-frequency bands up {hi_growth:.1f}x with the spectral centroid shifted "
            f"{centroid_shift:.2f}x and no kinematic bearing lines -- consistent with "
            "boundary lubrication or grease starvation, not yet a spall"
        ),
        action=(
            "Re-lubricate to the specified grease quantity and re-measure after 2-4 h of running. "
            "If the high-frequency level does not drop, treat as early bearing damage."
        ),
        iso_zone=ctx.iso_zone,
    )


@register_rule
def rule_cavitation(ctx: RuleContext) -> Optional[Diagnosis]:
    """Pump cavitation: broadband, high spectral entropy, non-stationary."""
    entropy = ctx.f("shape.entropy")
    hi_growth = max(ctx.ratio_to_baseline("band.09_rms"), ctx.ratio_to_baseline("band.10_rms"))
    flatness = ctx.f("shape.flatness")
    if entropy < 0.55 or hi_growth < 2.0:
        return None
    confidence = min(0.8, 0.15 + 0.6 * entropy + 0.1 * min(hi_growth, 4.0))
    return Diagnosis(
        fault="cavitation",
        confidence=confidence,
        severity=_severity_from(hi_growth),
        evidence={
            "spectral_entropy": entropy,
            "spectral_flatness": flatness,
            "hf_band_growth": hi_growth,
        },
        explanation=(
            f"Broadband energy with spectral entropy {entropy:.2f} and flatness {flatness:.2f}, "
            "no discrete lines -- classic cavitation noise"
        ),
        action=(
            "Check suction conditions: NPSH margin, strainer blockage, suction valve position "
            "and liquid temperature. Do not treat as a bearing fault."
        ),
        iso_zone=ctx.iso_zone,
    )


@register_rule
def rule_iso_severity(ctx: RuleContext) -> Optional[Diagnosis]:
    """Fallback: ISO zone C/D with no specific signature identified.

    Reported as ``elevated_vibration_unclassified`` rather than being forced
    into the nearest fault class. "The level is high and we do not yet know why"
    is a legitimate and useful output; a confidently wrong fault label is not.
    """
    if ctx.iso_zone not in ("C", "D"):
        return None
    return Diagnosis(
        fault="elevated_vibration_unclassified",
        confidence=0.5 if ctx.iso_zone == "C" else 0.7,
        severity="advanced" if ctx.iso_zone == "C" else "severe",
        evidence={
            "velocity_rms_mm_s": ctx.f("iso.velocity_rms_mm_s"),
            "time_rms_g": ctx.f("time.rms"),
            "time_kurtosis": ctx.f("time.kurtosis"),
        },
        explanation=(
            f"Broadband velocity in ISO zone {ctx.iso_zone} "
            f"({ctx.f('iso.velocity_rms_mm_s'):.2f} mm/s) with no matched fault signature"
        ),
        action=(
            "Collect a full-resolution waveform and route to an ISO 18436 category II analyst. "
            "Check sensor mounting before assuming a machine fault."
        ),
        iso_zone=ctx.iso_zone,
    )


def diagnose(
    frame,
    *,
    baseline: Optional[Dict[str, float]] = None,
    min_confidence: float = 0.3,
    max_results: int = 4,
) -> List[Diagnosis]:
    """Run every rule and return the surviving diagnoses, best first.

    Multiple simultaneous diagnoses are expected and are not deduplicated:
    misalignment destroying a bearing is the single most common real pairing,
    and reporting only the strongest signature loses the root cause, which is
    the one that actually needs fixing.
    """
    ctx = RuleContext.from_frame(frame, baseline=baseline)
    out: List[Diagnosis] = []
    for rule in RULES:
        try:
            d = rule(ctx)
        except Exception as exc:  # a broken rule must not take down the pipeline
            out.append(
                Diagnosis(
                    fault="rule_error",
                    confidence=0.0,
                    severity="incipient",
                    evidence={},
                    explanation=f"{rule.__name__} raised {type(exc).__name__}: {exc}",
                    action="File a bug; this is a platform defect, not a machine fault.",
                )
            )
            continue
        if d is not None and d.confidence >= min_confidence:
            out.append(d)
    out.sort(key=lambda d: -d.confidence)
    # If a specific fault was identified, drop the unclassified fallback.
    specific = [d for d in out if d.fault != "elevated_vibration_unclassified"]
    if specific:
        out = specific + [d for d in out if d.fault == "elevated_vibration_unclassified"][:0]
    return out[:max_results]


def baseline_from_vectors(vectors: np.ndarray, names: Sequence[str]) -> Dict[str, float]:
    """Median per feature over a healthy window -> baseline dict for rules."""
    X = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    med = np.median(X, axis=0)
    return {n: float(v) for n, v in zip(names, med)}
