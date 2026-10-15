"""Rates that carry the count they were computed from.

Why this module exists
----------------------
``docs/08_branch_catalogue.md`` tabulates six "checks that cannot fail" found in
this repository. One of them is a general statistical trap rather than a local
bug, and it is restated there as a lesson:

    A gate expressed as a ratio of rates is really a gate on a count, and at
    small counts the ratio is the wrong parameterisation.

The decision layer is where that trap is most dangerous, because **maintenance
is made of small counts**. A plant with 40 monitored assets and one failure per
asset per five years sees eight failures a year. "We prevented 60 % of
failures" is then a statement about five events, and five events have a 95 %
Poisson interval of 1.6-11.7 -- so "60 %" and "20 %" are not distinguishable
from one year of data. A programme-payback figure quoted as a percentage of
avoided failures, with no count attached, is not a measurement; it is a
rounding of noise.

The structural fix is to make it impossible to report a rate without its
count. Everything in this repository's decision layer that divides by an
exposure returns a :class:`CountedRate`, which refuses to exist without the
numerator count and always serialises its exact Poisson interval alongside the
point estimate.

What is here
------------
:class:`CountedRate`      a rate that cannot be separated from its count.
:func:`poisson_rate_ci`   exact (Garwood) interval on a rate from a count.
:func:`poisson_count_band` the acceptance band on a *count* for a target rate.
:func:`counts_distinguishable`
                          whether two observed counts could be the same rate.
:func:`require_counts`    a gate that refuses to express a ratio that rests on
                          too few events, and says how many it needed.

The Garwood interval is used rather than a normal approximation because the
normal approximation is worst exactly where this matters: at a count of 3 the
Wald interval includes negative rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = [
    "CountedRate",
    "SmallCountError",
    "poisson_rate_ci",
    "poisson_count_band",
    "counts_distinguishable",
    "require_counts",
]


class SmallCountError(ValueError):
    """Raised when a ratio is requested that the event count cannot support."""


# ---------------------------------------------------------------------------
# chi-square quantiles without scipy
# ---------------------------------------------------------------------------
# scipy is an optional dependency of ``core`` (see core/pyproject.toml), and the
# decision layer has to run on a gateway where only numpy is installed. The
# Garwood bounds need chi-square quantiles with even degrees of freedom only,
# which is the regularised lower incomplete gamma at integer/half-integer shape
# -- available in the standard library via math.lgamma plus a bisection, with no
# accuracy concern at the precision a maintenance report needs.
def _gammainc_lower_regularised(s: float, x: float) -> float:
    """Regularised lower incomplete gamma P(s, x), series/continued fraction."""
    if x < 0.0 or s <= 0.0:
        raise ValueError("gammainc requires s > 0 and x >= 0")
    if x == 0.0:
        return 0.0
    if x < s + 1.0:
        # Series expansion.
        term = 1.0 / s
        total = term
        n = s
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    # Continued fraction for the upper tail Q(s, x), then P = 1 - Q.
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
    return 1.0 - q


def _chi2_ppf(p: float, df: float) -> float:
    """Chi-square quantile by bisection on the regularised incomplete gamma."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    s = df / 2.0
    lo, hi = 0.0, max(10.0, df * 4.0 + 20.0)
    while _gammainc_lower_regularised(s, hi / 2.0) < p:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - unreachable for df we use
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc_lower_regularised(s, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------
def poisson_rate_ci(count: int, exposure: float, *, conf: float = 0.95) -> Tuple[float, float]:
    """Exact (Garwood) confidence interval on a Poisson rate.

    ``count`` events observed over ``exposure`` units (asset-years, months,
    whatever the denominator is). Returns ``(lo, hi)`` in events per unit.

    The lower bound at ``count == 0`` is exactly 0, which is correct and is the
    reason a zero-failure year does not prove a zero failure rate: the upper
    bound is still ``3.0 / exposure`` at 95 %.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    if exposure <= 0:
        raise ValueError("exposure must be > 0")
    alpha = 1.0 - conf
    lo = 0.0 if count == 0 else 0.5 * _chi2_ppf(alpha / 2.0, 2.0 * count)
    hi = 0.5 * _chi2_ppf(1.0 - alpha / 2.0, 2.0 * (count + 1))
    return lo / exposure, hi / exposure


def poisson_count_band(lam: float, *, k_sigma: float = 3.0, pad: float = 3.0) -> Tuple[float, float]:
    """Acceptance band on a *count* whose expectation is ``lam``.

    ``lam +- k_sigma*sqrt(lam)`` with an additive ``pad``, which is the fix the
    Jetson branch applied to its own false-alarm gate after the ratio version
    rejected stationary data about a third of the time. The pad matters at small
    ``lam``: at ``lam = 1.8`` the 3-sigma Gaussian half-width is 4.0 but the
    Poisson distribution still puts 1.5 % of its mass at 6 or above, so a gate
    without the pad is a gate that fires on noise.

    Returns ``(lo, hi)`` as real numbers; compare an integer count against them.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    half = k_sigma * math.sqrt(lam) + pad
    return max(0.0, lam - half), lam + half


def counts_distinguishable(
    count_a: int, exposure_a: float, count_b: int, exposure_b: float, *, conf: float = 0.95
) -> bool:
    """Could these two observed counts have come from the same rate?

    ``True`` means the Poisson intervals do not overlap, i.e. the difference is
    larger than counting noise. Deliberately conservative (non-overlapping
    intervals is a stricter test than a formal two-sample test) because the
    consequence of a false "we improved" claim in a maintenance report is that
    the programme is credited with a saving it did not make.
    """
    lo_a, hi_a = poisson_rate_ci(count_a, exposure_a, conf=conf)
    lo_b, hi_b = poisson_rate_ci(count_b, exposure_b, conf=conf)
    return hi_a < lo_b or hi_b < lo_a


def require_counts(count: int, *, minimum: int, what: str) -> None:
    """Refuse to express a ratio that rests on fewer than ``minimum`` events.

    The relative standard error of a Poisson count is ``1/sqrt(n)``: 58 % at
    n = 3, 32 % at n = 10, 10 % at n = 100. ``minimum`` should be chosen from
    the precision the claim needs, and the error message says what precision the
    available count actually supports so the caller can quote that instead of
    refusing outright.
    """
    if count < minimum:
        rse = float("inf") if count == 0 else 100.0 / math.sqrt(count)
        raise SmallCountError(
            f"{what}: {count} event(s) cannot support a ratio -- {minimum} needed. "
            f"Relative standard error at n={count} is "
            + ("undefined" if count == 0 else f"{rse:.0f} %")
            + f"; quote the count and its Poisson interval instead."
        )


@dataclass(frozen=True)
class CountedRate:
    """A rate that cannot be reported without the count behind it.

    ``count`` events over ``exposure`` units of ``unit``. ``as_dict`` always
    emits the count, the exposure and the exact Poisson interval, so no figure
    and no document generated from this repository's decision results can quote
    a rate whose evidence base is invisible.
    """

    count: int
    exposure: float
    unit: str
    conf: float = 0.95
    label: str = ""

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be >= 0")
        if self.exposure <= 0:
            raise ValueError(f"exposure must be > 0 (got {self.exposure})")

    @property
    def rate(self) -> float:
        return self.count / self.exposure

    @property
    def ci(self) -> Tuple[float, float]:
        return poisson_rate_ci(self.count, self.exposure, conf=self.conf)

    @property
    def relative_width(self) -> float:
        """Interval width as a fraction of the point estimate.

        ``inf`` at a count of zero. Above ~0.5 the rate should be read as an
        order of magnitude, not a number -- which at this repository's failure
        rates is the normal case, not an edge case.
        """
        lo, hi = self.ci
        return float("inf") if self.count == 0 else (hi - lo) / self.rate

    def scaled(self, value_per_event: float) -> Tuple[float, float, float]:
        """``(point, lo, hi)`` of ``rate * value_per_event``.

        Used to turn "failures per asset-year" into "euros per asset-year"
        while carrying the counting uncertainty through, instead of quoting a
        euro figure whose error bar was dropped at the multiplication.
        """
        lo, hi = self.ci
        return self.rate * value_per_event, lo * value_per_event, hi * value_per_event

    def as_dict(self) -> Dict[str, object]:
        lo, hi = self.ci
        return {
            "label": self.label,
            "count": int(self.count),
            "exposure": round(float(self.exposure), 4),
            "unit": self.unit,
            "rate": round(float(self.rate), 6),
            "ci_lo": round(float(lo), 6),
            "ci_hi": round(float(hi), 6),
            "conf": self.conf,
            "relative_width": (
                None if math.isinf(self.relative_width) else round(self.relative_width, 3)
            ),
        }

    def __str__(self) -> str:
        lo, hi = self.ci
        return (
            f"{self.rate:.4g}/{self.unit} [{lo:.4g}, {hi:.4g}] "
            f"(n={self.count} over {self.exposure:g} {self.unit})"
        )


def rate_from_counts(
    count: int, exposure: float, unit: str, *, label: str = "", conf: float = 0.95
) -> CountedRate:
    """Convenience constructor, named so call sites read as an assertion."""
    return CountedRate(count=int(count), exposure=float(exposure), unit=unit,
                       conf=conf, label=label)


def summarise(rates: Dict[str, Optional[CountedRate]]) -> Dict[str, object]:
    """Serialise a mapping of named rates, keeping ``None`` entries visible."""
    return {k: (None if v is None else v.as_dict()) for k, v in rates.items()}
