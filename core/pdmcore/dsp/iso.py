"""ISO 20816 / ISO 10816 vibration severity zones.

Zone semantics (ISO 20816-1 clause 7):

====  ====================================================================
A     newly commissioned machine
B     acceptable for unrestricted long-term operation
C     unsatisfactory for long-term operation; operate only until a
      suitable opportunity for remedial action
D     sufficiently severe to cause damage; stop the machine
====  ====================================================================

The zone boundaries below are the broadband velocity RMS limits of
ISO 10816-3 / ISO 20816-3 for in-situ measurement on non-rotating parts, in the
10-1000 Hz band (2-1000 Hz for machines under 600 rpm).

What this is *not*
------------------
A severity zone is **not** a diagnosis and **not** a trend. A fan can sit in
zone B for a decade and then fail in three weeks with the broadband level
barely moving, because the bearing energy lives at 4 kHz where the velocity
measurement is deliberately band-limited away. So the platform reports the ISO
zone because customers and insurers ask for it, and bases alarms on the trended
health index in :mod:`pdmcore.health.index`. Treating ISO zones as the alarm
engine is the most common way a condition-monitoring deployment fails to catch
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = ["IsoClass", "ISO_20816_3_LIMITS", "iso_zone", "iso_zone_margin", "classify_machine"]


@dataclass(frozen=True)
class IsoClass:
    """An ISO 20816-3 machine group / support condition pairing."""

    key: str
    description: str
    a_b_mm_s: float
    b_c_mm_s: float
    c_d_mm_s: float


#: ISO 20816-3 (formerly ISO 10816-3) zone boundaries, velocity RMS in mm/s.
#:
#: Group 1 = large machines, 300 kW-50 MW, shaft height >= 315 mm.
#: Group 2 = medium machines, 15-300 kW, shaft height 160-315 mm.
ISO_20816_3_LIMITS: Dict[str, IsoClass] = {
    "group1_rigid": IsoClass("group1_rigid", "Group 1 (>300 kW), rigid support", 2.3, 4.5, 7.1),
    "group1_flexible": IsoClass("group1_flexible", "Group 1 (>300 kW), flexible support", 3.5, 7.1, 11.0),
    "group2_rigid": IsoClass("group2_rigid", "Group 2 (15-300 kW), rigid support", 1.4, 2.8, 4.5),
    "group2_flexible": IsoClass("group2_flexible", "Group 2 (15-300 kW), flexible support", 2.3, 4.5, 7.1),
}


def iso_zone(velocity_rms_mm_s: float, machine_class: str = "group2_rigid") -> str:
    """Return ``"A"``, ``"B"``, ``"C"`` or ``"D"``."""
    if velocity_rms_mm_s < 0:
        raise ValueError("velocity RMS must be non-negative")
    try:
        limits = ISO_20816_3_LIMITS[machine_class]
    except KeyError as exc:
        raise ValueError(
            f"unknown machine class {machine_class!r}; expected one of {sorted(ISO_20816_3_LIMITS)}"
        ) from exc
    if velocity_rms_mm_s <= limits.a_b_mm_s:
        return "A"
    if velocity_rms_mm_s <= limits.b_c_mm_s:
        return "B"
    if velocity_rms_mm_s <= limits.c_d_mm_s:
        return "C"
    return "D"


def iso_zone_margin(velocity_rms_mm_s: float, machine_class: str = "group2_rigid") -> Tuple[str, float]:
    """``(zone, headroom_mm_s)`` -- distance to the next worse boundary.

    Headroom is more actionable than the letter on its own: "zone B with
    0.1 mm/s of headroom" is a work order, "zone B with 1.6 mm/s" is a note.
    Returns ``inf`` headroom in zone D (no worse zone exists).
    """
    limits = ISO_20816_3_LIMITS[machine_class]
    zone = iso_zone(velocity_rms_mm_s, machine_class)
    next_edge = {"A": limits.a_b_mm_s, "B": limits.b_c_mm_s, "C": limits.c_d_mm_s}.get(zone)
    if next_edge is None:
        return zone, float("inf")
    return zone, float(next_edge - velocity_rms_mm_s)


def classify_machine(power_kw: float, *, flexible_support: bool) -> str:
    """Pick the ISO class key from nameplate power and mounting.

    "Rigid" means the first support natural frequency is above the running
    speed; in practice, machines bolted to a massive foundation or a thick steel
    baseplate. Skid- or spring-mounted machines, and anything on a mezzanine
    floor, are flexible. Guessing wrong by one class shifts every threshold by
    roughly 1.6x, so this is recorded as asset metadata and reviewed at
    commissioning rather than inferred at runtime.
    """
    if power_kw <= 0:
        raise ValueError("power_kw must be positive")
    group = "group1" if power_kw > 300.0 else "group2"
    support = "flexible" if flexible_support else "rigid"
    return f"{group}_{support}"
