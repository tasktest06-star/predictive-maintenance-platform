"""Motor current signature analysis -- the electrical view of a mechanical machine.

Why this is a subpackage and not a fork of ``pdmcore.dsp``
---------------------------------------------------------
A current spectrum is still a calibrated single-sided spectrum, so everything in
:mod:`pdmcore.dsp.spectrum` applies unchanged and is used unchanged --
power-domain averaging, coherent gain, ENBW division in ``rms_in_band``, the
single-sided fold. :mod:`pdmcore.dsp.bearing` already scores a harmonic family
and a sideband pair against a local floor, and both are reused verbatim.

Three things genuinely differ, and they are what lives here:

1. **Dynamic range, not noise.** A vibration measurement is limited by the
   sensor's noise floor. A current measurement is limited by *spectral leakage
   from the supply fundamental*, which is 40-60 dB larger than the fault
   sideband and only 0.3-3 Hz away from it. A Hann window -- correct everywhere
   else in this repository -- leaks more than the signal being measured at that
   offset. :mod:`pdmcore.mcsa.features` therefore computes the leakage budget
   explicitly and refuses a configuration that cannot support the claim.

2. **The frequency axis depends on the operating point.** Bearing frequencies
   scale with shaft speed, which is one unknown. Current fault frequencies
   depend on *slip*, which is the small difference between two speeds and is
   therefore known to far worse relative precision -- a 0.5 % speed error is a
   45 % slip error at rated slip, which moves every predicted sideband by more
   than a resolution bandwidth. :mod:`pdmcore.mcsa.frequencies` derives the
   families and the position error they inherit.

3. **Three phases, not three axes.** Clarke/Park transforms, the extended
   Park's vector approach, the instantaneous-power spectrum and the
   negative-sequence impedance all need the phases *together*, which no
   vibration channel ever does. :mod:`pdmcore.mcsa.park`.

:mod:`pdmcore.mcsa.supply` holds the gates: a measured supply unbalance, a slip
estimate with a stated error bound, and VFD detection. They exist because the
confounders are not small -- a 2 % supply voltage unbalance produces a larger
negative-sequence current than a shorted turn does, and a load oscillation at
the right frequency is indistinguishable from a broken bar.
"""

from __future__ import annotations

from .frequencies import (
    MCSAFrequencies,
    bearing_current_sidebands,
    eccentricity_sidebands,
    min_record_duration_s,
    mcsa_frequencies,
    rotor_bar_sidebands,
    rotor_slot_harmonics,
    sideband_position_error_hz,
    slip_from_shaft_hz,
    shaft_hz_from_slip,
    sync_speed_hz,
)
from .park import (
    ParkVector,
    clarke_transform,
    epva_spectrum,
    instantaneous_power,
    park_vector,
    sequence_components,
)
from .features import (
    CurrentFeatureConfig,
    CurrentFeatureExtractor,
    CurrentFrame,
    leakage_budget_db,
    window_leakage_db,
)
from .supply import (
    SupplyAssessment,
    SlipEstimate,
    assess_supply,
    estimate_slip_from_sidebands,
    mcsa_admissible,
)

__all__ = [
    "MCSAFrequencies",
    "mcsa_frequencies",
    "rotor_bar_sidebands",
    "eccentricity_sidebands",
    "rotor_slot_harmonics",
    "bearing_current_sidebands",
    "sideband_position_error_hz",
    "min_record_duration_s",
    "slip_from_shaft_hz",
    "shaft_hz_from_slip",
    "sync_speed_hz",
    "ParkVector",
    "clarke_transform",
    "park_vector",
    "epva_spectrum",
    "instantaneous_power",
    "sequence_components",
    "CurrentFeatureConfig",
    "CurrentFeatureExtractor",
    "CurrentFrame",
    "window_leakage_db",
    "leakage_budget_db",
    "SupplyAssessment",
    "SlipEstimate",
    "assess_supply",
    "estimate_slip_from_sidebands",
    "mcsa_admissible",
]

#: Current-domain feature contract version. Separate from
#: :data:`pdmcore.FEATURE_SPEC_VERSION` on purpose: a current frame is a
#: different quantity with a different length, and the cloud must not be able to
#: score one with a vibration model. The payload carries the current spec
#: version in its own field (see ``mcsa/production/README.md``).
CURRENT_FEATURE_SPEC_VERSION = 1
