"""pdmcore -- production core for a wireless condition-monitoring platform.

Layer map
---------
``pdmcore.dsp``     Deterministic signal processing. The hot paths have a
                    portable C twin in ``cdsp/`` (on the ``hw/*`` branches) so
                    that the number a microcontroller computes equals the number
                    the cloud computes; generated golden vectors pin that down.
``pdmcore.sim``     Physics-flavoured machine simulator. Lets the whole stack be
                    exercised -- and regression-tested -- with no hardware
                    attached, which is how the per-kit POCs run in CI.
``pdmcore.models``  Feature-based and learned detectors, int8 quantisation and
                    C-array export for edge targets.
``pdmcore.eval``    Leakage-resistant splits, operating-point metrics and
                    calibration. Industrial PdM is a low-prevalence problem, so
                    headline accuracy is close to meaningless here.
``pdmcore.health``  Health index fusion, alarm logic with hysteresis, rule-based
                    diagnosis and particle-filter RUL.
``pdmcore.io``      Wire payload schema (shared with firmware), MQTT transport
                    with store-and-forward, archival writers.

Design rules that the rest of the platform depends on
-----------------------------------------------------
1. ``dsp`` is pure: float64 in, float64 out, no global state, no I/O.
2. Nothing in the import path of ``pdmcore`` requires scikit-learn or torch.
3. Units are explicit in names and in :mod:`pdmcore.units`: ``_g``, ``_mm_s``,
   ``_hz``, ``_degc``. Mixed-unit bugs are the most common field defect in
   condition monitoring, so the type names carry the unit.
"""

from __future__ import annotations

__version__ = "0.3.0"

# Wire-format version negotiated between firmware and ingest. Bump on any
# breaking change to pdmcore.io.payload.
PAYLOAD_SCHEMA_VERSION = 3

# Feature-vector contract version. Edge firmware stamps this into every frame so
# the cloud can refuse to score a frame produced by an incompatible extractor.
FEATURE_SPEC_VERSION = 2

__all__ = ["PAYLOAD_SCHEMA_VERSION", "FEATURE_SPEC_VERSION", "__version__"]
