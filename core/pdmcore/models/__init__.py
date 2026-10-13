"""Detectors, classifiers, quantisation and edge export.

Deployment order, which is also the order of these modules:

1. :mod:`pdmcore.models.anomaly` -- unsupervised, works on day one with only
   healthy data. This is what ships to a new site.
2. :mod:`pdmcore.models.classifier` -- supervised fault classification, enabled
   once confirmed failures exist (or transferred from public data and then
   validated against the site's own).
3. :mod:`pdmcore.models.quantize` / :mod:`pdmcore.models.export` -- int8
   post-training quantisation and C-array emission for the MCU targets.
4. :mod:`pdmcore.models.registry` -- model cards, checksums and the
   feature-config binding that stops a model being fed the wrong vector.
"""

from __future__ import annotations

from .anomaly import (
    AnomalyDetector,
    DeepSVDDDetector,
    MahalanobisDetector,
    PCAResidualDetector,
    RobustThresholdDetector,
    StandardScaler,
    build_detector,
)

from .export import QuantisedModel, emit_c_header, export_deep_svdd, quantise_deep_svdd

__all__ = [
    "StandardScaler", "AnomalyDetector", "MahalanobisDetector", "PCAResidualDetector",
    "RobustThresholdDetector", "DeepSVDDDetector", "build_detector",
    "QuantisedModel", "quantise_deep_svdd", "export_deep_svdd", "emit_c_header",
]
