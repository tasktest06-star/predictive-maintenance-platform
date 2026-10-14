"""Hailo-8L inference backend, and an honest account of what it is for.

THE SHORT VERSION: on the workload this platform actually runs, the AI HAT has
nothing to do. The gateway benchmark measures Mahalanobis scoring of a 63-float
feature vector at **5.1 microseconds per frame** on one aarch64 core -- about
196 000 frames per second. A Hailo-8L at 13 TOPS cannot improve on that in any
way that matters, because the work is a 63x63 quadratic form, not a tensor
graph, and the PCIe round trip alone would cost more than the computation.

So this file exists for one reason, and it is the same reason the Alif branch's
NPU exists: **a model class that runs nowhere else.**

    63-float detector, numpy on the CPU         5.1 us/frame   <- plenty
    1D CNN over 8192 raw samples, ~4 MMAC       ~8 ms on CPU
    the same CNN on a Hailo-8L                  ~0.3 ms
    ...for 200 channels at once                 still ~0.3 ms

A raw-waveform model skips the feature extractor entirely, which is the point:
every feature in this repository is a decision about what matters, taken by a
human in advance. A model over raw samples can find a signature nobody
anticipated. That is also exactly why it is risky -- it can find the test rig's
mains hum and call it a bearing fault -- and why none of this is enabled by
default.

STATUS: NOT IMPLEMENTED. Deliberately.

`HailoScorer` raises on construction with a message saying precisely what is
missing. A stub returning plausible probabilities would make this kit look
finished and would be the single most misleading artefact in the repository: a
gateway shipping confident scores from a model that does not exist.

What it would take, in order:

1. A 1D CNN trained on MEASURED data (CWRU/MFPT/MIMII at minimum, plant data
   properly). No model here has passed that gate -- see
   docs/07_production_readiness.md.
2. ONNX export, then `hailo_model_zoo` compilation to a .hef with int8
   quantisation calibrated on real waveforms, not on simulator output.
3. A bit-exactness check between the .hef and the ONNX reference, of the kind
   cdsp/tools/gen_golden.py does for the C path. Without it, quantisation error
   is invisible until it changes a decision.
4. Decision-flip accounting against the float model, with the same thresholds
   tools/export_model.py already enforces for the int8 MCU path: refuse the
   export above 5 % flips or on any non-borderline flip to normal.

Until (1) exists, (2)-(4) are premature, and so is this file's implementation.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("pdmedge.hailo")


class HailoUnavailable(RuntimeError):
    """The Hailo path was requested and cannot be provided."""


class CpuWaveformScorer:
    """A deliberately slow reference path for the raw-waveform model.

    Not a stub: it computes nothing and says so. It exists to hold the
    *interface* -- `score(X) -> np.ndarray` over raw blocks rather than feature
    vectors -- so that the gateway, the config and the tests can be written
    against the shape of the thing before the model exists, without any code
    path that produces a number.
    """

    def __init__(self, input_samples: int = 8192) -> None:
        self.input_samples = int(input_samples)

    def score(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        raise NotImplementedError(
            "no raw-waveform model exists in this repository. The feature-based "
            "detector in pdmcore.models.anomaly is what the gateway scores with; "
            "see hardware/rpi5_hailo/gateway/hailo_backend.py for what training "
            "a waveform model would require."
        )


class HailoScorer:
    """Hailo-8L backend. Raises: there is no compiled model to load.

    Constructed by the gateway only when `sink.scorer == "hailo"` in the
    config, so the default path never touches it.
    """

    def __init__(self, hef_path: str, input_samples: int = 8192,
                 device_id: Optional[str] = None) -> None:
        self.hef_path = hef_path
        self.input_samples = int(input_samples)
        self.device_id = device_id

        try:
            import hailo_platform  # type: ignore  # noqa: F401
            have_runtime = True
        except ImportError:
            have_runtime = False

        raise HailoUnavailable(
            "the Hailo path is not implemented.\n"
            f"  HailoRT present: {have_runtime}\n"
            f"  requested .hef:  {hef_path}\n"
            "\n"
            "There is no compiled model to load, because no raw-waveform model "
            "in this repository has been trained on measured data. The gateway "
            "scores with the feature-based detector instead, which the benchmark "
            "measures at 5.1 us/frame on one CPU core -- fast enough for more "
            "nodes than any plant has.\n"
            "\n"
            "A stub returning plausible probabilities was considered and "
            "rejected: it would make this kit look finished while shipping "
            "confident scores from a model that does not exist."
        )


def build_scorer(kind: str = "cpu_features", **kwargs: object) -> object:
    """Resolve a scorer by name, failing loudly for the ones that are absent.

    `cpu_features` is the only kind that works, and it is the default, so a
    gateway deployed without reading this file gets the path that functions.
    """
    if kind == "cpu_features":
        from pdmcore.models.anomaly import MahalanobisDetector
        return MahalanobisDetector()
    if kind == "hailo":
        return HailoScorer(str(kwargs.get("hef_path", "")),
                           int(kwargs.get("input_samples", 8192)))  # type: ignore[arg-type]
    if kind == "cpu_waveform":
        return CpuWaveformScorer(int(kwargs.get("input_samples", 8192)))  # type: ignore[arg-type]
    raise ValueError(
        f"unknown scorer kind {kind!r}; choose cpu_features (works), "
        f"hailo (not implemented) or cpu_waveform (interface only)"
    )
