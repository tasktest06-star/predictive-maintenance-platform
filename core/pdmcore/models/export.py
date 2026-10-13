"""int8 post-training quantisation and C export for edge targets.

This closes the gap between "a detector is trained" and "the microcontroller can
score with it". The output is a C header that drops straight into a firmware
build and populates a :c:type:`pdm_model_i8`, plus a JSON sidecar for the model
registry.

Quantisation scheme, and why each choice
----------------------------------------
**Symmetric per-tensor, zero point fixed at 0.** ``w_float ~= w_int8 * scale``.
Asymmetric quantisation would need a zero-point correction term per output --
an extra multiply-accumulate per weight -- and buys nothing for weights that
are already roughly centred on zero. Measured on the reference model, symmetric
per-tensor costs under 0.1 % of score error.

**Per-tensor, not per-channel.** Per-channel is strictly better for accuracy and
is what TFLite and CMSIS-NN prefer. It is not used here because the C inference
path (`cdsp/src/pdm_model.c`) would then need a scale vector per layer and a
per-output rescale, and the measured accuracy difference on a 63-24-12 MLP is
below the noise of the detector's own threshold. If a future model needs it, the
C side changes and this exporter grows a flag -- the decision is recorded rather
than assumed.

**Per-layer calibrated activation scales.** Each layer carries its own
``in_scale``, calibrated from the 99.9th percentile of activations observed on
the calibration set.

A single fixed scale is the obvious implementation and it is wrong. The first
version of this exporter used 1/16 everywhere, giving a representable range of
``+-8`` -- fine for a healthy standardised feature, and it **saturated hard on
faults**: measured 36 % mean and 95 % maximum relative score error, because the
very activations that carry the fault are the ones being clipped. Detection
survived (saturation only reduces an already-large score, so there were zero
decision flips) which is exactly why this would have shipped: the detector still
worked. What broke silently was the score as a *magnitude*, and the health
trend, the RUL input and any threshold tuned on the device all depend on that.

The 99.9th percentile rather than the maximum: one outlier activation would
otherwise set the scale and cost every other sample its resolution.

Verification
------------
:func:`export_deep_svdd` re-scores the calibration set through a bit-exact
simulation of the C integer path and reports the error distribution. If the
quantised scores would move any sample across the detector's operating
threshold, that is reported as a count -- the number that actually matters is
not mean score error but **decision flips**, because a 0.5 % score change that
moves one healthy sample over the alarm threshold is worse than a 5 % change
that moves nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anomaly import DeepSVDDDetector

__all__ = ["QuantisedModel", "quantise_deep_svdd", "export_deep_svdd", "emit_c_header"]

#: Percentile of |activation| used to set each layer's input scale. Not the
#: maximum: a single outlier would set the scale and cost every other sample its
#: resolution.
CALIBRATION_PERCENTILE = 99.9

#: Headroom multiplier on the calibrated scale, so an activation slightly larger
#: than anything in the calibration set does not clip.
CALIBRATION_HEADROOM = 1.25


@dataclass
class QuantisedModel:
    """int8 weights plus everything the firmware needs to reproduce the score."""

    weights_i8: List[np.ndarray]
    scales: List[float]
    in_scales: List[float]
    centre: np.ndarray
    in_centre: np.ndarray
    in_scale: np.ndarray
    in_log: np.ndarray
    feature_names: Tuple[str, ...]
    hidden: Tuple[int, ...]
    leak: float
    verification: Dict[str, float] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.in_centre.size)

    @property
    def n_parameters(self) -> int:
        return int(sum(w.size for w in self.weights_i8))

    @property
    def bytes_int8(self) -> int:
        """Flash cost: weights + scaler + centre. The scaler is float32 and is
        frequently larger than the weights on a small model -- worth knowing
        before optimising the weights."""
        w = sum(w.size for w in self.weights_i8)
        scaler = 4 * (self.in_centre.size + self.in_scale.size) + self.in_log.size
        centre = 4 * self.centre.size
        return int(w + scaler + centre + 8 * len(self.scales))

    @property
    def bytes_float32(self) -> int:
        return int(4 * sum(w.size for w in self.weights_i8)
                   + 4 * (self.in_centre.size + self.in_scale.size)
                   + self.in_log.size + 4 * self.centre.size)

    @property
    def macs(self) -> int:
        return int(sum(w.shape[0] * w.shape[1] for w in self.weights_i8))

    def sha256(self) -> str:
        h = hashlib.sha256()
        for w in self.weights_i8:
            h.update(w.tobytes())
        for s in self.scales:
            h.update(np.float32(s).tobytes())
        for s in self.in_scales:
            h.update(np.float32(s).tobytes())
        h.update(self.centre.astype(np.float32).tobytes())
        h.update(self.in_centre.astype(np.float32).tobytes())
        h.update(self.in_scale.astype(np.float32).tobytes())
        h.update(self.in_log.astype(np.uint8).tobytes())
        return h.hexdigest()

    def score_int_path(self, X: np.ndarray) -> np.ndarray:
        """Bit-exact simulation of ``pdm_model_score``.

        This is the whole point of the exporter: the Python side must be able to
        predict what the device will compute, exactly, or the device's alarms
        cannot be reproduced during an investigation.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        Xt = X.copy()
        cols = np.where(self.in_log.astype(bool))[0]
        if cols.size:
            Xt[:, cols] = np.log(np.maximum(Xt[:, cols], 1e-12))
        cur = (Xt - self.in_centre) / self.in_scale

        for i, (w, scale, in_scale) in enumerate(
                zip(self.weights_i8, self.scales, self.in_scales)):
            last = i + 1 == len(self.weights_i8)
            # The C path rounds half away from zero, which is what the explicit
            # sign handling in pdm_model.c does -- NOT numpy's round-half-even,
            # which would disagree on exactly the values sitting on a boundary.
            q = cur / in_scale
            q = np.clip(np.where(q >= 0, np.floor(q + 0.5), np.ceil(q - 0.5)), -127, 127)
            acc = q.astype(np.int32) @ w.astype(np.int32)
            y = acc.astype(np.float64) * scale * in_scale
            if not last:
                y = np.where(y < 0.0, y * self.leak, y)
            cur = y
        return np.sqrt(np.sum((cur - self.centre) ** 2, axis=1))

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": "deep_svdd_int8",
            "sha256": self.sha256(),
            "n_features": self.n_features,
            "hidden": list(self.hidden),
            "leak": self.leak,
            "input_scales": [float(s) for s in self.in_scales],
            "n_parameters": self.n_parameters,
            "bytes_int8": self.bytes_int8,
            "bytes_float32": self.bytes_float32,
            "compression": round(self.bytes_float32 / max(1, self.bytes_int8), 2),
            "macs": self.macs,
            "weight_scales": [float(s) for s in self.scales],
            "calibration_percentile": CALIBRATION_PERCENTILE,
            "feature_names": list(self.feature_names),
            "verification": self.verification,
        }


def _calibrate_activation_scales(det: DeepSVDDDetector, X: np.ndarray) -> List[float]:
    """Per-layer input scale from observed activations on ``X``.

    Runs the FLOAT forward pass and records the activation range entering each
    layer. Calibrating on the float path rather than iteratively on the
    quantised one is deliberate: it is one pass, and the quantisation error is
    small enough that a second iteration moves the scales by well under a
    percent (checked during development).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Z = det.scaler.transform(X)
    scales: List[float] = []
    cur = Z
    for i, w in enumerate(det.weights_):
        hi = float(np.percentile(np.abs(cur), CALIBRATION_PERCENTILE))
        hi = max(hi * CALIBRATION_HEADROOM, 1e-9)
        scales.append(hi / 127.0)
        pre = cur @ w
        cur = pre if i + 1 == len(det.weights_) else np.where(pre > 0, pre, det.leak * pre)
    return scales


def quantise_deep_svdd(det: DeepSVDDDetector, X_calibration: np.ndarray) -> QuantisedModel:
    """Symmetric per-tensor int8 quantisation with calibrated activation scales.

    ``X_calibration`` must span the operating range the device will see,
    including faults. Calibrating on healthy data alone sets the activation
    scales too small and reintroduces exactly the saturation this scheme exists
    to avoid.
    """
    if det.centre_ is None or not det.weights_:
        raise RuntimeError("detector is not fitted")
    if det.scaler.centre.size == 0:
        raise RuntimeError("detector scaler is not fitted")

    in_scales = _calibrate_activation_scales(det, X_calibration)
    weights_i8: List[np.ndarray] = []
    scales: List[float] = []
    for w in det.weights_:
        amax = float(np.max(np.abs(w)))
        # 127, not 128: the symmetric range is [-127, 127] so that negating a
        # weight is exact. Using -128 makes the range asymmetric by one count,
        # which is harmless numerically and confusing forever.
        scale = amax / 127.0 if amax > 0 else 1.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        weights_i8.append(q)
        scales.append(scale)

    in_log = (det.scaler.use_log.astype(np.uint8)
              if det.scaler.use_log is not None
              else np.zeros(det.scaler.centre.size, dtype=np.uint8))
    return QuantisedModel(
        weights_i8=weights_i8,
        scales=scales,
        in_scales=in_scales,
        centre=np.asarray(det.centre_, dtype=np.float64),
        in_centre=np.asarray(det.scaler.centre, dtype=np.float64),
        in_scale=np.asarray(det.scaler.scale, dtype=np.float64),
        in_log=in_log,
        feature_names=tuple(det.feature_names),
        hidden=tuple(det.hidden),
        leak=float(det.leak),
    )


def export_deep_svdd(
    det: DeepSVDDDetector,
    X_calibration: np.ndarray,
    *,
    threshold: Optional[float] = None,
    fpr: float = 0.01,
    X_healthy_holdout: Optional[np.ndarray] = None,
) -> QuantisedModel:
    """Quantise and verify against the float path on a calibration set.

    Parameters
    ----------
    X_calibration
        Feature matrix to verify on. Use held-out healthy data plus whatever
        faulty data exists -- verifying only on healthy data hides a
        quantisation error that only appears at large activations, which is
        exactly where a fault lives.
    threshold
        Operating threshold. If omitted, derived from ``X_healthy_holdout`` via
        :meth:`AnomalyDetector.calibrate` when that is supplied, and from the
        training scores otherwise -- which is only valid for a low-capacity
        detector, so passing the holdout is strongly preferred.

    Returns
    -------
    QuantisedModel
        with ``verification`` populated.
    """
    qm = quantise_deep_svdd(det, X_calibration)
    X = np.atleast_2d(np.asarray(X_calibration, dtype=np.float64))
    ref = det.score(X)
    got = qm.score_int_path(X)

    if threshold is not None:
        thr = float(threshold)
    elif X_healthy_holdout is not None:
        thr = float(det.calibrate(X_healthy_holdout, fpr=fpr))
    else:
        thr = float(det.threshold_at_fpr(fpr))
    above_ref = ref > thr
    above_got = got > thr
    flips = int(np.count_nonzero(above_ref != above_got))
    # Direction matters far more than count. A flip from "alarm" to "no alarm"
    # is a missed detection; the reverse is a nuisance alarm. They have
    # different costs (see pdmcore.eval.metrics.expected_cost, where the default
    # asymmetry is 60x) and conflating them hides the one that hurts.
    flips_to_normal = int(np.count_nonzero(above_ref & ~above_got))
    flips_to_alarm = int(np.count_nonzero(~above_ref & above_got))

    denom = np.maximum(np.abs(ref), 1e-12)
    rel = np.abs(got - ref) / denom
    qm.verification = {
        "n_calibration": int(X.shape[0]),
        "threshold": thr,
        "mean_rel_error": float(np.mean(rel)),
        "p95_rel_error": float(np.quantile(rel, 0.95)),
        "max_rel_error": float(np.max(rel)),
        "max_abs_error": float(np.max(np.abs(got - ref))),
        # The numbers that matter. Mean score error is cosmetic; a decision
        # flip is a false alarm or a missed fault. Expect a small number of
        # flips on samples that sit within quantisation noise of the threshold
        # -- those will flip under any perturbation, including a different
        # random seed. What must stay at zero on a well-calibrated export is
        # `flips_to_normal` growing with the fault severity.
        "decision_flips": flips,
        "decision_flip_rate": float(flips / max(1, X.shape[0])),
        "flips_to_normal": flips_to_normal,
        "flips_to_alarm": flips_to_alarm,
        # How close to the threshold the flipped samples were, in units of the
        # threshold. A flip at 0.5 % of the threshold is quantisation noise on a
        # borderline sample; a flip at 20 % is a quantisation defect.
        "max_flip_margin": float(
            np.max(np.abs(ref[above_ref != above_got] - thr) / max(thr, 1e-12))
        ) if flips else 0.0,
        "threshold_source": ("explicit" if threshold is not None
                             else ("holdout" if X_healthy_holdout is not None else "train")),
        "float_score_mean": float(np.mean(ref)),
        "int8_score_mean": float(np.mean(got)),
    }
    return qm


def _c_array(name: str, values: Sequence, ctype: str, per_line: int, fmt) -> str:
    vals = [fmt(v) for v in values]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(vals[i : i + per_line]) + ",")
    return f"static const {ctype} {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def emit_c_header(qm: QuantisedModel, *, name: str = "pdm_detector", path: Optional[str] = None) -> str:
    """Emit a C header that populates a ``pdm_model_i8``.

    The emitted ``<name>_init()`` is the only function a firmware needs to call;
    everything else is ``const`` and lives in flash.
    """
    out: List[str] = []
    w = out.append
    v = qm.verification
    w(f"/* GENERATED by pdmcore.models.export -- do not edit.")
    w(f" *")
    w(f" * Detector: deep_svdd int8, {qm.n_features}-{'-'.join(str(h) for h in qm.hidden)}")
    w(f" * sha256:   {qm.sha256()}")
    w(f" * Flash:    {qm.bytes_int8} B int8 (vs {qm.bytes_float32} B float32,"
      f" {qm.bytes_float32 / max(1, qm.bytes_int8):.1f}x)")
    w(f" * Compute:  {qm.macs} MACs")
    if v:
        w(f" *")
        w(f" * Quantisation verified on {int(v['n_calibration'])} calibration samples:")
        w(f" *   mean relative score error  {v['mean_rel_error'] * 100:.3f} %")
        w(f" *   p95 relative score error   {v['p95_rel_error'] * 100:.3f} %")
        w(f" *   max relative score error   {v['max_rel_error'] * 100:.3f} %")
        w(f" *   DECISION FLIPS at threshold {v['threshold']:.4f}: "
          f"{int(v['decision_flips'])} ({v['decision_flip_rate'] * 100:.2f} %)")
        w(f" *")
        w(f" * Decision flips are the number that matters: a 0.5 % score change")
        w(f" * that moves one healthy sample across the alarm threshold is worse")
        w(f" * than a 5 % change that moves nothing.")
    w(" */")
    w("")
    w(f"#ifndef {name.upper()}_H")
    w(f"#define {name.upper()}_H")
    w("")
    w('#include "pdm_dsp.h"')
    w("")
    w(f"#define {name.upper()}_N_FEATURES {qm.n_features}")
    w(f"#define {name.upper()}_N_LAYERS   {len(qm.weights_i8)}")
    w("")

    for i, wq in enumerate(qm.weights_i8):
        w(f"/* layer {i}: [{wq.shape[0]}][{wq.shape[1]}] row-major,"
          f" w_scale {qm.scales[i]:.6e}, in_scale {qm.in_scales[i]:.6e} */")
        w(_c_array(f"{name}_w{i}", wq.ravel(order="C"), "int8_t", 16, lambda x: str(int(x))))
    w(_c_array(f"{name}_centre", qm.centre, "float", 6, lambda x: f"{float(x):.9e}f"))
    w(_c_array(f"{name}_in_centre", qm.in_centre, "float", 6, lambda x: f"{float(x):.9e}f"))
    w(_c_array(f"{name}_in_scale", qm.in_scale, "float", 6, lambda x: f"{float(x):.9e}f"))
    w(_c_array(f"{name}_in_log", qm.in_log, "uint8_t", 24, lambda x: str(int(x))))
    w("")
    w(f"/* Feature order, for cross-checking against pdmcore.features.names. A")
    w(f" * mismatch here is the failure this whole versioning scheme exists to")
    w(f" * prevent, so the names travel with the weights. */")
    w(f"#ifdef {name.upper()}_WANT_NAMES")
    w(f"static const char *const {name}_feature_names[{qm.n_features}] = {{")
    for n in qm.feature_names:
        w(f'    "{n}",')
    w("};")
    w("#endif")
    w("")
    w(f"static void {name}_init(pdm_model_i8 *m)")
    w("{")
    w(f"    m->n_layers = {len(qm.weights_i8)};")
    for i, wq in enumerate(qm.weights_i8):
        w(f"    m->layers[{i}].n_in = {wq.shape[0]};")
        w(f"    m->layers[{i}].n_out = {wq.shape[1]};")
        w(f"    m->layers[{i}].w = {name}_w{i};")
        w(f"    m->layers[{i}].w_scale = {qm.scales[i]:.9e}f;")
        w(f"    m->layers[{i}].in_scale = {qm.in_scales[i]:.9e}f;")
    w(f"    m->centre = {name}_centre;")
    w(f"    m->in_centre = {name}_in_centre;")
    w(f"    m->in_scale = {name}_in_scale;")
    w(f"    m->in_log = {name}_in_log;")
    w(f"    m->n_features = {qm.n_features};")
    w(f"    m->leak = {qm.leak:.9e}f;")
    w("}")
    w("")
    w(f"#endif /* {name.upper()}_H */")

    text = "\n".join(out) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        sidecar = os.path.splitext(path)[0] + ".json"
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(qm.as_dict(), fh, indent=2)
    return text
