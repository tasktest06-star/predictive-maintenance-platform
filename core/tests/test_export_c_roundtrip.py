"""The training-to-device round trip: Python exports, C reproduces.

This is the highest-value test in the repository. Everything else verifies one
side of the boundary; this one verifies that a detector trained in Python and
exported to a C header produces **the same score** when evaluated by the actual
firmware inference path.

Without it, "the model is deployed" is an assumption. With it, a disagreement
between a cloud score and a device score is a bug with a known location rather
than a mystery.

It compiles ``cdsp/src/pdm_model.c`` with a host C compiler and runs it. Skipped
(not failed) when no compiler is available, because the Python suite must stay
runnable on a machine without a toolchain -- but CI has one, so the skip is
visible rather than silent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pdmcore.features import FeatureConfig, FeatureExtractor
from pdmcore.models.anomaly import DeepSVDDDetector
from pdmcore.models.export import emit_c_header, export_deep_svdd
from pdmcore.sim import MACHINE_LIBRARY, simulate

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CDSP = os.path.join(_REPO, "cdsp")

_HARNESS = r"""
#include "pdm_dsp.h"
#include "pdm_detector.h"
#include <stdio.h>
#include <math.h>

int main(void)
{
    pdm_model_i8 m;
    pdm_detector_init(&m);
    static float scratch[2 * PDM_MODEL_MAX_WIDTH];
    static float feat[PDM_DETECTOR_N_FEATURES];
    for (;;) {
        int got = 0;
        for (int i = 0; i < PDM_DETECTOR_N_FEATURES; ++i) {
            if (fscanf(stdin, "%f", &feat[i]) != 1) { got = -1; break; }
            ++got;
        }
        if (got != PDM_DETECTOR_N_FEATURES) break;
        float score = 0.0f;
        if (pdm_model_score(&m, feat, scratch, &score) != PDM_OK) return 1;
        printf("%.9e\n", (double)score);
    }
    return 0;
}
"""


def _have_toolchain() -> bool:
    return (
        shutil.which("cc") is not None
        and os.path.isdir(os.path.join(_CDSP, "include"))
        and os.path.isfile(os.path.join(_CDSP, "src", "pdm_model.c"))
    )


pytestmark = pytest.mark.skipif(
    not _have_toolchain(), reason="no host C compiler or cdsp/ not present"
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Train a detector on simulated healthy data, quantise, export, compile."""
    spec = MACHINE_LIBRARY["motor_pump_37kw"]
    fx = FeatureExtractor(FeatureConfig(
        fs_hz=spec.fs_hz, bearing=spec.bearing, demod_band_hz=(2500.0, 6000.0)))

    healthy = [simulate(spec, None, duration_s=0.3, seed=3000 + i) for i in range(90)]
    faulty = [
        simulate(spec, {mode: sev}, duration_s=0.3, seed=4000 + i)
        for i, (mode, sev) in enumerate(
            [(m, s) for m in ("bearing_outer", "bearing_inner", "unbalance")
             for s in (0.3, 0.7)] * 3)
    ]
    Xh = np.vstack([fx.extract(r.accel_g, spec.shaft_hz).vector for r in healthy])
    Xf = np.vstack([fx.extract(r.accel_g, spec.shaft_hz).vector for r in faulty])

    det = DeepSVDDDetector(feature_names=fx.names, hidden=(24, 12), epochs=200, seed=1)
    det.fit(Xh[:60])
    holdout = Xh[60:]
    X_cal = np.vstack([holdout, Xf])
    qm = export_deep_svdd(det, X_cal, X_healthy_holdout=holdout, fpr=0.01)

    work = tmp_path_factory.mktemp("export")
    header = work / "pdm_detector.h"
    emit_c_header(qm, name="pdm_detector", path=str(header))
    harness = work / "harness.c"
    harness.write_text(textwrap.dedent(_HARNESS))

    binary = work / "harness"
    proc = subprocess.run(
        ["cc", "-std=c99", "-O2", "-Wall", "-Wextra",
         f"-I{os.path.join(_CDSP, 'include')}", f"-I{work}",
         str(harness), os.path.join(_CDSP, "src", "pdm_model.c"),
         "-lm", "-o", str(binary)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"C harness failed to compile:\n{proc.stdout}\n{proc.stderr}")

    return {"qm": qm, "det": det, "binary": str(binary),
            "X_cal": X_cal, "holdout": holdout, "Xf": Xf, "names": fx.names}


def _run_c(binary: str, X: np.ndarray) -> np.ndarray:
    payload = "\n".join(" ".join(f"{v:.9e}" for v in row) for row in np.atleast_2d(X))
    proc = subprocess.run([binary], input=payload, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"C harness exited {proc.returncode}: {proc.stderr}"
    return np.array([float(line) for line in proc.stdout.split()], dtype=np.float64)


def test_c_matches_python_exactly(exported):
    """The headline check: C and the Python simulation agree to float32.

    1e-5 relative, not 1e-7: the C path accumulates in float32 at the layer
    boundaries while the Python simulation uses float64 for the dequantisation
    multiply. The int8 MACs themselves are exact in both.
    """
    sample = exported["X_cal"][::7]
    got = _run_c(exported["binary"], sample)
    want = exported["qm"].score_int_path(sample)
    assert got.size == want.size
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-9)
    assert float(np.max(rel)) < 1e-5, f"worst relative difference {np.max(rel):.3e}"


def test_quantisation_flips_only_borderline_decisions(exported):
    """Score error is cosmetic; a decision flip is a false alarm or a miss.

    Zero flips is the wrong assertion. A sample sitting within quantisation
    noise of the threshold will flip under *any* perturbation -- a different
    random seed, a different compiler's rounding -- and demanding zero would
    make this test flaky while telling you nothing. What is asserted instead:

    * the flip rate is small (a few percent at most);
    * flips happen only to samples that were already within ~25 % of the
      threshold, i.e. they are borderline, not a quantisation defect;
    * the aggregate operating point is unchanged.
    """
    qm = exported["qm"]
    v = qm.verification
    assert v["threshold_source"] == "holdout"
    assert v["decision_flip_rate"] <= 0.05, (
        f"{v['decision_flips']} of {v['n_calibration']} samples "
        f"({v['decision_flip_rate'] * 100:.1f} %) changed side of the threshold"
    )
    if v["decision_flips"]:
        assert v["max_flip_margin"] <= 0.25, (
            f"a sample {v['max_flip_margin'] * 100:.0f} % away from the threshold flipped; "
            f"that is a quantisation defect, not a borderline case"
        )


def test_quantised_detector_still_separates(exported):
    """Detection must survive quantisation at the calibrated operating point."""
    qm, det = exported["qm"], exported["det"]
    thr = qm.verification["threshold"]
    tpr = float(np.mean(qm.score_int_path(exported["Xf"]) > thr))
    fpr = float(np.mean(qm.score_int_path(exported["holdout"]) > thr))
    assert tpr > 0.9, f"int8 TPR {tpr:.3f} at the calibrated threshold"
    assert fpr <= 0.1, f"int8 FPR {fpr:.3f} at the calibrated threshold"
    # And the ranking must be essentially unchanged.
    from pdmcore.eval import auc

    y = np.r_[np.zeros(len(exported["holdout"])), np.ones(len(exported["Xf"]))]
    auc_f = auc(y, det.score(exported["X_cal"]))
    auc_q = auc(y, qm.score_int_path(exported["X_cal"]))
    assert auc_q > auc_f - 0.02, f"quantisation cost {auc_f - auc_q:.3f} AUC"


def test_holdout_threshold_differs_from_train_threshold(exported):
    """Pins the overfitting trap that motivated `calibrate`.

    Deep SVDD with 1 800 parameters fitted on 60 frames drives its training
    samples almost onto the centre, so a training-score threshold sits far below
    where held-out healthy data lands. Measured here at roughly an order of
    magnitude. If this test ever starts failing because the two agree, either the
    model stopped overfitting (good, check why) or `calibrate` stopped using the
    holdout (bad).
    """
    det, qm = exported["det"], exported["qm"]
    train_thr = det.threshold_at_fpr(0.01)
    holdout_thr = qm.verification["threshold"]
    assert holdout_thr > 3.0 * train_thr, (
        f"train threshold {train_thr:.4f} vs holdout {holdout_thr:.4f}: expected the "
        f"holdout threshold to be much higher for an overparameterised detector"
    )


def test_header_carries_the_feature_order(exported):
    """The names travel with the weights.

    A model fed features in a different order than it was trained on produces
    confident nonsense, and nothing in the numbers reveals it. The header
    therefore embeds the names behind a compile-time flag so firmware can assert
    against `pdmcore.features` at build time.
    """
    qm = exported["qm"]
    header = emit_c_header(qm, name="pdm_detector")
    assert "PDM_DETECTOR_WANT_NAMES" in header
    for name in ("time.rms", "iso.velocity_rms_mm_s", "bearing.outer_race.ratio"):
        assert f'"{name}"' in header, f"{name} missing from the emitted header"
    assert f"#define PDM_DETECTOR_N_FEATURES {len(exported['names'])}" in header


def test_sha256_is_stable_and_sensitive(exported):
    """The registry hash must be reproducible, and must change when weights do."""
    qm = exported["qm"]
    first = qm.sha256()
    assert first == qm.sha256()
    perturbed = qm.weights_i8[0].copy()
    perturbed[0, 0] = np.int8((int(perturbed[0, 0]) + 1) % 127)
    original = qm.weights_i8[0]
    qm.weights_i8[0] = perturbed
    try:
        assert qm.sha256() != first
    finally:
        qm.weights_i8[0] = original
