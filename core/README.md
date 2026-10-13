# `pdmcore` — production core

Shared, hardware-independent core for the predictive-maintenance platform: DSP,
machine simulation, detectors, evaluation, health/diagnosis/prognostics, and the
wire contract between firmware and cloud.

Every `hw/*` branch in this repository builds on this package, so the number a
microcontroller computes and the number the cloud computes are the same number.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e "core[ml,field,dev]"      # torch is optional, see below
```

`numpy` and `scipy` are the only hard dependencies. `scikit-learn` and `torch`
are optional extras on purpose: the whole core — including the unsupervised
detectors that ship to a new site — runs without them, so it installs on a
gateway or a CI runner with no wheels available.

## Quick start

```python
from pdmcore.sim import MACHINE_LIBRARY, simulate
from pdmcore.features import FeatureExtractor, FeatureConfig
from pdmcore.health import diagnose, baseline_from_vectors
import numpy as np

spec = MACHINE_LIBRARY["motor_pump_37kw"]
fx = FeatureExtractor(FeatureConfig(fs_hz=spec.fs_hz, bearing=spec.bearing,
                                    demod_band_hz=(2500.0, 6000.0)))

# 60 healthy acquisitions establish the baseline the rules compare against.
healthy = [simulate(spec, None, duration_s=0.4, seed=900 + i) for i in range(60)]
base = baseline_from_vectors(
    np.vstack([fx.extract(r.accel_g, spec.shaft_hz).vector for r in healthy]), fx.names)

faulty = simulate(spec, {"bearing_outer": 0.35}, duration_s=1.0, seed=1)
for d in diagnose(fx.extract(faulty.accel_g, spec.shaft_hz), baseline=base):
    print(d.fault, round(d.confidence, 2), d.severity, "|", d.explanation)
```

```
bearing_outer_race 1.0 advanced | BPFO harmonic family stands 11.5x above the
local envelope-spectrum floor with 5 matched harmonics
```

Or from the shell:

```bash
pdm simulate --fault bearing_inner --severity 0.4 --out /tmp/run.npz
pdm diagnose /tmp/run.npz
pdm demo --machine gearbox_132kw        # full commissioning -> alarm -> RUL walk-through
```

## Layout

| Module | Contents |
|---|---|
| `pdmcore.dsp` | Windows, calibrated spectra, envelope/kurtogram, bearing kinematics, order tracking, ISO 20816 zones. Pure functions, float64, no I/O. |
| `pdmcore.sim` | Machine signal simulator with ground truth; `degradation_series`, `fleet_dataset`. |
| `pdmcore.features` | **The feature contract.** 63 floats, versioned, identical on MCU and in cloud. |
| `pdmcore.models` | Unsupervised detectors (Mahalanobis, PCA residual, robust-z, Deep SVDD), int8 quantisation, C export. |
| `pdmcore.eval` | Leakage-resistant splits, operating-point metrics, cost model, calibration. |
| `pdmcore.health` | Health index, hysteretic alarm state machine, rule-based diagnosis, particle-filter RUL. |
| `pdmcore.io` | Wire payload schema, MQTT with store-and-forward, archival writers. |

## Design positions worth knowing before you change anything

**Spectra are calibrated, and the tests prove it.** `rms_in_band` agrees with
the time-domain RMS for broadband content *and* reads a tone's amplitude
correctly, for every scaling and every window. That requires the single-sided
fold to be applied to power (not magnitude), block averaging to happen in the
power domain, and the ENBW to be divided out when summing bins. Each of those
three was wrong at some point during development and each produced an error
between 1.0 and 1.4x that no integration test would have caught. See
`tests/test_dsp_spectrum.py`.

**Features, not waveforms, on the radio.** A 1 s 25.6 kHz 16-bit acquisition is
51 kB; the feature vector is 300 B on the wire. Raw waveforms are sent only on alarm. This
is what makes a multi-year battery life arithmetically possible, and it is why
the feature set has to be rich enough to *diagnose* from, not just detect.

**Unsupervised first.** A new site has thousands of healthy hours and zero
labelled failures. Supervised classification is a later capability, and the
shipped default is the detector a maintenance team can argue with.

**A random train/test split is never acceptable.** Partition by physical unit
(`pdmcore.eval.splits`) or the reported accuracy is measuring recording
recognition. On the simulated fleet, a grouped split gives AUC 0.95–0.99 where
an ungrouped one gives 1.000.

**Accuracy is not the metric.** At realistic prevalence (~1 in 13 000 hourly
acquisitions) a model that always says "healthy" scores 99.99 %. Use
`partial_auc`, `alarms_per_day`, `detection_delay` and `expected_cost`.

**The simulator is a signal model, not multibody dynamics.** It reproduces the
signatures diagnosis relies on; it does not derive them from contact mechanics.
Anything trained only on it must be validated on measured data (CWRU/MFPT/MIMII
or the customer's own) before it can touch a work order. Keep `nuisance=True`:
without per-acquisition load and resonance variation the healthy distribution is
unrealistically tight and every threshold tuned against it will be too
sensitive in the field.

## Tests

```bash
pytest core/tests -q
```

The suite is deliberately heavy on invariants rather than snapshots: Parseval
identities, window gain constants, kinematic frequency relations, leakage audits
and alarm state-machine transitions.

The portable C implementation (`cdsp/`) and the golden vectors that pin it
against this Python reference are added on the `hw/*` branches, where they are
compiled for each target; `core/tests/golden/` is generated by
`make golden` there.
