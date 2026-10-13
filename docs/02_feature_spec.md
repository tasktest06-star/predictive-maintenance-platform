# Feature specification (`FEATURE_SPEC_VERSION = 2`)

**This document defines the contract between firmware and cloud.** 63 float32
values per acquisition per axis, in a fixed order, stamped with the spec version
so the ingest service can refuse a frame whose layout it does not know.

Regenerate the authoritative list with:

```bash
python3 -c "from pdmcore.features import FeatureExtractor; print('\n'.join(FeatureExtractor().names))"
```

If that output and this document disagree, the code is right and this document
is stale — but the code is also the thing that must bump
`FEATURE_SPEC_VERSION`, and a changed layout without a bumped version is the
defect this whole mechanism exists to prevent.

## Why 63 and not 6

Because the frame must support **diagnosis**, not just detection. A detector can
run on RMS and kurtosis. Telling a millwright *which part of the bearing* and
*why* needs the bearing-frequency evidence, and that cannot be reconstructed in
the cloud from summary statistics — the information is destroyed at the device.

At 300 bytes the frame is still 170× smaller than the waveform it summarises, so
the richness is nearly free in radio terms. See `docs/00_architecture.md`.

## Budget on the target

| Stage | Cost (Cortex-M33 @ 160 MHz, CMSIS-DSP) | Produces |
|---|---:|---|
| Streaming moments, one pass | 0.4 ms | 14 `time.*` |
| Window + 4096-point real FFT | 1.1 ms | — |
| Scale, band RMS, shape, orders, ISO | 0.5 ms | 12 `band.*`, 7 `shape.*`, 2 `iso.*`, 8 `order.*` |
| Band-pass + rectify + decimate + 2048-point FFT | 2.8 ms | — |
| Harmonic SNR × 4 modes × 5 harmonics | 0.9 ms | 20 `bearing.*` |
| int8 detector inference | 0.03 ms | anomaly score |
| **Total** | **~5.7 ms** | 63 features + score |

At one acquisition per hour that is 1.6 ppm duty cycle for compute — the radio
and the sensor's own settling time dominate the energy budget, not the DSP. This
is the measurement that justifies doing full envelope analysis on the device
rather than shipping waveforms.

## Layout

### `time.*` — 14 features

| # | Name |
|---:|---|
| 0 | `time.mean` |
| 1 | `time.rms` |
| 2 | `time.std` |
| 3 | `time.peak` |
| 4 | `time.peak_to_peak` |
| 5 | `time.crest` |
| 6 | `time.kurtosis` |
| 7 | `time.skewness` |
| 8 | `time.shape` |
| 9 | `time.impulse` |
| 10 | `time.clearance` |
| 11 | `time.margin` |
| 12 | `time.zcr` |
| 13 | `time.entropy` |

### `band.*` — 12 features

| # | Name |
|---:|---|
| 14 | `band.00_rms` |
| 15 | `band.01_rms` |
| 16 | `band.02_rms` |
| 17 | `band.03_rms` |
| 18 | `band.04_rms` |
| 19 | `band.05_rms` |
| 20 | `band.06_rms` |
| 21 | `band.07_rms` |
| 22 | `band.08_rms` |
| 23 | `band.09_rms` |
| 24 | `band.10_rms` |
| 25 | `band.11_rms` |

### `shape.*` — 7 features

| # | Name |
|---:|---|
| 26 | `shape.centroid_hz` |
| 27 | `shape.spread_hz` |
| 28 | `shape.skew` |
| 29 | `shape.kurtosis` |
| 30 | `shape.rolloff95_hz` |
| 31 | `shape.flatness` |
| 32 | `shape.entropy` |

### `iso.*` — 2 features

| # | Name |
|---:|---|
| 33 | `iso.velocity_rms_mm_s` |
| 34 | `iso.zone_index` |

### `order.*` — 8 features

| # | Name |
|---:|---|
| 35 | `order.0.5x_g` |
| 36 | `order.1x_g` |
| 37 | `order.2x_g` |
| 38 | `order.3x_g` |
| 39 | `order.4x_g` |
| 40 | `order.5x_g` |
| 41 | `order.sum_1_5x_g` |
| 42 | `order.subharmonic_ratio` |

### `bearing.*` — 20 features

| # | Name |
|---:|---|
| 43 | `bearing.outer_race.ratio` |
| 44 | `bearing.outer_race.peak_snr` |
| 45 | `bearing.outer_race.n_matched` |
| 46 | `bearing.outer_race.sideband_ratio` |
| 47 | `bearing.inner_race.ratio` |
| 48 | `bearing.inner_race.peak_snr` |
| 49 | `bearing.inner_race.n_matched` |
| 50 | `bearing.inner_race.sideband_ratio` |
| 51 | `bearing.rolling_element.ratio` |
| 52 | `bearing.rolling_element.peak_snr` |
| 53 | `bearing.rolling_element.n_matched` |
| 54 | `bearing.rolling_element.sideband_ratio` |
| 55 | `bearing.cage.ratio` |
| 56 | `bearing.cage.peak_snr` |
| 57 | `bearing.cage.n_matched` |
| 58 | `bearing.cage.sideband_ratio` |
| 59 | `bearing.env_kurtosis` |
| 60 | `bearing.env_crest` |
| 61 | `bearing.demod_lo_hz` |
| 62 | `bearing.demod_hi_hz` |
## What each group is for

### `time.*` (14)
Single-pass indicators, computed every cycle before any FFT. See
`cdsp/src/pdm_features.c` for the streaming algorithm.

Diagnostic character, in the order they react:

| Indicator | Reacts first to |
|---|---|
| `rms` | overall energy: unbalance, looseness, late-stage wear |
| `peak` | the single worst impact in the block |
| `crest` | early localised defects — peak rises before RMS does |
| `kurtosis` | impulsiveness; the classic early bearing indicator |
| `clearance` | very early pitting; most sensitive and most noise-prone |
| `skewness` | asymmetric rubs, soft foot, directional looseness |
| `zcr` | dominant-frequency proxy; jumps when a resonance is excited |

**`kurtosis` is non-excess (Pearson): 3.0 for Gaussian noise.** Mixing excess
and non-excess conventions between edge and cloud shifts every threshold by
exactly 3.

**`kurtosis` is not monotonic with damage.** It rises as a defect localises,
then falls back toward 3 once the defect spreads over the raceway and the signal
becomes quasi-Gaussian again. A rule that watches kurtosis alone will declare a
severely damaged bearing healthy. This is why the health index fuses it with the
RMS trend rather than thresholding it.

### `band.*` (12)
Log-spaced band RMS from 5 Hz to 0.45·fs. Log, not linear: the information in a
vibration spectrum is roughly log-distributed, and a linear split wastes 11 of
12 bands on the high-frequency region where only broadband energy lives.

`band.10_rms` and `band.11_rms` are the high-frequency bands that rise with
lubrication distress days to weeks before any kinematic frequency appears. They
are also what the prognostics model consumes as its degradation indicator,
because unlike the health index they do not saturate.

### `shape.*` (7)
Spectral centroid, spread, skew, kurtosis, 95 % rolloff, flatness, entropy.
These describe *how* broadband energy is distributed, which separates cavitation
and lubrication starvation (energy climbs at high frequency, entropy high) from
looseness (energy spreads at low frequency) without needing a tone model.

### `iso.*` (2)
Broadband velocity RMS in mm/s over the configured band, and the ISO 20816-3
zone index (0=A … 3=D).

Reported because customers and insurers ask for it. **Alarms are not based on
it.** A fan can sit in zone B for a decade and fail in three weeks with the
broadband level barely moving, because the bearing energy lives at 4 kHz where
the velocity measurement is deliberately band-limited away.

One configuration trap, encoded in the extractor: the default measurement band
is 10–1000 Hz, which is correct above 600 rpm. Below that the 1x line itself
falls in the stop band. The simulated 395 rpm agitator reads **1.24 mm/s** over
2–1000 Hz and **0.30 mm/s** over 10–1000 Hz — a 4× under-read that would pass a
genuinely unbalanced machine. `FeatureConfig` switches the band on `shaft_hz < 10`.

### `order.*` (8)
Narrow-band RMS at 0.5x, 1x, 2x, 3x, 4x, 5x of shaft speed, plus the 1x–5x sum
and the subharmonic ratio (0.5x / 1x).

±6 % bands: wide enough to survive the speed drift of a DOL motor and a modest
VFD, narrow enough that 1x and 2x do not merge. The ratios between these — not
the absolute values — are what separate unbalance (1x dominant, 2x/1x < 0.45)
from misalignment (2x/1x ≥ 0.45, often with 3x) from looseness (subharmonics).

### `bearing.*` (20)
Per fault mode (outer race, inner race, rolling element, cage): mean
per-harmonic SNR, peak SNR, matched-harmonic count, sideband ratio. Plus
envelope kurtosis, envelope crest and the demodulation band edges actually used.

`ratio` is the **mean per-harmonic SNR against a local median floor**, not
summed amplitude. Summed amplitude over one floor estimate scales with the
harmonic count, so the number is uninterpretable alone and every threshold has
to be retuned when the count changes. With the tolerance windows used here a
pure-noise family scores ~2–3, so 4–6 is a real detection and the number reads
as "harmonics stand 5× above the local floor".

The sideband ratio is what distinguishes the modes, and it is gated: with no
carrier present, the ratio of two noise samples is unbounded and happily reports
6.0 on a healthy machine. If the carrier does not stand clear of its local floor
the result is `0.0` with `sideband_valid = 0`, so "not modulated" and "no
carrier to modulate" are distinguishable downstream.

`bearing.demod_lo_hz` / `demod_hi_hz` are included **in the frame** so the cloud
knows which band produced the evidence. Without them, a configuration change
looks like a machine change.

## The one place edge and cloud differ numerically

The envelope. The cloud uses the analytic (Hilbert) envelope; the device uses
full-wave rectification plus a low-pass, because the analytic signal needs a 90°
all-pass across the whole band — a long FIR (RAM) or a second FFT (cycles).

Rectify-and-smooth gives the same modulation spectrum up to a DC term and a known
2/π factor on the fundamental. All bearing statistics are *ratios* against a
local floor, so the decision is unaffected; only absolute envelope amplitudes
differ. Consequences, which are accepted deliberately:

- `bearing.*.ratio`, `peak_snr`, `n_matched`, `sideband_ratio` — **comparable**
  between edge and cloud.
- `bearing.env_kurtosis`, `env_crest` — **not** numerically comparable; they are
  trended per device and never compared across the edge/cloud boundary.

This is the only exception to the golden-vector agreement, and it is why the
envelope path is verified by property (does it recover a known modulation rate
and reject a decoy at 1.7× that rate?) rather than against a golden value.
Measured margin: 164×.

## Non-finite handling

A feature vector with a NaN is repaired, not rejected: the value is zeroed and
`n_nonfinite` is recorded in the frame's extras. One bad acquisition must not
stop a fleet pipeline, but a device producing them steadily is a defect, so the
count is surfaced and alarmable.

## Changing this spec

1. Change `FeatureExtractor._build_names` and the extraction.
2. Bump `FEATURE_SPEC_VERSION` in `core/pdmcore/__init__.py` **and**
   `PDM_FEATURE_SPEC_VERSION` in `cdsp/include/pdm_dsp.h`. They are asserted
   equal by the C tests.
3. Regenerate the golden vectors: `make -C cdsp golden && make -C cdsp test`.
4. Retrain and re-register every model: a model is bound to a feature config in
   `model_version.feature_config` and must not be fed a different layout.
5. Plan the fleet rollout. Ingest will **reject** old-version frames rather than
   mis-score them, so an un-upgraded fleet goes dark rather than wrong — which
   is the correct failure direction and still an outage.
