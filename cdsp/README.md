# `cdsp` — portable C99 condition-monitoring DSP

The firmware twin of `core/pdmcore.dsp`. Every `hw/*` branch links this library,
so the number a microcontroller computes in the field is the number the cloud
computes from the archived waveform.

```bash
make -C cdsp golden    # regenerate golden vectors from the Python reference
make -C cdsp test      # build natively and cross-check   -> PASS 1049 checks
make -C cdsp lib       # libpdmdsp.a for the firmware build
make -C cdsp size      # per-object code size
```

## Why two implementations

Because a maintenance decision must not depend on *where* the number was
computed. The edge ships a 300-byte feature frame; the cloud can recompute the
same frame from a raw snapshot. If the two disagree, every threshold, every
baseline and every trend is quietly wrong in a way that is almost impossible to
debug from a dashboard.

They are held together by generated golden vectors: `tools/gen_golden.py`
evaluates `pdmcore` on fixed inputs and emits `tests/golden.h`;
`tests/test_cdsp.c` recomputes them in C and compares. It runs on the build
host, so the agreement is checked on every commit with no hardware in the loop.

Tolerances are per-check and justified in the test — they are the expected
float32 accumulation error for that specific operation, not "close enough". A
tolerance that has to be loosened is a signal, not a nuisance.

**This caught a real bug.** The twiddle table was allocated as `3n/4` floats
where the algorithm needs `n+2`; the split-stage table overflowed into itself
and corrupted exactly bins `n/8+1 … 3n/8` of every transform. Bins outside that
range were perfect, so the FFT looked fine on a tone sweep and on Parseval, and
a spectrum plot would have looked plausible. 395 of 1049 checks failed and
pointed straight at it. Hence `PDM_RFFT_TWIDDLE_LEN(n)` — size the table with
the macro, never by hand.

## Measured footprint

Code size, `-Os`, x86-64 host (ARM Thumb-2 is typically 10–20 % smaller for
this kind of integer/float mix, so treat these as an upper bound; the per-kit
READMEs carry the real `arm-none-eabi-size` figures):

| Object | `-Os` | What it is |
|---|---:|---|
| `pdm_envelope.o` | 2 132 B | Butterworth design + biquad cascade + decimation |
| `pdm_spectrum.o` | 1 996 B | Scaling, band RMS, peak interpolation, ISO, median |
| `pdm_features.o` | 1 556 B | Streaming moments and indicators |
| `pdm_fft.o` | 1 348 B | Real FFT (split-radix real-from-complex) |
| `pdm_bearing.o` | 1 108 B | Kinematics and harmonic-family SNR |
| `pdm_model.o` | 868 B | int8 MLP inference |
| `pdm_window.o` | 636 B | Window generation and correction factors |
| **Total** | **9 644 B** | |

Static RAM for a 4096-point pipeline, which is the platform's standard
acquisition:

| Buffer | Bytes |
|---|---:|
| Window coefficients | 16 384 |
| FFT twiddle table | 16 392 |
| Bit-reversal table | 4 096 |
| Sample / FFT buffer (in place) | 16 384 |
| Magnitude array | 8 196 |
| Scaled spectrum | 8 196 |
| Envelope buffer | 16 384 |
| Model scratch | 1 024 |
| **Total** | **87 056 (85 kB)** |

85 kB is the number that decides which parts are in scope. It fits comfortably
on STM32U585 (786 kB), nRF9160 (256 kB), ESP32-S3 (512 kB internal), RA8D1
(1 MB) and Alif E7. It does **not** fit a 64 kB part, and halving `n_fft` to
2048 — which halves the RAM to ~43 kB at the cost of 6.25 Hz resolution instead
of 3.125 Hz — is the documented fallback.

The window and twiddle tables can move to flash as `const` data on parts where
RAM is tighter than flash; that is a per-kit decision recorded in each
`hardware/<kit>/README.md`.

## Deliberate constraints

- **float32 throughout.** Cortex-M4F/M33/M55 have single-precision FPUs and no
  double. A stray `double` drops into software emulation and a 4096-point FFT
  goes from ~1 ms to ~40 ms. The build uses `-Wdouble-promotion -Werror` to
  make that impossible to do by accident. The one place `double` is used
  deliberately — the moment accumulators in `pdm_features.c` — is documented
  with its measured cost (~0.4 ms per 4096-sample block on a 160 MHz M33) and
  its justification (float32 loses the small contributions to the fourth moment
  past ~2000 samples and kurtosis drifts by several percent, which is enough to
  move a bearing alarm).
- **No dynamic allocation.** No `malloc`, no VLAs, every buffer caller-provided.
  A heap on a device that must run unattended for five years is a fragmentation
  bug waiting for the worst possible moment.
- **No stdio, no errno, no locale.** Only `<math.h>`, `<string.h>`,
  `<stdint.h>`, so the library links into a `-nostdlib` image.
- **No recursion.** Stack depth is statically known.
- **In-place FFT**, because a second 4096-float buffer is 6 % of RAM on a
  256 kB part.

## CMSIS-DSP / vendor-accelerator swap-in

Three functions are the hot path and all three have a drop-in vendor
equivalent. The structure here mirrors theirs deliberately, so the swap is
local:

| `cdsp` | CMSIS-DSP | Ethos-U55 / vendor |
|---|---|---|
| `pdm_rfft` | `arm_rfft_fast_f32` | — |
| `pdm_biquad_run` | `arm_biquad_cascade_df1_f32` | — |
| `pdm_model_score` | `arm_fully_connected_s8` | NPU operator |

The real-FFT output packing is the CMSIS convention (DC in slot 0, Nyquist in
slot 1) specifically so that `arm_rfft_fast_f32` can be substituted without any
surrounding code changing — and the golden-vector tests still apply afterwards,
which is how you know the substitution was correct.

## One deliberate edge/cloud divergence

The envelope. `pdm_envelope_rectify` is **rectify-and-low-pass**, not the
Hilbert/analytic envelope the cloud uses. The analytic signal needs a 90° all-pass
across the whole band, which on an MCU means either a long FIR (RAM) or a second
FFT (cycles).

Full-wave rectification plus a low-pass gives the same modulation spectrum up to
a DC term and a known 2/π scale factor on the fundamental. The bearing
statistics are *ratios* against a local floor, so the decision is unaffected;
only absolute envelope amplitudes differ. This is the single place where the two
implementations are not expected to agree numerically, it is called out in
`docs/02_feature_spec.md`, and the envelope path is therefore verified by
*property* (does it recover a known modulation rate, and reject a decoy at
1.7× that rate?) rather than against a golden value.

Measured in the test: true-rate correlation 756 against decoy 4.6, a 164×
margin.
