# Alif Ensemble E7 — design note

Companion to [`hardware/alif_e7/README.md`](../../hardware/alif_e7/README.md).

## What an NPU is for, measured rather than assumed

The question this kit exists to answer: does an NPU-bearing microcontroller
improve a vibration condition-monitoring node?

**On the baseline pipeline, no.** The numbers:

| | Latency | Share of the energy cycle |
|---|---:|---:|
| 63-float detector, plain C | ~23 µs | 0.0005 % |
| …CMSIS-NN on the M55 | ~8 µs | — |
| …Ethos-U55 (128 MAC) | ~2 µs | — |
| Whole DSP pipeline, Helium | 2.4 ms | 0.5 % |
| **Acquisition window** | **307 ms** | **85 %** |

An NPU that takes model inference from 23 µs to 2 µs is optimising 0.0005 % of
the budget. Helium, which halves the FFT, is optimising 0.5 %. Neither moves the
needle, and a node built on this part has materially the same battery life as
one built on a €90 STM32U5 — in fact slightly worse, because the sleep current
is a little higher.

**What the NPU buys is a model that does not otherwise fit.** At ~20 GMAC/s the
small Ethos-U55 runs a 2–5 MMAC 1D CNN over the raw waveform in ~0.25 ms. The
same model is 100–250 ms on the HE core alone and ~600 ms on the nRF9160's
64 MHz M33 — longer than the acquisition it would be analysing.

That architecture is what the 2025–2026 literature actually uses: end-to-end 1D
CNN on raw vibration, dual-domain multi-scale CNN at −4 dB SNR, lightweight
transformers benchmarked for on-device fault detection (see
`research/code_backed_papers_2025_2026.md`). None of them run on any other kit
here.

So the NPU changes *which hypothesis the node can test*, not how fast it tests
the current one. That is a real and interesting capability, and it is a different
claim from "the NPU makes condition monitoring faster".

## Why the experimental path is off by default

`APP_WAVEFORM_MODEL=0`.

1. **Comparability.** Eight kits exist to be compared. A kit whose default
   configuration differs produces frames that cannot be put beside the others.
2. **Energy.** The waveform model runs only after the feature path has
   escalated. Running it every cycle costs more than the entire rest of the
   cycle on a healthy machine — and 99.99 % of acquisitions are on healthy
   machines. The gate is `tier >= 2 && score > 0.55 * threshold`, the same
   escalation logic the pipeline already uses.
3. **Epistemics.** The waveform result is reported *alongside* the feature
   frame, never instead of it. The interesting output of this kit is the
   **disagreement set**: acquisitions where one path alarms and the other does
   not. That set is either the waveform model's added value or its failure mode,
   and only measured data distinguishes them. Replacing the comparable path with
   the experimental one destroys the only evidence that would settle it.

## The core split, and why power domains matter more than TOPS

| Core | Clock | Current | Role |
|---|---|---:|---|
| M55 HE | 160 MHz | ~4.5 mA | everything in the normal cycle |
| M55 HP | 400 MHz | ~22 mA | **powered off** except for the waveform model |

A 22 mA core left idle between hourly acquisitions costs 79 mA·s per hour — 17×
the entire rest of the cycle. The E7's power domains let it be genuinely absent
rather than quiet, and `pdm_board_sleep_ms` requests subsystem-off.

This is the actual argument for a heterogeneous SoC in a battery application,
and it is an argument about gating, not about throughput. A part whose big core
cannot be fully powered down is worse in this application than a small one, no
matter what its TOPS figure says.

## The build flag that costs 2× and is invisible to every test

`-mfpu=auto`, not `-mfpu=fpv5-sp-d16`.

On a Cortex-M55 it is `-mfpu=auto` that enables the MVE (Helium) vector unit.
Without it the compiler emits scalar code and CMSIS-DSP silently falls back to
its generic paths. The FFT then takes twice as long — and **produces identical
numbers**.

Every test in this repository passes either way: the golden vectors match, the
edge/cloud feature agreement matches, the quantisation round-trip matches. A 2×
performance regression invisible to the entire test suite is precisely the kind
of defect that ships.

The only detection is a cycle-count measurement on hardware, which is why
"PMU cycle counts proving MVE is active" is on the production checklist and why
the flag carries a comment in `CMakeLists.txt` instead of just being set.

The same hazard exists for the NPU: the Vela compiler's operator fallbacks are
silent, and a model with one unsupported operator runs on the CPU at 1/100th the
speed while giving the same answer.

## Open problems

- **The waveform model is not written.** It is the kit's entire reason for
  existing. Writing it needs a 1D CNN trained on *measured* data (validation
  stage 3, which no model in this repository has passed), Vela compilation, and
  hardware to measure. A stub returning a plausible probability would make the
  kit look complete and would be the most misleading artefact here.
- **No on-board radio.** The DevKit has none, so an external BLE module is
  required and the energy model borrows the STWIN.box's BLE figures rather than
  measuring this kit's.
- **Helium and NPU figures are arithmetic.** The 2× Helium speed-up is a vendor
  benchmark; the 2 µs NPU latency is MAC/cycle × clock from the Ethos-U55 TRM.
  Neither is measured.
- **MRAM instead of flash for the queue** removes the endurance question that
  constrains the STM32 kit (MRAM is ~10⁶ cycles against flash's 10⁴), which is a
  quiet advantage worth noting.
