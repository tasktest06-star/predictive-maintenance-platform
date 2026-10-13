# Alif Ensemble E7 — NPU node

Two Cortex-M55s with Helium, two Ethos-U55 NPUs, 5.5 MB of on-chip SRAM. The
most capable silicon in the lineup, and the kit with the most honest story about
what that capability is for.

**Status: never flashed.** Written against the Alif Ensemble SDK; the DSP above
`board_alif_e7.c` is verified by the shared pipeline's host tests.

---

## The NPU does not make the baseline pipeline better

Stating this first because it is the result, and because every "AI
microcontroller" evaluation should start here:

| | Baseline detector | Measured on this kit |
|---|---|---|
| Model | 63-24-12 MLP, 1 800 MACs | — |
| Plain C on the HE core | ~23 µs | 0.5 % of compute |
| CMSIS-NN on the M55 | ~8 µs | — |
| **Ethos-U55 (128 MAC)** | **~2 µs** | — |
| Compute share of the energy cycle | | **0.5 %** |

Saving 21 µs once an hour is not a reason to choose a part. The whole DSP
pipeline is 2.4 ms at 9 mA — 0.02 mA·s of a 4.7 mA·s cycle — and the dominant
term here, as on every BLE kit, is the 307 ms acquisition window at 85 %.

**Helium is the part that helps.** MVE halves the FFT against a scalar M33:
1.1 ms instead of 2.2 ms for 3 × 4096-point transforms. That is a real
improvement and it is *also* only 0.5 % of the cycle.

So: on compute grounds this kit has no advantage over the €90 STWIN.box.

---

## What the NPU actually buys: a different model

At 128 MAC/cycle and 160 MHz the small Ethos-U55 sustains roughly 20 GMAC/s.
That puts a **1D CNN over the raw waveform** — 2–5 MMAC per inference — at
~0.25 ms. On the HE core alone it would be 100–250 ms; on the nRF9160's 64 MHz
M33 it is 600 ms, longer than the acquisition.

That architecture is what the recent literature actually uses. From
`research/code_backed_papers_2025_2026.md`:

- end-to-end 1D CNN on raw vibration, 99 % on CWRU at 0 HP
- dual-domain multi-scale CNN, 92.5 % at −4 dB SNR
- lightweight transformers benchmarked for on-device fault detection

None of those run on any other kit here. **The NPU changes which hypothesis the
node can test, not how fast it tests the current one.**

### So the kit ships two inference paths

```
                    ┌─ 63-float feature detector (HE core, 2 µs)   PRIMARY
acquisition ────────┤
                    └─ 2048-sample 1D CNN (HP core + NPU, 0.25 ms) EXPERIMENT
```

`APP_WAVEFORM_MODEL` is **off by default**. Three reasons:

1. A kit whose experimental path is on by default produces data that cannot be
   compared against the other seven.
2. The waveform model runs only when the feature path has already escalated.
   Running it every cycle makes this kit's energy budget *worse* than the
   STWIN.box's for no diagnostic gain on a healthy machine — and 99.99 % of
   acquisitions are on healthy machines.
3. The result is reported **alongside** the feature frame, never instead of it.
   Two models disagreeing is information; replacing the comparable path with the
   experimental one throws it away.

The waveform input is the acquisition decimated 4× to 6667 Hz and truncated to
2048 samples. That keeps the 2–3 kHz resonance band and loses everything above
3.3 kHz, which is a real cost and the reason the feature path stays primary.

---

## The core split is the engineering

| Core | Clock | Current | NPU | Role |
|---|---|---:|---|---|
| M55 HE | 160 MHz | ~4.5 mA | 128 MAC | Everything in the normal cycle |
| M55 HP | 400 MHz | ~22 mA | 256 MAC | Powered **off** except for the waveform model |
| 2× A32 | — | — | — | Unused |

The HP core at 22 mA, left idle between acquisitions, would cost 79 mA·s per
hour — **17× the entire rest of the cycle**. So it is genuinely powered down, not
idled, and `pdm_board_sleep_ms` requests subsystem-off.

That is the real argument for this part: its power domains let expensive silicon
be *absent* rather than merely quiet. A heterogeneous SoC whose big core cannot
be fully gated is worse than a small one.

---

## Bill of materials

| Item | Part | Qty | Notes |
|---|---|---:|---|
| Dev kit | **Alif Ensemble E7 DevKit** (DK-E7) | 1 | ~€200. Registration needed for the SDK |
| Accelerometer | IIS3DWB (STEVAL-MKI208V1K) | 1 | SPI1 |
| BLE | External module (the DevKit has no radio) | 1 | **Note: no on-board radio** |
| Cell | Same 480 mAh LiPo reference as the STWIN.box | 1 | Replace with LiSOCl₂ AA in production |
| Mount | M8 stud | 1 | |

The DevKit has no radio. That is a genuine inconvenience: this is the only MCU
kit here needing an external transport module, and it is why the energy model
uses the same BLE figures as the STWIN.box rather than measured ones.

---

## Energy

```
$ python3 ../../tools/power_budget.py --kit alif_e7
    settle      0.3000 mA.s    6.9 %
    acquire     4.0000 mA.s   91.8 %  ##################################
    compute     0.0216 mA.s    0.5 %  #
    radio       0.2852 mA.s    6.5 %  ##
```

Compute is **0.5 %**. Helium halved it from 1 % and the NPU would take it to
0.02 %. Neither matters.

Battery life at hourly: **1.47 y** on the reference 480 mAh LiPo — slightly
*worse* than the STWIN.box's 1.55 y, because the sleep current is marginally
higher. And as on that kit the cell's 20 µA self-discharge exceeds the circuit's
17 µA draw, so the number is a property of the battery, not the silicon.

With a LiSOCl₂ AA: ~14 years. Same conclusion, same arithmetic:

```
python3 ../../tools/power_budget.py --kit alif_e7 \
    --battery-mah 2400 --self-discharge-pct-month 0.08
```

---

## The build flag that silently costs 2×

`-mfpu=auto`, not `-mfpu=fpv5-sp-d16`.

On a Cortex-M55, `-mfpu=auto` is what enables the MVE (Helium) vector unit.
Without it the compiler emits scalar code, CMSIS-DSP falls back to its generic
paths, and the FFT takes twice as long.

**It produces identical numbers.** Every test in this repository passes either
way; the golden vectors match, the feature agreement test matches, the
quantisation round-trip matches. A 2× performance regression that is invisible to
the entire test suite is exactly the kind of thing that ships, which is why the
flag carries a comment in `CMakeLists.txt` rather than just being set.

The only way to catch it is a cycle-count measurement on hardware, which is on
the production checklist.

---

## Validation protocol

Stages 1–4 as in `hardware/stwinbx1/README.md`, plus:

- **Confirm Helium is actually being used.** Read the PMU cycle counter around
  `pdm_rfft` and compare against the scalar build. A ~2× difference means MVE is
  active; no difference means `-mfpu=auto` was lost somewhere in the build.
- **Confirm the HP subsystem is really off.** Measure current with the waveform
  model compiled in but not triggered. If it is more than ~1 mA above the HE-only
  figure, the subsystem is idling rather than gated and the kit's energy
  advantage is gone.
- **Validate the waveform model against the feature model, not instead of it.**
  The interesting output is the disagreement set: acquisitions where one path
  alarms and the other does not. Those are either the waveform model's added
  value or its failure mode, and only measured data distinguishes them.

---

## Production checklist

- [ ] `tools/check_config.py` clean in CI
- [ ] Cross-compile for the HE core in CI
- [ ] PMU cycle counts proving MVE is active
- [ ] HP subsystem current proving it is gated, not idle
- [ ] Vela-compiled model checked against its float reference before deployment — the Vela compiler's operator fallbacks are silent, and a model that falls back to the CPU runs 100× slower while producing the same answer
- [ ] Secure enclave used for clock/power services, not direct register writes — the enclave owns them and an application that bypasses it works until the enclave disagrees
- [ ] External BLE module selected, and its energy measured rather than borrowed from the STWIN.box figures
- [ ] MRAM queue endurance: MRAM is ~10⁶ cycles, far better than flash, so the queue wear question that constrains the STM32 kit does not arise here
- [ ] TrustZone / secure boot via the enclave

---

## Honest status

| | |
|---|---|
| Shared DSP verified | ✅ via platform/core |
| Tables in flash | ✅ enabled |
| `app_config.h` cross-checked | ✅ |
| **Compiled for Cortex-M55** | ❌ no Alif SDK in this environment |
| **Flashed** | ❌ |
| Helium speed-up | ❌ vendor benchmark, not measured |
| NPU latency | ❌ Ethos-U55 TRM arithmetic (MAC/cycle × clock), not measured |
| Waveform model | ⬜ **not written.** `waveform_model.c` and the Vela output are declared by the build and absent |
| `ble_transport.c`, `mram_queue.c` | ⬜ declared, not implemented |

The waveform model is the kit's entire reason for existing and it is not
written. Writing it needs: a 1D CNN trained on measured data (stage 3 of the
validation protocol, which no model here has passed), Vela compilation, and
hardware to measure. Shipping a stub that returns a plausible probability would
make the kit look complete and would be the most misleading thing in this
repository.
