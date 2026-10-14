# Branch catalogue

What is on each branch, file by file, and what each one found.

Eleven branches. `platform/core` is the shared foundation; eight `hw/*` branches
each take it to one hardware kit; `research/*` is the literature work. Every
`hw/*` branch is cut from `platform/core` and contains the **whole platform**
plus its own kit — so a reader can clone one branch and have a complete,
buildable system, rather than needing two.

| Branch | Files | Commits ahead of `main` | Kit |
|---|---:|---:|---|
| [`platform/core`](#platformcore) | 130 | 11 | — |
| [`research/code-backed-deep-dive`](#researchcode-backed-deep-dive) | 35 | 1 | — |
| [`hw/stwinbx1-stm32u5`](#hwstwinbx1-stm32u5) | 127 | 7 | ST STEVAL-STWINBX1 |
| [`hw/esp32s3-adxl355`](#hwesp32s3-adxl355) | 131 | 8 | ESP32-S3 + ADXL355 / IIS3DWB |
| [`hw/nrf9160-cellular`](#hwnrf9160-cellular) | 132 | 10 | Nordic nRF9160 DK |
| [`hw/alif-e7-ethos-u55`](#hwalif-e7-ethos-u55) | 130 | 9 | Alif Ensemble E7 |
| [`hw/renesas-ra8d1-helium`](#hwrenesas-ra8d1-helium) | 154 | 9 | Renesas EK-RA8D1 |
| [`hw/cn0549-piezo-ae`](#hwcn0549-piezo-ae) | 142 | 10 | ADI CN0549 |
| [`hw/rpi5-hailo-gateway`](#hwrpi5-hailo-gateway) | 144 | 10 | Pi 5 + Hailo-8L |
| [`hw/jetson-orin-nano`](#hwjetson-orin-nano) | 140 | 10 | Jetson Orin Nano Super |

> **`main`** is the upstream repository as it was found: seven planning
> documents (`plans/00`–`07`), a research summary, and resource lists. No code.
> Everything below was built on top of it and none of it was pushed to `main`.

---

## Reading this catalogue

Each entry has the same five sections:

- **Purpose** — why the branch exists, in one paragraph.
- **Contents** — the files that are *specific to this branch*, with what each
  one is for. Shared files are listed once, under `platform/core`.
- **Configuration** — the numbers that decide what the measurements mean.
- **Findings** — what the branch measured, including the results that
  contradicted its own design.
- **Verification and gaps** — what was actually run, and what was not.

---

## `platform/core`

### Purpose

Everything that is not specific to one piece of silicon. Three
implementations of the same signal chain that must agree: a Python reference
(`core/`), a C twin for microcontrollers (`cdsp/`), and a Python agent for
Linux-class boards (`edge/`). Plus the cloud side, the documentation set, and
the tools that generate per-kit artefacts.

The design rule the whole repository rests on: **the cloud must not be able to
tell which tier a device belongs to.** One wire format, one feature spec, one
model format, one ingest path.

### Contents

#### `core/` — the Python reference, 63 features, 80 tests

| File | What it is |
|---|---|
| `pdmcore/dsp/spectrum.py` | The calibration heart. Coherent gain, ENBW, scallop loss, single-sided power folding, power-domain averaging. `rms_in_band()` divides out ENBW; `_block_power()` folds **power**, not magnitude |
| `pdmcore/dsp/windows.py` | Window functions with their coherent gain and ENBW as data, not as literals scattered through the code |
| `pdmcore/dsp/bearing.py` | BPFO/BPFI/BSF/FTF kinematics, `harmonic_energy_ratio()` (mean per-harmonic SNR against a local floor that excludes the peak), `sideband_ratio()` gated on carrier SNR with a `valid` field |
| `pdmcore/dsp/envelope.py` | Band-pass → rectify → decimate, plus `select_demodulation_band()` — a kurtogram with an energy floor, because spectral kurtosis is blind in an empty band |
| `pdmcore/dsp/integrate.py` | Acceleration → velocity in the frequency domain, with the high-pass that stops integration amplifying DC into nonsense |
| `pdmcore/dsp/iso.py` | ISO 20816-3 / 10816-3 velocity zones by machine class |
| `pdmcore/dsp/ordertrack.py` | Tacho and tacholess speed estimation |
| `pdmcore/dsp/features.py` | Time-domain moments, shape factors, zero-crossing rate, spectral entropy |
| `pdmcore/features.py` | `FeatureConfig` + `FeatureExtractor`: the **63-feature contract**. `time.*` 14 @0, `band.*` 12 @14, `shape.*` 7 @26, `iso.*` 2 @33, `order.*` 8 @35, `bearing.*` 20 @43 |
| `pdmcore/io/payload.py` | The wire format: 44-byte header + N float32 + CRC-32, 300 B for 63 features. `FrameFlags` with the channel field in bits 8–9 |
| `pdmcore/models/anomaly.py` | One-class detectors: Mahalanobis with Ledoit-Wolf shrinkage, PCA residual, robust threshold, Deep SVDD. Robust median/MAD scaling with clipping; `calibrate()` on a **held-out** healthy set |
| `pdmcore/models/export.py` | int8 post-training quantisation, per-layer calibrated activation scales, `score_int_path()` — a bit-exact Python simulation of the C integer path |
| `pdmcore/health/index.py` | Operating-point-binned baseline, EWMA trend, 0–100 health index with top-k contributors |
| `pdmcore/health/rules.py` | Rule-based diagnosis per ISO 13373-3 / 18436 practice: unbalance, misalignment, looseness, bearing, lubrication, cavitation |
| `pdmcore/health/rul.py` | Bootstrap particle-filter RUL with abstention — it declines rather than extrapolating from too few observations |
| `pdmcore/eval/splits.py` | Unit-grouped, stratified-group and time splits, plus a leakage audit |
| `pdmcore/eval/metrics.py` | Partial AUC, alarms/day, detection delay, an explicit cost model, ECE/Brier |
| `pdmcore/sim/machine.py` | The machine simulator: 1×/2×/3× lines, bearing impulse trains, resonance, gear mesh, and `nuisance=True` load/resonance/speed variation |
| `pdmcore/cli.py` | `pdm simulate`, `pdm diagnose`, `pdm payload` |
| `tests/` | 80 tests: 47 on spectrum calibration (Parseval invariants across 4 scalings × 4 windows), 20 on the payload, 6 on the C round trip, 14 on the virtual device |

#### `cdsp/` — the C twin, 1 113 generated checks

Portable C99: no `malloc`, no `stdio`, no recursion, `float32` only.

| File | What it is |
|---|---|
| `include/pdm_dsp.h` | Real FFT via an N/2 complex FFT with CMSIS-compatible DC/Nyquist packing; `PDM_RFFT_TWIDDLE_LEN(n) == n+2` |
| `include/pdm_board.h` | **The board contract.** Every kit implements exactly this: init, power, settle, acquire, shaft speed, temperature, battery, RSSI, transmit, queue, sleep, identify, watchdog — plus the multi-channel extension |
| `include/pdm_pipeline.h` | `PDM_FRAME_*` flags, the tiered pipeline result |
| `include/pdm_app.h` | The vendor-independent application state machine. A kit's `main()` is four lines |
| `src/pdm_fft.c` | The transform. The twiddle-table length bug here cost 395 of 1 049 checks |
| `src/pdm_spectrum.c` | Windowing, power-domain averaging, calibrated scaling — the C mirror of `spectrum.py` |
| `src/pdm_envelope.c` | Direct Form I biquads, Butterworth design with centre-frequency normalisation, rectify and decimate |
| `src/pdm_features.c` | Streaming Welford moments to 4th order |
| `src/pdm_bearing.c` | Harmonic SNR against a local floor |
| `src/pdm_model.c` | The int8 inference path the exporter targets |
| `src/pdm_payload.c` | Explicit byte writes, union type-punning, bitwise CRC-32 |
| `src/pdm_pipeline.c` | The tiered pipeline: time → spectral → envelope/bearing, escalating on score |
| `src/pdm_app.c` | Battery gate → sensor power/settle → speed-or-refuse → acquire → pipeline → encode → transmit-or-queue → drain → adaptive sleep. Loops over `APP_N_AXES` channels |
| `tests/test_cdsp.c` | 1 056 checks against generated golden vectors |
| `tests/test_pipeline.c` | 57 checks: the whole pipeline against the Python reference |
| `tests/test_flash_tables.c` | The `PDM_TABLES_IN_FLASH` path, proven bit-identical |
| `tests/virtual_device.c` + `board_host.c` | **The virtual device**: real firmware driven from a sample file, emitting real frames on stdout. Injectable `--tx-fail-every`, `--overrun-every`, `--battery-decay-mv`, `--selftest-fail` |
| `tools/gen_golden.py`, `gen_feature_layout.py`, `gen_tables.py` | Generate the golden vectors, the feature-layout header and the const tables from the Python reference, so the C side cannot drift |

#### `edge/` — the Linux-tier agent, 36 tests

| File | What it is |
|---|---|
| `pdmedge/agent.py` | The same loop shape as `pdm_app.c`, per channel. `HealthGate` replaces the battery check with disk and thermal |
| `pdmedge/calibration.py` | ADC code → volts → g → g-corrected. **Refuses to default a sensitivity.** `noise_budget(bandwidth_hz)` says whether a channel is sensor- or converter-limited |
| `pdmedge/queue.py` | Durable SQLite store-and-forward, WAL + `synchronous=FULL`. Evicts by **decimating the middle**, keeping both ends |
| `pdmedge/source.py` | `Source` contract + `SimSource`, `FileSource`, `IIOSource`. The IIO source **reads its sample rate back** after setting it |
| `pdmedge/sink.py` | `MqttSink` at QoS 1 with the PUBACK waited for; `StdoutSink`; `FailingSink` for testing outages |
| `pdmedge/config.py` | File-based config, validated on load — refuses rather than defaulting |
| `pdmedge/__main__.py` | `python -m pdmedge --self-test` proves the stack with no sensor attached |
| `systemd/pdm-edge.service` | Hardened unit: `ProtectSystem=strict`, `MemoryDenyWriteExecute`, `@system-service` syscall filter, `SupplementaryGroups=iio` |

#### `cloud/`, `docs/`, `tools/`, `slides/`

| Path | What it is |
|---|---|
| `cloud/ingest/app.py` | The ingest service: idempotency, staleness, gap accounting, commissioning windows, work-order suppression, 13 Prometheus metrics. `--replay` / `--mqtt` / `--http` |
| `cloud/docker-compose.yml` | Mosquitto + VictoriaMetrics + Postgres + Grafana. **Never started** — no Docker here |
| `cloud/sql/001_schema.sql` | CMMS schema with asset registry, work orders, and a `programme_performance` view |
| `cloud/grafana/dashboards/fleet_overview.json` | Provisioned dashboard |
| `docs/00`–`07` | Architecture, data contract, feature spec, model cards, security, ops runbook, test strategy, production readiness |
| `docs/07_production_readiness.md` | **The honest-status document.** Includes "things that are easy to believe and are not true" |
| `tools/power_budget.py` | The shared energy model for all eight kits. `--sweep`, `--battery-mah`, `--self-discharge-pct-month` |
| `tools/export_model.py` | Trains and exports the int8 detector. **Refuses** exports with >5 % decision flips or any non-borderline flip to normal |
| `slides/platform_core.tex` | The core deck, with figures generated by running the real pipeline |
| `ci/github/core-ci.yml` | CI, in `ci/` rather than `.github/workflows/` — see [Known constraints](#known-constraints) |

### Configuration

The reference configuration, in `cdsp/tests/ref_config/app_config.h`: 25.6 kHz,
4096-point FFT, 8192-sample acquisition, Hann, 2500–6000 Hz demodulation band,
1024-point envelope FFT after decimating by 8, SKF6205 geometry, 5 harmonics.

### Findings

**The DSP calibration bugs, all caught by invariant tests rather than by
inspection.** A √2 power-fold error; a π/4 bias from averaging magnitudes
instead of powers; a missing ENBW division; harmonic-ratio false positives from
a floor that included the peak; an unguarded sideband ratio; an energy-blind
kurtogram that chose empty bands. Each one produces plausible numbers.

**The C twin found one real bug and it was not subtle.** The twiddle table was
sized `3n/4` instead of `n+2`, corrupting bins `n/8+1` through `3n/8` — 395 of
1 049 checks. Without generated golden vectors it would have shipped.

**int8 export: a fixed 1/16 input scale produced 36 % mean and 95 % max score
error with zero decision flips.** It would have shipped on a flip test alone.
Per-layer calibrated scales fixed it; `score_int_path()` now proves the C
integer path bit-exactly in Python.

**`threshold_at_fpr` on training scores was 15.4× too low.** Calibrating a
one-class detector on the data it was fitted to is the same error the Jetson
branch later found in a different guise.

### Verification and gaps

| | |
|---|---|
| Python reference | ✅ 80 tests |
| Edge agent | ✅ 36 tests |
| C twin vs Python | ✅ 1 056 + 57 checks, flash path bit-identical |
| Firmware → cloud loop | ✅ 14 virtual-device tests with 4 negative controls |
| Wire format | ✅ CRC matches IEEE 802.3 vectors |
| Cloud ingest | ✅ via `--replay` |
| **Compose stack started** | ❌ no Docker |
| **Any model validated on measured data** | ❌ **the gate nothing has passed** |

---

## `research/code-backed-deep-dive`

### Purpose

A list of 2025–26 papers that actually ship code, built by a reproducible
pipeline rather than curated by hand — and filtered, because "papers with code"
is mostly a lie told by aggregator repositories.

### Contents

| File | What it is |
|---|---|
| `research/tools/harvest_arxiv_code.py` | 22-topic arXiv harvester, stdlib only. Topic expressions use short phrases ANDed with terms — long phrases are too narrow (`abs:"bearing fault diagnosis"` returns 47 hits all-time). Includes `search_github_by_arxiv_id()` reverse lookup, rate-limited to 30 req/min |
| `research/tools/curate.py` | Aggregator removal and relevance filtering. `AGGREGATOR_PAPER_THRESHOLD = 3`; name patterns catch `*-arxiv-daily`, `awesome-*`, `paper-list`, `paper-digest`; `RELEVANCE_THRESHOLD = 6.0` |
| `research/data/harvest_20261013.json` | The raw harvest, committed so every row is traceable |
| `research/data/curated.json`, `papers_with_code.csv` | The filtered output, 45 papers |
| `research/code_backed_papers_2025_2026.md` | The generated list, with the audit trail and the caveats above it |
| `research/deep_dive/00_method.md` | How the search was built and what it cannot see |
| `research/deep_dive/01_what_changed.md` | What actually moved in the field over the window |
| `research/deep_dive/02_gap_analysis_vs_plans.md` | The literature against this repository's own plans |
| `research/deep_dive/03_adoption_shortlist.md` | What is worth adopting, and what the adoption would cost |
| `slides/research_deep_dive.tex` | 13-frame deck |

### Findings

The funnel is the finding:

| | |
|---|---:|
| Unique papers harvested | 1 639 |
| Print a code URL in the abstract | 103 (6 %) |
| With a resolvable repository after reverse lookup | 331 |
| After aggregator removal (54 repos, 140 papers) | 191 |
| After relevance filtering (146 papers) | **45** |

**Only 6 % of papers print a code URL in the abstract.** Taking that as "papers
with code" would have missed two thirds of the real ones; asking GitHub which
repositories cite each arXiv ID recovered them.

**54 repositories were paper *lists*, not paper code** — `*-arxiv-daily`,
`awesome-*`, or anything cited by ≥3 different papers. They accounted for 140
papers whose only "code" was a listing.

**27 of the 45 have no licence**, which grants no reuse rights at all. They can
be read for method detail and not vendored.

### Verification and gaps

Every repository was resolved through the GitHub API on the generation date: it
exists, and its licence, size, stars and last-push date are recorded as
returned. That is a deliberately low bar — **it means "the code exists", not
"the code reproduces the paper". None of these repositories were executed.**

The list is arXiv-only. `MSSP`, `IEEE TIM` and the PHM Society conferences carry
much of the serious condition-monitoring literature and ship no code.

---

## `hw/stwinbx1-stm32u5`

### Purpose

The reference node, and the baseline every other kit is compared against.
Highest-bandwidth MEMS accelerometer in the lineup, the most RAM, and the only
kit whose vendor ships a condition-monitoring reference firmware to compare
against.

### Contents

| File | What it is |
|---|---|
| `hardware/stwinbx1/firmware/app_config.h` | 26 667 Hz (the IIS3DWB's only rate), 4096-point FFT, 8192-sample acquisition, 2500–6000 Hz demodulation band |
| `hardware/stwinbx1/firmware/board_stwinbx1.c` | IIS3DWB over SPI2 + GPDMA, self-test on all three axes, Stop 2 sleep via LPTIM |
| `hardware/stwinbx1/firmware/main.c` | Four lines of application logic |
| `hardware/stwinbx1/firmware/CMakeLists.txt` | `arm-none-eabi`, including `-Wdouble-promotion -Werror` |
| `hardware/stwinbx1/firmware/model/pdm_detector.h`/`.json` | The exported int8 detector and its model card |
| `hardware/stwinbx1/tools/check_config.py` | Cross-checks `app_config.h` against the Python reference before every build |
| `hardware/stwinbx1/tools/flash.sh` | ST-LINK, with an OpenOCD alternative |
| `hardware/stwinbx1/tests/test_kit.py` | 13 config property tests, kit-agnostic so every branch inherits them |
| `docs/hardware/stwinbx1.md` | Design note |
| `slides/stwinbx1.tex` | Deck |

### Configuration

| Parameter | Value | Why |
|---|---|---|
| `APP_FS_HZ` | 26 667 | The IIS3DWB's **only** rate. Decimating would make this an expensive ISM330 |
| `APP_ACQ_SAMPLES` | 8192 (307 ms) | **Set by the envelope stage, not the FFT.** Envelope resolution is a function of record duration; 4096 samples gives 6.5 Hz, where the 1.5-bin tolerance floor alone is ±9 % at BPFO — wide enough to call an inner-race fault an outer-race one |
| `APP_SENSOR_KEEPALIVE_THRESHOLD_S` | 1 | The crossover where keeping the sensor powered beats paying settling. The first value was 90 s against a 0.27 s crossover — a 330× energy error |

### Findings

**The acquisition window dominates — not the radio, not the DSP.** 63.5 % of
per-cycle energy is the time the sensor spends awake. Most published vibration-
node designs optimise the radio (14.7 %) and the FFT (6.2 %).

**The shipped 480 mAh LiPo is the binding constraint, not the firmware.** Its
3 %/month self-discharge is 20 µA — *more* than the circuit's 14.5 µA average.
Battery life saturates at ~1.6 years no matter how long the interval. A
lithium-thionyl-chloride AA at 0.08 %/month gives **1.59 → 15.2 years with no
code change.**

**A 2048-point envelope FFT was configured from a 4096-sample block** — 512
samples into a 2048-point transform. The pipeline correctly refused and
*silently stayed at tier 1*, so the bearing block read as all-zero on every
frame, indistinguishable from "no bearing fault". There are now compile-time
`#error` guards.

**The metrological limitation to close first:** the IIS3DWB's internal
oscillator is ±3 % and the board offers no way to measure the true rate. A 3 %
error moves a 106 Hz BPFO line by 3.2 Hz, which the ±2 % matcher tolerance
absorbs only just.

### Verification and gaps

Shared DSP ✅ 1 113 checks · config ✅ 13 tests · application loop ✅ via the
virtual device · **compiled for Cortex-M33 ❌** · **flashed ❌** · timing and
current ❌ estimates. `ble_transport.c` and `flash_queue.c` are referenced by
the build and deliberately **not written**, so the gap is a link error rather
than a stub that returns success.

---

## `hw/esp32s3-adxl355`

### Purpose

The low-cost, high-volume option, and the only branch that carries **two sensor
profiles on one board** — so the bandwidth-versus-noise trade is a compile-time
switch rather than an argument.

### Contents

| File | What it is |
|---|---|
| `hardware/esp32s3/firmware/app_config.h` | Dual profiles via `PDM_SENSOR_ADXL355` (4 kHz, 4096 samples, 500–1000 Hz band, **flagged weak**) or `PDM_SENSOR_IIS3DWB` (26.667 kHz) |
| `hardware/esp32s3/firmware/board_esp32s3.c` | ESP-IDF v5.x: PSRAM, deep sleep with `RTC_DATA_ATTR` state, dual-core pinning so the DSP does not contend with the Wi-Fi stack |
| `hardware/esp32s3/firmware/sdkconfig.defaults` | The settings that matter: PSRAM, flash speed, task stacks |
| `hardware/esp32s3/firmware/partitions.csv` | OTA A/B plus the flash queue region |
| `hardware/esp32s3/tools/check_config.py` | Rewritten to **compile** the header rather than parse it, so the `#ifdef` profile it reports is the one the build uses |

### Findings

**The ADXL355 is quieter and finer and the wrong choice for bearing work.**
25 µg/√Hz and 20 bits against the IIS3DWB's 75 µg/√Hz and 16 bits — but it rolls
off at 1 kHz, *below the structural resonances bearing impacts excite*. It can
see that a machine is rougher; it cannot see which element is damaged. The
config says so in a comment and the kit tests enforce the narrower band.

**The radio dominates, not the acquisition** — the only MEMS kit where that is
true, because Wi-Fi association costs more than a BLE notification by an order
of magnitude.

**Deep sleep loses SRAM, so the sequence counter must be restored.** The ingest
service deduplicates on `(device_id, sequence)`, so a node that restarts its
counter at zero has every frame after the first wake discarded as a duplicate.
`pdm_app_restore_sequence()` exists for this and `RTC_DATA_ATTR` carries it.

### Verification and gaps

Shared DSP ✅ · config ✅ 13 tests, both profiles · **ESP-IDF build ❌** (no
toolchain here) · **flashed ❌** · Wi-Fi and the flash queue declared, not
written.

---

## `hw/nrf9160-cellular`

### Purpose

The kit with no gateway. Cellular LTE-M direct to cloud, which inverts every
other kit's optimisation priority.

### Contents

| File | What it is |
|---|---|
| `hardware/nrf9160/firmware/prj.conf` | nRF Connect SDK / Zephyr: PSM with a ~10.5 h TAU, NB-IoT fallback **off**, MCUboot, deferred logging |
| `hardware/nrf9160/firmware/nrf9160dk_nrf9160_ns.overlay` | Non-secure devicetree overlay |
| `hardware/nrf9160/firmware/board_nrf9160.c` | Per-chunk float conversion — no 48 kB staging buffer — and a **measured `fs_hz_actual`** from `k_uptime_get()` |
| `hardware/nrf9160/firmware/model/pdm_tables.h` | Const tables in flash: `PDM_TABLES_IN_FLASH=1`, because 256 kB is tight |

### Configuration

Battery thresholds 3200/3050 mV. The first values were 2900 mV — **below the
part's 3.0 V minimum**, so the node would have browned out before deciding it
was low. Caught *after* the branch was already pushed and fixed in a follow-up
commit.

### Findings

**The radio is 98.2 % of the cycle energy.** 95.4 mA·s against the acquisition's
1.35 mA·s. Every lever that matters on the other kits is a rounding error here,
and every lever that matters here — PSM TAU, payload size, retry policy — is a
rounding error there. PSM gives 9.56 years.

**This is the only kit that measures its own sample rate.** `k_uptime_get()`
around the acquisition gives a true `fs_hz_actual`, which goes into the frame so
the cloud can re-derive the kinematic frequencies correctly rather than trusting
a ±3 % MEMS oscillator.

**Flash tables cost ~3 % on the transform and are worth it**, because on a part
whose sleep mode loses RAM the table-build cost is paid every cycle rather than
once at boot.

### Verification and gaps

Shared DSP ✅ · config ✅ 13 tests · flash-table path ✅ bit-identical ·
**Zephyr build ❌** · **flashed ❌** · **no modem, no SIM, no network** — the
PSM behaviour that produces the 9.56-year figure is entirely unexercised.

---

## `hw/alif-e7-ethos-u55`

### Purpose

The NPU question, asked honestly: what does a neural accelerator buy a
condition-monitoring node?

### Contents

| File | What it is |
|---|---|
| `hardware/alif_e7/firmware/app_config.h` | Same DSP config as the other IIS3DWB kits for cross-kit comparability, plus `APP_WAVEFORM_MODEL` defaulting to **0** |
| `hardware/alif_e7/firmware/board_alif_e7.c` | HE core only; the HP core and the large NPU are **powered off**. Secure-enclave unique ID, `pm_core_enter_deep_sleep_request_subsys_off` |
| `hardware/alif_e7/firmware/main.c` | NPU path gated on `r.tier >= 2u && r.score > 0.55f * threshold` |
| `hardware/alif_e7/firmware/CMakeLists.txt` | Documents `-mfpu=auto` and why |

### Findings

**The NPU optimises 0.0005 % of the energy budget.** Detector inference goes
from 23 µs in plain C to 2 µs on the Ethos-U55. Helium, which halves the FFT,
optimises 0.5 %.

**A node on this €200 part has slightly *worse* battery life than one on the €90
STM32U5** — 1.50 y against 1.59 y, because the Alif's sleep current is higher
and the compute it saves is 1 % of the cycle.

**The HP core left *idle* would cost 79 mA·s/hour** — 17× the rest of the cycle
combined. Power-domain gating matters more than TOPS, and it is the one thing
the datasheet does not sell.

**What the NPU actually buys is a model class:** a 2–5 MMAC 1D CNN over the raw
waveform at ~0.25 ms, which runs on no other kit here. **That model is not
written**, because writing it needs training on measured data and no model in
this repository has passed that gate. A stub returning a plausible probability
would be the most misleading artefact in the repository.

**`-mfpu=auto` versus `-mfpu=fpv5-sp-d16` is a 2× performance regression that
produces identical numbers and passes every test.** No test can catch it; only a
disassembly can.

### Verification and gaps

Shared DSP ✅ · config ✅ 13 tests · **Alif SDK build ❌** · **flashed ❌** ·
**NPU never run** · the waveform model **deliberately not written**.

---

## `hw/renesas-ra8d1-helium`

### Purpose

The fastest compute in the lineup, and the only kit that does something with it:
three live axes and a real tachometer.

### Contents

| File | What it is |
|---|---|
| `hardware/ra8d1/firmware/app_config.h` | `APP_N_AXES 3`, `APP_TACHO_ENABLED`, `APP_ORDER_SAMPLES_PER_REV 256`, `APP_ORDER_TRACK_DRIFT_PCT 0.5`, `APP_QUEUE_DEPTH 1536` (3 frames per acquisition) |
| `hardware/ra8d1/firmware/board_ra8d1.c` | Renesas FSP. IIS3DWB on SCI0 simple-SPI; one FIFO burst serves all three axes; GPT0 input capture for the keyphasor; `pdm_board_extra_flags()` contributes `SPEED_UNSTEADY` |
| `hardware/ra8d1/firmware/board_ra8d1.h` | The kit's extensions beyond the shared board contract |
| `hardware/ra8d1/firmware/pdm_order.c`/`.h` | **Angular resampling.** Catmull-Rom interpolation at constant angular increments; refuses a non-ascending mark list |
| `hardware/ra8d1/tests/test_order.c` | 14 checks that **measure** the resampling claim, with negative controls |
| `hardware/ra8d1/tests/fsp_stub/` | A stub FSP (`hal_data.h`, `fsp_stub.c`) so the driver can be compiled and linked on a host with no Renesas toolchain |
| `hardware/ra8d1/tools/syntax_check.sh` | Compiles the driver, `main.c` and the shared sources under `-Werror -Wconversion -Wdouble-promotion` against the stub |
| `hardware/ra8d1/tests/Makefile` | `make -C hardware/ra8d1/tests` runs both |

Plus shared-core changes this branch introduced: the channel loop in
`cdsp/src/pdm_app.c`, the multi-channel contract in `cdsp/include/pdm_board.h`,
frame flag bits 8–11, `virtual_device_mc`, and four multi-axis end-to-end tests.

### Findings

**Speed buys no energy.** The M85 is 6.6× faster than the M33 and draws 3× the
current: a 2× win on a term that is 9 % of the cycle. If the only argument for
a €100 M85 is speed, the €90 STM32U5 wins on price.

**Order tracking is nearly pointless at bearing orders**, and the module's own
docstring claimed otherwise. Measured at 4 % drift over a 307 ms record:

| | peak vs ideal | width | gain |
|---|---:|---:|---:|
| order 6 (bearing), frequency domain | 0.84 (−1.50 dB) | 2 bins | — |
| order 6, angle domain | 1.00 | 1 bin | **1.19×** |
| order 40 (gear mesh), frequency domain | 0.52 (−5.67 dB) | 6 bins | — |
| order 40, angle domain | 1.00 | 1 bin | **1.92×** |

The docstring said 8–10 dB at order 6. It was wrong by a lot, in the flattering
direction: at 3.26 Hz bins a 7 Hz smear is barely a Hann main lobe.

**An unlooked-for result: the angle domain also removes scallop loss.** On a
*perfectly steady* shaft the frequency domain still loses 1.16 dB because the
line falls between bins. For amplitude-based trending — most of ISO 20816
practice — that is arguably worth more than the smearing recovery.

**Two bugs the multi-axis test found on its first run**, both plausible-and-
wrong: the shared loop powers the sensor down as soon as channel 0's burst is in
RAM, and both boards checked "is the sensor powered?" *before* serving the cached
channels — so the node shipped **one** frame per acquisition with correct
features, correct flags and a healthy-looking dashboard. And
`pdm_board_accel_power(0)` cleared the retained block, so the extra axes would
have re-read the sensor: three different instants.

**The node deliberately does not ship order-domain features.**
`pdm_pipeline_run` takes its rate from `app_config.h` at compile time, so a
resampled record would label orders as hertz. It flags `SPEED_UNSTEADY` and
requests a snapshot instead.

### Verification and gaps

Shared DSP ✅ 1 113 checks · resampling ✅ 14 measured checks · three-axis path
✅ 4 end-to-end tests via `virtual_device_mc` · driver ✅ compiles and links
under the real warning set against a stub FSP · config ✅ 19 tests ·
**compiled for Cortex-M85 ❌** · **flashed ❌** · **Helium use unverified** ·
**tacho has seen only synthetic marks** · `ble_transport.c`, `flash_queue.c`,
`ra8_platform.c` declared, not written.

---

## `hw/cn0549-piezo-ae`

### Purpose

The reference instrument. The only kit whose numbers are **traceable** — a
calibrated IEPE piezo, a 24-bit converter, and a certificate. Its job is to be
the measurement a fleet of €90 MEMS nodes is compared against, deployed in ones
rather than hundreds.

First branch to use the shared `edge/` agent rather than firmware.

### Contents

| File | What it is |
|---|---|
| `hardware/cn0549/edge/cn0549_source.py` | `IIOSource` subclass for the AD7768-1 with the three checks an IEPE chain needs: bias classification, a mandatory filter choice, and DC removal **after** the clip check |
| `hardware/cn0549/config/agent.cn0549.json` | 32 kHz, 32768-sample blocks, 8192-point FFT, 15-minute interval, 2500–12000 Hz band |
| `hardware/cn0549/config/calibration.cn0549.json` | The calibration file, with **placeholder datasheet values** and a comment saying so |
| `hardware/cn0549/tools/experiments.py` | The two experiments that decide whether the kit earns its cost |
| `hardware/cn0549/results/experiments.json` | Their committed output; the deck's figures are generated from it |
| `hardware/cn0549/tests/test_cn0549.py` | 15 tests |
| `slides/figures/make_cn0549_figures.py` | Reads the JSON — a slide cannot disagree with its measurement |

### Findings

**The 24-bit converter buys nothing.** The first version of experiment 1
quantised one record to 16, 20 and 24 bits and compared the BPFO harmonic ratio.
All three agreed to three decimal places at every severity from 0.002 to 0.6.
Not a bug — the answer:

| chain | ADC noise | sensor noise | sensor/ADC |
|---|---:|---:|---:|
| IIS3DWB 16-bit ±16 g | 1.11 µg/√Hz | 75 µg/√Hz | 67× |
| ADXL355 20-bit ±8 g | 0.03 µg/√Hz | 25 µg/√Hz | 718× |
| CN0549 24-bit ±41 g | 0.01 µg/√Hz | 7 µg/√Hz | 628× |

Every chain is sensor-limited by two to three orders of magnitude. And when the
experiment was rewritten to sweep noise density instead, **a 10× quieter sensor
bought about one grid step, not ten**: the two quieter chains cross the 3 σ line
at severity 0.01 where the 16-bit chain needs 0.02 — one step on a factor-of-two
grid — and both cross only marginally (margins 1.24 and 1.39 against the 16-bit
chain's 1.71). The machine's own broadband noise, not the sensor's, sets the
floor for all three.

That number flipped between runs before it was pinned down: an earlier sweep put
all three chains at 0.01, because the 3 σ limit depends on a healthy-case
standard deviation that moves with the seed count and the grid. The experiment
now reports **how far past the limit each chain crossed**, and the README's
output block is pasted from the committed JSON rather than from a terminal — a
single number that flips is worse than a number with its margin attached.

What survives as justification: **bandwidth** (DC–40 kHz against 6.3 kHz;
lubrication distress emits at 20–40 kHz, which no MEMS part here can see at
all), **traceability**, and **DC accuracy**. A bandwidth-and-traceability
purchase wearing a resolution label.

**A disconnected sensor does not look healthy. It looks faulty, and gets
diagnosed.** The prediction was that converter noise would score *below*
healthy, making the worst possible condition invisible. Wrong:

| | median score | alarms |
|---|---:|---:|
| threshold (1 % FPR) | 9.7 | — |
| healthy | 5.8 | 0/20 |
| disconnected | 52.9 | 20/20 |
| outer-race 0.35 | 84.8 | 20/20 |

Distance from healthy is not ordered by smoothness. The real failure is a level
up: the rule base diagnoses the noise as **mechanical looseness at 0.97
confidence.** A fitter is sent to a healthy machine; the real fault is a
connector.

So `CN0549Source.open()` **refuses to start** when the IEPE bias check fails.
The costs are asymmetric: a refusal is loud and correctly attributed, while a
node that starts on a cut cable produces months of data showing a beautifully
smooth machine — and nobody questions good news.

### Verification and gaps

IEPE decision table ✅ 5 tests · config and calibration cross-checked ✅ ·
the sensor-limited claim ✅ asserted as a test · both experiments ✅ reproducible
· **run against a CN0549 ❌** · **IIO attribute names unverified ❌** ·
**reference-shaker calibration ❌** (the stage that justifies the kit) ·
**cross-validation against a MEMS node on one housing ❌ — the highest-value
missing experiment in this repository** · **20–40 kHz AE band never measured.**

---

## `hw/rpi5-hailo-gateway`

### Purpose

The gateway tier: many nodes in, one uplink out, autonomous when the link drops.
The first branch whose numbers come from a **stopwatch** rather than a cycle
count.

### Contents

| File | What it is |
|---|---|
| `hardware/rpi5_hailo/gateway/aggregator.py` | `Gateway.handle()` — decode, validate, gap-detect, score locally, forward or queue, drain. Plus `silent_nodes()`, the view that only the gateway can compute |
| `hardware/rpi5_hailo/gateway/hailo_backend.py` | `HailoScorer` **raises on construction** and lists the four things missing. `CpuWaveformScorer` holds the interface without producing a number |
| `hardware/rpi5_hailo/tools/benchmark.py` | Measured throughput: features, scoring, wire format, durable queue, fan-out. Names its host and flags an ineffective fsync |
| `hardware/rpi5_hailo/tools/holdover.py` | UPS sizing, which refuted the kit's own plan |
| `hardware/rpi5_hailo/results/benchmark.json`, `holdover.json` | Committed, with the caveat string |
| `hardware/rpi5_hailo/tests/test_gateway.py` | 18 tests |

### Findings

**Measured on this aarch64 host — not a Pi 5, and the benchmark says so:**

| Stage | Measured |
|---|---:|
| Frame decode (300 B, CRC checked) | 4.0 µs |
| Frame encode | 5.2 µs |
| **Score one frame** | **16.9 µs** |
| Feature extraction, 8192 samples | 3.46 ms |

**Compute is not the gateway's constraint.** One core at 75 % offered keeps up
with **781 331 hourly nodes.** A large site has a thousand — three orders of
magnitude of headroom. What actually binds a gateway is the broker's concurrent
connections, the database write rate, the uplink, and BLE airtime, **none of
which this benchmark measures.**

**The AI HAT has no job.** 13 TOPS cannot improve on 17 µs: the work is a 63×63
quadratic form, not a tensor graph, and the PCIe round trip costs more than the
arithmetic. Same conclusion as the Alif branch, reached independently on
different silicon — which makes it a general result.

**A benchmark that measured nothing, and says so.** The durable queue reported
39 828 frames/s with `synchronous=FULL`, which is not plausible. The benchmark
detects that `durable` and `nosync` agree within 20 % and warns that the fsync
is not reaching storage, so the SD-card case is explicitly **not** validated.

**UPS holdover is not the correctness requirement the plan assumed.**
`synchronous=FULL` already fsyncs every frame, and the flush happens when mains
*and* uplink are back — 50 nodes over a 14-day outage is 5 MB, clearing in 20 s
on contended 4G. So the UPS buys a graceful SQLite close and brownout
ride-through: **60 s of supercapacitor, 0.12 Wh.** The correction that matters
more is **storage**: if the SD card dies on an unclean power-down the backlog is
gone whatever the holdover.

**Silent-node detection is the feature that cannot live anywhere else.** A node
that stops reporting is indistinguishable downstream from a healthy quiet
machine. Three decisions, each with a failure behind it: the interval is
*learned* (fleet config drifts), learned *slowly* (one early frame must not
raise the threshold and mask the outage that follows), and **three** missed
intervals rather than one (crying wolf trains the operator to ignore it). And
silence is measured from **arrival**, not the frame's timestamp.

### Verification and gaps

18 gateway tests ✅ including wrap-safe gap detection, local alarms with the
uplink permanently down, a corrupt frame rejected rather than forwarded, and a
failing scorer degrading to forward-only · **run on a Pi 5 ❌** ·
**fsync cost unmeasurable here ❌** · **compose stack never started ❌** ·
**Hailo-8L raises by design** · **the radio termination is deliberately not
written** — `handle()` takes bytes, and what delivers them is the one part that
genuinely depends on the radio hardware chosen.

---

## `hw/jetson-orin-nano`

### Purpose

The hardest kit to justify, so the branch was written to kill it. The brief was
inverted: rather than "what can a GPU do?", **"is there a workload here where a
GPU is the difference between possible and impossible?"**

### Contents

| File | What it is |
|---|---|
| `hardware/jetson_orin/tools/workload_study.py` | Adversarial by construction: three candidate workloads, measured, with a verdict each |
| `hardware/jetson_orin/edge/train_onsite.py` | **On-site training** — the one justification that survived. Trains from the gateway's own queue, read-only, with five refusal gates |
| `hardware/jetson_orin/config/agent.edge.json` | 2 M-frame queue (600 MB, 4.5 years of a 50-node site), weekly training schedule |
| `hardware/jetson_orin/results/workload_study.json` | Committed; the figures read it |
| `hardware/jetson_orin/tests/test_jetson.py` | 13 tests |

### Findings

**All three candidates negative:**

| Candidate | Needs | One core supplies | Headroom |
|---|---:|---:|---:|
| Re-score a site-year (1000 nodes × 365 d, 8.76 M frames) | 43.8 s | 600 s | 14× |
| Fit the detectors (5000 frames) | 0.10 s | 600 s | 6 010× |
| Raw-waveform inference (1000 nodes hourly) | 0.0011 GMAC/s | ~15 GMAC/s | 13 500× |

The study records where candidate 3 would flip: **3 750 inferences/s, i.e. 13.5
million nodes at hourly cadence.** And it names its own weak spot — candidate 2's
fits are closed-form and SVD, not gradient descent; `torch` is not installed, so
**that number is absent rather than guessed**, and a test asserts the absence so
the gap cannot be forgotten.

**Two justifications survived, neither about speed.** Data sovereignty — a plant
that will not send data off site must train on site — is implemented. Multimodal
fusion is the genuine GPU case and has no model. **Three accelerators in this
lineup (Ethos-U55, Hailo-8L, Ampere) and all three are justified by models that
do not exist yet.**

**Two bugs, both "checks that cannot fail":**

The *circular check*. The first version fitted on 75 %, calibrated the threshold
on the last 25 %, then measured the false-alarm rate on **that same 25 %** —
which equals the target by construction. The check could never fail, and a
deliberate 4× monotone drift sailed through reporting exactly 0.010. **A
suspiciously exact number is how it announced itself.** Fixed with three windows
in time order; the same drift now gives 96 alarms where 1.8 were expected.

The *statistically wrong gate*. The replacement refused if the verification
*rate* exceeded the target by 4×. On a 175-frame window at 1 % the expected
count is 1.8, and Poisson noise alone spans 0–6 at 3 σ — so it refused
stationary data about a third of the time. In a plant that means re-running
training until it passes, **selecting for whichever window looked quiet**: a
biased model reached by a procedure that feels like diligence. Fixed with a 3 σ
Poisson band on the alarm *count*: `allowed = λ + 3√λ + 3`.

> The general lesson is worth more than the fix: **a gate expressed as a ratio
> of rates is really a gate on a count, and at small counts the ratio is the
> wrong parameterisation.** The same trap is open in the alarms/day budget and
> the n-of-m alarm logic.

**The refusals are most of the value.** 400 frames minimum; **7 days of span
minimum** — the gate that bites, because a commissioning engineer with a day on
site has plenty of frames and nowhere near enough span, and a detector that has
never seen a Monday start-up alarms on the first one; unscoreable frames
excluded, because a detector trained on clipped acquisitions learns the
artefacts; `SPEED_UNSTEADY` excluded, because training on smeared records teaches
the detector that smearing is normal.

**Energy, and the cost of being wrong:** 7.5 W Pi 5 = €13/y; 25 W Orin Super =
€44/y. €31/year per site for an idle GPU is the cheapest mistake in a
deployment, **which is precisely why nobody examines it.**

**Legacy Jetson Nano (2019) is end-of-life** — last JetPack 4.6 on Ubuntu 18.04.
Everything here runs on it in principle, but deploying an unpatched
internet-connected Linux to a plant network is a finding, not a trade-off.

### Verification and gaps

13 tests ✅ including both refusal paths with their numbers, the time-ordered
split proved against shuffled input, and the Poisson band checked at three
window sizes for both admitting noise and rejecting drift · **run on a Jetson
❌** · **`torch` absent ❌** · **GPU not used at all, by design** ·
**multimodal fusion has no model** · model rollout/rollback and the
`pdm-train` systemd units not written — on-site training without rollback is a
one-way door.

---

## Cross-branch results

Two findings recur, and both are more useful than any single kit's.

### The accelerator is never the answer

Four of the eight kits carry an accelerator or a premium part. Three independent
measurements, on three different accelerators, said the same thing:

> The feature extractor is ~200× the cost of the model, and the acquisition
> window is ~100× the cost of the extractor.

| Kit | Accelerator | What it optimises |
|---|---|---|
| Alif E7 | Ethos-U55, 2×128/256 MAC | 0.0005 % of the energy budget |
| Pi 5 | Hailo-8L, 13 TOPS | 17 µs that is already fast enough for 781 331 nodes |
| Jetson Orin | 1024-core Ampere, 67 TOPS | a workload with 13 500× headroom |

All three are justified by **models that do not exist yet** — and that is stated
the same way in all three branches, rather than hidden behind a stub that returns
a plausible number.

### The defects that survive longest are checks that cannot fail

| Where | The check | Why it could not fail |
|---|---|---|
| `tools/export_model.py` | decision flips after int8 quantisation | 0 flips with 36 % mean score error — the flip test alone would have shipped it |
| `core/pdmcore/models/anomaly.py` | `threshold_at_fpr` on training scores | calibrating on the data that was fitted: 15.4× too low |
| `edge/pdmedge/calibration.py` | `noise_density_g > resolution_g` | a density compared to a quantisation step — dimensionally meaningless, right answer by luck |
| `hardware/jetson_orin/edge/train_onsite.py` | held-out FPR vs target | calibrated and verified on the same window: equal by construction |
| `hardware/jetson_orin/edge/train_onsite.py` | FPR ratio gate | a ratio on an expected count of 1.8, where noise spans 0–6 |
| `slides/figures/make_ra8d1_figures.py` | measured markers | regex matched the 0 %-drift control too, giving 1.14× instead of 1.19× |
| `hardware/cn0549/README.md` | detection severity | a marginal 3 σ crossing quoted from a terminal run, which the committed JSON later contradicted |

Each is recorded in the branch that produced it, with the number that exposed
it. Two were exposed by **suspiciously round results** — exactly 2.000, exactly
0.010 — which is a detection method worth keeping. The last was exposed by
writing this catalogue: every numeric claim in it was cross-checked against the
committed JSON rather than against memory, and one disagreed.

---

## Known constraints

These apply to every branch and are not oversights.

**Nothing has been flashed.** No `arm-none-eabi-gcc`, ESP-IDF, nRF Connect SDK,
Alif SDK or Renesas FSP in this environment. The RA8D1 branch adds a stub FSP so
its driver compiles and links under `-Werror -Wconversion -Wdouble-promotion`;
the other four firmware branches are review-quality only.

**No Docker**, so `cloud/docker-compose.yml` has never been started, and the
gateway's real constraints — broker connections, database write rate — are
unmeasured.

**No model has passed validation on measured data.** CWRU, MFPT and MIMII are
the gate. Every detection number in this repository is a number about a
simulator whose noise model was written by the same process that wrote the
features, so some of the agreement is self-fulfilling.

**Timing and current are datasheet arithmetic**, except the gateway benchmark —
which is measured, on the wrong machine, and says so in its own output.

**CI lives in `ci/github/`, not `.github/workflows/`.** The credential used to
push these branches lacked `workflow` scope, so GitHub rejects any push touching
that path. `make ci-install` copies it into place; see `ci/README.md`. This must
be done from a session with `workflow` scope.

**The energy-model correction was made on a kit branch, not here.** The
acquisition window was corrected on `hw/renesas-ra8d1-helium`; `platform/core`
and the three branches cut after it carried the wrong model until that fix was
ported back. The four earlier kit branches will pick up the property test that
prevents a recurrence on their next merge from core — until then, their READMEs
quote acquire-share percentages that are overstated (85 % where the corrected
figure is 63.5 %). Battery-life figures moved under 3 % everywhere, because
self-discharge dominates the load on four of the five battery kits, so the
*conclusions* in those branches survive and the *percentages* do not.
