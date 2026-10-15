# Branch summary — the 16 branches built for this programme

You are on **`platform/core`**, the first branch of this programme and the
shared foundation the other fifteen are built on. This file summarises all
sixteen.

Three documents cover the branches at different depths — use whichever matches
what you need:

| Document | Depth |
|---|---|
| **This file** | One table row and one section per branch — what it is, what it found |
| [`docs/08_branch_catalogue.md`](docs/08_branch_catalogue.md) | **File by file**, with configuration, verification status and gaps per branch |
| [`docs/09_programme_models_and_twins.md`](docs/09_programme_models_and_twins.md) | Why the `models/*` and `twin/*` programmes exist, and the shared evaluation standard, recorded *before* their results |

The same summary is on `main` alongside `BRANCHES.md`, which additionally covers
the twelve pre-existing upstream branches (`impl/*`, `implementation_*`).

Each branch contains the **whole platform plus its own contribution** — clone
one and you have a complete, buildable system rather than a fragment.

---

## Quick table

| Branch | Summary |
|---|---|
| **`platform/core`** | The shared foundation: a calibrated DSP layer, a C firmware twin verified against it by 1 113 golden-vector checks, a Linux edge agent, a cloud ingest path, and eleven design documents.<br>Significance: it is what makes the other fifteen branches comparable — one wire format, one feature spec, so a result from one kit means the same as a result from another. |
| **`research/code-backed-deep-dive`** | A reproducible harvest of 2025–26 papers that actually ship code: 1 639 harvested → 331 with a repository → **45** after removing aggregators and adjacent fields.<br>Significance: only **6 %** of papers print a code URL in the abstract, and 27 of the surviving 45 have **no licence at all**, so they can be read but not vendored. |
| **`hw/stwinbx1-stm32u5`** | ST STEVAL-STWINBX1 reference node — IIS3DWB at 6.3 kHz flat bandwidth on an STM32U585, BLE, Stop-2 sleep.<br>Significance: the shipped cell **self-discharges faster than the circuit drains it**, capping life at 1.6 y regardless of firmware; a different chemistry gives **15.9 y with no code change**. |
| **`hw/esp32s3-adxl355`** | ESP32-S3 with two compile-time sensor profiles (ADXL355 or IIS3DWB), ESP-IDF v5, deep sleep with sequence-counter restore.<br>Significance: the ADXL355 is the **quieter sensor and the wrong one** for bearing work — it rolls off at 1 kHz, below the resonances bearing impacts excite. |
| **`hw/nrf9160-cellular`** | Nordic nRF9160 cellular node, no gateway — Zephyr with PSM, MCUboot, flash-resident tables.<br>Significance: **the radio is 98 % of the energy budget**, which inverts every other kit's optimisation priority; PSM yields 9.6 years. |
| **`hw/alif-e7-ethos-u55`** | Alif Ensemble E7 with two Ethos-U55 NPUs, run on the efficiency core with the rest powered down.<br>Significance: **the NPU optimises 0.010 % of the budget**, and this €200 part has *worse* battery life than the €90 STM32U5 — the accelerator's only justification is a model nobody has trained. |
| **`hw/renesas-ra8d1-helium`** | Renesas EK-RA8D1 (Cortex-M85, 1 MB SRAM) — three live axes and the only real tachometer in the lineup, plus angular resampling.<br>Significance: **speed buys no energy** (6.6× faster, 3× the current); and the branch corrected its own documentation — order tracking recovers **1.19×** at bearing orders, not the 8–10 dB claimed. |
| **`hw/cn0549-piezo-ae`** | ADI CN0549 reference instrument — calibrated IEPE piezo into a 24-bit AD7768-1 on Linux, with two measured experiments.<br>Significance: **the 24-bit converter buys nothing** (every chain is sensor-limited by 67–718×), and **a cut cable is diagnosed as mechanical looseness at 97 % confidence**. |
| **`hw/rpi5-hailo-gateway`** | Raspberry Pi 5 + Hailo-8L gateway — fan-in, gap detection, local scoring, silent-node detection, with a measured benchmark.<br>Significance: a frame scores in **17 µs** and one core covers **781 331 hourly nodes**, so compute is not the constraint and the 13-TOPS accelerator has no job. |
| **`hw/jetson-orin-nano`** | NVIDIA Jetson Orin Nano Super — an adversarial workload study plus an on-site training pipeline with five refusal gates.<br>Significance: **a branch that argues against its own hardware**, by 14× to 13 500×; what survives is not speed but data sovereignty — training without data leaving the site. |
| **`models/foundation-ts-zeroshot`** | Time-series foundation models against the platform's own detector, on **real CWRU and MFPT data** with checksummed provenance.<br>Significance: **the first measured-data validation in this repository**, and it found AUC identical (1.000) across leaky and grouped splits while the false-alarm rate moved **0.8 % → 50.5 %**. |
| **`models/supervised-cnn-crossdataset`** | A numpy 1D-CNN of the WDCNN lineage, engineered-feature baselines, a split ladder, and an audit of the upstream 1D-CNN branch.<br>Significance: prices the three evaluation flaws behind this literature's 99 % accuracy claims; the upstream audit reports **six findings, all confirmed by running that branch's own code**. |
| **`models/mcsa-current-signature`** | Motor current signature analysis on the platform's calibrated spectrum machinery — fault-frequency derivations, Park's vector, supply and VFD confounders, a three-phase simulator.<br>Significance: a current clamp needs no sensor on the machine, no battery and no radio, so if it covers enough fault modes **the economics of a programme change, not just its accuracy**. |
| **`twin/motor-physics-core`** | An induction-motor twin — equivalent circuit, rotor dynamics, defect excitation, thermal model, validity domain, and ~900 lines of parameter identification.<br>Significance: asks what can actually be **calibrated** from what a plant has, and reports posterior widths rather than point estimates — an unidentifiable transfer path means no absolute-amplitude prediction, so no ISO-zone alarming. |
| **`twin/hybrid-residual-ukf`** | A physics core plus a learned residual, UKF/particle state estimation, virtual sensing, and a CUSUM residual test, with a **preregistered** protocol.<br>Significance: tests the real twin claim — earlier detection **at matched false-alarm rate** — and quantifies the capacity trap, where a residual model large enough to fit well absorbs the fault it exists to expose. |
| **`twin/decisions-rul-scheduling`** | Scenario-conditional RUL, a cost model, a value-of-information calculation, a fleet scheduler, and five test files — the most complete of the six.<br>Significance: asks **how much a better RUL estimate is worth**; if the action is "intervene at the next shutdown" either way, improving ±4 weeks to ±2 weeks is worth nothing. |

---

## Contents and significance, in detail

### `platform/core` — the branch you are on

**Contents.** Three implementations of one signal chain that must agree.

- `core/pdmcore/` — the Python reference: calibrated spectra (coherent gain,
  ENBW, power-domain averaging), envelope demodulation with kurtogram band
  selection, order tracking, a 63-feature extractor, bearing kinematics,
  ISO 20816 zones, four one-class detectors, int8 export, a health index with a
  hysteretic alarm state machine, a particle-filter RUL that abstains rather
  than extrapolating, grouped and time splits with a leakage audit, and a
  machine simulator. 80 tests.
- `cdsp/` — a C99 firmware twin: no malloc, no stdio, float32 only, pinned to
  the Python reference by **1 113 generated golden-vector checks**. Includes a
  "virtual device" that runs the real firmware from a sample file and emits real
  300-byte wire frames.
- `edge/pdmedge/` — the Linux-tier agent: durable SQLite store-and-forward,
  calibration that refuses to guess a sensitivity, MQTT at QoS 1. 36 tests.
- `cloud/` — MQTT ingest with idempotency and gap accounting,
  VictoriaMetrics, a Postgres CMMS schema, Grafana dashboards.
- `docs/00`–`10` — architecture through dataset provenance.
- `tools/` — the shared energy model for all eight kits, the int8 model
  exporter, and the dataset checksum manifest.

**Significance.** This branch is why the other fifteen can be compared at all.
One 300-byte wire format, one feature spec, one model format — the cloud cannot
tell which tier a device belongs to. The golden-vector checks pinning C to
Python are what make a firmware number credible without hardware, and they
earned it: they caught a twiddle-table sizing bug that corrupted a third of the
spectrum.

**Summary.** The foundation. Nothing else makes sense without it.

---

### `research/code-backed-deep-dive`

**Contents.** A 22-topic arXiv harvester, a curator that removes aggregator
repositories and adjacent fields, the raw harvest committed for traceability,
the generated 45-paper list with its audit trail, four deep-dive notes, and a
13-frame deck.

**Significance.** The funnel is the finding. 1 639 papers harvested; only **103
(6 %)** print a code URL in the abstract, so treating that as "papers with code"
misses most of them — a reverse lookup against GitHub recovered the rest. 54
repositories turned out to be paper *lists* rather than paper code, accounting
for 140 papers. And **27 of the final 45 carry no licence**, which grants no
reuse rights at all.

**Summary.** A practical adoption filter, and a caution about how much published
"code availability" is real.

---

### The eight hardware kits

Each adds `hardware/<kit>/` (config, board driver, build glue, kit property
tests), a design note in `docs/hardware/`, and a slide deck.

**`hw/stwinbx1-stm32u5`** — the reference node, and the source of the
programme's most actionable finding: acquisition is 63.5 % of the energy budget,
not the radio (14.7 %) or the FFT (6.2 %), which is the opposite of where most
published designs optimise. Then the finding that overrides all of them — the
shipped 480 mAh LiPo's 20 µA self-discharge exceeds the circuit's 14.5 µA
average draw, so no firmware change can help and the cell choice is the entire
decision.

**`hw/esp32s3-adxl355`** — two sensor profiles on one board, so the
bandwidth-versus-noise trade is a compile-time switch rather than an argument.
Also the branch where a config test was found to have been **passing against the
wrong sensor profile for its whole history**, because its parser read the last
`#define` in the file rather than the one the compiler sees.

**`hw/nrf9160-cellular`** — the kit with no gateway, and the one that inverts
every priority: the LTE-M connection costs 95 mA·s against 1.7 mA·s for
everything else. It is also the only kit that **measures its own sample rate**,
which matters because a ±3 % oscillator shifts every bearing frequency by 3 %.

**`hw/alif-e7-ethos-u55`** — the honest answer to "should a monitoring node have
an NPU". The NPU takes inference from 23 µs to 2 µs and so optimises 0.010 % of
the cycle; the idle high-performance core would cost 17× the rest of the cycle,
so power-domain gating matters more than TOPS. The model that would justify the
part is **deliberately not written**, because writing it needs measured data.

**`hw/renesas-ra8d1-helium`** — three axes and a real tachometer, with
multi-channel support added to the shared app loop and a stub FSP so the driver
compiles under `-Werror` with no vendor toolchain. The angular-resampling
measurement contradicted this branch's own docstring, and an unlooked-for side
result came with it: the angle domain also removes scallop loss, so it is
amplitude-accurate in a way a plain FFT is not even on a steady shaft.

**`hw/cn0549-piezo-ae`** — the reference *instrument*, and the first branch
built on the Linux edge agent rather than firmware. Its two experiments both
refuted their own premise: the 24-bit converter buys dynamic range rather than
resolution, and a disconnected sensor does not look healthy — it looks faulty,
and the rule base explains the noise at 97 % confidence, which is why the IEPE
bias check **refuses to start** rather than warning.

**`hw/rpi5-hailo-gateway`** — the only branch whose numbers come from a
stopwatch. Its benchmark also **detected that its own fsync measurement was
invalid** on this host and said so rather than publishing it, and its holdover
study refuted the kit's own plan: `synchronous=FULL` is the correctness
requirement, not the UPS, and the storage choice matters more than either. Its
unique capability is silent-node detection — a node that stops reporting is
indistinguishable downstream from a healthy quiet machine.

**`hw/jetson-orin-nano`** — written to kill its own hardware and succeeding.
Three candidate GPU workloads all negative; re-scoring a 1 000-node site-year
takes 44 seconds of CPU. It also produced two "checks that cannot fail": a
threshold calibrated and verified on the *same* window, and a gate on a rate
where the expected count was 1.8 and Poisson noise spans 0–6.

---

### The three applied-model branches

**`models/foundation-ts-zeroshot`** — contents: CWRU and MFPT fetch, convert and
verify tooling with checksummed manifests, a windowing layer, the 63-feature
baseline, a foundation-model runner, and a deployable detector that calibrates
per operating point. Significance: **the first validation on measured data in
this repository**. CWRU drive-end detection turned out *saturated* — AUC 1.000
for the detector and also for plain window RMS with no model at all, so it
cannot rank detectors. MFPT does discriminate (0.986, where RMS alone is 0.348,
*below chance*). And the result with production consequences: AUC is 1.000 under
both a leaky and a grouped split while the realised false-alarm rate moves
**0.8 % → 50.5 %**, 1 212 alarms/day/100 assets against a budget of 24 — a 63×
difference invisible in the metric the literature reports.

**`models/supervised-cnn-crossdataset`** — contents: CWRU/MFPT loaders, a numpy
1D-CNN, engineered-feature baselines, a split ladder, and an audit of the
upstream `impl/ml-supervised-1dcnn`. Significance: prices the three flaws that
make 99 % accuracy claims untrustworthy — leaking random splits, cross-load
transfer, cross-dataset transfer — and audits existing code rather than writing
a fourth implementation, reporting six findings confirmed by running it.

**`models/mcsa-current-signature`** — contents: fault-frequency derivations from
first principles, Park's vector and EPVA, supply-unbalance and VFD confounders,
a three-phase simulator, a leakage budget. Significance: commercially the most
consequential question here, because a clamp in a switchgear panel needs no
mounting, no battery and no radio. The branch treats bearing-faults-in-current
adversarially, since that coupling is weak and is exactly where MCSA is
oversold.

---

### The three digital-twin branches

Split along the only boundary that matters — can a twin be **calibrated**, does
it **detect earlier**, does it **change the decision**.

**`twin/motor-physics-core`** — the physics and, more importantly, the
identification. "Digital twin" usually fails by never being calibrated against
the specific machine it claims to twin, so this branch asks what can be pinned
down from nameplate alone, nameplate plus a commissioning record, or nameplate
plus a coast-down — and reports posterior widths. The distinction it exists to
draw: an unidentifiable structural transfer path means absolute amplitude cannot
be predicted, so the twin cannot drive ISO-zone alarming even if its frequencies
are perfect.

**`twin/hybrid-residual-ukf`** — physics plus a learned residual, with state
estimation and a residual-domain change test. Tests the actual commercial claim
against a competent baseline at **matched false-alarm rate**, because alarming
earlier by alarming more often achieves nothing. Its most original element is
the capacity trap: a residual model with enough capacity absorbs the fault
signature it was built to reveal.

**`twin/decisions-rul-scheduling`** — the decision layer, and the branch that
tells the other five whether their accuracy gains matter. Scenario-conditional
RUL that preserves abstention, a cost model including the cost of a false alarm,
a fleet scheduler under crew and shutdown constraints, and a
value-of-information calculation that asks the uncomfortable question: if the
action does not change, a better estimate is worth nothing.

---

## Status

Honest, because these branches are explicit about it themselves. The
platform-wide version is in
[`docs/07_production_readiness.md`](docs/07_production_readiness.md), including
its section "Things that are easy to believe and are not true".

| | |
|---|---|
| `platform/core`, `research/*`, all eight `hw/*` | **Complete** |
| The six `models/*` and `twin/*` branches | **In progress** — research and prototype landed; `production/`, tests, design notes and decks outstanding on most |
| Flashed to hardware | **No** — no vendor toolchain was available; the RA8D1 compiles against a stub FSP under the real warning set |
| Docker / compose stack started | **No** |
| Validated on measured data | **Begun** — CWRU and MFPT run through the real pipeline on `models/foundation-ts-zeroshot`; every `hw/*` number is still simulator output or datasheet arithmetic |

One licence constraint to carry forward: MFPT is CC BY-NC-SA 4.0. Validating on
it is ordinary scholarly use; **fitting deployable weights on it is not
permitted** without the data owner's consent. See
[`docs/10_dataset_provenance.md`](docs/10_dataset_provenance.md).

---

## Two results that recur

**The accelerator is never the answer.** Four kits carry an accelerator or a
premium part, and three independent measurements on three different ones —
Ethos-U55, Hailo-8L, Ampere — agreed: the feature extractor is ~200× the cost of
the model, and the acquisition window is ~100× the cost of the extractor. All
three are justified by **models that do not exist yet**, and all three branches
say so rather than shipping a stub that returns a plausible number.

**The defects that survive longest are checks that cannot fail.** Seven are
tabulated in [`docs/08_branch_catalogue.md`](docs/08_branch_catalogue.md), each
with the number that exposed it. Two were caught by noticing a **suspiciously
round result** — exactly 2.000, exactly 0.010 — which turns out to be a usable
detection method.
