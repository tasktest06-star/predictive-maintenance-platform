# Branch guide

This repository is a predictive-maintenance programme for rotating machinery —
motors, pumps, fans, gearboxes. It is organised as **29 branches** rather than
one trunk, because each branch answers a different question and most of them
would conflict if merged.

`main` holds the **planning baseline only**: seven product plans, a case study
and two resource lists. No code. Everything buildable lives on a branch.

**Read this before trusting a number anywhere in this repository:** almost
nothing here has been flashed to hardware, no Docker stack has been started,
and until recently no model had been validated on measured data. Each branch
carries its own honest-status table saying which of its numbers are measured,
which are cycle-count estimates, and which are simulator output. The branches
that reached a negative conclusion say so in their own titles.

---

## Where to start

| If you want… | Go to |
|---|---|
| The working platform — DSP, firmware, cloud, docs | **`platform/core`** |
| A complete per-file catalogue of every branch | `platform/core` → [`docs/08_branch_catalogue.md`](https://github.com/tasktest06-star/predictive-maintenance-platform/blob/platform/core/docs/08_branch_catalogue.md) |
| One specific hardware kit, buildable end to end | any `hw/*` |
| Whether published ML actually transfers | any `models/*` |
| Whether a digital twin earns its cost | any `twin/*` |
| 2025–26 papers that really ship code | `research/code-backed-deep-dive` |

---

## The platform

### `platform/core` — 134 files, 15 commits

**Contents.** The shared foundation, in three implementations that must agree:

- `core/pdmcore/` — the Python reference. A calibrated DSP layer (coherent
  gain, ENBW, power-domain averaging, envelope demodulation, kurtogram band
  selection, order tracking), a 63-feature extractor, bearing fault kinematics,
  ISO 20816 zones, one-class detectors (Mahalanobis, PCA residual, robust,
  Deep SVDD), int8 export, a health index with a hysteretic alarm state
  machine, a particle-filter RUL that *abstains*, grouped/time splits with a
  leakage audit, and a machine simulator. 80 tests.
- `cdsp/` — a C99 firmware twin of that DSP, pinned to the Python reference by
  **1 113 generated golden-vector checks**. No malloc, no stdio, float32 only.
  Includes a "virtual device": the real firmware driven from a sample file,
  emitting real wire frames on stdout.
- `edge/pdmedge/` — the Linux-tier agent for mains-powered nodes and gateways.
  Durable SQLite store-and-forward, calibration that refuses to guess a
  sensitivity, MQTT at QoS 1. 36 tests.
- `cloud/` — MQTT ingest with idempotency and gap accounting, VictoriaMetrics,
  a Postgres CMMS schema, Grafana dashboards.
- `docs/00`–`10` — architecture, data contract, feature spec, model cards,
  security, ops runbook, test strategy, production readiness, the branch
  catalogue, the programme framing, and dataset provenance.

**Significance.** This is the branch that makes the other 15 comparable. One
wire format (300 bytes), one feature spec, one model format — so the cloud
cannot tell which tier a device belongs to, and a result from one kit means the
same thing as a result from another. The C-against-Python verification is what
makes a firmware number trustworthy without hardware.

### `research/code-backed-deep-dive` — 35 files

**Contents.** A reproducible harvest of 2025–26 papers that actually ship code:
1 639 papers harvested, 331 with a resolvable repository, **45 after removing
aggregators and irrelevant fields.** Harvester and curator scripts, the raw
data, and four deep-dive notes.

**Significance.** The funnel is the finding. Only **6 %** of papers print a code
URL in the abstract; taking that as "papers with code" misses two thirds of the
real ones. 54 repositories turned out to be paper *lists* rather than paper
code, accounting for 140 papers. And **27 of the 45 have no licence at all**,
which grants no reuse rights — they can be read for method detail and not
vendored. A practical filter for anyone deciding what to adopt.

---

## The eight hardware kits

Each `hw/*` branch contains the **whole platform plus one kit** — clone one
branch and you have a complete system. Each adds `hardware/<kit>/` (config,
board driver, build glue, kit tests), a design note, and a slide deck.

### `hw/stwinbx1-stm32u5` — ST STEVAL-STWINBX1

**Contents.** The reference node. IIS3DWB (6.3 kHz flat bandwidth) over SPI+DMA
on an STM32U585, BLE, Stop-2 sleep. 16 config property tests.

**Significance.** The baseline every other kit is measured against, and the
source of the programme's most actionable finding: **the shipped 480 mAh LiPo's
self-discharge (20 µA) exceeds the circuit's entire average draw (14.5 µA)**, so
battery life saturates at ~1.6 years no matter what the firmware does. Swapping
to a lithium-thionyl-chloride cell gives **15.9 years with no code change.** Also
established that acquisition — not the radio, not the FFT — is 63.5 % of the
energy budget, inverting where most published designs optimise.

### `hw/esp32s3-adxl355` — ESP32-S3 + ADXL355 / IIS3DWB

**Contents.** Two sensor profiles selectable at compile time, ESP-IDF v5,
PSRAM, deep sleep with sequence-counter restore, OTA A/B partitions.

**Significance.** The low-cost option, and an honest warning: the ADXL355 is
quieter and finer (25 µg/√Hz, 20-bit) and **the wrong sensor for bearing work**,
because it rolls off at 1 kHz — below the structural resonances bearing impacts
excite. It can tell you a machine is rougher; it cannot tell you which element
is damaged. Also the branch where a config test was found to have been **passing
against the wrong sensor profile for its entire history**, because its parser
read the last `#define` in the file rather than the one the compiler sees.

### `hw/nrf9160-cellular` — Nordic nRF9160 DK

**Contents.** Cellular LTE-M direct to cloud, no gateway. Zephyr/nRF Connect
SDK with PSM (~10.5 h TAU), MCUboot, const tables in flash, and per-chunk float
conversion to avoid a 48 kB staging buffer.

**Significance.** **The radio is 98 % of the cycle energy** — 95 mA·s against
1.7 mA·s for everything else. Every optimisation that matters on the other kits
is a rounding error here, and vice versa. PSM gives 9.6 years. It is also the
only kit that **measures its own sample rate**, which matters because a ±3 %
MEMS oscillator moves every bearing frequency by the same 3 %.

### `hw/alif-e7-ethos-u55` — Alif Ensemble E7

**Contents.** Dual Cortex-M55 with two Ethos-U55 NPUs. Runs on the efficiency
core with the high-performance core and large NPU powered off.

**Significance.** The honest answer to "should a condition-monitoring node have
an NPU": **the NPU optimises 0.010 % of the energy budget**, and a node on this
€200 part has slightly *worse* battery life (1.50 y) than one on the €90
STM32U5 (1.59 y). What an NPU would buy is a model class — a 1D CNN over the raw
waveform — and that model is **deliberately not written**, because writing it
needs training on measured data. Also found that the idle high-performance core
would cost 17× the rest of the cycle, so power-domain gating matters more than
TOPS.

### `hw/renesas-ra8d1-helium` — Renesas EK-RA8D1

**Contents.** Cortex-M85 with Helium, 1 MB SRAM. **Three live axes and a real
tachometer** — the only kit with either. Adds angular resampling
(`pdm_order.c`), a stub FSP so the driver compiles under `-Werror` with no
Renesas toolchain, and multi-channel support in the shared app loop.

**Significance.** Shows that **speed buys no energy** — the M85 is 6.6× faster
and draws 3× the current, a 2× win on 9 % of the cycle. What the headroom buys
is capability. And it corrected its own documentation: order tracking was
claimed to recover 8–10 dB at bearing orders and **measured 1.19×** there,
against 1.92× at gear-mesh orders. A side finding nobody was looking for: the
angle domain also removes scallop loss, so it is amplitude-accurate in a way a
plain FFT is not even on a perfectly steady shaft.

### `hw/cn0549-piezo-ae` — ADI CN0549

**Contents.** Calibrated IEPE piezo → CN0540 front end → AD7768-1 24-bit →
Cora Z7 running Linux. The reference *instrument*, built on the `edge/` agent
rather than firmware. Two measured experiments committed as JSON.

**Significance.** Two results that matter beyond this kit:

1. **The 24-bit converter buys nothing.** Every sensing chain in the lineup is
   sensor-limited by 67–718×, so adding converter bits buys dynamic range, not
   resolution. A 10× quieter sensor bought **one grid step** of detection
   severity, not ten. The kit's real justification is **bandwidth** (DC–40 kHz
   against 6.3 kHz, where lubrication distress emits) and traceability.
2. **A cut cable is diagnosed as mechanical looseness at 97 % confidence.** The
   anomaly detector flags a disconnected sensor loudly — featureless is as
   anomalous as faulty — and the rule base then explains the noise. A fitter is
   dispatched to a healthy machine. Unfixable in the data, which is why the
   IEPE bias check **refuses to start** rather than warning.

### `hw/rpi5-hailo-gateway` — Raspberry Pi 5 + Hailo-8L

**Contents.** The gateway tier: many nodes in, one uplink out. Fan-in
accounting, gap detection, local scoring, silent-node detection, a measured
benchmark and a UPS-sizing study. 18 tests.

**Significance.** The only branch whose figures come from a **stopwatch**:
scoring a frame takes **17 µs**, and one core keeps up with **781 331 hourly
nodes** — a large site has a thousand. So compute is three orders of magnitude
from being the gateway's constraint and the 13-TOPS accelerator has no job.
Two further corrections: the benchmark **detected that its own fsync measurement
was invalid** on this host and said so rather than publishing the number; and
UPS holdover turned out *not* to be the correctness requirement the plan assumed
— `synchronous=FULL` already is, and the **storage choice** matters more than
the UPS. Its unique feature is silent-node detection: a node that stops
reporting is indistinguishable downstream from a healthy quiet machine.

### `hw/jetson-orin-nano` — NVIDIA Jetson Orin Nano Super

**Contents.** An adversarial workload study plus an on-site training pipeline
with five refusal gates.

**Significance.** **A branch that argues against its own hardware.** Three
candidate GPU workloads, all negative by 14× to 13 500×: re-scoring a
1 000-node site-year takes 44 seconds of CPU. What survives is not speed but
**data sovereignty** — a plant that will not send vibration data off site must
train on site — and that pipeline is what the branch implements. It also
produced two "checks that cannot fail": a threshold calibrated and verified on
the *same* window (which returns the target by construction, so a deliberate 4×
drift sailed through reporting exactly 0.010), and a gate on a *rate* where the
expected count was 1.8 and Poisson noise spans 0–6.

---

## The applied-model programme

Three branches asking whether the published literature transfers. Each carries
`research/` (protocol **and preregistered expectation**, written before
results), `prototype/`, and `production/`. **In progress** — see each branch's
own status table.

### `models/foundation-ts-zeroshot` — 151 files

**Contents.** CWRU and MFPT fetch/convert/verify tooling with checksummed
manifests, a windowing and presentation layer, the `pdmcore` 63-feature
baseline, a foundation-model runner, and a deployable detector.

**Significance.** **The first validation on measured data in this repository** —
the gate `docs/07_production_readiness.md` had carried from the first commit.
Three results, two unexpected:

1. **CWRU drive-end detection is saturated.** The 63-feature detector scores
   AUC 1.000 on all four folds — and so does plain window RMS with no model and
   no features. It cannot rank detectors; its benchmark reputation comes from
   multi-class severity work, a different task.
2. **MFPT discriminates and nobody quotes it.** RMS alone scores 0.348 —
   *below chance*, because the faulted records are quieter than baseline — while
   the full detector reaches 0.986.
3. **The finding with production consequences:** AUC is **1.000 under both a
   leaky and a grouped split**, while the realised false-alarm rate goes from
   0.8 % to **50.5 %** — 1 212 alarms/day/100 assets against a budget of 24. A
   63× difference, invisible in the metric the literature reports. The
   `production/` detector therefore calibrates **per operating point** and
   refuses an uncalibrated one rather than borrowing a threshold.

### `models/supervised-cnn-crossdataset` — 147 files

**Contents.** CWRU/MFPT loaders, a numpy 1D-CNN (WDCNN lineage), engineered-
feature baselines, a split ladder, and an audit of the upstream
`impl/ml-supervised-1dcnn` branch.

**Significance.** Prices the three evaluation flaws that make this literature's
99 % accuracy figures untrustworthy: random splits that leak across consecutive
windows of one recording, cross-load transfer, and cross-dataset transfer. The
upstream audit reports **six findings, all confirmed by running that branch's
own code** — which is more useful than any number from a fresh implementation.

### `models/mcsa-current-signature` — 141 files

**Contents.** Motor current signature analysis built on `pdmcore`'s calibrated
spectrum machinery: fault-frequency derivations, Park's vector and EPVA, supply
unbalance and VFD confounders, a three-phase simulator, and a leakage budget.

**Significance.** Commercially the most consequential question in the
repository. A current clamp sits in a switchgear panel that already has power
and a network — no sensor bolted to a machine, no battery, no radio. If current
covers enough fault modes, the economics of a whole programme change rather than
its accuracy. The branch treats the bearing-fault-in-current case adversarially,
because the mechanical-to-electrical coupling is weak and that is exactly where
MCSA is oversold.

---

## The digital-twin programme

Three branches split along the only boundary that matters: can a twin be
*calibrated*, does it *detect earlier*, does it *change the decision*.
**In progress.**

### `twin/motor-physics-core` — 140 files

**Contents.** An induction-motor twin: equivalent circuit, rotor dynamics,
bearing defect excitation, a thermal model, a parameter catalogue, a validity
domain, and ~900 lines of identification machinery.

**Significance.** "Digital twin" is the most oversold term in industrial IoT and
the usual failure is a model that was never calibrated against the specific
machine it claims to twin. This branch asks how much can be identified from what
a plant actually has — nameplate alone, nameplate plus one commissioning record,
nameplate plus a coast-down — and reports **posterior width, not point
estimates**. The distinction it exists to make: a twin whose structural transfer
path is unidentifiable cannot predict absolute vibration amplitude, and
therefore cannot drive ISO-zone alarming, even if it predicts frequencies
perfectly.

### `twin/hybrid-residual-ukf` — 142 files

**Contents.** A physics core plus a learned residual, UKF and particle-filter
state estimation, virtual sensing, a CUSUM harness on the residual, and a
baseline arm. Includes a **preregistered** evaluation protocol.

**Significance.** Tests the actual commercial claim behind digital twins —
earlier detection — against a competent baseline rather than a strawman:
detection delay **at matched false-alarm rate**, because a twin that alarms
earlier by alarming more often has achieved nothing. Its most original piece is
the **capacity trap**: a learned residual with enough capacity absorbs the fault
signature it exists to expose, and the branch quantifies how much capacity is
too much.

### `twin/decisions-rul-scheduling` — 153 files

**Contents.** Scenario-conditional RUL, a cost model, a value-of-information
calculation, a fleet scheduler, a count-based gate module, and five test files.
The most complete of the six.

**Significance.** Closes the loop from measurement to money, and asks the
question that is uncomfortable for the entire category: **how much is a better
RUL estimate actually worth?** If the action is "intervene at the next shutdown"
either way, improving the estimate from ±4 weeks to ±2 weeks is worth exactly
nothing. That number is what tells the other five branches whether their
accuracy improvements matter — which is why it was commissioned alongside them
rather than after.

---

## Upstream implementation branches

Twelve branches predating the platform work above. They are **not merged into
`platform/core`**, and none carries evidence of validation on measured data or
of evaluation under a leakage-audited grouped split. They are torch/TensorFlow
based, where the platform branches are numpy/scipy.

### `impl/*` — eight branches, ~1 500–3 200 lines each

| Branch | Contents |
|---|---|
| `impl/ml-supervised-1dcnn` | Supervised 1D-CNN: dataset, model, train, export, serve (~2 000 lines) |
| `impl/ml-transformer-faultformer` | Transformer with pretrain, finetune and **cross-domain** evaluation (~3 200 lines) |
| `impl/ml-unsupervised-vae` | VAE anomaly detection with train/infer/serve (~1 050 lines) |
| `impl/sensor-mems-lowpower` | MEMS node firmware with BOM (~1 950 lines) |
| `impl/sensor-piezo-ae` | Piezo/AE node with analogue front-end notes and BOM (~1 570 lines) |
| `impl/pipeline-kafka-flink` | Kafka + Flink streaming with a Java alert job (~2 630 lines) |
| `impl/pipeline-thingsboard` | ThingsBoard ingest with a VictoriaMetrics exporter (~2 160 lines) |
| `impl/dashboard-superset` | Superset dashboards for machine health and OEE (~2 550 lines) |

**Significance.** These are the breadth-first pass: one branch per architectural
choice, each self-contained and deployable on its own terms. Three of them
overlap directly with the `models/*` programme, which treats them as **subjects
of evaluation rather than competitors** — the valuable question is not "can we
write a fourth 1D-CNN" but "do these generalise when evaluated properly".

### `implementation_*` — four branches, ~850–1 100 lines each

`implementation_classical-ml`, `implementation_deep-learning`,
`implementation_llm-augmented`, `implementation_streaming-pipeline`.

**Significance.** An earlier, smaller scaffold with a shared `src/common/`
(data generator, models) and a per-approach package. `implementation_llm-augmented`
is the only exploration anywhere in the repository of LLM-assisted diagnosis,
which the later branches do not revisit.

---

## Two results that recur across branches

Worth more than any single branch's finding.

**The accelerator is never the answer.** Four kits carry an accelerator or a
premium part, and three independent measurements on three different
accelerators — Ethos-U55, Hailo-8L, Ampere — agreed:

> The feature extractor is ~200× the cost of the model, and the acquisition
> window is ~100× the cost of the extractor.

All three are justified by **models that do not exist yet**, and all three
branches say so rather than shipping a stub that returns a plausible number.

**The defects that survive longest are checks that cannot fail.** Seven are
tabulated in `platform/core`'s branch catalogue, each with the number that
exposed it — a threshold calibrated and verified on the same data, a noise
comparison between a density and a quantisation step, a config test reading the
wrong `#ifdef` profile, a gate on a rate where the expected count was 1.8. Two
were caught by noticing a **suspiciously round result** (exactly 2.000, exactly
0.010), which turns out to be a usable detection method.

---

## Honest status, programme-wide

- **Nothing has been flashed.** No ARM, ESP-IDF, Zephyr, Alif or Renesas
  toolchain was available; the RA8D1 branch compiles its driver against a stub
  FSP under the real warning set, the other four firmware branches are
  review-quality.
- **No Docker**, so the compose stack has never started and the gateway's real
  constraints — broker connections, database write rate — are unmeasured.
- **Measured-data validation has just begun.** CWRU and MFPT now run through
  the real pipeline on `models/foundation-ts-zeroshot`. Every number on the
  `hw/*` branches is still simulator output or datasheet arithmetic.
- **One dataset restricts what may be shipped.** MFPT is CC BY-NC-SA 4.0:
  validating on it is ordinary scholarly use, but **fitting deployable weights
  on it is not permitted** without the data owner's consent. See
  `platform/core` → `docs/10_dataset_provenance.md`.
- CI workflows live in `ci/github/` rather than `.github/workflows/` and must be
  copied into place from a session with `workflow` scope.
