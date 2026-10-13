# Production readiness

An honest assessment of what is built, what is designed, and what is missing.
Nothing here is marked done because it exists on disk; "done" means exercised.

Legend: **✅ done and exercised** · **🟡 built, not exercised** · **⬜ designed
only** · **❌ not started**

## Platform core

| Item | State | Evidence / gap |
|---|---|---|
| Calibrated DSP | ✅ | 64 tests; Parseval holds for 4 scalings × 4 windows |
| Machine simulator | ✅ | 5 asset presets, degradation series, labelled fleet |
| Feature contract (63, versioned) | ✅ | Pinned by tests; C twin agrees |
| C library vs Python agreement | ✅ | 1 049 golden-vector checks pass |
| One-class detectors | ✅ | 4 detectors, unit-grouped AUC 0.95–0.99 |
| Leakage-resistant splits + audit | ✅ | `leakage_report` clean on the fleet split |
| Operating-point metrics | ✅ | pAUC, alarms/day, detection delay, cost model |
| Health index + alarm machine | ✅ | 0/30 false diagnoses on healthy holdout |
| Rule-based diagnosis | ✅ | Correct mode on all 4 bearing modes |
| Particle-filter RUL | ✅ | Abstains until detectable; intervals behave |
| Wire payload + CRC | ✅ | 20 tests incl. bit-flip and version rejection |
| int8 quantisation + C export | ✅ | `tools/export_model.py`; round trip verified against the compiled C path to 1e-7 |
| MQTT transport | 🟡 | Code present, no broker exercised here |
| CLI | ✅ | `simulate / features / diagnose / demo / payload` |

## Cloud

| Item | State | Evidence / gap |
|---|---|---|
| Ingest decode → score → alarm → work order | ✅ | `--replay` end-to-end, 4 negative controls pass |
| Idempotency, gap accounting, staleness | ✅ | All three verified in `--replay` |
| Prometheus metrics | ✅ | 13 counters/gauges exposed |
| HTTP + MQTT transports | 🟡 | Same `handle_raw` path as replay; untested against real clients |
| VictoriaMetrics write | 🟡 | Fire-and-forget code path, no server exercised |
| CMMS schema | 🟡 | Loaded by compose; **the service does not write to it yet** |
| Grafana dashboard | 🟡 | Valid JSON, provisioned; not rendered against live data |
| Compose stack | ⬜ | No Docker in this environment |
| Kafka/Flink | ⬜ | Deliberately deferred; trigger conditions in `docs/00` |

## Firmware

| Item | State | Evidence / gap |
|---|---|---|
| Portable C99 DSP | ✅ | Builds `-Werror -Wpedantic -Wconversion`; 9.6 kB at `-Os` |
| Real FFT, windows, features, envelope, bearing, ISO, int8 model | ✅ | Cross-checked against Python |
| Per-kit board bring-up | 🟡 | Structurally complete per branch, **never flashed** |
| Cross-compilation in CI | ❌ | `arm-none-eabi-gcc` unavailable here. **Largest single gap** |
| Timing on real silicon | ❌ | Figures are cycle-count estimates, labelled as such |
| Power measurement | ❌ | Budgets are datasheet arithmetic, labelled as such |
| Secure boot, signed OTA | ⬜ | Specified in `docs/04`; not implemented |

## Validation

| Stage | State | Gap |
|---|---|---|
| Simulated | ✅ | Pipeline, logic, calibration |
| Public measured (CWRU/MFPT/MIMII/DCASE) | ❌ | **No model has passed this.** Gate before any work order |
| Pilot (20–50 sensors, multi-month) | ❌ | The only stage that tests the real deployment; no public dataset substitutes |

## The four things to do next, in order

1. **Cross-compile in CI.** ~20 lines per kit. Until the firmware compiles for
   its target, "the firmware is complete" is a claim about structure, not about
   software. Everything else on this list is less important than this.
2. **Validate on CWRU and MFPT with bearing-wise partitioning.** Until this is
   done, every detection number in this repository is a number about a
   simulator. It is also cheap — the data is public and the harness exists.
3. **Wire the ingest service to Postgres.** The schema is complete and the
   service holds work orders in memory. Without the transactional write there is
   no audit trail, and the audit trail is the thing that makes a disputed work
   order resolvable.
5. **Adopt the regime-based split and residual calibration from
   arXiv 2607.08273.** The closest published work to this exact problem,
   MIT-licensed, with the authors' own code. See
   `research/deep_dive/03_adoption_shortlist.md`.

## Things that are easy to believe and are not true

Stated explicitly, because every one of them would be a reasonable inference
from reading the repository:

- **"The firmware works."** It is structurally complete and its DSP is verified
  against a reference. It has not run on hardware.
- **"The models are good."** They score 0.95–0.99 AUC *on simulated data under a
  grouped split*. The simulator was written by the same process that wrote the
  features, so some of that agreement is self-fulfilling.
- **"The timing numbers are measured."** The 5.7 ms feature-extraction budget is
  derived from cycle counts and vendor benchmarks. Only the host-side numbers
  (code size, RAM, the 1 049 golden checks) are measured.
- **"The power budgets are measured."** They are datasheet arithmetic. Real
  consumption is typically 20–50 % worse than datasheet arithmetic.
- **"The compose stack runs."** It has never been started; there is no Docker in
  the environment it was written in.
- **"The research list is complete."** It is arXiv-only. `MSSP`, `IEEE TIM` and
  the PHM Society conferences carry much of the serious condition-monitoring
  literature and ship no code.
