# Architecture

```
FIELD                        EDGE                     PLATFORM              USER
─────                        ────                     ────────              ────
┌──────────────┐
│ MEMS accel   │──SPI──┐
│ mic / AE     │──ADC──┤   ┌──────────────────┐
│ temperature  │──I2C──┼──▶│  MCU             │
│ tacho / CT   │──TIM──┘   │  cdsp pipeline   │
└──────────────┘           │  63 floats       │
                           │  int8 detector   │
                           └────────┬─────────┘
                                    │ 300 B frame, CRC-32
                                    │ MQTT/TLS, BLE, LoRa, LTE-M
                           ┌────────▼─────────┐
                           │  gateway         │   (optional: Pi 5, Jetson)
                           │  store & forward │
                           │  local inference │
                           └────────┬─────────┘
                                    │ MQTT/TLS
                  ┌─────────────────▼──────────────────┐
                  │  ingest: decode → validate → score │
                  │          → alarm → work order      │
                  └──────┬───────────────┬─────────────┘
                         │               │
               ┌─────────▼──────┐  ┌─────▼──────────┐    ┌──────────────┐
               │ VictoriaMetrics│  │ Postgres       │───▶│ Grafana /    │
               │ trends         │  │ assets, orders │    │ CMMS UI      │
               └────────────────┘  └────────────────┘    └──────────────┘
```

## The one decision everything else follows from

**Features on the radio, waveforms only on demand.**

A 1 s, 25.6 kHz, 16-bit acquisition is 51 kB. The feature frame is 300 B — 170×
smaller. On LTE-M at a realistic 20 kbit/s that is 120 ms of airtime instead of
20 s, and radio energy is the dominant term in a battery-powered sensor's budget.

| Path | Payload | LTE-M airtime | Sub-GHz at 6.25 kbit/s |
|---|---:|---:|---:|
| Feature frame | 300 B | 120 ms | 384 ms |
| Raw waveform | 51 kB | 20.4 s | 65 s |

Everything else is a consequence:

- The feature set must be rich enough to **diagnose** from, not merely detect —
  hence 63 features including bearing-frequency evidence, not 6 summary
  statistics. `docs/02_feature_spec.md`.
- The edge needs a real FFT, envelope demodulation and a model, not a threshold.
  Hence `cdsp/`.
- Edge and cloud must compute identical features, or trends break at the
  boundary between a feature frame and a recomputed snapshot. Hence the golden
  vectors.
- A raw snapshot is an *event*, triggered by an alarm or a request, and it is
  the thing an analyst looks at when a work order is disputed.

## Tiered acquisition

The device does not compute everything every cycle. Escalation is driven by
measured uncertainty, not by a fixed schedule — a policy adopted from
arXiv 2511.12240 (see `research/deep_dive/02_gap_analysis_vs_plans.md`):

| Tier | Work | Cost (M33 @ 160 MHz) | When |
|---|---|---:|---|
| 0 | Time-domain indicators, streaming | ~0.4 ms | every cycle |
| 1 | + windowed FFT, band RMS, orders, ISO velocity | ~2 ms | every cycle |
| 2 | + envelope demodulation, bearing evidence | ~5 ms | tier 1 uncertain |
| 3 | + raw waveform snapshot to cloud | ~300 ms radio | tier 2 alarms |

The common case terminates at tier 1. The energy saving comes from that; the
diagnostic quality comes from tier 3 existing at all.

## Why no Kafka yet, and when to add it

The upstream plan specifies Kafka + Flink. That is the right architecture at
scale and the wrong one to start with: MQTT → service → VictoriaMetrics handles
a few thousand sensors on one node, and each sensor emits 300 bytes per hour.

Add Kafka when **any** of these becomes true — written down so the decision is
made on evidence:

1. **More than one consumer needs every frame.** The moment a second service
   (an LLM agent layer, a customer's own data lake, a second model being
   shadow-evaluated) must see the same stream, fan-out in the ingest service
   becomes the bottleneck and the failure mode becomes "consumer B being slow
   stalls consumer A".
2. **Ingest must be replayable.** Re-scoring six months of history against a
   retrained model is a routine need, and doing it from VictoriaMetrics is
   impossible — the features are not stored, only the derived series.
3. **Sustained ingest exceeds ~5 000 frames/second**, or the service can no
   longer be restarted without losing frames.
4. **Multi-region or multi-tenant isolation** requires a durable buffer at the
   boundary.

None of those is true for a 50-sensor pilot. All of them will be true for a
10 000-sensor deployment. The migration is deliberately cheap: `handle_raw()`
takes bytes and has no transport dependency, so a Kafka consumer calls the same
function.

## Where inference runs, and why there are three answers

| Where | What it can run | Why there |
|---|---|---|
| **Sensor MCU** | int8 MLP, ~2 k params, ~26 µs | No link needed to raise an alarm. A sensor that can only alarm when the network is up is not a safety instrument. |
| **Gateway** (Pi 5, Jetson) | ONNX/TensorRT, transformer-class, multi-sensor fusion | Cross-sensor and cross-asset context; survives a WAN outage; keeps raw waveforms on site when the customer requires it. |
| **Cloud** | Anything; training; kurtogram band selection | Fleet-wide models, retraining, and the compute-heavy analysis whose *results* are pushed down as configuration. |

The division is not arbitrary: **the cloud decides the configuration, the edge
executes it.** The demodulation band is chosen by a kurtogram in the cloud and
pushed to the device as two numbers; the device runs one fixed filter. That
keeps the expensive, data-hungry decision where the data and the cycles are, and
the cheap deterministic execution where the power budget is.

## Failure modes designed for

| Failure | Response |
|---|---|
| Network down | Device queues frames in flash; gateway stores and forwards. Replays are deduplicated on `(device_id, sequence)`, so a reconnection does not re-drive the alarm machine. |
| Sensor goes silent | `STALE` is a first-class alarm state. A silent sensor is indistinguishable from a healthy one on a dashboard — that is how a dead battery becomes an unmonitored machine for six months. |
| ADC clips / FIFO overruns | Device sets a flag; ingest refuses to score the frame. Scoring a clipped acquisition produces a confident anomaly on an artefact. |
| Firmware/cloud version skew | Frames carry `feature_spec_version`; ingest rejects versions it does not know rather than scoring a v1 vector with a v2 model. |
| Model is wrong about an asset | Rule-based diagnosis runs alongside and is inspectable; `AnomalyDetector.explain` attributes the score per feature; work orders carry the evidence that produced them. |
| Baseline drifts (seasons, grease, mount relaxation) | Baseline adapts **only while the asset is in NORMAL**, so a developing fault cannot be absorbed into "normal". |

## Component map

| Path | Contents |
|---|---|
| `core/pdmcore/dsp` | Calibrated spectra, envelope/kurtogram, bearing kinematics, order tracking, ISO zones |
| `core/pdmcore/sim` | Machine simulator with ground truth — how every POC runs without hardware |
| `core/pdmcore/features.py` | **The feature contract.** 63 floats, versioned |
| `core/pdmcore/models` | One-class detectors, int8 quantisation, C export |
| `core/pdmcore/eval` | Unit-grouped splits with leakage audit, operating-point metrics |
| `core/pdmcore/health` | Health index, alarm state machine, rule diagnosis, particle-filter RUL |
| `core/pdmcore/io` | Wire payload, MQTT, archival |
| `cdsp/` | The C99 twin of `pdmcore.dsp`, cross-checked by golden vectors |
| `cloud/` | Compose stack, ingest service, Grafana, CMMS schema |
| `hardware/<kit>/` | Per-kit firmware, BOM, power budget, validation (on the `hw/*` branches) |
