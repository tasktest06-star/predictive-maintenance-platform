# ThingsBoard IoT Pipeline — Predictive Maintenance Platform

Alternative to the Kafka+Flink stack for teams that need faster time-to-market
and a smaller operational footprint.

---

## What Is ThingsBoard?

ThingsBoard Community Edition is an open-source (Apache 2.0) IoT platform that
bundles the following capabilities in a single deployable unit:

- **Device management** — provision devices, manage credentials, store attributes
- **MQTT / HTTP / CoAP ingestion** — built-in broker; no separate HiveMQ/VerneMQ needed
- **Rule engine** — visual, node-based stream processing (filter → transform → action)
- **Alarm system** — threshold rules, severity levels, lifecycle management
- **Dashboards** — drag-and-drop real-time widgets bound to device telemetry
- **REST + WebSocket API** — for integration with external services

---

## When to Use ThingsBoard vs Kafka+Flink

| Criterion | Use ThingsBoard | Use Kafka+Flink |
|---|---|---|
| Sensor count | < 5,000 sensors | > 10,000 sensors |
| Time to first working system | Days (compose up) | Weeks (cluster tuning) |
| Ops team size | 1–3 engineers | Dedicated data-eng team |
| Dashboard needs | Built-in dashboards sufficient | Custom BI / advanced analytics |
| Feature extraction | Rule engine scripts (JS) adequate | Custom Java/Python transformations |
| ML inference latency | < 5 s acceptable | Sub-second, colocated with Flink |
| Multi-tenancy complexity | Low — single factory / SME | High — multi-tenant SaaS |
| Total cost of ownership | Lower (one process) | Higher (Kafka + ZK + Flink + connectors) |

**Rule of thumb:** Start with ThingsBoard for pilots and single-site deployments.
Migrate to Kafka+Flink when sensor count exceeds ~5,000 or when you need custom
feature-extraction logic that cannot be expressed in the JS rule engine.

---

## Architecture

```
 Industrial Machinery
 ┌─────────────────────────────────────┐
 │  Vibration Sensor (STM32 + ADXL357) │
 │  Electrical Monitor (ACS712)        │
 └───────────────┬─────────────────────┘
                 │ MQTT over TLS (port 8883)
                 │ topic: v1/devices/me/telemetry
                 ▼
 ┌───────────────────────────────────────────────────────┐
 │                  ThingsBoard CE 3.6                   │
 │                                                       │
 │  ┌────────────┐   ┌─────────────────────────────┐    │
 │  │  Device    │   │        Rule Engine           │    │
 │  │  Registry  │   │                              │    │
 │  │ (profiles, │   │  Filter (vibration only)     │    │
 │  │  tokens,   │   │      ↓                       │    │
 │  │  attrs)    │   │  Script (RMS, crest factor)  │    │
 │  └────────────┘   │      ↓                       │    │
 │                   │  Threshold check (ISO 10816) │    │
 │  ┌────────────┐   │      ↓                       │    │
 │  │ Dashboard  │◄──┤  Alarm creation / severity   │    │
 │  │  (HTTP UI) │   │      ↓                       │    │
 │  └────────────┘   │  Webhook → ML Engine         │    │
 │                   │      ↓                       │    │
 └───────────────────│  Email / SMS notification    │────┘
                     │      ↓                       │
                     │  Save to PostgreSQL           │
                     └─────────────────────────────-┘
                                   │
                 ┌─────────────────┴──────────────────┐
                 ▼                                     ▼
       VictoriaMetrics                            PostgreSQL
     (long-term metrics,                     (device metadata,
      PromQL queries)                          alarm history)
                 │
                 ▼
             Grafana
      (advanced analytics,
       multi-site comparison)
```

### Data Flow Summary

1. Sensors publish JSON telemetry over MQTT every 60 s (or on-demand waveform burst).
2. ThingsBoard rule engine filters by `sensor_type`, computes derived metrics
   (RMS, kurtosis, crest factor) via a JavaScript transformation node.
3. Alarm thresholds (ISO 10816 / ISO 20816) trigger 4-stage severity alarms.
4. Critical alarms POST to the external ML engine webhook for fault classification.
5. Notifications route by severity: email (WARNING), SMS (MAJOR), PagerDuty (CRITICAL).
6. All telemetry is archived to VictoriaMetrics via a lightweight sidecar exporter.
7. Grafana connects to VictoriaMetrics for historical trend analysis and multi-site views.

---

## Trade-offs vs Kafka+Flink

### Advantages of ThingsBoard

- **Faster deployment:** `docker compose up` in minutes vs multi-day Kafka cluster tuning.
- **Built-in dashboard:** no need to build a separate front-end for ops teams.
- **Alarm lifecycle management:** OPEN/ACK/CLEAR with audit trail out of the box.
- **Lower resource consumption:** a single JVM process vs Kafka + ZooKeeper + Flink cluster.
- **Device management UI:** provision, group, and configure devices without custom tooling.

### Limitations vs Kafka+Flink

- **Rule engine expressiveness:** JavaScript nodes cover most needs, but complex
  stateful joins (multi-sensor temporal alignment) require workarounds.
- **Throughput ceiling:** a single ThingsBoard node handles ~10,000 msgs/s;
  Kafka+Flink scales horizontally without architectural changes.
- **ML integration:** webhook callbacks add latency; Flink can colocate model
  inference for sub-second scoring.
- **Replay capability:** ThingsBoard has no built-in event-log replay.
  Kafka retains raw messages for re-processing with a new model version.
- **Vendor-specific rule chain format:** migrating processing logic to Flink later
  requires a re-implementation effort.

---

## Directory Structure

```
impl/pipeline_thingsboard/
├── README.md                         ← this file
├── docker-compose.yml                ← local dev stack
├── rules/
│   └── vibration_alert_rule_chain.json   ← ThingsBoard rule chain export
├── provision/
│   ├── provision_devices.py          ← REST API provisioning script
│   └── dashboard_template.json       ← Machine Health dashboard export
├── mqtt/
│   └── sensor_simulator.py           ← MQTT sensor data simulator
└── archival/
    └── victoria_exporter.py          ← VictoriaMetrics archival sidecar
```

---

## Quick Start

```bash
# 1. Start the stack
docker compose up -d

# 2. Wait for ThingsBoard to be ready (~60 s on first boot)
docker compose logs -f thingsboard | grep "Started ThingsBoard"

# 3. Provision devices and get access tokens
pip install requests
python provision/provision_devices.py --host localhost --port 8080

# 4. Run the sensor simulator (replace TOKEN with output from step 3)
pip install paho-mqtt
python mqtt/sensor_simulator.py \
    --tokens TOKEN_SENSOR_001,TOKEN_SENSOR_002 \
    --broker localhost --port 1883 \
    --sensors 10 --interval 60

# 5. Open ThingsBoard dashboard
open http://localhost:8080   # admin@thingsboard.org / sysadmin

# 6. Open Grafana
open http://localhost:3000   # admin / admin
```

---

## References

- ThingsBoard docs: https://thingsboard.io/docs/
- ISO 10816-3 vibration severity zones: Zone A (new) B (normal) C (warning) D (critical)
- ISO 20816-1 (updated 10816): https://www.iso.org/standard/63180.html
- VictoriaMetrics remote write: https://docs.victoriametrics.com/#how-to-import-data-in-prometheus-exposition-format
