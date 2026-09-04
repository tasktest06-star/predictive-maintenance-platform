# Predictive Maintenance Platform — Product Portfolio Overview
> TRACTIAN-class competitive system | September 2026

---

## Strategic Context

TRACTIAN raised $120M Series C and serves 1,500+ manufacturers with an "Industrial Copilot" combining wireless condition monitoring sensors + AI diagnostics + integrated CMMS. Their core moat: 3.5B training samples, patented sensor design eliminating battery resonance artifacts, and a bundled CMMS that incumbent players (SKF, Emerson, Fluke) cannot offer.

This portfolio defines seven discrete products that together constitute a competitive alternative. Each has its own requirements and implementation plan.

---

## Product Portfolio

| # | Product | Layer | Primary Differentiator Target |
|---|---|---|---|
| 1 | [Smart Sensor Hardware](./01_smart_sensor_hardware.md) | Physical | True AE sensing (100–400 kHz), magnetic mount option, open hardware |
| 2 | [Edge Firmware](./02_edge_firmware.md) | Embedded | Hierarchical on-device inference, power-aware sampling |
| 3 | [IoT Data Pipeline](./03_iot_data_pipeline.md) | Infrastructure | Open-source Kafka/Flink stack, no vendor lock-in |
| 4 | [ML/AI Diagnostic Engine](./04_ml_ai_engine.md) | AI | Explainable AI + uncertainty quantification, few-shot new fault adaptation |
| 5 | [Cloud Platform & API](./05_cloud_platform.md) | Backend | Multi-tenant, FedRAMP-ready, open integrations |
| 6 | [CMMS & Monitoring Dashboard](./06_cmms_dashboard.md) | Frontend | Real-time spectrum display, mobile-first, offline-capable |
| 7 | [Electrical Monitoring](./07_electrical_monitoring.md) | Physical | Non-invasive multi-phase current + power quality monitoring |

---

## Cross-Cutting Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  FIELD LAYER                                                     │
│  Product 1: Smart Sensor  ←→  Product 7: Electrical Monitor     │
│  (vibration + acoustic + temp + RPM)   (current + voltage)      │
└──────────────────┬──────────────────────────────────────────────┘
                   │ 915 MHz / 4G-LTE / BLE (MQTT, AES-256)
┌──────────────────▼──────────────────────────────────────────────┐
│  EDGE/GATEWAY LAYER                                              │
│  Product 2: Edge Firmware (on-sensor MCU inference)             │
└──────────────────┬──────────────────────────────────────────────┘
                   │ MQTT over TLS
┌──────────────────▼──────────────────────────────────────────────┐
│  INGESTION LAYER                                                 │
│  Product 3: IoT Data Pipeline (Kafka + Flink + VictoriaMetrics) │
└──────────────────┬──────────────────────────────────────────────┘
                   │ gRPC / REST
┌──────────────────▼──────────────────────────────────────────────┐
│  INTELLIGENCE LAYER                                             │
│  Product 4: ML/AI Diagnostic Engine (AutoDiagnosis equivalent)  │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│  PLATFORM LAYER                                                  │
│  Product 5: Cloud Platform & API                                 │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│  USER LAYER                                                      │
│  Product 6: CMMS & Monitoring Dashboard (web + mobile)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Development Phases (Cross-Product)

**Phase 0 — Foundation (Months 1–3)**
- Set up Kafka + VictoriaMetrics + MQTT broker infrastructure
- Prototype sensor hardware with COTS components (evaluation board)
- Establish ML experimentation environment (MLflow + Airflow)
- Begin CWRU/MFPT/MIMII dataset training for baseline models

**Phase 1 — MVP (Months 4–9)**
- First sensor PCB revision; edge firmware with FFT + basic threshold
- VAE-based anomaly detection deployed to cloud
- ThingsBoard dashboard for sensor telemetry + alert display
- Basic CMMS work order generation on fault detection

**Phase 2 — Beta (Months 10–15)**
- Full fault classification (bearing, unbalance, misalignment)
- Multi-modal fusion (vibration + acoustic + temp)
- Customer pilot with 20–50 sensors across 2–3 sites
- Electrical monitoring hardware MVP

**Phase 3 — Commercial (Months 16–24)**
- ATEX/IECEx certification for sensor hardware
- FedRAMP authorization for cloud platform
- SAP/IBM Maximo integration connectors
- Enterprise CMMS features (SSO, custom KPI dashboards)
- Sub-GHz mesh gateway product

---

## Key Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| Sensing | MEMS accelerometer + MEMS microphone + AE transducer | TRACTIAN uses vibration+mic; adding true AE (100–400 kHz) as differentiator |
| MCU | ARM Cortex-M33/M55 (e.g., STM32H7, Nordic nRF9160) | Cortex-M55 includes Helium SIMD for DSP; nRF9160 includes cellular modem |
| Radio | Sub-GHz (915 MHz, IEEE 802.15.4g) + LTE-M/NB-IoT | Matches TRACTIAN's proven wall-penetration advantage; LTE-M for remote assets |
| Edge ML | TensorFlow Lite Micro + CMSIS-NN | Established ecosystem; CMSIS-NN accelerates INT8 ops on Cortex-M |
| Streaming | Apache Kafka | Industry standard; proven at sensor data scale |
| Stream processing | Apache Flink | Sub-second windowed feature extraction at scale |
| Time-series DB | VictoriaMetrics | High write throughput; Prometheus-compatible; lower cost than InfluxDB Cloud |
| ML training | PyTorch + MLflow + Airflow | De facto industrial ML stack |
| Dashboard | React + Apache ECharts + ThingsBoard | ECharts for spectrum visualization; ThingsBoard for IoT device management |
| CMMS | Custom (React + PostgreSQL) | No compelling Apache-licensed CMMS exists; build for differentiation |
