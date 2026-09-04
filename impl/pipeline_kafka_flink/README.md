# Kafka + Apache Flink Streaming Pipeline

## Architecture Overview

```
Sensor Nodes (50,000+ concurrent)
         │  MQTT over TLS 1.3
         ▼
┌─────────────────────┐
│   HiveMQ CE Cluster  │  MQTT broker — handles device auth, QoS 0/1/2,
│   port 1883 / 8883   │  keep-alive, last-will messages
└────────┬────────────┘
         │  hivemq-kafka-extension (built-in bridge)
         ▼
┌─────────────────────────────────────────────────────┐
│            Apache Kafka — 3-broker cluster           │
│                                                      │
│  sensor.telemetry   (100 partitions, 7d retention)  │
│  sensor.waveforms   ( 50 partitions, 3d retention)  │
│  sensor.features    (100 partitions, 90d retention) │
│  sensor.alerts      ( 20 partitions, 30d retention) │
│  sensor.diagnostics ( 20 partitions, 1yr retention) │
└──────┬──────────────────────────────────────────────┘
       │                         │
       ▼                         ▼
┌────────────────┐      ┌────────────────────┐
│  Flink Job 1   │      │   Flink Job 2       │
│  Feature       │      │   Alert Detection   │
│  Extraction    │      │   + CEP Dedup       │
│  (5min slide)  │      │   (15min suppress)  │
└──────┬─────────┘      └──────┬─────────────┘
       │                       │
       ▼                       ▼
┌──────────────────┐   ┌───────────────┐   ┌──────────────┐
│  VictoriaMetrics │   │  Kafka        │   │  PostgreSQL  │
│  Time-series DB  │   │  sensor.alerts│   │  Alert state │
│  (PromQL API)    │   │               │   │  machine     │
└──────────────────┘   └───────────────┘   └──────────────┘
       │
       ▼
   Grafana Dashboards + Cloud Platform API (Product 5)
```

## Why Kafka + Flink Over ThingsBoard

### ThingsBoard Limitations at Scale

ThingsBoard is an all-in-one IoT platform that bundles MQTT, rule engine, storage, and
dashboards into a single application. This is convenient for small deployments (hundreds of
devices) but creates hard ceilings for the TRACTIAN-scale use case:

| Concern | ThingsBoard | Kafka + Flink |
|---|---|---|
| MQTT concurrency | ~5,000 per node; horizontal scaling is paid (PE/Cloud only) | HiveMQ CE handles 50,000+ per node; cluster-ready in CE |
| Stream processing | Embedded JS/Go rule chains, no true stateful windowing | Flink: event-time, stateful, exactly-once, 5ms-latency windows |
| Throughput ceiling | ~10,000 msgs/sec (Cassandra write bound) | Kafka: millions of msgs/sec; decoupled from processing speed |
| Feature computation | Threshold rules only; no spectral/bearing frequency analysis | Custom `AggregateFunction` — RMS, kurtosis, BPFO/BPFI/BSF/FTF |
| Data replay | Not supported — cannot re-process historical data | Kafka retention allows full pipeline replay from any offset |
| ML integration | HTTP callback only; no low-latency model serving path | Flink async I/O → gRPC model serving; sub-second scoring |
| Ops footprint | Single Java monolith (simpler) | Multiple services (higher ops complexity) |

**Bottom line:** ThingsBoard is the right choice for a team deploying 500 sensors quickly.
Kafka + Flink is the right choice when you are signing contracts with large industrial
customers (automotive plants, mining, steel mills) where 10,000–100,000 sensors per site
is normal.

### Sub-second Latency Path

```
Sensor publish → HiveMQ receive → Kafka produce → Flink watermark →
  Flink window trigger (1-min slide) → VictoriaMetrics write

Typical end-to-end: 800ms – 2s
Alert path (threshold breach): < 5s to Kafka sensor.alerts
```

## Tradeoffs

### Kafka + Flink Advantages
- **Durability and replay**: Kafka stores raw messages for 7 days; any pipeline bug can
  be corrected by replaying the topic with a fixed Flink job version.
- **Independent scaling**: Scale Kafka brokers, Flink task managers, and VictoriaMetrics
  storage nodes independently based on actual bottlenecks.
- **Exactly-once guarantees**: Flink + Kafka transactional producer ensures no duplicate
  feature records or double-fired alerts.
- **Multi-tenant isolation**: Kafka ACLs enforce per-customer topic namespacing with no
  cross-tenant data leakage.

### Kafka + Flink Disadvantages vs ThingsBoard
- **Operational complexity**: 6+ distinct services to deploy, monitor, and upgrade versus
  one ThingsBoard binary. Requires engineers familiar with JVM tuning, Kafka consumer
  group lag, and Flink checkpoint management.
- **Time to first demo**: ThingsBoard can be running with MQTT dashboards in an afternoon.
  This stack requires a week of infrastructure setup before a first sensor appears in
  Grafana.
- **Cost at small scale**: Running 3 Kafka brokers + Flink cluster + VictoriaMetrics costs
  ~$800–1,200/month in cloud compute even with zero sensors connected. ThingsBoard
  community edition runs on a single $80/month VM.

### Recommendation
Use ThingsBoard for pilot deployments (<1,000 sensors, single customer site). Migrate to
this Kafka + Flink stack when any of these triggers are hit:
- Sensor count exceeds 5,000
- Customer requests data replay or custom ML model integration
- Alert latency >10 seconds becomes a contractual requirement
- Multi-factory (multi-tenant) deployment is needed

## Component Versions

| Component | Version | License |
|---|---|---|
| Apache Kafka | 3.7 (Confluent images 7.x) | Apache 2.0 |
| Apache Flink | 1.18 | Apache 2.0 |
| HiveMQ CE | 2024.x | Apache 2.0 |
| VictoriaMetrics | 1.99 | Apache 2.0 |
| Grafana | 10.x | AGPL-3.0 |
| ZooKeeper | 3.8 (via Confluent) | Apache 2.0 |

## Quick Start (Local Dev)

```bash
# Start the full stack
docker compose up -d

# Wait for Kafka to be ready, then create topics
bash kafka/topics.sh

# Verify topics
docker exec kafka-1 kafka-topics --bootstrap-server localhost:9092 --list

# Submit Flink jobs
# (Package with Maven first)
mvn -f flink/pom.xml package -DskipTests
docker cp flink/target/pdm-pipeline-1.0.jar flink-jobmanager:/opt/flink/
docker exec flink-jobmanager flink run /opt/flink/pdm-pipeline-1.0.jar \
  --class com.pdm.pipeline.flink.SensorFeatureExtractionJob

# Access UIs
open http://localhost:8080   # HiveMQ web UI
open http://localhost:8081   # Flink Dashboard
open http://localhost:3000   # Grafana (admin/admin)
open http://localhost:8428   # VictoriaMetrics
open http://localhost:8082   # Kafka UI
```
