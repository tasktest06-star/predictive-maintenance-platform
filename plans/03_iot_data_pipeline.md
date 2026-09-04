# Product 3: IoT Data Pipeline
> Sensor data ingestion, streaming, and storage infrastructure

---

## Product Overview

The data pipeline layer ingests sensor telemetry from thousands of devices, computes streaming features, routes data to the ML engine and storage systems, and provides the data foundation for the CMMS dashboard and cloud API. Built entirely on Apache 2.0 open-source components.

---

## Functional Requirements

### FR-PIPE-01: MQTT Ingestion
- Receive MQTT messages from sensor nodes (QoS 0/1/2)
- Support ≥50,000 concurrent device connections per broker cluster
- Message topics: `sensors/{device_id}/telemetry`, `sensors/{device_id}/alerts`, `sensors/{device_id}/raw_waveform`
- TLS 1.3 with mutual certificate authentication
- Message payload: CBOR or JSON, <4 KB typical, up to 256 KB for raw waveform bursts

### FR-PIPE-02: Event Streaming
- All sensor messages published to Kafka topics after MQTT ingestion
- Topic partitioning by device ID for ordered per-device processing
- Retention: 7 days raw messages; 90 days feature events
- Exactly-once semantics for alert processing pipeline

### FR-PIPE-03: Stream Processing (Feature Computation)
- Sliding and tumbling windows: 1 min, 5 min, 1 hour, 24 hours
- Per-window aggregations: RMS, peak, mean, std-dev, kurtosis, crest factor
- Bearing frequency magnitude tracking per window
- RPM-normalized features for variable-speed machines
- Multi-sensor join: align vibration + temperature + AE by timestamp
- Anomaly score streaming from deployed ML model (via model serving endpoint)

### FR-PIPE-04: Time-Series Storage
- Write all sensor features to time-series database at ≥100,000 writes/second
- Configurable retention per metric type (30 days hot / 365 days cold / 7 years archive)
- PromQL-compatible query interface for dashboard and API
- Long-term storage: cold tier on S3-compatible object storage

### FR-PIPE-05: Raw Waveform Storage
- Store raw time-domain waveform captures (triggered by anomaly or scheduled)
- Object storage (S3): `waveforms/{device_id}/{timestamp}/x_axis.bin`
- Retention: 90 days default; indefinite for confirmed fault events
- Indexed metadata in PostgreSQL for efficient query

### FR-PIPE-06: Alert Pipeline
- Real-time alert processing: sensor anomaly → alert event → notification
- Deduplication window: suppress duplicate alerts from same device within 15 minutes
- Severity routing: P1 (critical) → immediate PagerDuty/SMS; P2 → email; P3 → dashboard only
- Alert state machine: OPEN → ACKNOWLEDGED → RESOLVED

### FR-PIPE-07: Data Quality and Monitoring
- Detect and flag: missing device heartbeats, sensor calibration drift, outlier spikes
- Data lineage tracking per feature value
- OpenTelemetry metrics for pipeline health (throughput, lag, error rates)
- Grafana dashboards for operator pipeline observability

---

## Non-Functional Requirements

### NFR-PIPE-01: Throughput
- Ingest: ≥50,000 sensor messages/second sustained
- Feature computation: end-to-end latency <5 seconds (sensor publish to feature store)
- Alert latency: sensor publish to notification <30 seconds

### NFR-PIPE-02: Availability
- Pipeline SLA: 99.9% uptime (8.7 hours downtime/year)
- No single point of failure: all components clustered
- Sensor messages buffered on device for 24h if pipeline unavailable

### NFR-PIPE-03: Scalability
- Horizontal scaling: add Kafka partitions + Flink task managers to scale with device count
- Cold path storage (S3) scales independently

### NFR-PIPE-04: Security
- mTLS everywhere: MQTT broker, Kafka, Flink, API endpoints
- Kafka ACLs: per-topic authorization by service identity
- Data encryption at rest: AES-256 for object storage, encrypted volumes for databases

---

## Technical Architecture

```
Sensor Nodes (MQTT publish)
         ↓ TLS MQTT
HiveMQ CE / VerneMQ Cluster (MQTT Broker)
         ↓ Kafka Producer
Apache Kafka Cluster (event bus)
    Topic: sensor.telemetry
    Topic: sensor.alerts
    Topic: sensor.waveforms
         ↓
Apache Flink (stream processor)
    Job 1: Feature aggregation (windowed RMS, kurtosis, crest factor)
    Job 2: ML anomaly scoring (calls ML serving gRPC endpoint)
    Job 3: Alert correlation and deduplication
         ↓                    ↓                  ↓
VictoriaMetrics        PostgreSQL           S3 Object Store
(metrics store)        (metadata, alerts)   (raw waveforms)
         ↓
Cloud Platform API (Product 5)
```

### Component Choices

| Component | Choice | Rationale |
|---|---|---|
| MQTT Broker | HiveMQ CE (primary) or VerneMQ (alternative) | Apache 2.0; enterprise IoT; Erlang/Java; TLS + ACL built-in |
| Event bus | Apache Kafka | Industry standard; proven at sensor scale; replay capability |
| Stream processor | Apache Flink | Sub-second windows; event-time; exactly-once; Kafka connector |
| Time-series DB | VictoriaMetrics | Higher write throughput than InfluxDB; PromQL API; Apache 2.0 |
| Metadata DB | PostgreSQL | ACID; full-text search; extensible; wide support |
| Object storage | MinIO (on-prem) or S3 (cloud) | Apache 2.0 MinIO; S3-compatible API |
| Observability | OpenTelemetry Collector → VictoriaMetrics | Vendor-neutral; Apache 2.0 |
| Orchestration | Kubernetes (K8s) | Standard container orchestration |

### Flink Feature Extraction Job (Java/Scala)
```java
DataStream<SensorEvent> raw = env.fromKafka("sensor.telemetry");

raw.keyBy(SensorEvent::getDeviceId)
   .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1)))
   .aggregate(new VibrationFeatureAggregator())  // RMS, kurtosis, crest, bearing freqs
   .addSink(new VictoriaMetricsSink());
```

---

## Implementation Phases

### Phase 0: Local Dev Stack (Month 1–2)
- [ ] Docker Compose: Kafka + ZooKeeper + HiveMQ CE + VictoriaMetrics + Grafana
- [ ] MQTT test publisher simulating 100 sensors
- [ ] Kafka consumer logging messages to VictoriaMetrics
- [ ] Validate end-to-end from MQTT publish to metric query

### Phase 1: MVP Pipeline (Months 3–5)
- [ ] Flink Feature Job: 1-min window RMS + kurtosis + crest factor
- [ ] Alert pipeline: threshold-based alerts → PostgreSQL + email notify
- [ ] S3 waveform storage with metadata index
- [ ] Kubernetes deployment (Helm charts for each component)
- [ ] HiveMQ CE → Kafka bridge with mTLS

### Phase 2: Production Hardening (Months 6–9)
- [ ] Multi-node Kafka cluster (3 brokers, replication factor 3)
- [ ] Flink checkpointing (exactly-once); auto-recovery from failure
- [ ] VictoriaMetrics cluster mode (vmstorage + vminsert + vmselect)
- [ ] OpenTelemetry pipeline health metrics
- [ ] Load test: 50,000 msgs/second sustained

### Phase 3: Advanced Features (Months 10–12)
- [ ] ML anomaly scoring in Flink (gRPC call to ML serving)
- [ ] Multi-sensor temporal join (vibration + temp + AE aligned by device + timestamp)
- [ ] Adaptive retention tiering (hot → cold → archive)
- [ ] Data replay capability (re-process historical Kafka topics with new model version)

---

## Key GitHub Repos

| Repo | Use |
|---|---|
| [hivemq/hivemq-community-edition](https://github.com/hivemq/hivemq-community-edition) | MQTT broker |
| [vernemq/vernemq](https://github.com/vernemq/vernemq) | Alternative MQTT broker |
| [apache/kafka](https://github.com/apache/kafka) | Event streaming bus |
| [apache/flink](https://github.com/apache/flink) | Stream processing + feature extraction |
| [VictoriaMetrics/VictoriaMetrics](https://github.com/VictoriaMetrics/VictoriaMetrics) | Time-series database |
| [open-telemetry/opentelemetry-collector](https://github.com/open-telemetry/opentelemetry-collector) | Pipeline observability |
| [apache/nifi](https://github.com/apache/nifi) | Alternative: visual ETL routing |
| [thingsboard/thingsboard](https://github.com/thingsboard/thingsboard) | Alternative: all-in-one IoT platform |

---

## Key Academic Papers

| arXiv | Relevance |
|---|---|
| 2411.07168 | Hierarchical edge/gateway/cloud architecture — informs tiered processing design |
| 2211.09406 | Federated multi-factory learning — informs multi-tenant data isolation |
| 2605.07860 | On-device generative models — informs where computation should live in the stack |

---

## Success Metrics

| Metric | Target |
|---|---|
| Sustained ingest throughput | ≥50,000 msgs/sec |
| Feature computation latency | <5 seconds end-to-end |
| Alert latency | <30 seconds |
| Pipeline availability | 99.9% |
| Kafka message loss | 0 (exactly-once) |
| Cold storage cost | <$0.01/GB/month |
