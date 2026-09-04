#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# topics.sh — Create all Kafka topics for the PDM pipeline
#
# Usage:
#   ./kafka/topics.sh [BOOTSTRAP_SERVER]
#
# Default bootstrap server: localhost:9092
# Run after the Kafka cluster is healthy.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BOOTSTRAP="${1:-localhost:9092}"
KAFKA_CMD="kafka-topics"

# Check connectivity
echo "[topics.sh] Waiting for Kafka broker at ${BOOTSTRAP}..."
until ${KAFKA_CMD} --bootstrap-server "${BOOTSTRAP}" --list &>/dev/null; do
    sleep 2
done
echo "[topics.sh] Kafka is ready."

# Helper function
create_topic() {
    local topic="$1"
    local partitions="$2"
    local replication="$3"
    shift 3
    local extra_configs=("$@")

    if ${KAFKA_CMD} --bootstrap-server "${BOOTSTRAP}" --describe --topic "${topic}" &>/dev/null 2>&1; then
        echo "[topics.sh] Topic '${topic}' already exists — skipping."
        return 0
    fi

    echo "[topics.sh] Creating topic '${topic}' (partitions=${partitions}, replication=${replication})"
    local cmd=(
        "${KAFKA_CMD}"
        --bootstrap-server "${BOOTSTRAP}"
        --create
        --topic "${topic}"
        --partitions "${partitions}"
        --replication-factor "${replication}"
    )

    for cfg in "${extra_configs[@]}"; do
        cmd+=(--config "${cfg}")
    done

    "${cmd[@]}"
    echo "[topics.sh]   -> Created."
}

# ──────────────────────────────────────────────────────────────────────────────
# sensor.telemetry
#   Raw sensor telemetry: vibration XYZ, temperature, RPM from all devices.
#   100 partitions — keyed by device_id hash for ordered per-device processing.
#   7-day retention; lz4 compression; max.message.bytes 256KB for waveform bursts.
# ──────────────────────────────────────────────────────────────────────────────
create_topic "sensor.telemetry" 100 3 \
    "retention.ms=604800000" \
    "compression.type=lz4" \
    "max.message.bytes=262144" \
    "min.insync.replicas=2" \
    "unclean.leader.election.enable=false" \
    "message.timestamp.type=LogAppendTime"

# ──────────────────────────────────────────────────────────────────────────────
# sensor.alerts
#   Processed alert events. Compacted: one latest alert per device+fault_type key.
#   30-day retention; low partition count — alert volume is orders of magnitude
#   lower than telemetry.
# ──────────────────────────────────────────────────────────────────────────────
create_topic "sensor.alerts" 20 3 \
    "retention.ms=2592000000" \
    "cleanup.policy=compact" \
    "min.compaction.lag.ms=3600000" \
    "max.compaction.lag.ms=43200000" \
    "min.insync.replicas=2" \
    "unclean.leader.election.enable=false" \
    "compression.type=lz4"

# ──────────────────────────────────────────────────────────────────────────────
# sensor.waveforms
#   Raw time-domain waveform captures (triggered by anomaly or scheduled).
#   Large messages (up to 256 KB). 3-day retention (cold path → S3 after ingest).
#   Replication=2 (cost vs. durability tradeoff; S3 is the durable store).
# ──────────────────────────────────────────────────────────────────────────────
create_topic "sensor.waveforms" 50 2 \
    "retention.ms=259200000" \
    "compression.type=lz4" \
    "max.message.bytes=268435456" \
    "min.insync.replicas=1" \
    "segment.bytes=536870912"

# ──────────────────────────────────────────────────────────────────────────────
# sensor.features
#   Computed features from Flink (RMS, kurtosis, bearing freqs, crest factor).
#   90-day retention — feeds ML training and retrospective analysis.
#   Same partition count as telemetry so Flink can write with same key routing.
# ──────────────────────────────────────────────────────────────────────────────
create_topic "sensor.features" 100 3 \
    "retention.ms=7776000000" \
    "compression.type=lz4" \
    "min.insync.replicas=2" \
    "unclean.leader.election.enable=false"

# ──────────────────────────────────────────────────────────────────────────────
# sensor.diagnostics
#   Long-lived diagnostic snapshots: health scores, calibration state, ML
#   model inferences, maintenance event timestamps.
#   365-day retention; compacted+delete hybrid policy.
# ──────────────────────────────────────────────────────────────────────────────
create_topic "sensor.diagnostics" 20 3 \
    "retention.ms=31536000000" \
    "cleanup.policy=compact,delete" \
    "min.compaction.lag.ms=86400000" \
    "min.insync.replicas=2" \
    "unclean.leader.election.enable=false" \
    "compression.type=lz4"

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[topics.sh] All topics created. Current topic list:"
${KAFKA_CMD} --bootstrap-server "${BOOTSTRAP}" --list | grep "sensor\." | sort
echo ""
echo "[topics.sh] Done."
