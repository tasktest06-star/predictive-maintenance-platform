package com.pdm.pipeline.flink;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.PatternSelectFunction;
import org.apache.flink.cep.PatternStream;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.jdbc.JdbcConnectionOptions;
import org.apache.flink.connector.jdbc.JdbcExecutionOptions;
import org.apache.flink.connector.jdbc.JdbcSink;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * AlertJob
 *
 * Reads computed feature vectors from Kafka topic {@code sensor.features},
 * applies configurable alert rules (threshold, trend, bearing frequency),
 * deduplicates alerts using Flink CEP within a 15-minute suppression window,
 * and writes alerts to:
 *   1. Kafka topic {@code sensor.alerts}
 *   2. PostgreSQL alerts table (via JDBC sink)
 *
 * Alert Severity:
 *   P1 (CRITICAL) — immediate fan-out via PagerDuty/SMS (handled by downstream consumer)
 *   P2 (WARNING)  — email notification
 *   P3 (INFO)     — dashboard only
 */
public class AlertJob {

    static final String KAFKA_BROKERS   = System.getenv().getOrDefault(
            "KAFKA_BROKERS", "kafka-1:29092,kafka-2:29093,kafka-3:29094");
    static final String INPUT_TOPIC     = "sensor.features";
    static final String OUTPUT_TOPIC    = "sensor.alerts";
    static final String CONSUMER_GROUP  = "flink-alert-evaluator";
    static final String POSTGRES_URL    = System.getenv().getOrDefault(
            "POSTGRES_URL", "jdbc:postgresql://postgres:5432/pdm");
    static final String POSTGRES_USER   = System.getenv().getOrDefault("POSTGRES_USER", "pdm");
    static final String POSTGRES_PASS   = System.getenv().getOrDefault("POSTGRES_PASS", "pdm_dev_secret");

    // Alert rule thresholds (in production these are loaded from PostgreSQL)
    static final double RMS_CRITICAL_THRESHOLD = 15.0;   // m/s² — P1
    static final double RMS_WARNING_THRESHOLD  = 8.0;    // m/s² — P2
    static final double KURTOSIS_THRESHOLD     = 6.0;    // >6 indicates impulsive fault
    static final double CREST_FACTOR_THRESHOLD = 6.5;    // bearing defect indicator
    static final double BPFO_SIGMA_MULTIPLIER  = 3.0;    // 3σ above rolling baseline
    static final double KURTOSIS_TREND_INCREASE = 0.20;  // 20% increase over 30 min
    static final long   SUPPRESSION_WINDOW_MS  = 900_000L; // 15 minutes

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment();

        env.enableCheckpointing(30_000L);
        env.getCheckpointConfig().setCheckpointingMode(
                org.apache.flink.streaming.api.CheckpointingMode.EXACTLY_ONCE);

        // ── Kafka Source ───────────────────────────────────────────────────
        KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setTopics(INPUT_TOPIC)
                .setGroupId(CONSUMER_GROUP)
                .setStartingOffsets(OffsetsInitializer.latest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .setProperty("enable.auto.commit", "false")
                .setProperty("isolation.level", "read_committed")
                .build();

        // ── Parse features ─────────────────────────────────────────────────
        DataStream<FeatureEvent> features = env
                .fromSource(kafkaSource,
                        WatermarkStrategy.<String>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                                .withTimestampAssigner((json, ts) -> parseTimestamp(json)),
                        "Kafka sensor.features")
                .map(new ParseFeatureEventFn())
                .filter(f -> f != null);

        KeyedStream<FeatureEvent, String> keyedFeatures =
                features.keyBy(f -> f.deviceId);

        // ── Alert Rule Evaluation ──────────────────────────────────────────
        DataStream<AlertEvent> rawAlerts = keyedFeatures
                .process(new AlertRuleEvaluator());

        // ── CEP Deduplication — suppress same device+fault_type for 15 min ─
        KeyedStream<AlertEvent, String> keyedAlerts =
                rawAlerts.keyBy(a -> a.deviceId + ":" + a.faultType);

        DataStream<AlertEvent> dedupedAlerts = keyedAlerts
                .process(new AlertDeduplicationFunction(SUPPRESSION_WINDOW_MS));

        // ── Serialize alerts to JSON ───────────────────────────────────────
        DataStream<String> alertJson = dedupedAlerts
                .map(new AlertToJsonFn());

        // ── Sink 1: Kafka sensor.alerts ────────────────────────────────────
        KafkaSink<String> kafkaSink = KafkaSink.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setRecordSerializer(
                        KafkaRecordSerializationSchema.builder()
                                .setTopic(OUTPUT_TOPIC)
                                .setValueSerializationSchema(new SimpleStringSchema())
                                .build())
                .setDeliverGuarantee(
                        org.apache.flink.connector.kafka.sink.DeliveryGuarantee.EXACTLY_ONCE)
                .setTransactionalIdPrefix("flink-alert-")
                .build();

        alertJson.sinkTo(kafkaSink);

        // ── Sink 2: PostgreSQL ─────────────────────────────────────────────
        dedupedAlerts.addSink(
                JdbcSink.sink(
                        "INSERT INTO alerts (device_id, fault_type, severity, message, " +
                        "rms_x, rms_z, kurtosis_x, bpfo_magnitude, event_ts, created_at) " +
                        "VALUES (?,?,?,?,?,?,?,?,to_timestamp(?/1000.0),NOW()) " +
                        "ON CONFLICT DO NOTHING",
                        (stmt, alert) -> {
                            stmt.setString(1, alert.deviceId);
                            stmt.setString(2, alert.faultType);
                            stmt.setString(3, alert.severity);
                            stmt.setString(4, alert.message);
                            stmt.setDouble(5, alert.rmsX);
                            stmt.setDouble(6, alert.rmsZ);
                            stmt.setDouble(7, alert.kurtosisX);
                            stmt.setDouble(8, alert.bpfoMagnitude);
                            stmt.setLong(9, alert.eventTs);
                        },
                        JdbcExecutionOptions.builder()
                                .withBatchSize(100)
                                .withBatchIntervalMs(1000)
                                .withMaxRetries(3)
                                .build(),
                        new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                                .withUrl(POSTGRES_URL)
                                .withDriverName("org.postgresql.Driver")
                                .withUsername(POSTGRES_USER)
                                .withPassword(POSTGRES_PASS)
                                .build()
                )
        );

        env.execute("PDM AlertJob");
    }

    // ── Utility: extract timestamp from raw JSON without full parse ────────

    static long parseTimestamp(String json) {
        try {
            int idx = json.indexOf("\"window_end_ms\":");
            if (idx < 0) return System.currentTimeMillis();
            int start = idx + 16;
            int end = json.indexOf(',', start);
            if (end < 0) end = json.indexOf('}', start);
            return Long.parseLong(json.substring(start, end).trim());
        } catch (Exception e) {
            return System.currentTimeMillis();
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Data Models
    // ══════════════════════════════════════════════════════════════════════════

    public static class FeatureEvent {
        public String deviceId;
        public long   windowEndMs;
        public double rmsX, rmsY, rmsZ;
        public double peakX, peakY, peakZ;
        public double kurtosisX, kurtosisY, kurtosisZ;
        public double crestFactorX, crestFactorY, crestFactorZ;
        public double temperatureAvg;
        public double rpmAvg;
        public double bpfoMagnitude, bpfiMagnitude, bsfMagnitude, ftfMagnitude;
    }

    public static class AlertEvent {
        public String deviceId;
        public String faultType;   // e.g. "HIGH_RMS_Z", "BEARING_BPFO", "KURTOSIS_TREND"
        public String severity;    // "P1", "P2", "P3"
        public String message;
        public double rmsX, rmsZ;
        public double kurtosisX;
        public double bpfoMagnitude;
        public long   eventTs;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // ParseFeatureEventFn
    // ══════════════════════════════════════════════════════════════════════════

    public static class ParseFeatureEventFn implements MapFunction<String, FeatureEvent> {
        private static final ObjectMapper mapper = new ObjectMapper();

        @Override
        public FeatureEvent map(String json) {
            try {
                JsonNode root = mapper.readTree(json);
                FeatureEvent f = new FeatureEvent();
                f.deviceId       = root.get("device_id").asText();
                f.windowEndMs    = root.get("window_end_ms").asLong();
                f.rmsX           = root.path("rms_x").asDouble();
                f.rmsY           = root.path("rms_y").asDouble();
                f.rmsZ           = root.path("rms_z").asDouble();
                f.peakX          = root.path("peak_x").asDouble();
                f.peakY          = root.path("peak_y").asDouble();
                f.peakZ          = root.path("peak_z").asDouble();
                f.kurtosisX      = root.path("kurtosis_x").asDouble();
                f.kurtosisY      = root.path("kurtosis_y").asDouble();
                f.kurtosisZ      = root.path("kurtosis_z").asDouble();
                f.crestFactorX   = root.path("crest_factor_x").asDouble();
                f.crestFactorY   = root.path("crest_factor_y").asDouble();
                f.crestFactorZ   = root.path("crest_factor_z").asDouble();
                f.temperatureAvg = root.path("temperature_avg").asDouble();
                f.rpmAvg         = root.path("rpm_avg").asDouble();
                f.bpfoMagnitude  = root.path("bpfo_magnitude").asDouble();
                f.bpfiMagnitude  = root.path("bpfi_magnitude").asDouble();
                f.bsfMagnitude   = root.path("bsf_magnitude").asDouble();
                f.ftfMagnitude   = root.path("ftf_magnitude").asDouble();
                return f;
            } catch (Exception e) {
                return null;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // AlertRuleEvaluator
    //
    // Stateful KeyedProcessFunction that evaluates three rule types:
    //
    // 1. THRESHOLD — instantaneous value exceeds static threshold
    //    - rms_z > 15 m/s²   → P1 HIGH_RMS_Z
    //    - rms_z > 8 m/s²    → P2 ELEVATED_RMS_Z
    //    - kurtosis_x > 6    → P2 HIGH_KURTOSIS_X (impulsive fault)
    //    - crest_factor_z > 6.5 → P2 HIGH_CREST_FACTOR
    //
    // 2. TREND — kurtosis increasing >20% over the last 30 minutes
    //    - Tracked using a ring buffer of kurtosis values in ValueState
    //
    // 3. BEARING_FREQUENCY — BPFO magnitude >3σ above rolling 24-hour baseline
    //    - Rolling mean and M2 maintained in state (Welford's algorithm)
    // ══════════════════════════════════════════════════════════════════════════

    public static class AlertRuleEvaluator
            extends KeyedProcessFunction<String, FeatureEvent, AlertEvent> {

        // State: kurtosis history for trend detection (ring buffer via MapState idx→value)
        private transient MapState<Integer, Double> kurtosisHistory;
        private transient ValueState<Integer>       kurtosisIdx;
        private transient ValueState<Long>          kurtosisOldestTs;

        // State: Welford's online algorithm for BPFO baseline
        private transient ValueState<Long>   bpfoN;
        private transient ValueState<Double> bpfoMean;
        private transient ValueState<Double> bpfoM2;

        private static final int KURTOSIS_HISTORY_SIZE = 30;  // 30 one-minute windows = 30 min

        @Override
        public void open(Configuration parameters) throws Exception {
            kurtosisHistory = getRuntimeContext().getMapState(
                    new MapStateDescriptor<>("kurtosis-history", Types.INT, Types.DOUBLE));
            kurtosisIdx = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("kurtosis-idx", Types.INT));
            kurtosisOldestTs = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("kurtosis-oldest-ts", Types.LONG));

            bpfoN    = getRuntimeContext().getState(new ValueStateDescriptor<>("bpfo-n",    Types.LONG));
            bpfoMean = getRuntimeContext().getState(new ValueStateDescriptor<>("bpfo-mean", Types.DOUBLE));
            bpfoM2   = getRuntimeContext().getState(new ValueStateDescriptor<>("bpfo-m2",   Types.DOUBLE));
        }

        @Override
        public void processElement(FeatureEvent f, Context ctx, Collector<AlertEvent> out)
                throws Exception {

            // ── Rule 1: Threshold — RMS ──────────────────────────────────
            if (f.rmsZ > RMS_CRITICAL_THRESHOLD) {
                out.collect(buildAlert(f, "HIGH_RMS_Z", "P1",
                        String.format("Critical vibration: rms_z=%.2f m/s² (threshold=%.1f)",
                                f.rmsZ, RMS_CRITICAL_THRESHOLD)));
            } else if (f.rmsZ > RMS_WARNING_THRESHOLD) {
                out.collect(buildAlert(f, "ELEVATED_RMS_Z", "P2",
                        String.format("Elevated vibration: rms_z=%.2f m/s² (threshold=%.1f)",
                                f.rmsZ, RMS_WARNING_THRESHOLD)));
            }

            // ── Rule 1: Threshold — Kurtosis ─────────────────────────────
            if (f.kurtosisX > KURTOSIS_THRESHOLD) {
                out.collect(buildAlert(f, "HIGH_KURTOSIS_X", "P2",
                        String.format("Impulsive fault indicator: kurtosis_x=%.2f (threshold=%.1f)",
                                f.kurtosisX, KURTOSIS_THRESHOLD)));
            }

            // ── Rule 1: Threshold — Crest Factor ─────────────────────────
            if (f.crestFactorZ > CREST_FACTOR_THRESHOLD) {
                out.collect(buildAlert(f, "HIGH_CREST_FACTOR_Z", "P2",
                        String.format("Crest factor elevated: crest_factor_z=%.2f (threshold=%.1f)",
                                f.crestFactorZ, CREST_FACTOR_THRESHOLD)));
            }

            // ── Rule 2: Kurtosis Trend ────────────────────────────────────
            updateKurtosisHistory(f.kurtosisX, f.windowEndMs);
            Double trendAlert = evaluateKurtosisTrend();
            if (trendAlert != null) {
                out.collect(buildAlert(f, "KURTOSIS_INCREASING_TREND", "P2",
                        String.format("Kurtosis trend: %.0f%% increase over 30 min (current=%.2f)",
                                trendAlert * 100, f.kurtosisX)));
            }

            // ── Rule 3: Bearing Frequency (BPFO 3σ) ──────────────────────
            updateBpfoBaseline(f.bpfoMagnitude);
            Long n     = bpfoN.value();
            Double mean = bpfoMean.value();
            Double m2   = bpfoM2.value();
            if (n != null && n > 30 && mean != null && m2 != null) {
                double variance = m2 / n;
                double sigma    = Math.sqrt(variance);
                double threshold = mean + BPFO_SIGMA_MULTIPLIER * sigma;
                if (f.bpfoMagnitude > threshold) {
                    out.collect(buildAlert(f, "BEARING_BPFO_ANOMALY", "P1",
                            String.format("BPFO magnitude %.4f exceeds baseline %.4f + 3σ (%.4f)",
                                    f.bpfoMagnitude, mean, sigma)));
                }
            }
        }

        private AlertEvent buildAlert(FeatureEvent f, String faultType,
                                       String severity, String message) {
            AlertEvent a = new AlertEvent();
            a.deviceId       = f.deviceId;
            a.faultType      = faultType;
            a.severity       = severity;
            a.message        = message;
            a.rmsX           = f.rmsX;
            a.rmsZ           = f.rmsZ;
            a.kurtosisX      = f.kurtosisX;
            a.bpfoMagnitude  = f.bpfoMagnitude;
            a.eventTs        = f.windowEndMs;
            return a;
        }

        private void updateKurtosisHistory(double kurtosis, long tsMs) throws Exception {
            Integer idx = kurtosisIdx.value();
            if (idx == null) idx = 0;
            kurtosisHistory.put(idx % KURTOSIS_HISTORY_SIZE, kurtosis);
            kurtosisIdx.update((idx + 1) % KURTOSIS_HISTORY_SIZE);
        }

        /**
         * Returns the fractional kurtosis increase if >20% over 30 min, else null.
         */
        private Double evaluateKurtosisTrend() throws Exception {
            double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
            int count = 0;
            for (Map.Entry<Integer, Double> entry : kurtosisHistory.entries()) {
                double v = entry.getValue();
                min = Math.min(min, v);
                max = Math.max(max, v);
                count++;
            }
            if (count < KURTOSIS_HISTORY_SIZE || min <= 0) return null;
            double increase = (max - min) / min;
            return (increase > KURTOSIS_TREND_INCREASE) ? increase : null;
        }

        /**
         * Welford's online algorithm for computing running mean and variance.
         */
        private void updateBpfoBaseline(double value) throws Exception {
            Long   n    = bpfoN.value();
            Double mean = bpfoMean.value();
            Double m2   = bpfoM2.value();
            if (n == null) { n = 0L; mean = 0.0; m2 = 0.0; }

            n++;
            double delta  = value - mean;
            mean += delta / n;
            double delta2 = value - mean;
            m2 += delta * delta2;

            bpfoN.update(n);
            bpfoMean.update(mean);
            bpfoM2.update(m2);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // AlertDeduplicationFunction
    //
    // Suppresses repeat alerts for the same device+fault_type within a
    // configurable time window (default 15 minutes).
    //
    // State: lastAlertTs per (deviceId, faultType) composite key.
    // A timer is registered to clear state after the suppression window expires.
    // ══════════════════════════════════════════════════════════════════════════

    public static class AlertDeduplicationFunction
            extends KeyedProcessFunction<String, AlertEvent, AlertEvent> {

        private final long suppressionWindowMs;
        private transient ValueState<Long> lastAlertTs;

        public AlertDeduplicationFunction(long suppressionWindowMs) {
            this.suppressionWindowMs = suppressionWindowMs;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            lastAlertTs = getRuntimeContext().getState(
                    new ValueStateDescriptor<>("last-alert-ts", Types.LONG));
        }

        @Override
        public void processElement(AlertEvent alert, Context ctx, Collector<AlertEvent> out)
                throws Exception {

            Long last = lastAlertTs.value();
            long now  = alert.eventTs;

            if (last == null || (now - last) >= suppressionWindowMs) {
                // First occurrence or suppression window has passed — emit
                out.collect(alert);
                lastAlertTs.update(now);

                // Register cleanup timer so state does not accumulate indefinitely
                ctx.timerService().registerEventTimeTimer(now + suppressionWindowMs + 1);
            }
            // else: duplicate within suppression window — drop silently
        }

        @Override
        public void onTimer(long timestamp, OnTimerContext ctx, Collector<AlertEvent> out)
                throws Exception {
            // Clear dedup state after suppression window expires
            lastAlertTs.clear();
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // AlertToJsonFn
    // ══════════════════════════════════════════════════════════════════════════

    public static class AlertToJsonFn implements MapFunction<AlertEvent, String> {
        private static final ObjectMapper mapper = new ObjectMapper();

        @Override
        public String map(AlertEvent a) throws Exception {
            ObjectNode node = mapper.createObjectNode();
            node.put("device_id",      a.deviceId);
            node.put("fault_type",     a.faultType);
            node.put("severity",       a.severity);
            node.put("message",        a.message);
            node.put("rms_x",          a.rmsX);
            node.put("rms_z",          a.rmsZ);
            node.put("kurtosis_x",     a.kurtosisX);
            node.put("bpfo_magnitude", a.bpfoMagnitude);
            node.put("event_ts",       a.eventTs);
            node.put("status",         "OPEN");
            return mapper.writeValueAsString(node);
        }
    }
}
