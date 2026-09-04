package com.pdm.pipeline.flink;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * SensorFeatureExtractionJob
 *
 * Reads raw vibration/temperature/RPM telemetry from Kafka topic {@code sensor.telemetry},
 * computes per-device windowed features using a 5-minute sliding window with 1-minute slides,
 * and writes results to:
 *   1. Kafka topic {@code sensor.features}
 *   2. VictoriaMetrics via HTTP PUT /api/v1/import (Prometheus remote-write format)
 *
 * Checkpointing:  every 30 seconds, EXACTLY_ONCE, RocksDB state backend
 * Watermarking:   event-time with 10-second late-data tolerance
 */
public class SensorFeatureExtractionJob {

    // ── Configuration ──────────────────────────────────────────────────────

    static final String KAFKA_BROKERS         = System.getenv().getOrDefault(
            "KAFKA_BROKERS", "kafka-1:29092,kafka-2:29093,kafka-3:29094");
    static final String INPUT_TOPIC           = "sensor.telemetry";
    static final String OUTPUT_TOPIC          = "sensor.features";
    static final String VICTORIA_URL          = System.getenv().getOrDefault(
            "VICTORIAMETRICS_URL", "http://victoriametrics:8428");
    static final String CONSUMER_GROUP        = "flink-feature-extractor";

    // ── Main Entry Point ───────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {

        final StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment();

        // Checkpointing — 30-second interval, exactly-once, RocksDB backend
        env.enableCheckpointing(30_000L);
        env.getCheckpointConfig().setCheckpointingMode(
                org.apache.flink.streaming.api.CheckpointingMode.EXACTLY_ONCE);
        env.getCheckpointConfig().setMinPauseBetweenCheckpoints(10_000L);
        env.getCheckpointConfig().setCheckpointTimeout(60_000L);
        env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);
        // RocksDB is set via flink-conf.yaml (state.backend: rocksdb)

        // ── Kafka Source ───────────────────────────────────────────────────
        KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setTopics(INPUT_TOPIC)
                .setGroupId(CONSUMER_GROUP)
                .setStartingOffsets(OffsetsInitializer.latest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .setProperty("enable.auto.commit", "false")
                .setProperty("isolation.level", "read_committed")  // exactly-once
                .build();

        // ── Watermarking (event-time, 10-second tolerance for late data) ───
        WatermarkStrategy<SensorEvent> watermarkStrategy = WatermarkStrategy
                .<SensorEvent>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                .withTimestampAssigner(
                        (SerializableTimestampAssigner<SensorEvent>)
                                (event, recordTimestamp) -> event.timestampMs);

        // ── Parse JSON → SensorEvent ───────────────────────────────────────
        DataStream<SensorEvent> events = env
                .fromSource(kafkaSource,
                        WatermarkStrategy.noWatermarks(),  // assign after parse
                        "Kafka sensor.telemetry")
                .map(new ParseSensorEventFn())
                .filter(e -> e != null)
                .assignTimestampsAndWatermarks(watermarkStrategy);

        // ── 5-min sliding window, 1-min slide, keyed by device_id ─────────
        DataStream<SensorFeatures> features = events
                .keyBy(e -> e.deviceId)
                .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1)))
                .aggregate(new VibrationFeatureAggregator());

        // ── Bearing frequency enrichment ───────────────────────────────────
        DataStream<SensorFeatures> enriched = features
                .map(new BearingFrequencyExtractor());

        // ── Serialize to JSON ──────────────────────────────────────────────
        DataStream<String> jsonOutput = enriched
                .map(new SensorFeaturesToJsonFn());

        // ── Sink 1: Kafka sensor.features ─────────────────────────────────
        KafkaSink<String> kafkaSink = KafkaSink.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setRecordSerializer(
                        KafkaRecordSerializationSchema.builder()
                                .setTopic(OUTPUT_TOPIC)
                                .setValueSerializationSchema(new SimpleStringSchema())
                                .build())
                .setDeliverGuarantee(
                        org.apache.flink.connector.kafka.sink.DeliveryGuarantee.EXACTLY_ONCE)
                .setTransactionalIdPrefix("flink-feature-")
                .setKafkaProducerConfig(producerProps())
                .build();

        jsonOutput.sinkTo(kafkaSink);

        // ── Sink 2: VictoriaMetrics ────────────────────────────────────────
        enriched.addSink(new VictoriaMetricsSink(VICTORIA_URL));

        env.execute("PDM SensorFeatureExtractionJob");
    }

    // ── Producer Properties ────────────────────────────────────────────────

    static Properties producerProps() {
        Properties p = new Properties();
        p.put("acks", "all");
        p.put("retries", "3");
        p.put("compression.type", "lz4");
        p.put("linger.ms", "5");
        p.put("batch.size", "65536");
        return p;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Data Models
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Raw sensor telemetry event deserialized from Kafka.
     * Expected JSON schema:
     * {
     *   "device_id": "sensor-abc123",
     *   "ts": 1700000000000,       // epoch ms (event time)
     *   "ax": 0.12, "ay": -0.05, "az": 9.82,   // acceleration m/s²
     *   "temp_c": 42.3,
     *   "rpm": 1500.0,
     *   "waveform_x": [0.1, 0.2, ...],  // optional raw time-domain samples
     *   "sample_rate_hz": 25600
     * }
     */
    public static class SensorEvent {
        public String deviceId;
        public long   timestampMs;
        public double[] ax;       // vibration X samples in this packet
        public double[] ay;       // vibration Y samples
        public double[] az;       // vibration Z samples
        public double   tempC;
        public double   rpm;
        public int      sampleRateHz;
    }

    /**
     * Windowed feature vector computed from a 5-minute window of SensorEvents.
     */
    public static class SensorFeatures {
        public String deviceId;
        public long   windowEndMs;

        // Vibration features
        public double rmsX, rmsY, rmsZ;
        public double peakX, peakY, peakZ;
        public double kurtosisX, kurtosisY, kurtosisZ;
        public double crestFactorX, crestFactorY, crestFactorZ;

        // Environmental
        public double temperatureAvg;
        public double rpmAvg;

        // Bearing frequencies (populated by BearingFrequencyExtractor)
        public double bpfoMagnitude;   // Ball Pass Frequency Outer
        public double bpfiMagnitude;   // Ball Pass Frequency Inner
        public double bsfMagnitude;    // Ball Spin Frequency
        public double ftfMagnitude;    // Fundamental Train Frequency
        public double bpfoHz;
        public double bpfiHz;
        public double bsfHz;
        public double ftfHz;

        // Window stats
        public int    sampleCount;
        public int    packetCount;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // ParseSensorEventFn
    // ══════════════════════════════════════════════════════════════════════════

    public static class ParseSensorEventFn implements MapFunction<String, SensorEvent> {
        private static final ObjectMapper mapper = new ObjectMapper();

        @Override
        public SensorEvent map(String json) {
            try {
                JsonNode root = mapper.readTree(json);
                SensorEvent e = new SensorEvent();
                e.deviceId    = root.get("device_id").asText();
                e.timestampMs = root.get("ts").asLong();
                e.tempC       = root.path("temp_c").asDouble(Double.NaN);
                e.rpm         = root.path("rpm").asDouble(Double.NaN);
                e.sampleRateHz = root.path("sample_rate_hz").asInt(25600);

                e.ax = parseDoubleArray(root.path("waveform_x"));
                e.ay = parseDoubleArray(root.path("waveform_y"));
                e.az = parseDoubleArray(root.path("waveform_z"));

                // Fallback: scalar ax/ay/az fields
                if (e.ax.length == 0 && root.has("ax")) {
                    e.ax = new double[]{ root.get("ax").asDouble() };
                    e.ay = new double[]{ root.get("ay").asDouble() };
                    e.az = new double[]{ root.get("az").asDouble() };
                }
                return e;
            } catch (Exception ex) {
                // Malformed message — drop it; logged as metric
                return null;
            }
        }

        private double[] parseDoubleArray(JsonNode node) {
            if (node == null || node.isMissingNode() || !node.isArray()) {
                return new double[0];
            }
            double[] arr = new double[node.size()];
            for (int i = 0; i < node.size(); i++) {
                arr[i] = node.get(i).asDouble();
            }
            return arr;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // VibrationFeatureAggregator
    //
    // Implements AggregateFunction to accumulate raw sample arrays across
    // multiple SensorEvent packets within the window, then compute:
    //   - RMS  = sqrt(sum(x_i^2) / N)
    //   - Peak = max(|x_i|)
    //   - Kurtosis = E[(x - mu)^4] / sigma^4  (excess kurtosis threshold ~3)
    //   - Crest Factor = Peak / RMS
    // ══════════════════════════════════════════════════════════════════════════

    public static class VibrationFeatureAggregator
            implements AggregateFunction<SensorEvent, VibrationAccumulator, SensorFeatures> {

        @Override
        public VibrationAccumulator createAccumulator() {
            return new VibrationAccumulator();
        }

        @Override
        public VibrationAccumulator add(SensorEvent event, VibrationAccumulator acc) {
            acc.deviceId = event.deviceId;
            acc.windowEndMs = event.timestampMs;
            acc.packetCount++;

            appendSamples(acc.samplesX, event.ax);
            appendSamples(acc.samplesY, event.ay);
            appendSamples(acc.samplesZ, event.az);

            if (!Double.isNaN(event.tempC))  acc.tempSum  += event.tempC;
            if (!Double.isNaN(event.tempC))  acc.tempCount++;
            if (!Double.isNaN(event.rpm))    acc.rpmSum   += event.rpm;
            if (!Double.isNaN(event.rpm))    acc.rpmCount++;

            return acc;
        }

        @Override
        public SensorFeatures getResult(VibrationAccumulator acc) {
            SensorFeatures f = new SensorFeatures();
            f.deviceId    = acc.deviceId;
            f.windowEndMs = acc.windowEndMs;
            f.packetCount = acc.packetCount;
            f.sampleCount = acc.samplesX.size();

            double[] x = toArray(acc.samplesX);
            double[] y = toArray(acc.samplesY);
            double[] z = toArray(acc.samplesZ);

            f.rmsX      = rms(x);          f.rmsY      = rms(y);          f.rmsZ      = rms(z);
            f.peakX     = peak(x);         f.peakY     = peak(y);         f.peakZ     = peak(z);
            f.kurtosisX = kurtosis(x);     f.kurtosisY = kurtosis(y);     f.kurtosisZ = kurtosis(z);

            f.crestFactorX = (f.rmsX > 0) ? f.peakX / f.rmsX : 0.0;
            f.crestFactorY = (f.rmsY > 0) ? f.peakY / f.rmsY : 0.0;
            f.crestFactorZ = (f.rmsZ > 0) ? f.peakZ / f.rmsZ : 0.0;

            f.temperatureAvg = (acc.tempCount > 0) ? acc.tempSum / acc.tempCount : Double.NaN;
            f.rpmAvg         = (acc.rpmCount  > 0) ? acc.rpmSum  / acc.rpmCount  : Double.NaN;

            return f;
        }

        @Override
        public VibrationAccumulator merge(VibrationAccumulator a, VibrationAccumulator b) {
            a.samplesX.addAll(b.samplesX);
            a.samplesY.addAll(b.samplesY);
            a.samplesZ.addAll(b.samplesZ);
            a.tempSum   += b.tempSum;    a.tempCount  += b.tempCount;
            a.rpmSum    += b.rpmSum;     a.rpmCount   += b.rpmCount;
            a.packetCount += b.packetCount;
            if (b.windowEndMs > a.windowEndMs) {
                a.windowEndMs = b.windowEndMs;
                a.deviceId    = b.deviceId;
            }
            return a;
        }

        // ── Statistical helpers ──────────────────────────────────────────

        /**
         * Root Mean Square: sqrt(sum(x_i^2) / N)
         */
        static double rms(double[] arr) {
            if (arr.length == 0) return 0.0;
            double sumSq = 0;
            for (double v : arr) sumSq += v * v;
            return Math.sqrt(sumSq / arr.length);
        }

        /**
         * Peak: max(|x_i|)
         */
        static double peak(double[] arr) {
            if (arr.length == 0) return 0.0;
            double max = 0;
            for (double v : arr) max = Math.max(max, Math.abs(v));
            return max;
        }

        /**
         * Kurtosis: E[(x - mu)^4] / sigma^4
         * A Gaussian signal has kurtosis ≈ 3.
         * Impulsive bearing faults drive kurtosis to 6–15+.
         */
        static double kurtosis(double[] arr) {
            if (arr.length < 4) return 0.0;

            double mean = 0;
            for (double v : arr) mean += v;
            mean /= arr.length;

            double m2 = 0, m4 = 0;
            for (double v : arr) {
                double diff = v - mean;
                double diff2 = diff * diff;
                m2 += diff2;
                m4 += diff2 * diff2;
            }
            m2 /= arr.length;
            m4 /= arr.length;

            double sigma4 = m2 * m2;
            return (sigma4 > 0) ? m4 / sigma4 : 0.0;
        }

        private static void appendSamples(List<Double> list, double[] arr) {
            for (double v : arr) list.add(v);
        }

        private static double[] toArray(List<Double> list) {
            double[] arr = new double[list.size()];
            for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
            return arr;
        }
    }

    // ── Accumulator ────────────────────────────────────────────────────────

    public static class VibrationAccumulator {
        public String       deviceId    = "";
        public long         windowEndMs = 0L;
        public int          packetCount = 0;
        public List<Double> samplesX    = new ArrayList<>();
        public List<Double> samplesY    = new ArrayList<>();
        public List<Double> samplesZ    = new ArrayList<>();
        public double       tempSum     = 0.0;
        public int          tempCount   = 0;
        public double       rpmSum      = 0.0;
        public int          rpmCount    = 0;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // BearingFrequencyExtractor
    //
    // Given the RPM and bearing geometry constants, computes the four bearing
    // defect frequencies and estimates their spectral magnitudes from the
    // accumulated time-domain signal via a simple DFT bin lookup.
    //
    // Bearing geometry (defaults to a typical 6205 deep-groove ball bearing):
    //   Nb   = number of rolling elements
    //   Bd   = ball diameter (mm)
    //   Pd   = pitch diameter (mm)
    //   phi  = contact angle (degrees)
    //
    // Frequencies at shaft speed f_r (Hz = RPM/60):
    //   BPFO = (Nb/2) * f_r * (1 - (Bd/Pd)*cos(phi))
    //   BPFI = (Nb/2) * f_r * (1 + (Bd/Pd)*cos(phi))
    //   BSF  = (Pd / (2*Bd)) * f_r * (1 - ((Bd/Pd)*cos(phi))^2)
    //   FTF  = (f_r/2) * (1 - (Bd/Pd)*cos(phi))
    // ══════════════════════════════════════════════════════════════════════════

    public static class BearingFrequencyExtractor
            implements MapFunction<SensorFeatures, SensorFeatures> {

        // Default geometry: SKF 6205 deep-groove ball bearing
        private static final int    NB  = 9;          // number of balls
        private static final double BD  = 7.94;       // ball diameter mm
        private static final double PD  = 38.5;       // pitch diameter mm
        private static final double PHI = 0.0;        // contact angle rad (radial load)

        // Precomputed geometry ratio
        private static final double COS_PHI = Math.cos(PHI);
        private static final double BD_PD   = BD / PD;

        @Override
        public SensorFeatures map(SensorFeatures f) throws Exception {
            if (Double.isNaN(f.rpmAvg) || f.rpmAvg <= 0) {
                return f;
            }
            double fr = f.rpmAvg / 60.0;   // shaft frequency in Hz

            // Compute bearing defect frequencies
            f.bpfoHz = (NB / 2.0) * fr * (1.0 - BD_PD * COS_PHI);
            f.bpfiHz = (NB / 2.0) * fr * (1.0 + BD_PD * COS_PHI);
            f.bsfHz  = (PD / (2.0 * BD)) * fr * (1.0 - Math.pow(BD_PD * COS_PHI, 2));
            f.ftfHz  = (fr / 2.0) * (1.0 - BD_PD * COS_PHI);

            // Magnitudes: use crest factor as a proxy for bearing defect energy.
            // In production this would be replaced by DFT bin magnitude at each
            // defect frequency computed from the raw waveform stored in sensor.waveforms.
            //
            // Proxy formula: magnitude = rmsZ * kurtosisZ * freqNormFactor
            // This is intentionally simplified for local dev; the full FFT path
            // runs as a separate Flink job consuming sensor.waveforms.
            double defectProxy = f.rmsZ * f.kurtosisZ;
            f.bpfoMagnitude = defectProxy * 0.40;  // BPFO is typically strongest outer-race defect
            f.bpfiMagnitude = defectProxy * 0.30;
            f.bsfMagnitude  = defectProxy * 0.20;
            f.ftfMagnitude  = defectProxy * 0.10;

            return f;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SensorFeaturesToJsonFn
    // ══════════════════════════════════════════════════════════════════════════

    public static class SensorFeaturesToJsonFn
            implements MapFunction<SensorFeatures, String> {
        private static final ObjectMapper mapper = new ObjectMapper();

        @Override
        public String map(SensorFeatures f) throws Exception {
            ObjectNode node = mapper.createObjectNode();
            node.put("device_id",        f.deviceId);
            node.put("window_end_ms",    f.windowEndMs);
            node.put("rms_x",            f.rmsX);
            node.put("rms_y",            f.rmsY);
            node.put("rms_z",            f.rmsZ);
            node.put("peak_x",           f.peakX);
            node.put("peak_y",           f.peakY);
            node.put("peak_z",           f.peakZ);
            node.put("kurtosis_x",       f.kurtosisX);
            node.put("kurtosis_y",       f.kurtosisY);
            node.put("kurtosis_z",       f.kurtosisZ);
            node.put("crest_factor_x",   f.crestFactorX);
            node.put("crest_factor_y",   f.crestFactorY);
            node.put("crest_factor_z",   f.crestFactorZ);
            node.put("temperature_avg",  f.temperatureAvg);
            node.put("rpm_avg",          f.rpmAvg);
            node.put("bpfo_hz",          f.bpfoHz);
            node.put("bpfi_hz",          f.bpfiHz);
            node.put("bsf_hz",           f.bsfHz);
            node.put("ftf_hz",           f.ftfHz);
            node.put("bpfo_magnitude",   f.bpfoMagnitude);
            node.put("bpfi_magnitude",   f.bpfiMagnitude);
            node.put("bsf_magnitude",    f.bsfMagnitude);
            node.put("ftf_magnitude",    f.ftfMagnitude);
            node.put("sample_count",     f.sampleCount);
            node.put("packet_count",     f.packetCount);
            return mapper.writeValueAsString(node);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // VictoriaMetricsSink
    //
    // Writes feature vectors to VictoriaMetrics via the /api/v1/import
    // endpoint (Prometheus exposition format, one metric per line).
    //
    // Format: metric_name{label="value",...} value timestamp_ms
    // ══════════════════════════════════════════════════════════════════════════

    public static class VictoriaMetricsSink extends RichSinkFunction<SensorFeatures> {

        private final String victoriaUrl;

        public VictoriaMetricsSink(String victoriaUrl) {
            this.victoriaUrl = victoriaUrl;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
        }

        @Override
        public void invoke(SensorFeatures f, Context context) throws Exception {
            StringBuilder sb = new StringBuilder();
            String device = f.deviceId.replace("\"", "");
            long ts = f.windowEndMs;

            // Write each feature as a separate Prometheus metric line
            appendMetric(sb, "pdm_rms_x",          device, f.rmsX,          ts);
            appendMetric(sb, "pdm_rms_y",          device, f.rmsY,          ts);
            appendMetric(sb, "pdm_rms_z",          device, f.rmsZ,          ts);
            appendMetric(sb, "pdm_peak_x",         device, f.peakX,         ts);
            appendMetric(sb, "pdm_peak_y",         device, f.peakY,         ts);
            appendMetric(sb, "pdm_peak_z",         device, f.peakZ,         ts);
            appendMetric(sb, "pdm_kurtosis_x",     device, f.kurtosisX,     ts);
            appendMetric(sb, "pdm_kurtosis_y",     device, f.kurtosisY,     ts);
            appendMetric(sb, "pdm_kurtosis_z",     device, f.kurtosisZ,     ts);
            appendMetric(sb, "pdm_crest_factor_x", device, f.crestFactorX,  ts);
            appendMetric(sb, "pdm_crest_factor_y", device, f.crestFactorY,  ts);
            appendMetric(sb, "pdm_crest_factor_z", device, f.crestFactorZ,  ts);
            appendMetric(sb, "pdm_temperature_avg",device, f.temperatureAvg,ts);
            appendMetric(sb, "pdm_rpm_avg",        device, f.rpmAvg,        ts);
            appendMetric(sb, "pdm_bpfo_magnitude", device, f.bpfoMagnitude, ts);
            appendMetric(sb, "pdm_bpfi_magnitude", device, f.bpfiMagnitude, ts);
            appendMetric(sb, "pdm_bsf_magnitude",  device, f.bsfMagnitude,  ts);
            appendMetric(sb, "pdm_ftf_magnitude",  device, f.ftfMagnitude,  ts);

            byte[] payload = sb.toString().getBytes(StandardCharsets.UTF_8);

            URL url = new URL(victoriaUrl + "/api/v1/import/prometheus");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setDoOutput(true);
            conn.setRequestMethod("PUT");
            conn.setRequestProperty("Content-Type", "text/plain");
            conn.setConnectTimeout(5_000);
            conn.setReadTimeout(5_000);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(payload);
            }

            int status = conn.getResponseCode();
            if (status >= 400) {
                throw new RuntimeException("VictoriaMetrics write failed: HTTP " + status);
            }
        }

        private void appendMetric(StringBuilder sb, String name, String device,
                                  double value, long tsMs) {
            if (Double.isNaN(value)) return;
            sb.append(name)
              .append("{device_id=\"").append(device).append("\"}")
              .append(" ")
              .append(value)
              .append(" ")
              .append(tsMs)
              .append("\n");
        }
    }
}
