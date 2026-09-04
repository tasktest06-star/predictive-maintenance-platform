# Apache 2.0 Licensed GitHub Repositories
> 32 repositories for predictive maintenance stack development — verified licenses

---

## Category 1 — Signal Processing & Acoustic Analysis

| Repo | Stars | Description | Use in Stack |
|---|---|---|---|
| [tyiannak/pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) | ~6,259 | Audio feature extraction (MFCCs, chroma, spectral features), classification, segmentation | Baseline feature extraction for acoustic/vibration classifier inputs |
| [ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) | ~1,076 | Embedded DSP for Cortex-M/A: FFT, FIR/IIR, matrix math, statistics | On-device FFT and filtering at the sensor MCU level |

---

## Category 2 — Machine Learning for Anomaly Detection

| Repo | Stars | Description | Use in Stack |
|---|---|---|---|
| [unit8co/darts](https://github.com/unit8co/darts) | ~9,508 | Unified time-series forecasting + anomaly detection: ARIMA, N-BEATS, Transformer, LSTM | Anomaly scoring on vibration/thermal time-series; multivariate multi-sensor support |
| [awslabs/gluonts](https://github.com/awslabs/gluonts) | ~5,229 | Probabilistic time-series: DeepAR, TFT, uncertainty quantification | RUL prediction with confidence intervals for maintenance scheduling |
| [chickenbestlover/RNN-Time-series-Anomaly-Detection](https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection) | ~1,305 | LSTM/RNN anomaly detection via reconstruction error on multivariate series (PyTorch) | Vibration/temperature stream anomaly detection without labeled fault data |
| [linkedin/luminol](https://github.com/linkedin/luminol) | ~1,230 | Anomaly detection + cross-series correlation | Correlating anomaly windows across vibration + temp + current channels |
| [PaddlePaddle/PaddleTS](https://github.com/PaddlePaddle/PaddleTS) | ~547 | Deep time-series: forecasting, representation learning, anomaly detection (PaddlePaddle) | Pre-built anomaly detection + embedding generation for similarity-based fault detection |
| [baidu/Curve](https://github.com/baidu/Curve) | ~535 | Semi-supervised anomaly detection with labeling UI, training, evaluation | Label scarce industrial data; semi-supervised anomaly threshold training |
| [thedatumorg/TSB-AD](https://github.com/thedatumorg/TSB-AD) | ~322 | Comprehensive time-series anomaly benchmark with algorithm implementations | Benchmark candidate anomaly detectors before production deployment |
| [ExpediaGroup/adaptive-alerting](https://github.com/ExpediaGroup/adaptive-alerting) | ~212 | Streaming anomaly detection with auto model selection and adaptive thresholds | Auto-calibrating thresholds as sensor baselines drift over time |
| [thedatumorg/TSB-UAD](https://github.com/thedatumorg/TSB-UAD) | ~221 | Univariate time-series anomaly detection benchmark suite | Validate single-channel vibration/acoustic models |
| [lytics/anomalyzer](https://github.com/lytics/anomalyzer) | ~298 | Probabilistic anomaly detection in Go | Low-latency edge gateway alert microservice |

---

## Category 3 — IoT & Sensor Data Pipelines

| Repo | Stars | Description | Use in Stack |
|---|---|---|---|
| [thingsboard/thingsboard](https://github.com/thingsboard/thingsboard) | ~22,357 | Full IoT platform: device management, MQTT/HTTP/CoAP ingestion, rule engine, dashboards | End-to-end sensor-to-dashboard; rule engine for threshold alerts; MQTT ingestion |
| [apache/kafka](https://github.com/apache/kafka) | ~33,661 | Distributed event streaming for high-throughput real-time pipelines | High-frequency vibration stream backbone; decouples acquisition from ML inference |
| [apache/flink](https://github.com/apache/flink) | ~26,325 | Stateful stream + batch processing with exactly-once semantics and event-time windowing | Real-time windowed aggregations (RMS, kurtosis, crest factor); sub-second anomaly scoring |
| [apache/spark](https://github.com/apache/spark) | ~43,947 | Unified analytics: batch, streaming (Structured Streaming), ML (MLlib) | Batch feature extraction over historical sensor data; model retraining pipelines |
| [apache/nifi](https://github.com/apache/nifi) | ~6,219 | Visual data routing and transformation with 300+ processors | Low-code ETL for routing sensor data from MQTT to databases/ML services/cloud |
| [apache/beam](https://github.com/apache/beam) | ~8,658 | Unified batch + streaming model runnable on Flink/Spark/Dataflow | Write signal processing logic once; deploy to cloud or on-premises runners |
| [hivemq/hivemq-community-edition](https://github.com/hivemq/hivemq-community-edition) | ~1,207 | Java MQTT 3.x/5 broker; enterprise IoT workloads | Sensor-to-gateway MQTT transport; WebSockets + TLS support |
| [vernemq/vernemq](https://github.com/vernemq/vernemq) | ~3,626 | Distributed MQTT broker (Erlang/OTP); high availability industrial | Fault-tolerant always-on MQTT for critical industrial systems |
| [VictoriaMetrics/VictoriaMetrics](https://github.com/VictoriaMetrics/VictoriaMetrics) | ~17,647 | Fast time-series database + monitoring; Prometheus-compatible | High-cardinality sensor metric storage (RMS, peak freq, temp per sensor) |
| [open-telemetry/opentelemetry-collector](https://github.com/open-telemetry/opentelemetry-collector) | ~7,492 | Vendor-neutral telemetry collection: traces, metrics, logs | Standardize heterogeneous sensor telemetry from OPC-UA/MQTT/proprietary protocols |

---

## Category 4 — Predictive Maintenance Reference Implementations

| Repo | Stars | Description | Use in Stack |
|---|---|---|---|
| [awslabs/predictive-maintenance-using-machine-learning](https://github.com/awslabs/predictive-maintenance-using-machine-learning) | ~110 | End-to-end SageMaker demo: RUL prediction from NASA Turbofan Engine dataset | Reference architecture for LSTM-based RUL prediction; feature engineering patterns |
| [awslabs/aws-fleet-predictive-maintenance](https://github.com/awslabs/aws-fleet-predictive-maintenance) | ~74 | ML-based PdM for vehicle fleets using multi-signal CAN bus data | Multi-signal fusion patterns for vibration + temp + pressure + electrical |

---

## Category 5 — Edge Computing & Embedded Inference

| Repo | Stars | Description | Use in Stack |
|---|---|---|---|
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | ~198,793 | ML framework including TF Lite (embedded/mobile) and TF Serving (production) | Deploy trained fault-detection models on MCUs/Jetson via TF Lite; INT8 quantization |
| [onnx/onnx](https://github.com/onnx/onnx) | ~21,410 | Open format for ML model interoperability (PyTorch/TF/sklearn → ONNX Runtime) | Train in Python, deploy with ONNX Runtime C++ on edge hardware |
| [apache/tvm](https://github.com/apache/tvm) | ~13,712 | Deep learning compiler: compiles ML models to CPU/GPU/FPGA native code | Hardware-optimal inference on sensor gateway; critical for high-frequency vibration analysis |
| [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) | ~10,799 | Intel toolkit for AI inference on CPU/GPU/VPU/FPGA with INT8/FP16 quantization | Optimize models for Intel-based edge nodes and industrial PCs in factory environments |

---

## Category 6 — Monitoring, ML Lifecycle & Dashboards

| Repo | Stars | Description | Use in Stack |
|---|---|---|---|
| [apache/superset](https://github.com/apache/superset) | ~74,616 | Data exploration + BI dashboards; connects to VictoriaMetrics/TimescaleDB/InfluxDB | Maintenance dashboards without custom frontend; sensor health trends and anomaly events |
| [apache/echarts](https://github.com/apache/echarts) | ~67,230 | JavaScript charting library: 20+ chart types, rich interactivity | Embeddable HMI/SCADA web UI for real-time spectrum visualization; gauge charts |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | ~27,797 | ML lifecycle: experiment tracking, model registry, deployment, monitoring | Version and manage fault detection models; detect model drift as equipment ages |
| [apache/airflow](https://github.com/apache/airflow) | ~46,719 | Workflow orchestration for data pipelines as DAGs | Periodic model retraining; batch feature extraction; multi-step preprocessing-to-deployment |

---

## Reference Architecture Stack

```
Sensor Node
  CMSIS-DSP (on-device FFT, FIR filters)
  TF Lite Micro / ONNX Runtime (edge inference)
      ↓ MQTT (AES-256)
MQTT Broker (HiveMQ CE or VerneMQ)
      ↓
Event Bus (Apache Kafka)
      ↓
Stream Processing (Apache Flink — windowed feature extraction)
      ↓
Storage (VictoriaMetrics — metrics | Kafka — raw replay)
      ↓
ML Training (darts / GluonTS + MLflow + Airflow retraining)
      ↓
Edge Inference (TF Lite / ONNX + TVM or OpenVINO)
      ↓
Dashboard (ThingsBoard IoT native -OR- Superset + ECharts)
```
