# Product 4: ML/AI Diagnostic Engine
> AutoDiagnosis equivalent — fault detection, classification, and prognosis

---

## Product Overview

The intelligence layer of the platform. Ingests multi-modal sensor features from the data pipeline, runs anomaly detection, fault classification, severity staging, and remaining useful life (RUL) estimation. Produces named fault diagnoses with confidence scores, severity levels, and LLM-generated plain-language work instructions. This is the direct equivalent of TRACTIAN's AutoDiagnosis™ engine.

**Key differentiators from TRACTIAN:**
- Explainable AI: SHAP-based feature attribution per alert
- Uncertainty quantification: probabilistic confidence intervals
- Few-shot new fault adaptation: MAML meta-learning for novel failure modes
- Open model registry: all models versioned and auditable

---

## Functional Requirements

### FR-ML-01: Unsupervised Anomaly Detection (Baseline — no labeled data needed)
- VAE (Variational Autoencoder) trained on normal operating data
- Input: multivariate feature vector (vibration RMS + kurtosis + crest + AE RMS + temp + RPM per window)
- Output: anomaly score (reconstruction error) + percentile rank vs historical baseline
- Per-machine personalized model: conditioned on machine ID (arXiv 2007.05314 approach)
- Cold-start: deployable on a new machine with 7 days of normal operation data
- Adaptive threshold: updates rolling baseline as machine ages / conditions change

### FR-ML-02: Supervised Fault Classification
- 1D CNN + LSTM hybrid classifier (arXiv 1909.07801)
- Input: 1024-point FFT magnitude spectrum + time-domain features
- Output classes: Normal, Outer-race defect, Inner-race defect, Ball defect, Cage defect, Unbalance, Misalignment, Lubrication, Looseness, Cavitation, Gearbox fault, Electrical fault
- Confidence score per class
- Training datasets: CWRU + MFPT (vibration), MIMII (acoustic), proprietary customer data
- Evaluation: bearing-wise partitioned per arXiv 2509.22267 (no data leakage)

### FR-ML-03: Severity Staging
- 4-stage severity scale: Stage 1 (watch), Stage 2 (plan), Stage 3 (act soon), Stage 4 (critical)
- Stage determined from: anomaly score trend + fault class confidence + time-to-failure estimate
- Stage history tracked per asset for trend visualization

### FR-ML-04: Remaining Useful Life (RUL) Estimation
- Probabilistic RUL: median estimate + 80% confidence interval (days)
- Model: GluonTS DeepAR or TFT on degradation trend features
- Input: rolling anomaly score trend + severity stage history
- Calibrated on NASA PRONOSTIA and FEMTO bearing run-to-failure datasets
- Output shown in dashboard as "estimated time to maintenance needed: X days (±Y days)"

### FR-ML-05: Multi-Modal Fusion
- Late fusion: independent per-modality networks (vibration branch + acoustic branch + thermal branch) combined by a lightweight attention-weighted combiner
- Early fusion option: concatenated feature vector as single model input
- Physics-informed regularization: penalize predictions contradicting known mechanical relationships (arXiv 2508.07536)

### FR-ML-06: Few-Shot New Fault Adaptation
- MAML-based meta-learning (arXiv 2007.12851): adapt to new fault class with ≤20 examples
- Diffusion-based data augmentation (arXiv 2511.15174): synthesize additional training samples from few real examples
- New fault workflow: customer collects 5–20 confirmed fault examples → model fine-tuned in <30 minutes

### FR-ML-07: Explainability
- SHAP (SHapley Additive exPlanations) values per prediction: which frequency bands / features drove the alert
- Feature importance visualization: top-5 contributing features shown in dashboard
- Counterfactual explanation: "If bearing defect frequency component decreased by X%, alert would not have fired"

### FR-ML-08: LLM Work Instruction Generation
- On fault classification: invoke LLM (Claude claude-haiku-4-5-20251001 for speed, claude-sonnet-5 for complex faults) to generate:
  - Plain-language fault description for the maintenance technician
  - Step-by-step inspection and repair procedure
  - Required parts list (from bearing database)
  - Safety precautions (LOTO, PPE)
- Prompt includes: fault class, severity, asset metadata, historical repair history

### FR-ML-09: Model Management
- All models versioned in MLflow Model Registry
- Canary deployments: new model version receives 10% of traffic before full rollout
- A/B testing framework: compare two model versions on same sensor stream
- Automated retraining trigger: data drift detected OR new labeled data exceeds threshold
- Model performance monitoring: precision/recall on new confirmed fault events

---

## Non-Functional Requirements

### NFR-ML-01: Inference Latency
- Anomaly score (VAE): <100 ms per device per window
- Fault classification (CNN): <500 ms per event
- RUL estimate: <2 seconds
- LLM work instruction: <10 seconds

### NFR-ML-02: Throughput
- Handle ≥10,000 concurrent devices with rolling window inference
- Batch retraining: complete within 4 hours for 1M labeled samples

### NFR-ML-03: Model Quality
- Fault classification F1 score: >0.92 weighted average across fault classes
- False positive rate: <5% per asset per 30-day period (minimize alert fatigue)
- RUL MAE: <15% of actual remaining life

### NFR-ML-04: Privacy
- Federated learning option for customers that cannot share raw data
- Differential privacy noise addition during federated aggregation
- Data never leaves customer network in federated mode

---

## Technical Architecture

```
Feature Store (VictoriaMetrics + PostgreSQL)
         ↓
┌────────────────────────────────────────────┐
│  Anomaly Detection Layer                    │
│  VAE per machine (PyTorch + MLflow)         │
│  → anomaly_score, percentile_rank           │
├────────────────────────────────────────────┤
│  Fault Classification Layer                 │
│  1D CNN + LSTM (PyTorch)                    │
│  → fault_class, confidence[12 classes]      │
├────────────────────────────────────────────┤
│  Multi-Modal Fusion                         │
│  Attention combiner (vibration+AE+thermal)  │
│  → fused_fault_class, fused_confidence      │
├────────────────────────────────────────────┤
│  Severity + RUL Layer                       │
│  GluonTS DeepAR trend model                 │
│  → severity_stage[1-4], rul_days            │
├────────────────────────────────────────────┤
│  Explainability                             │
│  SHAP TreeExplainer / DeepExplainer         │
│  → feature_attributions[]                   │
├────────────────────────────────────────────┤
│  LLM Work Instruction                       │
│  Claude Haiku 4.5 API (Anthropic)           │
│  → work_instruction_text, parts_list[]      │
└────────────────────────────────────────────┘
         ↓
Diagnostic Event → Cloud Platform API (Product 5)
         ↓
Auto Work Order → CMMS (Product 6)
```

### Model Serving
- **TorchServe** (PyTorch native, Apache 2.0) for CNN/LSTM and VAE inference
- **MLflow Models** for model registry + deployment abstraction
- **gRPC** API between Flink stream processor and model serving

### Training Pipeline
```
Airflow DAG: retrain_anomaly_detector
  Task 1: Load labeled data from PostgreSQL (last 30 days)
  Task 2: Feature normalization + bearing-wise train/test split
  Task 3: Train VAE (PyTorch Lightning) on normal data
  Task 4: Train CNN classifier on fault data
  Task 5: Evaluate on held-out test set; gate on F1 > 0.92
  Task 6: Register model in MLflow registry
  Task 7: Canary deploy to 10% of devices
  Task 8: Monitor false-positive rate for 48h
  Task 9: Full rollout if no regression
```

---

## Implementation Phases

### Phase 0: Baseline Models (Months 1–3)
- [ ] Train VAE on CWRU normal-only data (PyTorch); evaluate reconstruction error
- [ ] Train 1D CNN classifier on CWRU + MFPT (bearing-wise split)
- [ ] Evaluate: F1, precision, recall, confusion matrix
- [ ] MLflow experiment tracking setup
- [ ] Benchmark against arXiv baselines (Papers 9, 15, 32)

### Phase 1: Multi-Modal Models (Months 4–7)
- [ ] Train acoustic branch on MIMII dataset (VAE per machine type)
- [ ] Late fusion combiner (vibration + acoustic)
- [ ] SHAP explainability integration
- [ ] Model serving with TorchServe; gRPC API
- [ ] Airflow retraining DAG

### Phase 2: Advanced Features (Months 8–12)
- [ ] GluonTS RUL model on PRONOSTIA dataset
- [ ] MAML few-shot adapter for new fault types
- [ ] FaultDiffusion data augmentation pipeline
- [ ] LLM work instruction generation (Claude Haiku 4.5)
- [ ] Model A/B testing framework

### Phase 3: Fleet Scale (Months 13–18)
- [ ] Federated learning framework (privacy-preserving multi-site)
- [ ] Foundation model pretraining on accumulated customer data
- [ ] Quantized edge versions (INT8) for on-device Tier-2 inference
- [ ] Physics-informed regularization for variable operating conditions

---

## Key GitHub Repos

| Repo | Use |
|---|---|
| [unit8co/darts](https://github.com/unit8co/darts) | Anomaly detection + forecasting toolkit |
| [awslabs/gluonts](https://github.com/awslabs/gluonts) | RUL probabilistic forecasting |
| [chickenbestlover/RNN-Time-series-Anomaly-Detection](https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection) | LSTM reconstruction error anomaly detection |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | Model registry + experiment tracking |
| [apache/airflow](https://github.com/apache/airflow) | Retraining pipeline orchestration |
| [awslabs/predictive-maintenance-using-machine-learning](https://github.com/awslabs/predictive-maintenance-using-machine-learning) | Reference architecture |

---

## Key Academic Papers

| arXiv | Relevance |
|---|---|
| 1912.01096 | Semi-supervised VAE — primary unsupervised anomaly detection approach |
| 2007.05314 | ID-conditioned autoencoder — per-machine personalized models |
| 1909.07801 | CNN+LSTM end-to-end — fault classification architecture |
| 2312.02380 | FaultFormer transformer pretraining — foundation model approach |
| 2409.17604 | RmGPT — GPT-style multi-task foundation model |
| 2007.12851 | MAML few-shot — new fault adaptation |
| 2511.15174 | FaultDiffusion — data augmentation for scarce faults |
| 2508.07536 | Physics-informed multimodal CNN |
| 2412.10792 | Deep SVDD — parameter-efficient anomaly detection |
| 2509.22267 | Correct CWRU evaluation protocol |

---

## Success Metrics

| Metric | Target |
|---|---|
| Fault classification F1 (weighted) | >0.92 |
| False positive rate | <5% per asset per 30 days |
| Cold-start detection delay | <7 days of normal data needed |
| RUL MAE | <15% of actual |
| New fault adaptation | ≤20 labeled examples needed |
| Inference latency (anomaly score) | <100 ms |
| Inference latency (fault class) | <500 ms |
| LLM work instruction generation | <10 seconds |
