# Supervised 1D CNN Fault Classification

## Overview

This module implements a supervised 1D Convolutional Neural Network for industrial bearing
and machinery fault classification, trained on labeled fault data from the CWRU (Case Western
Reserve University) and MFPT (Machinery Failure Prevention Technology) benchmark datasets.

The model operates directly on the raw FFT magnitude spectrum of vibration signals, enabling
named fault classification with associated confidence scores — a key differentiator over
unsupervised anomaly detection approaches.

## Architecture

**Input**: 1D tensor of shape `(batch, 1, 2048)` — raw FFT magnitude spectrum computed from
a 2048-sample vibration window at the sensor sampling rate.

**Network**: Four stacked Conv1D blocks, each consisting of:
- Conv1D (with increasing filter depth)
- Batch Normalization
- ReLU activation
- MaxPool1D (stride 2)
- Dropout (0.25)

Block configuration:
| Block | Filters (in→out) | Kernel Size | Stride |
|-------|-----------------|-------------|--------|
| 1     | 1 → 64          | 64          | 2      |
| 2     | 64 → 128        | 32          | 1      |
| 3     | 128 → 256       | 16          | 1      |
| 4     | 256 → 256       | 8           | 1      |

**Classifier head**: Global Average Pooling → FC(256→128) → ReLU → Dropout → FC(128→12) → Softmax

**Variant**: `BearingFaultCNNLSTM` extends the base CNN with a 2-layer LSTM (hidden=128) that
processes a sequence of overlapping windows, providing temporal context for transient faults.

## Fault Classes (12)

| ID | Class Label   | Description                             |
|----|---------------|-----------------------------------------|
| 0  | Normal        | Healthy bearing, no fault               |
| 1  | BPFO          | Ball Pass Frequency Outer race defect   |
| 2  | BPFI          | Ball Pass Frequency Inner race defect   |
| 3  | BSF           | Ball Spin Frequency (ball defect)       |
| 4  | FTF           | Fundamental Train Frequency (cage)      |
| 5  | Unbalance     | Rotor mass unbalance                    |
| 6  | Misalignment  | Shaft or coupling misalignment          |
| 7  | Lubrication   | Insufficient or degraded lubrication    |
| 8  | Looseness     | Structural looseness / bearing looseness|
| 9  | Cavitation    | Hydraulic cavitation (pump/compressor)  |
| 10 | Gearbox       | Gear tooth defect / wear                |
| 11 | Electrical    | Electrical fault (motor stator/rotor)   |

## Datasets

- **CWRU Bearing Dataset**: DE/FE accelerometer signals at 12 kHz / 48 kHz, four load
  conditions (0–3 HP), three fault diameters (0.007", 0.014", 0.021") for outer/inner/ball
  faults, plus normal baseline.
- **MFPT Bearing Dataset**: 97,656 Hz sampling rate, three load conditions, outer/inner/ball
  fault classes plus normal.

## Critical: Bearing-Wise Train/Test Split

**All experiments use bearing-wise partitioning** — each physical bearing appears in either
the training set or the test set, never both. This prevents data leakage from correlated
windows belonging to the same bearing run, which inflates accuracy by up to 40 percentage
points on standard random splits.

This methodology follows the evaluation protocol described in:
> arXiv 2509.22267 — "Data Leakage in Bearing Fault Diagnosis: A Systematic Review"

Failure to apply bearing-wise splits is the single most common source of overly optimistic
benchmark results in the fault diagnosis literature.

## References

- **arXiv 2602.09699** — 1D CNN on CWRU dataset, establishing baseline architecture and
  preprocessing pipeline used here.
- **arXiv 1909.07801** — CNN+LSTM hybrid for bearing fault diagnosis with temporal context,
  basis for `BearingFaultCNNLSTM`.
- **arXiv 2509.22267** — Systematic review of data leakage in bearing fault diagnosis,
  motivating the bearing-wise split requirement.
- **arXiv 2304.09100** — TF Lite INT8 quantization for MCU deployment of vibration models.

## Tradeoffs vs. VAE Anomaly Detection

| Aspect               | Supervised 1D CNN (this module)          | VAE Anomaly Detection                  |
|----------------------|------------------------------------------|----------------------------------------|
| Output               | Named fault class + confidence score     | Anomaly score only                     |
| Interpretability     | High — tells operators what is wrong     | Low — only signals "something is wrong"|
| Data requirement     | Labeled fault examples required          | Normal data only (easier to collect)   |
| New fault types      | Requires retraining with new labels      | Detects any deviation automatically    |
| Cold-start           | Poor on new machine types                | Works immediately after normal baseline|
| Actionability        | Direct maintenance action recommendation | Requires further diagnosis             |
| Benchmark accuracy   | ~98% (CWRU, bearing-wise split ~85%)     | AUROC ~0.92 typical                    |

**Recommendation**: Deploy VAE for anomaly detection on new machines or fault types not in
training data; use supervised CNN when fault-type specificity is needed for maintenance
scheduling and parts procurement.
