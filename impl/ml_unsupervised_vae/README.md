# Unsupervised VAE Anomaly Detection

## Overview

This module implements a **Variational Autoencoder (VAE)**-based anomaly detection system for
industrial machinery vibration signals. Unlike supervised approaches (e.g., CNN fault classifiers),
this model trains **exclusively on normal operation data** — no fault labels are required.

---

## Core Principle: Reconstruction Error as Anomaly Score

A VAE learns a compressed latent representation of healthy machinery behavior. During inference:

1. A feature vector is passed through the **encoder** to produce a distribution in latent space.
2. A sample is drawn and passed through the **decoder** to reconstruct the input.
3. The **reconstruction error** (MSE between input and reconstruction) is the anomaly score.

Because the model has only ever learned to reconstruct normal patterns, anomalous inputs
(bearing faults, imbalance, looseness) will reconstruct poorly, yielding high error.

```
feature vector (1024-dim)
        │
        ▼
  ┌───────────┐
  │  Encoder  │  1024 → 512 → 256 → [mu, log_var] (64-dim each)
  └───────────┘
        │
        ▼  reparameterization: z = mu + eps * exp(0.5 * log_var)
        │
  ┌───────────┐
  │  Decoder  │  64 → 256 → 512 → 1024
  └───────────┘
        │
        ▼
  reconstruction (1024-dim)
        │
        ▼
  anomaly_score = MSE(input, reconstruction)
```

---

## Architecture Details

| Layer       | Input Dim | Output Dim | Activation         |
|-------------|-----------|------------|--------------------|
| enc_fc1     | 1024      | 512        | BatchNorm + ReLU   |
| enc_fc2     | 512       | 256        | BatchNorm + ReLU   |
| enc_mu      | 256       | 64         | —                  |
| enc_logvar  | 256       | 64         | —                  |
| dec_fc1     | 64        | 256        | BatchNorm + ReLU   |
| dec_fc2     | 256       | 512        | BatchNorm + ReLU   |
| dec_out     | 512       | 1024       | Sigmoid            |

**Input features (1024-dim):**
- FFT magnitude spectrum (e.g., 900 bins up to Nyquist)
- Bearing fault frequencies: BPFO, BPFI, BSF, FTF magnitudes and harmonics
- Time-domain statistics: RMS, kurtosis, crest factor, skewness, peak-to-peak

**Loss function:**
```
L = MSE(x, x_hat) + beta * KL(q(z|x) || p(z))
```
where `beta` (default 1.0) controls the weight of the KL regularization term.
A lower `beta` yields better reconstruction; a higher `beta` yields a more
disentangled latent space. Recommended range for anomaly detection: 0.5–2.0.

---

## MachineID-Conditioned Variant

`MachineIDConditionedVAE` extends the base VAE with a learnable **machine ID embedding**
(128-dim) concatenated to the encoder input. This allows the model to learn
**per-machine personalized normal profiles**, accounting for unit-to-unit variation
in vibration characteristics.

Reference: *ID-Conditioned Expert AE for Anomaly Detection* — arXiv 2007.05314

---

## When to Use This vs. Supervised CNN

| Criterion                         | VAE (this module)          | Supervised CNN             |
|-----------------------------------|----------------------------|----------------------------|
| Labeled fault data available      | Not required               | Required                   |
| Cold-start / new machine          | Yes — deploy immediately   | No — must collect faults   |
| Fault type classification         | No                         | Yes (BPFO, BPFI, etc.)     |
| Sensitive to novel fault modes    | Yes — any anomaly detected | Only trained fault types   |
| False positive rate               | Higher (tunable threshold) | Lower (explicit classes)   |
| Recommended deployment phase      | Early deployment           | After fault history builds |

**Recommended strategy:** Deploy VAE first (unsupervised, instant cold start). As labeled
fault events accumulate over 6–12 months, train the supervised CNN alongside it. Use the
CNN when confident fault labels exist; keep the VAE as a catch-all for novel fault modes.

---

## References

- Kingma & Welling (2013). *Auto-Encoding Variational Bayes.* arXiv 1312.6114
- Xu et al. (2019). *Unsupervised Anomaly Detection via Variational Auto-Encoder.* arXiv 1912.01096
- Hendrycks et al. (2020). *ID-Conditioned Expert AE for Anomaly Detection.* arXiv 2007.05314
- Smith (1999). *The Local Mean Decomposition and Its Application.* (CWRU Bearing Dataset)
