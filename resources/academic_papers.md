# Academic Papers: Acoustic & Vibration Sensing for Predictive Maintenance
> 32 papers compiled from arXiv searches — organized by system component

---

## Section 1 — Acoustic Emission Sensing

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 1 | AE + Deep Transfer Learning (Bolted Joints) | 2024 | 2405.20887 | CWT image encoding of AE signals as CNN inputs; super-convergence for multi-sensor AE |
| 2 | Non-Gaussian AE Clustering (DBSCAN + Envelope Spectrum) | 2025 | 2502.11786 | Spectral clustering for isolating fault signals in non-Gaussian industrial noise |

---

## Section 2 — Vibration-Based Predictive Maintenance

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 3 | Quadratic TFR + CNN (Variable Speed) | 2024 | 2401.01172 | Solves non-stationary speed challenge; TFR-CNN outperforms FFT under variable load |
| 4 | WPT + FFT + Random Forest (Low Latency) | 2022 | 2208.06051 | Efficient classical pipeline; near real-time detection; strong baseline |
| 5 | Wavelet + Self-Supervised (1% Labels) | 2022 | 2207.10432 | CWT + contrastive self-supervised pretraining; >90% accuracy with 1% labeled data |
| 6 | Spectral Graph Wavelet Networks | 2023 | 2303.14958 | Multi-scale graph wavelets; captures both global trends and localized fault impulses |
| 7 | Statistical Batch + FFT (SPC) | 2024 | 2407.17236 | Explainable statistical process control; competitive with deep learning; interpretable alerts |
| 8 | Sound vs. Vibration Benchmark | 2023 | 2312.10742 | Acoustic sensing more robust than vibration in typical conditions; new dual-modality dataset |
| 9 | 1D CNN Raw Vibration (CWRU) | 2026 | 2602.09699 | End-to-end 1D CNN; 99.14% accuracy at 0 HP; lightweight baseline |
| 10 | Dual-Domain Multi-Scale CNN (Noise Robust) | 2026 | 2608.09174 | Time + frequency domain combined; 92.50% at -4 dB SNR |

---

## Section 3 — Deep Learning for Industrial Anomaly Detection

### Supervised (CNN / LSTM / Transformer)

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 11 | FaultFormer (Transformer Pretraining) | 2024 | 2312.02380 | Masked pretraining for vibration transformers; strong generalization to new machines |
| 12 | CNN + LSTM End-to-End | 2019 | 1909.07801 | Classic CNN+LSTM hybrid; complementary spatial and temporal feature learning |
| 13 | Transformer Vibration Denoising | 2023 | 2308.02166 | Learned denoising pre-stage; adapts to changing noise profiles vs fixed filters |
| 14 | RmGPT Foundation Model | 2024 | 2409.17604 | GPT-style foundation model for fault diagnosis + RUL; 82% one-shot 16-class accuracy |

### Unsupervised (Autoencoders / VAE)

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 15 | Semi-Supervised VAE (CWRU) | 2019 | 1912.01096 | VAE reconstruction error as health indicator; 3–30% gain with few labeled examples |
| 16 | ID-Conditioned Autoencoder (MIMII) | 2020 | 2007.05314 | Machine-ID conditioning for per-unit normal profiles in fleet deployments |
| 17 | On-Device VAE/GAN/Diffusion Federated | 2026 | 2605.07860 | Systematic comparison of generative models for federated edge PdM |
| 18 | SCVAE Compressed VAE (Edge IoT) | 2017 | 1712.06343 | Depthwise separable convolutions for VAE; fits on edge MCU memory budgets |

### Few-Shot Learning

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 19 | MAML Few-Shot Bearing | 2020 | 2007.12851 | Meta-learning adapts to new fault types with very few examples; +25% vs Siamese baseline |
| 20 | Meta Transformer (1% Labels) | 2025 | 2509.09251 | Multi-head attention meta-learning; 99% accuracy with 1% labeled data |
| 21 | FaultDiffusion (Diffusion Data Augmentation) | 2025 | 2511.15174 | Diffusion models synthesize realistic fault time-series for training data augmentation |

---

## Section 4 — Edge AI for Industrial IoT

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 22 | CNN on STM32 MCU (19 ms inference) | 2023 | 2304.09100 | 98.9% accuracy; 19 ms per cycle on STM32H743VI; full communication stack |
| 23 | TinyML Hierarchical Inference (Mining) | 2024 | 2411.07168 | 3-tier edge/gateway/cloud hierarchy; >90% accuracy at 3.33 ms; 44% power reduction |
| 24 | Federated Learning Multi-Factory | 2022 | 2211.09406 | Federated clustering for multi-site PdM without raw data sharing |

---

## Section 5 — Multi-Sensor Fusion

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 25 | Physics-Informed Multimodal CNN | 2025 | 2508.07536 | Physics constraints as regularizer in vibration + current fusion; improves cross-condition generalization |
| 26 | Self-Supervised Phase-Aware Complex Network | 2023 | 2312.13615 | Phase information in acoustic signals carries diagnostic value; self-supervised phase objectives |
| 27 | CP Tensor Decomposition + DL (Multi-channel) | 2021 | 2107.09519 | Non-negative CP tensor decomposition for multi-channel spectral denoising + compression |

---

## Section 6 — Datasets and Benchmarks

| # | Title | Year | arXiv | Key Contribution |
|---|---|---|---|---|
| 28 | MIMII Dataset Paper | 2019 | 1909.09347 | First public acoustic anomaly detection dataset; valves, pumps, fans, slide rails with factory noise |
| 29 | Universal Vibration Dataset Framework | 2025 | 2504.11581 | Multi-dataset framework (CWRU+MFPT+DIRG) for transfer learning; ImageNet-style pretraining |
| 30 | Realistic CWRU Evaluation (Data Leakage) | 2025 | 2509.22267 | Bearing-wise partitioning required to avoid inflated CWRU accuracy; methodology corrective |
| 31 | Anomalous Sound Detection Systematic Review | 2021 | 2102.07820 | Survey of 31 studies; mel-spectrogram + autoencoder/CNN is dominant approach |
| 32 | Deep SVDD on MIMII (7.4× fewer params) | 2024 | 2412.10792 | One-class deep SVDD outperforms reconstruction autoencoders; 7.4× parameter reduction |

---

## Design Guidance from Literature

**Sensing layer:** Papers 8 and 25 establish acoustic + vibration as complementary modalities. Paper 8 specifically finds acoustic sensing more robust under typical conditions.

**Feature extraction convergence:**
- CWT/STFT images for CNN inputs (stationary conditions)
- Wavelet packet decomposition for multi-scale sub-band analysis
- End-to-end 1D CNN for raw waveforms
- Papers 3 and 10: dual time-frequency representations best under variable speed

**Model selection guide:**
- Supervised, sufficient data: 1D CNN (Paper 9) or FaultFormer transformer (Paper 11)
- Unsupervised / cold-start: VAE reconstruction error (Paper 15) or Deep SVDD (Paper 32)
- Scarce fault data: MAML meta-learning (Paper 19) or diffusion augmentation (Paper 21)

**Edge deployment targets:**
- STM32H743VI: 19 ms inference (Paper 22)
- Three-tier hierarchy cuts power 44% (Paper 23)
- SCVAE model compression for MCU memory budgets (Paper 18)

**Evaluation protocol:** Use bearing-wise CWRU partitioning (Paper 30) — standard split inflates accuracy by data leakage.

**Benchmark suite:** CWRU + MFPT (vibration), MIMII (acoustic). Dataset fusion as in Paper 29 for foundation model pretraining.
