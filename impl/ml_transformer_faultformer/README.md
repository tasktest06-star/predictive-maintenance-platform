# FaultFormer: Transformer Foundation Model for Vibration-Based Fault Diagnosis

## Concept

FaultFormer applies the BERT-style masked pretraining paradigm to industrial vibration time-series. A transformer encoder is pretrained in a self-supervised manner on large amounts of **unlabeled** vibration data — both normal and fault signals without class labels. The pretrained backbone is then fine-tuned on a small amount of **labeled** fault data.

This mirrors how BERT pretrains on unlabeled text and is then fine-tuned on downstream NLP tasks. The key insight for predictive maintenance: raw vibration data is abundant and cheap, but expert-labeled fault data is scarce and expensive. Pretraining on unlabeled data lets the model learn rich representations of mechanical signal structure before it ever sees a labeled example.

Reference paper: **FaultFormer** (arXiv:2312.02380) — the primary architecture this implementation follows.
Related work: **RmGPT** (arXiv:2409.17604) — GPT-style autoregressive multi-task foundation model for rotating machinery.

---

## Architecture

### Signal Tokenization: Patch Embedding

A 2048-point vibration window (captured at e.g. 12 kHz) is divided into **32 non-overlapping patches** of 64 points each. Each patch is projected to a 256-dimensional embedding vector via a learned linear layer. Learnable position embeddings are added so the transformer can distinguish patch order (temporal position).

```
Vibration signal: [2048 points]
       ↓ split into patches
Patches: [32 × 64 points]
       ↓ linear projection
Patch embeddings: [32 × 256]
       ↓ add positional embedding
Token sequence: [32 × 256]
```

A special `[CLS]` token is prepended, giving a sequence length of 33. After transformer encoding, the `[CLS]` token carries a global summary of the entire window and is used as the signal representation for classification or regression heads.

### Transformer Encoder

| Hyperparameter        | Value  |
|-----------------------|--------|
| Layers                | 6      |
| Attention heads       | 8      |
| Embedding dimension   | 256    |
| Feedforward dimension | 1024   |
| Dropout               | 0.1    |
| Normalization         | Pre-LN (LayerNorm before attention, following GPT-2) |

Pre-LN (pre-normalization) is used rather than post-LN for improved training stability with deep transformers.

---

## Two-Stage Training

### Stage 1: Self-Supervised Pretraining (Masked Patch Reconstruction)

**Data:** All available vibration data — CWRU, MFPT, MIMII — with **no labels required**. Both normal operation and fault signals are used, treating them all as unlabeled.

**Masking strategy:** 75% of the 32 patches are randomly masked (zeroed out or replaced with a learnable mask token). The model must reconstruct the masked patches from the remaining 25% of visible patches plus positional information.

```
Input:  [CLS] [P1] [MASK] [MASK] [P4] [MASK] ... [P32]
Output: reconstruction of all masked patches
Loss:   MSE on masked patch positions only (not visible patches)
```

The 75% masking ratio is validated in the FaultFormer paper and follows MAE (Masked Autoencoders, He et al. 2022). High masking forces the model to learn semantic correlations across the signal rather than simple local interpolation.

**Optimizer:** Adam with cosine LR schedule + linear warmup over first 10% of training epochs.

**Output:** `models/faultformer_pretrained.pt` — saved backbone weights.

### Stage 2: Supervised Fine-Tuning (Fault Classification)

The pretrained backbone is loaded. A classification head (linear layer: 256 → 12 classes) is attached to the `[CLS]` token output. Encoder layers 0–3 are frozen; layers 4–5 and the classification head are fine-tuned.

**Classes (12):** Normal, Outer-race defect, Inner-race defect, Ball defect, Cage defect, Unbalance, Misalignment, Lubrication defect, Looseness, Cavitation, Gearbox fault, Electrical fault.

**Evaluation protocol:** Bearing-wise train/test split per arXiv:2509.22267 — no bearing appears in both train and test sets, preventing data leakage from the same bearing at different fault severities.

---

## Advantages Over 1D CNN Baseline

| Property                        | 1D CNN + LSTM         | FaultFormer (this)         |
|---------------------------------|-----------------------|----------------------------|
| Labeled data required           | High (full dataset)   | Low (after pretraining)    |
| Transfer to new machine type    | Requires retraining   | Few-shot fine-tune (~50 examples) |
| Cross-domain (vibration→acoustic) | Poor               | Better (shared token space)|
| Inference speed                 | Fast (<10 ms)         | Slower (~50 ms on GPU)     |
| Model size                      | ~500 K params         | ~4 M params               |
| Few-shot (1% labels) F1         | ~0.60                 | ~0.82 (expected)           |

The primary advantage is **transfer learning**: once pretrained on a large unlabeled corpus, FaultFormer can be fine-tuned on a **new machine type** with as few as 20–50 labeled examples, dramatically outperforming training from scratch. This is the critical capability for scaling to a new customer's machinery without collecting months of labeled fault data.

---

## RUL Prognosis Head

The same pretrained backbone can be reused for **Remaining Useful Life (RUL) estimation** by replacing the classification head with a regression head (Linear(256, 1) + Sigmoid → fraction of remaining life in [0, 1]).

This shares backbone weights across fault classification and RUL estimation tasks, enabling multi-task learning or efficient deployment of a single model for both tasks.

---

## Tradeoffs and When to Use

**Use FaultFormer when:**
- Deploying to a **new machine type** with little labeled data
- **Cross-site generalization** is needed (different bearing geometries, sample rates)
- You have abundant unlabeled data from the target domain
- Explainability via **embedding similarity search** is valuable

**Use 1D CNN when:**
- Inference latency is critical (<10 ms requirement)
- Large labeled dataset is available for the target machine
- Edge deployment with constrained memory

**Relationship to the broader ML engine (plans/04_ml_ai_engine.md):**
FaultFormer is the "Foundation Model Pretraining" component described in Phase 3 of the ML roadmap. It complements, not replaces, the 1D CNN baseline — both are maintained in the model registry and the serving layer can route to either depending on context.

---

## File Structure

```
impl/ml_transformer_faultformer/
├── README.md              # This document
├── model.py               # PyTorch model definitions (FaultFormer, heads)
├── pretrain.py            # Self-supervised pretraining script
├── finetune.py            # Fine-tuning + few-shot evaluation
├── cross_domain.py        # Cross-domain generalization experiments
├── serve.py               # FastAPI inference server
└── requirements.txt       # Python dependencies
```

---

## References

- FaultFormer: arXiv:2312.02380 — Transformer masked pretraining for bearing fault diagnosis
- RmGPT: arXiv:2409.17604 — GPT-style rotating machinery foundation model
- MAE (Masked Autoencoders): He et al., arXiv:2111.06377 — inspiration for 75% masking ratio
- Correct CWRU evaluation protocol: arXiv:2509.22267 — bearing-wise split methodology
- Physics-informed multimodal CNN: arXiv:2508.07536
