# Model cards

Every model that can influence a work order has a card. A card states what the
model was trained on, how it was evaluated, where it must not be used, and who
approved it. The registry entry (`model_version` in `cloud/sql/001_schema.sql`)
carries the same fields so a disputed work order can be traced to the exact
weights that produced it, years later.

## The honest headline

**Every number below is from simulated data.** `core/pdmcore/sim` reproduces the
spectral signatures diagnosis relies on, but it does not derive them from contact
mechanics. Models trained only on it must be validated on measured data before
they can touch a work order. That gate is stated per model and is not yet passed
by any of them.

The validation sequence, in order:

1. **Simulated** — pipeline correctness, feature sanity, alarm logic. Done.
2. **Public measured data** — CWRU and MFPT for vibration, MIMII and DCASE 2026
   Task 2 for acoustic, with bearing-wise/unit-wise partitioning. **Not done.**
3. **Pilot data** — 20–50 sensors, 2–3 sites, multi-month. The only stage that
   tests a wireless MEMS sensor on a real machine over a real degradation, which
   no public dataset covers. **Not done** (`plans/00_overview.md` Phase 2, and
   therefore on the critical path for the ML work, not parallel to it).

---

## `robust_threshold` — the one that ships first

| | |
|---|---|
| **Kind** | Max robust z-score across features |
| **Trains on** | Healthy commissioning window only |
| **Parameters** | 2 per feature (median, MAD) = 126 |
| **Inference** | ~1 µs on an M33 |
| **Status** | Default for a new site |

**Why this is the default despite being the weakest.** Every alarm reduces to
one sentence: "feature X is N robust sigmas above its commissioning
distribution." A maintenance team can check that against the machine. That is
what converts a pilot into a contract, and it is worth more than the few AUC
points a learned model adds before anyone trusts it.

**Performance.** See the shared evaluation table below — all four detectors are
measured on one dataset under one protocol, which is the only way the comparison
means anything.

**Cannot do:** find a correlation-structure anomaly (two features individually
normal but jointly impossible). **Must not be used** when the commissioning
window is under 30 frames or contains a known fault.

---

## `mahalanobis` — the default upgrade

| | |
|---|---|
| **Kind** | Gaussian with Ledoit-Wolf shrunk covariance |
| **Parameters** | 63 + 63² /2 ≈ 2 079 |
| **Status** | Candidate; needs stage 2 validation |

**The constraint that matters:** with 60 commissioning samples and 63 features
the sample covariance is singular. Shrinkage is forced upward below 3
samples/feature (`MIN_SAMPLES_PER_FEATURE`); at 0.95 samples/feature the
measured shrinkage was 0.678. Without that guard, near-collinear features
produce enormous spurious distances — which is also why `explain()` normalises
contributions to fractions of the score rather than reporting raw ones.

**Must not be used** with fewer than ~40 commissioning frames; prefer
`robust_threshold` there.

---

## `pca_residual` — best measured detector here

| | |
|---|---|
| **Kind** | Reconstruction residual outside the top-k principal subspace |
| **Parameters** | 63·k, k chosen for 95 % explained variance (typically 8–15) |
| **Status** | Candidate; needs stage 2 validation |

Residual decomposes per feature, so "why" is free. `k` is chosen from explained
variance rather than fixed, because the intrinsic dimension of a machine's
healthy feature cloud depends on how many operating points it has — a
single-speed pump 3–5, a VFD compressor sweeping 30–60 Hz 10–15.

---

## `deep_svdd` — the edge model

| | |
|---|---|
| **Kind** | One-class deep SVDD, 63-24-12 MLP, no biases |
| **Parameters** | 1 800 |
| **Size** | 7.2 kB float32 → **1.8 kB int8** |
| **Inference** | 2 052 MACs; ~26 µs plain C, ~9 µs CMSIS-NN, ~2 µs Ethos-U55 |
| **Status** | Candidate for on-device scoring |

**No bias terms, deliberately.** Deep SVDD with biases can collapse to the
constant centre and reach zero training loss having learned nothing
(hypersphere collapse, Ruff et al. 2018). The centre is fixed from the initial
forward pass and never trained, and near-zero centre coordinates are nudged away
from zero — a coordinate at exactly 0 is a free direction along which the
network can collapse. `cdsp/src/pdm_model.c` omits biases to match; a firmware
that added them would silently mis-score every frame.

---

## Shared evaluation

One dataset, one protocol, all four detectors. Regenerate with
`python3 slides/figures/make_core_figures.py` — the figure and this table come
from the same run.

**Dataset.** Simulated fleet: 3 machine classes × 3 units × (24 healthy + 6 per
fault per severity), 594 acquisitions, 9 physical units, 378 faulty / 216
healthy. Each unit gets ±3 % nameplate-speed jitter so a model cannot memorise
an exact shaft frequency.

**Protocol.** `stratified_group_split`, 35 % test, **grouped by physical unit**,
7 split seeds, `leakage_report` clean on every one. Detectors see only healthy
training frames. Reported as mean ± std across seeds.

| Detector | AUC | pAUC @ FPR 0.1 | AP | TPR @ FPR 1 % |
|---|---|---|---|---|
| `robust_threshold` | 0.904 ± 0.020 | 0.883 ± 0.026 | 0.955 ± 0.010 | 0.753 ± 0.061 |
| `mahalanobis` | 0.954 ± 0.023 | 0.938 ± 0.029 | 0.979 ± 0.011 | 0.867 ± 0.066 |
| `pca_residual` | 0.958 ± 0.036 | 0.942 ± 0.043 | 0.980 ± 0.017 | 0.871 ± 0.094 |
| `deep_svdd` | **0.966 ± 0.016** | **0.948 ± 0.023** | **0.984 ± 0.008** | **0.878 ± 0.052** |

### Read this table carefully — three things it says

**1. The top three are not separable.** `mahalanobis`, `pca_residual` and
`deep_svdd` are within one standard deviation of each other on every metric. With
9 physical units the split-to-split variation (±0.02 to ±0.09) exceeds the
between-detector difference. Ranking them on this data would be reading noise.
Choose on other grounds: explainability, parameter count, whether it fits on the
MCU.

**2. AUC compresses the difference that matters.** The AUC spread across the four
is 0.062; the spread at 1 % FPR is 0.125 — twice as large. `robust_threshold`
gives up 6 points of AUC and **12 points of recall at a usable alarm rate**. AUC
averages over false-positive rates no plant would accept, which is precisely why
`docs/06` insists on operating-point metrics.

**3. The split protocol is worth more than the model choice.** The same
`pca_residual`, the same data, a random instead of unit-grouped split:

| Split | AUC |
|---|---|
| Random (leaky) | 0.982 ± 0.005 |
| Unit-grouped | 0.958 ± 0.036 |

A 2.4-point inflation, and note the *variance* collapse — a leaky split also
looks more reproducible, which is how it survives review. This is a mild version
of the effect; arXiv 2607.16493 measures 20–60 % → 99.9 % on C-MAPSS with
window-level leakage. Our grouping is at the unit level and the simulator does
not reuse recordings, so 2.4 points is the floor of the effect, not its scale.

### Degradation behaviour

One 90-day simulated run-to-failure, `bearing_outer`, 60-frame commissioning:

| Milestone | Day |
|---|---|
| First correct diagnosis (`bearing_outer_race`) | 25 |
| `normal → watch` | 27 |
| `normal → watch` … `alarm` | 55 |
| `alarm → critical` | 67 |
| **Kurtosis first leaves the Gaussian band (> 3.5)** | **63** |

The last row is the useful one. Kurtosis is the textbook early bearing
indicator, and here the fused health index escalates **36 days before kurtosis
moves at all** — because the index also sees the high-frequency band energy,
which rises first. A programme that watches kurtosis alone loses five weeks.

---

## Rule-based diagnosis

Not a learned model, and it has a card because it influences work orders.

**7 rules** (bearing localisation, unbalance, misalignment, looseness,
lubrication distress, cavitation, ISO-severity fallback), encoding ISO 13373-3 /
ISO 18436 diagnostic practice.

**Measured on simulated data:** 0 false diagnoses on a 30-frame healthy holdout;
correct mode identified on all four bearing fault modes; correct discrimination
between unbalance and misalignment via the 2x/1x ratio.

Two design choices worth the card:

- **Multiple simultaneous diagnoses are reported, not deduplicated.** A
  misaligned coupling destroying the bearing it drives is the most common real
  pairing, and reporting only the strongest signature loses the root cause —
  which is the one that needs fixing.
- **`elevated_vibration_unclassified` exists.** "The level is high and we do not
  know why" is a legitimate output. Forcing it into the nearest fault class is
  how a programme produces confidently wrong work orders.

Two rules required a growth-over-baseline gate during development, because the
ratio alone fired on healthy machines: a machine whose 2x/1x ratio has *always*
been 0.5 has a stiff coupling, not a new misalignment.

---

## `particle_filter_rul` — prognostics

| | |
|---|---|
| **Kind** | Bootstrap particle filter, exponential degradation, 1 000 particles |
| **Output** | Median + 10/90 credible interval, or an explicit abstention |
| **Status** | Candidate; usable window is narrow |

**It abstains far more often than it answers, by design.** Gates: ≥ 12
observations, health above the normal band, growth rate ≥ 5e-3/day, R² ≥ 0.30 on
the recent 30-point window. A single RUL number is worse than none — "23 days
left" gets acted on as fact, and when the machine fails in 9 the programme loses
credibility permanently.

**Feed it an amplitude-domain indicator, not the health index.** The health index
is capped (`Z_CAP`) so the alarm logic saturates sensibly, and an exponential fit
to a saturating signal concludes the crossing already happened. The ingest
service uses `band.10_rms` normalised to its healthy median; the failure
threshold is then "20× the healthy high-frequency level", which is a statement
about the asset rather than about the index.

**Measured** on a 90-day simulated run-to-failure: abstains until day 64, then
tracks with intervals that widen and narrow sensibly (e.g. day 66: 4 d [2, 7];
day 74: 3 d [0, 14]). Conservative against the simulator's nominal endpoint,
which is expected — the threshold defines failure as "the state at which we stop
the machine", not total seizure.

**Two failure modes found and fixed, both worth knowing about:**

- Proportional-only resampling jitter meant that once the posterior collapsed it
  could never re-explore, emitting `3 d [3, 3]`. A zero-width interval is the
  most dangerous possible output because it looks maximally confident. Fixed
  with an absolute jitter floor.
- A clamped interval at zero is now reported as clamped, not narrow.

---

## Approval gate

Before a model version may create work orders:

1. Stage 2 validation passed (public measured data, unit-wise partitioned).
2. `leakage_report` clean, split protocol named in the registry row.
3. Partial AUC at the customer's agreed alarm budget, plus alarms/day/100 assets.
4. Calibration reported (ECE, Brier) if it emits a probability.
5. Detection delay on at least one run-to-failure series.
6. Named human approver recorded in `model_version.approved_by`.

No model in this repository has passed step 1.
