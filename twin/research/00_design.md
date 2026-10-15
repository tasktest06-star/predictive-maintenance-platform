# A hybrid residual twin for the condition-monitoring node — design

Branch `twin/hybrid-residual-ukf`, based on `platform/core`.

## The question, stated so it can come out "no"

> Does a digital twin detect a developing fault **earlier** than the
> feature-space one-class detector this platform already ships — at the same
> false-alarm rate — or does it just cost more to run?

That is the whole commercial claim behind digital twins in condition
monitoring. It is rarely tested against a competent baseline, and the baseline
here is competent: a Mahalanobis/PCA one-class model on 63 calibrated features,
fitted on a commissioning window, calibrated on held-out healthy data, 17 µs
per frame measured on this host, with per-feature attribution for free.

The design below is arranged so that a negative result requires no rewriting.
Every arm shares the same datasets, the same windows, the same alarm logic and
the same false-alarm matching, and the twin's own structural advantages and
handicaps are each isolated by an arm of their own.

## What the twin is

### 1. A physics core (`pdmtwin/physics.py`)

Not a multibody model and not `pdmcore.sim`. A lumped forward model of the
*observable features*:

| component | physics | scales with |
|---|---|---|
| `line_1x` | residual unbalance, force = m r ω² | rotating-force state, ω² |
| `line_2x` | coupling/alignment reaction | rotating-force state, ω² |
| `line_3x` | higher coupling harmonic | rotating-force state, ω² |
| `line_elec` | 2 × line frequency, induction motor | rotating-force state |
| `floor` | broadband structural background, 1/f | floor state |
| `noise` | sensor electronic noise | **nothing** — datasheet |
| `impact` | rolling-element impacts ringing the resonances | damage × force state |

Five coefficients are identified on the healthy commissioning window. The
sensor's mount roll-off and bandwidth come from `SensorSpec` — the datasheet —
and the resonance map from `MachineSpec.resonances`, which on a real asset is a
bump test.

**The measurement model is the platform's own feature extractor.** This node
does not transmit waveforms; it transmits 63 numbers. `order.1x_g` is the RMS
inside a ±6 % band around the shaft line, which at `n_fft = 4096` is one or two
FFT bins wide, so how much of a 29.7 Hz line lands in it is a question about
window leakage, not about mechanics. Rather than reimplement that algebra, the
physics core **probes `FeatureExtractor`**: each component is synthesised alone
at unit RMS, run through the real extractor, and the gain into every channel is
recorded. Band RMS is quadrature-additive for incoherent components —
`Spectrum.rms_in_band` divides out the ENBW precisely so that it is — so the
probe yields an exact power-superposition model

```
y_c² = Σ_j M_cj L_j²
```

with `L_j` the component RMS levels in g. The probe is a per-asset measurement
calibration, done once, costing ~3 s.

### 2. A learned residual (`pdmtwin/residual.py`)

`ẑ = h_phys(x; θ) + r̂(u)`, in log space, because every observation channel is a
positive RMS whose healthy variation is multiplicative (load, mount gain,
sensitivity error are all gains). RBF kernel ridge, chosen over an MLP for one
reason: its capacity is **measurable** in closed form,

```
edf = tr( K (K + αI)⁻¹ )
```

so "how much capacity is too much" can be answered with a number another
project can reuse rather than with a hyperparameter name. An MLP width sweep is
run alongside, with the caveat that its parameter count is not its effective
capacity.

### 3. State estimation (`pdmtwin/filters.py`)

A scaled unscented Kalman filter over the twin's state at the *acquisition*
timescale (6 h), not the rotational one. The state is:

- `u_line` — log level of the rotating-force content;
- `u_floor` — log level of the broadband structural floor;
- `damage` — impact level (estimation filter only, never the detection filter).

Ornstein–Uhlenbeck dynamics on the two nuisance states rather than a random
walk, deliberately: a random walk slowly forgets the commissioning reference,
which means it adapts to a developing fault. `pdmcore.health.index` already
states that rule for the baseline ("adapt the baseline *only* while the asset is
in NORMAL"); it applies identically to a twin's state.

A bootstrap particle filter over the same model is implemented in the style of
`pdmcore.health.rul.ParticleFilterRUL` and used as a cross-check, because when
the UKF and the particle filter disagree, the UKF is wrong.

### 4. A change test on the residual (`pdmtwin/detect.py`)

One-sided CUSUM on the whitened innovation energy, referred to its χ² moments
so the statistic does not depend on how many channels survived gating. The same
CUSUM is available to the baseline, so that "sequential testing beats per-frame
testing" cannot be mistaken for "the twin beats the baseline".

## The parameterisation the simulator forced

The first design had the state as *(load, mount gain)*, the textbook pair. It is
not identifiable here, and the reason is worth recording because it contradicts
the simulator's own documentation.

`pdmcore/sim/machine.py` says:

```python
# Load scales the whole mechanical signal; resonance gain scales only the
# impact-driven high-frequency content, which is how a temperature change
# or a relaxing mount actually shows up.
x *= load_factor * resonance_gain
```

The comment describes a selective gain; the code applies a global one. The two
nuisance factors therefore multiply **the same array by the same product**, and
only `load_factor × resonance_gain` is identifiable from any measurement. No
filter can separate them, and a virtual load sensor has an information ceiling
set by the mount-gain spread, which `run_virtual.py` computes and measures
against.

The second thing the code does, which the comment does not mention, is apply
the product *before* the broadband background is added:

```python
x *= load_factor * resonance_gain
x += spec.base_noise_g * _pink_noise(n, rng)
```

So the shaft lines move with the nuisance and the structural floor does not.
Measured healthy log-standard-deviation per channel: ≈0.16 where a shaft line
dominates, ≈0.04 where the floor does. A twin state that scaled everything
would be fighting the data, which is why the state here is *(line level, floor
level)* — a decomposition the data supports and the kinematics also justifies.

**Neither of these is a bug the twin branch should fix.** Changing the
simulator's nuisance model would move every detection number in the repository.
It is a docstring that contradicts its code, recorded in
`twin/research/02_findings.md` and pinned by a test so the next person to read
the comment is not misled by it.

## What is circular, and what is not

The repository is careful about this regarding its own simulator, and the same
care is owed here.

**Circular, and declared:** the *structure* of the physics core — which lines
exist, a 1/f floor, impacts ringing resonances — was chosen knowing how
`pdmcore.sim` generates signals. On a real asset that structure has to be
discovered and getting it wrong is the normal case. So the physics core here is
about as right as a physics core ever gets: the head-to-head is **generous to
the twin**, which is the direction an adversarial test should err in.

**Not circular:** no module in `pdmtwin` calls `simulate()`; the coefficients
are fitted only to healthy frames; the damage law is a generic `level ∝ d` where
the simulator's is `0.12·√s + 4·s^1.4`, so the twin has the shape of the fault
growth wrong, which is realistic and shows up in the virtual-severity bias.

**Excluded on purpose:** `temperature_c` is in the frame, and in `pdmcore.sim`
the temperature model is `ambient + 18 + 22·min(1, severity) + N(0, 0.4)`. A
22 K span against a 0.4 K noise makes temperature a tens-of-sigma fault detector
*in the simulator* and in no plant on earth, where ambient, duty, lubrication
and the sensor's thermal path all confound it. Neither arm uses it.
`run_identifiability.py` measures the artefact so the exclusion is a number.

## The gate that fired on the platform's own default asset

The twin needs its observation channels to be real numbers. Checking that
produced the branch's incidental finding: `order.0.5x_g` and `order.1x_g` are
**identically zero** on `motor_pump_37kw`, because a ±6 % band around 29.7 Hz is
3.56 Hz wide and the FFT bin spacing is 6.25 Hz, so the band contains no bin and
`rms_in_band` returns exactly 0.0. Full table and the consequence — a
severity-0.9 looseness diagnosed as *cavitation at 0.73 confidence* — in
`twin/research/02_findings.md`.

`ObsSpec.gated` therefore refuses a channel whose band contains no FFT bin
anywhere in the asset's speed range, and prints the bin arithmetic in the
refusal. It drops `order.1x_g` on the default asset, which is why the twin
observes 9 channels and not 10.

## Cost

The twin and the baseline consume the **same 300-byte frame**, so there is no
radio-energy difference — the comparison is purely compute. `run_cost.py`
measures per-frame wall time for every arm on this host, names the host, and
reports what the latency means for gateway fan-out against the 17 µs the
Mahalanobis detector costs. The twin's identification cost (the extractor probe)
is reported separately, because it is per-asset-commissioning, not per-frame,
and on a fleet of thousands of assets a per-asset human step is the real cost of
a twin.

## Deployment position

`twin/production/twin_monitor.py` is the variant that would be deployed, behind
`.score(X)` and a frame interface, with validity-domain gates that refuse rather
than extrapolate. Whether it *should* be deployed is answered in that file and
in `twin/research/03_decision.md`, from the measured numbers, either way.
