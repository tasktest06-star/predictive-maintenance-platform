# Piezoelectric + Acoustic Emission (AE) Sensor — Implementation Design

## Overview

This design extends TRACTIAN-class predictive maintenance with a true acoustic emission channel, enabling fault detection 4-6 weeks earlier than vibration-only approaches (arXiv 2405.20887). It targets high-criticality assets where premium BOM cost is justified by the value of early warning.

---

## Target: Earlier Fault Detection

Standard vibration sensors (including TRACTIAN Smart Trac) detect bearing faults at ISO 13373 Stage 2+, when damage has already progressed to detectable vibration levels.

This design adds acoustic emission sensing to detect Stage 0–1:
- **Stage 0**: Subsurface microcracking and subsurface fatigue (AE only detects this)
- **Stage 1**: Surface pit formation begins (AE counts spike 4-6 weeks before vibration rises)
- **Stage 2**: Vibration spike — detectable by conventional sensors
- **Stage 3**: Severe damage, imminent failure

---

## Sensing Modalities

### 1. Piezoelectric Accelerometer (Vibration Channel)
- **Bandwidth**: 0–80 kHz triaxial
- **Range**: ±60g
- **Example part**: Kistler 8305A or equivalent IEPE accelerometer
- **Sample rate**: 160 kSps (2× Nyquist for 80 kHz)
- **Primary use**: conventional vibration analysis, high-frequency band envelope demodulation (20–40 kHz)
- **FFT resolution**: 64k-point (64,000 samples, ~0.4 s window at 160 kSps), giving ~2.4 Hz frequency resolution

### 2. Acoustic Emission Transducer (AE Channel)
- **Part**: Physical Acoustics R15α (150 kHz resonant frequency)
- **Sensitivity**: 60 dB re 1V/(m/s) (very high)
- **Bandwidth**: 100–400 kHz
- **ADC sample rate**: 1 Msps (adequate to capture 400 kHz AE without aliasing)
- **AE features extracted**:
  - **Amplitude** (dBAE): peak absolute value, log-scaled
  - **Counts**: number of threshold crossings per event
  - **Duration** (µs): time from first to last threshold crossing
  - **Energy**: integral of rectified signal over event duration
  - **Rise time** (µs): time from first threshold crossing to peak amplitude
  - **RMS (100 ms rolling)**: background AE activity level

---

## Microcontroller: STM32H743

- **Core**: Cortex-M7, 480 MHz
- **Why needed**: 64k-point FFT + 1 Msps AE ADC simultaneously requires high compute
- **Hardware AE threshold detection**: comparator peripheral (COMP) — keeps AE monitoring alive at very low power
- **DMA**: used for ADC → memory transfers without CPU involvement
- **DSP extensions**: Cortex-M7 FPU and DSP instructions (SIMD) accelerate FFT
- **Memory**: 1 MB SRAM (needed for 64k float buffers + AE ring buffer)

---

## Frequency Ranges

| Channel | Range | Purpose |
|---------|-------|---------|
| Vibration (low band) | 0–5 kHz | Imbalance, misalignment, looseness |
| Vibration (mid band) | 5–20 kHz | Bearing outer/inner race defects |
| Vibration (HF band) | 20–80 kHz | Early bearing fatigue via envelope demodulation |
| AE waveform capture | 100–400 kHz | Micro-crack propagation, surface fatigue burst events |
| AE background RMS | 100–400 kHz | Continuous surface wear monitoring |

---

## Why AE Matters: Research Basis

**arXiv 2405.20887** (He et al., 2024) demonstrated that AE features (counts, amplitude, energy) detect bearing surface fatigue 4–6 weeks before vibration-based sensors show any anomaly. The mechanism:

1. Subsurface cracks under Hertzian contact stress produce ultrasonic stress waves (AE events)
2. These waves propagate to the bearing outer ring and machine housing
3. AE transducer detects these micro-bursts — individual events are invisible in vibration spectra
4. As damage progresses, AE count rate and amplitude increase; vibration rises later when spalls are large enough to cause impulse forces

This makes AE the gold standard for Stage 0–1 detection of rolling element bearing failures.

---

## Tradeoffs vs MEMS-Only Design

| Aspect | MEMS Low-Power | Piezo + AE (This Design) |
|--------|---------------|--------------------------|
| BOM cost | ~$45 at 5k | ~$120 at 5k |
| Power consumption | ~8 mW active | ~35 mW active |
| Battery life | 5–7 years | 2–3 years |
| Frequency range | 0–10 kHz | 0–80 kHz + 100–400 kHz AE |
| Early detection lead time | Weeks (Stage 2) | Months (Stage 0–1) |
| Signal chain complexity | Simple | Complex (AE preamp + filter) |
| Mounting | Magnetic or adhesive | Epoxy bond required (AE) |
| Fault classes detected | Imbalance, misalign, bearing Stage 2+ | All of left + bearing Stage 0–1, cavitation, micro-cracking |

---

## When to Deploy This Design

**Use piezo + AE for:**
- Large compressors (reciprocating or centrifugal), turbines, critical pumps
- Assets where unplanned downtime cost exceeds $50k/event
- Rotating equipment with bearing replacement cost > $5k
- Any application requiring ISO 13373-3 Stage 1 detection capability
- Cavitation monitoring in pumps (AE signature is distinct and reliable)

**Use MEMS low-power for:**
- General-purpose condition monitoring across a large fleet
- Cost-sensitive deployments (>500 nodes)
- Assets with tolerable Stage 2 detection (most HVAC, conveyor, fan applications)

---

## References

- arXiv 2405.20887: He et al., "Acoustic Emission-Based Early Fault Detection for Rolling Element Bearings"
- ISO 22096: Condition monitoring and diagnostics of machines — Acoustic emission
- ISO 13373-3: Vibration condition monitoring — Diagnostic techniques
- Physical Acoustics R15α datasheet: 150 kHz resonant AE transducer
- STM32H743 Reference Manual: RM0433
