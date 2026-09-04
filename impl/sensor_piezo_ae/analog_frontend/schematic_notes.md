# Analog Front-End Design Notes — AE Preamplifier and Signal Chain

## Overview

The AE analog front-end conditions the signal from the Physical Acoustics R15α transducer (100–400 kHz) before it enters the STM32H743 ADC. The design goals are:

1. Amplify the transducer's weak signal (µV range) to the ADC's input range (3.3V pp)
2. Reject EMI from motor drives and inverters (very high at industrial sites)
3. Anti-alias before 1 Msps ADC sampling
4. Maintain <1 µV/√Hz input-referred noise floor to detect Stage 0 AE events

---

## AE Transducer

**Part: Physical Acoustics R15α**

| Parameter | Value |
|-----------|-------|
| Resonant frequency | 150 kHz |
| Operating bandwidth | 100–400 kHz |
| Sensitivity | 60 dB re 1V/(m/s) |
| Output impedance | ~100 Ω |
| Connector | BNC or integral cable |
| Operating temp | -20 to +120°C |

The R15α is a resonant transducer optimized for bearing and structural fatigue applications. Its 150 kHz resonance maximizes sensitivity in the frequency range where rolling element bearing AE is strongest.

**Acoustic coupling**: Apply a thin layer of silicone acoustic couplant (e.g. Molykote 111 or Soundsafe gel) between the transducer face and the machined stainless steel base. This reduces the acoustic impedance mismatch between the steel structure and the transducer ceramic element.

**Mounting — critical**: The R15α must be epoxy bonded to the base plate, not magnetically mounted. Magnetic mounts introduce acoustic reflections and micro-motion at the interface that degrade AE coupling above 50 kHz. Use M4 screw-down or epoxy (Loctite EA 9462 or equivalent low-outgassing structural adhesive). This is a permanent installation — AE is not suited for clip-on sensors.

---

## Preamplifier

**Part: INA128 Instrumentation Amplifier (Texas Instruments)**

| Parameter | Value |
|-----------|-------|
| Gain | 40 dB (×100) — set by R_G = 499 Ω (G = 1 + 50kΩ/R_G) |
| Input-referred noise | 8 nV/√Hz at 1 kHz (excellent for AE frequencies) |
| Bandwidth | DC to >1 MHz at G=100 |
| CMRR | 120 dB (rejects motor EMI common-mode) |
| Supply | ±5V (use dedicated LDO, not shared with digital) |

The INA128 is configured for differential input from the transducer, providing 120 dB CMRR against common-mode noise from motor drive cables and inverters that run near the sensor mounting point.

**Gain resistor calculation:**
```
G = 1 + 50,000 / R_G
R_G = 50,000 / (G - 1) = 50,000 / 99 ≈ 499 Ω  (use 499 Ω 0.1% thin-film)
```

**Reference pin (REF)**: Connect to ADC GND reference through a low-pass RC filter (10 Ω + 100 nF) to reject digital ground noise from the MCU.

---

## Anti-Aliasing Filter

**Type: 7th-order Butterworth, Fc = 500 kHz**

Rationale: The ADC samples at 1 Msps. A Butterworth filter at 500 kHz (Nyquist) provides >80 dB attenuation at ≥1 MHz, preventing aliasing of out-of-band noise back into the 100–400 kHz AE band.

Implementation: Sallen-Key topology using OPA657 (1.6 GHz GBW, low-noise FET-input op-amp). A 7th-order filter provides steep rolloff without group delay distortion in the pass-band.

**Filter stages (7th order = 3 × 2nd-order + 1 × 1st-order):**
```
Stage 1: 1st order RC,       Fc = 500 kHz  (R=318Ω, C=1nF)
Stage 2: 2nd order Sallen-Key, Fc = 500 kHz, Q=0.532
Stage 3: 2nd order Sallen-Key, Fc = 500 kHz, Q=0.707
Stage 4: 2nd order Sallen-Key, Fc = 500 kHz, Q=1.932
```
Values computed with FilterPro (TI) or Analog Filter Wizard (ADI).

---

## Noise Floor Specification

**Target: <1 µV/√Hz input-referred noise**

Budget:
- INA128 at G=100: 8 nV/√Hz referred to input → 8 nV/√Hz RTI (excellent)
- Johnson noise of source resistance (100 Ω source): sqrt(4kTR) = 1.3 nV/√Hz at 25°C
- OPA657 anti-alias filter noise: ~4.8 nV/√Hz
- Total combined RTI (root-sum-square): ~9.4 nV/√Hz

This is well below the 1 µV/√Hz target. The dominant noise source at 100–400 kHz will be the ADC quantization noise and the transducer's self-noise.

---

## Differential Signaling from Preamp to MCU ADC

The preamplifier board (near the transducer) transmits to the MCU board via differential signal pairs (100 Ω characteristic impedance). This rejects EMI induced on the cable run between the preamp and MCU.

- **Line driver**: THAT1510 or ADI AD8130 differential line receiver at the MCU end
- **Cable**: twisted-pair, 100 Ω characteristic impedance, max 500 mm
- **Termination**: 100 Ω differential at ADC end (matched to cable impedance)

This scheme provides an additional 40–60 dB of EMI rejection on the signal path after the INA128.

---

## Decoupling Strategy

Noise coupling from the MCU's switching regulators and radio PA into the AE analog chain is a major design risk. Mitigation:

```
Each supply pin of AE analog chain:
  - 100 nF ceramic (X7R) as close as possible to IC pin (<2mm)
  - 10 µF tantalum or MLCC bulk decoupling
  - 1 Ω series ferrite bead between digital and analog supply planes

Analog GND plane:
  - Separate from digital GND plane; join at single star point near power entry
  - AE preamplifier on isolated sub-board or poured copper island
```

**PCB layout rules for AE section:**
1. AE analog section on separate PCB layer, solid copper ground reference
2. No digital signals routed under AE analog traces
3. Transducer cable shield connected to analog GND at one end only
4. INA128 + anti-alias filter within 5 mm of ADC input pins to minimize trace capacitance

---

## Vibration Channel (Piezoelectric Accelerometer Interface)

The Kistler 8305A / equivalent IEPE accelerometer uses a current-excitation scheme:

- **IEPE constant-current supply**: 2–20 mA, supplied by MCU-side IEPE driver IC (e.g. AD8244 or PCB Electronics ICP buffer)
- **Bias voltage**: ~8–12V DC bias on the IEPE signal line (AC-coupled at ADC input via 100 nF capacitor)
- **Bandwidth**: DC coupling not needed — use 1 Hz HPF to remove DC bias
- **SPI interface**: accelerometer with digital SPI output preferred over analog IEPE for this design (reduces ADC channel count requirement)

---

## Signal Chain Block Diagram (Text)

```
[R15α AE transducer]
       |  (coax or shielded twisted pair)
       v
[INA128 — 40 dB gain, diff input]
       |
       v
[7th-order Butterworth AAF — Fc=500 kHz]
       |  (differential line drive, 100Ω)
       v
[AD8130 diff receiver at MCU]
       |
       v
[STM32H743 ADC1 — 1 Msps, 12-bit]
       |
       v
[DMA → AE burst buffer → ae_features.c]

[Kistler 8305A accelerometer]
       |  (SPI digital)
       v
[STM32H743 SPI1 — 5 MHz]
       |
       v
[DMA → vibration buffer → vibration_hf.c]
```

---

## Standards and References

- **ISO 22096**: Condition monitoring and diagnostics of machines — Acoustic emission (primary AE standard for machinery)
- **ASTM E1316**: Standard Terminology for Nondestructive Examinations (defines amplitude dBAE, counts, duration, energy, rise time)
- **INA128 datasheet**: SBOS051D, Texas Instruments
- **Physical Acoustics R15α datasheet**: Physical Acoustics Corp (Parker Hannifin)
- **OPA657 datasheet**: SBOS197, Texas Instruments
- **STM32H743 ADC application note**: AN5354
