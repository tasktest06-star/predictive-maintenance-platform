# Product 7: Electrical Monitoring
> Non-invasive current, voltage, and power quality monitoring

---

## Product Overview

A dedicated non-invasive electrical monitoring device that clamps onto power supply cables to motor-driven assets. Detects motor electrical faults, power factor drift, voltage unbalance, harmonic distortion, and current imbalance — symptoms that vibration alone cannot detect. Integrates with the cloud platform to correlate electrical anomalies with mechanical fault events from the Smart Sensor.

**Key differentiator:** Combines electrical monitoring data with vibration data in the ML fusion engine to detect faults at their earliest electrical signature, before mechanical degradation becomes detectable.

---

## Functional Requirements

### FR-ELEC-01: Current Monitoring
- Three-phase current sensing (L1, L2, L3) via Rogowski coils or split-core Hall-effect CTs
- Current range: 1A–3,000A (configurable CT ratio)
- Accuracy: ±0.5% of reading + 0.1% FS
- Bandwidth: DC to 10 kHz (captures harmonics up to order ~166 at 60 Hz)
- Sampling rate: 10 kHz minimum per phase

### FR-ELEC-02: Voltage Monitoring
- Three-phase voltage (L1-N, L2-N, L3-N, L1-L2, L2-L3, L3-L1)
- Measurement via resistive voltage dividers on screw-terminal connections
- Voltage range: 0–690 VAC RMS
- Accuracy: ±0.2% RMS

### FR-ELEC-03: Computed Power Quality Parameters
- Per-phase and three-phase: Active power (W), Reactive power (VAR), Apparent power (VA), Power factor (PF), Displacement power factor
- Voltage and current THD (Total Harmonic Distortion) up to order 50
- Individual harmonic magnitudes (1st–50th)
- Voltage unbalance (%)
- Current unbalance (%)
- Frequency: 45–65 Hz auto-detect

### FR-ELEC-04: Motor Fault Detection
- Motor current signature analysis (MCSA): detect eccentricity, broken rotor bars, stator winding faults from current spectrum
- Sidebands at f ± 2sf (s=slip) for broken rotor bar detection
- Stator fault: 3rd and 5th harmonic elevation
- Bearing fault via current: BPFO/BPFI sidebands in current spectrum (for machines where vibration sensor is impractical)
- Power factor trend: gradual decline indicates insulation deterioration
- Starting current profile: compare each motor start to baseline (arcing contacts, capacitor bank degradation)

### FR-ELEC-05: Connectivity and Communication
- Sub-GHz 915 MHz (same network as vibration sensors) or WiFi 2.4/5 GHz
- MQTT publish of electrical parameters at 1 Hz continuous
- High-speed transient capture (10 kHz/phase, 10-second window) on event trigger
- AES-256 encryption

### FR-ELEC-06: Power and Physical
- Powered from the monitored circuit itself (energy harvesting from one CT coil)
- Backup: 3.6V LiSOCl₂ primary cell for monitoring during power-off periods
- IP65 enclosure (panel-mount or surface-mount DIN rail option)
- Operating temperature: -20°C to 70°C
- Certifications: UL 508A, CE (EMC + Low Voltage Directive)

### FR-ELEC-07: Integration with Platform
- Electrical metrics published to same Kafka pipeline as vibration metrics
- Diagnostic events correlated in ML engine: electrical anomaly + vibration anomaly on same asset = higher confidence fault diagnosis
- Dashboard: dedicated electrical tab on asset detail page (voltage/current trends, power quality, harmonic spectrum, fault events)

---

## Non-Functional Requirements

### NFR-ELEC-01: Safety
- Isolation: CT inputs galvanically isolated from MCU (1kV)
- Overvoltage protection: TVS diodes on all measurement inputs
- CT open-circuit protection (built into Rogowski coil design)
- UL 508A listing required before commercial release

### NFR-ELEC-02: Accuracy
- Meets IEC 61000-4-30 Class A for power quality measurements
- Revenue-grade accuracy not required (instrumentation-grade sufficient)

### NFR-ELEC-03: EMI Immunity
- IEC 61000-4-2/4/5/8 (ESD, burst, surge, magnetic field)
- Operate without error in presence of VFD switching noise

---

## Technical Architecture

### Hardware Block Diagram

```
3-Phase Current → Rogowski Coils (3×) → Integrator circuits
3-Phase Voltage → Resistive dividers → Isolation amplifiers
                           ↓
                   ADC (Sigma-Delta, 24-bit, 4-channel simultaneous)
                   e.g., ADS131M06 (Texas Instruments, 6-ch, 32 kSPS)
                           ↓
                   MCU: STM32G474 (ARM Cortex-M4, 170 MHz, FPU, CORDIC)
                     → FFT of current signal (CMSIS-DSP)
                     → Compute RMS, PF, THD, harmonics, unbalance
                     → MCSA: detect rotor bar sidebands
                           ↓
                   CC1352P Sub-GHz + BLE radio (or ESP32-S3 for WiFi option)
                           ↓
                   MQTT → Kafka pipeline → ML engine correlation
```

### MCSA (Motor Current Signature Analysis) Pipeline

```python
# Detect broken rotor bars via sideband analysis
f_slip = f_supply * slip_ratio        # e.g., 60 Hz * 0.03 = 1.8 Hz
f_lower_sideband = f_supply - 2*f_slip  # 56.4 Hz
f_upper_sideband = f_supply + 2*f_slip  # 63.6 Hz

# Extract sideband magnitude from current FFT
# Broken rotor bar severity = sideband_dB - fundamental_dB
# Threshold: >-40 dB ratio indicates developing fault
```

---

## Implementation Phases

### Phase 0: Proof of Concept (Months 1–3)
- [ ] COTS evaluation: ADS131M06 eval board + STM32G474 Nucleo
- [ ] Capture 3-phase current from test motor; compute FFT
- [ ] Validate MCSA: inject rotor bar fault condition; detect sideband
- [ ] MQTT publish of computed parameters

### Phase 1: Alpha Hardware (Months 4–7)
- [ ] Custom PCB: ADC + MCU + power harvesting + radio
- [ ] Rogowski coil integration and integrator circuit design
- [ ] IP65 enclosure design (DIN rail mount)
- [ ] Validate accuracy vs power analyzer reference

### Phase 2: Platform Integration (Months 8–11)
- [ ] Kafka pipeline integration (same topics as vibration sensors)
- [ ] Dashboard: electrical tab in asset detail
- [ ] ML fusion: combine electrical + vibration anomaly scores
- [ ] MCSA fault detection (rotor bar, eccentricity) in cloud ML engine

### Phase 3: Certification (Months 12–16)
- [ ] UL 508A listing
- [ ] CE marking (EMC Directive + Low Voltage Directive)
- [ ] IEC 61000-4-30 Class A validation
- [ ] ATEX variant (for Zone 2 environments)

---

## Key GitHub Repos

| Repo | Use |
|---|---|
| [ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) | FFT of current signal for MCSA |
| [apache/kafka](https://github.com/apache/kafka) | Electrical metrics to same pipeline |
| [apache/flink](https://github.com/apache/flink) | Correlate electrical + vibration events |
| [apache/echarts](https://github.com/apache/echarts) | Harmonic spectrum display in dashboard |

---

## Key Academic Papers

| arXiv | Relevance |
|---|---|
| 2508.07536 | Physics-informed multimodal CNN — vibration + current fusion |
| 2312.10742 | Sound vs vibration comparison — context for why electrical is needed |

---

## Success Metrics

| Metric | Target |
|---|---|
| Current measurement accuracy | ±0.5% |
| Broken rotor bar detection | ≥2 broken bars detectable |
| THD measurement accuracy | ±1% absolute |
| Power harvesting sufficient for operation | 3VA from 1 CT |
| Integration with vibration fusion model | Improved F1 vs vibration-only |
| UL 508A listing | Achieved by Month 16 |
