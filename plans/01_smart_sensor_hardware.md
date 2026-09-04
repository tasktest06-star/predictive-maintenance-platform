# Product 1: Smart Sensor Hardware
> Wireless multi-modal condition monitoring sensor unit

---

## Product Overview

A wireless, battery-powered condition monitoring sensor combining vibration, acoustic emission, temperature, and tachometer sensing in a single hardened industrial housing. Targets rotating machinery: motors, pumps, fans, compressors, gearboxes, conveyor drives.

**Key differentiator from TRACTIAN Smart Trac Ultra:** adds true acoustic emission (AE) sensing at 100–400 kHz for very early fatigue crack and surface defect detection; offers both permanent and magnetic mount variants.

---

## Functional Requirements

### FR-HW-01: Vibration Sensing
- Triaxial MEMS accelerometer
- Frequency range: 0–80,000 Hz (exceeds TRACTIAN's 64 kHz)
- Dynamic range: ≥70g peak
- Resolution: ≤0.5 mg RMS noise floor
- Configurable sampling: 1 kHz / 2 kHz / 4 kHz / 8 kHz / 16 kHz / 32 kHz / 64 kHz
- FFT resolution: ≥131,072 lines per axis

### FR-HW-02: Acoustic Emission Sensing
- Narrowband AE transducer: resonant peak 150 kHz (standard) or 300 kHz (optional)
- Broadband AE transducer option: 50–400 kHz flat ±3 dB
- Pre-amplifier: 40 dB gain, low-noise (<1 μV/√Hz)
- RMS energy, peak amplitude, counts, rise-time, duration outputs
- Enables detection of: early bearing fatigue, micro-cracks, friction, cavitation onset

### FR-HW-03: Temperature Sensing
- Contact thermocouple or RTD (PT100): -40°C to 150°C, ±0.5°C accuracy
- Optional non-contact IR thermopile: spot measurement, 5:1 field of view
- Resolution: 0.1°C

### FR-HW-04: RPM / Tachometer
- Hall-effect based tachometer from rotating shaft/coupling magnetic target
- Range: 1–60,000 RPM
- Resolution: 1 RPM or better at <1,000 RPM

### FR-HW-05: Connectivity
- Primary: Sub-GHz 915 MHz ISM band, IEEE 802.15.4g mesh
  - Obstructed range: ≥100 m
  - Line-of-sight range: ≥1,000 m
- Secondary: LTE-M / NB-IoT (onboard cellular modem, global eSIM)
- Optional: BLE 5.2 for local commissioning and firmware updates
- All wireless: AES-256 encryption

### FR-HW-06: Battery and Power
- Primary: 3.6V lithium thionyl chloride (LiSOCl₂) non-rechargeable
- Battery life: ≥5 years at default sampling intervals
- Ultra-low-power sleep: <5 μA quiescent current
- Battery level reporting to cloud

### FR-HW-07: Environmental Protection
- IP rating: IP69K (high-pressure wash-down)
- Operating temperature: -40°C to 125°C
- Vibration resistance: IEC 60068-2-64
- Shock resistance: IEC 60068-2-27

### FR-HW-08: Mounting
- Variant A: Permanent mount — adhesive-bonded stainless steel base (short + tall options)
- Variant B: Magnetic mount — neodymium magnet base, quick-release locking collar, field-repositionable
- Mounting interface: Ø10mm–Ø25mm selectable stud mount (1/4-28 UNF, M8, M12 thread options)

### FR-HW-09: Hazardous Area Certification
- ATEX Zone 1/2 (Ex ia IIC T4 Gb)
- IECEx equivalent
- NFPA 70 Class I Division 1 Groups C/D (NEC)
- FCC Part 15 / IC

### FR-HW-10: Physical
- Maximum dimensions: 50 × 80 × 50 mm (excluding base)
- Maximum weight: 200g (sensor) + 25g (base)
- Housing: 316L stainless steel or high-performance polymer (PEEK)
- Connector: IP69K-rated M12 4-pin (for wired AE output option)

---

## Non-Functional Requirements

### NFR-HW-01: Mechanical Signal Fidelity
- Zero resonant peaks in accelerometer frequency response below 40 kHz
- Transverse sensitivity <5% across full bandwidth
- Battery and PCB must be mechanically decoupled from measurement axis (patented TRACTIAN design insight — use compliant polymer mount)
- Validate against traceable reference per ISO 16063-21

### NFR-HW-02: EMI/RFI Immunity
- Operate without measurement artifacts near 50/60 Hz motors and VFDs
- EMC: IEC 61000-4-2 (ESD), IEC 61000-4-4 (EFT), IEC 61000-4-6 (RF immunity)

### NFR-HW-03: Regulatory
- CE marking (EU)
- UKCA marking
- FCC / IC (North America)
- RoHS compliant

### NFR-HW-04: Manufacturing
- Components: COTS availability from ≥2 distributors
- Design for automated PCB assembly (PCBA)
- BOM cost target: <$80 USD per unit at 10k volume
- ITAR-free component selection

---

## Technical Architecture

### MCU Selection
**STM32H7 series (STM32H743/H723)** — primary choice
- ARM Cortex-M7 @ 480 MHz
- Helium SIMD unavailable on H7 (requires M55), but M7 FPU sufficient for FFT
- 1 MB SRAM, 2 MB Flash
- CMSIS-DSP FFT: 4096-point FFT in <1 ms

**Alternative: STM32U5 (Cortex-M33 + TrustZone)**
- Ultra-low-power: 19 μA/MHz in Run mode, 300 nA Standby
- Better for battery life; slightly less compute

**Gateway MCU: Nordic nRF9160** (for LTE-M/NB-IoT variant)
- Integrated LTE-M/NB-IoT modem + ARM Cortex-M33 application MCU
- Built-in GPS option (nRF9161)

### Sensor ICs
- **Vibration:** Analog Devices ADXL1004 (20 kHz BW), or Bosch BMI088 (6-axis, 24g)
  - For high-frequency (>20 kHz): Kistler 8305A piezoelectric + signal conditioning
- **AE transducer:** Physical Acoustics R15α (resonant 150 kHz) or VS30-V (wideband)
  - On-board AE preamplifier: Analog Devices AD8221 or INA128 (40 dB gain)
- **Temperature:** Maxim DS18B20 (contact) or Texas Instruments TMP006 (IR non-contact)
- **Hall tachometer:** Allegro ACS725 or discrete Hall IC with NdFeB target

### Radio Module
- **Sub-GHz:** Texas Instruments CC1352P (dual-band: 868/915 MHz + 2.4 GHz BLE)
  - Supports IEEE 802.15.4g, proprietary protocols, BLE simultaneously
  - +20 dBm TX power (long range with wall penetration)
- **Cellular option:** Nordic nRF9160 SiP (LTE-M + NB-IoT + GPS)

### Power Management
- LiSOCl₂ primary cell: SAFT LS 26500 (7.7 Ah)
- TI BQ2970 for protection and fuel gauge
- DC-DC converter: TI TPS62840 (hysteretic, 96% efficiency at 10 μA)
- Wake on motion interrupt from accelerometer to duty-cycle main MCU

---

## Implementation Phases

### Phase 0: Proof of Concept (Months 1–3)
- [ ] COTS evaluation board (STM32H743 Nucleo + ADXL1004 breakout + CC1352P LaunchPad)
- [ ] Validate signal chain: measure known bearing fault (CWRU dataset frequencies)
- [ ] Implement CMSIS-DSP FFT pipeline in firmware
- [ ] Demonstrate 915 MHz MQTT publish to HiveMQ broker
- [ ] Test AE transducer with pencil-break test on steel plate

### Phase 1: Alpha Hardware (Months 4–7)
- [ ] First custom PCB: MCU + radio + sensor ICs + power management
- [ ] Design IP65-rated prototype enclosure (3D printed)
- [ ] Validate FFT accuracy vs benchtop spectrum analyzer
- [ ] AE transducer PCB integration and preamplifier tuning
- [ ] Battery life measurement with duty-cycle firmware
- [ ] BLE commissioning app prototype

### Phase 2: Beta Hardware (Months 8–12)
- [ ] Second PCB revision: thermal, EMI, and layout improvements
- [ ] Finalize stainless steel housing design (CNC machined)
- [ ] Both mount variants: adhesive base + magnetic base
- [ ] IP69K validation (pressure wash test)
- [ ] Temperature range validation (-40°C / +125°C thermal cycling)
- [ ] Shock and vibration immunity testing per IEC 60068
- [ ] FCC/CE pre-compliance testing

### Phase 3: Production Release (Months 13–18)
- [ ] ATEX/IECEx certification submission
- [ ] FCC/IC certification
- [ ] ISO 16063-21 calibration validation
- [ ] Finalize BOMs and manufacturing supply chain
- [ ] Production test fixture design

---

## Key GitHub Repos

| Repo | Use |
|---|---|
| [ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) | On-device FFT, FIR filters, RMS computation |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | TF Lite Micro for on-sensor ML inference |
| [onnx/onnx](https://github.com/onnx/onnx) | Export trained fault models for ONNX Runtime on MCU |

---

## Key Academic Papers

| arXiv | Relevance |
|---|---|
| 2405.20887 | AE + CWT image encoding for early fault detection |
| 2304.09100 | CNN on STM32 — 19 ms inference; directly applicable hardware |
| 2411.07168 | TinyML hierarchical inference — 44% power reduction architecture |
| 2312.10742 | Acoustic vs vibration comparison — justifies dual-modality design |
| 2502.11786 | Non-Gaussian noise handling for real industrial environments |

---

## Success Metrics

| Metric | Target |
|---|---|
| Frequency range | 0–80,000 Hz |
| Battery life | ≥5 years at 1-hour sampling interval |
| IP rating | IP69K |
| Hazardous area | ATEX Zone 1 / Class I Div 1 |
| Bearing fault detection (BPFO/BPFI/BSF) | >98% on CWRU/MFPT datasets |
| False positive rate | <2% per 30-day window |
| Installation time | <15 min (permanent mount), <2 min (magnetic) |
| Unit BOM cost | <$80 at 10k volume |
