# Bill of Materials — Piezoelectric + AE Sensor Variant

**Target**: ~$120 per unit at 5,000-unit volume

---

## BOM Table

| Ref | Part Number | Supplier | Qty | Unit Cost @ 5k | Extended | Description |
|-----|-------------|----------|-----|----------------|----------|-------------|
| U1 | STM32H743VIT6 | ST Micro | 1 | $8.50 | $8.50 | Cortex-M7 480 MHz MCU, 2 MB flash, 1 MB SRAM |
| U2 | CC1352P1FRGZT | Texas Instruments | 1 | $5.20 | $5.20 | Sub-GHz + 2.4 GHz multi-protocol radio (SimpleLink) |
| U3 | INA128UA | Texas Instruments | 1 | $3.80 | $3.80 | Instrumentation amplifier, 40 dB gain, AE preamp |
| U4 | OPA657N | Texas Instruments | 3 | $2.20 | $6.60 | 1.6 GHz FET-input op-amp, AAF stages |
| U5 | AD8130ARZ | Analog Devices | 1 | $2.90 | $2.90 | Differential line receiver, 270 MHz |
| U6 | TPS62840DRLR | Texas Instruments | 1 | $0.85 | $0.85 | 750 mA step-down, 60 nA quiescent, 3.3V rail |
| U7 | TPS7A4700RGWT | Texas Instruments | 1 | $1.40 | $1.40 | Ultra-low-noise LDO, 1A, ±5V analog supply for INA128 |
| U8 | MAX9064EXT | Maxim / Analog Devices | 1 | $0.95 | $0.95 | Ultra-low-power comparator, AE threshold monitor |
| X1 | R15α | Physical Acoustics (Parker) | 1 | $45.00 | $45.00 | 150 kHz resonant AE transducer, 60 dB sensitivity |
| X2 | Kistler 8315A2 | Kistler | 1 | $28.00 | $28.00 | Triaxial MEMS+piezo accelerometer, 0–80 kHz, ±60g, SPI |
| BT1 | SL-2790 | Tadiran | 1 | $4.50 | $4.50 | LiSOCl₂ D-cell, 19 Ah (larger cell for higher power budget) |
| ANT1 | 0433AT62A0020E | Johanson | 1 | $0.35 | $0.35 | 433/915 MHz ceramic chip antenna |
| J1 | CL331-0550-0-00 | TE Connectivity | 1 | $1.20 | $1.20 | M8 4-pin connector, IP67, sensor cable entry |
| PCB | Custom 4-layer, 80×60 mm | PCBWay / JLC | 1 | $3.20 | $3.20 | Main board + analog daughter board |
| XTAL1 | ABM8-32.000MHZ-B2-T | Abracon | 1 | $0.45 | $0.45 | 32 MHz MCU XTAL |
| XTAL2 | ABS07-32.768KHZ-T | Abracon | 1 | $0.30 | $0.30 | 32.768 kHz RTC crystal |
| ENC | Custom IP67 stainless housing | — | 1 | $6.00 | $6.00 | IP67 316L stainless steel housing with acoustic base |
| COUP | Molykote 111 compound | Dow Corning | 1 | $0.20 | $0.20 | Silicone acoustic couplant, R15α to base |
| MISC | Passives (R, C, L, ferrite) | — | — | $1.80 | $1.80 | Decoupling, filter, biasing components |

---

## Cost Summary

| Category | Cost |
|----------|------|
| MCU + Radio | $13.70 |
| AE transducer (R15α) | $45.00 |
| Piezo accelerometer | $28.00 |
| Analog signal chain | $15.65 |
| Power management | $3.25 |
| Enclosure + connector | $7.20 |
| PCB | $3.20 |
| Battery (LiSOCl₂ D-cell) | $4.50 |
| Crystal + passives + misc | $2.80 |
| **Total BOM (@ 5k units)** | **$123.30** |

> Note: Assembly (SMT + test) adds ~$12–18 depending on region, bringing COGS to ~$135–140 at 5k. Typical sale price for critical-asset sensors at this tier is $800–1,200 per node.

---

## Key Component Notes

### Physical Acoustics R15α ($45 at 5k)
The dominant BOM cost. At 100k+ units, factory-direct pricing can reach ~$28–32 with blanket PO. Alternatively, the Olympus V103-RM (similar spec, ~$30 direct) or Fuji Ceramics AE-900M-WB (Japanese equivalent, ~$35) can substitute.

### LiSOCl₂ D-cell (19 Ah)
Larger than the MEMS design (which uses AA-size, 2.4 Ah). Required because:
- Active power: ~35 mW (vs ~8 mW for MEMS-only)
- Vibration acquisition current: ~80 mA peak (64k FFT on STM32H743)
- AE preamplifier: ~12 mA continuous (comparator + INA128 on ±5V rails)
- Target lifetime: 2–3 years at 60-second vibration snapshots + continuous AE threshold monitoring

### STM32H743 vs MEMS design STM32L4
- $8.50 vs $3.10 — the H7 is justified by:
  - 64k-point FFT in <50 ms (vs several seconds on L4)
  - Simultaneous 1 Msps ADC DMA for AE channel
  - Larger SRAM (1 MB) for concurrent FFT + AE ring buffers

### TI CC1352P Radio
Same radio as MEMS low-power variant for common firmware radio stack. Supports both 915 MHz (US/AU) and 868 MHz (EU) in one part number.

---

## Comparison: MEMS Low-Power vs Piezo+AE BOM

| Component class | MEMS Low-Power | Piezo+AE (this) |
|----------------|----------------|-----------------|
| MCU | STM32L451 ($3.10) | STM32H743 ($8.50) |
| Accelerometer | ST IIS3DWB ($4.20) | Kistler 8315A2 ($28.00) |
| AE transducer | — | Physical Acoustics R15α ($45.00) |
| Analog chain | Simple RC + buffer ($0.80) | INA128 + AAF + diff drive ($15.65) |
| Battery | LiSOCl₂ AA ($1.80) | LiSOCl₂ D-cell ($4.50) |
| **Total BOM** | **~$45** | **~$123** |

The 2.7× cost premium is justified for critical assets where 4–6 weeks additional lead time for bearing fault detection can prevent catastrophic failures.
