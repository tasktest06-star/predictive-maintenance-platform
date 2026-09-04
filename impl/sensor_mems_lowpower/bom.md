# Bill of Materials — MEMS Low-Power Sensor (Variant: MEMS-Only)

Target: < $45 total BOM @ 10 000 unit volume  
Battery life target: ≥ 8 years  
Last updated: 2026-09-04

---

## Active Components

| Ref | Part Number | Manufacturer | Description | Qty | Unit Cost @ 10k | Extended |
|-----|-------------|--------------|-------------|-----|-----------------|----------|
| U1 | STM32U575CIT6 | STMicroelectronics | MCU, Cortex-M33, 160 MHz, 786 KB Flash, 256 KB SRAM, 48-LQFP, 300 nA STOP2, TrustZone | 1 | $3.85 | $3.85 |
| U2 | BMI088 | Bosch Sensortec | 6-axis MEMS IMU, ±24g accelerometer, 2 kHz ODR, FIFO, SPI/I2C, –40 to +85°C, LGA-16 | 1 | $2.10 | $2.10 |
| U3 | CC1352P1F3RGZT | Texas Instruments | Sub-GHz + BLE 5.2 dual-band RF SoC, +20 dBm Sub-GHz, Arm Cortex-M4, 48-VQFN | 1 | $4.20 | $4.20 |
| U4 | TPS62840DRLR | Texas Instruments | 750 nA quiescent DC-DC step-down, 1.8–6.5 V in, 0.4–3.6 V out, 750 mA, SOT-563 | 1 | $0.85 | $0.85 |
| U5 | TXB0104RUTR | Texas Instruments | 4-channel bi-directional logic level translator 1.2–3.6 V, 12-WQFN (BMI088 SPI level shifting) | 1 | $0.55 | $0.55 |

## Memory

| Ref | Part Number | Manufacturer | Description | Qty | Unit Cost @ 10k | Extended |
|-----|-------------|--------------|-------------|-----|-----------------|----------|
| U6 | AT25SF041B-SSHD-T | Renesas (Dialog) | 4 Mbit SPI NOR Flash, 104 MHz, 1.65–3.6 V, 8-SOIC (FOTA image storage) | 1 | $0.38 | $0.38 |

## Power — Primary Cell Battery

| Ref | Part Number | Manufacturer | Description | Qty | Unit Cost @ 10k | Extended |
|-----|-------------|--------------|-------------|-----|-----------------|----------|
| BT1 | LS 26500 | SAFT | LiSOCl₂ primary cell, 3.6 V, 7500 mAh, C-size bobbin, –60 to +85°C, 20-year shelf life | 1 | $6.50 | $6.50 |

> Note: LiSOCl₂ batteries have internal impedance that limits peak current.
> The TPS62840 buck converter and a 100 µF bulk capacitor (C10) buffer the
> 38–42 mA CC1352P TX pulses to stay within the LS 26500's recommended
> 10 mA continuous / 150 mA peak pulse specification.

## RF Components

| Ref | Part Number | Manufacturer | Description | Qty | Unit Cost @ 10k | Extended |
|-----|-------------|--------------|-------------|-----|-----------------|----------|
| ANT1 | 0433AT62A0020E | Johanson Technology | 433/868/915 MHz ceramic chip antenna, 50 Ω, 0805 | 1 | $0.42 | $0.42 |
| ANT2 | 2450AT18A100 | Johanson Technology | 2.4 GHz chip antenna (BLE 5.2), 50 Ω, 0805 | 1 | $0.32 | $0.32 |
| FL1 | BPF-A3029 | Taiyo Yuden | Sub-GHz band-pass filter, 868/915 MHz, insertion loss 1.5 dB, LTCC | 1 | $0.28 | $0.28 |
| FL2 | BPF-A2401 | Taiyo Yuden | 2.4 GHz band-pass filter, 2.4–2.5 GHz, LTCC | 1 | $0.22 | $0.22 |

## Crystals / Oscillators

| Ref | Part Number | Manufacturer | Description | Qty | Unit Cost @ 10k | Extended |
|-----|-------------|--------------|-------------|-----|-----------------|----------|
| X1 | LFXTAL036090 | IQD | 32.768 kHz crystal, RTC/LPTIM wakeup clock, –40 to +85°C, ±20 ppm, 2-pad SMD | 1 | $0.35 | $0.35 |

## Passive Components

| Ref | Part Number | Manufacturer | Description | Qty | Unit Cost @ 10k | Extended |
|-----|-------------|--------------|-------------|-----|-----------------|----------|
| C1–C9 | GRM155R71C104KA88D | Murata | 100 nF, 16 V, X7R, 0402 (decoupling, power rails) | 9 | $0.02 | $0.18 |
| C10 | GRM21BR60J107ME39L | Murata | 100 µF, 6.3 V, X5R, 0805 (CC1352P TX bulk capacitor) | 1 | $0.18 | $0.18 |
| C11–C12 | GRM1555C1H120JA01D | Murata | 12 pF, 50 V, C0G, 0402 (X1 crystal load capacitors) | 2 | $0.02 | $0.04 |
| C13–C16 | GRM155R70J105KA12D | Murata | 1 µF, 6.3 V, X7R, 0402 (VREG bypass) | 4 | $0.02 | $0.08 |
| R1–R4 | RC0402FR-071KL | Yageo | 1 kΩ, 1%, 100 mW, 0402 (pull-up/down resistors) | 4 | $0.01 | $0.04 |
| R5 | RC0402FR-0710KL | Yageo | 10 kΩ, 1%, 0402 (BMI088 CS pull-up) | 1 | $0.01 | $0.01 |
| L1 | LQM21PN2R2NGRD | Murata | 2.2 µH, 500 mA, DCR 0.29 Ω, 0805 (TPS62840 inductor) | 1 | $0.22 | $0.22 |
| FB1 | BLM21PG221SN1D | Murata | Ferrite bead 220 Ω @ 100 MHz, 0805 (power supply filtering) | 1 | $0.04 | $0.04 |

## Connectors / Mechanical

| Ref | Part Number | Manufacturer | Description | Qty | Unit Cost @ 10k | Extended |
|-----|-------------|--------------|-------------|-----|-----------------|----------|
| J1 | 1051640001 | Molex | SWD/JTAG debug header, 0.8 mm pitch, 10-pin, SMT (removed in production) | 1 | $0.45 | $0.45 |
| J2 | PTS810 SJK 250 SMTR LFS | C&K | Test point / programming pogo pin pad, 2.54 mm (battery holder contact) | 2 | $0.12 | $0.24 |
| TP1–TP8 | 5001 | Keystone | SMT test point, 1 mm diameter (UART, SWD, power rails) | 8 | $0.04 | $0.32 |

## PCB and Assembly

| Item | Description | Unit Cost @ 10k | Extended |
|------|-------------|-----------------|----------|
| PCB | 4-layer PCB, 55 × 35 mm, 0.8 mm FR4, ENIG, controlled impedance for RF traces | $2.80 | $2.80 |
| Assembly | SMT assembly + reflow + electrical test (China EMS at 10k volume) | $4.50 | $4.50 |
| Enclosure | IP65 ABS snap-fit enclosure, 60 × 40 × 28 mm (custom injection mold amortized) | $1.80 | $1.80 |
| Magnets | N42 neodymium disc magnets × 2 (15 mm OD, 3 mm thick) for sensor mounting | $0.60 | $0.60 |
| Labels / QR | Asset identification label + laser-etched QR code | $0.15 | $0.15 |

---

## BOM Cost Summary

| Category | Cost |
|----------|------|
| Active ICs (U1–U5) | $11.55 |
| Memory (U6) | $0.38 |
| Battery (BT1) | $6.50 |
| RF components | $1.24 |
| Crystal | $0.35 |
| Passives | $0.59 |
| Connectors / test points | $1.01 |
| PCB + assembly | $7.30 |
| Enclosure + magnets | $2.40 |
| Labels | $0.15 |
| **Total BOM** | **$31.47** |
| 10% yield/scrap reserve | $3.15 |
| **All-in unit cost** | **$34.62** |

> **Target met: $34.62 < $45.00 at 10 k volume.**  
> This compares favorably to the Piezo+AE design (~$80–120) while covering the
> majority of HVAC/pump/conveyor monitoring use cases.

---

## Notes

1. **Battery selection rationale**: LiSOCl₂ (SAFT LS 26500) was chosen over Li-ion
   rechargeable to eliminate the need for charge management circuitry, USB port, and
   associated waterproofing complexity.  At a 10-minute measurement interval, the
   7500 mAh capacity delivers a conservative 8+ year service life with no maintenance.

2. **TPS62840 selection**: The 750 nA quiescent current buck converter is essential —
   a conventional LDO would waste >500 µA in steady-state conversion from 3.6 V battery
   to the 1.8 V core rail, which would reduce battery life to under 2 years.

3. **CC1352P vs LoRa module**: The CC1352P was chosen over a LoRa module (e.g. RFM95W)
   because it integrates both BLE 5.2 and Sub-GHz, eliminating a second radio chip.
   The CC1352P's Arm Cortex-M4 core also handles Sub-GHz MAC and FHSS independently,
   freeing the STM32U5 from radio protocol overhead.

4. **Antenna placement**: ANT1 (Sub-GHz) and ANT2 (BLE) must be placed at opposite
   corners of the PCB with ground plane keepout.  RF trace routing should be ≤ 50 mm
   with controlled 50 Ω impedance.

5. **Certifications required before shipment**: FCC Part 15, CE RED, IC Canada,
   UKCA, RoHS, REACH.  CC1352P and STM32U5 are both pre-certified modules (CC1352P
   available in CC1352P1ELAUNCHXL certified module form for faster time to market).
