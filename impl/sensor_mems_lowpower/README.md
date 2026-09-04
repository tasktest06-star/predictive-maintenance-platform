# MEMS-Only Low-Power Sensor Design

## Design Philosophy

This implementation targets maximum battery life and minimum BOM cost for general-purpose
industrial predictive maintenance on lower-criticality assets (HVAC fans, pumps, conveyors).
It deliberately omits acoustic emission (AE) transducers to reduce cost and power draw,
accepting the tradeoff of no sub-surface micro-crack detection in exchange for an 8+ year
maintenance-free service life and a sub-$45 unit cost at 10 k volume.

This approach directly competes with TRACTIAN's Smart Sensor on cost and longevity while
delivering comparable vibration analytics for the majority of deployment scenarios.

---

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Target battery life | **≥ 8 years** (vs TRACTIAN's 3–5 year claim) |
| Frequency range | 0 – 10 kHz (MEMS accelerometer bandwidth) |
| Vibration sensor | Bosch BMI088, 6-axis, ±24g, SPI |
| MCU | STM32U5 (Cortex-M33, 19 µA/MHz run, 300 nA STOP2 standby) |
| Radio | TI CC1352P — BLE 5.2 + Sub-GHz (868/915 MHz) |
| BOM cost | < $45 @ 10 k units |
| Active time / measurement | < 500 ms |
| Active current | ~8 mA peak |
| Standby current | < 2 µA system-level |
| Form factor | 55 × 35 × 25 mm (sensor + battery housing) |

---

## Hardware Architecture

### MCU — STM32U5

The STM32U5 (specifically STM32U575) is the cornerstone of this design's power budget:

- **Run mode**: 19 µA/MHz — allows 48 MHz FFT computation at minimal energy cost
- **STOP2 mode**: 300 nA — CPU/RAM retained, LPTIM and wake-up pins active
- **RTC wakeup**: Configurable periodic interrupt (default: every 10 minutes)
- **Cortex-M33 + TrustZone**: Enables secure key storage for device authentication
- **DMA2D + hardware divider**: Accelerates feature extraction
- **OCTOSPI**: High-speed flash for firmware OTA via Sub-GHz

### Accelerometer — Bosch BMI088

Selected for high vibration resistance and wide temperature range:

- **Axes**: 3-axis accelerometer (acc) + 3-axis gyroscope (gyro)
- **Accelerometer range**: ±3g / ±6g / ±12g / **±24g** (configured to ±24g for machinery)
- **ODR**: Up to 1600 Hz normal; 2000 Hz in high-performance mode
- **FIFO**: 1024-byte FIFO with watermark interrupt — enables DMA burst without CPU polling
- **Interface**: SPI up to 10 MHz
- **Shock tolerance**: 10 000 g / 0.2 ms — survives industrial hammer strikes
- **Temperature range**: −40°C to +85°C
- **Built-in self-test (BIST)**: Validates sensor integrity at power-on

### Radio — TI CC1352P

Dual-band radio supporting both short-range and long-range use cases:

- **BLE 5.2**: Commissioning, local dashboard streaming, smartphone direct connect
- **Sub-GHz (868/915 MHz)**: Long-range mesh to gateway; penetrates steel enclosures
- **TX power**: Up to +20 dBm (Sub-GHz), +5 dBm (BLE)
- **RX sensitivity**: −121 dBm @ 50 kbps Sub-GHz
- **Active TX/RX current**: 42 mA TX peak; kept < 50 ms per measurement burst
- **Sleep current**: 185 nA — SPI interface to STM32U5 for wakeup coordination

### Power — LiSOCl₂ Primary Cell (SAFT LS 26500)

Non-rechargeable lithium-thionyl chloride chemistry chosen for:

- **Nominal capacity**: 7500 mAh @ C/500 rate
- **Self-discharge**: < 1%/year (20-year shelf life rated)
- **Voltage**: 3.6 V nominal — matches STM32U5 VDD range
- **Operating temperature**: −60°C to +85°C
- **Bobbin construction**: High energy density, optimized for pulse loads via TPS62840 buffer

---

## Firmware Architecture — Power State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMS Sensor Power FSM                        │
│                                                                 │
│  ┌──────────┐   RTC tick    ┌──────────┐   DMA done  ┌───────┐ │
│  │  STOP2   │──────────────>│  WAKE /  │────────────>│  FFT  │ │
│  │ 300 nA   │               │ ACQUIRE  │             │ FEAT  │ │
│  │(default) │               │ 8 mA     │             │EXTRACT│ │
│  └──────────┘               └──────────┘             └───┬───┘ │
│       ↑                                                   │     │
│       │        < 50 ms TX                                 │     │
│       └──────────────────────────────────────────────────┘     │
│                    TRANSMIT via CC1352P                         │
└─────────────────────────────────────────────────────────────────┘
```

**Typical cycle (10-minute interval):**

| Phase | Duration | Current | Energy |
|-------|----------|---------|--------|
| STOP2 idle | 599.5 s | 1.8 µA | 1.08 mJ |
| Wake + BMI088 init | 5 ms | 5 mA | 0.025 mJ |
| DMA acquisition (2048 × 3-axis @ 2 kHz) | ~310 ms | 7.5 mA | 2.3 mJ |
| FFT + feature extraction | 80 ms | 12 mA | 0.96 mJ |
| Sub-GHz transmit | 45 ms | 38 mA | 1.7 mJ |
| **Total / cycle** | ~600 s | — | **~6.1 mJ** |

**Battery life estimate (7500 mAh = 97.2 kJ @ 3.6 V):**
- Energy per cycle: 6.1 mJ
- Cycles per day (10-min interval): 144
- Energy per day: 878 mJ
- Estimated life: 97 200 / 0.878 ≈ **110 750 days ÷ 365 ≈ 303 years** (theoretical)
- With 50% derating for temperature + self-discharge: **~12 years**
- Conservative estimate (2× derating): **≥ 8 years**

---

## Tradeoffs vs Piezo + AE Design

| Capability | MEMS-Only (this design) | Piezo + AE Design |
|-----------|------------------------|-------------------|
| Battery life | ≥ 8 years | 3–5 years |
| BOM cost | < $45 | ~$80–120 |
| Frequency range | 0–10 kHz | 0–100 kHz |
| Micro-crack detection | No | Yes (via AE) |
| Bearing fault detection | Yes (BPFO/BPFI/BSF) | Yes (earlier detection) |
| Imbalance / misalignment | Yes | Yes |
| Looseness detection | Yes | Yes |
| High-speed (>10 kRPM) | Limited | Full |
| Suitable assets | HVAC, pumps, conveyors, fans | Gearboxes, compressors, high-criticality |

### When to Choose MEMS-Only

- Cost-driven deployments (>100 sensor installations)
- Lower-criticality assets where early micro-crack detection is not required
- Assets running below 10 kRPM shaft speed
- HVAC systems, centrifugal pumps, conveyor drives, cooling tower fans
- Applications where 8+ year maintenance-free operation is paramount

### When to Use Piezo + AE Instead

- High-criticality assets (primary compressors, turbines, safety-critical drives)
- High-speed machinery (>10 kRPM) where MEMS bandwidth is insufficient
- Applications requiring early sub-surface defect detection
- Assets with history of catastrophic failure causing production loss > $100k/hour

---

## Software Stack

- **RTOS**: Zephyr RTOS 3.6 (Apache 2.0 license)
- **DSP**: ARM CMSIS-DSP 1.15 (arm_rfft_fast_f32, 2048-point)
- **Encoding**: CBOR (TinyCBOR) for compact MQTT payload (~200 bytes)
- **Security**: TLS 1.3 via Mbed TLS; device certificate in STM32U5 TrustZone secure storage
- **OTA**: FOTA via Sub-GHz using MCUboot + Zephyr image management

---

## Directory Structure

```
impl/sensor_mems_lowpower/
├── README.md                  (this file)
├── bom.md                     (bill of materials)
└── firmware/
    ├── CMakeLists.txt         (Zephyr build configuration)
    ├── prj.conf               (Zephyr Kconfig options)
    ├── main.c                 (power state machine, BMI088 driver, CC1352P comms)
    └── dsp_features.c         (FFT, RMS, kurtosis, bearing frequency extraction)
```
