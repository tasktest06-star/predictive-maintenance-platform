# Product 2: Edge Firmware
> Embedded software running on the sensor MCU

---

## Product Overview

The firmware that runs on the sensor hardware MCU. Responsible for signal acquisition, real-time DSP (FFT, filtering, feature extraction), on-device ML inference, power management, wireless communication, and OTA updates. The firmware is the primary determinant of battery life, signal fidelity, and detection latency.

---

## Functional Requirements

### FR-FW-01: Signal Acquisition
- Configurable sampling rates: 500 Hz, 1 kHz, 2 kHz, 4 kHz, 8 kHz, 16 kHz, 32 kHz, 64 kHz per axis
- Triaxial simultaneous sampling (hardware-synchronized ADC channels)
- AE channel: continuous RMS monitoring + burst capture on threshold crossing
- Temperature: 1 Hz polling
- RPM: hardware capture timer; edge-detect on Hall sensor pulses

### FR-FW-02: On-Device DSP (Signal Processing)
- Real FFT: up to 65,536 points (using CMSIS-DSP arm_rfft_fast_f32)
- Spectral features computed per measurement window:
  - RMS velocity (ISO 10816-3 velocity bands)
  - Peak acceleration
  - Crest factor
  - Kurtosis
  - Skewness
  - Spectral kurtosis (per frequency band)
  - Envelope spectrum (demodulated high-frequency band)
  - Bearing defect frequencies (BPFO, BPFI, BSF, FTF) from known bearing geometry + current RPM
- FIR bandpass filter bank: 10–1,000 Hz (low-freq), 1–10 kHz (mid), 10–40 kHz (HF enveloping)
- AE features: AE RMS, counts per second, peak amplitude, duration, energy

### FR-FW-03: On-Device ML Inference
- Tier-1 (always on, <1 ms): threshold-based anomaly score from extracted features
- Tier-2 (on anomaly trigger, <50 ms): 1D CNN fault classifier (INT8 quantized, TF Lite Micro)
  - Classes: normal, unbalance, misalignment, outer-race defect, inner-race defect, ball defect, lubrication issue
  - Confidence score per class
- Tier-3 (gateway/cloud escalation): raw waveform capture + upload for deep model re-evaluation

### FR-FW-04: Power Management
- Deep sleep between measurement cycles: <5 μA current
- Wake sources: RTC alarm (scheduled), motion interrupt (from accelerometer), RPM presence
- Adaptive sampling: reduce sampling rate and frequency when RPM is zero (machine stopped)
- Power state machine: SLEEP → ACQUIRE → COMPUTE → TRANSMIT → SLEEP
- Low battery alert: transmit warning at <15% remaining

### FR-FW-05: Communication
- Publish sensor features + inference results via MQTT over Sub-GHz radio
- Payload: JSON or CBOR (configurable); CBOR preferred for bandwidth efficiency
- MQTT QoS 1 (at-least-once) with local retry queue (up to 24h offline buffering in Flash)
- Cellular fallback: auto-switch to LTE-M when Sub-GHz fails for >N consecutive attempts
- Message priority queue: alerts before routine telemetry
- BLE: local commissioning only (read sensor ID, configure RPM/bearing geometry, check signal quality)

### FR-FW-06: Configuration Management
- Remote configuration via MQTT topics: sampling rate, window size, thresholds, feature set
- Configuration stored in non-volatile memory with fallback to defaults on corruption
- Asset metadata (bearing geometry, RPM nominal, criticality) stored locally and mirrored to cloud

### FR-FW-07: OTA Firmware Updates
- Dual-bank Flash with A/B firmware partition
- OTA via MQTT or BLE; SHA-256 signed firmware packages
- Automatic rollback to previous firmware on consecutive boot failures
- Delta updates supported to minimize radio transmission time

### FR-FW-08: Diagnostics and Self-Test
- Boot-time self-test: accelerometer noise floor, AE amplifier bias, radio link check
- Watchdog timer with panic dump to Flash (last 4 KB of state before reset)
- Heartbeat MQTT publish every N minutes to confirm liveness

---

## Non-Functional Requirements

### NFR-FW-01: Real-Time Constraints
- Acquisition-to-feature publish latency: <2 seconds for standard measurement
- Alert publish latency (Tier-1 anomaly detected): <500 ms
- No sample drops during acquisition window

### NFR-FW-02: Memory Constraints
- Firmware image: <512 KB Flash (target for A/B OTA with 2 MB Flash total)
- RAM usage during FFT: <128 KB peak (use CMSIS-DSP in-place)
- TF Lite Micro model: <64 KB Flash, <32 KB RAM for inference

### NFR-FW-03: Security
- Secure boot: firmware signature verification at startup
- Unique per-device private key stored in hardware secure element (STSAFE-A100 or ATECC608)
- MQTT TLS 1.3 with mutual authentication (device certificate + server certificate)
- No plaintext secrets in Flash; all keys in secure element

### NFR-FW-04: Reliability
- MTBF target: ≥50,000 hours (no hardware failures)
- Firmware must recover from any single fault without human intervention
- Zero unrecoverable states in normal operating envelope

---

## Technical Architecture

### Firmware Stack

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer                                       │
│  main_task | acquisition_task | ml_task | comm_task     │
├─────────────────────────────────────────────────────────┤
│  Middleware                                              │
│  MQTT client (Eclipse Paho C) | TF Lite Micro           │
│  CMSIS-DSP | Ring buffer | Config store                 │
├─────────────────────────────────────────────────────────┤
│  RTOS                                                    │
│  FreeRTOS or Zephyr RTOS                                │
├─────────────────────────────────────────────────────────┤
│  HAL / Drivers                                           │
│  STM32 HAL or Zephyr device drivers                     │
│  Accelerometer SPI | AE ADC | Temp I2C | Hall GPIO      │
│  Sub-GHz radio (CC1352) | BLE (CC1352) | LTE-M (nRF9160)│
├─────────────────────────────────────────────────────────┤
│  Hardware                                                │
│  STM32H7 / STM32U5 MCU + peripherals                   │
└─────────────────────────────────────────────────────────┘
```

### RTOS Choice
**Zephyr RTOS** (preferred) — Apache 2.0 licensed; native support for STM32, Nordic; built-in BLE, LTE-M, TLS stacks; hardware-agnostic device tree; Bluetooth SIG mesh.

**FreeRTOS** (alternative) — MIT licensed; simpler; lower RAM overhead; better for ultra-low-power M33 targets.

### DSP Pipeline (per measurement cycle)
```
ADC DMA capture → circular ring buffer
  → Anti-aliasing FIR (CMSIS-DSP arm_fir_f32)
  → Windowing (Hann window, arm_mult_f32)
  → arm_rfft_fast_f32 (N-point real FFT)
  → arm_cmplx_mag_squared_f32 (power spectrum)
  → Feature extraction (RMS, kurtosis, crest factor, spectral kurtosis)
  → Bearing frequency magnitude extraction (BPFO, BPFI, BSF, FTF)
  → Envelope demodulation: bandpass → abs → low-pass → FFT
  → Assemble feature vector → publish
```

### TF Lite Micro Model (Tier-2 classifier)
- Architecture: 1D CNN (3 conv layers + 2 dense) — from arXiv 2602.09699 design
- Input: 1024-point FFT magnitude spectrum (normalized)
- Output: 7-class softmax + confidence
- Size target: <50 KB quantized INT8
- Training: CWRU + MFPT datasets with bearing-wise partitioning (arXiv 2509.22267)
- Convert: PyTorch → ONNX → TF Lite → INT8 quantization

---

## Implementation Phases

### Phase 0: Signal Chain Validation (Months 1–2)
- [ ] Port CMSIS-DSP to target MCU; benchmark FFT timing
- [ ] Implement acquisition task with DMA double-buffer
- [ ] Validate FFT output against MATLAB/scipy reference on known signals
- [ ] Implement bearing defect frequency extraction from known geometry + RPM

### Phase 1: MVP Firmware (Months 3–6)
- [ ] Full DSP pipeline: FIR → FFT → feature vector
- [ ] MQTT publish over Sub-GHz radio (Zephyr net_mgmt + MQTT client)
- [ ] Power state machine; deep sleep between cycles
- [ ] BLE commissioning: device discovery, config write, signal quality check
- [ ] Measure actual battery life with power profiler

### Phase 2: On-Device ML (Months 7–10)
- [ ] Train 1D CNN on CWRU/MFPT datasets (PyTorch); export to TF Lite INT8
- [ ] Integrate TF Lite Micro: allocate tensor arena in SRAM
- [ ] Implement Tier-1 / Tier-2 / Tier-3 escalation logic
- [ ] Adaptive sampling: throttle when machine stopped
- [ ] OTA firmware update system (A/B partition, signed packages)

### Phase 3: Security Hardening (Months 11–14)
- [ ] Secure boot (STM32 read-out protection + signature check)
- [ ] Hardware secure element integration (STSAFE or ATECC608)
- [ ] MQTT TLS 1.3 mutual auth; provision per-device certificates
- [ ] LTE-M failover and fallback logic
- [ ] Offline buffering: 24h data queued in Flash when radio unavailable

---

## Key GitHub Repos

| Repo | Use |
|---|---|
| [ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) | Core DSP: FFT, FIR, RMS, kurtosis |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | TF Lite Micro runtime (embedded) |
| [onnx/onnx](https://github.com/onnx/onnx) | Model export pipeline (PyTorch → ONNX → TF Lite) |
| [apache/tvm](https://github.com/apache/tvm) | Alternative: compile model to native MCU code for maximum speed |

---

## Key Academic Papers

| arXiv | Relevance |
|---|---|
| 2304.09100 | CNN on STM32 — validates architecture and timing targets |
| 2411.07168 | Hierarchical 3-tier inference — power reduction architecture |
| 2602.09699 | 1D CNN architecture for on-device classifier |
| 1712.06343 | Compressed VAE for edge IoT memory budgets |
| 2208.06051 | WPT + FFT + RF baseline — computationally efficient reference |
| 2509.22267 | Correct CWRU evaluation protocol (bearing-wise split) |

---

## Success Metrics

| Metric | Target |
|---|---|
| Acquisition-to-publish latency | <2 seconds |
| Alert latency (threshold breach) | <500 ms |
| On-device Tier-2 inference time | <50 ms |
| Firmware image size | <512 KB |
| TF Lite model size | <64 KB (INT8) |
| Deep sleep current | <5 μA |
| Battery life (1-hr cycle) | ≥5 years |
| Fault classification accuracy (CWRU) | >98% |
| OTA success rate | >99.9% |
