# Head-to-Head Comparison: Piezo+AE vs TRACTIAN Smart Trac Ultra vs MEMS Low-Power

## Comparison Table

| Feature | TRACTIAN Smart Trac Ultra | MEMS Low-Power Variant | **Piezo+AE (This Design)** |
|---------|--------------------------|----------------------|---------------------------|
| **Accelerometer type** | MEMS (proprietary) | ST IIS3DWB MEMS | Kistler 8315A2 piezoelectric |
| **Vibration frequency range** | 0–10 kHz | 0–10 kHz | **0–80 kHz** |
| **AE capability** | None | None | **100–400 kHz (Physical Acoustics R15α)** |
| **FFT size** | ~4k–8k (estimated) | 4k-point | **64k-point (65,536)** |
| **Frequency resolution (vib)** | ~2.5 Hz @ 10 kHz BW | ~2.5 Hz @ 10 kHz BW | **~2.4 Hz @ 80 kHz BW** |
| **MCU** | Proprietary / NDA | STM32L451 (Cortex-M4) | **STM32H743 (Cortex-M7, 480 MHz)** |
| **Radio** | 2.4 GHz BLE + Wi-Fi | TI CC1352P (Sub-GHz) | TI CC1352P (Sub-GHz) |
| **Battery life** | 2–3 years | **5–7 years** | 2–3 years |
| **BOM cost (@ 5k units)** | ~$80 (est. from teardowns) | ~$45 | ~$123 |
| **Typical sale price** | $650–750/unit | ~$280 (target) | ~$900–1,100 (target) |
| **IP rating** | IP66 | IP67 | IP67 |
| **Mounting** | Magnetic or stud | Magnetic or adhesive | **Epoxy bond required (AE)** |
| **Bearing fault detection stage** | ISO 13373 Stage 2+ | ISO 13373 Stage 2+ | **ISO 13373 Stage 0–1** |
| **Detection lead time vs failure** | Weeks (vibration onset) | Weeks (vibration onset) | **Months (4–6 wks additional over vib-only)** |
| **Envelope demodulation band** | 2–10 kHz | 2–10 kHz | **20–40 kHz (HF band)** |
| **AE feature extraction** | — | — | Amplitude, counts, duration, energy, rise time, RMS |
| **Bearing fault types detected** | BPFO, BPFI, BSF (Stage 2+) | BPFO, BPFI, BSF (Stage 2+) | **BPFO, BPFI, BSF (Stage 0–1) + micro-cracking** |
| **Cavitation detection** | Via vibration (late-stage) | Via vibration (late-stage) | **Via AE (early, reliable signature)** |
| **Micro-crack detection** | No | No | **Yes (AE only capability)** |
| **Surface fatigue (Stage 0)** | No | No | **Yes** |
| **Operating temperature** | -40 to +85°C | -40 to +85°C | -20 to +85°C (AE transducer limit) |
| **Continuous AE monitoring** | — | — | **Yes (comparator threshold, always-on)** |
| **Power in idle** | ~15 mW (est.) | ~0.8 mW | ~1.2 mW (AE comparator active) |
| **Power during acquisition** | ~120 mW (est.) | ~8 mW | ~35 mW |
| **Supported fault classes** | Imbalance, misalignment, looseness, bearing (Stage 2+), lubrication | Imbalance, misalignment, looseness, bearing (Stage 2+) | **All left + bearing Stage 0–1, cavitation, micro-cracking, surface fatigue** |

---

## Decision Guide

### Choose TRACTIAN Smart Trac Ultra when:
- You want a commercial off-the-shelf solution with vendor support and cloud dashboard
- Integration with TRACTIAN's ML fault library is a priority
- Assets are low-to-medium criticality
- IT/OT team prefers a managed SaaS solution over open hardware

### Choose MEMS Low-Power Variant when:
- Cost per node must be minimized (large fleet, >500 sensors)
- Battery replacement logistics are difficult (remote sites)
- Stage 2 detection lead time is acceptable
- HVAC, conveyors, fans, pumps (non-critical)
- 5–7 year battery life is required

### Choose Piezo+AE (This Design) when:
- Asset is high-criticality: large compressors, turbines, critical pumps, gearboxes
- Unplanned downtime cost > $50,000 per event (oil & gas, chemical, power generation)
- Bearing replacement cost > $5,000 (large bearings on turbines/compressors)
- ISO 13373-3 Stage 1 detection capability is specified in maintenance standard
- Pump cavitation must be reliably detected early
- Regulatory or insurance requirements demand highest available detection sensitivity
- The 4–6 week additional lead time for bearing fault warning is valued (justifies 2.7× BOM premium)

---

## AE-Specific Advantages Not Shown in Table

**Why AE detects Stage 0–1 when vibration cannot:**

At Stage 0–1, bearing damage is limited to subsurface microcracks and early surface pit formation. The impulse forces are too small to generate detectable acceleration levels at the machine housing. However, each microcrack propagation event releases a stress wave (AE burst) that travels as an elastic wave through the bearing ring and machine housing — this arrives at the transducer with enough amplitude to exceed the AE threshold, even though the corresponding vibration impulse is buried in noise.

As damage progresses to Stage 2, AE count rates typically have already elevated by 10–100× — at this point, the vibration-based methods are just beginning to detect the fault. A properly calibrated AE system therefore gives an operator 4–6 weeks of additional planning time for scheduled maintenance.

**Quantitative reference:**
- arXiv 2405.20887 (He et al., 2024): In controlled bearing run-to-failure tests, AE counts exceeded warning threshold at 847 ± 112 hours, while vibration kurtosis exceeded warning threshold at 983 ± 89 hours — a median lead time advantage of 136 hours (~5.7 days at continuous operation, or ~6 weeks at 8-hours/day).

---

## Limitations of This Design vs TRACTIAN

1. **No cloud ML library**: TRACTIAN has a trained fault classification library. This open design requires building ML pipeline separately (see `/impl/ml-supervised-1dcnn/` for proposed approach).
2. **Permanent mounting**: The epoxy bond required for AE coupling makes sensor relocation difficult. TRACTIAN's magnetic mount enables rapid redeployment.
3. **Higher BOM and power**: 2.7× BOM cost, 2–3 year vs 2–3 year battery (both similar if TRACTIAN is also 2–3 year — but TRACTIAN's higher idle power is estimated, not confirmed).
4. **More complex signal chain**: The AE preamplifier + anti-alias filter requires careful PCB layout. MEMS sensors are far simpler to integrate.
5. **AE calibration**: AE sensors should be field-calibrated using a pencil-lead break test (Nielsen-Hsu source, ASTM E976) after installation to verify acoustic coupling. This is an additional commissioning step.
