# ESP32-S3 + ADXL355 — low-cost retrofit node

The cheapest node in the lineup and the one with the clearest limitation. A
DevKitC-1 plus an ADXL355 is roughly €25 of parts against €90 for the ST kit,
and it buys a genuinely better low-frequency instrument and a genuinely worse
bearing instrument.

**Status: never flashed.** Written against ESP-IDF v5.2 and the part datasheets;
the DSP above `board_esp32s3.c` is verified by the shared pipeline's host tests.

---

## Read this before choosing the sensor

The branch supports two parts over the same SPI bus, selected at build time.
**They are not substitutes.**

| | ADXL355 (default) | IIS3DWB |
|---|---|---|
| Usable bandwidth | **1 kHz** | **6.3 kHz** |
| Noise density | **25 µg/√Hz** | 75 µg/√Hz |
| Resolution | **20-bit** | 16-bit |
| Full scale | ±8 g | ±16 g |
| Operating current | **200 µA** | 1.1 mA |
| Price | ~€12 | ~€15 |

Bearing impacts excite structural resonances at **2–20 kHz**. The diagnostic
information is not at BPFO — it is in the modulation *at* BPFO of those
resonances. The ADXL355's anti-alias filter is at 1 kHz by design, so those
resonances are not merely attenuated, they are filtered out before digitisation.

The config checker says so, unprompted:

```
$ python3 tools/check_config.py
  fs 4000 Hz, n_fft 4096 (0.98 Hz), acq 4096 (1024 ms)
  envelope: decimate 2 -> 2000 Hz, 1024 pts (1.95 Hz), band 500-1000 Hz
  NOTE  the demodulation band (500-1000 Hz) is below the 2-20 kHz range where
        bearing impacts excite structural resonances. Bearing evidence from
        this configuration is weak; treat a negative result as 'not measured'
```

### What each profile is for

**ADXL355 — low-frequency instrument.** 0.98 Hz spectral resolution over a 1 s
record, a noise floor three times better than the IIS3DWB's, and four extra
bits. Excellent at:

- unbalance, misalignment, looseness, soft foot, belt faults
- ISO 20816 broadband velocity — the measurement customers and insurers ask for
- **slow machines**, where its 1 Hz resolution and low-frequency reach matter
  and where a wideband part's noise floor is the limitation instead

Use it for: fans, slow agitators, large fabricated structures, anything where
the question is "is this machine running smoothly" rather than "which bearing
element is damaged".

**IIS3DWB — bearing instrument.** Same configuration as the ST kit. Use it
whenever bearing diagnosis is the point.

```bash
tools/flash.sh              # ADXL355, low-frequency
tools/flash.sh --iis3dwb    # wideband
```

A negative bearing result from the ADXL355 profile means *not measured*, and the
frame's `tier` field tells the cloud as much. Reporting it as "no bearing fault"
would be the single most misleading thing this kit could do.

### The profiles need different harmonic counts, and a test only just caught it

| Profile | Envelope Nyquist | Harmonics | Top BPFI harmonic at 3600 rpm |
|---|---:|---:|---:|
| ADXL355 (4 kHz ÷ 2) | 1000 Hz | **3** | 975 Hz |
| IIS3DWB (26.667 kHz ÷ 8) | 1667 Hz | **5** | 1625 Hz |

The ADXL355 profile's 2× decimation puts the envelope Nyquist at 1000 Hz, which
is deliberately the sensor's *own* flat bandwidth — there is no point resolving
above a band the part has already rolled off. But BPFI on the configured bearing
is 325 Hz at 3600 rpm, so **five** harmonics reach 1625 Hz: above the decimated
Nyquist *and* above the sensor. Two of them would have aliased onto
plausible-looking lines.

`APP_BEARING_HARMONICS` is therefore per profile: 3 for the ADXL355, 5 for the
IIS3DWB.

This is a **speed** limit as much as a harmonic limit. At 1780 rpm BPFI is
161 Hz and 1000 Hz holds six harmonics, so a slower machine could use five;
3 is the 3600 rpm worst case, and the config is one number for the fleet.

**How it was missed, and how it was found.** `tests/test_kit.py` had always
checked this relationship — but its config parser was a regex over `#define`
lines, which reports whichever definition comes *last* in the file. That is the
IIS3DWB profile. So the check had been silently passing against the wrong
profile for this kit's entire history. The parser now runs the real C
preprocessor, exactly as `tools/check_config.py` already did, and the check
failed on the first run.

A test that reads the wrong configuration is worse than no test: it was
reporting a pass on a question it had never actually asked.

---

## Bill of materials

| Item | Part | Qty | ~Cost | Notes |
|---|---|---:|---:|---|
| MCU board | ESP32-S3-DevKitC-1-N8R8 | 1 | €14 | **N8R8**: 8 MB flash, 8 MB **PSRAM** — the R8 part is required |
| Accelerometer | EVAL-ADXL355 or bare ADXL355 | 1 | €12 | LGA-14, 3.3 V |
| — alternative | IIS3DWB (STEVAL-MKI208V1K) | 1 | €15 | Wideband profile |
| Cell | 18650 Li-ion, 2600 mAh + holder | 1 | €6 | **Li-ion, not LiSOCl₂** — see [Energy](#energy) |
| Regulator | 3.3 V buck, ≥500 mA | 1 | €3 | Wi-Fi peaks at 350 mA |
| Mount | M8 stud or epoxy pad | 1 | — | |
| Enclosure | IP65, 100×68×50 mm | 1 | €8 | Plastic — do not shield the antenna |
| **Total** | | | **~€43** | |

### Wiring

| ADXL355 / IIS3DWB | ESP32-S3 | GPIO |
|---|---|---|
| SCLK | SPI2 CLK | 12 |
| MOSI | SPI2 MOSI | 11 |
| MISO | SPI2 MISO | 13 |
| CS | SPI2 CS | 10 |
| VDD / VDDIO | 3V3 | — |
| GND | GND | — |
| Battery ÷2 | ADC1_CH3 | 4 |

SPI runs at 8 MHz rather than the parts' 10 MHz maximum. On a DevKitC with
jumper leads the limiting factor is the wiring, not the silicon: a long
unshielded MISO lead is the most common cause of intermittent `WHO_AM_I`
failures, and those get blamed on the sensor. Keep leads under 10 cm and ground
them alongside.

---

## Energy — and this is the kit's defining constraint

```
$ python3 ../../tools/power_budget.py --kit esp32s3
  per acquisition cycle (300 B payload):
    settle      0.1200 mA.s    0.0 %  #
    acquire    12.2880 mA.s    4.9 %  #
    compute     0.1722 mA.s    0.1 %  #
    radio     240.1465 mA.s   95.0 %  ######################################
    total     252.7267 mA.s
  dominant term: radio
  at 3600 s interval: 105.2 uA average -> 1.67 years
```

**Wi-Fi association is 240 mA·s — 1 400× the sensing and compute combined.**
Association, DHCP, a TLS handshake and a publish is ~2 s at ~120 mA average with
peaks to 350 mA. Nothing about the DSP, the sensor or the acquisition length
changes that number.

Three consequences, all visible in the configuration:

1. **Default interval is 6 hours, not 1.** `APP_DEFAULT_INTERVAL_S` is 21600.
   Sampling hourly on this kit is a decision to spend the battery, not a
   default.
2. **The queue is deep (4096 frames) and the transmit path batches.** The point
   is to amortise one association across many frames. A conventional
   one-frame-per-connection design flattens a 2600 mAh cell in under two months.
3. **Deep sleep, not light sleep.** 7 µA against 240 µA. At a 6-hour interval
   light sleep alone would be 5.2 mA·h per cycle — 40× the entire acquisition
   budget. Deep sleep loses SRAM, which is why the sequence number lives in RTC
   slow memory and `pdm_app_restore_sequence` exists.

| Interval | Avg current | Life (2600 mAh) |
|---|---:|---:|
| 5 min | 876 µA | 0.31 y |
| 15 min | 316 µA | 0.77 y |
| 1 h | 105 µA | 1.67 y |
| 6 h | 46.7 µA | **2.49 y** |
| 24 h | 37.9 µA | 2.69 y |

**Why Li-ion and not the LiSOCl₂ the other kits want.** Lithium-thionyl-chloride
has a wonderful 0.08 %/month self-discharge but a pulse-current limit of tens of
milliamps; Wi-Fi's 350 mA peaks would collapse its terminal voltage and trip the
brownout detector. Li-ion's 2.5 %/month self-discharge is 72 µA — more than the
circuit's draw above a 1-hour interval — but it can actually deliver the current.

**The honest recommendation: power this kit from mains or PoE.** A €43 node
whose battery lasts 2.5 years at a 6-hour cadence is a worse proposition than
the same node on a USB supply sampling every 15 minutes. Its real niche is a
brownfield retrofit where a socket or a PoE drop exists nearby and the asset
count is high enough that €43 versus €90 matters. On battery, the ST or Nordic
kits are better at their own jobs.

---

## Memory

The only kit here with real RAM pressure. 512 kB internal SRAM, minus ~180 kB
for Wi-Fi, LWIP and mbedTLS.

| Buffer | Bytes | Where |
|---|---:|---|
| Raw FIFO staging (4096 × 3 axes × 3 B) | 36 864 | **PSRAM** |
| Acquisition `float[4096]` | 16 384 | internal |
| FFT in-place | 16 384 | internal |
| Envelope `float[2048]` | 8 192 | internal |
| Magnitude + spectrum | 16 392 | internal |
| Window + twiddle | 32 776 | internal |
| **Pipeline, internal** | **~90 kB** | of ~330 kB available |

The staging buffer is in PSRAM; the FFT working set is not. A 4096-point
transform out of PSRAM is roughly 3× slower, and `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL`
is set to 16 kB so the heap does not quietly put Wi-Fi buffers there and collapse
throughput.

**Dual-core is required, not preferred.** `CONFIG_FREERTOS_UNICORE=n`. The
ADXL355 FIFO is 96 samples — 24 ms at 4 kHz — and the Wi-Fi task runs at a higher
priority than the application. On a single core a Wi-Fi stall of tens of
milliseconds overruns the FIFO, and the resulting block is a concatenation of two
different times whose spectrum contains a step that looks exactly like an impact.
Acquisition is pinned to core 1, the network stack to core 0, and the application
serialises radio and acquisition anyway.

---

## Timing

Estimates, not measurements. The LX7 at 240 MHz with ESP-DSP's vector FFT is
roughly 1.6× faster than the Cortex-M33 at 160 MHz on this workload.

| Stage | Estimate |
|---|---:|
| Sensor settling (ADXL355) | 10 ms |
| Acquisition (4096 @ 4 kHz) | 1 024 ms |
| 3 × 4096-point FFT | 2.1 ms |
| Everything else | 3.6 ms |
| **Compute total** | **~5.7 ms** |
| Wi-Fi associate + TLS + publish | ~2 000 ms |

Compute is 0.6 % of the acquisition window and 0.3 % of a transmission.

Note the acquisition is **1.024 s** here against 307 ms on the ST kit: the
ADXL355's 4 kHz rate needs a longer record for the same 4096-point FFT. That is
also why the spectral resolution is 0.98 Hz instead of 6.51 Hz, which is the
profile's genuine advantage.

---

## Validation protocol

Stages 1–4 as in `hardware/stwinbx1/README.md`, with one addition this kit can
do and the ST kit cannot:

**Sample-rate calibration is free here.** The ADXL355's ODR comes from an
internal oscillator with no tolerance better than ±5 % over temperature. Unlike
the STWIN.box, this board has a ±10 ppm crystal and `esp_timer`, so the true
elapsed time of an acquisition is measurable to microseconds. Dividing the sample
count by the measured interval gives the real rate, which `pdm_board_acquire`
should report through `fs_hz_actual`. **This is the cheapest accuracy win on the
kit and it is not yet implemented** — the function currently reports the nominal
rate with a comment saying so.

---

## Production checklist

- [ ] `tools/check_config.py` clean for **both** sensor profiles in CI
- [ ] Cross-compile both profiles in CI (`idf.py build`)
- [ ] Measured `fs_hz_actual` from `esp_timer` (see above)
- [ ] ADC calibration from the eFuse curve — the S3's ADC is ±5 % without it, enough for a 3.4/3.2 V decision and nowhere near a state-of-charge display
- [ ] Verify the RTC sequence survives deep sleep: the second wake must emit sequence 1, not 0. Sequence 0 twice means every frame after the first wake is deduplicated away and the node looks healthy while transmitting nothing
- [ ] Confirm the FIFO does not overrun with Wi-Fi active on the other core
- [ ] Per-device X.509 client certificates in NVS; **no shared fleet PSK**
- [ ] A/B OTA with rollback (`partitions.csv` already provides the partitions)
- [ ] Brownout detector threshold set above the Li-ion knee under a 350 mA pulse
- [ ] Antenna keep-out respected: a metal enclosure is not an option on this kit
- [ ] 2.4 GHz coexistence survey — a plant with a VFD and a dozen other Wi-Fi nodes is the realistic worst case

---

## Honest status

| | |
|---|---|
| Shared DSP verified | ✅ via platform/core |
| `app_config.h` cross-checked, both profiles | ✅ and the checker warns about the ADXL355 band |
| **Compiled for Xtensa** | ❌ no ESP-IDF in this environment |
| **Flashed** | ❌ |
| IIS3DWB acquisition path | ⬜ returns `ERR_NOT_READY`; the ADXL355 path is written |
| `mqtt_transport.c`, `nvs_queue.c` | ⬜ declared, not implemented |
| Measured rate via `esp_timer` | ⬜ identified, not implemented |

The IIS3DWB branch of `pdm_board_acquire` deliberately returns
`PDM_BOARD_ERR_NOT_READY` rather than silently falling through to the ADXL355
code path. A wrong register map produces plausible numbers; an explicit refusal
does not.
