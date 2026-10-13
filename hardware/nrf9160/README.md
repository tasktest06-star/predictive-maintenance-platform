# Nordic nRF9160 — cellular node

The only kit that needs no gateway. Also the only one whose battery life is not
limited by the cell, and the one where RAM forced two optimisations every kit
should adopt.

**Status: never flashed.** Written against nRF Connect SDK v2.x; the DSP above
`board_nrf9160.c` is verified by the shared pipeline's host tests.

---

## Why this kit exists

A BLE or sub-GHz node needs a gateway within radio range, mains power for that
gateway, and someone to install it. For a remote pump station, a tank-farm
agitator, a quarry conveyor or anything outside a building, the gateway is the
hard part of the deployment — not the sensor.

LTE-M removes it. The node talks directly to the network, and the only site
requirement is coverage.

| | nRF9160 | STWIN.box (BLE) | ESP32-S3 (Wi-Fi) |
|---|---|---|---|
| Gateway needed | **no** | yes | yes (AP) |
| Radio energy per transmission | 95 mA·s | 0.29 mA·s | 240 mA·s |
| Battery life, hourly | **9.4 y** | 1.6 y | 1.7 y |
| Cell | D LiSOCl₂ | 480 mAh LiPo | 18650 Li-ion |
| CPU | 64 MHz M33 | 160 MHz M33 | 240 MHz LX7 |
| RAM | **256 kB** | 786 kB | 512 kB + PSRAM |

The counter-intuitive row is the third. **The kit with the most expensive radio
has the longest battery life** — 5.8× the Wi-Fi node's — because LTE-M's pulse
profile (200–250 mA) is within a lithium-thionyl-chloride D-cell's capability
where Wi-Fi's 350 mA bursts are not. LiSOCl₂ self-discharges at 0.08 %/month
against Li-ion's 2.5 %, so the cell contributes 4 µA instead of 72 µA. **The
radio choice determines the chemistry, and the chemistry determines the life.**

---

## Bill of materials

| Item | Part | Qty | Notes |
|---|---|---:|---|
| Dev kit | **nRF9160 DK** (nRF9160-DK) | 1 | ~€140. Includes the LTE antenna and a SIM slot |
| SIM | iBasis (bundled) or any LTE-M profile | 1 | The bundled SIM is for evaluation; a commercial deployment needs its own APN |
| Accelerometer | IIS3DWB (STEVAL-MKI208V1K) | 1 | On SPI3 — **not** SPI2, which collides with the Arduino header |
| Cell | Tadiran TL-5930 D, 19 Ah nominal | 1 | Derated to **3600 mAh usable** at LTE-M pulse currents — see below |
| Capacitor | 1 F hybrid-layer or 2× 470 µF low-ESR | 1 | **Required.** A LiSOCl₂ D-cell cannot source 250 mA alone |
| Mount | M8 stud | 1 | |
| Enclosure | IP67, non-metallic window for the antenna | 1 | A metal enclosure makes this kit a BLE kit without a gateway |

### The capacitor is not optional

A LiSOCl₂ D-cell has high internal resistance — tens of ohms once passivated —
and LTE-M draws 200–250 mA for tens of milliseconds. Without a bulk capacitor
the terminal voltage collapses, the modem browns out mid-transmission, and the
node reboots. The classic symptom is a device that works on a bench supply and
fails in the field exactly when it tries to report something.

The 19 Ah nameplate is derated to 3600 mAh usable in the energy model for the
same reason: pulse loading, not capacity, is the limit.

---

## The two RAM optimisations

256 kB total, of which Zephyr takes ~25 kB and the Nordic modem library ~40 kB.
That leaves ~190 kB. The pipeline layout the STWIN.box uses with 786 kB to spare
needs ~168 kB, which would leave 20 kB for everything else — not viable.

| | Naive | This kit |
|---|---:|---:|
| Raw int16 FIFO staging | 49 152 | **0** |
| Window + twiddle + bit-reversal | 46 096 | **0** (flash) |
| Acquisition `float[8192]` | 32 768 | 32 768 |
| Envelope `float[8192]` | 32 768 | 32 768 |
| FFT in place | 16 384 | 16 384 |
| Magnitude + spectrum | 16 392 | 16 392 |
| Chunk scratch `int16[512×3]` | — | 3 072 |
| Model scratch | 1 024 | 1 024 |
| **Total** | **~168 kB** | **~101 kB** |

**1. No raw staging buffer.** The other kits read the FIFO into an int16 block
and convert to float afterwards. Here each 512-sample FIFO chunk is converted as
it arrives, straight into the caller's float buffer, through a 3 kB scratch that
is reused. Saves 48 kB and costs nothing — the conversion was going to happen
anyway. Every kit should do this; this one had to.

**2. Tables in flash.** `PDM_TABLES_IN_FLASH=1` with
`cdsp/tools/gen_tables.py`. 46 kB of a 1 MB part buys 46 kB of a 256 kB part,
and removes ~40 ms of boot-time transcendentals. Costs ~3 % on the FFT because
flash reads are a couple of cycles slower on the M33 bus.

`cdsp/tests/test_flash_tables.c` proves the two builds agree: 0 of 43 features
outside 5e-3, worst 2.4e-6. That matters because the two builds get deployed to
different kits on the same fleet, and a drift would make every threshold tuned
on one subtly wrong on the other.

---

## Energy

```
$ python3 ../../tools/power_budget.py --kit nrf9160
    settle      0.3000 mA.s    0.3 %
    acquire     4.4000 mA.s    4.4 %
    compute     0.0653 mA.s    0.1 %
    radio      95.3516 mA.s   95.2 %  ######################################
    total     100.1169 mA.s
  at 3600 s interval: 39.8 uA average -> 9.37 years
    (load 39.8 uA + self-discharge 4.0 uA)
```

| Interval | Avg current | Life | +1 snapshot/day |
|---|---:|---:|---:|
| 5 min | 346 µA | 1.17 y | 1.17 y |
| 15 min | 123 µA | 3.23 y | 3.18 y |
| 1 h | 39.8 µA | **9.37 y** | 9.01 y |
| 6 h | 16.6 µA | 19.9 y | 18.3 y |
| 24 h | 13.2 µA | 23.9 y | 21.7 y |

**The radio is 95 % of the budget, so the design levers are all about
connections, not bytes.** Three consequences in the configuration:

1. **PSM with a ~10.5 h TAU, 0 s active time.** Stay registered, sleep deeply.
   An RRC connection is 95 mA·s; a full attach after losing registration is
   roughly 1000. `prj.conf` configures this explicitly, and `pdm_board_sleep_ms`
   deliberately does **not** use System OFF — it would reach ~2 µA instead of
   ~3 µA and cost an attach on every wake. Saving 1 µA to spend 900 mA·s is a
   bad trade at any interval under about a year.
2. **NB-IoT fallback disabled.** Better coverage, ~10× worse latency, and a
   modem that silently falls back changes the energy budget by an order of
   magnitude without telling anyone. If a site needs NB-IoT that is a per-device
   deployment decision, not a runtime one.
3. **Snapshots at most daily.** A 51 kB waveform is ~540 s of LTE-M airtime and
   ~2 600 mA·s — 27 acquisitions' worth of energy for one waveform.
   `APP_SNAPSHOT_MIN_INTERVAL_S` is 86400 here against 3600 on the BLE kits.

---

## This kit can measure its own sample rate

`pdm_board_acquire` reports a **measured** `fs_hz_actual`, not the nominal one.
`k_uptime_get()` runs from the 32.768 kHz LFCLK, a ±20 ppm crystal on the DK, so
timing a 307 ms acquisition to 1 ms resolution gives the rate to 0.33 % — an
order of magnitude better than the IIS3DWB's ±3 % oscillator spec.

The STWIN.box cannot do this (nothing there times the acquisition against a
crystal) and the ESP32-S3 could but does not yet. It matters because a 3 % rate
error puts the fifth BPFO harmonic 16 Hz outside a ±10.6 Hz matching window, so
`n_matched` under-reports exactly when the evidence is strongest.

**This is implemented here** — the only kit where it is.

---

## Timing

Estimates. 64 MHz is the slowest core in the lineup and the pipeline is ~2.5×
the STWIN.box's.

| Stage | Estimate |
|---|---:|
| Sensor settling | 100 ms |
| Acquisition (8192 @ 26.667 kHz) | 307 ms |
| 3 × 4096-point FFT | 8.2 ms |
| Everything else | 5.4 ms |
| **Compute total** | **~13.6 ms** |
| LTE-M RRC connection + publish | ~1 500 ms |

Compute is 3.3 % of the sensing window and 0.9 % of a transmission.
`-Wdouble-promotion` matters more here than anywhere: at 64 MHz a software-float
FFT would take ~300 ms, longer than the acquisition it is processing.

---

## Validation protocol

Stages 1–4 as in `hardware/stwinbx1/README.md`, plus three specific to cellular:

- **Brown-out under a TX pulse.** Scope the cell terminal voltage during an LTE-M
  transmission with the bulk capacitor fitted and removed. This is the failure
  that only appears in the field.
- **Attach-vs-RRC accounting.** Measure the energy of a transmission after PSM
  sleep and after a forced detach. If the ratio is not roughly 1:10, the PSM
  configuration is not taking effect — operators do not always grant the
  requested TAU.
- **Coverage survey before deployment.** RSRP at the intended mounting position,
  inside the intended enclosure. A metal enclosure costs 20–30 dB and turns a
  9-year node into a queue that never drains.

---

## Production checklist

- [ ] `tools/check_config.py` clean in CI
- [ ] Cross-compile with `west build` in CI
- [ ] Measured current under a real LTE-M transaction, with the capacitor
- [ ] Verify the operator actually grants the requested PSM TAU (`AT+CEREG?`)
- [ ] Per-device certificates **in the modem's key store** via `modem_key_mgmt` — the private key then never appears in application flash or RAM, which is the strongest key custody of any kit here
- [ ] MCUboot A/B with rollback. Non-negotiable: a bad image on a remote asset with no rollback is a truck roll, sometimes a helicopter
- [ ] Watchdog ≥ 120 s. A watchdog shorter than the worst-case LTE connection reboots the node exactly when coverage is poor — when its data matters most
- [ ] Queue wear: 1024 frames × 300 B in NVS, at one write/hour for 9 years ≈ 79 k writes. Needs the partition sized so no sector exceeds its endurance
- [ ] LiSOCl₂ passivation: a cell stored for a year needs a depassivation pulse before first use, or the first transmission browns out
- [ ] IP67 with a non-metallic antenna window

---

## Honest status

| | |
|---|---|
| Shared DSP verified | ✅ via platform/core |
| RAM budget fits with margin | ✅ ~101 kB of ~190 kB available |
| Flash tables identical to RAM tables | ✅ `make -C cdsp test-flash-tables` |
| **Measured sample rate** | ✅ implemented (only kit) |
| **Compiled for Cortex-M33** | ❌ no nRF Connect SDK in this environment |
| **Flashed** | ❌ |
| LTE energy figures | ❌ Nordic app-note arithmetic, not measured |
| `lte_transport.c`, `nvs_queue.c` | ⬜ declared, not implemented |

The LTE figures are the least certain numbers in this repository. Nordic's
application notes give RRC connection energy under their own conditions; real
networks vary by operator, band, coverage and how the operator configures PSM.
A 2× error either way would not be surprising, and it would move the 9.4-year
figure to between 5 and 19 years.
