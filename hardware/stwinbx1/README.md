# ST STEVAL-STWINBX1 — wideband MEMS vibration node

The reference industrial node. Highest-bandwidth MEMS accelerometer in the
lineup (6.3 kHz flat), the most RAM, and the only kit whose vendor ships a
condition-monitoring reference firmware to compare against.

**Status: never flashed.** The firmware is structurally complete and its DSP is
verified against the Python reference (1 113 C checks, 80 Python tests), but no
board has been on a bench. Register values, DMA wiring and all timing figures
are review-quality, not measured. See [Honest status](#honest-status).

---

## See it work now, without the board

```bash
# The real firmware — pdm_app.c, pdm_pipeline.c, pdm_payload.c — compiled for
# the host, driven from simulated samples, emitting real 300-byte frames.
make -C ../../cdsp virtual-device
python3 -m pytest ../../core/tests/test_virtual_device.py -v

# And the full chain into the ingest service:
python3 ../../cloud/ingest/app.py --replay
```

The virtual device runs the identical application loop, pipeline and wire
encoder that this kit's firmware runs. What it does **not** exercise is
`board_stwinbx1.c`: the IIS3DWB driver, DMA, clocks, BLE and sleep modes. That
is the part that needs hardware.

---

## Bill of materials

| Item | Part | Qty | Notes |
|---|---|---:|---|
| Dev kit | **STEVAL-STWINBX1** | 1 | SensorTile Wireless Industrial Node, ~€90 |
| — MCU | STM32U585AII6Q | — | Cortex-M33, 160 MHz, 2 MB flash, **786 kB SRAM**, TrustZone |
| — Vibration | **IIS3DWB** | — | ±16 g, 26.667 kHz ODR, **6.3 kHz flat BW**, 75 µg/√Hz |
| — IMU | ISM330BX | — | 6-axis + MLC/ISPU. Unused here; see [Future channels](#future-channels) |
| — Inclinometer | IIS2ICLX | — | ±3 g high-precision. Unused |
| — Microphone | IMP23ABSU | — | Analogue MEMS, 80 kHz BW. Unused — the most valuable unused part on the board |
| — Temperature | STTS22H | — | ±0.5 °C, I²C |
| — Pressure | ILPS22QS | — | Unused |
| — Radio | BlueNRG-M2SA | — | BLE 5.2 module |
| Battery | 480 mAh LiPo | 1 | As shipped. **Replace it** — see [Energy](#energy) |
| Programmer | ST-LINK/V3 | 1 | Or any SWD probe; `tools/flash.sh` also prints an OpenOCD line |
| Mount | M8 stud + adhesive pad | 1 | Stud. **Not magnetic** — see [Mounting](#mounting) |
| Enclosure | IP65, metal, 80×60×30 mm | 1 | Not supplied with the kit |

Nothing is wired: every sensor is on the board. The only hardware task is
mounting and (strongly recommended) replacing the cell.

---

## Why this kit

The **IIS3DWB** is the reason. Its flat response reaches 6.3 kHz, which is where
bearing resonances actually live. The alternative MEMS parts in this lineup roll
off at 1–2 kHz, which is below the structural resonances that bearing impacts
excite — so they can see that a machine is rougher but cannot see *which*
bearing element is damaged.

| | IIS3DWB | ADXL355 | ISM330DHCX |
|---|---|---|---|
| Flat bandwidth | **6.3 kHz** | 1 kHz | 0.9 kHz |
| Noise density | 75 µg/√Hz | **25 µg/√Hz** | 70 µg/√Hz |
| Full scale | ±16 g | ±8 g | ±16 g |
| Resolution | 16-bit | **20-bit** | 16-bit |

The ADXL355 is quieter and finer, and the ESP32-S3 branch uses it for exactly
that reason. For bearing diagnosis, bandwidth wins: a 3.2 kHz resonance ringing
at 106 Hz is invisible to a 1 kHz sensor no matter how low its noise floor.

---

## Configuration

Everything that affects the numbers is in
[`firmware/app_config.h`](firmware/app_config.h), cross-checked against the
Python reference by `tools/check_config.py` before every build:

```
$ python3 tools/check_config.py
app_config.h: 36 definitions
  fs 26667 Hz, n_fft 4096 (6.51 Hz), acq 8192 (307 ms)
  envelope: decimate 8 -> 3333 Hz, 1024 pts (3.26 Hz), band 2500-6000 Hz
  bearing SKF6205 at 29.67 Hz: BPFO 106.3, BPFI 160.7, BSF 69.9, FTF 11.82 Hz
  NOTE  envelope tolerance floor is +-4.6 % at BPFO; acceptable but the
        +-2 % slip term is no longer dominant
config is consistent with the Python reference
```

| Parameter | Value | Why |
|---|---|---|
| `APP_FS_HZ` | 26 667 Hz | The IIS3DWB's **only** rate. No decimation — throwing away the 6.3 kHz bandwidth would make this an expensive ISM330 |
| `APP_N_FFT` | 4096 | 6.51 Hz resolution; 50 %-overlapped, 3 blocks averaged |
| `APP_ACQ_SAMPLES` | 8192 (307 ms) | **Set by the envelope stage, not the spectral one.** See below |
| `APP_ENVELOPE_N_FFT` | 1024 | 3.26 Hz after decimating by 8 |
| `APP_DEMOD_LO/HI_HZ` | 2500–6000 | The first structural resonance. Chosen by the cloud's kurtogram at commissioning and pushed down; this is the default |
| `APP_SENSOR_SETTLE_MS` | 100 | IIS3DWB turn-on time to full accuracy (DS12749) |
| `APP_SENSOR_KEEPALIVE_THRESHOLD_S` | 1 | Effectively never keep the sensor alive — see [Energy](#energy) |

### The acquisition length is set by the envelope, not the FFT

Envelope resolution is a function of record **duration**, not of decimation. A
4096-sample block at 26.667 kHz is 154 ms, so no amount of decimation gets below
6.5 Hz. BPFO here is 106.3 Hz and the harmonic matcher uses a ±2 % tolerance
with a 1.5-bin floor — at 6.5 Hz that floor alone is ±9 %, wide enough to match
a neighbouring family and call an inner-race fault an outer-race one.

8192 samples (307 ms) gives 3.26 Hz, where the floor is ±4.6 % and the ±2 % slip
term starts to matter again. The tolerance note above is the checker saying so.

The first version of this config asked for a 2048-point envelope FFT from a
4096-sample block — 512 samples into a 2048-point transform. The pipeline
correctly refused and **silently stayed at tier 1**, so the bearing block was
never computed and every frame read as "no bearing fault". Caught by
`cdsp/tests/test_pipeline.c`; there are now compile-time `#error` guards on the
relationship.

---

## Build and flash

```bash
# One-time: STM32CubeU5 (~1.5 GB, not vendored, not redistributable)
# https://www.st.com/en/embedded-software/stm32cubeu5.html

python3 tools/check_config.py                 # before anything else
python3 ../../tools/export_model.py \
    --out firmware/model/pdm_detector.h       # regenerate the int8 detector

cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
      -DSTM32CUBE_U5=$HOME/STM32Cube/Repository/STM32Cube_FW_U5_V1.5.0
cmake --build build -j
tools/flash.sh build/pdm_stwinbx1.bin
```

`-Wdouble-promotion -Werror` is not optional on this part: the M33 FPU is
single-precision only, so an accidental promotion to `double` drops into
software emulation and a 4096-point FFT goes from ~1.1 ms to ~40 ms.

---

## Memory

Measured on the host build of the shared sources; ARM Thumb-2 is typically
10–20 % smaller for this code, so treat these as an upper bound. The real
`arm-none-eabi-size` numbers go here once the kit has been built.

| | Bytes | Of available |
|---|---:|---:|
| **Flash** | | **2 MB** |
| `cdsp` library (`-Os`) | 9 644 | 0.5 % |
| Exported int8 detector | 2 431 | 0.1 % |
| Window + twiddle tables (`const`, in flash) | 36 872 | 1.8 % |
| STM32 HAL + BlueNRG stack (estimate) | ~140 000 | 7 % |
| **RAM** | | **786 kB** |
| Acquisition buffer (`float[8192]`) | 32 768 | 4.2 % |
| IIS3DWB FIFO staging (`int16[8192×3]`, SRAM2) | 49 152 | 6.3 % |
| FFT in-place buffer | 16 384 | 2.1 % |
| Envelope buffer | 32 768 | 4.2 % |
| Magnitude + spectrum | 16 392 | 2.1 % |
| Model scratch | 1 024 | 0.1 % |
| **Pipeline total** | **148 488** | **18.9 %** |

786 kB is luxurious for this workload, and it is why this kit can afford the
16384-sample acquisition (1.63 Hz envelope resolution) that the constrained kits
cannot. The 8192 default is for comparability across branches.

The FIFO staging buffer is placed in **SRAM2** by the linker script, away from
the DSP working set in SRAM1. The GPDMA reaches both; keeping them apart means a
DMA burst does not stall the core's access to the FFT buffer.

---

## Timing

**Estimates from cycle counts and CMSIS-DSP benchmarks. Not measured.**

| Stage | Estimate |
|---|---:|
| Sensor settling | 100 ms |
| Acquisition (8192 @ 26.667 kHz) | 307 ms |
| Streaming moments, 8192 samples | 0.8 ms |
| 3 × 4096-point real FFT | 3.3 ms |
| Band RMS, shape, orders, ISO | 0.5 ms |
| Band-pass + rectify + decimate | 2.8 ms |
| 1024-point envelope FFT | 0.3 ms |
| Harmonic SNR, 4 modes × 5 harmonics | 0.9 ms |
| int8 detector (1 800 MACs) | 0.03 ms |
| **Compute total** | **~8.6 ms** |
| BLE connect + 2 notifications | ~30 ms |

Compute is 2 % of the acquisition window. The DSP is not the bottleneck on this
kit and there is no case for the NPU-bearing alternatives on compute grounds
alone — see the Alif branch for what an NPU actually buys.

With CMSIS-DSP's `arm_rfft_fast_f32` substituted for `pdm_rfft` (a one-line
change — the output packing convention is deliberately identical), expect ~40 %
off the FFT line.

---

## Energy

Computed by `tools/power_budget.py`, not hand-tabulated:

```
$ python3 ../../tools/power_budget.py --kit stwinbx1
  per acquisition cycle (300 B payload):
    settle      0.3000 mA.s   15.5 %  ######
    acquire     1.2288 mA.s   63.5 %  #########################
    compute     0.1197 mA.s    6.2 %  ##
    radio       0.2852 mA.s   14.7 %  #####
    total       1.9336 mA.s
  dominant term: acquire
  at 3600 s interval: 14.5 uA average -> 1.59 years
    (load 14.5 uA + self-discharge 20.0 uA)
    NOTE: self-discharge exceeds the load -- the cell's shelf life, not this circuit, sets the replacement interval
```

### Two findings worth acting on

**1. The acquisition window dominates — not the radio, not the DSP.**
**63.5 %** of the per-cycle energy is the 307 ms the sensor and MCU spend awake
capturing samples. Most published vibration-node designs optimise the radio and
the FFT; on this kit those are 14.7 % and 6.2 %. The levers that matter are a
shorter record (costs envelope resolution) and the IIS3DWB's 1.1 mA operating
current (fixed by the part).

*An earlier version of this README said 85 %.* That came from a model which used
a flat 1000 ms "standard acquisition" for every kit while this firmware acquires
8192 samples at 26.667 kHz = 307 ms — a 3.3× error in the largest term. The
window is now derived from `acq_samples / fs_hz` and
[`tests/test_kit.py`](tests/test_kit.py) asserts it matches `app_config.h`, so
it cannot drift again. The *ordering* was right and survives; the percentage was
not.

The 100 ms settling time alone (0.30 mA·s) costs more than all the DSP (0.12
mA·s) — but **not** more than leaving the sensor powered. Settling costs 0.30
mA·s once per cycle; the IIS3DWB's 1.1 mA standing current costs that much every
0.27 s. So the sensor is powered down between acquisitions at every realistic
interval, and `APP_SENSOR_KEEPALIVE_THRESHOLD_S` is 1 s — effectively never.

An earlier version of this file claimed a ~90 s crossover and set the threshold
accordingly. That would have burned 99 mA·s per cycle instead of 0.30 — a 20×
increase in total node energy and a battery life measured in weeks. It was
caught by `tests/test_kit.py`, which recomputes the crossover from the energy
model rather than trusting the constant. The lesson generalises: a constant
derived from an estimate and never re-derived is a defect waiting to be
measured.

**2. The shipped 480 mAh LiPo is the binding constraint, not the firmware.**
Its 3 %/month self-discharge is 20 µA — *more than the circuit's 14.5 µA average
draw*. Battery life therefore saturates at ~1.6 years no matter how long the
acquisition interval:

| Interval | Avg current | Life |
|---|---:|---:|
| 60 s | 46.2 µA | 0.83 y |
| 15 min | 16.1 µA | 1.51 y |
| 1 h | 14.5 µA | 1.59 y |
| 24 h | 14.0 µA | **1.61 y** ← saturated |

Swap the cell for a lithium-thionyl-chloride AA (0.08 %/month):

```
$ python3 ../../tools/power_budget.py --kit stwinbx1 \
    --battery-mah 2400 --self-discharge-pct-month 0.08
  at 3600 s interval: 14.5 uA average -> 15.91 years
```

**1.59 → 15.9 years from the cell alone, with no firmware change.** LiSOCl₂ has
real drawbacks — passivation after long storage, and a pulse-current limit that
matters for the cellular kits but not for BLE — but for a fit-and-forget
vibration node it is the correct chemistry and the shipped LiPo is a
development-kit convenience, not a design choice.

---

## Mounting

Mount type changes what the sensor can see more than any firmware setting.

| Mount | Usable bandwidth | Verdict |
|---|---|---|
| **M8 stud into a spot-faced boss** | >10 kHz | Correct. Use this |
| Adhesive (cyanoacrylate / epoxy) | ~5 kHz | Acceptable; the practical choice on a curved housing |
| Magnet | ~2 kHz | **Throws away the point of this kit** |
| Handheld probe | <1 kHz | Trending only, never for diagnosis |

A magnetic mount rolls off above roughly 2 kHz, which removes the 2.5–6 kHz
demodulation band this kit's whole bearing-diagnosis path depends on. With a
magnet, the IIS3DWB's 6.3 kHz bandwidth is wasted and a 1 kHz part would perform
identically.

The mount type is recorded per device in the asset registry (`device.mount_type`
in `cloud/sql/001_schema.sql`) because a surprising spectrum is explained by it
more often than by the machine.

Orientation: the **radial** axis sees bearing and unbalance energy. Which of
x/y/z that is depends on how the unit was bolted on. The commissioning tool
records the axis, and the cloud cross-checks it against the gravity vector
visible in the DC level (`time.mean`) — a sensor mounted 90° from its recorded
orientation shows ~1 g where ~0 g is expected.

---

## Validation protocol

Nothing below has been run. This is the plan the kit is built against; the
`docs/hardware/stwinbx1.md` design note carries the detail.

**Stage 1 — bench, no machine.** Accelerometer self-test passes; a shaker at
known frequency and amplitude reads back within 5 %; the FFT of a 1 kHz
reference tone lands in the right bin; the frame decodes in the cloud;
`pdm_board_sleep_ms` measures within 1 % against a reference clock.

**Stage 2 — bench, instrumented rotor rig.** Known bearing defects at known
severities. Compare BPFO/BPFI/BSF detection against a calibrated piezo channel
(the CN0549 branch exists partly to be that reference). Confirm the kurtogram's
band choice agrees with the piezo's.

**Stage 3 — measured datasets.** CWRU and MFPT through the same pipeline with
bearing-wise partitioning. **This is the gate no model has passed**; until it
does, every detection number in this repository is a number about a simulator.

**Stage 4 — plant pilot.** 20–50 nodes, 2–3 sites, multi-month. The only stage
that tests a wireless MEMS node on a real machine over a real degradation, which
no public dataset covers.

The metrological limitation to close in stage 1: the IIS3DWB's internal
oscillator is ±3 %, and this board offers no way to measure the true rate (the
FIFO is untimestamped at 26.667 kHz). A 3 % error moves a 106 Hz BPFO line by
3.2 Hz, which the ±2 % matcher tolerance absorbs only just. The planned fix is a
reference tone injected through the self-test actuator at commissioning.

---

## Production checklist

- [ ] `tools/check_config.py` clean in CI
- [ ] Cross-compile in CI (**the largest open gap** — ~20 lines of workflow)
- [ ] `arm-none-eabi-size` numbers replacing the host estimates above
- [ ] Measured timing from a GPIO toggle around each stage
- [ ] Measured current with a Joulescope or an N6705; expect 20–50 % worse than the table
- [ ] Sampling-rate calibration against a reference tone
- [ ] Secure boot + signed OTA with A/B rollback (`docs/04_security.md`)
- [ ] Per-device X.509 provisioning; **no shared fleet secret**
- [ ] Flash-queue wear test: 512 frames/day × 5 years ≈ 930 k writes against the U5's 10 k cycles/page — needs the wear-levelling spare sized and proven
- [ ] IP65 enclosure with the stud thermally and mechanically coupled to the housing
- [ ] ATEX/IECEx assessment if the asset is in a classified zone (**not** a software task; a sealed-battery design is a different enclosure programme)
- [ ] EMC pre-compliance: a 160 MHz MCU and a BLE radio on a machine with a VFD is the realistic worst case

---

## Future channels

Two parts on this board are unused and both are more interesting than anything
left to optimise in the vibration path.

**IMP23ABSU analogue microphone, 80 kHz bandwidth.** Acoustic sensing is
complementary to vibration and sometimes better: the DCASE lineage
(`research/code_backed_papers_2025_2026.md`) is built on it, and it needs no
mechanical coupling at all — a microphone hears a bearing through air where an
accelerometer needs a stud into the housing. 80 kHz also reaches into the
low-ultrasonic region where lubrication distress appears earliest. The analogue
output goes to the U5's ADC, so this is firmware work, not hardware.

**ISM330BX with MLC/ISPU.** An in-sensor programmable core that can run a
decision tree on the accelerometer itself, waking the MCU only when something
changes. Given that 63.5 % of this kit's energy is the acquisition window, a
"wake me when the vibration changes" primitive in the sensor is the single
largest available energy win — potentially an order of magnitude at long
intervals.

---

## Honest status

| | |
|---|---|
| Shared DSP verified against Python | ✅ 1 113 C checks |
| Feature vector matches the cloud | ✅ 43/43 non-bearing features, ≤2e-3 |
| Wire format byte-identical | ✅ |
| Application loop exercised | ✅ via `virtual_device` on the host |
| `app_config.h` cross-checked | ✅ `tools/check_config.py` |
| **Compiled for Cortex-M33** | ❌ no `arm-none-eabi-gcc` in this environment |
| **Flashed to a board** | ❌ |
| **Timing measured** | ❌ estimates only |
| **Current measured** | ❌ datasheet arithmetic only |
| `board_stwinbx1.c` exercised | ❌ needs hardware |
| BLE transport, flash queue | ⬜ declared, not implemented (`ble_transport.c`, `flash_queue.c`) |

Two files are referenced by the build and not written: `ble_transport.c` and
`flash_queue.c`. They are the BlueNRG host-stack glue and the flash ring buffer
— both board-specific, both needing hardware to develop against, and both
listed here rather than stubbed so the gap is visible in the build rather than
hidden behind a function that returns success.
