# Renesas EK-RA8D1 — three-axis node with a real tachometer

The fastest compute in the lineup, and the only kit that does something with it.

**Status: never flashed.** The firmware compiles clean against a stub FSP under
`-Werror -Wconversion -Wdouble-promotion`, its DSP is pinned to the Python
reference by 1 113 generated checks, and its distinctive module — angular
resampling — is measured by a host test. No Renesas board has been on a bench.
See [Honest status](#honest-status).

---

## The argument for this part, stated honestly

A faster core does not save energy. Energy is work, and the work is fixed:

| | clock | pipeline | current | energy |
|---|---:|---:|---:|---:|
| STM32U585 (M33) | 160 MHz | 8.6 ms | 21 mA | 0.18 mA·s |
| **RA8D1 (M85 + Helium)** | **480 MHz** | **1.3 ms** | **62 mA** | **0.08 mA·s** |

Two-fold, on a term that is 6 % of the cycle. Anyone selling an M85 as a
battery-life improvement for this workload is selling arithmetic that does not
add up.

What the headroom actually buys is a **bigger workload**, and this kit spends it
on three things no other kit here can do:

1. **Three axes.** The whole pipeline runs once per axis out of a single FIFO
   burst — 3 × 1.3 ms. Misalignment, looseness and thrust problems are far
   clearer with an axial channel; bearing faults rarely need it.
2. **A real tachometer.** A counter input wired to a keyphasor, so shaft speed
   is *measured*. Every other kit's frames carry `SPEED_ESTIMATED`, and a
   diagnosis computed against a guessed speed is the first thing an analyst
   questions.
3. **Order tracking.** With a tacho the record can be resampled into the angle
   domain — the only correct way to analyse a VFD-driven machine whose speed
   moves during the record. [Measured below](#order-tracking-what-it-is-worth).

The cost of all three, from the energy model: 10 % of the cycle. That is the
trade, and it is a good one.

---

## See it work now, without the board

```bash
# The order-tracking module, measured rather than asserted:
make -C tests test-order

# The driver and the shared firmware, compiled under the real warning set
# against a stub FSP (catches types and signatures, not register values):
sh tools/syntax_check.sh

# The three-axis application path, end to end, in the virtual device:
make -C ../../cdsp virtual-device-mc
python3 -m pytest ../../core/tests/test_virtual_device.py -k multi_axis -v

# And the config property tests:
python3 -m pytest tests/test_kit.py -q
```

`test_virtual_device.py -k multi_axis` runs the **same `pdm_app.c`** this kit
runs, compiled with `APP_N_AXES=3`, and checks that each acquisition produces
three frames with channel tags 0/1/2, one shared timestamp, consecutive
sequence numbers, features in the right per-axis ratio, and one — not three —
diagnostic frames when the FIFO overruns.

---

## Bill of materials

Unlike the STWIN and the Alif kit, **this one needs wiring.** The EK-RA8D1 is a
general-purpose board aimed at graphics and AI demos; it has no accelerometer
and no radio.

| Item | Part | Qty | Notes |
|---|---|---:|---|
| Dev kit | **EK-RA8D1** | 1 | ~€100. Cortex-M85 @ 480 MHz, 2 MB flash, **1 MB SRAM**, TrustZone |
| Vibration | **IIS3DWB** breakout | 1 | Same sensor as the STWIN and Alif branches, so frames are comparable across kits |
| Tacho pickup | Optical or inductive, open-collector | 1 | The part that makes this kit different. Omron E3Z or equivalent |
| Reflective target | Retroreflective tape, 10 mm | 1 | One mark per revolution = a keyphasor |
| Radio | PMOD BLE (RN4870 / BlueNRG) | 1 | Or Wi-Fi via PMOD; the transport is behind `pdm_board_transmit` |
| Battery | 2600 mAh 18650 + holder | 1 | See [Energy](#energy) — at 2 %/month its self-discharge is 3× the circuit |
| Programmer | On-board J-Link OB | — | No external probe needed |
| Mount | M8 stud + spot-faced boss | 1 | **Three axes means orientation matters three times as much** |

### Wiring

| EK-RA8D1 | Signal | To |
|---|---|---|
| P409 (SCI0 MOSI) | SPI MOSI | IIS3DWB SDI |
| P410 (SCI0 MISO) | SPI MISO | IIS3DWB SDO |
| P411 (SCI0 SCK) | SPI SCK | IIS3DWB SCL |
| P406 | CS (GPIO) | IIS3DWB CS |
| P105 (GTIOC0A) | **tacho input capture** | Pickup output, pulled up |
| AN000 | battery sense | 2:1 divider off the cell |
| P600 | status LED | On-board |

The tacho input must be a **hardware capture** pin, not a GPIO interrupt. A
software interrupt timestamps the pulse after an unpredictable latency, and the
jitter that introduces is indistinguishable from real speed variation — which
is the exact quantity this kit exists to measure.

---

## Order tracking: what it is worth

`firmware/pdm_order.c` resamples the record at constant angular increments
rather than constant time increments, interpolating with a 4-point Catmull-Rom
kernel. `tests/test_order.c` measures what that recovers, on a synthetic signal
whose revolution marks are known in closed form:

```
$ make -C tests test-order
angular resampling: 29 Hz +4 % over 307 ms, 256 samples/rev
  ok  drift estimate recovers the sweep [%]                  3.4541
  low  order  6:  time 0.4209 (-1.50 dB) over 2 bins | angle 0.5000 (-0.00 dB) over 1 bin | gain 1.19x
  high order 40:  time 0.2603 (-5.67 dB) over 6 bins | angle 0.4995 (-0.01 dB) over 1 bin | gain 1.92x
  flat order  6:  time 0.4375 (-1.16 dB) over 2 bins | angle 0.5000 (-0.00 dB) over 1 bin | gain 1.14x
PASS  14 checks
```

**The finding is the order dependence, and it is not what this kit's design
assumed.** Smearing goes as (order × drift × record length), and at a 307 ms
record only the first factor is large:

- At **order 6** — bearing territory — a 4 % sweep costs 1.5 dB and resampling
  buys 1.19×. The 8192-point FFT has 3.26 Hz bins and the smear is ~7 Hz, which
  is barely wider than a Hann main lobe. **Order tracking is nearly pointless
  for bearing diagnosis at this record length.**
- At **order 40** — gear mesh, blade pass, electrical lines — the same drift
  costs 5.67 dB, spreads the line over 6 bins, and resampling buys 1.92× and
  puts it back in one bin.

The README originally claimed "8–10 dB at order 6". That was wrong, the test
said so, and the test is what the figure in `slides/ra8d1.tex` now plots.

Two side findings worth knowing:

- **Resampling also removes scallop loss.** The steady-shaft control shows
  −1.16 dB in the frequency domain, where the 174 Hz line falls between bins;
  the angle domain puts it exactly on an order bin. So even on a perfectly
  constant machine the angle domain is amplitude-accurate in a way a plain FFT
  is not.
- **Interpolation is cheap but not free.** Catmull-Rom was chosen over linear
  (2.4 dB attenuation at a quarter of Nyquist, against 0.3 dB) because an
  amplitude error that grows with order is indistinguishable from the smearing
  the function exists to remove.

### Configuration

| Parameter | Value | Why |
|---|---|---|
| `APP_ORDER_SAMPLES_PER_REV` | 256 | Maximum order 128; 0.125-order bins over 8 revolutions |
| `APP_ORDER_TRACK_DRIFT_PCT` | 0.5 | Below this the interpolation error is comparable to the smearing removed — measured by the steady-shaft control above |
| `APP_TACHO_MAX_JITTER_PCT` | 5.0 | Gate on the *speed reading*, inside the ±2 % harmonic tolerance it feeds |
| `APP_TACHO_PULSES_PER_REV` | 1 | Keyphasor. A 60-tooth gear target would be 60 |

### The gap, stated plainly

**The node does not ship order-domain features.** The shared feature spec is
frequency-domain and `pdm_pipeline_run` takes its sample rate from
`app_config.h` at compile time, so feeding it a resampled record would label
orders as hertz — numbers that look right and are wrong by whatever the speed
happened to be. Until the pipeline accepts a runtime rate, the node:

- flags the frame **`PDM_FRAME_SPEED_UNSTEADY`** (new in this branch), so the
  cloud widens its harmonic tolerance and discounts high orders instead of
  alarming on smearing;
- requests a raw snapshot, so the cloud — which has no compute limit — does the
  angle-domain analysis;
- computes the local order spectrum anyway, for the deployment where snapshot
  bandwidth is not available.

It deliberately does **not** set `PDM_FRAME_ORDER_TRACKED`. Claiming the angle
domain while shipping hertz would have the cloud compare against the wrong
baseline, which is worse than no flag at all.

---

## Three axes on the wire

A multi-axis node ships one frame per axis per acquisition, distinguished by a
2-bit channel field in bits 8–9 of the frame flags. Those bits were previously
always zero, so **every single-axis kit's frames are byte-identical to before
this field existed** and the schema version does not move.

What the shared app loop guarantees, and what the kit tests check:

- the three axes come from **one** FIFO burst, so they are the same instant of
  the same machine — three separate acquisitions would make cross-axis
  comparison meaningless;
- sequence numbers are **per frame**, not per acquisition: the ingest service
  deduplicates on `(device_id, sequence)`, so three axes sharing a sequence
  would have two silently discarded;
- channel 0 alone drives the health index and the alarm state machine. Letting
  an axial channel overwrite `last_score` would make the alarm hysteresis chase
  whichever axis happened to be noisiest;
- an overrun costs **one** frame, not three. All three axes come from the same
  burst, so if that burst lost samples every axis is corrupt, and emitting two
  clean-looking axes beside one flagged one would be worse than emitting
  nothing.

Two bugs this path produced while being written, both of the plausible-and-wrong
kind:

1. The shared loop powers the sensor down the moment channel 0's burst is in
   RAM. Both the host board and the real driver checked "is the sensor powered?"
   *before* serving the cached channels, so axes 1 and 2 returned `NOT_READY`
   and the node shipped **one** frame per acquisition while looking healthy.
   Caught by `test_multi_axis_emits_one_tagged_frame_per_axis`.
2. `pdm_board_accel_power(0)` cleared the retained raw block, so the extra axes
   would have re-read the sensor — three different instants. Now only
   power-*up* invalidates it.

---

## Energy

```
$ python3 ../../tools/power_budget.py --kit ra8d1
  per acquisition cycle (300 B payload):
    settle      0.3000 mA.s   10.9 %  ####
    acquire     1.8432 mA.s   67.3 %  ##########################
    compute     0.2418 mA.s    8.8 %  ###
    radio       0.3555 mA.s   13.0 %  #####
    total       2.7404 mA.s
  at 3600 s interval: 22.8 uA average -> 3.12 years
    (load 22.8 uA + self-discharge 72.2 uA)
    NOTE: self-discharge exceeds the load
```

Acquisition still dominates, as on every MEMS kit here. Compute is 8.8 % — three
times a single-axis kit's share and still small — and the radio 13 %, also 3×.
**The entire three-axis, tacho-equipped capability costs about 10 % of the cycle
energy.**

The 18650's 2 %/month self-discharge is 72 µA, more than three times the
circuit's 22.8 µA, so battery life saturates at ~3.1 years no matter what the
firmware does. A lithium-thionyl-chloride D cell (19 000 mAh, 0.08 %/month)
takes it past 20 years — but LiSOCl₂ has a pulse-current limit that a 62 mA
compute burst plus a BLE transmission can exceed, so it needs a supercapacitor
across the cell. That is a hardware design task, noted on the checklist.

### A correction that affects the other four kit branches

While building this kit the energy model was found to be using a flat
**1000 ms "standard acquisition"** for every kit, while the firmware acquires
8192 samples at 26.667 kHz = **307 ms**. Acquisition is the largest term in
every battery kit's budget, so that was a 3.3× error in the dominant term.

`tools/power_budget.py` now derives the window from each kit's `acq_samples /
fs_hz`, and `tests/test_kit.py` asserts it matches `app_config.h` so it cannot
drift again. Corrected figures:

| kit | acquire share | was | avg current | battery life | was |
|---|---:|---:|---:|---:|---:|
| stwinbx1 | 63.5 % | 85.0 % | 14.5 µA | 1.59 y | 1.55 y |
| esp32s3 | 4.9 % | 4.8 % | 105.2 µA | 1.67 y | 1.67 y |
| nrf9160 | 1.4 % | 4.4 % | 39.0 µA | 9.56 y | 9.37 y |
| alif_e7 | 66.9 % | 86.8 % | 16.5 µA | 1.50 y | 1.47 y |
| ra8d1 | 67.3 % | 87.0 % | 22.8 µA | 3.12 y | 3.09 y |

(The ESP32-S3 is unaffected because its default ADXL355 profile really does
acquire 4096 samples at 4 kHz = 1.02 s, which is where the flat 1000 ms came
from in the first place. It is radio-dominated either way.)

The battery-life figures barely moved, because on four of the five kits
self-discharge dominates the load — which was those branches' headline finding
and survives intact. The **percentage** claims in those READMEs ("85 % of the
energy is the acquisition window") were overstated; the ordering and the
conclusions were not.

---

## Memory

1 MB of SRAM is the other reason for this part, and it is the one the datasheet
does not sell. Measured on the host build of the shared sources; ARM Thumb-2 is
typically 10–20 % smaller.

| | Bytes | Of available |
|---|---:|---:|
| **Flash** | | **2 MB** |
| `cdsp` library (`-Os`) | 9 644 | 0.5 % |
| `pdm_order.c` | ~1 100 | 0.1 % |
| Exported int8 detector | 2 431 | 0.1 % |
| FSP + BLE host stack (estimate) | ~120 000 | 6 % |
| **RAM** | | **1 MB** |
| **Three-axis raw block** (`int16[8192×3]`) | 49 152 | 4.7 % |
| Channel-0 float record (for the order pass) | 32 768 | 3.1 % |
| Acquisition buffer (`float[8192]`) | 32 768 | 3.1 % |
| FFT in-place buffer | 16 384 | 1.6 % |
| Envelope buffer | 32 768 | 3.1 % |
| Twiddle + window tables (**in RAM here**) | 36 872 | 3.5 % |
| Angle-domain buffer (`float[2304]`) | 9 216 | 0.9 % |
| Magnitude + spectrum + model scratch | 17 416 | 1.7 % |
| **Pipeline total** | **227 344** | **21.7 %** |

227 kB would not fit the 256 kB kits at all — the three-axis retained block plus
the float record is 82 kB on its own. `PDM_TABLES_IN_FLASH` is **0** here
because there is no reason to pay the ~3 % transform penalty for RAM this part
has to spare.

---

## Timing

**Estimates from cycle counts and Arm's Helium figures. Not measured.**

| Stage | Per axis | × 3 axes |
|---|---:|---:|
| Streaming moments, 8192 samples | 0.2 ms | 0.6 ms |
| 3 × 4096-point real FFT (Helium) | 0.6 ms | 1.8 ms |
| Band RMS, shape, orders, ISO | 0.15 ms | 0.45 ms |
| Band-pass + rectify + decimate | 0.5 ms | 1.5 ms |
| Harmonic SNR, 4 modes × 5 harmonics | 0.2 ms | 0.6 ms |
| int8 detector (1 800 MACs) | 0.01 ms | 0.03 ms |
| Angular resampling (2304 out) | — | 0.9 ms |
| **Total** | **1.66 ms** | **5.88 ms** |

5.9 ms against a 407 ms sensing window: **1.4 %**. The compute budget is not a
constraint on this kit, which is the whole point — there is room for a
16384-point FFT (1.63 Hz envelope resolution), several bearings per node, or
cyclostationary analysis, none of which fit elsewhere in this lineup.

`-mfpu=auto` matters: `-mfpu=fpv5-sp-d16` compiles, links, produces
bit-identical numbers, passes every test in this repository, and runs the FFT
about twice as slow because it hides the vector unit. **No test can catch
this.** It is a review item, recorded in `firmware/CMakeLists.txt`.

---

## Build and flash

```bash
# One-time: e2 studio + FSP 5.x, and a generated project for the EK-RA8D1
# (~400 MB, not vendored). https://github.com/renesas/fsp
#
# In the configurator: SCI0 in simple-SPI mode, GPT0 in input-capture mode with
# tacho_capture_callback registered, ADC0 on AN000, WDT, RTC on the sub-clock.

python3 tools/check_config.py                    # before anything else
sh tools/syntax_check.sh                         # compiles without the FSP
python3 ../../tools/export_model.py \
    --out firmware/model/pdm_detector.h --demod 2500 6000

cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
      -DFSP_ROOT=$HOME/ra8d1_pdm
cmake --build build -j
tools/flash.sh build/pdm_ra8d1.bin               # J-Link OB on the board
```

---

## Mounting

Three axes make orientation matter three times as much as on a single-axis kit.

| Mount | Usable bandwidth | Verdict |
|---|---|---|
| **M8 stud into a spot-faced boss** | >10 kHz | Correct. Use this |
| Adhesive (cyanoacrylate / epoxy) | ~5 kHz | Acceptable on a curved housing |
| Magnet | ~2 kHz | Throws away the demodulation band entirely |

The commissioning tool records which of x/y/z is radial, tangential and axial
(`device.mount_type` and the axis map in `cloud/sql/001_schema.sql`). The cloud
cross-checks it against the gravity vector visible in each channel's DC level:
a unit mounted 90° from its recorded orientation shows ~1 g on the wrong axis.

Get this wrong and the axial channel's misalignment rule fires on radial
unbalance — a plausible, confident, wrong diagnosis, which is the failure mode
this whole repository is organised around avoiding.

The tacho pickup needs a line of sight to the reflective target and a standoff
within the sensor's rated range. A pickup that sees the target twice per
revolution (two pieces of tape, or a reflective keyway) halves every reported
speed and therefore doubles every order — and the jitter gate will not catch it,
because the readings are perfectly consistent. `APP_TACHO_PULSES_PER_REV` is a
commissioning parameter and must be verified against a handheld tachometer.

---

## Validation protocol

Nothing below has been run. `docs/hardware/ra8d1.md` carries the detail.

**Stage 1 — bench, no machine.** Self-test passes on all three axes; a shaker
reads back within 5 % per axis; cross-axis sensitivity below 5 %; the tacho
reads a known-speed motor within 0.1 %; `pdm_board_sleep_ms` within 1 % of a
reference clock; a frame from each channel decodes in the cloud with the right
tag.

**Stage 2 — instrumented rotor rig with a VFD.** The stage that tests what this
kit is for: sweep the drive 1–5 % during a record and confirm the order-domain
peak at a known high order stays put while the frequency-domain peak smears by
the predicted amount. Compare against a calibrated piezo channel (the CN0549
branch exists partly to be that reference).

**Stage 3 — measured datasets.** CWRU and MFPT through the same pipeline with
bearing-wise partitioning. **The gate no model in this repository has passed.**
Note that neither dataset has a tacho channel or a varying speed, so neither
can validate the order-tracking path — that needs a rig or a plant.

**Stage 4 — plant pilot on VFD-driven assets specifically.** The population this
kit is aimed at, and the one where fixed-frequency analysis is progressively
wrong as the duty varies.

---

## Production checklist

- [ ] `tools/check_config.py` and `tools/syntax_check.sh` clean in CI
- [ ] Cross-compile with `arm-none-eabi-gcc` in CI (**the largest open gap**)
- [ ] Verify `-mfpu=auto` reached the vector unit — disassemble and look for MVE
      instructions; no test can do this for you
- [ ] `arm-none-eabi-size` numbers replacing the host estimates
- [ ] Measured timing from a GPIO toggle around each stage, per axis
- [ ] Measured current with a Joulescope; expect 20–50 % worse than the table
- [ ] Tacho verified against a handheld tachometer, including
      `APP_TACHO_PULSES_PER_REV`
- [ ] Cross-axis sensitivity characterised; axis map recorded per device
- [ ] Supercapacitor sized if LiSOCl₂ is used — 62 mA bursts exceed the cell's
      pulse rating
- [ ] `ble_transport.c`, `flash_queue.c`, `ra8_platform.c` written
- [ ] Flash-queue wear: 1 536 frames/day × 5 years ≈ 2.8 M writes, three times a
      single-axis kit's. Needs the wear-levelling spare sized and proven
- [ ] TrustZone split decided: secure boot and the modem keys in the secure
      partition, DSP in non-secure
- [ ] EMC pre-compliance — a 480 MHz core beside a tacho pickup on a VFD-driven
      machine is the realistic worst case, and the pickup is the vulnerable part

---

## Honest status

| | |
|---|---|
| Shared DSP verified against Python | ✅ 1 113 C checks |
| Angular resampling measured | ✅ 14 checks, `tests/test_order.c` |
| Three-axis application path exercised | ✅ 4 end-to-end tests, `virtual_device_mc` |
| Driver compiles under the real warning set | ✅ against a stub FSP |
| Config cross-checked against Python | ✅ 19 property tests |
| Wire format byte-identical, channel tags round-trip | ✅ |
| **Compiled for Cortex-M85** | ❌ no `arm-none-eabi-gcc`, no FSP in this environment |
| **Flashed to a board** | ❌ |
| **Helium actually used** | ❌ unverified; needs a disassembly |
| **Timing measured** | ❌ estimates only |
| **Current measured** | ❌ datasheet arithmetic only |
| **Tacho exercised** | ❌ synthetic marks only; no pickup, no rotor |
| Order-domain features on the wire | ⬜ deliberately not shipped — see [the gap](#the-gap-stated-plainly) |
| `ble_transport.c`, `flash_queue.c`, `ra8_platform.c` | ⬜ declared, not implemented |

Three files are referenced by the build and not written. They are listed here
rather than stubbed so the gap shows up as a link error — `tools/syntax_check.sh`
prints exactly which symbols are missing — instead of hiding behind a function
that returns success.
