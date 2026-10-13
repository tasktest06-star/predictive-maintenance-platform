# STEVAL-STWINBX1 — design note

Companion to [`hardware/stwinbx1/README.md`](../../hardware/stwinbx1/README.md),
which covers BOM, build, energy and the production checklist. This note records
the decisions and the open metrological problems.

## Why the IIS3DWB and not the ISM330BX that is also on the board

Both are on the STWIN.box. The ISM330BX is a 6-axis IMU with an in-sensor
programmable core; the IIS3DWB is a single-purpose wideband accelerometer with
one output rate and no features. The wideband part wins for this application for
one reason: **bandwidth**.

Bearing impacts excite structural resonances between roughly 2 and 20 kHz. The
diagnostic information is not at BPFO — it is in the *modulation* of those
resonances at BPFO. A sensor that rolls off at 900 Hz (the ISM330BX in
high-performance mode) sees the modulation envelope only if the resonance
happens to be low, which on a small machine with a stiff housing it is not.

| | IIS3DWB | ISM330BX accel |
|---|---|---|
| Flat bandwidth | 6.3 kHz | ~0.9 kHz |
| Output rate | 26.667 kHz (fixed) | up to 7.68 kHz |
| Noise density | 75 µg/√Hz | ~70 µg/√Hz |
| Operating current | 1.1 mA | 0.55 mA |
| In-sensor processing | none | MLC + ISPU |

The ISM330BX's 0.55 mA and its in-sensor core are genuinely attractive — see
[Future work](#future-work) — but not at the cost of the demodulation band.

## The fixed output rate is a feature

The IIS3DWB has exactly one ODR. There is no anti-alias filter to configure, no
bandwidth/ODR ratio to get wrong, and no mode where the part silently delivers
less bandwidth than expected. On parts with configurable rates, a
mis-ordered register write that leaves the ODR at a default is a common and
nearly undetectable bug: the data looks fine, just smoother.

The cost is that 26.667 kHz is the only choice. Lower rates must come from
decimation in software, which this firmware deliberately does not do — the whole
reason for the part is the bandwidth.

## The metrological problem: ±3 % sample rate

**This is the kit's main unresolved limitation.**

The IIS3DWB's output rate comes from an internal oscillator specified at ±3 %.
There is no way to measure the true rate on this board: the FIFO carries no
timestamps at 26.667 kHz, and there is no external clock path into the sensor.

Consequence: at the 3 % extreme, a 106.3 Hz BPFO line is computed as if it were
at 103.1 or 109.5 Hz. The harmonic matcher's ±2 % tolerance absorbs that — just.
By the fifth harmonic the error is 16 Hz against a ±10.6 Hz tolerance, so the
top harmonics fall outside their windows and `n_matched` under-reports.

Three ways to fix it, in increasing order of effort:

1. **Reference-tone calibration at commissioning.** The IIS3DWB's self-test
   actuator produces a known mechanical deflection. Driving it at a known rate
   from an MCU timer and measuring where the tone lands in the FFT gives the
   true sample rate to the FFT's resolution (6.5 Hz out of 26 667, i.e. 0.02 %).
   One extra second at commissioning, stored as a per-device calibration
   constant. **This is the planned fix.**
2. **Widen the matcher tolerance to ±4 %.** Cheap and wrong: it doubles the
   chance of matching a neighbouring family, which is the failure mode that
   turns an inner-race diagnosis into an outer-race one.
3. **Derive the rate from a known line in the signal.** The 2× line-frequency
   electrical component (100 or 120 Hz) is present on every induction motor and
   is as accurate as the grid. Tracking it gives a sample-rate estimate for
   free, at the cost of assuming the asset is mains-driven — false for anything
   on a VFD, which is an increasing fraction of the installed base.

Until this is closed, `pdm_board_acquire` reports the nominal rate and the
limitation is recorded here rather than hidden.

## Axis selection and the gravity cross-check

The firmware reads the Z axis. Which physical direction that is depends on how
the enclosure was bolted on, and the **radial** axis is the one that sees
bearing and unbalance energy.

Getting this wrong is not catastrophic — bearing energy appears on all axes
through cross-coupling — but it costs 6 to 12 dB of signal, which is the
difference between detecting an incipient spall and not.

The cloud cross-checks it: `time.mean` carries the DC level, which is the
projection of gravity onto the measured axis. A sensor recorded as
radially-mounted that reads ~1 g of DC is actually mounted axially (or
vertically), and the commissioning record is wrong. This check costs nothing and
catches a mistake that is otherwise invisible until someone climbs back up to
look at the sensor.

## Why internal flash for the queue, not the microSD slot

The STWIN.box has a microSD slot. The store-and-forward queue does not use it.

- An SD card draws 20–100 mA while mounted, against a few µA for the flash
  controller. At 512 queued frames that is the difference between a rounding
  error and a measurable dent in the battery.
- SD write latency is unbounded in the general case (wear levelling, internal
  garbage collection), and the acquisition loop has a watchdog.
- SD cards fail silently and in ways that return success.

The 2 MB internal flash holds 512 × 300 B = 150 kB of queue in the top 256 kB.
The open problem is **endurance**: 512 frames/day over 5 years is ~930 k writes
against the U5's 10 k cycles/page specification, so the ring buffer needs
wear-levelling across enough pages that no page sees more than ~10 k erases.
That sizing is on the production checklist and is not yet done.

The SD slot stays useful for raw-waveform capture during commissioning, when the
node is on a bench with a cable and the energy cost is irrelevant.

## Sleep mode: Stop 2, not Standby

Standby is ~0.3 µA against Stop 2's ~2.5 µA, which looks like an easy win over
an hour of sleep — 2.2 µA × 3600 s = 7.9 mA·s saved.

It is not a win, because Standby loses SRAM. Waking from it means re-running
`pdm_pipeline_init`: window generation (4096 cosines), the FFT twiddle table
(3072 sin/cos), and the Butterworth band-pass design (~40 transcendentals). On
an M33 without a hardware transcendental unit that is roughly 40 ms at 19 mA =
0.76 mA·s, against 7.9 mA·s saved. Standby wins.

Except the 36 kB of tables could live in flash as `const`, making the wake-up
free — and then Standby does win. The reason it is still Stop 2 is the IIS3DWB
FIFO staging buffer in SRAM2: it must survive the sleep, because the FIFO is
re-armed before the sleep and read after it. Moving the tables to flash and the
staging buffer's lifetime entirely inside the awake window would make Standby
viable and is worth roughly 2.2 µA, i.e. 14 % of the current average draw.

Recorded as a real, quantified optimisation rather than done, because it is only
worth the risk once the kit has run on hardware.

## Future work

**The IMP23ABSU microphone (80 kHz) is the most valuable unused part on this
board.** Acoustic sensing is complementary rather than redundant: it needs no
mechanical coupling, so it is unaffected by the mounting problem that dominates
accelerometer practice, and 80 kHz reaches into the low-ultrasonic band where
lubrication distress appears earliest. The DCASE lineage in
`research/code_backed_papers_2025_2026.md` is built on exactly this signal, with
a maintained MIT-licensed toolkit. The analogue output goes to the U5's ADC, so
this is firmware work.

**ISM330BX MLC/ISPU as a wake-up filter.** 85 % of this kit's energy is the
307 ms acquisition window. A decision tree running inside the ISM330BX on
low-rate data, waking the MCU only when the vibration signature changes, attacks
the dominant term directly. At long intervals the saving could be an order of
magnitude. The cost is a second model to train, validate and version — and a
new failure mode where the wake-up filter's false-negative rate silently becomes
the system's detection limit. Worth doing; worth doing carefully.

**Multi-axis.** The firmware uses one axis. Three axes triple the acquisition
energy, so the case has to be made per asset: misalignment and looseness are
much clearer with an axial channel, bearing faults rarely need it.
