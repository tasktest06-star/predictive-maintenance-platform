# ADI CN0549 — the reference instrument

The expensive one, and the only kit here whose numbers are **traceable**: a
calibrated IEPE piezo, a 24-bit converter, and a certificate that says what the
sensor produced on a reference shaker.

Its job is not to be deployed at scale. It is to be the measurement the MEMS
nodes are compared against — so that when a €90 battery node says 2.1 mm/s,
somebody has checked what 2.1 mm/s really was.

**Status: never run on hardware.** No CN0549 in this environment. The IIO
attribute names are from the pyadi-iio AD7768-1 driver and the reference design
docs; treat them as review-quality. What *is* measured is in
[Two experiments](#two-experiments-and-both-answers-were-surprising), and both
answers were not what the design assumed.

---

## See it work now

```bash
# The two experiments that decide whether this kit earns its cost (~90 s)
PYTHONPATH=core:edge python3 hardware/cn0549/tools/experiments.py

# 15 kit tests: the IEPE bias logic, the config, the calibration, the experiments
PYTHONPATH=core:edge python3 -m pytest hardware/cn0549/tests -q

# The shared agent, with no sensor attached
PYTHONPATH=core:edge python3 -m pdmedge --self-test
```

---

## Bill of materials

| Item | Part | Qty | Notes |
|---|---|---:|---|
| Front end | **EVAL-CN0549-ARDZ** | 1 | ~€250. CN0540 + AD7768-1, Arduino shield form |
| — ADC | AD7768-1 | — | 24-bit Σ-Δ, DC–40 kHz, sinc5 / wideband filters |
| — Front end | CN0540 | — | IEPE current source, DC-excluding amplifier, programmable offset |
| Carrier | **Cora Z7-07S** | 1 | ~€130. Zynq-7007S; runs the ADI Kuiper Linux image |
| Sensor | **TE 8911** or equivalent IEPE accelerometer | 1 | ~€300 **with certificate** — the certificate is the point |
| Cable | 2-pin MIL-C-5015 or BNC, screened | 1 | Length affects the bias voltage; see [the bias check](#the-iepe-bias-check) |
| Mount | M8 stud, spot-faced boss | 1 | A piezo's 40 kHz bandwidth is thrown away by a magnet |
| Supply | 5 V 3 A | 1 | Mains. 3.0 W continuous — see [Power](#power) |
| Shaker (stage 1) | Reference shaker, e.g. B&K 4294 | 1 | 10 m/s² at 159.2 Hz. Borrowed or a calibration-lab visit |

Total around **€700** against €90 for the STWIN. That has to be justified, and
the justification turned out not to be the one on the datasheet.

---

## Two experiments, and both answers were surprising

[`tools/experiments.py`](tools/experiments.py) — both results are JSON in
`results/experiments.json`, and the deck's figures are generated from that file,
so a slide cannot disagree with the measurement.

### 1. The 24-bit converter buys nothing. The sensor noise floor is everything.

The first version of this experiment quantised the same simulated record to 16,
20 and 24 bits and compared the BPFO harmonic-energy ratio. **All three agreed
to three decimal places at every severity from 0.002 to 0.6.** That is not a
bug, it is the answer:

| chain | bit depth | sensor noise | converter noise | ratio |
|---|---:|---:|---:|---:|
| IIS3DWB | 16-bit, ±16 g | 75 µg/√Hz | 1.11 µg/√Hz | **67×** |
| ADXL355 | 20-bit, ±8 g | 25 µg/√Hz | 0.03 µg/√Hz | **718×** |
| CN0549 | 24-bit, ±41 g | 7 µg/√Hz | 0.01 µg/√Hz | **628×** |

Every chain in this lineup is **sensor-limited by two to three orders of
magnitude**. Adding converter bits to a sensor-limited measurement buys dynamic
range, not resolution. A 16-bit MEMS part's ADC is *not* what limits it.

So the experiment was rewritten to sweep what actually differs — each part's
noise density — and the answer there is also not flattering:

```
  severity        IIS3DWB 16-bit      ADXL355 20-bit       CN0549 24-bit
     0.000                 1.571               1.747               1.659
     0.002                 1.610               1.655               1.670
     0.005                 1.708               1.808               1.945
     0.010                 2.000               2.728               2.930
     0.020                 3.661               4.886               5.198
     0.050                 7.419               8.388               8.295
     0.100                 9.224               9.492               9.495
     0.300                 9.668               9.711               9.705
  first severity detected at 3 sigma above healthy, with the margin
  by which it crossed (near 1.0 = one grid step from flipping):
    IIS3DWB 16-bit         0.02   margin 1.71
    ADXL355 20-bit         0.01   margin 1.24
    CN0549 24-bit          0.01   margin 1.39
```

**A 10× quieter sensor buys about one grid step of detection, not ten.** The two
quieter chains cross the 3 σ line at severity 0.01 where the 16-bit MEMS chain
needs 0.02 — one step on a factor-of-two grid — and both cross only marginally
(margins 1.24 and 1.39, against the 16-bit chain's 1.71 at its own step). So the
advantage is real, modest, and at the edge of what this sweep can resolve.

The piezo also gives a stronger signature once there *is* one, which helps
*severity estimation* more than detection. The machine's own broadband noise,
not the sensor's, sets the floor for both.

**A number that flipped between runs, and the fix.** An earlier run of this
sweep put all three chains at 0.01; the committed JSON put the 16-bit chain at
0.02. Both were true — the 3 σ limit depends on the healthy-case standard
deviation, which moves with the seed count and the grid. A single number that
flips is worse than a number with its margin attached, so the experiment now
reports **how far past the limit each chain crossed**, and this README's block
is pasted from the committed JSON rather than from a terminal.

Caveat that matters: this is a simulator whose noise model was written in this
repository. It can support an argument about physics — noise density dominates
bit depth, and the machine dominates both. It cannot give the true detection
threshold of a real piezo on a real bearing. That needs stage 2 below.

**So why buy this kit?** Not resolution. Three things the table above does not
show:

- **Bandwidth.** DC–40 kHz against the IIS3DWB's 6.3 kHz. Lubrication distress
  and early crack propagation emit in the 20–40 kHz region, which no MEMS part
  in this lineup can see at all. This is the real argument, and it is
  categorical rather than incremental.
- **Traceability.** A certificate, a serial number and a date. When a fleet of
  MEMS nodes disagrees with each other, this is what settles it.
- **DC accuracy.** The CN0540 is DC-coupled through the converter, so tilt,
  thrust and very low-frequency content are real measurements rather than
  artefacts of an AC-coupling corner.

### 2. A disconnected sensor does not look healthy. It looks faulty — and gets diagnosed.

The prediction was that an AC-coupled input with no sensor — converter noise,
the smoothest signal the system will ever see — would score as *less* anomalous
than a healthy machine, making the worst possible condition invisible.

**Wrong.** The measurement:

| | median score | alarms |
|---|---:|---:|
| alarm threshold (1 % FPR) | 9.7 | — |
| healthy machine | 5.8 | 0/20 |
| **disconnected sensor** | **52.9** | **20/20** |
| outer-race fault, severity 0.35 | 84.8 | 20/20 |

A Mahalanobis detector flags a dead channel loudly, because converter noise is
not near the healthy centre — it is far outside it, in a different direction.
Featureless is as anomalous as faulty.

The real failure is one level up. The detector alarms, and the **rule base then
diagnoses the noise**:

```
    disconnected -> mechanical_looseness   confidence 0.97  severity developing
    real fault   -> bearing_outer_race     confidence 1.00  severity severe
```

**A cut cable produces a 97 %-confidence mechanical-looseness work order.** A
planner sends a fitter to a machine that is fine, while the actual fault is a
connector. Nothing in the data distinguishes the two cases — which is exactly
why the check has to be out of band.

---

## The IEPE bias check

An IEPE sensor sits on a 2–20 mA constant-current supply and idles at a known
bias voltage. Read the bias and you know the state of the cable:

| Bias | State | Meaning |
|---|---|---|
| ≥ 85 % of supply (≥20.4 V on 24 V) | `open` | Cut cable, unmated connector, or no sensor |
| 6–16 V | `ok` | Live sensor |
| ≤ 1 V | `short` | Shorted line, or the current source is off |
| anything else | `unknown` | Wrong sensor type, very long cable, failing source |

`classify_bias` is a pure function so it is tested without hardware, and
`CN0549Source.open()` **refuses to start** when the check fails. Refusing is
deliberate: a node that starts happily on a cut cable produces months of data
showing a beautifully smooth machine, and nobody questions good news.

`unknown` is not treated as a pass. The band between a short and the normal
window is where a wrong sensor type and a failing current source live, and
"probably fine" is how a 97 %-confidence wrong work order starts.

---

## The anti-alias filter is a choice

The AD7768-1 offers two decimation paths, and neither is a safe default — so
`filter_type` is a required argument and the agent prints it at start.

| Filter | Transition | Group delay | Use for |
|---|---|---|---|
| `sinc5` | sharp | long (tens of samples) | magnitude spectra |
| `wideband` | flat to 0.433 × Nyquist | short | **impacts, envelopes, bearings** |

The shipped config uses `wideband`, because bearing work is envelope work and
the sinc5 path's long impulse response smears impact timing. If a slide ever
shows an envelope computed through `sinc5`, that is the reason the impacts look
wider than they are.

### DC removal

This chain is DC-coupled through the converter, so the bias appears as a large
DC offset. Leaving it in puts a huge value in bin 0 that leaks into the low
orders through any window's skirts.

The offset is removed at the source — **after** the clipping check, not before.
The converter clips on the total signal including the offset, and subtracting
first would hide exactly the condition being looked for. The amount removed is
reported (`last_dc_v`) rather than discarded, because a drifting DC offset is
itself a diagnostic: it tracks the current source's health and the sensor's
temperature.

---

## Power

```
$ python3 ../../tools/power_budget.py --kit cn0549
  MAINS POWERED: 3.0 W continuous
```

No battery analysis, and **not because nobody did one**: an IEPE sensor needs a
continuous 2–20 mA at 24 V — 48–480 mW *just to bias the sensor*, before the
converter, the FPGA or the Linux userspace. At 4 mA and 24 V that is 96 mW
continuous, which on a 2600 mAh 18650 is about 11 days.

IEPE excitation makes low power impossible by design. That is not a limitation
of this implementation; it is the trade a charge-mode piezo chain makes, and it
is why battery condition monitoring is a MEMS business.

---

## Bring-up

```bash
# 1. ADI Kuiper Linux on the Cora Z7 (SD image, ~4 GB)
#    https://wiki.analog.com/resources/tools-software/linux-software/kuiper-linux

# 2. On the target
sudo apt install python3-numpy python3-scipy
pip3 install pyadi-iio paho-mqtt
sudo mkdir -p /opt/pdm /etc/pdm /var/lib/pdm
# copy core/ and edge/ to /opt/pdm, config to /etc/pdm

# 3. Prove the stack with no sensor attached
PYTHONPATH=/opt/pdm/core:/opt/pdm/edge python3 -m pdmedge --self-test

# 4. Then the sensor chain, which is the first thing that can be wrong
PYTHONPATH=/opt/pdm/core:/opt/pdm/edge \
    python3 hardware/cn0549/edge/cn0549_source.py --uri ip:localhost
#   CN0549 on ip:localhost
#     rate         32000.0 Hz (read back from the device)
#     filter       wideband -- flat to 0.433 x Nyquist, short delay
#     IEPE bias    12.1 V

# 5. Install the service
sudo cp edge/systemd/pdm-edge.service /etc/systemd/system/
sudo systemctl enable --now pdm-edge
```

Step 3 before step 4 is the point: it draws a line so that a failure afterwards
is unambiguously the sensor, its cable or its calibration.

---

## Calibration

[`config/calibration.cn0549.json`](config/calibration.cn0549.json) ships with
**placeholder values from the datasheet, not a certificate**, and says so in the
file. The nominal sensitivity of a 100 mV/g sensor is typically within 5 % of
its certified value — a 5 % amplitude error on every reading, which is enough to
move an ISO 20816 zone boundary.

The agent prints the whole chain at start:

```
calibration: 1 channel(s) from /etc/pdm/calibration.json
  voltage0    100.00 mV/g  gain  1.0  full scale    41.0 g  resolution    4.88 ug  cert 2026-06-01
             over 16.0 kHz: sensor    885.4 ug RMS, ADC   1.41 ug RMS -- sensor-limited by 56 dB
```

The last line is the kit's own argument against itself, printed on every
start-up. It is also a test (`test_noise_budget_says_sensor_limited`): if it
ever fails, the chain became converter-limited and the README needs rewriting.

The frequency response in the file matters more here than on a MEMS kit. A
piezo rolls off below ~2 Hz and rises towards its mounted resonance above
~10 kHz, and the correction is applied in the frequency domain and **holds its
end values rather than extrapolating** — a guessed slope near resonance would be
confidently wrong exactly where the correction is largest.

---

## Validation protocol

**Stage 1 — reference shaker.** The stage that only this kit can do, and the one
that makes it worth owning. 10 m/s² at 159.2 Hz; read back within 2 % of the
certificate. Then sweep 10 Hz–10 kHz and measure the response curve, replacing
the datasheet placeholders with measured values. Without this the kit is an
expensive MEMS node.

**Stage 2 — the cross-validation this kit exists for.** Mount a CN0549 channel
and a MEMS node on the same bearing housing, run a seeded-defect rotor rig, and
compare: velocity RMS, the kurtogram's band choice, BPFO/BPFI harmonic ratios,
and the detection severity. **This is the measurement that would replace
experiment 1's simulator result with a real one**, and it is the single highest-
value experiment left anywhere in this repository.

**Stage 3 — the AE band.** 20–40 kHz on a bearing under deliberate lubrication
starvation. No MEMS part here can see this band at all, so there is nothing to
compare against — the question is whether it gives earlier warning than the
2.5–6 kHz envelope does, and by how long.

**Stage 4 — the disconnected-sensor test, on hardware.** Unplug the cable
mid-run and confirm: the bias check catches it, the agent refuses rather than
acquiring, and — if the check is disabled — the resulting frames really are
diagnosed as looseness. Experiment 2 predicts 97 % confidence. Verifying that on
hardware is a one-afternoon experiment with a strong result either way.

---

## Production checklist

- [ ] Real certificate values replacing the datasheet placeholders
- [ ] Measured frequency response replacing the datasheet curve
- [ ] `filter_type` decision recorded per asset, not left at the default
- [ ] IEPE bias logged as a trend, not just checked — a slowly falling bias is a
      failing current source or a wetting cable
- [ ] DC offset trended for the same reason
- [ ] Kuiper Linux image pinned and reproducible; `apt` sources frozen
- [ ] Device X.509 provisioned; **no shared fleet secret** (`docs/04_security.md`)
- [ ] Queue partition sized: 200 000 frames × 300 B = 60 MB, plus WAL
- [ ] SD-card wear: the queue fsyncs once per frame. At 15-minute cadence that
      is 96 writes/day, trivial; at 1 Hz it is not, and an eMMC carrier would be
      required
- [ ] Mains failure behaviour: this kit has no UPS, so a power cut loses the
      cycle in progress. The queue survives (`synchronous=FULL`), the
      acquisition does not
- [ ] Thermal: 3.0 W in a sealed IP65 enclosure at 40 °C ambient needs a
      calculation nobody has done

---

## Honest status

| | |
|---|---|
| IEPE bias classification | ✅ 5 tests, incl. open / short / out-of-band |
| Shipped config and calibration validated | ✅ 6 tests |
| Both experiments reproducible | ✅ JSON committed, figures generated from it |
| Shared agent, end to end | ✅ 36 tests in `edge/tests` |
| Noise-budget claim asserted as a test | ✅ |
| **Run against a CN0549** | ❌ no hardware in this environment |
| **IIO attribute names verified** | ❌ from the driver docs, not from a device |
| **Reference-shaker calibration** | ❌ the stage that justifies the kit |
| **Cross-validation against a MEMS node** | ❌ **the highest-value missing experiment** |
| **The 20–40 kHz AE band** | ❌ never measured; the kit's best categorical argument is untested |
| Calibration certificate | ⬜ placeholder datasheet values, flagged in the file |

The two experiments are simulator results. They are enough to show that the
*bit-depth* argument for this kit is wrong and that the *bandwidth and
traceability* arguments are what remain — but the numbers that would decide a
purchase need stage 2.
