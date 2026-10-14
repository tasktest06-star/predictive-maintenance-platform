# Design note — Renesas EK-RA8D1 three-axis node

Companion to [`hardware/ra8d1/README.md`](../../hardware/ra8d1/README.md). The
README says how to build and what it costs; this note records *why* the design
is shaped this way, which decisions were wrong first, and what is still open.

---

## 1. The brief this kit answers

The other four MCU branches all answer the same question — "can a battery node
do credible bearing diagnosis?" — with different silicon. This one answers a
different question:

> What can a node do that the 256 kB, 64–160 MHz kits cannot do at all?

That framing matters because the obvious answer — "the same thing, faster" — is
worthless. Energy is work; the work is fixed; a 2× faster core on a 6 %-of-cycle
term is a rounding error. If the only argument for a €100 M85 is speed, the
€90 STM32U5 wins on price and the comparison is over.

So the kit is configured to spend the headroom on capability rather than on
finishing early:

| Capability | Why it needs this part |
|---|---|
| Three live axes | 3× the pipeline (5.9 ms) and a 49 kB retained raw block + a 33 kB float record — 82 kB of buffers alone, which does not fit in 256 kB |
| A hardware tachometer | Not a compute problem, but an *architectural* one: the only board here with a capture input routed to a connector |
| Angular resampling | 0.9 ms and a 9 kB angle-domain buffer; ~11 ms on an M33 at 64 MHz, on top of an 18 ms budget |
| Headroom for more | 16384-point FFT, several bearings per node, cyclostationary analysis — 227 kB of 1 MB used |

---

## 2. Decisions, with the reasoning

### 2.1 Three axes, one burst

The three axes **must** come from a single FIFO burst. Three separate
acquisitions a few hundred milliseconds apart are three different instants of
the machine, and cross-axis relationships — the entire reason to have three
channels — become noise.

This produced the board contract now documented in `cdsp/include/pdm_board.h`:

```
select_channel(0); acquire(...)   <- reads the sensor, caches the block
select_channel(1); acquire(...)   <- RAM copy out of that same block
select_channel(2); acquire(...)   <- likewise
release_block()                   <- next acquire(0) hits the sensor again
```

**Where it lives, and why not in this kit's `main()`.** Everything around the
pipeline — the battery gate, the sticky-flag discipline, the sequence accounting
the cloud deduplicates on, the opportunistic drain — is precisely the part that
must not be reimplemented per kit. So `pdm_app.c` grew a channel loop guarded by
`APP_N_AXES`, and with that macro undefined the loop runs once and the emitted
bytes are identical to before it existed. That last property is what keeps the
1 113 golden checks and the ten single-axis end-to-end tests meaningful, and it
was verified by rebuilding and re-running them, not assumed.

### 2.2 Channel identity on the wire

Three frames per acquisition need to be distinguishable, and the obvious options
were both wrong:

- *Three axes share a sequence number.* The ingest service deduplicates on
  `(device_id, sequence)`, so two of the three would be silently discarded. The
  symptom is a fleet that mysteriously only ever reports its radial axis.
- *Three axes, three consecutive sequence numbers, no channel tag.* Gap
  detection works, dedup works, and the cloud sees three frames with the same
  timestamp and wildly different features — which looks exactly like a failing
  sensor.

So: **sequence per frame, channel in flag bits 8–9.** Those bits were previously
always zero, so single-axis frames are byte-identical and `PAYLOAD_SCHEMA_VERSION`
does not move. A decoder that ignores the bits is no worse off than before; a
decoder that reads them gets the axis.

Bit 10 (`ORDER_TRACKED`) and bit 11 (`SPEED_UNSTEADY`) were added at the same
time, for reasons in §3.

### 2.3 Channel 0 owns the health chain

Only channel 0 updates `last_score` and `last_tier`, and therefore only channel
0 drives the alarm state machine's hysteresis and the EWMA trend.

The alternative — score all three and alarm on the worst — sounds more sensitive
and is worse. The three channels are strongly correlated and differently
scaled, so the "worst" channel is whichever happens to be noisiest, and the
alarm hysteresis ends up chasing it. The dwell and n-of-m logic exists to make
alarms stable; feeding it a max over three correlated channels undoes that.

The axial channel's diagnostic value is realised in the **cloud**, where the
rule base can ask "is 1x high on the axial channel but not the radial?" — which
is the actual misalignment signature — rather than on the node, which has no
view of the other channels' baselines.

### 2.4 Retaining the raw block past `release_block()`

`release_block()` marks the block **spent**, not wiped. The next channel-0
acquire re-reads the sensor, so the node cannot re-analyse stale samples; but
the samples stay readable for the order-domain pass, which deliberately runs
*after* `pdm_app_step()` has returned so it cannot delay the acquisition or hold
the radio window open.

Collapsing "valid" and "spent" into one flag gives a choice between
re-analysing stale samples forever (a perfectly flat trend on a perfectly
healthy-looking dashboard) and having no raw record for the diagnostic. Two
flags cost one byte.

### 2.5 The tacho gate is on speed, not on jitter

`APP_TACHO_MAX_JITTER_PCT` rejects a *speed reading*, not a pulse. The reasoning
is that the harmonic matcher works to ±2 %, so a speed accepted at 10 % error
puts every derived bearing frequency outside its own tolerance window — and the
node then reports "no fault" with complete confidence. A gate feeding a ±2 %
tolerance has to be tighter than it.

Gross jitter (a factor of 4 over the limit) is treated as a missed or doubled
pulse rather than real speed variation, and the reading is dropped entirely in
favour of the configured nominal with `SPEED_ESTIMATED` set. Degrading to the
behaviour of every other kit in the lineup is an acceptable failure mode;
shipping a confident wrong speed is not.

---

## 3. Order tracking, and the claim that was wrong

The design assumed speed smearing was a large effect at bearing orders. It is
not, at this record length, and `tests/test_order.c` was written to measure it
rather than to confirm it.

| | peak vs ideal | −3 dB width | resampling gain |
|---|---:|---:|---:|
| order 6, 4 % drift, frequency domain | 0.84 (−1.50 dB) | 2 bins | — |
| order 6, 4 % drift, angle domain | 1.00 (−0.00 dB) | 1 bin | **1.19×** |
| order 40, 4 % drift, frequency domain | 0.52 (−5.67 dB) | 6 bins | — |
| order 40, 4 % drift, angle domain | 1.00 (−0.01 dB) | 1 bin | **1.92×** |
| order 6, 0 % drift, frequency domain | 0.88 (−1.16 dB) | 2 bins | — |
| order 6, 0 % drift, angle domain | 1.00 (−0.00 dB) | 1 bin | 1.14× |

Smearing goes as (order × drift × record length). At 307 ms the FFT bins are
3.26 Hz and a 6x line at 4 % drift smears over ~7 Hz — barely wider than a Hann
main lobe. **So order tracking is nearly pointless for bearing diagnosis at
this record length**, and the honest place to deploy it is gear mesh, blade
pass and electrical lines, or on a longer record.

The last row is a result nobody was looking for: on a *perfectly steady* shaft
the frequency domain still loses 1.16 dB, because the 174 Hz line falls between
bins. Scallop loss. The angle domain puts the line exactly on an order bin, so
angle-domain amplitudes are accurate in a way plain-FFT amplitudes are not,
independent of any speed variation. For amplitude-based trending — which is
most of ISO 20816 practice — that is arguably worth more than the smearing
recovery.

### 3.1 Why the node does not ship order-domain features

`pdm_pipeline_run` takes its sample rate from `app_config.h` at compile time.
Feeding it a resampled record would label orders as hertz: numbers that look
right and are wrong by whatever the speed happened to be. The options were:

1. **Runtime-rate pipeline.** Correct, and a change to shared code that every
   other kit depends on, with a full golden-vector re-verification. Not
   something to do as a side effect of adding a kit.
2. **A second, order-domain feature spec.** Correct, and it forks the cloud's
   model training, baseline storage and rule base per asset.
3. **Flag it and escalate.** What was done.

So the node flags `SPEED_UNSTEADY`, requests a raw snapshot, and computes the
local order spectrum for the deployment where snapshot bandwidth is not
available. The cloud — no compute limit, and able to extend the feature set per
asset — does the analysis.

It deliberately does **not** set `ORDER_TRACKED`, because the features really
are in hertz. A flag that claims the angle domain while the payload is
frequency-domain would have the cloud compare against the wrong baseline, which
is a worse failure than no flag.

Option 1 is the right long-term answer and is the top item in §6.

### 3.2 Interpolation kernel

Catmull-Rom, 4 points. Linear attenuates ~2.4 dB at a quarter of the output
Nyquist against Catmull-Rom's ~0.3 dB, and an amplitude error that grows with
order is indistinguishable from the smearing the resampler exists to remove.
Windowed sinc needs 16–32 taps for a worthwhile further improvement — 8× the
arithmetic for a correction below the IIS3DWB's own 75 µg/√Hz noise floor at
these orders.

Angle is taken to advance linearly between marks. A quadratic fit through three
marks is marginally better under constant angular acceleration but unstable when
a mark is jittery, and tacho jitter is the common case while constant
acceleration is not.

### 3.3 Refusing a bad mark list

A non-ascending mark list means a missed or doubled pulse. One dropped
revolution shifts every subsequent angle by a full turn, and the resulting
spectrum is not noisy — it is confidently wrong, with lines at the right orders
and the wrong amplitudes. `pdm_order_resample` returns `PDM_ERR_ARG`, and
`pdm_order_speed_drift_pct` returns a negative value rather than a plausible
one. Both are covered by negative controls in the test.

---

## 4. Bugs this branch produced

Recorded because each one shipped a *plausible* result, which is the only kind
worth writing down.

**4.1 Three axes, one frame.** The shared loop powers the sensor down the moment
channel 0's burst is in RAM. Both the host board and the real driver checked
"is the sensor powered?" *before* serving the cached channels, so axes 1 and 2
returned `NOT_READY`, the channel loop broke, and the node shipped one frame per
acquisition — with correct features, correct flags, and a healthy-looking
dashboard. Found by `test_multi_axis_emits_one_tagged_frame_per_axis` on its
first run. The power gate now applies only to a real sensor read.

**4.2 Power-down dropped the retained block.** `pdm_board_accel_power(0)` cleared
`s_raw_valid`, so even with 4.1 fixed the extra axes would have re-read the
sensor — three different instants. Now only power-*up* invalidates.

**4.3 The 1-second "standard acquisition".** `tools/power_budget.py` modelled a
flat 1000 ms acquisition for every kit while the firmware acquires 8192 samples
at 26.667 kHz = 307 ms. Acquisition is the largest term in every battery kit's
budget, so this was a 3.3× error in the dominant term, and it had already been
quoted in four pushed READMEs. The window is now derived from `acq_samples /
fs_hz` and `test_kit.py` asserts it matches `app_config.h`. The corrected table
is in this kit's README; the qualitative conclusions of the other branches
survive, their percentages do not.

**4.4 The figure's measured markers came from the wrong rows.** The regex that
parses `test_order`'s output for the deck figure matched the 0 %-drift control
as well as the 4 % cases, so the order-6 marker read 1.14× instead of 1.19× —
close enough to look right. Now anchored to the `low`/`high` labels.

**4.5 `IDEAL_PEAK` off by the window's coherent gain.** The first version of
the test compared peaks against 0.25 (amplitude) rather than 0.5 (amplitude ×
Hann coherent gain), so every "fraction of ideal" came out at exactly 2.000 —
which is how the error announced itself. Suspiciously round numbers are worth
more than they look.

---

## 5. What a review should attack first

1. **`-mfpu=auto` vs `-mfpu=fpv5-sp-d16`.** The second compiles, links,
   produces bit-identical numbers, passes every test in this repository, and
   runs the FFT about twice as slow because it hides the vector unit. No test
   can catch it. Verify by disassembling and looking for MVE instructions.
2. **Cross-axis sensitivity.** The three-axis story assumes the channels are
   reasonably independent. Real MEMS parts have a few percent cross-axis
   coupling, and the mount adds more. If coupling is high, the axial channel's
   misalignment rule will fire on radial unbalance.
3. **`APP_TACHO_PULSES_PER_REV`.** A pickup that sees two marks per revolution
   halves every speed and doubles every order, perfectly consistently, so the
   jitter gate will not catch it. Commissioning must verify against a handheld
   tachometer.
4. **The 62 mA compute burst against a LiSOCl₂ pulse limit.** The battery
   recommendation and the current profile are in tension; it needs a
   supercapacitor and the sizing is not done.
5. **Flash-queue wear.** 1 536 frames/day is three times a single-axis kit's,
   so ~2.8 M writes over five years against 10 k cycles/page.

---

## 6. Open items, ordered by what they would change

| Item | Why it matters |
|---|---|
| Runtime-rate pipeline | The only way to ship order-domain features; unlocks §3.1 properly, and benefits any kit with a tacho |
| Cross-compile in CI | Everything in §5.1 is invisible until there is a real build |
| `ble_transport.c`, `flash_queue.c`, `ra8_platform.c` | Declared, not written. `tools/syntax_check.sh` prints exactly which symbols are missing |
| Longer record (16384 samples) | 1.63 Hz envelope resolution and twice the order resolution; this part has the RAM for it and the others do not |
| Time-synchronous averaging | Needs the mark-to-sample-0 offset the current code discards; the natural next use of a tacho after order tracking |
| Multi-bearing support | Several bearings per node is a compute and config problem, not a sensing one, and this is the part with the headroom |
| Secure boot via TrustZone | `docs/04_security.md` applies; the split is undecided |

---

## 7. What this kit is for, in one paragraph

A VFD-driven critical asset where a single radial channel is not enough: a
gearbox where the mesh order matters, a fan where blade pass matters, a
compressor whose duty varies through the day. It costs more than the STWIN,
draws more current, needs wiring the STWIN does not, and buys an axial channel,
a measured shaft speed, amplitude-accurate order-domain analysis, and room to
grow the DSP by an order of magnitude. On a non-VFD motor-pump set with one
accessible bearing, the STWIN is the better engineering choice and this part is
over-specified — which is the honest recommendation and the reason both branches
exist.
