# ESP32-S3 + ADXL355 — design note

Companion to [`hardware/esp32s3/README.md`](../../hardware/esp32s3/README.md).

## The sensor choice is the architecture

Most of this branch follows from one fact: **the ADXL355 cannot see bearing
resonances, and no signal processing recovers them.** Its anti-alias filter is at
1 kHz by design, so the 2–20 kHz band where bearing impacts ring the structure is
removed before digitisation. There is no information left to extract.

The branch does two things instead of pretending otherwise:

1. **Configures the envelope stage inside the usable band (500–1000 Hz) and
   labels the result as weak.** `tools/check_config.py` emits a NOTE on every
   run; the frame's `tier` field tells the cloud whether the bearing block was
   computed at all. A negative bearing result from this profile means *not
   measured*.
2. **Ships a second profile for the IIS3DWB**, so the same board can be a
   bearing node when that is the requirement.

The alternative — one profile with a band straddling 1 kHz and
confident-looking bearing numbers — would have been easier to write and would
have produced work orders from filtered-out information. That defect is
undetectable from the output, because every number is in range.

## What the ADXL355 is genuinely better at

It is not a worse sensor, it is a different one: 25 µg/√Hz against 75, 20 bits
against 16, and a 1 kHz band that is flat rather than rolling off.

For the simulated 395 rpm agitator the 1x line is at 6.6 Hz. The measurement
that matters is ISO 20816 velocity over 2–1000 Hz, and the limitation there is
the low-frequency noise floor, not bandwidth. The ADXL355 wins by roughly 10 dB.
That asset exists in `pdmcore.sim` partly to make the case testable.

## Wi-Fi dictates the battery chemistry

240 mA·s per association, 1 400× the sensing and compute combined. The chain is
worth writing out, because it is not obvious that a radio choice determines a
cell chemistry:

```
Wi-Fi association = 240 mA.s, 350 mA peaks
  -> rules out LiSOCl2 (pulse-current limit is tens of mA)
     -> must use Li-ion (2.5 %/month self-discharge = 72 uA)
        -> self-discharge exceeds circuit draw above a 1 h interval
           -> battery life saturates near 2.7 years
              -> honest recommendation: mains or PoE
```

Compare the Nordic branch: LTE-M costs 95 mA·s — a *more* expensive radio — but
its pulse profile permits a LiSOCl₂ D-cell whose 0.08 %/month self-discharge is
4 µA, giving 9.4 years at an hourly cadence. Five times the battery life from
the more expensive radio.

## Deep sleep changes the program structure

`esp_deep_sleep_start()` does not return: the chip resets into `app_main`, so the
"loop" is one iteration per wake and SRAM is gone.

Light sleep would preserve the loop at 240 µA — 5.2 mA·h per cycle at a 6-hour
interval, roughly 40× the entire acquisition budget. Deep sleep is not optional,
so the structure follows from the power budget.

What must survive lives in `RTC_DATA_ATTR` (8 kB of RTC slow memory):

| Variable | Consequence of losing it |
|---|---|
| `rtc_sequence` | **The dangerous one.** Ingest deduplicates on `(device_id, sequence)` so replays are idempotent. A node restarting at zero has every frame after the first wake discarded as a duplicate — it looks healthy and transmits nothing. `pdm_app_restore_sequence` was added to the shared core for this |
| `rtc_epoch_ms` | The RTC drifts minutes per day; without a synced epoch the trend cannot be aligned across the fleet |
| `rtc_threshold` | Reverts to computing tier 2 every cycle — correct but wasteful |
| `rtc_interval_s` | Reverts to the compiled default, undoing a cloud reconfiguration on every wake |

## Dual-core is a correctness requirement

`CONFIG_FREERTOS_UNICORE=n`, and not for throughput.

The ADXL355 FIFO is 96 samples — 24 ms at 4 kHz. The ESP-IDF Wi-Fi task runs at
a higher priority than application tasks and can hold the CPU for tens of
milliseconds during association. On a single core that overruns the FIFO, and
the resulting block is a concatenation of two different times: a spectrum with a
step discontinuity that looks exactly like an impact train.

Three defences, all needed:

1. Acquisition pinned to core 1, network stack to core 0.
2. The application never transmits during an acquisition — `pdm_app.c`
   serialises them, which also keeps radio supply noise out of the analogue
   front end.
3. The overrun flag is checked every chunk and the block is refused, not
   repaired.

## What this kit can do that the ST kit cannot

**Measure its own sample rate.** The ADXL355's ODR comes from an internal
oscillator with no tolerance better than ±5 % over temperature — *worse* than
the IIS3DWB's ±3 %. But this board has a ±10 ppm crystal and `esp_timer`, so the
true elapsed time of an acquisition is measurable to microseconds, and
`samples / elapsed` is the real rate.

That turns the kit's worst metrological spec into its best. It is the cheapest
accuracy win anywhere in this repository and it is **not yet implemented** —
`pdm_board_acquire` reports the nominal rate with a comment saying so. Closing it
is about a dozen lines.

## Open problems

- The IIS3DWB acquisition path returns `PDM_BOARD_ERR_NOT_READY` rather than
  falling through to the ADXL355 register map. Explicit refusal beats plausible
  numbers.
- `mqtt_transport.c` and `nvs_queue.c` are declared and not written. The
  former's batching is what makes the energy budget work, so it is not a thin
  wrapper around `esp_mqtt_client_publish`.
- The S3's ADC needs the eFuse calibration curve for better than ±5 %. Enough
  for a 3.4/3.2 V decision, nowhere near a state-of-charge display — which is
  why the frame carries millivolts and the dashboard shows a trend.
