# `pdmedge` — the Linux-class node and gateway agent

Three of the eight hardware kits run Linux rather than bare metal: the ADI
CN0549 (Cora Z7), the Raspberry Pi 5 + Hailo-8L, and the NVIDIA Jetson Orin
Nano. They share this agent the way the five MCU kits share [`cdsp/`](../cdsp).

```bash
# Prove the whole stack with no sensor attached. Run this first on a new board.
PYTHONPATH=core:edge python3 -m pdmedge --self-test

# Run against the simulator, emitting real frames on stdout
PYTHONPATH=core:edge python3 -m pdmedge --config edge/examples/agent.node.json \
    --cycles 3 --stdout --no-sleep

# The tests
PYTHONPATH=core:edge python3 -m pytest edge/tests -q      # 34 tests
```

`--self-test` exists because it draws a line: it proves the Python stack, the
DSP, the wire format and the queue before any sensor is attached. A failure
after it passes is the sensor, its wiring or its calibration — not this.

---

## The loop is the firmware's loop

Deliberately the same shape as [`cdsp/src/pdm_app.c`](../cdsp/src/pdm_app.c):

```
health gate -> acquire -> features -> encode -> publish or queue
            -> drain the backlog while the link is up -> sleep
```

Same order, same reasons:

| Step | MCU | Linux |
|---|---|---|
| Gate first | battery: a node at 3.25 V can beacon but not acquire | **disk**: a full queue partition means a failed publish loses its frame, and finding out after acquiring wastes the acquisition |
| Drain after a publish | the link is demonstrably up — the cheapest moment to flush | identical |
| Sequence advances on frame build | a failed acquisition must not look like lost data | identical |
| Diagnostic frame on overrun | a missing hour is indistinguishable from a dead sensor | identical |

A Linux node's frames are **byte-identical in format** to an MCU node's: same
300 bytes, same schema version, same feature-spec version, same CRC. The cloud
cannot tell which tier a device belongs to, which is the point — one ingest
path, one model format, one set of dashboards.

### What differs, and why

- **Per channel, not per device.** A gateway aggregates several sensors. The
  channel index rides in frame flag bits 8–9, exactly as it does for the
  RA8D1's three axes, so the cloud needs no new concept.
- **Features come from `pdmcore`** — the same code the cloud trains on. The MCU
  has a C twin because it must; this tier has no reason to maintain two.
- **No tiering.** The MCU computes a cheap tier and escalates to save energy. A
  mains-powered node has nothing to save, so it always computes the full
  vector. That also means its frames are never partially populated, and the
  cloud can distinguish a Linux node's "no bearing fault" from an MCU node's
  "bearing block not computed".

---

## Calibration: the thing that makes this tier different

A MEMS node reports g because its datasheet says so. A CN0549 channel reports g
because a calibration certificate says what its piezo produced on a reference
shaker. [`calibration.py`](pdmedge/calibration.py) implements the chain:

```
ADC code -> volts at the ADC -> volts at the sensor -> g -> g corrected
            (full scale,       (amplifier gain)      (mV/g)  (frequency
             bits)                                            response)
```

**It refuses to default a sensitivity.** An uncalibrated channel raises
`CalibrationError` and the agent skips it. The reasoning: shipping volts
labelled as g is not a degraded measurement, it is a *different quantity*, and
scoring it against a baseline in g produces an anomaly score that is confident
and meaningless.

The certificate's serial and date travel with the configuration, and the agent
prints the whole table at start:

```
calibration: 1 channel(s) from /etc/pdm/calibration.json
  voltage0    100.30 mV/g  gain  1.0  full scale    40.8 g  resolution    4.87 ug  cert 2026-02-10
             over 16.0 kHz: sensor    885.4 ug RMS, ADC   1.41 ug RMS -- sensor-limited by 56 dB
```

Printed, not logged at debug level, because a wrong sensitivity is the most
consequential configuration error on this tier and the only defence is that
someone reads it once. A 100 mV/g sensor entered as 10 mV/g is obvious — it
crosses every ISO zone at once. Entered as 102 mV/g it is a 2 % error nobody
will ever find.

The last line says whether the measurement is **sensor- or converter-limited**,
and it is compared over a *bandwidth* because the two numbers are not the same
quantity: the sensor's noise is a density (g/√Hz) and the converter's is a
quantisation step (g). Over bandwidth B, sensor RMS = density·√B and ADC RMS =
LSB/√12.

A 24-bit converter on a 7 µg/√Hz piezo over 16 kHz is **sensor-limited by
56 dB** — the expensive converter is buying dynamic range, not resolution,
which is worth knowing before paying for a better one.

The first version of that line printed `noise_density_g > resolution_g`, which
is dimensionally meaningless and happened to give the right verdict for the one
channel it was tried on. `noise_budget(bandwidth_hz)` replaced it.

### Frequency response

Optional, as (hz, relative_gain) pairs from the certificate. Applied in the
**frequency domain**, not as a time-domain filter: the correction is a smooth
gain curve, and a minimum-phase filter implementing it would add group delay
that matters for envelope and impact timing.

Outside the measured range it **holds the end value rather than
extrapolating**. A piezo's response rises towards its mounted resonance, and an
extrapolation that guessed the slope would be confidently wrong in exactly the
region where the correction matters.

---

## The queue, and an eviction policy worth arguing about

An MCU queues to internal flash with no filesystem in the way. A Linux gateway
holding a site's data in a page cache loses it on a power cut — and a gateway is
where an outage is most expensive, because it may be holding hundreds of nodes'
data.

So: SQLite, WAL, `synchronous=FULL`. One fsync per frame, ~1 ms on an SD card
for a 300-byte write. At one frame per node per hour that cost is irrelevant,
and it is the difference between a queue and a cache.

**A bounded queue has to drop something, and both obvious policies are wrong:**

| Policy | What it loses | Why that is bad |
|---|---|---|
| Refuse new frames | the hours before a failure | the only data anybody wants afterwards |
| Drop oldest (ring) | the baseline | the health index's operating-point baseline needs weeks; a trend missing its first half cannot say what normal was |

Both ends matter and the middle does not, so this queue **decimates the
middle**: at capacity it deletes every other frame from the central two-thirds,
keeping the oldest and newest intact. Time coverage degrades in *resolution*
rather than in *extent* — an hourly trend becomes two-hourly, then four-hourly,
while the alarm-relevant recent frames and the baseline-relevant old ones both
survive.

It is a real trade, not a free win: a decimated region cannot be un-decimated,
and a fault whose signature appears in one frame may be the frame that was
dropped. For 300-byte feature frames at hourly cadence the queue holds months
before it ever decimates, which is what makes it acceptable. Raw snapshots are
51 kB and their value is almost entirely in being recent, so those are dropped
oldest-first and counted separately.

```python
>>> q.stats()
QueueStats(depth=32, bytes_used=9600, oldest_ms=..., newest_ms=...,
           pushed=80, popped=0, evicted=48)
>>> q.stats().span_hours
79.0            # 32 frames still covering 79 hours
```

### Two-phase pop

`pop()` returns a frame without deleting it; `ack(id)` deletes it. A frame must
not leave the queue because a publish was *attempted* — the firmware pushes
frames back when the link drops mid-drain, and this is the Python equivalent.

---

## Sinks

`MqttSink` publishes at **QoS 1 and waits for the PUBACK** before returning.
Fire-and-forget would make the store-and-forward logic decorative: the agent
would dequeue frames that never arrived, invisibly at both ends.

TLS is on by default and `ca_cert` is required when it is. A gateway shipping a
plant's condition data over an unauthenticated plaintext link is a finding, not
a configuration choice; `tls=False` is available for a lab bring-up and says so.

`FailingSink` ships in the library rather than in a test file, because every
tier needs it — the firmware has `--tx-fail-every` for the same reason. A
store-and-forward path never exercised against a failing link is a feature
nobody has tested.

---

## Sources

| Source | Status |
|---|---|
| `SimSource` | ✅ exercised — drives all 34 tests |
| `FileSource` | ✅ exercised — replays a capture, so a field bug is reproducible |
| `IIOSource` | ⬜ **never run against hardware.** No IIO device in this environment |

`IIOSource` **reads the sample rate back** after setting it, and that is not
defensive programming. An IIO device's rate is a divider off a crystal: ask an
AD7768-1 for 32 kHz and you may get 32000.0 or 31250.0 depending on MCLK and
decimation. A 2.3 % error moves a 106 Hz BPFO line by 2.4 Hz, which is more than
the harmonic matcher's entire tolerance. The value read back goes into the
frame; when a source cannot measure its rate, the frame carries
`RATE_NOMINAL` so the cloud knows to widen its tolerances.

`pyadi-iio` is imported inside `open()`, so this module is importable and
testable on a machine with no IIO stack — every CI runner, and this one.

---

## Deployment

[`systemd/pdm-edge.service`](systemd/pdm-edge.service) is a hardened unit:
`ProtectSystem=strict`, `MemoryDenyWriteExecute`, a `@system-service` syscall
filter, `ReadWritePaths=/var/lib/pdm` and nothing else writable.

It runs as user `pdm` with `SupplementaryGroups=iio`, which is what grants
access to `/dev/iio:device*`. Running the agent as root "because it needs the
ADC" is the most common shortcut on this tier and the one that turns a sensor
bug into a root bug.

`StartLimitIntervalSec=0` is deliberate: a node that cannot reach its broker
must keep acquiring and queueing rather than hit a restart limit and stop. The
queue is the whole point.

---

## Honest status

| | |
|---|---|
| Agent loop, end to end against the simulator | ✅ 34 tests |
| Frames decode in the cloud, CRC-clean | ✅ |
| Store-and-forward with CRC recompute | ✅ incl. a bounded-drain test |
| Multi-channel tagging | ✅ 3 channels, distinct sequences |
| Durable queue across process restart | ✅ |
| Eviction policy bounded and order-preserving | ✅ 500-push stress |
| Calibration arithmetic | ✅ checked against hand arithmetic |
| **Run against a real IIO device** | ❌ |
| **MQTT against a real broker** | ❌ no broker in this environment |
| **systemd unit exercised** | ❌ never installed |
| **Thermal/throttle behaviour on real silicon** | ❌ the parser is tested; the hardware is not |

`MqttSink.publish` and `IIOSource.read_block` are the two functions that have
never executed. Both are marked `# pragma: no cover` with the reason, rather
than being given fake tests that would exercise a mock and prove nothing.
