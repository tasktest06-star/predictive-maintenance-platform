# Raspberry Pi 5 + AI HAT (Hailo-8L) — the gateway tier

Many nodes in, one uplink out, autonomous when the link drops.

**The headline, measured not estimated:** scoring a feature frame takes **17 µs**
on one aarch64 core, and one core keeps up with **781 331 hourly nodes**. A large
site has a thousand. So compute is not what a gateway is for, and the 13-TOPS
AI HAT has nothing to do on this workload — a conclusion the benchmark reached
rather than the datasheet.

**Status: never run on a Pi 5.** The gateway code and its 18 tests run here; the
benchmark was measured on this aarch64 host, which is explicitly *not* a Pi 5
and says so in its own output. The Hailo path raises on construction rather than
returning plausible numbers.

---

## See it work now

```bash
# The measured benchmark (the host is named in its own output)
PYTHONPATH=core:edge python3 hardware/rpi5_hailo/tools/benchmark.py

# UPS sizing, which turned out not to be what the plan assumed
PYTHONPATH=core:edge python3 hardware/rpi5_hailo/tools/holdover.py

# 18 gateway tests: fan-in, gap detection, local alarms, silent-node detection
PYTHONPATH=core:edge python3 -m pytest hardware/rpi5_hailo/tests -q

# The shared agent, with no sensor attached
PYTHONPATH=core:edge python3 -m pdmedge --self-test
```

---

## Bill of materials

| Item | Part | Qty | Notes |
|---|---|---:|---|
| Compute | **Raspberry Pi 5, 8 GB** | 1 | ~€85. 4× Cortex-A76 @ 2.4 GHz |
| Accelerator | **AI HAT+ (Hailo-8L, 13 TOPS)** | 1 | ~€70. **Not needed for this workload** — see [the AI HAT](#the-ai-hat-and-why-it-is-idle) |
| Storage | **NVMe SSD + M.2 HAT** | 1 | ~€40. **Not an SD card** — see [storage](#storage-is-the-real-data-integrity-decision) |
| Radio | BLE (built-in) + sub-GHz HAT | 1 | Terminates the node population |
| Holdover | 10 F supercapacitor module | 1 | 60 s at 7.5 W = 0.12 Wh — see [holdover](#holdover) |
| Enclosure | IP54 vented, DIN rail | 1 | 7.5 W sustained needs airflow, not a sealed box |
| Supply | 5 V 5 A USB-C PD | 1 | 12 W peak at boot sizes the supply, not the UPS |

Around **€200** and it carries a site. The cheapest part of a deployment per
node, which is the point of a gateway tier.

---

## What a gateway is actually for

Not compute. The benchmark settles that. Four things:

1. **Protocol translation.** BLE nodes cannot reach a cloud broker. Something
   must terminate BLE/sub-GHz and speak MQTT/TLS upstream, and that something is
   the only part of the chain that is mains-powered anyway.
2. **Store-and-forward for a whole site.** A node's flash queue holds days; a
   gateway's SSD holds months. When the site uplink goes, this is where the
   plant's data lives.
3. **Local autonomy.** An alarm that needs a round trip to a cloud is
   unavailable during the outage that most often *accompanies* a real failure.
   The gateway scores locally and alarms with no uplink at all —
   `test_local_alarm_does_not_need_the_uplink`.
4. **Fan-in accounting: which nodes have gone quiet.** The most commonly missed
   one, so it is implemented here rather than deferred to the cloud.

### Silent-node detection is the feature that cannot live anywhere else

A node that stops reporting is the single most common real event in a deployed
fleet, and from downstream it is **indistinguishable from a healthy quiet
machine**. Only the thing the nodes report *to* knows the difference between
"nothing to say" and "gone".

`Gateway.silent_nodes()` learns each node's interval from arrivals — rather than
reading it from config, because fleet intervals are changed from the cloud and a
gateway's copy of that goes stale — and flags a node that has missed **three**
of its own intervals. Three, not one: a single missed cycle is normal on a
congested BLE link, and an alarm that cries wolf on every missed frame trains an
operator to ignore it.

The interval is learned with a slow EWMA (0.8/0.2) so that one early frame
cannot reset the silence threshold and mask a genuine outage —
`test_interval_learning_is_slow_to_believe_a_change`.

Crucially, silence is measured from **arrival** time, not from the frame's own
timestamp. A node with a wrong clock would otherwise look permanently fresh.

---

## Measured throughput

Everything below is a stopwatch reading on this aarch64 host. **It is not a
Raspberry Pi 5**, and `tools/benchmark.py` prints that caveat in its own output
and stores it in the JSON. A Pi 5's A76 at 2.4 GHz is roughly 1.4–1.8× this
container per core on numpy-heavy float work; the range is wide, and the only
thing that closes it is running the script on a Pi.

| Stage | Measured |
|---|---:|
| Frame decode (300 B, CRC checked) | **4.0 µs** |
| Frame encode | **5.2 µs** |
| **Score one frame (Mahalanobis, 63 features)** | **16.9 µs** |
| Score 1000 frames (batched) | 4.9 ms → 4.9 µs/frame |
| Feature extraction, 4096 samples | 2.85 ms |
| **Feature extraction, 8192 samples** (what every MCU kit uses) | **3.46 ms** |
| Feature extraction, 16384 samples | 4.59 ms |
| Feature extraction, 32768 samples | 6.86 ms |

### Fan-out

At 3.46 ms per feature vector, offering 75 % of one core:

| Node interval | Nodes one core keeps up with |
|---|---:|
| 1 min | 13 022 |
| 5 min | 65 110 |
| 15 min | 195 332 |
| **1 h** | **781 331** |
| 1 h, four cores | 3 125 324 |

**These numbers are large enough to be the finding.** A large site has a
thousand nodes. Feature extraction is three orders of magnitude from being the
constraint.

What *does* bind a gateway, none of which this benchmark measures:

- the broker's concurrent connection count and per-connection memory;
- the time-series database's write rate and retention compaction;
- the uplink;
- BLE/sub-GHz radio airtime, which is a shared medium and the real fan-in limit.

Measuring those needs the compose stack running, and there is no Docker in this
environment — recorded in `docs/07_production_readiness.md`.

### A measurement that measured nothing, and says so

```
durable      39828 frames/s  (0.025 ms/frame)
nosync       41985 frames/s  (0.024 ms/frame)
durability costs 1.0x throughput on this filesystem
WARNING  durable and nosync are within 20 %: the fsync is not
         reaching a storage device on this host (overlay/tmpfs,
         or a drive that lies about write completion). This
         measurement does NOT validate the SD-card case, which
         is the one that matters -- expect 1-10 ms per fsync and
         100-1000 frames/s there.
```

A real fsync to real storage costs something. 40 000 frames/s with
`synchronous=FULL` is not plausible, so the benchmark detects the equality and
flags it rather than publishing the number. A benchmark that silently measures
nothing is worse than no benchmark, because the number gets quoted.

---

## The AI HAT, and why it is idle

| Workload | Latency |
|---|---:|
| 63-float detector, numpy on one CPU core | **17 µs** |
| 1D CNN over 8192 raw samples, ~4 MMAC, on CPU | ~8 ms |
| the same CNN on a Hailo-8L | ~0.3 ms |
| …for 200 channels at once | still ~0.3 ms |

A Hailo-8L cannot improve on 17 µs in any way that matters: the work is a 63×63
quadratic form, not a tensor graph, and the PCIe round trip alone costs more
than the computation.

So the AI HAT is justified by exactly one thing — **a model class that runs
nowhere else**: a 1D CNN over raw waveforms, which skips the feature extractor
entirely. That matters because every feature in this repository is a decision
about what matters, taken by a human in advance; a raw-waveform model can find a
signature nobody anticipated. It is also why it is risky: it can find the test
rig's mains hum and call it a bearing fault.

**`HailoScorer` raises on construction.** It is not implemented, deliberately,
and the exception says what is missing:

1. a 1D CNN trained on **measured** data — no model here has passed that gate;
2. ONNX export, then `hailo_model_zoo` compilation to a `.hef` with int8
   calibration on real waveforms, not simulator output;
3. a bit-exactness check between the `.hef` and the ONNX reference, of the kind
   `cdsp/tools/gen_golden.py` does for the C path;
4. decision-flip accounting against the float model, with the same thresholds
   `tools/export_model.py` already enforces for the int8 MCU path.

A stub returning plausible probabilities was considered and rejected: it would
make this kit look finished while shipping confident scores from a model that
does not exist. The same reasoning as the Alif branch's waveform model, and the
same conclusion.

**This is the honest recommendation: buy the Pi, skip the HAT**, until (1)
exists. Spend the €70 on the NVMe SSD instead.

---

## Holdover

`tools/holdover.py`, and the answer contradicted the plan this kit was written
against. The plan said "UPS holdover is a correctness requirement". It is not.

50 nodes × 14-day uplink outage = 5 MB of feature frames. Flush time when the
link returns:

| Uplink | Flush |
|---|---:|
| LTE Cat-M1 (shared) | 10 min |
| 4G LTE (plant, contended) | 20.2 s |
| Wi-Fi to site LAN | 2.5 s |
| Gigabit Ethernet | 0.1 s |

**The flush does not need battery at all** — it happens when mains and uplink
are both back. And the backlog is already safe: SQLite with
`synchronous=FULL` means committed frames survive a power cut with no holdover
whatsoever.

So what the UPS is actually for:

1. **A graceful shutdown** — close SQLite cleanly so the WAL is checkpointed and
   the next boot does not have to recover. 30 s is generous.
2. **Riding through brownouts** without a reboot. A Pi that reboots every time
   the plant's compressors start loses the cycle in progress each time.

| Holdover | Energy at 7.5 W | Technology |
|---|---:|---|
| 30 s | 0.06 Wh | supercapacitor |
| **60 s** | **0.12 Wh** | **supercapacitor — recommended** |
| 300 s | 0.62 Wh | LiFePO₄ pack |
| 900 s | 1.88 Wh | LiFePO₄ pack |

**60 s of supercapacitor plus a clean shutdown handler.** Enough for (1) and most
of (2), no battery chemistry in the enclosure, and no maintenance item. Longer
holdover buys uptime, not data integrity.

### Storage is the real data-integrity decision

A UPS does not fix the thing most likely to lose a site's backlog: **if the SD
card dies on an unclean power-down, the data is gone regardless.** An NVMe SSD
or eMMC carrier is a bigger data-integrity win than any UPS, and it costs less.

The queue fsyncs once per frame. At 50 nodes hourly that is 1 200 writes/day —
trivial. At 1 Hz aggregation across a large site it is not, and an SD card would
be consumed in months.

---

## Deployment

```bash
# Raspberry Pi OS (64-bit). Boot from the NVMe, not the SD card.
sudo apt install python3-numpy python3-scipy python3-paho-mqtt
sudo mkdir -p /opt/pdm /etc/pdm /var/lib/pdm
# copy core/ and edge/ to /opt/pdm; config to /etc/pdm

# Prove the stack
PYTHONPATH=/opt/pdm/core:/opt/pdm/edge python3 -m pdmedge --self-test

# The agent as a hardened service
sudo cp edge/systemd/pdm-edge.service /etc/systemd/system/
sudo systemctl enable --now pdm-edge

# And the local stack, if this gateway is also the site's server
docker compose -f cloud/docker-compose.yml up -d
```

A gateway can run the whole `cloud/` compose stack locally — Mosquitto,
VictoriaMetrics, Postgres, Grafana — which is how a site gets dashboards with no
cloud at all. **Never exercised**: there is no Docker in this environment, so the
compose file has never been started. The `MemoryMax=512M` in the systemd unit is
sized for the agent alone; running the stack beside it on an 8 GB Pi is
plausible and unmeasured.

---

## Production checklist

- [ ] Run `tools/benchmark.py` **on an actual Pi 5** and replace this README's
      numbers; the fsync result in particular is invalid here
- [ ] Start the compose stack and measure what actually binds: broker
      connections, database write rate, memory
- [ ] Boot from NVMe; verify the SD slot is empty so nobody "fixes" it later
- [ ] Supercapacitor holdover + a shutdown handler that closes SQLite cleanly
- [ ] Thermal: 7.5 W sustained in an IP54 DIN enclosure at 40 °C ambient. The Pi
      5 throttles at 80 °C; the agent flags `TEMP_OUT_OF_RANGE` but does not stop
- [ ] BLE airtime budget for the node population — the real fan-in limit, and
      nothing here has modelled it
- [ ] Per-device X.509 for the gateway's *upstream* link, and a separate
      credential per node for its *downstream* one (`docs/04_security.md`)
- [ ] Decide whether the gateway may hold node decryption keys. If it can read
      node payloads, compromising one gateway compromises a site
- [ ] Silent-node alarms routed somewhere a human sees during an uplink outage —
      the detection is local, the notification currently is not
- [ ] Queue partition on its own filesystem, so a full log partition cannot stop
      ingestion

---

## Honest status

| | |
|---|---|
| Gateway fan-in, forwarding, queueing | ✅ 18 tests |
| Gap detection, wrap-safe, multi-channel aware | ✅ |
| Local alarms during an uplink outage | ✅ |
| Silent-node detection with learned intervals | ✅ |
| A corrupt frame is rejected, not forwarded | ✅ |
| A failing scorer does not stop forwarding | ✅ |
| Throughput measured | ✅ on **this host**, named in the output |
| Holdover computed | ✅ and it refuted the kit's own plan |
| **Run on a Raspberry Pi 5** | ❌ |
| **fsync cost measured** | ❌ the host swallows it; the benchmark says so |
| **Compose stack started** | ❌ no Docker here |
| **Hailo-8L used** | ❌ raises by design; no trained waveform model exists |
| **BLE/sub-GHz node termination** | ⬜ the radio side is not written — `aggregator.Gateway.handle` takes bytes, and what delivers them is deployment-specific |
| Silent-node notification path | ⬜ detected locally, not yet routed |

`Gateway.handle()` takes an already-received payload. What receives it — a BLE
GATT server, a sub-GHz concentrator, or a local MQTT broker the nodes publish to
— is deployment-specific and not written. That is a deliberate boundary: it is
the one part of this kit that genuinely depends on the radio hardware chosen,
and stubbing it would hide the choice.
