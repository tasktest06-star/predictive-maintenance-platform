# Design note — Raspberry Pi 5 + Hailo-8L gateway

Companion to [`hardware/rpi5_hailo/README.md`](../../hardware/rpi5_hailo/README.md).
The README says what to buy and what was measured; this records why the tier
exists, which two of its design assumptions the measurements refuted, and what
is open.

---

## 1. The brief, and why it is not "a node with more CPU"

The five MCU branches and the CN0549 all answer "can this thing measure a
machine?". This tier answers a different question: **what has to exist between
a hundred battery nodes and a cloud?**

The tempting answer is "something with more compute". The benchmark says that is
wrong by three orders of magnitude: one aarch64 core feature-extracts for
781 331 hourly nodes, and a large site has a thousand. If compute were the
reason for a gateway, a gateway would not be needed.

The four real reasons, in the order they would be missed:

1. **Protocol translation.** A BLE node cannot reach a cloud broker. Something
   has to terminate the radio and speak MQTT/TLS upstream — and that something
   is the only mains-powered element in the chain anyway.
2. **Site-scale store-and-forward.** A node's flash queue holds days; a
   gateway's SSD holds months.
3. **Local autonomy.** An alarm requiring a cloud round trip is unavailable
   during the outage that most often accompanies a real failure.
4. **Knowing which nodes have gone quiet.** §3.

---

## 2. Two refuted assumptions

### 2.1 "The AI HAT accelerates scoring"

Measured: 16.9 µs to score a 63-feature frame with a Mahalanobis detector on one
core; 4.9 µs/frame batched. A Hailo-8L at 13 TOPS cannot improve on that, and
not because it is slow — because the work is a 63×63 quadratic form rather than
a tensor graph, and the PCIe round trip costs more than the arithmetic.

This is the same finding as the Alif branch, reached independently on different
silicon with a different accelerator, and it is worth stating as a general
result: **for feature-vector scoring, the accelerator is never the answer. The
feature extractor is 200× the cost of the model, and the acquisition is 100× the
cost of the extractor.**

What survives as a justification is a model class, not a speed-up: a 1D CNN over
raw waveforms (~4 MMAC, ~8 ms on CPU, ~0.3 ms on the HAT, and ~0.3 ms for 200
channels at once). That skips the feature extractor entirely, which is both the
appeal — every feature here is a human decision about what matters, taken in
advance — and the risk: a waveform model can learn the test rig's mains hum.

`HailoScorer` therefore **raises on construction** and lists the four things
missing (trained model, compilation, bit-exactness check, flip accounting). A
stub returning plausible probabilities would make the kit look finished while
shipping confident scores from a nonexistent model, which is the most
misleading artefact this repository could contain.

The resulting recommendation is uncomfortable and correct: **buy the Pi, skip
the HAT.** Spend the €70 on the NVMe SSD, where it buys data integrity.

### 2.2 "UPS holdover is a correctness requirement"

This kit's own plan said so. It is not.

The backlog is already safe: SQLite with `synchronous=FULL` fsyncs every frame,
so committed data survives a power cut with no holdover at all. And the flush,
when the uplink returns, does not happen on battery — it happens when mains and
uplink are both back:

| Uplink | Flush of 50 nodes × 14 days (5 MB) |
|---|---:|
| LTE Cat-M1 (shared) | 10 min |
| 4G LTE (plant, contended) | 20.2 s |
| Wi-Fi to site LAN | 2.5 s |
| Gigabit Ethernet | 0.1 s |

So the UPS exists for a graceful SQLite close (30 s is generous) and for riding
through brownouts without a reboot. **60 s of supercapacitor, 0.12 Wh at 7.5 W** —
no battery chemistry in the enclosure, no maintenance item.

And the correction that matters more than the UPS: **storage choice.** If the SD
card dies on an unclean power-down the backlog is gone whatever the holdover. An
NVMe SSD or eMMC carrier is a larger data-integrity win and costs less than the
UPS it partly replaces.

---

## 3. Silent-node detection, and why it belongs here

A node that stops reporting is the commonest real event in a deployed fleet and
is **invisible downstream**: a silent node and a healthy quiet machine produce
identical data, namely none. Only the thing the nodes report *to* can tell the
difference.

Three design decisions, each with a failure mode behind it:

**The interval is learned, not configured.** Fleet intervals are changed from the
cloud, so a gateway's configured copy goes stale and silence thresholds become
wrong in whichever direction the config drifted.

**Learning is slow (EWMA 0.8/0.2).** One early frame must not be allowed to pull
the learned interval down and raise the silence threshold — which would mask the
outage that follows. `test_interval_learning_is_slow_to_believe_a_change` pins
this: ten hourly observations followed by a one-minute arrival must leave the
estimate above 45 minutes.

**Three intervals, not one.** A single missed cycle is normal on a congested BLE
link. An alarm on every missed frame trains the operator to ignore the alarm,
which is the failure this whole repository is organised around avoiding.

**Silence is measured from arrival, not from the frame's timestamp.** A node with
a wrong clock would otherwise look permanently fresh. This is also why the
tests have to drive the gateway's clock explicitly, and why getting it wrong
produced three test failures before producing none.

**Open:** detection is local and the *notification* is not. A silent-node alarm
raised during an uplink outage currently goes to a log. Routing it somewhere a
human sees — a local HMI, an SMS modem, a dry contact — is unwritten and is the
first thing a real deployment would need.

---

## 4. Other decisions worth recording

### 4.1 A corrupt frame is rejected, not forwarded

Passing a frame that fails CRC upstream "so the cloud can sort it out" converts
a local radio problem into a fleet-wide data-quality problem — and the gateway
knows strictly more than the cloud about which link it came from. So it is
counted and dropped. `test_a_corrupt_frame_is_not_forwarded`.

### 4.2 A failing scorer must not stop forwarding

A corrupt model file, a bad quantisation, an exception in a new backend: any of
these must degrade the gateway to "forwards everything, scores nothing" rather
than taking the site's data path down. The scoring call is wrapped and the score
becomes NaN. `test_a_failing_scorer_does_not_take_the_gateway_down`.

### 4.3 Unscoreable frames are forwarded but not scored

A clipped or overrun frame must not be scored — a confident anomaly on an
artefact is worse than no data — but it must still reach the cloud, which needs
to know the acquisition failed. Both halves are tested.

### 4.4 Gap detection is per frame, not per acquisition

A three-axis node advances its sequence once per *frame*, so N, N+1, N+2 is
contiguous. Treating a multi-channel node's frames as one-per-acquisition would
report every such node as losing two thirds of its data.

The wrap-around case cost a test rewrite: 65534 → 1 means 65535 *and* 0 are
missing, which is two frames. The code was right; the first version of the test
asserted one. Wrap arithmetic is easier to get wrong in a test than in the thing
being tested.

---

## 5. What a review should attack first

1. **Every number is from the wrong machine.** The benchmark is measured, which
   is more than any MCU figure here, but it is measured on a 2-core aarch64
   container with a filesystem that swallows `fsync`. The script names its host
   and flags the fsync anomaly; it cannot fix either.
2. **The radio side does not exist.** `Gateway.handle()` takes bytes. What
   delivers them — a BLE GATT server, a sub-GHz concentrator, a local broker —
   is unwritten, and it is the one part that genuinely depends on the radio
   hardware chosen. Stubbing it would hide the choice.
3. **BLE airtime, not CPU, is the fan-in limit.** A shared medium with a hundred
   nodes waking on their own schedules is a contention problem nobody here has
   modelled, and it is the number that would actually size a gateway's node
   count.
4. **The compose stack has never started.** Running Mosquitto,
   VictoriaMetrics, Postgres and Grafana beside the agent on an 8 GB Pi is
   plausible and unmeasured. `MemoryMax=512M` in the systemd unit covers the
   agent alone.
5. **Key custody.** If the gateway can read node payloads, compromising one
   gateway compromises a site. The alternative — end-to-end encryption from node
   to cloud — breaks local scoring, which is reason (3) for having a gateway.
   That trade is undecided and should be decided deliberately.

---

## 6. Open items, ordered by what they would change

| Item | Why it matters |
|---|---|
| Run the benchmark on a Pi 5 | Every throughput number here is from a different machine |
| Start the compose stack and measure what binds | The CPU is three orders of magnitude from being the limit; something else is the limit and nobody knows what |
| Model BLE airtime for the node population | The real fan-in constraint |
| Write the radio termination | The gateway cannot receive anything until this exists |
| Route silent-node alarms to a human | Detection without notification is half a feature |
| Decide the key-custody model | Local scoring and end-to-end encryption are in tension |
| NVMe boot as the default, SD as unsupported | The largest data-integrity win available, and the cheapest |

---

## 7. What this tier is for, in one paragraph

One box per site, around €200, carrying a hundred nodes. It exists to terminate
radios the cloud cannot reach, to hold a site's data for months when the uplink
fails, to raise alarms when the cloud is unreachable, and to know which of its
nodes have stopped talking — the last of which nothing else in the system can
do. It does **not** exist to compute: one core covers three orders of magnitude
more nodes than any plant has, and the 13-TOPS accelerator beside it has nothing
to do until somebody trains a raw-waveform model on measured data. Buy the Pi,
skip the HAT, and spend the difference on an SSD.
