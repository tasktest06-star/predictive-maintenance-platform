# NVIDIA Jetson Orin Nano Super — the site edge tier

The hardest kit here to justify, so this branch tried to kill it.

**The finding: nothing in this platform needs a Jetson.**
`tools/workload_study.py` is adversarial by construction — it looks for a
workload a 67-TOPS GPU would help with and reports honestly when it cannot
find one:

| Candidate | Needs | One CPU core supplies | Headroom |
|---|---:|---:|---:|
| Re-score a site-year (1000 nodes × 365 d, 8.76 M frames) | 43.8 s | 600 s budget | **14×** |
| Fit the detectors (5000 frames) | 0.10 s | 600 s budget | **6 010×** |
| Raw-waveform inference (1000 nodes hourly) | 0.0011 GMAC/s | ~15 GMAC/s | **13 500×** |

Two justifications survived, and **neither is about speed**. One of them is
implemented here.

**Status: never run on a Jetson.** Every figure is measured on this 2-core
aarch64 host and says so. `torch` is not installed, so the one training workload
a GPU would genuinely change — gradient descent — is **absent rather than
guessed**, and a test asserts that absence so the gap cannot be forgotten.

---

## See it work now

```bash
# The adversarial study, which argues against its own hardware
PYTHONPATH=core:edge python3 hardware/jetson_orin/tools/workload_study.py

# On-site training against a gateway's own queue (the surviving justification)
PYTHONPATH=core:edge python3 hardware/jetson_orin/edge/train_onsite.py \
    --queue /var/lib/pdm/gateway.sqlite --out /var/lib/pdm/models

# 13 tests, including the two bugs the training gate used to hide
PYTHONPATH=core:edge python3 -m pytest hardware/jetson_orin/tests -q
```

---

## Bill of materials

| Item | Part | Qty | Notes |
|---|---|---:|---|
| Compute | **Jetson Orin Nano Super 8 GB dev kit** | 1 | ~€250. 6× A78AE + 1024-core Ampere, 67 TOPS at 25 W |
| Storage | NVMe SSD, 1 TB | 1 | ~€60. A site's history is the point — 2 M frames is 600 MB, 4.5 years of a 50-node site |
| Enclosure | Vented 19″ or DIN, with fan | 1 | 25 W sustained is not a sealed-box thermal budget |
| Supply | 19 V 65 W barrel | 1 | — |
| **Legacy note** | Jetson Nano (2019, 4 GB) | — | See [legacy](#legacy-jetson-nano) |

Around **€320**, plus €44/year of electricity against the Pi 5's €13 — the
cheapest part of being wrong about this board.

---

## What survived the study

### 1. Data sovereignty — implemented

A plant that will not send vibration data off site cannot have its models
trained in a cloud. Pharmaceutical, defence, and increasingly any site whose
legal team has read a data-processing agreement: the data stays inside the
fence. That makes the **site** the place where training happens, and a site that
trains needs more than a gateway.

[`edge/train_onsite.py`](edge/train_onsite.py) trains from what the site already
has — the gateway's own store-and-forward queue — and produces deployable model
cards without a frame leaving the building:

```
frames excluded before training:
      40  unscoreable (clipped/overrun/selftest)

3 device-channel(s) in the queue
  00000000000000a1/ch0  ACCEPTED  350/175/175 fit/cal/verify over 29.1 d,
                                  threshold 9.457, verify 2/175 alarms
                                  (<= 8.7 allowed), fit 90.3 ms
  00000000000000a2/ch0  REFUSED   frames span only 0.3 days; 7 needed to cover a
                                  weekly duty cycle. A detector that has never
                                  seen a Monday start-up will alarm on the first one
  00000000000000a3/ch0  REFUSED   96 false alarms on the later verification window
                                  of 175 frames, against 1.8 expected and 8.7
                                  allowed at 3 sigma: the population moved between
                                  the calibration and verification windows, so this
                                  threshold would not hold next week

1 of 3 accepted
```

**The refusals are most of the value.** A model trained on whatever happened to
be in a queue is a model trained on an unknown population. The gates:

- **400 usable frames minimum.** The 50/25/25 split gives the fit window 200,
  against the ~189 a 63-feature covariance estimate needs.
- **7 days of span minimum.** This is the refusal that matters most in
  practice: a commissioning engineer with a day on site has plenty of frames
  and nowhere near enough span. A detector that has never seen a Monday
  start-up will alarm on the first one.
- **No unscoreable frames.** A detector trained on clipped acquisitions learns
  the artefacts, then fails to alarm on the real thing because the real thing
  is *less* extreme than its training data.
- **No `SPEED_UNSTEADY` frames** by default — their high-order content is
  smeared, so training on them teaches the detector that smearing is normal.
- **Three windows in time order**, and a Poisson gate. See below.
- It does **not** consume the queue: those frames still have to go upstream.

### 2. Multimodal fusion — not implemented, and why

Vibration + acoustic + motor current + thermal imagery on one asset is a model
class that does not fit a 63-float feature vector at all, and the imagery alone
rules out the MCU and gateway tiers. It is the genuine GPU case.

It is **not** written, for the same reason the Alif branch's waveform model and
the Pi 5 branch's Hailo path are not: there is no model trained on measured data
to deploy. Three accelerators in this lineup — Ethos-U55, Hailo-8L, Ampere — and
all three are justified by models that do not exist yet. That is the honest
state of the art in this repository, and writing a stub for any of them would
make it look otherwise.

---

## Two bugs the training gate used to hide

Both found by its own tests, and both of the class this repository is organised
around: **checks that cannot fail.**

### A check that always passed

The first version split the span **two** ways: fit on the first 75 %, calibrate
the threshold on the last 25 %, then measure the false-alarm rate on **that same
25 %** — which equals the target *by construction*, because that is what
calibrating to an FPR means. The check could never fail. A deliberate 4×
monotone drift across the span sailed straight through it.

Three windows in time order fix it: **fit / calibrate / verify**, with the
threshold calibrated on the middle and verified on a later window it has never
seen. A population that moved then shows up as 96 alarms where 1.8 were
expected.

A random split would not do: it averages the drift into every window and
produces a threshold that looks fine and fails next week.

### A gate that was statistically wrong

The replacement check refused if the verification false-alarm *rate* exceeded
the target by 4×. On a 175-frame window with a 1 % target, the expected count is
1.8 — and Poisson noise alone spans 0 to 6 at 3 σ. A "4× the rate" gate
therefore refuses perfectly stationary data roughly a third of the time.

In a plant that means the commissioning engineer re-runs training until it
passes, **selecting for whichever window happened to look quiet** — worse than
no gate at all.

So the gate is a 3 σ Poisson band on the alarm *count*:

```
allowed = λ + 3·√λ + 3      where λ = n_verify × target_fpr
```

The floor of 3 keeps the band from collapsing when λ is small, and 3 σ is the
same convention the health index uses, so the numbers are comparable across the
platform. It admits ordinary noise (2 of 175 alarms passes) and still rejects a
population that moved (96 of 175 does not).

---

## Energy, and the cost of being wrong

| Board | Sustained | Per year | At €0.20/kWh |
|---|---:|---:|---:|
| Raspberry Pi 5 + SSD | 7.5 W | 66 kWh | **€13** |
| Jetson Orin Nano (15 W mode) | 15.0 W | 131 kWh | €26 |
| **Jetson Orin Nano Super (25 W)** | 25.0 W | 219 kWh | **€44** |

€31/year more than a Pi, per site. If the GPU is idle — and the study says it
is — that is the cheapest mistake in the deployment, which is precisely why it
goes unexamined. The study exists so somebody examines it.

The 25 W figure is the mode that reaches 67 TOPS. Running the board in 15 W mode
halves the electricity and loses roughly half the GPU, which for a workload that
needs 1/13 500th of one CPU core is a free saving.

---

## Legacy: Jetson Nano

The original 2019 Jetson Nano (4 GB, Maxwell, 472 GFLOPS) was named in this
project's kit selection. It is **end-of-life**: the last supported JetPack is
4.6.x on Ubuntu 18.04, with Python 3.6 and a CUDA stack that no current
framework targets.

Everything in this branch runs on it in principle — the agent is numpy and the
training is closed-form — but Ubuntu 18.04 has been out of standard support
since 2023, and deploying an unpatched internet-connected Linux to a plant
network is a finding, not a trade-off.

**Recommendation: Orin Nano Super for new work.** If existing Nano hardware must
be used, run it air-gapped behind a gateway, and read `docs/04_security.md`
first.

---

## Deployment

```bash
# JetPack 6.x (Ubuntu 22.04). Boot from NVMe.
sudo apt install python3-numpy python3-scipy python3-paho-mqtt
sudo mkdir -p /opt/pdm /etc/pdm /var/lib/pdm/models

PYTHONPATH=/opt/pdm/core:/opt/pdm/edge python3 -m pdmedge --self-test
sudo cp edge/systemd/pdm-edge.service /etc/systemd/system/
sudo systemctl enable --now pdm-edge

# Weekly on-site training, as a timer rather than a cron job so the
# dependency on the agent's queue is explicit
sudo systemctl enable --now pdm-train.timer    # unit not written; see below
```

`nvpmodel -m 0` selects the 25 W mode. Consider `-m 1` (15 W): see
[energy](#energy-and-the-cost-of-being-wrong).

---

## Production checklist

- [ ] Run `tools/workload_study.py` **on a Jetson** and with `torch` installed,
      so candidate 2 measures a gradient-descent fit — the only number that
      could change the conclusion
- [ ] Decide 15 W vs 25 W deliberately; the default is not free
- [ ] `pdm-train.timer` + service unit (not written)
- [ ] Model-card provenance: who trained it, on what window, and the refusal
      reasons for every channel that did *not* train
- [ ] Model rollout with rollback — a locally trained model that alarms on
      Monday mornings must be revertible without a site visit
- [ ] Thermal: 25 W sustained in a plant enclosure at 40 °C needs airflow and a
      calculation nobody has done
- [ ] NVMe boot; the eMMC/SD path is for development only
- [ ] If the site is air-gapped, the whole update path changes — signed offline
      bundles rather than OTA
- [ ] JetPack pinned; an unpinned `apt` source on a plant network is a silent
      dependency change

---

## Honest status

| | |
|---|---|
| Workload study, three candidates | ✅ all negative, committed as JSON |
| On-site training with gates | ✅ 13 tests |
| Three-window time split | ✅ and it caught a 4× drift the two-window version missed |
| Poisson verification gate | ✅ tolerates noise, rejects drift |
| Training does not consume the queue | ✅ |
| Unscoreable / unsteady frames excluded | ✅ |
| **Run on a Jetson** | ❌ every figure is from this 2-core aarch64 host |
| **`torch` installed** | ❌ so the one GPU-relevant training number is absent, asserted by a test |
| **GPU used at all** | ❌ the study says it should not be |
| **Multimodal fusion** | ⬜ the genuine GPU case; no model exists |
| `pdm-train` systemd units | ⬜ not written |
| Model rollout / rollback | ⬜ not written |

The conclusion of this branch is a recommendation against its own hardware for
the workload as it stands today. That is the useful answer: **buy the Pi 5 and
revisit this when a multimodal or waveform model exists, or when a site's legal
department makes on-site training mandatory.** The second condition is the one
that will arrive first.
