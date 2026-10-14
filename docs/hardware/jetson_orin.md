# Design note — NVIDIA Jetson Orin Nano Super site edge

Companion to [`hardware/jetson_orin/README.md`](../../hardware/jetson_orin/README.md).
The README says what was measured; this records how the kit was approached,
what the approach found, and the two bugs that fell out of it.

---

## 1. The brief: try to kill it

Seven kits had already been built when this one started, and two of them had
independently concluded that accelerators do nothing for feature-vector scoring
— the Alif branch on an Ethos-U55, the Pi 5 branch on a Hailo-8L. A third
negative result from a third accelerator would be a footnote.

So the brief was inverted: rather than ask "what can a Jetson do?", ask

> Is there a workload here where a GPU is the difference between **possible and
> impossible**, rather than between fast and faster?

`tools/workload_study.py` is written to answer that adversarially — it names
three candidates, measures them, and prints a verdict per candidate. All three
came back negative:

| Candidate | Needs | One core supplies | Headroom |
|---|---:|---:|---:|
| Re-score a site-year (8.76 M frames) | 43.8 s | 600 s | 14× |
| Fit the detectors (5000 frames) | 0.10 s | 600 s | 6 010× |
| Raw-waveform inference, 1000 nodes hourly | 0.0011 GMAC/s | ~15 GMAC/s | 13 500× |

Candidate 3 is reported as arithmetic rather than a measurement, because there
is no model to run; the point is the order of magnitude, and the study also
records the cadence at which it would flip — **3 750 inferences/s, i.e. 13.5
million nodes at hourly cadence.**

Candidate 2 is the honest weak spot and the study says so: these are
closed-form and SVD fits, not gradient descent. A Deep SVDD or a 1D CNN is the
one training workload a GPU would change, `torch` is not installed here, and so
**that number is absent rather than guessed**. A test asserts
`torch_available is False` and fails if torch ever appears, so the gap cannot be
silently forgotten.

---

## 2. What survived, and why one of them matters more than it sounds

### 2.1 Data sovereignty

A plant that will not send vibration data off site cannot have its models
trained in a cloud. This is not a hypothetical: pharmaceutical and defence sites
already refuse, and the direction of travel on data-processing agreements is one
way.

That makes the **site** the place where training happens — and a site that
trains needs more than a gateway. It is the only justification in this branch
that is about *where* the computation happens rather than *how fast*, which is
why it survives an argument about throughput.

So `edge/train_onsite.py` trains from the gateway's own store-and-forward queue.
Reading the queue directly, read-only, rather than through `FrameQueue`, is
deliberate: **training must not consume the queue**, because those frames still
have to go upstream.

### 2.2 Multimodal fusion

Vibration + acoustic + motor current + thermal imagery on one asset genuinely
does not fit a 63-float feature vector, and the imagery alone rules out the MCU
and gateway tiers. It is the real GPU case.

It is not written. Three accelerators in this lineup — Ethos-U55, Hailo-8L,
Ampere — and all three are justified by models that do not exist yet. Writing a
stub for any of them would make the platform look finished; the pattern is
consistent across the three branches and is stated the same way in each.

---

## 3. Two bugs, both "checks that cannot fail"

This is the class of defect the whole repository is organised around, and the
training gate produced two of them in succession.

### 3.1 The circular check

First version: fit on the first 75 % of the span, calibrate the threshold on the
last 25 %, then measure the false-alarm rate on **that same 25 %**.

That rate equals the target by construction — it is the definition of
calibrating to an FPR. The check could never fail. The test that was written to
prove it worked (a deliberate 4× monotone drift across the span) passed with a
reported false-alarm rate of exactly 0.010, which is how the bug announced
itself: **a suspiciously exact number.**

Fix: three windows in time order — fit / calibrate / verify — with the threshold
calibrated on the middle and verified on a later window it has never seen. The
same drift now produces 96 alarms where 1.8 were expected.

A random split would not do. It averages the drift into every window and
produces a threshold that looks fine and fails next week, which is the specific
failure a time split exists to surface.

### 3.2 The statistically wrong gate

The replacement refused if the verification false-alarm *rate* exceeded the
target by 4×. On a 175-frame window at a 1 % target the expected count is 1.8,
and Poisson noise alone spans 0–6 at 3 σ — so the gate refused perfectly
stationary data about a third of the time. The test for the *accepting* case
caught it immediately.

That failure mode is worse than it looks. In a plant it means the commissioning
engineer re-runs training until it passes, which **selects for whichever window
happened to look quiet** — a biased model, arrived at by a procedure that feels
like diligence.

Fix: a 3 σ Poisson band on the alarm **count**, not the rate:

```
allowed = λ + 3·√λ + 3,    λ = n_verify × target_fpr
```

The floor of 3 stops the band collapsing at small λ; 3 σ matches the health
index's convention so the numbers are comparable platform-wide. Tested at three
window sizes for both properties: the band must admit ordinary noise (> 2λ) and
still reject a moved population (< 25 % of the window).

**The general lesson, worth more than the fix:** a gate expressed as a ratio of
rates is a gate on a count, and at small counts a ratio is the wrong
parameterisation. The same mistake is available anywhere this platform converts
a rate target into a check — the alarms/day budget and the n-of-m alarm logic
both have the same shape.

---

## 4. The other refusals, and which one matters in practice

| Gate | Threshold | The failure it prevents |
|---|---|---|
| Minimum frames | 400 | A 63-feature covariance estimate needs ~189, and the 50/25/25 split gives the fit window half |
| **Minimum span** | **7 days** | **The one that bites.** A commissioning engineer with a day on site has plenty of frames and nowhere near enough span; a detector that has never seen a Monday start-up alarms on the first one |
| Unscoreable frames | excluded | A detector trained on clipped acquisitions learns the artefacts, then fails to alarm on the real thing because the real thing is *less* extreme |
| `SPEED_UNSTEADY` | excluded by default | Their high-order content is smeared, so training on them teaches the detector that smearing is normal |
| Feature-spec mismatch | excluded | Mixing spec versions in one fit silently mixes two different feature layouts |

The span gate is the one that will be argued about on site, and it should be:
seven days is a *minimum*, not a target. A monthly duty cycle needs a month.

---

## 5. What a review should attack first

1. **Every figure is from the wrong machine**, as on the Pi 5 branch — a 2-core
   aarch64 container, not a 6-core Orin. The conclusion has enormous margin
   (13 500×), so it survives the uncertainty; candidate 2 does not, and that is
   the one to re-measure.
2. **`torch` is absent**, so the only GPU-relevant training number is missing.
   The test that asserts its absence is the right behaviour, not a substitute
   for the measurement.
3. **The model-card provenance is thin.** It records the window, the counts and
   the threshold, but not which features drove the fit or the refusal reasons
   for channels that did *not* train. The second is arguably more valuable: a
   site with 40 nodes and 6 trained models needs to know why 34 were refused.
4. **No rollout or rollback.** A locally trained model that alarms on Monday
   mornings must be revertible without a site visit, and nothing here does that.
5. **The air-gapped update path.** If the site will not send data out, it
   probably will not accept OTA in. Signed offline bundles are a different
   programme from `docs/04_security.md`'s OTA design.

---

## 6. Open items, ordered by what they would change

| Item | Why it matters |
|---|---|
| Install `torch`, re-run candidate 2 | The only measurement that could overturn the conclusion |
| Record refusal reasons per channel in a site report | A site with 34 refused channels needs to know why |
| Model rollout with rollback | On-site training without rollback is a one-way door |
| `pdm-train` systemd timer + unit | Training is currently a manual command |
| Decide 15 W vs 25 W per site | Halves the electricity for a workload that needs none of the GPU |
| Multimodal fusion model | The genuine GPU case, and it does not exist |

---

## 7. What this tier is for, in one paragraph

A site that must keep its data inside the fence. Not for throughput: one CPU
core re-scores a thousand-node year in 44 seconds and fits every detector in a
tenth of a second, and the 67-TOPS GPU is idle by a factor of 13 500. Buy it
when a legal department makes on-site training mandatory, or when a multimodal
model exists that genuinely needs it — and until then buy the Pi 5 and spend the
€31/year difference on nothing at all. The honest output of this branch is a
recommendation against its own hardware, plus a training pipeline that is worth
having on whatever box the site ends up with.
