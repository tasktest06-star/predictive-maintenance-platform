# Operations runbook

For the people who get paged. Each section is a symptom, not a component.

## Health checks

```bash
curl -s localhost:8080/healthz                       # ingest liveness
curl -s localhost:8080/assets | python3 -m json.tool # per-asset state
curl -s localhost:8080/metrics | grep -E 'rejected|missed|stale'
curl -s localhost:8428/health                        # VictoriaMetrics
docker compose ps                                    # everything
```

## Symptom: a burst of alarms across many assets at once

**Almost never a fleet of machines failing simultaneously.** In order of
likelihood:

1. **A model or threshold was deployed.** Check `model_version` for a row added
   near the alarm timestamps. Roll back by repointing the service at the
   previous `sha256`.
2. **A plant-wide operating change** — a line speed change, a new product, a
   chiller starting for the season. Check whether the alarming assets share a
   speed or load bin. The operating baseline bins by speed and load for exactly
   this, but a *new* operating point has no baseline and legitimately reads as
   unknown.
3. **A feature-spec version mismatch.** `pdm_frames_rejected_version_total` will
   be non-zero. Frames are rejected, not mis-scored, so this shows as missing
   data rather than false alarms — if you see both, there are two problems.
4. **Clock skew.** Frames arriving with timestamps out of order will raise
   `ValueError` in the alarm machine (timestamps must be non-decreasing).
   Check device NTP/cellular time sync.

## Symptom: an asset has gone quiet

A silent sensor is indistinguishable from a healthy one on a trend chart. The
platform raises `STALE` after `stale_after_s` (default 3 h) — if you are looking
at a flat line and there is no STALE alarm, the staleness check is not running.

```bash
curl -s localhost:8080/assets | python3 -c "
import json,sys
for d,a in json.load(sys.stdin).items():
    print(d, a['state'], 'frames', a['frames'], 'gaps', a['gaps'])"
```

| Finding | Likely cause | Action |
|---|---|---|
| `battery_mv` trending down for weeks | end of life | schedule replacement; check the interval is not too aggressive |
| `rssi_dbm` degrading | new obstruction, gateway moved | survey; consider a repeater |
| `gaps` rising but frames still arriving | marginal link, queue overflowing | lengthen the interval or fix the link |
| Nothing at all, battery was fine | device fault, or physically removed | site visit; use `identify` on neighbours to confirm which one is missing |

## Symptom: frames are being rejected

Three counters, three different owners:

| Counter | Owner | Meaning |
|---|---|---|
| `pdm_frames_rejected_crc_total` | hardware/radio | Corruption *before* transmission: brown-out during the sensor FIFO read, or flash wear in the store-and-forward queue. A storm on one device is a failing device. A storm across a site is a power-quality problem. |
| `pdm_frames_rejected_version_total` | fleet management | Firmware and cloud disagree on the feature layout. The fleet is dark, not wrong. Finish the rollout or re-pin the service. |
| `pdm_frames_rejected_flags_total` | the sensor | `CLIPPED`, `FIFO_OVERRUN` or `SELFTEST_FAIL`. See below. |

### `CLIPPED`

The ADC hit full scale, so the amplitudes are invalid and the frame is not
scored. Causes, in order: the machine genuinely got much worse; the sensor's
full-scale range is too low for this asset (a ±2 g part on a machine with 5 g
impacts); mechanical impact on the sensor itself (something hit it).

Do **not** respond by raising the alarm threshold. Respond by checking the
machine, then by re-ranging the sensor.

### `FIFO_OVERRUN`

Samples were lost, so the spectrum is corrupted. Means the MCU did not service
the sensor FIFO in time — a firmware scheduling problem, usually a long
interrupt or a radio transmission colliding with acquisition. A per-device
pattern points at firmware; a fleet-wide pattern points at a release.

## Symptom: too many nuisance alarms

The number that decides whether anyone keeps reading the alerts. Target is
typically **under one nuisance alarm per sensor per month**.

```bash
# alarms per day across the fleet
curl -s localhost:8428/api/v1/query --data-urlencode \
  'query=sum(rate(pdm_alarm_transitions_total_watch[7d]))*86400'
```

In order:

1. **Is the commissioning window clean?** If a technician bumped the machine, or
   the asset ran dry for a minute during commissioning, the baseline is
   contaminated. The scaler is robust to ~50 % contamination, but the *operating
   baseline* is not robust to a fault being present. Re-commission.
2. **Does the asset have operating points with no baseline?** Check
   `baseline_cells` and the coverage map. A VFD asset sweeping 30–60 Hz needs
   samples in each speed bin; until it has them it is "unknown" and should not
   be alarming.
3. **Tighten `n_of_m` or `min_dwell_s` before touching thresholds.** Debouncing
   costs detection delay in proportion; raising a threshold costs sensitivity
   permanently. Quantify both with
   `pdmcore.eval.metrics.alarms_per_day(..., consecutive_required=N)` and
   `detection_delay(...)`.
4. **Only then re-derive thresholds** from the healthy distribution at an agreed
   false-positive rate (`AnomalyDetector.threshold_at_fpr`). Never pick a number
   because it looks calm.

## Symptom: a work order was wrong

This is the feedback loop, not an incident. Record the outcome:

```sql
UPDATE work_order
   SET status = 'closed', outcome = 'not_confirmed',
       outcome_notes = 'Bearing inspected, no damage. Coupling guard was loose and contacting.'
 WHERE id = 1234;
```

Then look at `programme_performance`. A PdM programme that cannot state its own
precision will eventually be switched off, and the only way to know it is from
recorded outcomes.

Note the example: "no bearing damage, but the coupling guard was loose" is a
`not_confirmed` *diagnosis* and a correct *detection*. Both facts matter, and
the notes field is where that distinction survives.

## Routine: commissioning a new asset

1. Record the asset in Postgres: tag, class, power, nameplate rpm, **mounting
   stiffness** (rigid vs flexible — it shifts every ISO threshold by ~1.6×, so
   it is a reviewed decision, not a guess) and bearing geometry.
2. Mount the sensor. **Record the mount type.** A magnet mount rolls off above
   ~2 kHz and removes most of the bearing band; a stud mount reaches past
   10 kHz. A surprising spectrum is explained by this more often than by the
   machine.
3. Let it collect ≥ 40 frames across the asset's normal operating points. The
   dashboard shows `commissioning`, not `normal` — the asset is *unknown*, and
   no alarm can fire.
4. Run a kurtogram on a snapshot to choose the demodulation band; push it as
   `cfg`.
5. Confirm the ISO zone is A or B. If a freshly commissioned machine is in C,
   either the machine has a problem or the class is wrong. Resolve before
   arming.

## Routine: deploying a new model

1. Validate against the gate in `docs/03_model_cards.md`.
2. Register: weights hash, feature config, split protocol, metrics, approver.
3. **Shadow first.** Score with both models, compare alarm decisions, and look
   at the disagreements individually. A model that agrees everywhere is not an
   upgrade; a model that disagrees everywhere is a bug.
4. Canary one site, then 5 %, then the fleet.
5. Watch `alarm_transitions_total_*` rates, not just AUC.

## Routine: backup and restore

```bash
docker compose exec postgres pg_dump -U pdm pdm | gzip > pdm-$(date +%F).sql.gz
# VictoriaMetrics: use vmbackup; the volume is not safe to copy while running
```

Postgres is the one that must be restored: it holds assets, alarm history, work
orders and the model registry. Losing the time series loses trend history, which
is painful; losing Postgres loses the audit trail, which is unrecoverable.

Test the restore. An untested backup is a belief.
