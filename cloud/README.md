# Cloud / server stack

Everything the server side needs, in one `docker-compose.yml`: MQTT broker,
ingest service, time-series store, CMMS database, dashboards.

```bash
cd cloud
cp .env.example .env && $EDITOR .env      # set the two passwords
docker compose up -d
docker compose logs -f ingest
open http://localhost:3000                # Grafana -> "Fleet overview"
```

## See it work without any of that

The whole chain — device-side feature extraction, wire encoding, CRC
validation, commissioning, health index, alarm escalation, diagnosis, work
order — runs from one command with no broker, no database and no Docker:

```bash
python3 cloud/ingest/app.py --replay
```

```
      day   sev    HI  trend     state  diagnosis                      RUL
     28.3  0.01  0.78   1.06     watch  -                              insufficient_evidence
     42.5  0.02  1.72   1.31     watch  bearing_outer_race (0.80)      insufficient_evidence
     56.6  0.05  5.38   2.53     alarm  bearing_outer_race (1.00)      insufficient_evidence
     70.8  0.19  6.47   4.61  critical  bearing_outer_race (1.00)      already_failed

validation checks
  corrupted CRC      -> rejected
  CLIPPED flag       -> rejected
  replayed sequence  -> deduplicated
  silent for 115 d   -> stale alarm raised
```

This is what every `hw/*` branch's README points at, and what runs in CI. The
four validation checks at the end are negative controls: a POC that only
demonstrates the happy path has not demonstrated the thing that matters.

## What is in the stack, and what is deliberately not

| Service | Image | Role |
|---|---|---|
| `mosquitto` | `eclipse-mosquitto:2.0` | MQTT broker, 1883 plain / 8883 TLS |
| `ingest` | built from `cloud/ingest/` | decode → validate → score → alarm → work order |
| `victoriametrics` | `victoria-metrics:v1.102.0` | time series, 2-year retention |
| `postgres` | `postgres:16-alpine` | assets, alarms, work orders, model registry |
| `grafana` | `grafana:11.3.0` | provisioned dashboards |

**No Kafka.** The upstream plan specifies Kafka + Flink, and that is the right
architecture at a certain scale — but not at the start. The MQTT → service →
VictoriaMetrics path handles a few thousand sensors on one node, and each sensor
produces 300 bytes per hour. Kafka earns its operational cost when ingest must
be *replayable* and *multi-consumer*; adding it on day one buys a
ZooKeeper-shaped problem and no capability. The trigger conditions are written
down in `docs/00_architecture.md` so the decision gets revisited on evidence
rather than on taste.

**No Flink.** Same reasoning. The windowed feature extraction Flink would do
already happens on the device, which is the whole point of the edge-feature
architecture.

**No Postgres writes from the ingest hot path yet.** The schema is complete and
loaded (`sql/001_schema.sql`), and the service holds work orders in memory. The
transactional write is the next increment; it is called out here rather than
implied to be done, because "the schema exists" and "the service uses it" are
different claims.

## Retention and sizing

Per sensor, per hour: one 300-byte frame → ~12 time series × 8 bytes after
VictoriaMetrics compression ≈ 100 B/h ≈ 0.9 MB/year. 1 000 sensors is under
1 GB/year of time series, which is why the retention is set to 24 months rather
than something anxious.

Raw waveform snapshots are different: 1 s at 25.6 kHz, 16-bit, is 51 kB. At one
snapshot per alarm plus one scheduled per asset per month, 1 000 sensors is
~600 MB/year. Those go to object storage with the `snapshot` table as the index,
not into Postgres.

## Securing the broker

The shipped `mosquitto.conf` has `allow_anonymous true`, which is acceptable
**only** on an isolated sensor VLAN. On any other topology an open broker lets
anyone on the network publish a fabricated alarm — a safety problem, not merely
a security one, because a fabricated *healthy* reading is just as easy to inject
as a fabricated alarm.

For anything beyond a lab:

```bash
docker compose exec mosquitto mosquitto_passwd -c /mosquitto/config/passwd pdm-fleet
# then in mosquitto.conf:
#   allow_anonymous false
#   password_file /mosquitto/config/passwd
```

For devices on a routed or cellular path, enable the 8883 TLS listener with
`require_certificate true` and per-device client certificates. Pre-shared
passwords across a fleet mean one extracted device compromises every device —
and extracting a sensor from a plant wall is not a sophisticated attack. See
`docs/04_security.md` for the provisioning model.

## Environment

`.env.example`:

```
POSTGRES_PASSWORD=change_me
GRAFANA_PASSWORD=change_me
```

Both have development defaults in `docker-compose.yml` so the stack starts
without configuration. Those defaults are fine on a laptop and are a finding in
any real deployment; the compose file says so at the point of definition.

## Ingest endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/telemetry` | one binary frame (`application/octet-stream`) |
| `GET` | `/metrics` | Prometheus text exposition |
| `GET` | `/healthz` | liveness |
| `GET` | `/assets` | per-asset summary: state, index, frames, gaps, open orders |

```bash
# A gateway that cannot speak MQTT
curl -s --data-binary @frame.bin \
     -H 'Content-Type: application/octet-stream' \
     http://localhost:8080/telemetry | python3 -m json.tool
```

The metrics worth alerting on in your own monitoring (not in the PdM product —
these are about the *platform's* health, which is a separate concern from the
machines'):

| Metric | Alert when |
|---|---|
| `pdm_frames_rejected_crc_total` | rising — radio or hardware fault in the field |
| `pdm_frames_rejected_version_total` | non-zero — a fleet is running incompatible firmware |
| `pdm_frames_missed_total` | rising — sensors dropping frames while appearing healthy |
| `pdm_alarm_transitions_total_stale` | rising — sensors going silent |
| `pdm_timeseries_write_errors_total` | rising — trend data being lost silently |

## Dashboards

`grafana/dashboards/fleet_overview.json` is provisioned automatically. Three
rows: fleet state, per-asset detail, and **device health kept separate from
machine health** — a flat battery and a failing bearing are different problems
with different owners, and merging them onto one screen is how one gets ignored.

Two panels carry a deliberate caveat in their description, because the number is
routinely over-read:

- **ISO velocity RMS** is shown because customers and insurers ask for it, and
  alarms are explicitly *not* based on it: bearing energy lives above the 1 kHz
  measurement band, so a bearing can fail with that line almost flat.
- **Kurtosis** is not monotonic with damage. It rises as a defect localises and
  falls back toward 3 as the defect spreads over the raceway, so a rule that
  watches kurtosis alone will declare a severely damaged bearing healthy.
