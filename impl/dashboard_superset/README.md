# Apache Superset Analytics Dashboard

## Overview

Apache Superset serves as the analytics and BI layer for the predictive maintenance platform,
connecting to VictoriaMetrics (sensor time-series) and PostgreSQL (alerts, work orders, asset
metadata) to produce manager-facing KPI dashboards and reliability engineer analytics views.

---

## Architecture

```
VictoriaMetrics  ─────────────────────────────────────┐
  (sensor metrics: RMS, kurtosis, temperature,         │
   bearing frequencies, anomaly scores)                │
                                                       ▼
PostgreSQL  ──────────────────────────────────►  Apache Superset  ──► Browser
  (alerts, work orders, assets, downtime events,       │
   maintenance costs, parts inventory)                 │
                                                       ▼
                                                  Celery + Redis
                                                (async reports, email
                                                 alerts, PDF exports)
```

**Superset connects to two datasources:**

| Datasource      | Connection type                          | Used for                                    |
|-----------------|------------------------------------------|---------------------------------------------|
| VictoriaMetrics | MetricsQL via vmselect HTTP API          | RMS trends, kurtosis, bearing frequencies   |
| PostgreSQL      | SQLAlchemy `postgresql+psycopg2://`      | Alerts, work orders, OEE, MTBF/MTTR, costs |

---

## Best For (Use Superset)

- **Manager / executive dashboards**: OEE, availability, performance, quality KPIs
- **Reliability engineer analytics**: fleet health heatmaps, fault distribution, MTBF/MTTR trends
- **Cross-asset analytics**: compare downtime across production lines or asset classes
- **Custom SQL KPIs**: complex aggregations not possible in ThingsBoard widgets
- **Executive PDF reports**: scheduled email with dashboard snapshots
- **RBAC by role**: managers see OEE summary; reliability engineers see detailed sensor trends

## Not For (Use ECharts Custom Component Instead)

- **Real-time FFT spectrum display**: requires Canvas rendering at 64 K frequency bins with
  sub-500 ms refresh — use the Apache ECharts `custom series` + WebSocket component in the
  React app
- **Live bearing defect frequency overlays**: computed client-side from bearing geometry + RPM
- **Real-time alert feed**: use the Socket.IO-backed alert panel in the React app
- **Waterfall / cascade plots**: 3-D time-frequency visualization requires the custom ECharts layer

---

## Superset vs ThingsBoard Decision Guide

| Scenario                                     | Use Superset | Use ThingsBoard |
|----------------------------------------------|:------------:|:---------------:|
| OEE trend chart (last 90 days)               | ✓            |                 |
| Complex SQL KPI (MTBF by asset class)        | ✓            |                 |
| Cross-fleet downtime Pareto chart            | ✓            |                 |
| Executive PDF report (scheduled email)       | ✓            |                 |
| Custom color-coded health score heatmap      | ✓            |                 |
| Real-time device online/offline status       |              | ✓               |
| Simple IoT gauge (current temperature)       |              | ✓               |
| Live sensor value widgets on mobile          |              | ✓               |
| Rule-based threshold alerts (built-in)       |              | ✓               |
| Quick provisioning for 100 + devices         |              | ✓               |

**Rule of thumb**: Superset when you need SQL aggregation, multi-datasource joins, or scheduled
reporting. ThingsBoard when you need live device status or simple threshold widgets.

---

## Tradeoffs

### Superset Advantages
- Rich chart library: 40+ chart types (Echarts, D3, Deck.gl)
- SQL Lab: ad-hoc query interface for reliability engineers
- RBAC: row-level security, dashboard-level permissions
- Scheduled reports: email PDF/image snapshots on a cron schedule
- Alerts: SQL-based threshold alerts with email/Slack notifications

### Superset Disadvantages
- **Requires SQL knowledge** to create new datasets and charts
- **Datasource setup overhead**: VictoriaMetrics requires MetricsQL adapter or ClickHouse-compat
  layer; not plug-and-play like ThingsBoard's MQTT ingestion
- **Not real-time by default**: minimum cache TTL 60 s; live spectrum requires custom component
- **Heavier stack**: Superset + Celery + Redis adds ~2 GB RAM to deployment

---

## Deployment

```bash
# Start the full analytics stack
docker compose up -d

# First-time setup (runs automatically via superset-init container)
# - Initialises database schema
# - Creates admin user
# - Imports OEE and machine health dashboard JSONs
# - Registers VictoriaMetrics and PostgreSQL datasources

# Access Superset
open http://localhost:8088
# Default credentials (change in production): admin / admin
```

See `docker-compose.yml` for service definitions and `init/import_dashboards.sh` for the
first-time import script.

---

## Files in This Directory

| File                                      | Purpose                                              |
|-------------------------------------------|------------------------------------------------------|
| `docker-compose.yml`                      | Superset + PostgreSQL + Redis + Celery + VM stack    |
| `superset_config.py`                      | Superset production configuration                    |
| `datasources/victoriametrics_datasource.py` | Register VM + PG datasources via REST API          |
| `dashboards/oee_dashboard.json`           | Plant OEE dashboard export                           |
| `dashboards/machine_health_dashboard.json`| Machine health & reliability dashboard export        |
| `sql/sensor_views.sql`                    | PostgreSQL views for Superset datasets               |
| `init/import_dashboards.sh`               | First-time import script (dashboards + datasources)  |
