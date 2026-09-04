"""
Register VictoriaMetrics and application PostgreSQL as Superset datasources,
then create virtual datasets for common sensor metric views.

Usage (run inside the superset-init container or locally):

  python datasources/victoriametrics_datasource.py \
      --superset-url http://localhost:8088 \
      --superset-user admin \
      --superset-password admin \
      --vm-url http://victoriametrics:8428 \
      --pg-url "postgresql+psycopg2://app_ro:app_ro@postgres:5432/maintenance"

VictoriaMetrics connectivity notes:
  - vmselect exposes a PromQL/MetricsQL HTTP API at /api/v1/query_range
  - For Superset SQL connectivity, we use the `sqlalchemy-victoriametrics` dialect
    which translates SELECT statements to MetricsQL queries.
  - Alternatively, VictoriaMetrics Enterprise ships a ClickHouse-compatible SQL
    frontend; the open-source edition uses the sqlalchemy-victoriametrics adapter.
  - Connection string: victoriametrics+http://victoriametrics:8428/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Superset REST API client (thin wrapper)
# ---------------------------------------------------------------------------


class SupersetClient:
    """Minimal Superset REST API client using cookie-based session auth."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._access_token: str = ""
        self._csrf_token: str = ""

    def login(self, username: str, password: str) -> None:
        resp = self.session.post(
            f"{self.base_url}/api/v1/security/login",
            json={"username": username, "password": password, "provider": "db", "refresh": True},
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {self._access_token}"})

        # Fetch CSRF token for state-mutating requests
        csrf_resp = self.session.get(f"{self.base_url}/api/v1/security/csrf_token/")
        csrf_resp.raise_for_status()
        self._csrf_token = csrf_resp.json()["result"]
        self.session.headers.update({"X-CSRFToken": self._csrf_token})

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self.session.get(f"{self.base_url}{path}", **kwargs)

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self.session.post(f"{self.base_url}{path}", **kwargs)

    def put(self, path: str, **kwargs: Any) -> requests.Response:
        return self.session.put(f"{self.base_url}{path}", **kwargs)


# ---------------------------------------------------------------------------
# Database (datasource) registration
# ---------------------------------------------------------------------------


def register_victoriametrics(client: SupersetClient, vm_url: str) -> int:
    """
    Register VictoriaMetrics as a Superset database using the
    sqlalchemy-victoriametrics dialect.

    Returns the Superset database ID.
    """
    vm_host = vm_url.rstrip("/").replace("http://", "").replace("https://", "")
    sqlalchemy_uri = f"victoriametrics+http://{vm_host}/"

    payload = {
        "database_name": "VictoriaMetrics (sensor metrics)",
        "sqlalchemy_uri": sqlalchemy_uri,
        "expose_in_sqllab": True,
        "allow_run_async": True,
        "allow_ctas": False,
        "allow_cvas": False,
        "allow_dml": False,
        "extra": json.dumps(
            {
                "metadata_params": {},
                "engine_params": {
                    "connect_args": {
                        "verify_ssl": False,
                        "extra_labels": {
                            # Default label filters applied to all MetricsQL queries
                            "job": "acoustic_sensor",
                        },
                    }
                },
                "cost_estimate_enabled": False,
                "allows_virtual_table_explore": True,
                # Cache sensor data queries for 10 minutes
                "cache_timeout": 600,
            }
        ),
        "impersonate_user": False,
        "is_managed_externally": False,
    }

    # Check if already registered
    existing = client.get("/api/v1/database/", params={"q": json.dumps({"filters": [{"col": "database_name", "opr": "eq", "val": "VictoriaMetrics (sensor metrics)"}]})})
    existing.raise_for_status()
    results = existing.json().get("result", [])
    if results:
        db_id = results[0]["id"]
        print(f"  VictoriaMetrics already registered (id={db_id}), updating...")
        resp = client.put(f"/api/v1/database/{db_id}", json=payload)
        resp.raise_for_status()
        return db_id

    resp = client.post("/api/v1/database/", json=payload)
    resp.raise_for_status()
    db_id: int = resp.json()["id"]
    print(f"  Registered VictoriaMetrics datasource (id={db_id})")
    return db_id


def register_postgres(client: SupersetClient, pg_url: str) -> int:
    """
    Register the maintenance application PostgreSQL database (read-only user)
    as a Superset datasource.

    Returns the Superset database ID.
    """
    payload = {
        "database_name": "Maintenance DB (alerts, work orders, assets)",
        "sqlalchemy_uri": pg_url,
        "expose_in_sqllab": True,
        "allow_run_async": True,
        "allow_ctas": False,
        "allow_cvas": False,
        "allow_dml": False,
        "extra": json.dumps(
            {
                "metadata_params": {},
                "engine_params": {"connect_args": {"application_name": "superset"}},
                "cost_estimate_enabled": True,
                "allows_virtual_table_explore": True,
                "cache_timeout": 300,
            }
        ),
        "impersonate_user": False,
        "is_managed_externally": False,
    }

    existing = client.get(
        "/api/v1/database/",
        params={"q": json.dumps({"filters": [{"col": "database_name", "opr": "eq", "val": "Maintenance DB (alerts, work orders, assets)"}]})},
    )
    existing.raise_for_status()
    results = existing.json().get("result", [])
    if results:
        db_id = results[0]["id"]
        print(f"  Maintenance DB already registered (id={db_id}), updating...")
        resp = client.put(f"/api/v1/database/{db_id}", json=payload)
        resp.raise_for_status()
        return db_id

    resp = client.post("/api/v1/database/", json=payload)
    resp.raise_for_status()
    db_id = resp.json()["id"]
    print(f"  Registered Maintenance DB datasource (id={db_id})")
    return db_id


# ---------------------------------------------------------------------------
# Virtual dataset registration
# ---------------------------------------------------------------------------

# Each dataset is a named SQL query that Superset treats as a virtual table.
# Analysts can build charts directly from these datasets without writing SQL.

VICTORIAMETRICS_DATASETS: list[dict[str, Any]] = [
    {
        "dataset_name": "sensor_rms_hourly",
        "sql": """
-- Hourly average RMS vibration per device
-- MetricsQL translated by sqlalchemy-victoriametrics dialect
SELECT
    device_id,
    timestamp,
    avg(rms_x) AS rms_x_avg,
    avg(rms_y) AS rms_y_avg,
    avg(rms_z) AS rms_z_avg
FROM rms_vibration
WHERE $__timeFilter(timestamp)
GROUP BY device_id, timestamp
ORDER BY timestamp DESC
""",
        "description": "Hourly average RMS vibration (X/Y/Z axes) per sensor device",
    },
    {
        "dataset_name": "sensor_health_daily",
        "sql": """
-- Daily sensor health summary per device
SELECT
    device_id,
    date_trunc('day', timestamp) AS date,
    max(kurtosis)               AS max_kurtosis,
    avg(temperature_celsius)    AS avg_temperature,
    count(CASE WHEN anomaly_score > 0.7 THEN 1 END) AS alert_count
FROM sensor_readings
WHERE $__timeFilter(timestamp)
GROUP BY device_id, date_trunc('day', timestamp)
ORDER BY date DESC, device_id
""",
        "description": "Daily sensor health summary: peak kurtosis, temperature, alert count",
    },
    {
        "dataset_name": "bearing_defect_frequencies",
        "sql": """
-- Bearing defect frequency magnitudes (BPFO, BPFI) per device per hour
SELECT
    device_id,
    timestamp,
    avg(bpfo_magnitude) AS bpfo_magnitude,
    avg(bpfi_magnitude) AS bpfi_magnitude,
    avg(bsf_magnitude)  AS bsf_magnitude,
    avg(ftf_magnitude)  AS ftf_magnitude
FROM bearing_frequencies
WHERE $__timeFilter(timestamp)
GROUP BY device_id, timestamp
ORDER BY timestamp DESC
""",
        "description": "Bearing defect frequency magnitudes (BPFO, BPFI, BSF, FTF) over time",
    },
]

POSTGRES_DATASETS: list[dict[str, Any]] = [
    {
        "dataset_name": "asset_oee",
        "sql": """
-- Overall Equipment Effectiveness per asset per day
-- Reads from the asset_oee_daily view (created by sql/sensor_views.sql)
SELECT
    asset_id,
    asset_name,
    production_line,
    site,
    date,
    availability_pct,
    performance_pct,
    quality_pct,
    oee_pct
FROM asset_oee_daily
WHERE date BETWEEN '{{ filter_values("start_date")[0] }}'
              AND '{{ filter_values("end_date")[0] }}'
ORDER BY date DESC, oee_pct ASC
""",
        "description": "OEE (availability × performance × quality) per asset per day",
    },
    {
        "dataset_name": "asset_mtbf_mttr",
        "sql": """
-- MTBF and MTTR per asset (rolling 90-day window)
SELECT
    a.asset_id,
    a.asset_name,
    a.asset_class,
    a.site,
    m.mtbf_hours,
    m.mttr_hours,
    m.failure_count,
    m.calculated_at
FROM assets a
JOIN asset_mtbf m USING (asset_id)
ORDER BY m.mtbf_hours ASC
""",
        "description": "MTBF and MTTR per asset (rolling 90-day window)",
    },
    {
        "dataset_name": "active_alerts",
        "sql": """
-- Active (unresolved) alerts with asset context
SELECT
    al.alert_id,
    al.created_at,
    a.asset_name,
    a.asset_class,
    a.production_line,
    a.site,
    al.fault_type,
    al.severity_stage,
    al.confidence_pct,
    al.status,
    al.assigned_to,
    EXTRACT(EPOCH FROM (NOW() - al.created_at)) / 3600 AS age_hours
FROM alerts al
JOIN assets a USING (asset_id)
WHERE al.status NOT IN ('resolved', 'closed')
ORDER BY al.severity_stage DESC, al.created_at ASC
""",
        "description": "Active (unresolved) alerts with asset context and age",
    },
    {
        "dataset_name": "work_order_status",
        "sql": """
-- Work order completion status
SELECT
    wo.work_order_id,
    wo.created_at,
    wo.due_date,
    a.asset_name,
    a.site,
    wo.priority,
    wo.status,
    wo.assigned_team,
    wo.labor_hours,
    wo.parts_cost_usd,
    (wo.labor_hours * 85.0 + COALESCE(wo.parts_cost_usd, 0)) AS total_cost_usd,
    CASE
        WHEN wo.status = 'completed' THEN 'Completed'
        WHEN wo.due_date < NOW() AND wo.status != 'completed' THEN 'Overdue'
        ELSE 'In Progress'
    END AS completion_status
FROM work_orders wo
JOIN assets a USING (asset_id)
WHERE wo.created_at >= NOW() - INTERVAL '90 days'
ORDER BY wo.due_date ASC
""",
        "description": "Work order status with cost and overdue classification (last 90 days)",
    },
    {
        "dataset_name": "maintenance_cost_monthly",
        "sql": """
-- Monthly maintenance cost per asset (from maintenance_cost_monthly view)
SELECT
    asset_id,
    asset_name,
    asset_class,
    site,
    month,
    labor_cost_usd,
    parts_cost_usd,
    total_cost_usd,
    work_order_count
FROM maintenance_cost_monthly
ORDER BY month DESC, total_cost_usd DESC
""",
        "description": "Monthly maintenance cost (labor + parts) per asset",
    },
    {
        "dataset_name": "fault_distribution",
        "sql": """
-- Fault type distribution across fleet (last 90 days)
SELECT
    fault_type,
    asset_class,
    site,
    alert_count,
    alert_pct,
    avg_confidence_pct,
    avg_severity_stage
FROM fault_distribution
ORDER BY alert_count DESC
""",
        "description": "Fault type distribution by asset class and site (last 90 days)",
    },
]


def create_virtual_dataset(
    client: SupersetClient,
    db_id: int,
    dataset: dict[str, Any],
) -> int:
    """Create a virtual dataset (SQL view) in Superset."""
    payload = {
        "database": db_id,
        "schema": "public",
        "table_name": dataset["dataset_name"],
        "sql": dataset["sql"].strip(),
        "description": dataset.get("description", ""),
        "is_managed_externally": False,
    }

    # Check for existing dataset
    existing = client.get(
        "/api/v1/dataset/",
        params={"q": json.dumps({"filters": [{"col": "table_name", "opr": "eq", "val": dataset["dataset_name"]}]})},
    )
    existing.raise_for_status()
    results = existing.json().get("result", [])
    if results:
        ds_id = results[0]["id"]
        print(f"    Dataset '{dataset['dataset_name']}' already exists (id={ds_id}), skipping")
        return ds_id

    resp = client.post("/api/v1/dataset/", json=payload)
    if resp.status_code == 422:
        print(f"    WARNING: Could not create dataset '{dataset['dataset_name']}': {resp.text}")
        return -1
    resp.raise_for_status()
    ds_id = resp.json()["id"]
    print(f"    Created dataset '{dataset['dataset_name']}' (id={ds_id})")
    return ds_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def wait_for_superset(url: str, max_attempts: int = 30) -> None:
    """Poll Superset /health until it responds."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"  Superset is ready (attempt {attempt})")
                return
        except requests.RequestException:
            pass
        print(f"  Waiting for Superset ({attempt}/{max_attempts})...")
        time.sleep(5)
    print("ERROR: Superset did not become ready in time", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Superset datasources and datasets")
    parser.add_argument("--superset-url", default="http://localhost:8088")
    parser.add_argument("--superset-user", default="admin")
    parser.add_argument("--superset-password", default="admin")
    parser.add_argument("--vm-url", default="http://victoriametrics:8428")
    parser.add_argument("--pg-url", default="postgresql+psycopg2://app_ro:app_ro@postgres:5432/maintenance")
    args = parser.parse_args()

    print(f"Connecting to Superset at {args.superset_url}...")
    wait_for_superset(args.superset_url)

    client = SupersetClient(args.superset_url)
    client.login(args.superset_user, args.superset_password)
    print("  Logged in successfully")

    print("\nRegistering datasources...")
    vm_db_id = register_victoriametrics(client, args.vm_url)
    pg_db_id = register_postgres(client, args.pg_url)

    print("\nCreating VictoriaMetrics virtual datasets...")
    for ds in VICTORIAMETRICS_DATASETS:
        create_virtual_dataset(client, vm_db_id, ds)

    print("\nCreating PostgreSQL virtual datasets...")
    for ds in POSTGRES_DATASETS:
        create_virtual_dataset(client, pg_db_id, ds)

    print("\nDatasource registration complete.")
    print(f"  Superset UI: {args.superset_url}")


if __name__ == "__main__":
    main()
