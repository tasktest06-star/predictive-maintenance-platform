"""
Superset configuration for the predictive maintenance analytics platform.

This file is mounted into the Superset container at:
  /app/pythonpath/superset_config.py

Environment variables (set in docker-compose.yml / Kubernetes secrets):
  DATABASE_URL   - PostgreSQL connection string for Superset metadata
  REDIS_URL      - Redis connection string for caching
  SUPERSET_SECRET_KEY - 32+ char random string for session signing
"""

import os
from celery.schedules import crontab
from typing import Any

# ---------------------------------------------------------------------------
# Core security
# ---------------------------------------------------------------------------

SECRET_KEY = os.environ["SUPERSET_SECRET_KEY"]

WTF_CSRF_ENABLED = True
WTF_CSRF_EXEMPT_LIST: list[str] = []
WTF_CSRF_TIME_LIMIT = 60 * 60 * 24 * 365  # 1 year

SESSION_COOKIE_SECURE = True          # HTTPS only in production
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# ---------------------------------------------------------------------------
# Database (Superset metadata)
# ---------------------------------------------------------------------------

SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]

# Pool sizing for production (PostgreSQL RDS)
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_size": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800,
    "max_overflow": 20,
}

# ---------------------------------------------------------------------------
# Cache — Redis backend
# ---------------------------------------------------------------------------

_redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")

CACHE_CONFIG: dict[str, Any] = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 300,          # 5 minutes default TTL
    "CACHE_KEY_PREFIX": "superset_",
    "CACHE_REDIS_URL": _redis_url,
}

# Per-datasource query result cache (sensor data can be cached longer)
DATA_CACHE_CONFIG: dict[str, Any] = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 600,          # 10 minutes for sensor queries
    "CACHE_KEY_PREFIX": "superset_data_",
    "CACHE_REDIS_URL": _redis_url,
}

# Dashboard filter state cache
FILTER_STATE_CACHE_CONFIG: dict[str, Any] = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 86400,        # 24 hours
    "CACHE_KEY_PREFIX": "superset_filter_",
    "CACHE_REDIS_URL": _redis_url,
}

# Explore form state cache
EXPLORE_FORM_DATA_CACHE_CONFIG: dict[str, Any] = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 86400,
    "CACHE_KEY_PREFIX": "superset_explore_",
    "CACHE_REDIS_URL": _redis_url,
}

# ---------------------------------------------------------------------------
# Celery (async reports, scheduled alerts, thumbnail generation)
# ---------------------------------------------------------------------------

_broker_url = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/1")
_result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/2")


class CeleryConfig:
    broker_url = _broker_url
    imports = (
        "superset.sql_lab",
        "superset.tasks.scheduler",
        "superset.tasks.thumbnails",
        "superset.tasks.cache",
    )
    result_backend = _result_backend
    worker_prefetch_multiplier = 10
    task_acks_late = True
    task_annotations = {
        "sql_lab.get_sql_results": {"rate_limit": "100/s"},
        "email_reports.send": {"rate_limit": "1/s", "time_limit": 120, "soft_time_limit": 150},
    }
    beat_schedule = {
        # Warm the chart thumbnail cache every 30 minutes
        "cache-warmup-hourly": {
            "task": "cache-warmup",
            "schedule": crontab(minute="*/30"),
            "kwargs": {"strategy": "top_n_dashboards", "top_n": 10},
        },
        # Run scheduled reports every 5 minutes
        "reports.scheduler": {
            "task": "reports.scheduler",
            "schedule": crontab(minute="*/5"),
        },
        # Prune old log entries daily at 02:00
        "reports.prune_log": {
            "task": "reports.prune_log",
            "schedule": crontab(minute=0, hour=2),
        },
    }


CELERY_CONFIG = CeleryConfig

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

FEATURE_FLAGS: dict[str, bool] = {
    # Role-based access control on individual dashboards
    "DASHBOARD_RBAC": True,
    # Scheduled reports and SQL-based threshold alerts
    "ALERT_REPORTS": True,
    # Dashboard thumbnail preview images
    "THUMBNAILS": True,
    # Native filter bar (replaces legacy filter box)
    "DASHBOARD_NATIVE_FILTERS": True,
    "DASHBOARD_CROSS_FILTERS": True,
    # Cache warmup API endpoint
    "CACHE_IMPERSONATION": False,
    # Enables row-level security on datasets
    "ROW_LEVEL_SECURITY": True,
    # Allow uploading CSV/Excel to create datasets (disable in prod if not needed)
    "ALLOW_FULL_CSV_EXPORT": True,
    # Drill-to-detail modal
    "DRILL_TO_DETAIL": True,
    # Async query execution (requires Celery)
    "GLOBAL_ASYNC_QUERIES": True,
}

# ---------------------------------------------------------------------------
# Security headers (Talisman)
# ---------------------------------------------------------------------------

TALISMAN_ENABLED = True
TALISMAN_CONFIG: dict[str, Any] = {
    "force_https": False,           # Set True in production behind TLS termination
    "strict_transport_security": True,
    "strict_transport_security_max_age": 31536000,
    "strict_transport_security_include_subdomains": True,
    "content_security_policy": {
        "default-src": ["'self'"],
        "img-src": [
            "'self'",
            "data:",
            "blob:",
            "*.tile.openstreetmap.org",   # Map tiles for fleet floor plan
        ],
        "worker-src": ["'self'", "blob:"],
        "connect-src": [
            "'self'",
            "http://victoriametrics:8428",  # Internal VM access
        ],
        "object-src": ["'none'"],
        "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "font-src": ["'self'", "data:"],
        "frame-ancestors": ["'none'"],
    },
    "referrer_policy": "strict-origin-when-cross-origin",
    "x_content_type_options": True,
    "x_xss_protection": True,
}

# ---------------------------------------------------------------------------
# Web server performance
# ---------------------------------------------------------------------------

# Increase for slow VictoriaMetrics MetricsQL queries over large time windows
SUPERSET_WEBSERVER_TIMEOUT = 120

# Max rows returned in SQL Lab (prevent analyst from pulling all sensor data)
SQL_MAX_ROW = 100_000
SAMPLES_ROW_LIMIT = 1_000

# ---------------------------------------------------------------------------
# Thumbnail / screenshot (for scheduled email reports)
# ---------------------------------------------------------------------------

SCREENSHOT_LOCATE_WAIT = 100
SCREENSHOT_LOAD_WAIT = 600
WEBDRIVER_TYPE = "chrome"
WEBDRIVER_OPTION_ARGS = [
    "--force-device-scale-factor=2.0",  # Retina quality
    "--high-dpi-support=2.0",
    "--headless",
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
]

# ---------------------------------------------------------------------------
# Email (for scheduled reports and alert notifications)
# ---------------------------------------------------------------------------

SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_STARTTLS = True
SMTP_SSL = False
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_MAIL_FROM = os.environ.get("SMTP_MAIL_FROM", "superset@example.com")

EMAIL_REPORTS_SUBJECT_PREFIX = "[Maintenance Platform] "

# ---------------------------------------------------------------------------
# Custom color palette — industrial health status theme
#
# Color semantics:
#   Healthy / good      → greens
#   Warning / elevated  → ambers
#   Critical / fault    → reds
#   Neutral / info      → blues and greys
# ---------------------------------------------------------------------------

# Superset extra-config color scheme (appears in chart color-scheme selector)
EXTRA_CATEGORICAL_COLOR_SCHEMES: list[dict[str, Any]] = [
    {
        "id": "industrial_health",
        "description": "Industrial health status palette",
        "label": "Industrial Health",
        "isDefault": True,
        "colors": [
            "#2ECC71",  # healthy green
            "#F39C12",  # warning amber
            "#E74C3C",  # critical red
            "#3498DB",  # info blue
            "#9B59B6",  # purple (secondary metric)
            "#1ABC9C",  # teal (good secondary)
            "#E67E22",  # orange (degraded)
            "#95A5A6",  # grey (offline / no data)
            "#F1C40F",  # yellow (elevated)
            "#C0392B",  # dark red (severe fault)
        ],
    },
    {
        "id": "maintenance_status",
        "description": "Work order and maintenance status palette",
        "label": "Maintenance Status",
        "isDefault": False,
        "colors": [
            "#27AE60",  # completed (green)
            "#2980B9",  # in-progress (blue)
            "#E74C3C",  # overdue (red)
            "#F39C12",  # pending (amber)
            "#7F8C8D",  # cancelled (grey)
        ],
    },
]

# Sequential scale for health score heatmaps (0=red → 100=green)
EXTRA_SEQUENTIAL_COLOR_SCHEMES: list[dict[str, Any]] = [
    {
        "id": "health_score",
        "description": "Health score 0-100 (red → amber → green)",
        "label": "Health Score",
        "isDiverging": False,
        "colors": [
            "#C0392B",  # 0  — severe fault
            "#E74C3C",  # 20 — critical
            "#E67E22",  # 40 — degraded
            "#F39C12",  # 60 — warning
            "#F1C40F",  # 70 — elevated
            "#2ECC71",  # 85 — good
            "#27AE60",  # 100 — healthy
        ],
    },
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

import logging

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
ENABLE_TIME_ROTATE = True
TIME_ROTATE_LOG_LEVEL = logging.INFO
FILENAME = "/app/superset_home/superset.log"
ROLLOVER = "midnight"
INTERVAL = 1
BACKUP_COUNT = 30

# ---------------------------------------------------------------------------
# SQL Lab
# ---------------------------------------------------------------------------

# Allow analysts to run long-running queries asynchronously
SQLLAB_ASYNC_TIME_LIMIT_SEC = 300
SQLLAB_TIMEOUT = 300

# Prevent exfiltration of large datasets
SQL_MAX_ROW = 100_000

# ---------------------------------------------------------------------------
# Mapbox (optional — for fleet floor plan / geo charts)
# ---------------------------------------------------------------------------

MAPBOX_API_KEY = os.environ.get("MAPBOX_API_KEY", "")
