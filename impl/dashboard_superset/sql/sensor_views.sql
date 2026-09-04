-- =============================================================================
-- Superset virtual datasets: PostgreSQL views for the predictive maintenance
-- analytics platform.
--
-- Assumes the following tables exist in the `maintenance` database:
--   assets          (asset_id, asset_name, asset_class, production_line,
--                    site, commissioned_at, rated_runtime_hours)
--   alerts          (alert_id, asset_id, created_at, resolved_at, fault_type,
--                    severity_stage, confidence_pct, status, assigned_to)
--   work_orders     (work_order_id, asset_id, alert_id, created_at, due_date,
--                    completed_at, status, assigned_team, labor_hours,
--                    parts_cost_usd, priority)
--   downtime_events (event_id, asset_id, started_at, ended_at, cause,
--                    planned BOOLEAN)
--   planned_production_minutes (asset_id, date, planned_minutes)
--
-- Run once during database initialisation:
--   psql -U app_admin -d maintenance -f sensor_views.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Asset health score (daily)
--    Derives a 0-100 health score from the anomaly alert history.
--    Score = 100 – penalty, where penalty accumulates from unresolved alerts
--    weighted by severity stage.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW asset_health_daily AS
WITH
-- Days in the analysis window
date_series AS (
    SELECT generate_series(
        CURRENT_DATE - INTERVAL '365 days',
        CURRENT_DATE,
        INTERVAL '1 day'
    )::date AS date
),
-- Active alerts on each day per asset
daily_alert_summary AS (
    SELECT
        a.asset_id,
        ds.date,
        COUNT(al.alert_id)                                             AS active_alert_count,
        COALESCE(MAX(al.severity_stage), 0)                           AS max_severity_stage,
        COALESCE(AVG(al.confidence_pct), 0)                           AS avg_confidence,
        -- Weighted penalty: stage 1=5pts, 2=15pts, 3=35pts, 4=60pts
        COALESCE(SUM(
            CASE al.severity_stage
                WHEN 1 THEN 5
                WHEN 2 THEN 15
                WHEN 3 THEN 35
                WHEN 4 THEN 60
                ELSE 0
            END * (al.confidence_pct / 100.0)
        ), 0)                                                          AS health_penalty
    FROM assets a
    CROSS JOIN date_series ds
    LEFT JOIN alerts al ON al.asset_id = a.asset_id
        AND al.created_at::date <= ds.date
        AND (al.resolved_at IS NULL OR al.resolved_at::date >= ds.date)
    GROUP BY a.asset_id, ds.date
)
SELECT
    das.asset_id,
    a.asset_name,
    a.asset_class,
    a.production_line,
    a.site,
    das.date,
    das.active_alert_count,
    das.max_severity_stage,
    das.avg_confidence,
    GREATEST(0, LEAST(100, ROUND(100 - das.health_penalty))) AS health_score
FROM daily_alert_summary das
JOIN assets a USING (asset_id)
ORDER BY das.date DESC, health_score ASC;


-- -----------------------------------------------------------------------------
-- 2. OEE (Overall Equipment Effectiveness) — daily per asset
--    OEE = Availability × Performance × Quality
--    This view uses downtime_events and planned_production_minutes.
--    Quality defaults to 98% unless a quality_events table is provided.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW asset_oee_daily AS
WITH

-- Total planned minutes per asset per day
planned AS (
    SELECT
        asset_id,
        date,
        GREATEST(planned_minutes, 1) AS planned_minutes
    FROM planned_production_minutes
),

-- Unplanned (corrective) downtime minutes per asset per day
unplanned_downtime AS (
    SELECT
        asset_id,
        started_at::date                                     AS date,
        SUM(
            EXTRACT(EPOCH FROM (
                LEAST(ended_at, (started_at::date + INTERVAL '1 day'))
                - GREATEST(started_at, started_at::date::timestamp)
            )) / 60.0
        )                                                    AS downtime_minutes
    FROM downtime_events
    WHERE planned = FALSE
      AND ended_at IS NOT NULL
    GROUP BY asset_id, started_at::date
),

-- Performance: proxy from average speed ratio stored in asset runtime table.
-- Defaulting to 95% where telemetry not available.
performance_proxy AS (
    SELECT
        asset_id,
        CURRENT_DATE AS date,
        0.95          AS performance_ratio
    FROM assets
),

oee_base AS (
    SELECT
        p.asset_id,
        p.date,
        p.planned_minutes,
        COALESCE(ud.downtime_minutes, 0)                                  AS downtime_minutes,
        -- Availability = (planned - unplanned downtime) / planned
        GREATEST(0, (p.planned_minutes - COALESCE(ud.downtime_minutes, 0)))
            / p.planned_minutes                                           AS availability,
        -- Performance proxy (ideally from production counter telemetry)
        COALESCE(pp.performance_ratio, 0.95)                              AS performance,
        -- Quality proxy: assume 98% first-pass yield unless quality_events table present
        0.98                                                              AS quality
    FROM planned p
    LEFT JOIN unplanned_downtime ud
        ON ud.asset_id = p.asset_id AND ud.date = p.date
    LEFT JOIN performance_proxy pp
        ON pp.asset_id = p.asset_id
)

SELECT
    ob.asset_id,
    a.asset_name,
    a.production_line,
    a.site,
    ob.date,
    ROUND((ob.availability  * 100)::numeric, 2) AS availability_pct,
    ROUND((ob.performance   * 100)::numeric, 2) AS performance_pct,
    ROUND((ob.quality       * 100)::numeric, 2) AS quality_pct,
    ROUND((ob.availability * ob.performance * ob.quality * 100)::numeric, 2) AS oee_pct,
    ob.planned_minutes,
    ob.downtime_minutes
FROM oee_base ob
JOIN assets a USING (asset_id)
ORDER BY ob.date DESC, ob.asset_id;


-- -----------------------------------------------------------------------------
-- 3. MTBF (Mean Time Between Failures) per asset
--    Rolling 90-day window.  A "failure" = an alert at severity stage >= 3
--    that required a corrective work order.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW asset_mtbf AS
WITH

failures AS (
    SELECT
        wo.asset_id,
        wo.created_at                                  AS failure_time,
        ROW_NUMBER() OVER (
            PARTITION BY wo.asset_id
            ORDER BY wo.created_at
        )                                              AS failure_seq
    FROM work_orders wo
    JOIN alerts al ON al.alert_id = wo.alert_id
    WHERE al.severity_stage >= 3
      AND wo.created_at >= NOW() - INTERVAL '90 days'
),

-- Time between consecutive failures per asset
intervals AS (
    SELECT
        f1.asset_id,
        EXTRACT(EPOCH FROM (f2.failure_time - f1.failure_time)) / 3600.0 AS hours_between
    FROM failures f1
    JOIN failures f2
        ON f2.asset_id = f1.asset_id
        AND f2.failure_seq = f1.failure_seq + 1
)

SELECT
    a.asset_id,
    a.asset_name,
    a.asset_class,
    a.site,
    COUNT(i.hours_between)                            AS failure_count,
    ROUND(AVG(i.hours_between)::numeric, 1)           AS mtbf_hours,
    NOW()                                             AS calculated_at
FROM assets a
LEFT JOIN intervals i USING (asset_id)
GROUP BY a.asset_id, a.asset_name, a.asset_class, a.site
ORDER BY mtbf_hours ASC NULLS LAST;


-- -----------------------------------------------------------------------------
-- 4. MTTR (Mean Time To Repair) per asset
--    Measured from work order creation to completion, rolling 90-day window.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW asset_mttr AS
WITH

repair_times AS (
    SELECT
        wo.asset_id,
        wo.work_order_id,
        wo.created_at,
        wo.completed_at,
        EXTRACT(EPOCH FROM (wo.completed_at - wo.created_at)) / 3600.0 AS repair_hours
    FROM work_orders wo
    WHERE wo.status = 'completed'
      AND wo.completed_at IS NOT NULL
      AND wo.created_at >= NOW() - INTERVAL '90 days'
      -- Exclude outliers (> 30 days repair time — likely data quality issues)
      AND wo.completed_at - wo.created_at < INTERVAL '30 days'
)

SELECT
    a.asset_id,
    a.asset_name,
    a.asset_class,
    a.site,
    COUNT(rt.work_order_id)                           AS repair_count,
    ROUND(AVG(rt.repair_hours)::numeric, 2)           AS mttr_hours,
    ROUND(PERCENTILE_CONT(0.5)
          WITHIN GROUP (ORDER BY rt.repair_hours)
          ::numeric, 2)                               AS mttr_median_hours,
    ROUND(PERCENTILE_CONT(0.9)
          WITHIN GROUP (ORDER BY rt.repair_hours)
          ::numeric, 2)                               AS mttr_p90_hours,
    NOW()                                             AS calculated_at
FROM assets a
LEFT JOIN repair_times rt USING (asset_id)
GROUP BY a.asset_id, a.asset_name, a.asset_class, a.site
ORDER BY mttr_hours DESC NULLS LAST;


-- -----------------------------------------------------------------------------
-- 5. Maintenance cost per asset per month
--    Combines labour costs (labor_hours × hourly rate) and spare parts costs
--    from work orders.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW maintenance_cost_monthly AS
WITH

-- Hourly labour rate by team (configurable — stored in a config table or hardcoded)
labour_rates (assigned_team, hourly_rate_usd) AS (
    VALUES
        ('Mechanical',  85.00),
        ('Electrical',  95.00),
        ('Instrumentation', 105.00),
        ('Contract',   125.00),
        ('Unknown',     85.00)
),

monthly_costs AS (
    SELECT
        wo.asset_id,
        DATE_TRUNC('month', wo.created_at)::date      AS month,
        wo.assigned_team,
        COUNT(wo.work_order_id)                       AS work_order_count,
        SUM(
            COALESCE(wo.labor_hours, 0) *
            COALESCE(lr.hourly_rate_usd, 85.0)
        )                                             AS labor_cost_usd,
        SUM(COALESCE(wo.parts_cost_usd, 0))           AS parts_cost_usd
    FROM work_orders wo
    LEFT JOIN labour_rates lr ON lr.assigned_team = wo.assigned_team
    GROUP BY wo.asset_id, DATE_TRUNC('month', wo.created_at), wo.assigned_team
)

SELECT
    mc.asset_id,
    a.asset_name,
    a.asset_class,
    a.site,
    mc.month,
    mc.assigned_team,
    mc.work_order_count,
    ROUND(mc.labor_cost_usd::numeric, 2)              AS labor_cost_usd,
    ROUND(mc.parts_cost_usd::numeric, 2)              AS parts_cost_usd,
    ROUND((mc.labor_cost_usd + mc.parts_cost_usd)::numeric, 2) AS total_cost_usd
FROM monthly_costs mc
JOIN assets a USING (asset_id)
ORDER BY mc.month DESC, total_cost_usd DESC;


-- -----------------------------------------------------------------------------
-- 6. Fault distribution by type (last 90 days)
--    Used in the donut chart showing which faults dominate the fleet.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW fault_distribution AS
WITH

alert_counts AS (
    SELECT
        al.fault_type,
        a.asset_class,
        a.site,
        COUNT(al.alert_id)          AS alert_count,
        AVG(al.confidence_pct)      AS avg_confidence_pct,
        AVG(al.severity_stage)      AS avg_severity_stage
    FROM alerts al
    JOIN assets a USING (asset_id)
    WHERE al.created_at >= NOW() - INTERVAL '90 days'
    GROUP BY al.fault_type, a.asset_class, a.site
),

totals AS (
    SELECT SUM(alert_count) AS total_alerts
    FROM alert_counts
)

SELECT
    ac.fault_type,
    ac.asset_class,
    ac.site,
    ac.alert_count,
    ROUND((ac.alert_count::numeric / NULLIF(t.total_alerts, 0) * 100), 2) AS alert_pct,
    ROUND(ac.avg_confidence_pct::numeric, 1)       AS avg_confidence_pct,
    ROUND(ac.avg_severity_stage::numeric, 2)       AS avg_severity_stage
FROM alert_counts ac
CROSS JOIN totals t
ORDER BY ac.alert_count DESC;


-- =============================================================================
-- Indexes to support common Superset query patterns
-- =============================================================================

-- These are on the underlying tables, not the views.

-- Alert queries by status and date
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_status_created
    ON alerts (status, created_at DESC);

-- Alert queries by asset
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_asset_id
    ON alerts (asset_id, created_at DESC);

-- Work order queries by status, date
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_wo_status_created
    ON work_orders (status, created_at DESC);

-- Downtime queries by asset and date
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_downtime_asset_date
    ON downtime_events (asset_id, started_at DESC)
    WHERE ended_at IS NOT NULL;

-- Planned production lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_planned_prod_asset_date
    ON planned_production_minutes (asset_id, date DESC);
