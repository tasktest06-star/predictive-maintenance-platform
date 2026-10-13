-- CMMS and asset-registry schema.
--
-- Time-series data does NOT live here -- it is in VictoriaMetrics. Postgres
-- holds the things that need transactions, foreign keys and an audit trail:
-- assets, their configuration, alarms, work orders and the human decisions
-- taken against them. Putting 25 kHz waveforms in Postgres is the classic
-- first mistake of an IIoT platform and it is unrecoverable by the time the
-- table is 2 TB.

BEGIN;

CREATE TABLE site (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    timezone    TEXT NOT NULL DEFAULT 'UTC',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE asset (
    id              SERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES site(id) ON DELETE RESTRICT,
    tag             TEXT NOT NULL,                   -- the plant's own tag, e.g. P-1203A
    name            TEXT NOT NULL,
    machine_class   TEXT NOT NULL,                   -- pump, fan, gearbox, compressor
    power_kw        NUMERIC(10,2),
    nameplate_rpm   NUMERIC(10,2),
    -- Mounting stiffness decides the ISO zone boundaries and shifts every
    -- threshold by ~1.6x if recorded wrongly. It is a commissioning decision,
    -- reviewed by a human, never inferred at runtime.
    flexible_support BOOLEAN NOT NULL DEFAULT FALSE,
    iso_class       TEXT NOT NULL DEFAULT 'group2_rigid',
    criticality     SMALLINT NOT NULL DEFAULT 3 CHECK (criticality BETWEEN 1 AND 5),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (site_id, tag)
);

CREATE TABLE bearing (
    id                  SERIAL PRIMARY KEY,
    asset_id            INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    position            TEXT NOT NULL,              -- drive_end, non_drive_end, ...
    designation         TEXT NOT NULL,              -- SKF 6205-2RS
    n_balls             SMALLINT NOT NULL CHECK (n_balls >= 3),
    ball_d              NUMERIC(10,5) NOT NULL,
    pitch_d             NUMERIC(10,5) NOT NULL,
    contact_angle_deg   NUMERIC(6,3) NOT NULL DEFAULT 0,
    installed_at        DATE,
    CHECK (ball_d > 0 AND pitch_d > ball_d)
);

CREATE TABLE device (
    id                   SERIAL PRIMARY KEY,
    device_uid           TEXT NOT NULL UNIQUE,       -- MCU unique ID, hex
    asset_id             INTEGER REFERENCES asset(id) ON DELETE SET NULL,
    hardware_kit         TEXT NOT NULL,              -- stwinbx1, esp32s3, nrf9160, ...
    firmware_version     TEXT,
    feature_spec_version SMALLINT NOT NULL,
    mount_type           TEXT,                       -- stud, adhesive, magnet
    -- Mount type is not cosmetic: a magnet mount rolls off above ~2 kHz, which
    -- removes most of the bearing band. Recorded so a surprising spectrum can
    -- be explained before anyone opens a machine.
    sample_rate_hz       INTEGER NOT NULL,
    commissioned_at      TIMESTAMPTZ,
    last_seen_at         TIMESTAMPTZ,
    battery_mv           INTEGER,
    CHECK (sample_rate_hz > 0)
);

CREATE INDEX device_asset_idx ON device(asset_id);
CREATE INDEX device_last_seen_idx ON device(last_seen_at);

-- Model registry. Which model scored which frame must be recoverable years
-- later, when a disputed work order is reviewed.
CREATE TABLE model_version (
    id                SERIAL PRIMARY KEY,
    name              TEXT NOT NULL,
    kind              TEXT NOT NULL,                -- mahalanobis, deep_svdd, ...
    sha256            CHAR(64) NOT NULL UNIQUE,
    feature_spec_version SMALLINT NOT NULL,
    feature_config    JSONB NOT NULL,
    trained_at        TIMESTAMPTZ NOT NULL,
    train_n_samples   INTEGER,
    split_protocol    TEXT NOT NULL,                -- must name the split; see docs/06
    metrics           JSONB NOT NULL,
    approved_by       TEXT,
    approved_at       TIMESTAMPTZ
);

CREATE TABLE alarm_event (
    id            BIGSERIAL PRIMARY KEY,
    device_id     INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    occurred_at   TIMESTAMPTZ NOT NULL,
    from_state    TEXT NOT NULL,
    to_state      TEXT NOT NULL,
    health_index  NUMERIC(10,4),
    reason        TEXT NOT NULL,
    evidence      JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_version_id INTEGER REFERENCES model_version(id),
    CHECK (from_state <> to_state)
);

CREATE INDEX alarm_event_device_time_idx ON alarm_event(device_id, occurred_at DESC);

CREATE TABLE work_order (
    id              BIGSERIAL PRIMARY KEY,
    asset_id        INTEGER NOT NULL REFERENCES asset(id) ON DELETE RESTRICT,
    alarm_event_id  BIGINT REFERENCES alarm_event(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    fault           TEXT NOT NULL,
    severity        TEXT NOT NULL,
    confidence      NUMERIC(5,4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    rul_median_days NUMERIC(10,2),
    rul_p10_days    NUMERIC(10,2),
    rul_p90_days    NUMERIC(10,2),
    recommended_action TEXT NOT NULL,
    evidence        JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open','acknowledged','in_progress','closed','rejected')),
    assigned_to     TEXT,
    closed_at       TIMESTAMPTZ,
    -- The feedback loop. Without a recorded outcome there is no way to measure
    -- precision in production, and a PdM programme that cannot state its own
    -- false-positive rate will eventually be switched off.
    outcome         TEXT CHECK (outcome IN ('confirmed','not_confirmed','inconclusive')),
    outcome_notes   TEXT,
    parts_replaced  TEXT[]
);

CREATE INDEX work_order_asset_status_idx ON work_order(asset_id, status);
CREATE INDEX work_order_open_idx ON work_order(asset_id, fault) WHERE status = 'open';

-- Raw-waveform snapshots live in object storage; this table is the index.
CREATE TABLE snapshot (
    id            BIGSERIAL PRIMARY KEY,
    device_id     INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    captured_at   TIMESTAMPTZ NOT NULL,
    reason        TEXT NOT NULL,                    -- alarm, scheduled, on_demand
    sample_rate_hz INTEGER NOT NULL,
    n_samples     INTEGER NOT NULL,
    object_key    TEXT NOT NULL UNIQUE,
    sha256        CHAR(64) NOT NULL,
    bytes         INTEGER NOT NULL
);

CREATE INDEX snapshot_device_time_idx ON snapshot(device_id, captured_at DESC);

-- Precision/recall of the programme itself, straight out of recorded outcomes.
-- Exposed as a view so it cannot drift from a hand-maintained report.
CREATE VIEW programme_performance AS
SELECT
    a.site_id,
    wo.fault,
    count(*)                                                        AS orders,
    count(*) FILTER (WHERE wo.outcome = 'confirmed')                AS confirmed,
    count(*) FILTER (WHERE wo.outcome = 'not_confirmed')            AS false_positive,
    count(*) FILTER (WHERE wo.outcome IS NULL AND wo.status <> 'closed') AS pending,
    round(
        count(*) FILTER (WHERE wo.outcome = 'confirmed')::numeric
        / NULLIF(count(*) FILTER (WHERE wo.outcome IN ('confirmed','not_confirmed')), 0),
        3
    ) AS precision
FROM work_order wo
JOIN asset a ON a.id = wo.asset_id
GROUP BY a.site_id, wo.fault;

COMMIT;
