-- Migration 0015: Platform Audit Log
--
-- Normalizes the gateway audit tables into a migration-managed schema.
-- This replaces the ad-hoc CREATE TABLE IF NOT EXISTS in:
--   - gateway/gateway/audit.py (audit_logs table)
--   - gateway/sidecar/main.py (gateway_audit_log table)
--
-- Both gateway components still create their tables on startup (for
-- backwards compatibility), but this migration ensures the schema exists
-- before first request and adds the unified platform_audit_log table
-- for critical event tracking.
--
-- See: docs/AUDIT_AND_INCIDENT_RECONSTRUCTION.md

-- ============================================================
-- 1. Gateway request audit (mirrors gateway/gateway/audit.py)
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    method VARCHAR(10) NOT NULL,
    path TEXT NOT NULL,
    query_string TEXT,
    user_id VARCHAR(128),
    project_id VARCHAR(128),
    email VARCHAR(256),
    client_ip VARCHAR(45),
    user_agent TEXT,
    status_code INTEGER,
    duration_ms FLOAT,
    response_size INTEGER,
    error TEXT,
    extra JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp
    ON audit_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id
    ON audit_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_project_id
    ON audit_logs (project_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_path
    ON audit_logs (path);

-- ============================================================
-- 2. Gateway sidecar audit (mirrors gateway/sidecar/main.py)
-- ============================================================
CREATE TABLE IF NOT EXISTS gateway_audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    client_ip TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,
    resource_id TEXT,
    bytes_transferred BIGINT,
    response_status INTEGER,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_gateway_audit_log_timestamp
    ON gateway_audit_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_gateway_audit_log_user_timestamp
    ON gateway_audit_log (user_id, timestamp DESC);

-- ============================================================
-- 3. Unified platform audit log (new — critical events)
-- ============================================================
-- This table provides crash-safe audit for security-critical events.
-- See docs/AUDIT_AND_INCIDENT_RECONSTRUCTION.md for event taxonomy.

CREATE TABLE IF NOT EXISTS platform_audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_time      TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_class     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'info',
    actor_id        TEXT,
    actor_type      TEXT,
    project_id      TEXT,
    resource_type   TEXT,
    resource_id     TEXT,
    action          TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    detail          JSONB,
    client_ip       INET,
    request_id      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_audit_event_time
    ON platform_audit_log (event_time);
CREATE INDEX IF NOT EXISTS idx_platform_audit_event_class
    ON platform_audit_log (event_class);
CREATE INDEX IF NOT EXISTS idx_platform_audit_actor_id
    ON platform_audit_log (actor_id);
CREATE INDEX IF NOT EXISTS idx_platform_audit_resource_id
    ON platform_audit_log (resource_id);
CREATE INDEX IF NOT EXISTS idx_platform_audit_request_id
    ON platform_audit_log (request_id);

-- Prevent accidental mutation of audit records
COMMENT ON TABLE platform_audit_log IS
    'Immutable audit log for security-critical events. Do not UPDATE or DELETE rows.';
