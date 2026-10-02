-- 0117: hub endpoint registry (ADR-060 U-1).
--
-- One UMRS hub: the endpoint registry moves from the four per-cluster
-- front_* databases into the shared umrs_db, venue-scoped like jobs
-- (0088). Each umrs-front server (UMRS_ENDPOINT_REGISTRY_SHARED=1) stamps
-- its site.yaml name on every row and reads only its own venue — one
-- table, per-site visibility; the row shape otherwise mirrors the
-- auto-DDL the PostgresBackend already maintained (positional contract
-- ADR-058). Per-cluster front_* tables are left untouched (retired with
-- their stacks in a later hub wave); live endpoints re-register through
-- the boot reconciler after each server's restart.

CREATE TABLE IF NOT EXISTS endpoint_registry (
    job_id               VARCHAR(255) PRIMARY KEY,
    project_id           VARCHAR(255) NOT NULL,
    model                VARCHAR(255) NOT NULL,
    host                 VARCHAR(255) NOT NULL,
    port                 INTEGER NOT NULL,
    status               VARCHAR(50) NOT NULL,
    backend              VARCHAR(50) NOT NULL,
    variant              VARCHAR(255),
    created_at           TIMESTAMP NOT NULL,
    ready_at             TIMESTAMP,
    last_used_at         TIMESTAMP,
    expires_at           TIMESTAMP,
    error_message        TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    metadata             JSONB DEFAULT '{}',
    backend_job_id       VARCHAR(255),
    venue                VARCHAR(32) NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_registry_project ON endpoint_registry(project_id);
CREATE INDEX IF NOT EXISTS idx_registry_status ON endpoint_registry(status);
CREATE INDEX IF NOT EXISTS idx_registry_venue ON endpoint_registry(venue);

COMMENT ON TABLE endpoint_registry IS
    'ADR-060 U-1: one hub endpoint registry in the shared umrs_db; venue column scopes each row to its site (jobs-style claim scoping, 0088). Per-cluster front_* copies are retired with their stacks.';
