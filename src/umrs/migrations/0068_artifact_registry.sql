-- 0068: artifact registry tables (boot-applied via migrations runner).
--
-- The artifact_registry module creates its schema only via ensure_schema(),
-- which running servers may never invoke on a fresh DB — GET /umrs/artifacts
-- then 500s with `relation "artifacts" does not exist`. Mirror the module's
-- canonical DDL (artifact_registry/__init__.py, version 1) so the table set
-- exists wherever migrations run. Idempotent; no data change.

CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type TEXT NOT NULL,
    name TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    project_id TEXT,
    uri TEXT NOT NULL,
    bucket TEXT,
    s3_key TEXT,
    size_bytes BIGINT,
    checksum TEXT,
    metadata JSONB DEFAULT '{}',
    execution_id TEXT,
    workflow_id TEXT,
    job_id TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS artifact_lineage (
    id BIGSERIAL PRIMARY KEY,
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    parent_artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    relation TEXT NOT NULL DEFAULT 'derived_from',
    UNIQUE (artifact_id, parent_artifact_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts (artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_owner ON artifacts (owner_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_execution ON artifacts (execution_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts (project_id);

CREATE TABLE IF NOT EXISTS artifact_registry_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO artifact_registry_schema_version (id, version)
VALUES (1, 1)
ON CONFLICT (id) DO NOTHING;
