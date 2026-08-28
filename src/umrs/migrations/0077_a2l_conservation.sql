-- 0077: A2L conservation domain — UMRS merge (ADR-034).
--
-- A2L becomes a bounded conservation domain inside UMRS: it stores the
-- *conserved additions* to execution history — deterministic artifact
-- identities, operational lineage edges, agent-execution annotations, and
-- human/operator decisions — keyed to the authoritative execution spine by
-- job_id/attempt_id. A2L joins against jobs/attempts at query time; it
-- never copies execution state.
--
-- Upgrade path: the standalone A2L service (a2l/, port 8002) created the
-- same tables in this shared Postgres. This migration is fully idempotent
-- (IF NOT EXISTS / IF EXISTS / DROP-then-ADD throughout) so it upgrades
-- existing deployments in place: the additive job_id/attempt_id columns are
-- new here; everything else matches the standalone DDL.
--
-- Spine-attribution keys (new in UMRS merge):

-- ============================================================================
-- a2l_artifacts — immutable conserved artifact identities (soft-delete)
-- ============================================================================
DO $$ BEGIN
    CREATE TYPE a2l_artifact_status AS ENUM ('pending', 'running', 'completed', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS a2l_artifacts (
    id              TEXT PRIMARY KEY,              -- SHA-256 content+context hash
    type            TEXT NOT NULL,
    type_schema     JSONB NOT NULL DEFAULT '{}',
    uri             TEXT NOT NULL,                 -- S3 object URI
    embedding       vector(1536),
    metadata        JSONB NOT NULL DEFAULT '{}',
    status          a2l_artifact_status NOT NULL DEFAULT 'pending',
    user_id         TEXT NOT NULL,
    project_id      TEXT,
    hash_version    INTEGER NOT NULL DEFAULT 1,
    artifact_class  TEXT,
    job_id          TEXT,                          -- ADR-034 spine key (new)
    attempt_id      TEXT,                          -- ADR-034 spine key (new)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error           TEXT,
    deleted_at      TIMESTAMPTZ
);

ALTER TABLE a2l_artifacts ADD COLUMN IF NOT EXISTS job_id TEXT;
ALTER TABLE a2l_artifacts ADD COLUMN IF NOT EXISTS attempt_id TEXT;

CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_type       ON a2l_artifacts (type);
CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_status     ON a2l_artifacts (status);
CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_user       ON a2l_artifacts (user_id);
CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_project    ON a2l_artifacts (project_id);
CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_created    ON a2l_artifacts (created_at);
CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_class      ON a2l_artifacts (artifact_class);
CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_job        ON a2l_artifacts (job_id);
CREATE INDEX IF NOT EXISTS idx_a2l_artifacts_metadata   ON a2l_artifacts USING gin (metadata);

-- ============================================================================
-- a2l_artifact_edges — operational lineage (parent → child)
-- ============================================================================
CREATE TABLE IF NOT EXISTS a2l_artifact_edges (
    parent_id           TEXT NOT NULL REFERENCES a2l_artifacts(id),
    child_id            TEXT NOT NULL REFERENCES a2l_artifacts(id),
    relationship_type   TEXT NOT NULL DEFAULT 'derived_from',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (parent_id, child_id)
);

CREATE INDEX IF NOT EXISTS idx_a2l_edges_child ON a2l_artifact_edges (child_id);

ALTER TABLE a2l_artifact_edges DROP CONSTRAINT IF EXISTS no_self_edge;
ALTER TABLE a2l_artifact_edges
    ADD CONSTRAINT no_self_edge CHECK (parent_id <> child_id);

-- ============================================================================
-- a2l_agent_executions — agent invocation annotations (trace/output/error)
-- ============================================================================
DO $$ BEGIN
    CREATE TYPE a2l_execution_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS a2l_agent_executions (
    id              TEXT PRIMARY KEY,              -- UUID
    agent_name      TEXT NOT NULL,
    agent_version   TEXT NOT NULL DEFAULT '',
    model_ref       TEXT NOT NULL DEFAULT '',
    prompt_hash     TEXT NOT NULL DEFAULT '',
    params          JSONB NOT NULL DEFAULT '{}',
    status          a2l_execution_status NOT NULL DEFAULT 'pending',
    user_id         TEXT NOT NULL,
    project_id      TEXT,
    execution_trace JSONB,
    job_id          TEXT,                          -- ADR-034 spine key (new)
    attempt_id      TEXT,                          -- ADR-034 spine key (new)
    error           TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE a2l_agent_executions ADD COLUMN IF NOT EXISTS job_id TEXT;
ALTER TABLE a2l_agent_executions ADD COLUMN IF NOT EXISTS attempt_id TEXT;

CREATE INDEX IF NOT EXISTS idx_a2l_executions_agent   ON a2l_agent_executions (agent_name);
CREATE INDEX IF NOT EXISTS idx_a2l_executions_status  ON a2l_agent_executions (status);
CREATE INDEX IF NOT EXISTS idx_a2l_executions_user    ON a2l_agent_executions (user_id);
CREATE INDEX IF NOT EXISTS idx_a2l_executions_created ON a2l_agent_executions (created_at);
CREATE INDEX IF NOT EXISTS idx_a2l_executions_job     ON a2l_agent_executions (job_id);
CREATE INDEX IF NOT EXISTS idx_a2l_executions_attempt ON a2l_agent_executions (attempt_id);

-- ============================================================================
-- a2l_execution_artifacts — execution ↔ artifact links (input/output)
-- ============================================================================
CREATE TABLE IF NOT EXISTS a2l_execution_artifacts (
    execution_id    TEXT NOT NULL REFERENCES a2l_agent_executions(id),
    artifact_id     TEXT NOT NULL REFERENCES a2l_artifacts(id),
    direction       TEXT NOT NULL CHECK (direction IN ('input', 'output')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (execution_id, artifact_id, direction)
);

CREATE INDEX IF NOT EXISTS idx_a2l_exec_artifacts_artifact
    ON a2l_execution_artifacts (artifact_id);

-- ============================================================================
-- a2l_decisions — human/operator decisions + annotations (NEW in UMRS merge)
-- The conservation-domain-only value: what a human decided and why, keyed to
-- the job/attempt that prompted it. No execution state is copied here.
-- ============================================================================
CREATE TABLE IF NOT EXISTS a2l_decisions (
    id              TEXT PRIMARY KEY,
    job_id          TEXT,
    attempt_id      TEXT,
    artifact_id     TEXT,
    decided_by      TEXT NOT NULL,                -- user id of the decider
    decision        TEXT NOT NULL,                -- e.g. approved/rejected/deferred
    rationale       TEXT NOT NULL DEFAULT '',
    payload         JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_a2l_decisions_job      ON a2l_decisions (job_id);
CREATE INDEX IF NOT EXISTS idx_a2l_decisions_artifact ON a2l_decisions (artifact_id);
CREATE INDEX IF NOT EXISTS idx_a2l_decisions_created  ON a2l_decisions (created_at);
