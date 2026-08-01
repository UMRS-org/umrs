-- 0064: knowledge_artifacts foundation (knowledge-subsystem shipping gap).
--
-- The knowledge subsystem (migrations 0055–0058 + production code in
-- knowledge/store.py, api/a2l.py, and jobs/knowledge_adapters.py) expects a
-- ``knowledge_artifacts`` base table that no migration ever created. Any
-- environment bootstrapped with the migrations CLI therefore uncovers the
-- flaw the first time 0056's FK REFERENCES knowledge_artifacts runs.
-- This shim closes it: additive-only, never overwrites columns that a
-- later canonical knowledge migration may define.
--
-- Columns follow the read/write surface exercised by store.py:
--   artifact_id, project_id, kind, source_uri, title, content, metadata,
--   created_at, updated_at.

CREATE TABLE IF NOT EXISTS knowledge_artifacts (
    -- Row-level PK for a2l-style cascades (0018/0019 FK, api/a2l.py JOIN)
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Business identifier used by store.py and migrations 0055/0056
    artifact_id  TEXT NOT NULL UNIQUE,
    project_id   TEXT NOT NULL DEFAULT '',
    kind         TEXT NOT NULL DEFAULT 'document',
    -- Lifecycle status: store.py queries with status = 'active'
    status       TEXT NOT NULL DEFAULT 'active',
    source_uri   TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    content      TEXT NOT NULL DEFAULT '',
    -- Deduplication key: store.py WHERE content_hash = $1 AND status='active'
    content_hash TEXT,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL DEFAULT ''
);
-- Convergence path for environments where the table pre-exists in an
-- older shape (created before this migration existed): idempotent ALTERs.
ALTER TABLE knowledge_artifacts
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE knowledge_artifacts
    ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_project
    ON knowledge_artifacts (project_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_kind
    ON knowledge_artifacts (kind);
CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_status
    ON knowledge_artifacts (status);
