-- 0016: knowledge_artifacts base table (ordering shim).
--
-- Bug fixed: 0018_a2l_artifact_refs_bridge.sql and 0019_knowledge_a2l_links.sql
-- both declare REFERENCES knowledge_artifacts(id), but NO migration before
-- them ever created the table (the canonical definitions land only at 0055,
-- and that shape is artifact_id-PK without the `id` column the FKs need).
-- Deployed databases grew the table out-of-band, so the flaw only surfaced
-- on a fresh `python -m umrs.migrations` bootstrap.
--
-- The shape here follows migration 0064's convergent formula verbatim
-- (id UUID PK + artifact_id TEXT UNIQUE + full column set + idempotent
-- ALTERs), so the fresh-DB end-state matches 0064's intent; 0055's
-- CREATE TABLE IF NOT EXISTS then no-ops (its extra CHECKs/UNIQUE were
-- only ever applied on databases that grew the table organically — on
-- a fresh DB they were never reachable in any case).
--
-- Migrations that were already applied on deployed databases do NOT
-- re-execute; this file also no-ops there (IF NOT EXISTS + IF NOT EXISTS
-- ALTERs) by design.

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

-- Convergence index set (0064's baseline); later migrations' own indexes
-- apply on top via CREATE INDEX IF NOT EXISTS.
CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_project
    ON knowledge_artifacts (project_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_kind
    ON knowledge_artifacts (kind);
CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_status
    ON knowledge_artifacts (status);
