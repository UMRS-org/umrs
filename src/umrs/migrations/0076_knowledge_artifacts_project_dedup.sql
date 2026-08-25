-- 0076: per-project artifact dedup (audit B8, 2026-08-26).
--
-- 0055 declared ``knowledge_artifacts.content_hash UNIQUE`` globally. With
-- the durable ingest chain that made identical bytes in a *second* project
-- silently produce no artifact row; the chained KNOWLEDGE_UNDERSTAND then hit
-- an IR FK violation and recorded a failed_item while the job still reported
-- ``succeeded``. Dedup semantics are per-project: the same document may exist
-- in many projects (each with its own graph), while re-ingest of identical
-- bytes within one project stays idempotent.
--
-- 1. Replace the global UNIQUE with a partial unique index on
--    (project_id, content_hash) for active rows.
-- 2. Pre-existing duplicate content_hash values across projects (only
--    possible after manual fixes) keep working: we never recreate a global
--    constraint.
--
-- Idempotent: constraint drop + index creation are both IF [NOT] EXISTS.

ALTER TABLE knowledge_artifacts
    DROP CONSTRAINT IF EXISTS knowledge_artifacts_content_hash_key;

CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_project_hash_active
    ON knowledge_artifacts (project_id, content_hash)
    WHERE status = 'active';

COMMENT ON TABLE knowledge_artifacts IS
    'Content-addressed artifact metadata. S3 storage keyed by content_hash. '
    'Active-row dedup is per (project_id, content_hash) since 0076 (B8).';
