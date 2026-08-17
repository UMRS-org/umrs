-- 0073: embedding metadata on knowledge artifact rows (stability S10).
--
-- Records WHICH embedding (model, dim, when) backs an artifact, so the
-- durable re-index pattern (JobType.KNOWLEDGE_REINDEX, sweep→INDEX_BUILD→
-- stamp) can select stale rows deliberately after a model change instead
-- of scanning/recomputing blindly. embedding_status vocabulary:
--   pending — never embedded (default)
--   current — embedding matches the deployment's target model
--   stale   — embedded under a different/older model, re-index due
ALTER TABLE knowledge_artifacts
    ADD COLUMN IF NOT EXISTS embedding_model  TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_dim    INT,
    ADD COLUMN IF NOT EXISTS embedded_at      TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_status TEXT NOT NULL DEFAULT 'pending';

CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_embedding_status
    ON knowledge_artifacts (embedding_status);
