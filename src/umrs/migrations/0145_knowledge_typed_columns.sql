-- 0145: typed columns beside the 7-bucket facet (ADR-081 D-1, W2a; slice
-- KN-S7 of decisions-2026-11-24).
--
-- Additive and idempotent; number provisional (taken at land time). Every
-- new column is NULL or has a constant default, so legacy rows stay valid
-- and Postgres adds the columns without a table rewrite. The node_type
-- CHECK stays as the coarse facet; the store derives it from the class's
-- bucket for typed writes.
--
-- Not here, on purpose: search_text (0120 already added it, with its own
-- lexical index), a generated search_tsv (a stored generated column
-- rewrites the table; the lexical lanes read 0120's index), pg_trgm
-- indexes (ADR-093 MD-4), node aliases (D-6 resolution) and claims (D-7),
-- which arrive with their own slices.
--
-- Down (manual): DROP INDEX the indexes below; ALTER TABLE ... DROP COLUMN
-- the columns below.

ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS node_class TEXT REFERENCES knowledge_terms (term_id);
ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS attrs JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS natural_key TEXT;
ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ;
ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS invalidated_by TEXT;
ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS invalid_reason TEXT;
ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS proposal_id TEXT;

CREATE INDEX IF NOT EXISTS idx_kn_class ON knowledge_nodes (graph_id, node_class);
CREATE INDEX IF NOT EXISTS idx_kn_attrs ON knowledge_nodes USING GIN (attrs jsonb_path_ops);
CREATE UNIQUE INDEX IF NOT EXISTS uq_kn_natural_key
    ON knowledge_nodes (graph_id, node_class, natural_key) WHERE natural_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kn_current
    ON knowledge_nodes (graph_id, node_class) WHERE invalidated_at IS NULL;

ALTER TABLE knowledge_edges
    ADD COLUMN IF NOT EXISTS relation_class TEXT REFERENCES knowledge_terms (term_id);
ALTER TABLE knowledge_edges
    ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ;
ALTER TABLE knowledge_edges
    ADD COLUMN IF NOT EXISTS proposal_id TEXT;

-- A typed duplicate (same endpoints, same relation class) is impossible;
-- legacy edges (relation_class NULL) are untouched by the index, and so is
-- the warn-mode fallback r4:relatedTo: two different unknown relations
-- between one pair both fall back to it and must both survive (their raw
-- relation is in content.rel_raw).
CREATE UNIQUE INDEX IF NOT EXISTS uq_ke_typed
    ON knowledge_edges (source_id, relation_class, target_id)
    WHERE relation_class IS NOT NULL AND relation_class <> 'r4:relatedTo';
CREATE INDEX IF NOT EXISTS idx_ke_src_rel ON knowledge_edges (source_id, relation_class);
CREATE INDEX IF NOT EXISTS idx_ke_tgt_rel ON knowledge_edges (target_id, relation_class);
