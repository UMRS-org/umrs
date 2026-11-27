-- 0151: the scope overlay's write side (ADR-108 D-4, slice S-W3a).
--
-- A write made under a work level's scope token lands proposed and tagged
-- with the level's path (``s-ab12/loop-7/iter-3``). The tag is a plain TEXT
-- column next to the tier: NULL is the root scope (the session's plain
-- identity), so every row written before this migration, and every root
-- write after it, is unchanged. No ``ltree`` (an install burden for
-- self-hosters, decisions-2026-11-24 DEC-084-108W3): a reader's visibility
-- is "established, or scope_path IN (its own path and its ancestors)", and
-- the ancestor list is computed reader-side (``scope_overlay.scope_lineage``,
-- at most 8 entries), so one btree per table serves it.
--
-- knowledge_artifacts gets the pair: ``tier`` (its ``status`` column is the
-- lifecycle, active/deleted, and stays so) and ``scope_path``. Rows written
-- before 0151 are 'established': they came from the vetted chain or a root
-- request. ``metadata.scope_path`` stamped by W2b is not copied into the
-- column: it is JSON a raw spec could also have set, and the column is the
-- server's record only (nothing issued scope tokens before W4).
--
-- Idempotent: every statement is IF NOT EXISTS.

ALTER TABLE knowledge_artifacts
    ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'established'
        CHECK (tier IN ('proposed', 'established')),
    ADD COLUMN IF NOT EXISTS scope_path TEXT
        CHECK (scope_path IS NULL OR length(scope_path) BETWEEN 1 AND 600);

ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS scope_path TEXT
        CHECK (scope_path IS NULL OR length(scope_path) BETWEEN 1 AND 600);

ALTER TABLE knowledge_edges
    ADD COLUMN IF NOT EXISTS scope_path TEXT
        CHECK (scope_path IS NULL OR length(scope_path) BETWEEN 1 AND 600);

CREATE INDEX IF NOT EXISTS idx_artifact_scope_path
    ON knowledge_artifacts (project_id, scope_path) WHERE scope_path IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_node_scope_path
    ON knowledge_nodes (project_id, scope_path) WHERE scope_path IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_edge_scope_path
    ON knowledge_edges (project_id, scope_path) WHERE scope_path IS NOT NULL;

COMMENT ON COLUMN knowledge_artifacts.tier IS
    'ADR-108 D-4: proposed (written under a work-level scope) | established (root or vetted chain). status stays the lifecycle.';
COMMENT ON COLUMN knowledge_artifacts.scope_path IS
    'ADR-108 D-4: the work-level scope path the row was written under; NULL = root scope.';
COMMENT ON COLUMN knowledge_nodes.scope_path IS
    'ADR-108 D-4: the work-level scope path the row was written under; NULL = root scope.';
COMMENT ON COLUMN knowledge_edges.scope_path IS
    'ADR-108 D-4: the work-level scope path the row was written under; NULL = root scope.';
