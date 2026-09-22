-- 0106: two-tier knowledge — status + confidence on nodes and edges
-- (gap row D-35, design suggestion 2: proposed → established).
--
-- Loop/ingest-extracted facts enter as 'proposed' with a confidence score;
-- they become 'established' only by a human decision (the 0069 proposal
-- decision path marks executed rows) — the same never-auto-approve rule
-- the compiled skills carry. Pre-0106 rows are 'establish' by default:
-- they were written by the vetted ingest chain, not by the loop, so the
-- new tier never retroactively demotes trusted content.

ALTER TABLE knowledge_nodes
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'established'
        CHECK (status IN ('proposed', 'established')),
    ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 0.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0);

ALTER TABLE knowledge_edges
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'established'
        CHECK (status IN ('proposed', 'established')),
    ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 0.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0);

CREATE INDEX IF NOT EXISTS idx_node_status
    ON knowledge_nodes(project_id, status);
CREATE INDEX IF NOT EXISTS idx_edge_status
    ON knowledge_edges(project_id, status);

COMMENT ON COLUMN knowledge_nodes.status IS
    'D-35 two-tier knowledge: proposed (loop/agent-written, pending review) | established (human-approved or vetted-chain ingest).';
COMMENT ON COLUMN knowledge_nodes.confidence IS
    'D-35: confidence score 0..1 carried from extraction/loop provenance.';
