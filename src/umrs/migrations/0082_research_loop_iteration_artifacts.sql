-- Migration 0082: iteration-produced artifact ids on research loop iterations
-- (ADR-036 D4, WP-05 round 1; G-P1-1)
--
-- The asset index derives ``loop → artifact`` lineage once an iteration
-- records the artifact ids it produced (knowledge artifacts minted from its
-- findings, workflow-produced artifacts). Additive column; existing rows read
-- as an empty list.

ALTER TABLE research_loop_iterations
    ADD COLUMN IF NOT EXISTS artifact_ids JSONB NOT NULL DEFAULT '[]';

COMMENT ON COLUMN research_loop_iterations.artifact_ids IS
    'Artifact ids produced by this iteration (finding artifacts, workflow outputs) — asset-index loop→artifact edges (ADR-036 D4).';
