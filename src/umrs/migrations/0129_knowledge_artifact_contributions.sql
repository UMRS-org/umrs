-- 0129: knowledge_artifact_contributions — who added an artifact, to which
-- graph, as what kind of actor (knowledge-sharing plan K1.2, 2026-10-24).
--
-- Artifacts are deduplicated per project on their bytes (0076, 0122), so the
-- artifact row records at most the first uploader, and it carries no graph:
-- a personal-graph upload kept the shared project_id and only its nodes moved
-- graph. One row per (artifact, graph, contributor) survives deduplication,
-- names every contributor and every graph the artifact was added to, and is
-- what the artifact paths join to apply the graph-readable predicate (the
-- retrieve verb's hidden_graphs rule): a member never lists another member's
-- personal-graph artifact.
--
-- origin_actor vocabulary (no CHECK, so later origins need no migration):
--   user     a person's explicit action (upload, add)
--   agent    an agent-marked request (x-r4-actor: agent; client-honest, K-12)
--   loop     a research-loop finding (user_id = the loop's owner)
--   workflow a workflow output pinned to the catalog (user_id = run owner)
--   import   a bundle import
--   unknown  backfilled rows that could not be traced (user_id '')
-- A row with user_id '' and origin 'unknown' is a placeholder: writing an
-- attributed row for the same (artifact, graph) removes it
-- (umrs.knowledge.contributions.record_contribution).
--
-- Backfill: python -m umrs.knowledge.contribution_backfill (dry run by
-- default; --apply writes, idempotent).
--
-- Rollback (forward-only): DROP TABLE knowledge_artifact_contributions; the
-- artifact listing then has no rows to list (it never falls back to the
-- unfiltered project list).
--
-- Number: 0129 was reserved for the RS-1 integrator (ADR-099), who landed
-- 0125–0128 without it; 0130 (multisite W0) and 0131–0138 stay reserved.

CREATE TABLE IF NOT EXISTS knowledge_artifact_contributions (
    artifact_id  TEXT        NOT NULL
                 REFERENCES knowledge_artifacts(artifact_id) ON DELETE CASCADE,
    graph_id     TEXT        NOT NULL,
    project_id   TEXT        NOT NULL,
    user_id      TEXT        NOT NULL DEFAULT '',
    origin_actor TEXT        NOT NULL DEFAULT 'unknown',
    title        TEXT        NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (artifact_id, graph_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_artifact_contributions_project_created
    ON knowledge_artifact_contributions (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_artifact_contributions_graph
    ON knowledge_artifact_contributions (graph_id);

COMMENT ON TABLE knowledge_artifact_contributions IS
    'Who added an artifact to which graph, as which actor (K1.2); the join '
    'every artifact path uses for the graph-readable predicate.';
