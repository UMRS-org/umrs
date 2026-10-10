-- 0121: the graph bridge lane's label index (ADR-093 D-14, review
-- profiles-graph-4, 2026-10-10).
--
-- The lane (umrs.knowledge.retrieval.graph_lane) links a query's terms to
-- extracted entity and concept nodes by a full-text match on their labels:
-- to_tsvector('english', coalesce(n.label, '')) @@ <the query's lexemes>,
-- over rows whose node_id starts with the semantic writer's prefix. Without
-- this index every retrieve computes each label's lexemes, so the lane runs
-- only while a scope holds 20,000 semantic nodes or fewer and reports
-- `unavailable` above that (graph_lane.LABEL_SCAN_MAX_ROWS).
--
-- The expression and the predicate are the lane's own, so the planner can
-- reach the index; tests/test_migration_0121.py pins both against the lane.
-- Partial: entity and concept rows only (59 on the stand-alone stack on
-- 2026-10-10), built inside the runner's transaction in well under a second.

CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_semantic_label_fts
    ON knowledge_nodes USING gin (to_tsvector('english'::regconfig, coalesce(label, '')))
    WHERE node_id LIKE 'node-sem-%';
