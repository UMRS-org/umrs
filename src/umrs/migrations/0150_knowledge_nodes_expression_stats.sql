-- 0150 (slice KN-S6): planner statistics for the JSON keys the knowledge
-- read lanes filter on (ADR-083 D-2).
--
-- The dense lane's ANN read joins knowledge_nodes and filters
-- content->>'kind' = 'chunk' and content->>'origin'. Without statistics on
-- those expressions Postgres guesses 0.5 % selectivity, decides the node side
-- is tiny, and never takes the per-space HNSW index (hnsw_<space_id>): every
-- query stays an exact scan. Expression statistics (PG 14+) give the planner
-- the real fractions; ANALYZE (autovacuum, or the one below) fills them.
--
-- Cheap and additive: no table rewrite, no lock beyond ANALYZE's.
-- Number: provisional; migration numbers are assigned in merge-train order at
-- land time (decisions-2026-11-24 §3 item 11).
--
-- Down (manual): DROP STATISTICS IF EXISTS knowledge_nodes_kind_stats,
--                knowledge_nodes_origin_stats;

CREATE STATISTICS IF NOT EXISTS knowledge_nodes_kind_stats
    ON (content->>'kind') FROM knowledge_nodes;

CREATE STATISTICS IF NOT EXISTS knowledge_nodes_origin_stats
    ON (content->>'origin') FROM knowledge_nodes;

ANALYZE knowledge_nodes;
