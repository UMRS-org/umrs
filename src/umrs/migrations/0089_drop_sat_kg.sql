-- 0089: sat_kg tombstone migration (ADR-026 wave C).
--
-- Drops the deprecated sat_kg storage face for the knowledge-graph API since
-- migration 0060: sat_kg_nodes + sat_kg_edges. The deprecated facade
-- (`umrs.knowledge_graph_api`) was already removed from the tree (2026-09-11
-- wave C, umrs 70afaef4), so NOTHING reads these tables any more — the
-- canonical store is the only producer now (knowledge_nodes/edges).
--
-- Scope (verified not to touch): `sat_embeddings` is NOT one of these — the
-- embeddings_api pg_store owns it for the embeddings service and remains a
-- live production oddity (not sat_kg: the satellite prefix collides by name
-- only, not by fate or contract; wave audit found writes/reads live this
-- register).
--
-- Wave note: user's pre-real-use waiver applies; no production data for
-- these sat_kg tables motivated the exist-sequence. Idempotent; safe to
-- re-run.

DROP TABLE IF EXISTS sat_kg_edges;
DROP TABLE IF EXISTS sat_kg_nodes;
