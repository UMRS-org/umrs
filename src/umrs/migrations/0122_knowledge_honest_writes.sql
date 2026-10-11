-- 0122: honest knowledge writes (ADR-093 D-11 amendment, wave F, 2026-10-11).
--
-- 1. Hashless artifacts are no identity. 0076's partial unique index on
--    (project_id, content_hash) WHERE status = 'active' made every active
--    row with content_hash '' collide with the first one: an `s3_uri` upload
--    registered without hashing was "deduped" against an unrelated hashless
--    artifact and went down the ingest chain with no row of its own
--    (art-3de698e03d87, 2026-10-08). KnowledgeStore.insert_artifact no longer
--    matches '' either; the index now holds hashed rows only. Existing rows
--    stay valid: the new predicate is narrower than the old one.
--
-- 2. knowledge_events.artifact_id loses NOT NULL. 0056 declared its FK
--    ON DELETE SET NULL on a NOT NULL column, so any delete of an artifact
--    row would abort. The FK stays SET NULL, not CASCADE: an artifact's
--    event trail outlives it (ADR-093 MD-16).
--
-- 3. An expression index on knowledge_nodes (project_id, content->>'artifact_id'):
--    the orphan repair (`python -m umrs.knowledge.orphan_repair`, MD-17) and
--    K3's live-artifact check read nodes by the artifact id their content
--    carries.
--
-- Everything runs inside the runner's transaction. The index build reads the
-- whole node table once (seconds on the stand-alone stack); a large
-- production table needs a maintenance window (ADR-093 MD-4).

DROP INDEX IF EXISTS idx_artifacts_project_hash_active;

CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_project_hash_active
    ON knowledge_artifacts (project_id, content_hash)
    WHERE status = 'active' AND content_hash <> '';

ALTER TABLE knowledge_events ALTER COLUMN artifact_id DROP NOT NULL;

CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_content_artifact
    ON knowledge_nodes (project_id, (content->>'artifact_id'));
