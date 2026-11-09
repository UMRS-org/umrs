-- 0133: which workflow version an execution ran (ADR-113 D-4).
--
-- A run pins the registered version it was triggered with and the digest of
-- that version's files (source_sha256, sha256 over the sorted relative paths
-- and contents). record_json carries both too; the columns make them
-- queryable ("which runs used 1.2.0", "which runs used these exact files").
-- Nullable: rows written before ADR-113 recorded neither.

ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS workflow_version TEXT;
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS source_sha256 TEXT;
CREATE INDEX IF NOT EXISTS idx_workflow_executions_version
    ON workflow_executions (workflow_name, workflow_version);
