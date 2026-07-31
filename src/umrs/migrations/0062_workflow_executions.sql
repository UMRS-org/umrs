-- 0062: Durable workflow execution registry (W1).
--
-- The in-process workflow_execution_registry lost every execution record on
-- UMRS restart or failover, leaving `r4 workflow status/watch` blind after a
-- bounce. This table snapshots full ExecutionRecord payloads (record_json)
-- with denormalized filter columns for listing.

CREATE TABLE IF NOT EXISTS workflow_executions (
    execution_id TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted',
    submitted_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    record_json JSONB NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow
    ON workflow_executions (workflow_name, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_user
    ON workflow_executions (user_id, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status
    ON workflow_executions (status);
