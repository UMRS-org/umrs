-- 0040_execution_traces.sql
-- Execution trace DAG for operational memory (MEM-B1).
--
-- Provides durable step-level execution traces linked to the durable
-- Job/Attempt model.  Each row represents one step execution within a
-- job or workflow, forming a DAG via parent_trace_id self-FK.
--
-- Design goals:
--   - Answer "what inputs, model refs, code versions, artifacts, and
--     policies produced this result?" for any workflow run.
--   - Enable resumable workflows, execution diff, and replay.
--   - Preserve session-level execution records across UMRS restarts.
--
-- Ownership: UMRS (durable state owner).

-- =============================================================================
-- execution_traces: step-level execution DAG
-- =============================================================================

CREATE TABLE IF NOT EXISTS execution_traces (
    trace_id        VARCHAR(64)  PRIMARY KEY,
    job_id          VARCHAR(64)  NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    attempt_id      VARCHAR(64),          -- NULL for job-level trace root
    execution_id    VARCHAR(64),          -- workflow execution ID (links to WorkflowExecutionRegistry)
    parent_trace_id VARCHAR(64)  REFERENCES execution_traces(trace_id) ON DELETE SET NULL,

    step_name       TEXT         NOT NULL,
    step_type       TEXT         NOT NULL DEFAULT 'generic',  -- generic | llm | tool | rag | mcp | workflow

    -- Execution state
    status          VARCHAR(32)  NOT NULL DEFAULT 'pending',  -- pending | running | succeeded | failed | skipped | cancelled
    progress        REAL         NOT NULL DEFAULT 0.0,

    -- Input/output hashes for deterministic replay
    inputs_hash     VARCHAR(64),          -- SHA-256 of serialized inputs
    outputs_hash    VARCHAR(64),          -- SHA-256 of serialized outputs
    inputs_ref      TEXT,                 -- S3 URI or inline JSONB key for full inputs
    outputs_ref     TEXT,                 -- S3 URI or inline JSONB key for full outputs

    -- Structured metadata
    model_ref       JSONB,               -- {"name": "...", "path": "...", "version": "..."}
    artifact_refs   JSONB        NOT NULL DEFAULT '[]'::jsonb,  -- [{uri, media_type, size_bytes, sha256}]
    tool_calls      JSONB        NOT NULL DEFAULT '[]'::jsonb,  -- [{tool_name, args_hash, result_hash}]
    retrieval_refs  JSONB        NOT NULL DEFAULT '[]'::jsonb,  -- [{source, query, chunk_ids, scores}]
    routing_decision JSONB,              -- {selected_target, reason, alternatives}

    -- Error tracking
    error           TEXT,
    error_detail    JSONB,

    -- Timing
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,

    -- Housekeeping
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    ttl_expires_at  TIMESTAMPTZ           -- NULL = permanent; set for session-ephemeral traces
);

-- Index for job-level trace lookups (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_execution_traces_job
    ON execution_traces(job_id, created_at);

-- Index for execution_id lookups (workflow session traces)
CREATE INDEX IF NOT EXISTS idx_execution_traces_execution
    ON execution_traces(execution_id, created_at)
    WHERE execution_id IS NOT NULL;

-- Index for DAG traversal (parent lookups)
CREATE INDEX IF NOT EXISTS idx_execution_traces_parent
    ON execution_traces(parent_trace_id)
    WHERE parent_trace_id IS NOT NULL;

-- Index for TTL cleanup (session-ephemeral traces)
CREATE INDEX IF NOT EXISTS idx_execution_traces_ttl
    ON execution_traces(ttl_expires_at)
    WHERE ttl_expires_at IS NOT NULL;

-- Index for step status queries
CREATE INDEX IF NOT EXISTS idx_execution_traces_status
    ON execution_traces(status)
    WHERE status NOT IN ('succeeded', 'failed', 'cancelled');

COMMENT ON TABLE execution_traces IS 'Step-level execution trace DAG for operational memory (MEM-B1)';
COMMENT ON COLUMN execution_traces.trace_id IS 'Unique trace identifier (trc-xxxxxxxxxxxx)';
COMMENT ON COLUMN execution_traces.parent_trace_id IS 'Self-FK forming execution DAG (NULL for root steps)';
COMMENT ON COLUMN execution_traces.inputs_hash IS 'SHA-256 of serialized inputs for replay verification';
COMMENT ON COLUMN execution_traces.outputs_hash IS 'SHA-256 of serialized outputs for diff/verification';
COMMENT ON COLUMN execution_traces.inputs_ref IS 'S3 URI to immutable input snapshot for replay';
COMMENT ON COLUMN execution_traces.outputs_ref IS 'S3 URI to immutable output snapshot';
COMMENT ON COLUMN execution_traces.model_ref IS 'Model reference used in this step (JSON ModelRef)';
COMMENT ON COLUMN execution_traces.tool_calls IS 'Tool invocations made during this step';
COMMENT ON COLUMN execution_traces.retrieval_refs IS 'RAG retrieval sources contributing to this step';
COMMENT ON COLUMN execution_traces.routing_decision IS 'Step routing decision with rationale';
COMMENT ON COLUMN execution_traces.ttl_expires_at IS 'NULL = permanent trace; set for session-ephemeral records';
