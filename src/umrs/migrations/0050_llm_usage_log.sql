-- Migration 0050: LLM Usage Log
--
-- Per-request token usage tracking for billing and quota enforcement.
-- Records every LLM chat/completion request with user identity, model,
-- token counts (including cache tokens), and timing.
--
-- The table is range-partitioned by event_time (monthly).  Postgres
-- requires child partitions to exist before INSERT.  The initial
-- migration creates one catch-all DEFAULT partition so inserts never
-- fail even without explicit monthly partitions.  Operators should
-- create monthly partitions ahead of time for best query performance:
--
--   CREATE TABLE llm_usage_log_2026_07 PARTITION OF llm_usage_log
--       FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
--
-- See: docs/OPERATOR_RUNBOOK.md (usage-log partitioning)

-- ============================================================
-- 1. Partitioned usage table
-- ============================================================
CREATE TABLE IF NOT EXISTS llm_usage_log (
    id                  BIGSERIAL        NOT NULL,
    request_id          VARCHAR(64),
    event_time          TIMESTAMPTZ      NOT NULL DEFAULT now(),

    -- Caller identity (from auth sidecar headers)
    user_id             TEXT             NOT NULL,
    project_id          TEXT             NOT NULL,

    -- Model / endpoint
    model               TEXT             NOT NULL,          -- model name as requested
    served_model        TEXT,                               -- actual served name (after alias rewrite)
    endpoint_job_id     VARCHAR(64),                        -- links to jobs(job_id)
    endpoint_tier       TEXT,                               -- system / project / user

    -- Token counts
    prompt_tokens       INTEGER          NOT NULL DEFAULT 0,
    completion_tokens   INTEGER          NOT NULL DEFAULT 0,
    total_tokens        INTEGER          NOT NULL DEFAULT 0,

    -- Cache tokens (Anthropic-style; 0 when provider does not report)
    cached_read_tokens  INTEGER          NOT NULL DEFAULT 0,
    cached_write_tokens INTEGER          NOT NULL DEFAULT 0,

    -- Request metadata
    is_streaming        BOOLEAN          NOT NULL DEFAULT false,
    duration_ms         DOUBLE PRECISION,
    status_code         INTEGER,

    -- Future extensibility (tool_call count, reasoning tokens, etc.)
    extra               JSONB            NOT NULL DEFAULT '{}'::jsonb,

    PRIMARY KEY (id, event_time)
) PARTITION BY RANGE (event_time);

-- Default partition — catches rows that don't match any explicit
-- monthly partition.  Prevents INSERT failures before operators
-- create the monthly partitions.
CREATE TABLE IF NOT EXISTS llm_usage_log_default
    PARTITION OF llm_usage_log DEFAULT;

-- ============================================================
-- 2. Indexes (defined on parent; propagate to child partitions)
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_llm_usage_user_time
    ON llm_usage_log (user_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_llm_usage_project_time
    ON llm_usage_log (project_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_llm_usage_model_time
    ON llm_usage_log (model, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_llm_usage_event_time
    ON llm_usage_log (event_time DESC);

-- ============================================================
-- 3. Monthly partitions for current + next month
-- ============================================================
-- These are safe to re-run (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS llm_usage_log_2026_06
    PARTITION OF llm_usage_log
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

CREATE TABLE IF NOT EXISTS llm_usage_log_2026_07
    PARTITION OF llm_usage_log
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

CREATE TABLE IF NOT EXISTS llm_usage_log_2026_08
    PARTITION OF llm_usage_log
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
