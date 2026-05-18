-- Migration 0044: Workflow outcome tracking (MEM-D1)
--
-- Provides durable storage for structured workflow execution outcomes
-- including success/failure metrics, resource usage, and model performance
-- per workflow template.

CREATE TABLE IF NOT EXISTS workflow_outcomes (
    outcome_id            VARCHAR(64)  PRIMARY KEY,
    job_id                VARCHAR(64)  NOT NULL,
    workflow_name         VARCHAR(256) NOT NULL,
    workflow_version      VARCHAR(64),
    user_id               VARCHAR(128) NOT NULL,
    project_id            VARCHAR(128) NOT NULL,

    status                VARCHAR(32)  NOT NULL,   -- succeeded | failed | cancelled
    execution_duration_sec REAL,
    step_count            INTEGER,
    failed_step_name      VARCHAR(256),

    model_used            VARCHAR(256),
    model_version         VARCHAR(64),
    embedding_model       VARCHAR(256),

    total_tokens          INTEGER,
    tool_call_count       INTEGER,
    retrieval_doc_count   INTEGER,

    resource_usage        JSONB        DEFAULT '{}',  -- {cpu_sec, memory_peak_mb, gpu_sec}
    failure_reason        TEXT,
    failure_category      VARCHAR(64),  -- timeout|oom|tool_error|llm_error|network|user_input

    step_metrics          JSONB        DEFAULT '[]',  -- [{step_name, status, duration, model, tokens}]

    started_at            TIMESTAMPTZ,
    completed_at          TIMESTAMPTZ,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Aggregate by workflow template
CREATE INDEX IF NOT EXISTS idx_workflow_outcomes_template
    ON workflow_outcomes (workflow_name, completed_at DESC);

-- Filter by model
CREATE INDEX IF NOT EXISTS idx_workflow_outcomes_model
    ON workflow_outcomes (model_used, completed_at DESC);

-- Filter by user/project
CREATE INDEX IF NOT EXISTS idx_workflow_outcomes_user
    ON workflow_outcomes (user_id, project_id, completed_at DESC);

-- Filter by status for failure analysis
CREATE INDEX IF NOT EXISTS idx_workflow_outcomes_status
    ON workflow_outcomes (status, workflow_name);
