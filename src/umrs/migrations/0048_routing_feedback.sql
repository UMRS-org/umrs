-- Migration 0048: Routing feedback table (MEM-D2)
--
-- Records model routing decisions and user feedback for
-- adaptive routing scoring (§4.14 Step Routing).

CREATE TABLE IF NOT EXISTS routing_feedback (
    feedback_id     VARCHAR(64)  PRIMARY KEY,
    job_id          TEXT         NOT NULL,
    step_name       TEXT         NOT NULL,
    model_selected  TEXT         NOT NULL,
    outcome         TEXT         DEFAULT '',
    user_accepted   BOOLEAN,
    user_correction TEXT,
    step_type       TEXT         DEFAULT '',
    routing_reason  TEXT         DEFAULT '',
    latency_ms      DOUBLE PRECISION,
    tokens_used     INTEGER,
    quality_score   DOUBLE PRECISION,
    project_id      TEXT         DEFAULT '',
    user_id         TEXT         DEFAULT '',
    created_at      TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rf_model
    ON routing_feedback(model_selected);

CREATE INDEX IF NOT EXISTS idx_rf_job
    ON routing_feedback(job_id);

CREATE INDEX IF NOT EXISTS idx_rf_step_type
    ON routing_feedback(step_type)
    WHERE step_type != '';

CREATE INDEX IF NOT EXISTS idx_rf_created
    ON routing_feedback(created_at DESC);
