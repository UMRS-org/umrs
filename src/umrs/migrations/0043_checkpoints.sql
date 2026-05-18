-- Migration 0043: Checkpoint persistence for long-running HPC steps (MEM-B6)
--
-- Provides durable storage for intermediate checkpoints during
-- long-running Slurm/HPC steps, enabling crash recovery without
-- full re-execution.

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id   VARCHAR(64)  PRIMARY KEY,
    job_id          VARCHAR(64)  NOT NULL,
    attempt_id      VARCHAR(64)  NOT NULL,
    step_name       VARCHAR(256) NOT NULL,
    sequence_no     INTEGER      NOT NULL DEFAULT 0,
    state_uri       TEXT,                          -- S3 path to serialized state
    state_size_bytes BIGINT      NOT NULL DEFAULT 0,
    state_inline    JSONB,                         -- inline payload for <64KB
    iteration       INTEGER      NOT NULL DEFAULT 0,
    progress        REAL         NOT NULL DEFAULT 0.0,
    metadata        JSONB        NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Fast lookup: latest checkpoint for an attempt
CREATE INDEX IF NOT EXISTS idx_checkpoints_attempt_seq
    ON checkpoints (attempt_id, sequence_no DESC);

-- Fast lookup: all checkpoints for a job
CREATE INDEX IF NOT EXISTS idx_checkpoints_job
    ON checkpoints (job_id, sequence_no DESC);

-- Fast lookup: checkpoints for a specific step within an attempt
CREATE INDEX IF NOT EXISTS idx_checkpoints_attempt_step
    ON checkpoints (attempt_id, step_name, sequence_no DESC);
