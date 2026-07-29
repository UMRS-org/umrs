-- Migration 0058: Research loops (RC durability)
--
-- Persists autonomous research loops so loop state survives UMRS restarts.
-- Mirrors the ResearchLoop / LoopIteration knowledge dataclasses.

-- =============================================================================
-- 1. Research loops
-- =============================================================================
CREATE TABLE IF NOT EXISTS research_loops (
    loop_id                 TEXT            PRIMARY KEY,
    project_id              TEXT            NOT NULL,
    research_goal           TEXT            NOT NULL,
    status                  TEXT            NOT NULL,
    phase                   TEXT            NOT NULL,
    max_iterations          INTEGER         NOT NULL DEFAULT 10,
    current_iteration       INTEGER         NOT NULL DEFAULT 0,
    plan                    JSONB,
    total_hypotheses_tested INTEGER         NOT NULL DEFAULT 0,
    total_experiments_run   INTEGER         NOT NULL DEFAULT 0,
    total_findings          INTEGER         NOT NULL DEFAULT 0,
    success_rate            REAL            NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    started_at              TEXT,
    completed_at            TEXT,
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    metadata                JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_research_loops_project ON research_loops(project_id);
CREATE INDEX IF NOT EXISTS idx_research_loops_status ON research_loops(status, updated_at DESC);

COMMENT ON TABLE research_loops IS
    'Autonomous research loops (replacing in-memory _loop_store). State survives restarts.';

-- =============================================================================
-- 2. Loop iterations
-- =============================================================================
CREATE TABLE IF NOT EXISTS research_loop_iterations (
    iteration_id     TEXT         PRIMARY KEY,
    loop_id          TEXT         NOT NULL REFERENCES research_loops(loop_id) ON DELETE CASCADE,
    iteration_number INTEGER      NOT NULL,
    phase            TEXT         NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    hypothesis       TEXT,
    experiment_id    TEXT,
    results          JSONB        NOT NULL DEFAULT '{}',
    analysis         TEXT,
    evaluation       TEXT,
    findings         JSONB        NOT NULL DEFAULT '[]',
    status           TEXT         NOT NULL DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_research_loop_iterations_loop
    ON research_loop_iterations(loop_id, iteration_number);

COMMENT ON TABLE research_loop_iterations IS
    'Per-iteration records of an autonomous research loop (hypothesis → evaluation trail).';
