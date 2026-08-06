-- Migration 0066: Research Sessions (ADR-025)
--
-- Durable investigation umbrella: sessions bind a project, a title/goal,
-- an authoritative-runtime hint, and the loops started under them. Local
-- (client-side, SQLite) loops are tracked by reference only.

-- =============================================================================
-- 1. research_sessions
-- =============================================================================
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    research_goal TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    authoritative_runtime TEXT NOT NULL DEFAULT 'umrs'
        CHECK (authoritative_runtime IN ('umrs', 'local', 'hybrid')),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_research_sessions_project
    ON research_sessions(project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_sessions_status
    ON research_sessions(status);

COMMENT ON TABLE research_sessions IS
    'Durable research investigation umbrella (ADR-025). Purely additive.';

-- =============================================================================
-- 2. research_session_loops (membership)
-- =============================================================================
CREATE TABLE IF NOT EXISTS research_session_loops (
    session_id TEXT NOT NULL REFERENCES research_sessions(session_id)
        ON DELETE CASCADE,
    loop_id TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT 'umrs'
        CHECK (venue IN ('umrs', 'local')),
    loop_role TEXT NOT NULL DEFAULT 'auto',
    attached_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (session_id, loop_id)
);

CREATE INDEX IF NOT EXISTS idx_session_loops_loop
    ON research_session_loops(loop_id);

COMMENT ON TABLE research_session_loops IS
    'Loop ↔ session membership. venue=local rows are references only.';
