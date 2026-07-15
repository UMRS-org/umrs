-- Migration 0057: Verification chat sessions (TASK-046)
--
-- Persists verification-chat sessions so that chat-resolved findings
-- survive UMRS restarts.  Each session records:
--   - the question asked
--   - evidence IDs cited (can regenerate answer)
--   - knowledge-graph traversal path
--   - confidence score
--   - timestamp
--
-- The ephemeral LLM answer is NOT stored (it is regenerated from
-- evidence IDs + graph traversal on demand).

-- =============================================================================
-- 1. Verification Chat Sessions (persistent evidence + traversal)
-- =============================================================================
CREATE TABLE IF NOT EXISTS verification_sessions (
    session_id     TEXT              PRIMARY KEY,
    project_id     TEXT              NOT NULL,
    question       TEXT              NOT NULL,
    evidence_ids   JSONB             NOT NULL DEFAULT '[]',
    graph_traversal TEXT            NOT NULL DEFAULT '[]',
    confidence     REAL              NOT NULL DEFAULT 0,
    timestamp      TIMESTAMPTZ       NOT NULL DEFAULT now()
);

COMMENT ON TABLE verification_sessions IS
    'Persistent verification-chat sessions (evidence IDs + graph traversal). Answer regenerated on read.';
