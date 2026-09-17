-- Migration 0098: Durable chat-turn approval intents (ADR-057 D-3)
--
-- Chat turns raise tool-grain approval gates locally in the r4 client today.
-- For sessions whose truth is mirrored (WP-04) or cross-client-visible, the
-- intent + decision must be durable server rows — same inbox contract as the
-- loop approvals (0081) but scoped to (session_id, turn_id) instead of
-- (loop_id, iteration). "always" mints no trust row here (chat approvals are
-- per-turn intents; session-scope trust stays the TRUST allowlist surface).

CREATE TABLE IF NOT EXISTS chat_turn_approvals (
    approval_id  TEXT        PRIMARY KEY,
    session_id   TEXT        NOT NULL,
    turn_id      TEXT        NOT NULL DEFAULT '',
    user_id      TEXT        NOT NULL DEFAULT '',
    action       JSONB       NOT NULL DEFAULT '{}',
    status       TEXT        NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'approved', 'rejected')),
    decision     TEXT        NOT NULL DEFAULT '',
    decided_by   TEXT        NOT NULL DEFAULT '',
    decided_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_turn_approvals_pending
    ON chat_turn_approvals(status, user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_turn_approvals_session
    ON chat_turn_approvals(session_id, turn_id);

COMMENT ON TABLE chat_turn_approvals IS
    'Durable tool-grain approval intents + decisions for chat turns (ADR-057 D-3).';
