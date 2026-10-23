-- 0127: chat_turn_questions — durable ask_user questions of remote sessions
-- (ADR-099 D-8, RS-1 lane A; remote-session platform plan 2026-10-22, E-4).
--
-- A remote session's worker used to auto-answer every ask_user call ("no
-- interactive user is attached"). The worker now raises the question through
-- its rail, polls for the answer, and the session's owner answers it from any
-- attached client (TUI ask-user overlay, `r4 session answer`). Unanswered
-- within R4_VENUE_QUESTION_TIMEOUT_SEC (default 1800 s) the worker marks it
-- expired and tells the model not to assume an answer.
--
-- Not chat_turn_approvals (0098): its states and decisions (once | always |
-- reject) do not fit a free-text answer, and answers are conversation
-- content, which admin views never show.
--
-- Terminal transitions only: pending -> answered | expired | cancelled.

CREATE TABLE IF NOT EXISTS chat_turn_questions (
    question_id TEXT        PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    turn_id     TEXT        NOT NULL DEFAULT '',
    user_id     TEXT        NOT NULL,
    question    TEXT        NOT NULL,
    options     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    status      TEXT        NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'answered', 'expired', 'cancelled')),
    answer      TEXT        NOT NULL DEFAULT '',
    answered_by TEXT        NOT NULL DEFAULT '',
    answered_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_turn_questions_session
    ON chat_turn_questions (session_id, status, created_at);

COMMENT ON TABLE chat_turn_questions IS
    'Durable ask_user questions raised by remote-session workers and answered '
    'by the session owner from any client (ADR-099 D-8). Conversation content: '
    'never shown in admin views.';
