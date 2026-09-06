-- Migration 0081: Durable approval inbox for server-hosted research loops
-- (ADR-036 amendment (a), WP-05 round 1; G-P8-1 / G-P6-2 pending-approval leg)
--
-- Pending tool-grain approval intents raised by UMRS-hosted loop drivers were
-- process state (umrs.knowledge.tool_approvals dict). A restart rebuilt the
-- intent idempotently from loop.metadata.awaiting_approval but the DECISION
-- was lost, so the loop re-asked. This table makes the intent AND its decision
-- durable rows, answerable from any client of the loop's owner.
--
-- Lives in the companion_trust schema component (schema.py v3) — the ADR-036
-- substrate decision: same component as the TRUST allowlist, dedicated table
-- (an approval is a per-action intent with a payload, not a (user, scope,
-- tool) allowlist row). ``always`` decisions additionally mint a
-- session-scope trust row (route layer).

CREATE TABLE IF NOT EXISTS companion_approval_inbox (
    approval_id  TEXT        PRIMARY KEY,
    loop_id      TEXT        NOT NULL,
    iteration    INTEGER     NOT NULL DEFAULT 0,
    user_id      TEXT        NOT NULL DEFAULT '',
    job_id       TEXT        NOT NULL DEFAULT '',
    attempt_id   TEXT        NOT NULL DEFAULT '',
    action       JSONB       NOT NULL DEFAULT '{}',
    status       TEXT        NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'approved', 'rejected')),
    decision     TEXT        NOT NULL DEFAULT '',
    decided_by   TEXT        NOT NULL DEFAULT '',
    decided_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_companion_approval_inbox_pending
    ON companion_approval_inbox(status, user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_companion_approval_inbox_loop
    ON companion_approval_inbox(loop_id, iteration);

COMMENT ON TABLE companion_approval_inbox IS
    'Durable tool-grain approval intents + decisions for server-hosted research loops (ADR-036 (a)).';
