-- Egress Audit (A-1204, Phase 2k, 2026-04-25)
--
-- Persists egress proxy decisions made by the Gateway
-- (`gateway/gateway/routes/egress.py`) so that the platform can offer:
--   - `r4 agent audit --last 24h` (A-1209)
--   - cross-tenant audit views for admins
--   - retention/scrubbing policies via the existing janitor
--
-- The Gateway writes one row per request (allowed or denied). Writes are
-- best-effort: a UMRS sink failure must not break an allowed egress request.

CREATE TABLE IF NOT EXISTS egress_requests (
    request_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            TEXT         NOT NULL,
    project_id         TEXT         NOT NULL,
    agent_session_id   UUID,
    tool_name          TEXT,
    destination_url    TEXT         NOT NULL,
    destination_domain TEXT         NOT NULL,
    method             TEXT         NOT NULL DEFAULT 'GET',
    payload_hash       TEXT         NOT NULL,
    payload_size_bytes INTEGER      NOT NULL DEFAULT 0,
    sensitivity_flags  JSONB        NOT NULL DEFAULT '{}'::jsonb,
    user_tags          TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
    action             TEXT         NOT NULL CHECK (action IN ('allowed', 'denied')),
    deny_reason        TEXT,
    response_status    INTEGER,
    response_size_bytes INTEGER,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Common query: per-user feed for `r4 agent audit`, ordered by recency.
CREATE INDEX IF NOT EXISTS idx_egress_requests_user_created
    ON egress_requests(user_id, created_at DESC);

-- Per-project admin view.
CREATE INDEX IF NOT EXISTS idx_egress_requests_project_created
    ON egress_requests(project_id, created_at DESC);

-- Filter denied requests fast (UI 'recent denials' view).
CREATE INDEX IF NOT EXISTS idx_egress_requests_action
    ON egress_requests(action, created_at DESC)
    WHERE action = 'denied';

-- Domain frequency analysis.
CREATE INDEX IF NOT EXISTS idx_egress_requests_domain
    ON egress_requests(destination_domain);

COMMENT ON TABLE egress_requests IS
    'Egress audit trail (A-1204) — one row per Gateway egress proxy decision.';
