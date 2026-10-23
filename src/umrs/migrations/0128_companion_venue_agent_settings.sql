-- 0128: companion_venue_agent_settings — the user's agent settings for a
-- remote session (ADR-099 D-8, RS-1 lane A).
--
-- A remote session behaves like the user's local session (maintainer
-- decision M-6, 2026-10-23). The client computes the agent settings its
-- local bridge applies (permission mode, interaction mode, write-requires-
-- plan, workflow arming, ask_user mode, approval TTL, tools approved
-- "always" in this session) and sends them before every remote start and on
-- every change (PUT /umrs/companion/sessions/{sid}/venue/settings). The
-- worker reads them through its rail at bootstrap and before each queued
-- turn, and applies them when `rev` grew.
--
-- A separate revisioned row, not the snapshot: the worker owns the snapshot
-- while it runs (server wins for frontend), so the owner's later changes
-- would race it.
--
-- `settings` keeps unknown keys (a newer client's); the worker ignores them.
-- A row belongs to `user_id`: a write by another user never replaces it.

CREATE TABLE IF NOT EXISTS companion_venue_agent_settings (
    session_id TEXT        PRIMARY KEY,
    user_id    TEXT        NOT NULL,
    settings   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    rev        INTEGER     NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE companion_venue_agent_settings IS
    'Agent settings the owner''s client mirrors to a remote-session worker, '
    'revisioned (ADR-099 D-8).';
