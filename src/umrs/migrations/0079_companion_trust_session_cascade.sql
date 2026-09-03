-- Migration 0079: companion trust v2 — session lifecycle cascade (ADR-035 item-8)
--
-- Two additions:
--   companion_session_archives — tombstones for archived companion sessions;
--     archiving a session marks it ended server-side (UMRS owns the durable
--     consequence, so the cascade hangs off the store that owns the rows).
--   companion_trust_revoke_log  — one audit row per grant revoked by the
--     cascade (reason=session_archived), so a later "why did my always-allow
--     disappear" has a durable answer.
--
-- Cascade contract: archive + revoke + audit commit together or not at all
-- (single transaction in TrustAllowlistStore.archive_session / the
-- research-session archive handler).

CREATE TABLE IF NOT EXISTS companion_session_archives (
    session_id TEXT PRIMARY KEY,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_by TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_companion_session_archives_at
    ON companion_session_archives(archived_at DESC);

COMMENT ON TABLE companion_session_archives IS
    'Tombstones for archived companion sessions (ADR-035 item-8 cascade).';

CREATE TABLE IF NOT EXISTS companion_trust_revoke_log (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_companion_trust_revoke_log_session
    ON companion_trust_revoke_log(session_id);

COMMENT ON TABLE companion_trust_revoke_log IS
    'Audit of session-lifecycle trust revocations (one row per revoked grant).';
