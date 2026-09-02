-- Migration 0078: Companion trust allowlist (wave T3c, ADR-035 follow-on)
--
-- Server-side per-user-scope TRUST records consulted by autopilot. One row
-- per (user_id, scope, tool_name) with DB-clock expiry; lazy GC on read (
-- expired rows delete themselves the next time a check touches them).
--
-- Scope is opaque at this layer:
--   'global' | 'project:<project_id>' | 'session:<session_id>'
-- The API layer resolves which scopes a caller may read/write.

CREATE TABLE IF NOT EXISTS companion_trust_allowlist (
    user_id     TEXT         NOT NULL,
    scope       TEXT         NOT NULL DEFAULT 'global',
    tool_name   TEXT         NOT NULL,
    expires_at  TIMESTAMPTZ  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    created_by  TEXT         NOT NULL DEFAULT '',

    PRIMARY KEY (user_id, scope, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_companion_trust_user
    ON companion_trust_allowlist(user_id, expires_at);

COMMENT ON TABLE companion_trust_allowlist IS
    'Per-user-scope TRUST allowlist (ADR-035). Server record consulted by autopilot.';
