-- Migration 0020: Admin Token Store
--
-- Moves the admin token registry from a local JSON file
-- (~/.config/umrs/admin_tokens.json) into Postgres for:
--   - Durability and backup coverage
--   - Multi-instance UMRS coordination
--   - Auditable lifecycle (create, rotate, revoke)
--
-- The JSON file remains a valid bootstrap fallback during the
-- migration window.  AdminTokenManager checks Postgres first,
-- falls back to JSON, and can migrate entries forward on first
-- connection.
--
-- See: docs/SYSTEM_AUTHORITY_TRUST_MODEL.md
--      docs/SECRET_LIFECYCLE.md

CREATE TABLE IF NOT EXISTS admin_tokens (
    name            TEXT PRIMARY KEY,
    token_hash      TEXT NOT NULL,
    token_salt      TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ,
    last_used_at    TIMESTAMPTZ,
    use_count       INTEGER NOT NULL DEFAULT 0,
    replaces        TEXT,
    rotated_to      TEXT,
    rotated_at      TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ,
    revoked_by      TEXT
);

-- Index for expiry sweep / listing active tokens
CREATE INDEX IF NOT EXISTS idx_admin_tokens_expires
    ON admin_tokens (expires_at)
    WHERE revoked_at IS NULL AND rotated_to IS NULL;

-- Revoked tokens are soft-deleted (retained for audit trail)
COMMENT ON COLUMN admin_tokens.revoked_at IS
    'When set, the token is revoked. Kept for audit — never hard-delete.';
