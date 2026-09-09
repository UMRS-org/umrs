-- 0085: durable secret-lease persistence for the broker (R0-2).
--
-- The broker previously held leases + audit in process memory (_leases
-- dict): a server restart wiped every active lease and its audit trail,
-- which session-venue revocation semantics cannot tolerate (design §3.2).
--
-- Plaintext secret values are NEVER stored (hash-only model unchanged:
-- only token_hash). Plaintext is returned exactly once at issue time.

CREATE TABLE IF NOT EXISTS secret_leases (
    lease_id           TEXT PRIMARY KEY,
    token_hash         TEXT NOT NULL UNIQUE,          -- SHA-256 of issued value
    job_id             TEXT NOT NULL DEFAULT '',
    secret_name        TEXT NOT NULL,
    scope              TEXT NOT NULL,
    owner_id           TEXT NOT NULL DEFAULT '',
    bound_to_step      TEXT NOT NULL DEFAULT '',
    tier               TEXT NOT NULL DEFAULT '',
    -- inference-scope bindings (R0-2 / §3.2)
    session_id         TEXT NOT NULL DEFAULT '',
    source_ids         UUID[] NOT NULL DEFAULT '{}',
    default_source_id  UUID,
    egress_ip          INET,
    project            TEXT NOT NULL DEFAULT '',
    -- lifecycle
    issued_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at         TIMESTAMPTZ NOT NULL,
    max_lifetime_at    TIMESTAMPTZ,                   -- hard cap across renewals
    is_revoked         BOOLEAN NOT NULL DEFAULT FALSE,
    revoked_at         TIMESTAMPTZ,
    access_count       INTEGER NOT NULL DEFAULT 0,
    last_accessed_at   TIMESTAMPTZ,
    last_used_at       TIMESTAMPTZ                    -- batched proxy heartbeat (R1)
);

COMMENT ON TABLE secret_leases IS
    'Durable secret-lease rows (hash-only; plaintext never persisted). '
    'Inference scope binds session_id/source_ids/egress_ip for the '
    'credential-injecting inference proxy authorization checks.';

CREATE INDEX IF NOT EXISTS idx_secret_leases_session
    ON secret_leases (session_id) WHERE NOT is_revoked;
CREATE INDEX IF NOT EXISTS idx_secret_leases_owner
    ON secret_leases (owner_id);
CREATE INDEX IF NOT EXISTS idx_secret_leases_expires
    ON secret_leases (expires_at) WHERE NOT is_revoked;

-- Durable audit trail (operations logged in-process before 0085 lived in
-- memory too; same reason for persistence).
CREATE TABLE IF NOT EXISTS secret_audit (
    entry_id    TEXT PRIMARY KEY,
    lease_id    TEXT NOT NULL REFERENCES secret_leases (lease_id),
    operation   TEXT NOT NULL,                        -- issue|access|revoke|expire|renew|push|unpush|deny
    actor       TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE secret_audit IS
    'Broker audit trail (durable). detail carries no secrets — hashes and '
    'metadata only.';

CREATE INDEX IF NOT EXISTS idx_secret_audit_lease ON secret_audit (lease_id);
CREATE INDEX IF NOT EXISTS idx_secret_audit_time ON secret_audit (created_at DESC);
