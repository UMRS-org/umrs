-- 0030_credential_provenance.sql
-- Unified credential provenance ledger.
--
-- Tracks the lifecycle of every platform credential so operators can answer:
--   "Who created this credential, when, for what purpose, and is it still live?"
--
-- This is an *append-mostly* ledger.  Rows are never hard-deleted — revocations
-- and rotations set status/timestamps.  The platform_audit_log captures the
-- individual mutation events; this table captures the *current state* of each
-- credential for inventory and compliance queries.

CREATE TABLE IF NOT EXISTS credential_provenance (
    credential_id   TEXT PRIMARY KEY,          -- opaque stable id (uuid or slug)
    credential_type TEXT NOT NULL
                    CHECK (credential_type IN (
                        'admin_token',         -- UMRS admin API token
                        'jwt_signing_key',     -- Gateway HS256 / RS256 key
                        'litellm_service_key', -- Gateway → LiteLLM bearer key
                        'litellm_master_key',  -- LiteLLM master key
                        'minio_access_key',    -- S3/MinIO access credentials
                        'pg_role_password',    -- Per-experience Postgres role
                        'ssh_deploy_key'       -- CI/CD deploy key
                    )),
    owner           TEXT NOT NULL DEFAULT 'system',  -- user or service that owns it
    description     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'rotated', 'revoked', 'expired')),

    -- Lifecycle timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ,               -- NULL = no expiry
    last_used_at    TIMESTAMPTZ,
    rotated_at      TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ,

    -- Rotation lineage
    replaces_id     TEXT,                      -- credential_id of predecessor
    replaced_by_id  TEXT,                      -- credential_id of successor

    -- Provenance metadata
    created_by      TEXT NOT NULL DEFAULT 'system',  -- actor who created
    created_via     TEXT NOT NULL DEFAULT 'manual',  -- cli | api | migration | bootstrap
    scope           TEXT NOT NULL DEFAULT 'system'
                    CHECK (scope IN ('system', 'project', 'user'))
);

-- Active credentials of each type for inventory queries.
CREATE INDEX IF NOT EXISTS idx_cred_prov_active
    ON credential_provenance (credential_type, owner)
    WHERE status = 'active';

-- Rotation chain traversal.
CREATE INDEX IF NOT EXISTS idx_cred_prov_lineage
    ON credential_provenance (replaces_id)
    WHERE replaces_id IS NOT NULL;

-- Expiry sweeps.
CREATE INDEX IF NOT EXISTS idx_cred_prov_expiry
    ON credential_provenance (expires_at)
    WHERE status = 'active' AND expires_at IS NOT NULL;

COMMENT ON TABLE credential_provenance IS 'Unified credential lifecycle ledger — every platform secret is tracked here';
