-- 0086: sealed user-credential store (R1-1, design §3.1.1).
--
-- Holds provider keys sealed with a recipient keypair (API holds the public
-- half to seal at push; the inference proxy holds the private half to
-- unseal per request). The API DB role must never read ciphertext columns
-- (column-level grants below); the routes additionally never SELECT them.
--
-- Companion to 0085's secret_leases: unpush revokes leases by source_id
-- through the broker cascade.

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

CREATE TABLE IF NOT EXISTS user_sources_sealed (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner           text NOT NULL,                 -- verified caller identity, never client-supplied
    name            text NOT NULL,
    provider_kind   text NOT NULL CHECK (provider_kind IN (
                        'openai', 'openai_compat', 'anthropic', 'google', 'azure_openai'
                    )),
    base_url        text NOT NULL,                 -- normalized by provider_url validator
    allow_host      text NOT NULL,                 -- 'https://host[:port]'
    allow_paths     text[] NOT NULL DEFAULT '{}',  -- opt-in extras: files, batches, images, audio, fine_tuning
    key_fingerprint text NOT NULL,                 -- 'sha256:' || first 8 hex of SHA-256(api_key)
    seal_key_id     text NOT NULL,                 -- recipient keypair version ('v1:<sha256(pub)[:6]>')
    dek_wrapped     bytea NOT NULL,                -- X25519 envelope per security/seal.py
    nonce           bytea NOT NULL,
    ciphertext      bytea NOT NULL,                -- AES-256-GCM(DEK, api_key, AAD)
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    tombstoned_at   timestamptz
);

COMMENT ON TABLE user_sources_sealed IS
    'Sealed provider credentials (R1-1): API seals with the recipient public '
    'key at push; inference proxy unseals with the private half per request. '
    'API must never SELECT dek_wrapped/nonce/ciphertext (grants + code).';

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_sources_sealed_owner_name
    ON user_sources_sealed (owner, name) WHERE tombstoned_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_user_sources_sealed_owner
    ON user_sources_sealed (owner);

-- Audit rows for push/unpush (lease-scoped ops stay in secret_audit).
CREATE TABLE IF NOT EXISTS user_source_audit (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    action      text NOT NULL,                      -- push | unpush
    actor_type  text NOT NULL DEFAULT 'user',       -- user | lease | proxy | system
    owner       text NOT NULL,
    source_id   uuid,
    detail      jsonb NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE user_source_audit IS
    'Push/unpush audit trail (R1-1). No credential material in detail.';

CREATE INDEX IF NOT EXISTS idx_user_source_audit_owner
    ON user_source_audit (owner, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_source_audit_source
    ON user_source_audit (source_id, created_at DESC);

-- ── Roles + column grants (design §3.1.1) ──────────────────────────────
-- Role creation is idempotent and skipped silently without sufficient
-- privilege (dev DBs); the grants only apply where the roles exist.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_api') THEN
        CREATE ROLE umrs_api NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_infer') THEN
        CREATE ROLE umrs_infer NOLOGIN;
    END IF;
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'skipping role creation (insufficient privilege) — grants below apply where roles exist';
END
$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_api') THEN
        -- API: metadata columns only; the ciphertext columns stay untouched
        -- (default role privileges include nothing on the new table).
        EXECUTE 'GRANT SELECT (id, owner, name, provider_kind, base_url, allow_host, allow_paths, key_fingerprint, seal_key_id, created_at, updated_at, tombstoned_at) ON TABLE user_sources_sealed TO umrs_api';
        EXECUTE 'GRANT INSERT (owner, name, provider_kind, base_url, allow_host, allow_paths, key_fingerprint, seal_key_id, dek_wrapped, nonce, ciphertext), UPDATE (tombstoned_at, updated_at) ON TABLE user_sources_sealed TO umrs_api';
        EXECUTE 'GRANT SELECT, INSERT ON user_source_audit TO umrs_api';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_infer') THEN
        EXECUTE 'GRANT SELECT ON user_sources_sealed TO umrs_infer';
        EXECUTE 'GRANT SELECT, UPDATE (last_accessed_at, access_count) ON secret_leases TO umrs_infer';
        EXECUTE 'GRANT SELECT ON secret_leases TO umrs_infer';
        EXECUTE 'GRANT INSERT ON secret_audit TO umrs_infer';
    END IF;
END
$$;
