-- 0110: accounts into Postgres beside projects (D-06 first leg; ADR-063 tail).
--
-- The JSON user files remain the BOOTSTRAP/RECOVERY lane (umrs-server admin
-- works with the daemon stopped; the offline-first contract is unchanged).
-- At boot the server syncs JSON → Postgres and runtime reads go PG-first
-- (the record's authoritative copy once synced); writes dual-write so the
-- JSON files never lag the truth the offline CLI needs.

CREATE TABLE IF NOT EXISTS users (
    username            TEXT PRIMARY KEY,
    fullname            TEXT NOT NULL DEFAULT '',
    email               TEXT NOT NULL DEFAULT '',
    role                TEXT NOT NULL DEFAULT 'user'
                        CONSTRAINT users_role_check CHECK (role IN ('user', 'admin', 'service')),
    disabled            BOOLEAN NOT NULL DEFAULT FALSE,
    slurm_account       TEXT,
    slurm_accounts      JSONB NOT NULL DEFAULT '{}'::jsonb,
    projects            JSONB NOT NULL DEFAULT '[]'::jsonb,
    quota               JSONB NOT NULL DEFAULT '{}'::jsonb,
    oidc_subject        TEXT NOT NULL DEFAULT '',
    curve_public_key    TEXT NOT NULL DEFAULT '',
    curve_key_revoked_at TEXT NOT NULL DEFAULT '',
    extras              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TEXT NOT NULL DEFAULT '',
    last_updated        TEXT NOT NULL DEFAULT ''
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oidc_subject
    ON users (oidc_subject) WHERE oidc_subject <> '';

COMMENT ON TABLE users IS
    'D-06 accounts beside projects: the authoritative runtime copy of the user records (JSON files stay the bootstrap/recovery lane; boot sync JSON→PG, writes dual-write).';
