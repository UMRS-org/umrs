-- 0025_user_config.sql
-- Postgres backend for user provisioning config.
--
-- Migrates user records from per-user JSON files (user_files/<username>.json)
-- to a durable Postgres table.  JSON files remain as bootstrap fallback during
-- migration window and for offline admin CLI usage.
--
-- See also: umrs/src/umrs/admin/user_store.py  (UserRecord dataclass)

CREATE TABLE IF NOT EXISTS user_config (
    username        TEXT PRIMARY KEY
                    CHECK (username ~ '^[a-z][a-z0-9._-]{1,62}$'),
    fullname        TEXT NOT NULL DEFAULT '',
    email           TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'user'
                    CHECK (role IN ('user', 'admin', 'service')),
    slurm_account   TEXT,
    projects        JSONB NOT NULL DEFAULT '[]'::jsonb,
    disabled        BOOLEAN NOT NULL DEFAULT FALSE,
    quota           JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Provisioning-specific fields (experiences, modalities, S3 mappings)
    -- stored as opaque JSONB so the migration does not need to model every
    -- nested key that the daemon/pipeline-SDK uses.
    provisioning    JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_updated    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Fast lookup for active users in a specific project.
CREATE INDEX IF NOT EXISTS idx_user_config_projects
    ON user_config USING gin (projects);

-- Filter disabled users out of normal queries.
CREATE INDEX IF NOT EXISTS idx_user_config_active
    ON user_config (username)
    WHERE NOT disabled;

COMMENT ON TABLE  user_config IS 'Durable user provisioning records (migrated from user_files/*.json)';
COMMENT ON COLUMN user_config.provisioning IS 'Opaque JSONB holding experiences, modalities, S3 config, and other daemon-specific fields';
