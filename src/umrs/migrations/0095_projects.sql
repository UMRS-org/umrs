-- 0095: durable projects registry (ADR-051 D-1).
--
-- Self-service casual projects: any verified user creates freely; the row
-- carries the owner attribution (creator). Retire = tombstone
-- (status='retired', retired_at), never DELETE, per the platform-wide rule.

CREATE TABLE IF NOT EXISTS projects (
    project_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    owner_id    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    retired_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT projects_status_check
        CHECK (status IN ('active', 'retired'))
);

CREATE INDEX IF NOT EXISTS projects_owner_status_idx
    ON projects (owner_id, status);

COMMENT ON TABLE projects IS
    'Durable self-service project registry (ADR-051 D-1) — creation is casual for any verified user; owner attribution rides the row. Retire is a tombstone, never a DELETE.';
