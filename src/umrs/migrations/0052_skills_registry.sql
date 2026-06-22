-- 0052: Skills Registry
--
-- Persists skill bundle registrations so they survive UMRS restarts.
-- Mirrors the in-memory SkillRegistry contract.
--
-- A skill is a bundle of:
--   - SKILL.md    (human-readable instructions)
--   - config.json (manifest with name, version, capabilities)
--   - mcp.json    (optional MCP server manifest)

CREATE TABLE IF NOT EXISTS skills (
    skill_id     VARCHAR(64)  PRIMARY KEY,
    name         VARCHAR(256) NOT NULL,
    version      VARCHAR(64)  NOT NULL DEFAULT '0.1.0',
    scope        VARCHAR(16)  NOT NULL DEFAULT 'user',
    owner_id     VARCHAR(128) NOT NULL DEFAULT '',
    project_id   VARCHAR(128),
    skill_md     TEXT         NOT NULL DEFAULT '',
    manifest     JSONB        NOT NULL DEFAULT '{}',
    enabled      BOOLEAN      NOT NULL DEFAULT TRUE,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(scope, owner_id, name)
);

CREATE INDEX IF NOT EXISTS idx_skills_name      ON skills (name);
CREATE INDEX IF NOT EXISTS idx_skills_scope     ON skills (scope);
CREATE INDEX IF NOT EXISTS idx_skills_owner     ON skills (owner_id) WHERE owner_id <> '';
CREATE INDEX IF NOT EXISTS idx_skills_project   ON skills (project_id) WHERE project_id IS NOT NULL;

COMMENT ON TABLE skills IS 'Persistent skill registry — skill bundles registered via UMRS API';
