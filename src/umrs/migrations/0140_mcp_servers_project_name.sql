-- 0140: MCP server names are unique per project, not across the registry.
--
-- 2026-11-23 hunt #26: `name` was UNIQUE registry-wide and the upsert
-- (ON CONFLICT (name)) rebound an existing row to the registering project,
-- so any project could take over the global default-rag or another
-- project's server. Guarding the upsert closed the takeover but left one
-- name space: the first project to register `fetch` blocked every other
-- project from it. The key is now (project, name); NULL, '' and padded
-- project ids all mean the global registry, as the store's lookups read
-- them. A project's own row shadows a global one of the same name for
-- that project's callers only.
--
-- The new index is built before the old constraint goes, so there is never
-- a window without a uniqueness rule. Existing rows cannot collide: names
-- were unique.

CREATE UNIQUE INDEX IF NOT EXISTS mcp_servers_project_name_key
    ON mcp_servers ((COALESCE(TRIM(project_id), '')), name);

ALTER TABLE mcp_servers DROP CONSTRAINT IF EXISTS mcp_servers_name_key;
