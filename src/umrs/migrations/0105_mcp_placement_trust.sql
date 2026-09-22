-- 0105: MCP registry placement + trust class (gap row D-19 / D-49).
--
-- Where an MCP server may run (placement) and what trust class the
-- catalog declares (trust_class, ADR-061):
--
--   placement      local | frontend | cpu:<host> | trusted:<host>
--                  cpu:<host> and trusted:<host> must be declared in
--                  site.yaml extra.venues.{cpu_hosts,trusted_hosts} —
--                  undeclared placements refuse at registration, in code.
--   trust_class    local-safe | front-end | trusted-host-only
--                  trusted-host-only <=> placement trusted:<host> exactly;
--                  front-end forbids cpu:/trusted: placements;
--                  local-safe rides anywhere.

ALTER TABLE mcp_servers
    ADD COLUMN IF NOT EXISTS placement TEXT NOT NULL DEFAULT 'local',
    ADD COLUMN IF NOT EXISTS trust_class TEXT NOT NULL DEFAULT 'local-safe';

COMMENT ON COLUMN mcp_servers.placement IS
    'D-19/D-49: local | frontend | cpu:<host> | trusted:<host> (site-declared hosts only).';
COMMENT ON COLUMN mcp_servers.trust_class IS
    'ADR-061 trust class: local-safe | front-end | trusted-host-only.';
