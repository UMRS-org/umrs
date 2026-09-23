-- 0108: D-24 — the catalog row declares the trust class per member MCP
-- server id (mcp_trust_classes JSONB, {server_id: trust_class}).
--
-- The declaration is a DERIVED projection resolved from the live MCP
-- registry at catalog create/update (fail-open: an unresolvable member
-- records "" — never an invented class). The registry remains the truth;
-- this column is the catalog's visible declaration of what each member
-- needs.

ALTER TABLE scientific_catalogs
    ADD COLUMN IF NOT EXISTS mcp_trust_classes JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN scientific_catalogs.mcp_trust_classes IS
    'D-24 declaration: {mcp_server_id: trust_class} resolved from the MCP registry at create/update; absent members carry "" — the registry stays the truth.';
