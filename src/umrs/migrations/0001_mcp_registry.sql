-- MCP Server Registry (Phase 2 of Gateway Redesign)
--
-- Persists MCP server registrations and tool mappings.
-- Replaces the in-memory MCPRegistry that lived inside Gateway.

CREATE TABLE IF NOT EXISTS mcp_servers (
    server_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(256) NOT NULL UNIQUE,
    version VARCHAR(32) NOT NULL DEFAULT '0.1.0',
    url TEXT NOT NULL,
    transport VARCHAR(32) NOT NULL DEFAULT 'http',
    status VARCHAR(32) NOT NULL DEFAULT 'unknown',
    description TEXT DEFAULT '',
    auth_type VARCHAR(32) NOT NULL DEFAULT 'none',
    auth_config JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    registered_by VARCHAR(128),
    project_id VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mcp_tools (
    tool_id VARCHAR(64) PRIMARY KEY,
    server_id VARCHAR(64) NOT NULL REFERENCES mcp_servers(server_id) ON DELETE CASCADE,
    name VARCHAR(256) NOT NULL,
    description TEXT DEFAULT '',
    input_schema JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(server_id, name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_status ON mcp_servers(status);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_project ON mcp_servers(project_id) WHERE project_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mcp_tools_server ON mcp_tools(server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);

COMMENT ON TABLE mcp_servers IS 'MCP server registry - persistent MCP server registrations';
COMMENT ON TABLE mcp_tools IS 'MCP tool registry - tools provided by registered MCP servers';
