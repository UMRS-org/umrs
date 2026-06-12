"""
Database schema for MCP Server Registry.

Creates and manages the mcp_servers and mcp_tools tables.
"""

from __future__ import annotations

import logging

import psycopg2

logger = logging.getLogger(__name__)

MCP_SCHEMA_VERSION = 1

CREATE_MCP_SERVERS_TABLE = """
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
"""

CREATE_MCP_TOOLS_TABLE = """
CREATE TABLE IF NOT EXISTS mcp_tools (
    tool_id VARCHAR(64) PRIMARY KEY,
    server_id VARCHAR(64) NOT NULL REFERENCES mcp_servers(server_id) ON DELETE CASCADE,
    name VARCHAR(256) NOT NULL,
    description TEXT DEFAULT '',
    input_schema JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(server_id, name)
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_status ON mcp_servers(status);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_project ON mcp_servers(project_id) WHERE project_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mcp_tools_server ON mcp_tools(server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);
"""

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS mcp_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);
"""


def ensure_schema(conn: psycopg2.extensions.connection) -> int:
    """
    Ensure MCP registry schema exists and is up to date.

    Args:
        conn: PostgreSQL connection

    Returns:
        Current schema version after any migrations
    """
    with conn.cursor() as cur:
        cur.execute(CREATE_SCHEMA_VERSION_TABLE)

        cur.execute("SELECT version FROM mcp_schema_version WHERE id = 1")
        row = cur.fetchone()

        if row is None:
            logger.info("Creating MCP registry schema (version %d)", MCP_SCHEMA_VERSION)
            cur.execute(CREATE_MCP_SERVERS_TABLE)
            cur.execute(CREATE_MCP_TOOLS_TABLE)
            cur.execute(CREATE_INDEXES)
            cur.execute(
                """
                INSERT INTO mcp_schema_version (id, version, updated_at)
                VALUES (1, %s, NOW())
                """,
                (MCP_SCHEMA_VERSION,),
            )
            conn.commit()
            return MCP_SCHEMA_VERSION

        current_version = row[0]
        if current_version < MCP_SCHEMA_VERSION:
            logger.info(
                "Migrating MCP schema from version %d to %d",
                current_version,
                MCP_SCHEMA_VERSION,
            )
            # Future migrations go here
            cur.execute(
                """
                UPDATE mcp_schema_version
                SET version = %s, updated_at = NOW()
                WHERE id = 1
                """,
                (MCP_SCHEMA_VERSION,),
            )
            conn.commit()
            return MCP_SCHEMA_VERSION

        logger.debug("MCP registry schema is up to date (version %d)", current_version)
        return current_version
