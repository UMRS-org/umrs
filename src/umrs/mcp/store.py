"""
MCP Server Registry data access layer.

CRUD operations for mcp_servers and mcp_tools tables.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2

from .models import MCPServer, MCPTool, generate_server_id, generate_tool_id

logger = logging.getLogger(__name__)


class MCPServerStoreError(Exception):
    """Base error for MCP server store operations."""


class MCPServerNotFoundError(MCPServerStoreError):
    """Raised when a server is not found."""


class DuplicateMCPServerError(MCPServerStoreError):
    """Raised when registering a server with a duplicate name."""


class MCPServerStore:
    """
    MCP Server Registry data access layer.

    Provides CRUD operations for MCP server and tool registrations.
    """

    def __init__(self, conn: psycopg2.extensions.connection):
        self.conn = conn

    # -------------------------------------------------------------------------
    # Server CRUD
    # -------------------------------------------------------------------------

    def register_server(
        self,
        name: str,
        url: str,
        *,
        version: str = "0.1.0",
        transport: str = "http",
        description: str = "",
        auth_type: str = "none",
        auth_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        registered_by: str | None = None,
        project_id: str | None = None,
    ) -> MCPServer:
        """
        Register or update an MCP server and its tools.

        If a server with the same name exists, it is updated (upsert).

        Returns:
            The registered MCPServer.
        """
        server_id = generate_server_id()

        with self.conn.cursor() as cur:
            # Upsert server
            cur.execute(
                """
                INSERT INTO mcp_servers (
                    server_id, name, version, url, transport, status,
                    description, auth_type, auth_config, metadata,
                    registered_by, project_id, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, 'healthy',
                    %s, %s, %s, %s,
                    %s, %s, NOW(), NOW()
                )
                ON CONFLICT (name) DO UPDATE SET
                    version = EXCLUDED.version,
                    url = EXCLUDED.url,
                    transport = EXCLUDED.transport,
                    status = 'healthy',
                    description = EXCLUDED.description,
                    auth_type = EXCLUDED.auth_type,
                    auth_config = EXCLUDED.auth_config,
                    metadata = EXCLUDED.metadata,
                    registered_by = EXCLUDED.registered_by,
                    project_id = EXCLUDED.project_id,
                    updated_at = NOW()
                RETURNING server_id, created_at, updated_at
                """,
                (
                    server_id,
                    name,
                    version,
                    url,
                    transport,
                    description,
                    auth_type,
                    json.dumps(auth_config or {}),
                    json.dumps(metadata or {}),
                    registered_by,
                    project_id,
                ),
            )
            row = cur.fetchone()
            actual_server_id = row[0]
            created_at = row[1]
            updated_at = row[2]

            # Replace tools: delete old, insert new
            cur.execute(
                "DELETE FROM mcp_tools WHERE server_id = %s",
                (actual_server_id,),
            )

            mcp_tools = []
            for tool_def in (tools or []):
                tool_name = tool_def.get("name", "")
                if not tool_name:
                    continue
                tool_id = generate_tool_id()
                # Normalize: accept both "inputSchema" and "parameters"
                input_schema = tool_def.get("inputSchema", tool_def.get("parameters", {}))
                tool_desc = tool_def.get("description", "")

                cur.execute(
                    """
                    INSERT INTO mcp_tools (tool_id, server_id, name, description, input_schema)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        tool_id,
                        actual_server_id,
                        tool_name,
                        tool_desc,
                        json.dumps(input_schema),
                    ),
                )
                mcp_tools.append(MCPTool(
                    tool_id=tool_id,
                    server_id=actual_server_id,
                    name=tool_name,
                    description=tool_desc,
                    input_schema=input_schema,
                ))

            self.conn.commit()

        return MCPServer(
            server_id=actual_server_id,
            name=name,
            url=url,
            version=version,
            transport=transport,
            status="healthy",
            description=description,
            auth_type=auth_type,
            auth_config=auth_config or {},
            metadata=metadata or {},
            registered_by=registered_by,
            project_id=project_id,
            tools=mcp_tools,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_server(self, server_id: str) -> MCPServer:
        """
        Get a server by ID.

        Raises:
            MCPServerNotFoundError: If server not found.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT server_id, name, version, url, transport, status,
                       description, auth_type, auth_config, metadata,
                       registered_by, project_id, created_at, updated_at
                FROM mcp_servers WHERE server_id = %s
                """,
                (server_id,),
            )
            row = cur.fetchone()
            if not row:
                raise MCPServerNotFoundError(f"Server not found: {server_id}")

            server = self._row_to_server(row)
            server.tools = self._get_tools_for_server(cur, server_id)
            return server

    def get_server_by_name(self, name: str) -> MCPServer:
        """
        Get a server by name.

        Raises:
            MCPServerNotFoundError: If server not found.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT server_id, name, version, url, transport, status,
                       description, auth_type, auth_config, metadata,
                       registered_by, project_id, created_at, updated_at
                FROM mcp_servers WHERE name = %s
                """,
                (name,),
            )
            row = cur.fetchone()
            if not row:
                raise MCPServerNotFoundError(f"Server not found: {name}")

            server = self._row_to_server(row)
            server.tools = self._get_tools_for_server(cur, server.server_id)
            return server

    def list_servers(
        self,
        project_id: str | None = None,
        status_filter: str | None = None,
    ) -> list[MCPServer]:
        """List all registered servers, optionally filtered."""
        conditions = []
        params: list = []

        if project_id is not None:
            conditions.append("project_id = %s")
            params.append(project_id)
        if status_filter is not None:
            conditions.append("status = %s")
            params.append(status_filter)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT server_id, name, version, url, transport, status,
                       description, auth_type, auth_config, metadata,
                       registered_by, project_id, created_at, updated_at
                FROM mcp_servers {where}
                ORDER BY name
                """,
                params,
            )
            servers = [self._row_to_server(row) for row in cur.fetchall()]

            # Load tools for each server
            for server in servers:
                server.tools = self._get_tools_for_server(cur, server.server_id)

            return servers

    def unregister_server(self, name: str) -> bool:
        """
        Unregister a server by name. Tools are cascade-deleted.

        Returns:
            True if deleted, False if not found.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM mcp_servers WHERE name = %s RETURNING server_id",
                (name,),
            )
            row = cur.fetchone()
            self.conn.commit()
            if row:
                logger.info("Unregistered MCP server: %s", name)
                return True
            return False

    def unregister_server_by_id(self, server_id: str) -> bool:
        """
        Unregister a server by ID.

        Returns:
            True if deleted, False if not found.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM mcp_servers WHERE server_id = %s RETURNING name",
                (server_id,),
            )
            row = cur.fetchone()
            self.conn.commit()
            if row:
                logger.info("Unregistered MCP server: %s (id=%s)", row[0], server_id)
                return True
            return False

    def update_status(self, server_id: str, status: str) -> None:
        """Update a server's status."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE mcp_servers SET status = %s, updated_at = NOW()
                WHERE server_id = %s
                """,
                (status, server_id),
            )
            self.conn.commit()

    # -------------------------------------------------------------------------
    # Tool queries
    # -------------------------------------------------------------------------

    def list_tools(
        self,
        server_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List all tools, optionally filtered by server name.

        Returns list of tool dicts with server name attached.
        """
        with self.conn.cursor() as cur:
            if server_name:
                cur.execute(
                    """
                    SELECT t.name, t.description, t.input_schema, s.name as server_name
                    FROM mcp_tools t
                    JOIN mcp_servers s ON t.server_id = s.server_id
                    WHERE s.name = %s
                    ORDER BY t.name
                    """,
                    (server_name,),
                )
            else:
                cur.execute(
                    """
                    SELECT t.name, t.description, t.input_schema, s.name as server_name
                    FROM mcp_tools t
                    JOIN mcp_servers s ON t.server_id = s.server_id
                    ORDER BY t.name
                    """
                )
            return [
                {
                    "name": row[0],
                    "description": row[1],
                    "inputSchema": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
                    "server": row[3],
                }
                for row in cur.fetchall()
            ]

    def get_server_for_tool(self, tool_name: str) -> MCPServer | None:
        """Find which server provides a given tool."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.server_id, s.name, s.version, s.url, s.transport, s.status,
                       s.description, s.auth_type, s.auth_config, s.metadata,
                       s.registered_by, s.project_id, s.created_at, s.updated_at
                FROM mcp_servers s
                JOIN mcp_tools t ON t.server_id = s.server_id
                WHERE t.name = %s
                LIMIT 1
                """,
                (tool_name,),
            )
            row = cur.fetchone()
            if not row:
                return None

            server = self._row_to_server(row)
            server.tools = self._get_tools_for_server(cur, server.server_id)
            return server

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Get registry statistics."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM mcp_servers")
            server_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM mcp_tools")
            tool_count = cur.fetchone()[0]
            return {"servers": server_count, "tools": tool_count}

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _row_to_server(self, row) -> MCPServer:
        """Convert a database row to MCPServer."""
        auth_config = row[8]
        if isinstance(auth_config, str):
            auth_config = json.loads(auth_config)
        metadata = row[9]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return MCPServer(
            server_id=row[0],
            name=row[1],
            version=row[2],
            url=row[3],
            transport=row[4],
            status=row[5],
            description=row[6] or "",
            auth_type=row[7],
            auth_config=auth_config or {},
            metadata=metadata or {},
            registered_by=row[10],
            project_id=row[11],
            created_at=row[12],
            updated_at=row[13],
        )

    def _get_tools_for_server(self, cur, server_id: str) -> list[MCPTool]:
        """Get all tools for a server."""
        cur.execute(
            """
            SELECT tool_id, server_id, name, description, input_schema, created_at
            FROM mcp_tools WHERE server_id = %s ORDER BY name
            """,
            (server_id,),
        )
        tools = []
        for row in cur.fetchall():
            input_schema = row[4]
            if isinstance(input_schema, str):
                input_schema = json.loads(input_schema)
            tools.append(MCPTool(
                tool_id=row[0],
                server_id=row[1],
                name=row[2],
                description=row[3] or "",
                input_schema=input_schema or {},
                created_at=row[5],
            ))
        return tools
