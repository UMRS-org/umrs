"""
MCP Server Registry data access layer.

CRUD operations for mcp_servers and mcp_tools tables.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import psycopg2
from psycopg2.extensions import TRANSACTION_STATUS_IDLE

from .models import MCPServer, MCPTool, generate_server_id, generate_tool_id

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Exception classes a dead session raises (Postgres restart, network cut, the
# R-14 idle_in_transaction_session_timeout reaper). Only a session psycopg2
# has marked ``closed`` counts as lost: QueryCanceled, deadlock, serialization
# and lock-timeout errors subclass OperationalError on a live session.
_CONNECTION_LOSS = (psycopg2.OperationalError, psycopg2.InterfaceError)


def _session_lost(conn, exc: BaseException) -> bool:
    return isinstance(exc, _CONNECTION_LOSS) and bool(conn.closed)


class _SessionLost(Exception):
    """Internal: the first attempt's session died before any commit."""


def _json_column(value):
    """JSONB column that may arrive as text (codec-less connections)."""
    return json.loads(value) if isinstance(value, str) else value


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

    def __init__(
        self,
        conn: psycopg2.extensions.connection | None = None,
        *,
        pool: Any | None = None,
    ):
        """``pool`` (preferred): every call borrows its own connection and
        hands it back. ``conn``: a caller-owned connection the store never
        replaces (per-request callers). Writes commit, as they always did; a
        read ends only a transaction it opened itself, so a caller's open
        transaction on ``conn`` (bundle_restore's uncommitted skill stubs)
        is left for the caller to commit or roll back."""
        if conn is None and pool is None:
            raise ValueError("MCPServerStore needs a connection or a pool")
        self.conn = conn
        self._pool = pool

    # -------------------------------------------------------------------------
    # Transaction plumbing
    # -------------------------------------------------------------------------

    def _run(self, work: Callable[[Any], _T], *, write: bool = False) -> _T:
        """Run ``work(cursor)``; a transaction the store opened always ends.

        Live bdrt 2026-09-27: reads never ended their implicit transaction,
        the boot-leased connection sat ``idle in transaction`` until R-14's
        reaper killed it, and GET /umrs/mcp/servers answered 500 ("server
        closed the connection unexpectedly", then "connection already closed"
        on every later call). Now writes commit and reads roll back what they
        opened. A session lost before the commit is dropped and the work
        retried once on a fresh pooled connection (nothing was committed); a
        loss during the commit is raised, never retried (the commit may have
        landed).
        """
        try:
            conn, owned, result = self._attempt(work, retry_on_loss=True)
        except _SessionLost as lost:
            logger.warning(
                "MCP store connection lost (%s); retrying once on a fresh one",
                lost.__cause__,
            )
            conn, owned, result = self._attempt(work, retry_on_loss=False)
        if write:
            self._commit(conn, owned)
        self._release(conn, owned=owned)  # a read's rollback ends its transaction
        return result

    def _attempt(self, work: Callable[[Any], _T], *, retry_on_loss: bool):
        """``(conn, owned, work(cursor))``; on failure ``conn`` is released.

        A lost pooled session raises ``_SessionLost`` when ``retry_on_loss``
        (only ``_run`` sees it); every other failure is raised as-is.
        """
        conn = self._acquire()
        owned = self._owns_transaction(conn)
        try:
            with conn.cursor() as cur:
                return conn, owned, work(cur)
        except BaseException as exc:
            lost = _session_lost(conn, exc)
            self._release(conn, owned=owned)
            if lost and retry_on_loss and self._pool is not None:
                raise _SessionLost from exc
            raise

    def _acquire(self):
        return self._pool.getconn() if self._pool is not None else self.conn

    def _owns_transaction(self, conn) -> bool:
        """A pooled connection is the store's own; a caller's connection only
        while the caller has no transaction open on it."""
        if self._pool is not None:
            return True
        return conn.get_transaction_status() == TRANSACTION_STATUS_IDLE

    def _commit(self, conn, owned: bool) -> None:
        try:
            conn.commit()
        except BaseException:
            self._release(conn, owned=owned)
            raise

    def _release(self, conn, *, owned: bool) -> None:
        """End the store's own transaction; hand a pooled conn back (closed
        when the session is gone, so the pool never reissues it)."""
        broken = bool(conn.closed)
        if owned and not broken:
            try:
                conn.rollback()
            except psycopg2.Error:
                broken = True
        if self._pool is not None:
            self._pool.putconn(conn, close=broken or bool(conn.closed))

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
        placement: str = "local",
        trust_class: str = "local-safe",
    ) -> MCPServer:
        """
        Register or update an MCP server and its tools.

        If a server with the same name exists, it is updated (upsert).
        Placement + trust class are validated (D-19/D-49: grammar, declared
        hosts, ADR-061 class map) before any write — a bad placement never
        reaches the registry.

        Returns:
            The registered MCPServer.
        """
        from umrs.shared.venue_placement import (
            validate_placement,
            validate_trust_class,
        )

        placement = validate_placement(placement)
        trust_class = validate_trust_class(placement, trust_class)
        server_id = generate_server_id()

        def _upsert(cur) -> tuple[Any, list[MCPTool]]:
            # Upsert server
            cur.execute(
                """
                INSERT INTO mcp_servers (
                    server_id, name, version, url, transport, status,
                    description, auth_type, auth_config, metadata,
                    registered_by, project_id, placement, trust_class,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, 'healthy',
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, NOW(), NOW()
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
                    placement = EXCLUDED.placement,
                    trust_class = EXCLUDED.trust_class,
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
                    placement,
                    trust_class,
                ),
            )
            row = cur.fetchone()
            actual_server_id = row[0]

            # Replace tools: delete old, insert new
            cur.execute(
                "DELETE FROM mcp_tools WHERE server_id = %s",
                (actual_server_id,),
            )

            mcp_tools = []
            for tool_def in tools or []:
                tool_name = tool_def.get("name", "")
                if not tool_name:
                    continue
                tool_id = generate_tool_id()
                # Normalize: accept both "inputSchema" and "parameters"
                input_schema = tool_def.get(
                    "inputSchema", tool_def.get("parameters", {})
                )
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
                mcp_tools.append(
                    MCPTool(
                        tool_id=tool_id,
                        server_id=actual_server_id,
                        name=tool_name,
                        description=tool_desc,
                        input_schema=input_schema,
                    )
                )
            return row, mcp_tools

        row, mcp_tools = self._run(_upsert, write=True)
        actual_server_id, created_at, updated_at = row[0], row[1], row[2]
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
            placement=placement,
            trust_class=trust_class,
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

        def _select(cur) -> MCPServer:
            cur.execute(
                """
                SELECT server_id, name, version, url, transport, status,
                       description, auth_type, auth_config, metadata,
                       registered_by, project_id, placement, trust_class,
                       created_at, updated_at
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

        return self._run(_select)

    def get_server_by_name(self, name: str) -> MCPServer:
        """
        Get a server by name.

        Raises:
            MCPServerNotFoundError: If server not found.
        """

        def _select(cur) -> MCPServer:
            cur.execute(
                """
                SELECT server_id, name, version, url, transport, status,
                       description, auth_type, auth_config, metadata,
                       registered_by, project_id, placement, trust_class,
                       created_at, updated_at
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

        return self._run(_select)

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

        def _select(cur) -> list[MCPServer]:
            cur.execute(
                f"""
                SELECT server_id, name, version, url, transport, status,
                       description, auth_type, auth_config, metadata,
                       registered_by, project_id, placement, trust_class,
                       created_at, updated_at
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

        return self._run(_select)

    def unregister_server(self, name: str) -> bool:
        """
        Unregister a server by name. Tools are cascade-deleted.

        Returns:
            True if deleted, False if not found.
        """

        def _delete(cur):
            cur.execute(
                "DELETE FROM mcp_servers WHERE name = %s RETURNING server_id",
                (name,),
            )
            return cur.fetchone()

        if self._run(_delete, write=True):
            logger.info("Unregistered MCP server: %s", name)
            return True
        return False

    def unregister_server_by_id(self, server_id: str) -> bool:
        """
        Unregister a server by ID.

        Returns:
            True if deleted, False if not found.
        """

        def _delete(cur):
            cur.execute(
                "DELETE FROM mcp_servers WHERE server_id = %s RETURNING name",
                (server_id,),
            )
            return cur.fetchone()

        row = self._run(_delete, write=True)
        if row:
            logger.info("Unregistered MCP server: %s (id=%s)", row[0], server_id)
            return True
        return False

    def update_status(self, server_id: str, status: str) -> None:
        """Update a server's status."""

        def _update(cur) -> None:
            cur.execute(
                """
                UPDATE mcp_servers SET status = %s, updated_at = NOW()
                WHERE server_id = %s
                """,
                (status, server_id),
            )

        self._run(_update, write=True)

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

        def _select(cur) -> list[dict[str, Any]]:
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
                    "inputSchema": row[2]
                    if isinstance(row[2], dict)
                    else json.loads(row[2]),
                    "server": row[3],
                }
                for row in cur.fetchall()
            ]

        return self._run(_select)

    def get_server_for_tool(self, tool_name: str) -> MCPServer | None:
        """Find which server provides a given tool."""

        def _select(cur) -> MCPServer | None:
            cur.execute(
                """
                SELECT s.server_id, s.name, s.version, s.url, s.transport, s.status,
                       s.description, s.auth_type, s.auth_config, s.metadata,
                       s.registered_by, s.project_id, s.placement, s.trust_class,
                       s.created_at, s.updated_at
                FROM mcp_servers s
                JOIN mcp_tools t ON t.server_id = s.server_id
                WHERE t.name = %s
                -- Tool names collide across servers (e.g. a user's hatch copy
                -- of a default package): prefer a healthy, most recently
                -- registered server instead of an arbitrary row (2026-09-25:
                -- two stale July entries on dead ports shared names with the
                -- default set).
                ORDER BY (s.status = 'healthy') DESC, s.updated_at DESC
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

        return self._run(_select)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Get registry statistics."""

        def _select(cur) -> dict[str, int]:
            cur.execute("SELECT COUNT(*) FROM mcp_servers")
            server_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM mcp_tools")
            tool_count = cur.fetchone()[0]
            return {"servers": server_count, "tools": tool_count}

        return self._run(_select)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _row_to_server(self, row) -> MCPServer:
        """Convert a database row to MCPServer."""
        auth_config = _json_column(row[8])
        metadata = _json_column(row[9])
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
            # Every server SELECT returns the same 16 columns in this order
            # (_SERVER_COLUMNS). 2026-09-25: the mapper read created_at from
            # the placement slot (list/get -> 500 on .isoformat()) and the
            # call-time lookup had no trust_class column, so D-19's
            # trusted-host-only refusal could never fire.
            placement=row[12] or "local",
            trust_class=row[13] or "local-safe",
            created_at=row[14],
            updated_at=row[15],
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
            tools.append(
                MCPTool(
                    tool_id=row[0],
                    server_id=row[1],
                    name=row[2],
                    description=row[3] or "",
                    input_schema=input_schema or {},
                    created_at=row[5],
                )
            )
        return tools
