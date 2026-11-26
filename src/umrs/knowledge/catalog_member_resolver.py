"""Project-scoped member lookups for the catalog import (DEC-D39, S-D39a).

``SqlMemberResolver`` implements ``catalog_member_map.MemberResolver`` over
one psycopg2 connection, for ONE target project fixed at construction. Every
query is filtered to what that project may reference:

- graphs:  active ``knowledge_graphs`` rows of the target project only;
- skills:  ``scope='project'`` rows of the target project, or
           ``scope='system'`` rows; user-scoped skills are never mapped
           (they belong to a person, not to the catalog's project);
- MCP:     servers of the target project, else global ones
           (``project_id IS NULL``); the trust class is the local registry's.

A bundle can therefore never make an imported catalog point at another
project's graph, skill or server. A failed lookup reads as "no match" and
rolls the connection back, so the catalog insert that follows on the same
connection still runs.

Sync (psycopg2), like the catalog store it feeds. No ``umrs.api`` import.
"""

from __future__ import annotations

import logging
from typing import Any

from .graph_registry import default_graph_id

logger = logging.getLogger(__name__)

_GRAPH_SQL = (
    "SELECT graph_id FROM knowledge_graphs"
    " WHERE project_id = %s AND name = %s AND status = 'active'"
    " ORDER BY created_at, graph_id LIMIT 1"
)
_PROJECT_SKILL_SQL = (
    "SELECT skill_id FROM skills"
    " WHERE scope = 'project' AND project_id = %s AND name = %s AND version = %s"
    " ORDER BY skill_id LIMIT 1"
)
_SYSTEM_SKILL_SQL = (
    "SELECT skill_id FROM skills"
    " WHERE scope = 'system' AND name = %s AND version = %s"
    " ORDER BY skill_id LIMIT 1"
)
_MCP_SQL = (
    "SELECT server_id, trust_class FROM mcp_servers"
    " WHERE name = %s AND version = %s"
    " AND (project_id = %s OR project_id IS NULL)"
    " ORDER BY (project_id IS NULL), server_id LIMIT 1"
)


class SqlMemberResolver:
    """``MemberResolver`` for one target project over a psycopg2 conn."""

    def __init__(self, conn: Any, project_id: str) -> None:
        if not str(project_id or "").strip():
            raise ValueError("SqlMemberResolver needs a target project id")
        self._conn = conn
        self._project_id = str(project_id)

    def _first(self, sql: str, params: tuple[Any, ...]) -> tuple[Any, ...] | None:
        try:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            return cur.fetchone()
        except Exception as exc:  # noqa: BLE001 — a missing table is "no match"
            logger.warning("catalog member lookup failed: %s", exc)
            self._conn.rollback()
            return None

    def default_graph_id(self) -> str:
        return default_graph_id(self._project_id)

    def graph_id_by_name(self, name: str) -> str | None:
        row = self._first(_GRAPH_SQL, (self._project_id, name))
        return str(row[0]) if row else None

    def skill_id(self, scope: str, name: str, version: str) -> str | None:
        if scope == "project":
            row = self._first(_PROJECT_SKILL_SQL, (self._project_id, name, version))
        elif scope == "system":
            row = self._first(_SYSTEM_SKILL_SQL, (name, version))
        else:
            return None
        return str(row[0]) if row else None

    def mcp_server(self, name: str, version: str) -> tuple[str, str] | None:
        row = self._first(_MCP_SQL, (name, version, self._project_id))
        if not row:
            return None
        return str(row[0]), str(row[1] or "")
