"""Research-session store — psycopg production store with sqlite3 fallback.

The sync facade mirrors the jobs/MCP stores: HTTP handlers call sync methods;
tests run without a Postgres pool thanks to the sqlite3 fallback (created in a
temp file, or shared memory when ``path=":memory:"``).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .schema import CREATE_LOOPS_TABLE, CREATE_SESSIONS_TABLE


def _sqliteify(sql: str) -> str:
    """Translate the Postgres DDL to sqlite3 (JSONB→TEXT, strip COMMENTs, etc.)."""
    import re

    # sqlite has no COMMENT ON TABLE syntax — drop the whole statement.
    sql = re.sub(r"COMMENT ON TABLE[^;]*;", "", sql, flags=re.S)
    return (
        sql.replace("JSONB", "TEXT")
        .replace("TIMESTAMPTZ", "TEXT")
        .replace("DEFAULT now()", "DEFAULT (datetime('now'))")
    )


@dataclass
class SessionLoopLink:
    session_id: str
    loop_id: str
    venue: str = "umrs"
    loop_role: str = "auto"
    attached_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "loop_id": self.loop_id,
            "venue": self.venue,
            "loop_role": self.loop_role,
            "attached_at": self.attached_at,
        }


@dataclass
class ResearchSession:
    session_id: str
    project_id: str
    title: str = ""
    research_goal: str = ""
    status: str = "active"
    authoritative_runtime: str = "umrs"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    archived_at: str | None = None
    loops: list[SessionLoopLink] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "title": self.title,
            "research_goal": self.research_goal,
            "status": self.status,
            "authoritative_runtime": self.authoritative_runtime,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived_at": self.archived_at,
            "loops": [link.to_dict() for link in self.loops],
        }


def _row_to_session(row: Any) -> ResearchSession:
    def _get(key: str, default: Any = None) -> Any:
        try:
            return row[key]
        except (IndexError, KeyError):
            return default
        except TypeError:
            return default

    meta = _get("metadata") or "{}"
    if isinstance(meta, dict):
        metadata = meta
    else:
        metadata = json.loads(meta or "{}")
    return ResearchSession(
        session_id=_get("session_id", ""),
        project_id=_get("project_id", ""),
        title=_get("title") or "",
        research_goal=_get("research_goal") or "",
        status=_get("status") or "active",
        authoritative_runtime=_get("authoritative_runtime") or "umrs",
        metadata=metadata,
        created_at=str(_get("created_at") or ""),
        updated_at=str(_get("updated_at") or ""),
        archived_at=str(_get("archived_at")) if _get("archived_at") else None,
        loops=[],
    )


class SessionStore:
    """Sync store over a DB-API connection (psycopg or sqlite3).

    ``conn`` may be a plain connection or a connection pool with
    ``getconn``/``putconn`` (psycopg.pool-like). Placeholders are written
    with ``%s`` — sqlite needs ``?`` so we translate on the sqlite path.
    """

    def __init__(self, conn: Any | None = None, *, path: str | None = None) -> None:
        self._pool = conn if hasattr(conn, "getconn") else None
        if conn is None and path is not None:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._sqlite = True
        elif conn is not None and not hasattr(conn, "getconn"):
            self._conn = conn
            self._sqlite = isinstance(conn, sqlite3.Connection)
        elif self._pool is not None:
            self._conn = None
            self._sqlite = False
        else:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._sqlite = True

        if self._sqlite and self._conn is not None:
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_sqliteify(CREATE_SESSIONS_TABLE))
            self._conn.executescript(_sqliteify(CREATE_LOOPS_TABLE))

    # -- low-level plumbing ------------------------------------------------ #

    def _borrow(self):
        return self._pool.getconn() if self._pool is not None else self._conn

    def _return(self, conn) -> None:
        if self._pool is not None:
            self._pool.putconn(conn)

    def _q(self, sql: str) -> str:
        if not self._sqlite:
            return sql
        return sql.replace("%s", "?").replace("now()", "datetime('now')")

    def _commit(self, conn) -> None:
        if self._sqlite:
            conn.commit()  # psycopg connections commit via callers in other stores

    # -- CRUD --------------------------------------------------------------- #

    def create(self, session: ResearchSession) -> ResearchSession:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    """INSERT INTO research_sessions
                       (session_id, project_id, title, research_goal, status,
                        authoritative_runtime, metadata)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)"""
                ),
                (
                    session.session_id,
                    session.project_id,
                    session.title,
                    session.research_goal,
                    session.status,
                    session.authoritative_runtime,
                    json.dumps(session.metadata),
                ),
            )
            for link in session.loops:
                self._attach_copy(cur, link)
            self._commit(conn)
            return session
        finally:
            self._return(conn)

    def get(self, session_id: str) -> ResearchSession | None:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q("SELECT * FROM research_sessions WHERE session_id = %s"),
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            session = _row_to_session(row)
            session.loops = self._list_loops_on(cur, session_id)
            return session
        finally:
            self._return(conn)

    def list(
        self, project_id: str | None = None, status: str | None = None
    ) -> list[ResearchSession]:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            sql = "SELECT * FROM research_sessions"
            clauses: list[str] = []
            params: list[Any] = []
            if project_id:
                clauses.append("project_id = %s")
                params.append(project_id)
            if status:
                clauses.append("status = %s")
                params.append(status)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY updated_at DESC"
            cur.execute(self._q(sql), tuple(params))
            rows = cur.fetchall()
            return [_row_to_session(r) for r in rows]
        finally:
            self._return(conn)

    def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchSession | None:
        current = self.get(session_id)
        if current is None:
            return None
        conn = self._borrow()
        try:
            cur = conn.cursor()
            if title is not None:
                cur.execute(
                    self._q(
                        "UPDATE research_sessions SET title = %s, "
                        "updated_at = now() WHERE session_id = %s"
                    ),
                    (title, session_id),
                )
            if status is not None:
                archived = "now()" if status == "archived" else "NULL"
                cur.execute(
                    self._q(
                        f"UPDATE research_sessions SET status = %s, "
                        f"archived_at = {archived}, updated_at = now() "
                        f"WHERE session_id = %s"
                    ),
                    (status, session_id),
                )
            if metadata is not None:
                cur.execute(
                    self._q(
                        "UPDATE research_sessions SET metadata = %s, "
                        "updated_at = now() WHERE session_id = %s"
                    ),
                    (json.dumps(metadata), session_id),
                )
            self._commit(conn)
        finally:
            self._return(conn)
        return self.get(session_id)

    def attach_loop(self, link: SessionLoopLink) -> None:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    """INSERT INTO research_session_loops
                       (session_id, loop_id, venue, loop_role)
                       VALUES (%s,%s,%s,%s)
                       ON CONFLICT (session_id, loop_id) DO NOTHING"""
                ),
                (link.session_id, link.loop_id, link.venue, link.loop_role),
            )
            cur.execute(
                self._q(
                    "UPDATE research_sessions SET updated_at = now() "
                    "WHERE session_id = %s"
                ),
                (link.session_id,),
            )
            self._commit(conn)
        finally:
            self._return(conn)

    def detach_loop(self, session_id: str, loop_id: str) -> None:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "DELETE FROM research_session_loops "
                    "WHERE session_id = %s AND loop_id = %s"
                ),
                (session_id, loop_id),
            )
            self._commit(conn)
        finally:
            self._return(conn)

    def list_loops(self, session_id: str) -> list[SessionLoopLink]:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            return self._list_loops_on(cur, session_id)
        finally:
            self._return(conn)

    def session_for_loop(self, loop_id: str) -> SessionLoopLink | None:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "SELECT * FROM research_session_loops WHERE loop_id = %s "
                    "ORDER BY attached_at DESC LIMIT 1"
                ),
                (loop_id,),
            )
            row = cur.fetchone()
            return _row_to_link(row) if row else None
        finally:
            self._return(conn)

    # -- helpers ------------------------------------------------------------ #

    def _attach_copy(self, cur, link: SessionLoopLink) -> None:
        cur.execute(
            self._q(
                """INSERT INTO research_session_loops
                   (session_id, loop_id, venue, loop_role)
                   VALUES (%s,%s,%s,%s)
                   ON CONFLICT (session_id, loop_id) DO NOTHING"""
            ),
            (link.session_id, link.loop_id, link.venue, link.loop_role),
        )

    def _list_loops_on(self, cur, session_id: str) -> list[SessionLoopLink]:
        cur.execute(
            self._q(
                "SELECT * FROM research_session_loops WHERE session_id = %s "
                "ORDER BY attached_at"
            ),
            (session_id,),
        )
        return [_row_to_link(r) for r in cur.fetchall()]


def _row_to_link(row: Any) -> SessionLoopLink:
    def _get(key: str, default: Any = "") -> Any:
        try:
            return row[key]
        except (IndexError, KeyError, TypeError):
            return default

    attached = _get("attached_at")
    return SessionLoopLink(
        session_id=_get("session_id"),
        loop_id=_get("loop_id"),
        venue=_get("venue") or "umrs",
        loop_role=_get("loop_role") or "auto",
        attached_at=str(attached) if attached else "",
    )
