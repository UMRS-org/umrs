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


_SESSION_COLUMNS = [
    "session_id",
    "project_id",
    "title",
    "research_goal",
    "status",
    "authoritative_runtime",
    "metadata",
    "created_at",
    "updated_at",
    "archived_at",
]


def _row_get(row: Any, key: str, columns: list[str], default: Any = None) -> Any:
    """Access one column from either a mapping row (sqlite3.Row/dict_row)
    or a plain tuple from a default psycopg cursor, by position."""
    try:
        return row[key]
    except TypeError:
        if isinstance(row, tuple) and key in columns:
            try:
                return row[columns.index(key)]
            except IndexError:
                return default
        return default
    except (IndexError, KeyError):
        return default


def _coerce_metadata(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw:
        return json.loads(raw)
    return {}


def _row_to_session(row: Any) -> ResearchSession:
    def _get(key: str, default: Any = None) -> Any:
        return _row_get(row, key, _SESSION_COLUMNS, default)

    return ResearchSession(
        session_id=_get("session_id", ""),
        project_id=_get("project_id", ""),
        title=_get("title") or "",
        research_goal=_get("research_goal") or "",
        status=_get("status") or "active",
        authoritative_runtime=_get("authoritative_runtime") or "umrs",
        metadata=_coerce_metadata(_get("metadata") or "{}"),
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

    def archive_aged_sessions(
        self,
        *,
        max_days: int,
        trust_store: Any,
    ) -> list[str]:
        """Archive non-archived sessions older than max_days via the ADR-035
        item-8 cascade (ONE transaction per session — archive + session-scope
        trust revoke + companion_session_archives tombstone).

        Rules:
        - max_days <= 0 → no-op (nothing archives without an explicit policy).
        - Pinned rows (metadata.pinned = true) are exempt.
        - Already-archived rows are good, never re-cascaded (idempotent side).
        """
        if max_days <= 0:
            return []
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cutoff_sql = "datetime('now', ?)" if self._sqlite else "now() - (%s * interval '1 day')"
            cutoff_param = f"-{max_days} days" if self._sqlite else max_days
            cur.execute(
                self._q(
                    "SELECT session_id, metadata FROM research_sessions "
                    "WHERE status != 'archived' "
                    f"AND updated_at < {cutoff_sql}"
                ),
                (cutoff_param,) if not self._sqlite else (cutoff_param,),
            )
            rows = cur.fetchall()
        finally:
            self._return(conn)
        ids: list[str] = []
        for row in rows:
            session_id = row[0]
            metadata_raw = row[1] or ""
            pinned = False
            if metadata_raw:
                import json as _json

                try:
                    pinned = bool(_json.loads(metadata_raw).get("pinned"))
                except (ValueError, TypeError):
                    pinned = False
            if pinned:
                continue
            session = self.archive_with_trust_cascade(session_id, trust_store)
            if session is not None:
                ids.append(session_id)
        return ids

    def archive_with_trust_cascade(
        self, session_id: str, trust_store: Any
    ) -> ResearchSession | None:
        """Archive a session AND revoke its session-scope trust rows in ONE
        transaction (borrow → update → cascade on the same conn → commit).

        ADR-035 item-8: lifecycle consequences of a server-owned object are
        enforced where the object lives. A crash between archive and revoke
        would leave live grants on a dead session — exactly the state this
        exists to prevent, hence the shared commit.
        """
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "UPDATE research_sessions SET status = %s, "
                    "archived_at = now(), updated_at = now() "
                    "WHERE session_id = %s"
                ),
                ("archived", session_id),
            )
            if trust_store is not None:
                # Tombstone INSIDE the same transaction — check() consults
                # companion_session_archives, so every archive channel
                # (bridge/CLI/ACP endpoint + this PATCH hook) must write it.
                # ON CONFLICT no-op: channels may race; first tombstone wins.
                tomb_sql = (
                    "INSERT OR IGNORE INTO companion_session_archives "
                    "(session_id, archived_by) VALUES (?, ?)"
                    if self._sqlite
                    else "INSERT INTO companion_session_archives (session_id, archived_by) "
                    "VALUES (%s, %s) ON CONFLICT (session_id) DO NOTHING"
                )
                cur.execute(tomb_sql, (session_id, "research-sessions.patch"))
                trust_store.revoke_session_scope_conn(conn, session_id)
            if self._sqlite:
                self._commit(conn)
            else:
                conn.commit()
        except Exception:
            # One transaction or not at all: anything between update and
            # revoke must not leave a live grant on a dead session (and
            # vice versa). Both DBs keep a pending transaction open on
            # raise without an explicit rollback.
            conn.rollback()
            raise
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
                self._q("UPDATE research_sessions SET updated_at = now() WHERE session_id = %s"),
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
                    "DELETE FROM research_session_loops WHERE session_id = %s AND loop_id = %s"
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
                "SELECT * FROM research_session_loops WHERE session_id = %s ORDER BY attached_at"
            ),
            (session_id,),
        )
        return [_row_to_link(r) for r in cur.fetchall()]


_LINK_COLUMNS = ["session_id", "loop_id", "venue", "loop_role", "attached_at"]


def _row_to_link(row: Any) -> SessionLoopLink:
    def _get(key: str, default: Any = "") -> Any:
        return _row_get(row, key, _LINK_COLUMNS, default)

    attached = _get("attached_at")
    return SessionLoopLink(
        session_id=_get("session_id"),
        loop_id=_get("loop_id"),
        venue=_get("venue") or "umrs",
        loop_role=_get("loop_role") or "auto",
        attached_at=str(attached) if attached else "",
    )
