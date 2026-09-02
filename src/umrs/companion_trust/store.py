"""Companion trust allowlist store — psycopg production store, sqlite fallback.

Sync facade mirrors the research_session/jobs stores: HTTP handlers call sync
methods; tests run without a Postgres pool thanks to the sqlite3 fallback
(shared memory when no ``conn`` is given).

Semantics:
- one row per (user_id, scope, tool_name) — upsert on grant refreshes expiry
- grant TTL is explicit (``ttl_sec``, default 30 min mirroring the client-side
  R4_PERMISSION_APPROVAL_TTL default — the environment contract for TTLs
  stays with the companion; the store just enforces expiry)
- ``check`` returns the row ONLY when unexpired; expired rows are deleted on
  read (lazy GC) so the table stays small without a sweeper
- ``scope`` is opaque here (global / project:<id> / session:<id>) — the API
  layer resolves what a caller may read or write
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from umrs.shared.scope_grammar import parse_lenient

from .schema import CREATE_TRUST_TABLE

DEFAULT_TTL_SEC = 1800  # 30 min — mirrors R4_PERMISSION_APPROVAL_TTL default


def _norm_scope(raw: str | None) -> str:
    """Normalize legacy scope aliases ('global' → 'system') at the store
    boundary so old rows stay readable while new writes are canonical."""
    try:
        return str(parse_lenient(raw))
    except Exception:
        return "system"


def _sqliteify(sql: str) -> str:
    """Translate the Postgres DDL to sqlite3."""
    import re

    sql = re.sub(r"COMMENT ON TABLE[^;]*;", "", sql, flags=re.S)
    return (
        sql.replace("JSONB", "TEXT")
        .replace("TIMESTAMPTZ", "TEXT")
        .replace("DEFAULT now()", "DEFAULT (datetime('now'))")
    )


@dataclass(frozen=True)
class TrustRow:
    """One allowlist record."""

    user_id: str
    scope: str
    tool_name: str
    expires_at: str
    created_at: str = ""
    created_by: str = ""

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "scope": self.scope,
            "tool_name": self.tool_name,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


_TRUST_COLUMNS = [
    "user_id",
    "scope",
    "tool_name",
    "expires_at",
    "created_at",
    "created_by",
]


def _row_get(row, key: str, default=None):
    """Access one column from a mapping row (sqlite3.Row) or tuple position."""
    try:
        return row[key]
    except TypeError:
        if isinstance(row, tuple) and key in _TRUST_COLUMNS:
            try:
                return row[_TRUST_COLUMNS.index(key)]
            except IndexError:
                return default
        return default
    except (IndexError, KeyError):
        return default


def _row_to_trust(row) -> TrustRow:
    def _get(key, default=None):
        return _row_get(row, key, default)

    return TrustRow(
        user_id=_get("user_id", ""),
        scope=_norm_scope(_get("scope", "system")),
        tool_name=_get("tool_name", ""),
        expires_at=str(_get("expires_at") or ""),
        created_at=str(_get("created_at") or ""),
        created_by=_get("created_by") or "",
    )


class TrustAllowlistStore:
    """Sync store over a DB-API connection (psycopg or sqlite3).

    ``conn`` may be a plain connection or a pool with ``getconn``/``putconn``.
    Placeholders are written ``%s`` — sqlite needs ``?`` (translated in ``_q``).
    """

    def __init__(self, conn=None, *, path: str | None = None) -> None:
        self._pool = conn if hasattr(conn, "getconn") else None
        if conn is None and path is not None:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._sqlite = True
        elif conn is not None and self._pool is None:
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
            self._conn.executescript(_sqliteify(CREATE_TRUST_TABLE))

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
            conn.commit()  # psycopg connections commit via pool owner

    # -- CRUD ---------------------------------------------------------------- #

    def grant(
        self,
        user_id: str,
        tool_name: str,
        scope: str = "system",
        ttl_sec: int = DEFAULT_TTL_SEC,
        created_by: str = "",
    ) -> TrustRow:
        """Upsert an allowlist row, refreshing its expiry."""
        scope = _norm_scope(scope)
        conn = self._borrow()
        try:
            cur = conn.cursor()
            if self._sqlite:
                cur.execute(
                    """INSERT INTO companion_trust_allowlist
                       (user_id, scope, tool_name, expires_at, created_by)
                       VALUES (?, ?, ?, datetime('now', ?), ?)
                       ON CONFLICT (user_id, scope, tool_name)
                       DO UPDATE SET expires_at = datetime('now', ?)""",
                    (
                        user_id,
                        scope,
                        tool_name,
                        f"+{ttl_sec} seconds",
                        created_by,
                        f"+{ttl_sec} seconds",
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO companion_trust_allowlist
                       (user_id, scope, tool_name, expires_at, created_by)
                       VALUES (%s, %s, %s, now() + (%s || ' seconds')::interval, %s)
                       ON CONFLICT (user_id, scope, tool_name)
                       DO UPDATE SET expires_at = now() + (%s || ' seconds')::interval""",
                    (user_id, scope, tool_name, str(ttl_sec), created_by, str(ttl_sec)),
                )
            self._commit(conn)
            return TrustRow(
                user_id=user_id,
                scope=scope,
                tool_name=tool_name,
                expires_at="",  # caller re-reads via check() if it needs the stamp
                created_by=created_by,
            )
        finally:
            self._return(conn)

    def check(self, user_id: str, tool_name: str, scope: str = "system") -> TrustRow | None:
        """Return the row when present AND unexpired; lazy-GC expired rows."""
        scope = _norm_scope(scope)
        row = self._fetch(user_id, scope, tool_name)
        if row is None:
            return None
        if not self._unexpired(row):
            self.revoke(user_id, tool_name, scope)
            return None
        return row

    def list(self, user_id: str, scope: str | None = None) -> list[TrustRow]:
        """Unexpired rows for the user, optionally scope-filtered."""
        if scope is not None:
            scope = _norm_scope(scope)
        conn = self._borrow()
        try:
            cur = conn.cursor()
            sql = "SELECT * FROM companion_trust_allowlist WHERE user_id = %s"
            params: tuple = (user_id,)
            if scope is not None:
                sql += " AND scope = %s"
                params = (user_id, scope)
            sql += " ORDER BY scope, tool_name"
            cur.execute(self._q(sql), params)
            rows = [self._row_guard(r) for r in cur.fetchall()]
            return [r for r in rows if r is not None]
        finally:
            self._return(conn)

    def revoke(self, user_id: str, tool_name: str, scope: str = "system") -> bool:
        """Delete a row. Returns True when a row existed."""
        scope = _norm_scope(scope)
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "DELETE FROM companion_trust_allowlist "
                    "WHERE user_id = %s AND scope = %s AND tool_name = %s"
                ),
                (user_id, scope, tool_name),
            )
            self._commit(conn)
            return (cur.rowcount or 0) > 0
        finally:
            self._return(conn)

    # -- internals ---------------------------------------------------------- #

    def _fetch(self, user_id: str, scope: str, tool_name: str) -> TrustRow | None:
        conn = self._borrow()
        try:
            cur = conn.cursor()
            cur.execute(
                self._q(
                    "SELECT * FROM companion_trust_allowlist "
                    "WHERE user_id = %s AND scope = %s AND tool_name = %s"
                ),
                (user_id, scope, tool_name),
            )
            row = cur.fetchone()
            return _row_to_trust(row) if row is not None else None
        finally:
            self._return(conn)

    def _unexpired(self, row: TrustRow) -> bool:
        """Expiry check against the DB clock (authoritative, no client skew)."""
        conn = self._borrow()
        try:
            cur = conn.cursor()
            now_fn = "datetime('now')" if self._sqlite else "now()"
            cur.execute(
                self._q(
                    f"SELECT 1 FROM companion_trust_allowlist "
                    f"WHERE user_id = %s AND scope = %s AND tool_name = %s "
                    f"AND expires_at > {now_fn}"
                ),
                (row.user_id, row.scope, row.tool_name),
            )
            return cur.fetchone() is not None
        finally:
            self._return(conn)

    def _row_guard(self, raw) -> TrustRow | None:
        """list-row filter: keep only unexpired; drop expired eagerly."""
        row = _row_to_trust(raw)
        if not self._unexpired(row):
            self.revoke(row.user_id, row.tool_name, row.scope)
            return None
        return row
