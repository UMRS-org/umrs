"""Test double for psycopg2 ThreadedConnectionPool-style pools.

Contract mirrored from production usage:
  pool.getconn() -> conn; pool.putconn(conn)
  conn used as context manager (with conn:)
  conn.cursor(cursor_factory=...) -> cursor context manager
  cursor.execute(sql, params); cursor.fetchone(); cursor.fetchall()

``script`` is a list of per-execute results (one item consumed per execute):
  - list  → fetchone() pops its first element, fetchall() returns it whole
  - dict  → fetchone() returns it
  - None  → fetchone() returns None, fetchall() returns []

Every executed (sql, params) pair is recorded in ``conn.record`` for
SQL-shape assertions.
"""

from __future__ import annotations

from typing import Any


class FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn
        self._current: Any = None

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        normalized = " ".join(str(sql).split())
        self._conn.record.append((normalized, params))
        self._current = self._conn.script.pop(0) if self._conn.script else None

    def fetchone(self) -> Any:
        if isinstance(self._current, list):
            return self._current.pop(0) if self._current else None
        result, self._current = self._current, None
        return result

    def fetchall(self) -> list[Any]:
        if isinstance(self._current, list):
            result, self._current = self._current, []
            return result
        return [self._current] if self._current is not None else []


class FakeConn:
    def __init__(self, script: list[Any] | None = None) -> None:
        self.script = list(script or [])
        self.record: list[tuple[str, Any]] = []

    def __enter__(self) -> FakeConn:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def cursor(self, cursor_factory: Any = None) -> FakeCursor:
        del cursor_factory  # psycopg2 signature parity; fakes never use factories
        return FakeCursor(self)


class FakePool:
    """Pool returning a single shared FakeConn (tests are single-threaded)."""

    def __init__(self, script: list[Any] | None = None) -> None:
        self.conn = FakeConn(script)

    @property
    def record(self) -> list[tuple[str, Any]]:
        return self.conn.record

    def getconn(self) -> FakeConn:
        return self.conn

    def putconn(self, conn: FakeConn) -> None:  # noqa: ARG002
        return None
