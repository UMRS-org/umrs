"""Where scope-token revocations live (ADR-108 D-3, migration 0132).

Every front shares the database, so a level stopped on one front is refused
on all of them. A failed revocation read is ``scopes.unavailable`` (fail
closed), never "not revoked".
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from .tokens import ScopeError, revoked_by


class RevocationStore(Protocol):
    def revoke(self, user: str, project: str, prefix: str) -> None: ...

    def is_revoked(self, user: str, project: str, path: str) -> bool: ...


class MemoryRevocationStore:
    """For a process without a database (tests, stand-alone dev)."""

    def __init__(self) -> None:
        self._rows: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()

    def revoke(self, user: str, project: str, prefix: str) -> None:
        with self._lock:
            self._rows.add((user, project, prefix))

    def is_revoked(self, user: str, project: str, path: str) -> bool:
        with self._lock:
            rows = [p for u, pr, p in self._rows if (u, pr) == (user, project)]
        return any(revoked_by(path, prefix) for prefix in rows)


_PREFIXES_OF = """
SELECT 1 FROM scope_token_revocations
 WHERE user_id = %s AND project_id = %s
   AND (path_prefix = %s OR left(%s, length(path_prefix) + 1) = path_prefix || '/')
 LIMIT 1
"""

_INSERT = """
INSERT INTO scope_token_revocations (user_id, project_id, path_prefix)
VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
"""


class PgRevocationStore:
    """``scope_token_revocations`` through the app's psycopg pool."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def _run(self, sql: str, params: tuple[Any, ...], *, fetch: bool) -> Any:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone() if fetch else None
            conn.commit()
            return row
        except Exception as exc:  # every database failure is the same refusal (re-raised)
            conn.rollback()
            raise ScopeError("unavailable", "scope revocations could not be read") from exc
        finally:
            self._pool.putconn(conn)

    def revoke(self, user: str, project: str, prefix: str) -> None:
        self._run(_INSERT, (user, project, prefix), fetch=False)

    def is_revoked(self, user: str, project: str, path: str) -> bool:
        return self._run(_PREFIXES_OF, (user, project, path, path), fetch=True) is not None


_MEMORY = MemoryRevocationStore()


def store_for(app: Any) -> RevocationStore:
    """The app's database store, else the process memory store."""
    pool = getattr(getattr(app, "state", None), "umrs_db_pool", None)
    return PgRevocationStore(pool) if pool is not None else _MEMORY
