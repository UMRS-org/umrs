"""Per-call transactions for the sync Postgres stores (R-14 session hygiene).

Live bdrt 2026-09-27: MCPServerStore held one boot-leased connection, its
reads never ended their implicit transaction, R-14's
idle_in_transaction_session_timeout (10 min) killed the session, and every
later /umrs/mcp/* call answered 500. The research-session, companion-trust,
approval-inbox and audit stores had the same lease; the first two also
committed only on sqlite, so on Postgres their writes sat in a transaction
the reaper rolled back. :class:`PgCalls` is the contract of the fixed
``MCPServerStore._run`` (umrs/mcp/store.py), shared:

- ``pool`` (preferred): every call borrows its own connection and hands it
  back. ``conn``: a caller-owned connection that is never replaced.
- A write commits; a read ends only a transaction it opened itself, so a
  caller's open transaction on ``conn`` is left for the caller.
- A pooled session lost before the commit is discarded (``close=True``) and
  the work retried once on a fresh connection (nothing was committed); a
  loss during the commit is raised, never retried (the commit may have
  landed). Only a session psycopg2 has marked ``closed`` counts as lost.

sqlite3 connections (the stores' test / DB-less fallback) take the same
path; they carry no ``closed`` flag, so they are never treated as lost.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from typing import Any, TypeVar

import psycopg2
from psycopg2.extensions import TRANSACTION_STATUS_IDLE

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Exception classes a dead session raises (Postgres restart, network cut, the
# R-14 reaper). QueryCanceled, deadlock, serialization and lock-timeout errors
# subclass OperationalError on a live session, hence the ``closed`` check.
_CONNECTION_LOSS = (psycopg2.OperationalError, psycopg2.InterfaceError)
_ROLLBACK_ERRORS = (psycopg2.Error, sqlite3.Error)


def session_lost(conn: Any, exc: BaseException) -> bool:
    """True when ``exc`` came from a session psycopg2 has marked closed."""
    return isinstance(exc, _CONNECTION_LOSS) and bool(getattr(conn, "closed", 0))


class _SessionLost(Exception):
    """Internal: the first attempt's session died before any commit."""


class PgCalls:
    """Run each store call in a transaction of its own (module docstring)."""

    def __init__(
        self,
        *,
        pool: Any | None = None,
        conn: Any | None = None,
        label: str = "store",
    ) -> None:
        if conn is None and pool is None:
            raise ValueError("PgCalls needs a connection or a pool")
        self.conn = conn
        self._pool = pool
        self._label = label

    def run(self, work: Callable[[Any], _T], *, write: bool = False) -> _T:
        """Run ``work(cursor)``; a transaction the runner opened always ends."""
        try:
            conn, owned, result = self._attempt(work, retry_on_loss=True)
        except _SessionLost as lost:
            logger.warning(
                "%s connection lost (%s); retrying once on a fresh one",
                self._label,
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
        (only ``run`` sees it); every other failure is raised as-is.
        """
        conn = self._acquire()
        owned = self._owns_transaction(conn)
        try:
            return conn, owned, _on_cursor(conn, work)
        except BaseException as exc:
            lost = session_lost(conn, exc)
            self._release(conn, owned=owned)
            if lost and retry_on_loss and self._pool is not None:
                raise _SessionLost from exc
            raise

    def _acquire(self) -> Any:
        return self._pool.getconn() if self._pool is not None else self.conn

    def _owns_transaction(self, conn: Any) -> bool:
        """A pooled connection is the runner's own; a caller's connection only
        while the caller has no transaction open on it."""
        if self._pool is not None:
            return True
        status = getattr(conn, "get_transaction_status", None)
        if status is None:  # sqlite3
            return not conn.in_transaction
        return status() == TRANSACTION_STATUS_IDLE

    def _commit(self, conn: Any, owned: bool) -> None:
        try:
            conn.commit()
        except BaseException:
            self._release(conn, owned=owned)
            raise

    def _release(self, conn: Any, *, owned: bool) -> None:
        """End the runner's own transaction; hand a pooled conn back (closed
        when the session is gone, so the pool never reissues it)."""
        broken = bool(getattr(conn, "closed", 0))
        if owned and not broken:
            try:
                conn.rollback()
            except _ROLLBACK_ERRORS:
                broken = True
        if self._pool is not None:
            self._pool.putconn(conn, close=broken or bool(conn.closed))


def _on_cursor(conn: Any, work: Callable[[Any], _T]) -> _T:
    # sqlite3 cursors are no context managers; close explicitly.
    cur = conn.cursor()
    try:
        return work(cur)
    finally:
        cur.close()


__all__ = ["PgCalls", "session_lost"]
