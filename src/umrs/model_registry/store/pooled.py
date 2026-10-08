"""Borrow-per-call facade over :class:`ModelStore` for long-lived servers.

The HTTP sidecar used to hand ``ModelStore`` one connection leased at boot.
``ModelStore`` reads never end the implicit transaction psycopg2 opens, so the
leased session sat ``idle in transaction`` until the R-14 reaper
(``idle_in_transaction_session_timeout``, 10 min, ``umrs.shared.pg_settings``)
terminated it. The next request then raised ``psycopg2.OperationalError:
server closed the connection unexpectedly``. Live 2026-10-07/08: every
``Exception in ASGI application`` on rccs (11 in 24 h) and the bdrt
model-store lookups ended that way (ADR-089 Ops Watch, first findings). The
MCP and audit stores were fixed for the same class on 2026-09-27; this is the
model store's turn.

Each public call now borrows a pooled connection, runs one ``ModelStore``
method on it, rolls back whatever is still open (a no-op after the method's
own commit), and hands the connection back — closed if the session is gone, so
the pool never reissues it. A read that lost its session is retried once on a
fresh connection; a write never is, because its commit may have landed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import psycopg2

from .acl import ACLEnforcer
from .crud import ModelStore
from .path_validator import PathValidator

logger = logging.getLogger(__name__)

#: Methods that only read. Only these are retried after a lost session.
READ_METHODS = frozenset(
    {
        "check_policy_access",
        "get_model",
        "get_model_by_name",
        "get_model_versions",
        "get_policy",
        "get_policy_by_model_name",
        "list_models",
        "list_models_with_reasons",
        "list_policies",
        "model_exists",
        "resolve_model_path",
    }
)

_SESSION_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


class _SessionLost(Exception):
    """The borrowed session died under a read; retry on a fresh one."""


class PooledModelStore:
    """Same public methods as :class:`ModelStore`; one pooled connection per call."""

    def __init__(
        self,
        pool: Any,
        *,
        path_validator: PathValidator | None = None,
        acl_enforcer: ACLEnforcer | None = None,
    ):
        self._pool = pool
        self.path_validator = path_validator or PathValidator()
        self.acl = acl_enforcer or ACLEnforcer()

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_") or not callable(getattr(ModelStore, name, None)):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, args, kwargs)

        call.__name__ = name
        return call

    def _call(self, name: str, args: tuple, kwargs: dict) -> Any:
        retry = name in READ_METHODS
        try:
            return self._once(name, args, kwargs, retry=retry)
        except _SessionLost as lost:
            logger.warning(
                "model store session lost in %s (%s); retrying once on a fresh connection",
                name,
                lost.__cause__,
            )
            return self._once(name, args, kwargs, retry=False)

    def _once(self, name: str, args: tuple, kwargs: dict, *, retry: bool) -> Any:
        conn = self._pool.getconn()
        try:
            store = ModelStore(
                conn, path_validator=self.path_validator, acl_enforcer=self.acl
            )
            return getattr(store, name)(*args, **kwargs)
        except _SESSION_ERRORS as exc:
            if retry and conn.closed:
                raise _SessionLost from exc
            raise
        finally:
            self._release(conn)

    def _release(self, conn: Any) -> None:
        release_connection(self._pool, conn)


def release_connection(pool: Any, conn: Any) -> None:
    """Hand a borrowed connection back with no transaction left open.

    Rolls back whatever is still open (a no-op after a commit) and closes the
    connection instead of pooling it when the session is gone. Returning it
    ``idle in transaction`` is what let the R-14 reaper kill pooled sessions
    under the next borrower.
    """
    broken = bool(conn.closed)
    if not broken:
        try:
            conn.rollback()
        except psycopg2.Error:
            broken = True
    pool.putconn(conn, close=broken or bool(conn.closed))
