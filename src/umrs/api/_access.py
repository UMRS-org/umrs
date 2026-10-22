"""API-side glue for row-carried access scopes (ADR-080).

The policy lives in ``umrs.shared.access_scopes`` (pure). This module binds
it to request identity: the caller's circles come from the shared
``umrs_db_pool``, resolved lazily — only when a ``circle:`` scope actually
decides — and memoized per caller instance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..shared.access_scopes import (
    AccessListError,
    ResourceCaller,
    access_admits,
    validate_access,
)
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)


def circles_resolver_for(raw_request: Any, user_id: str) -> Callable[[], list[str]]:
    """Lazy circle lookup bound to the request's app pool (0119 authority)."""

    def resolve() -> list[str]:
        app = getattr(raw_request, "app", None)
        pool = getattr(getattr(app, "state", None), "umrs_db_pool", None)
        if pool is None or not user_id:
            return []
        from .circles import circles_for_user

        conn = pool.getconn()
        try:
            return circles_for_user(conn, user_id)
        finally:
            pool.putconn(conn)

    return resolve


def circle_finder_for(raw_request: Any) -> Callable[[str], str | None]:
    """``find_active_circle`` on the request's pool. Unreadable (no pool, a
    database error) answers the ref itself: the share is stored unchecked,
    as before the check existed — a validation, not the access boundary."""

    def find(ref: str) -> str | None:
        app = getattr(raw_request, "app", None)
        pool = getattr(getattr(app, "state", None), "umrs_db_pool", None)
        if pool is None:
            return ref
        from .circles import find_active_circle

        conn = pool.getconn()
        try:
            return find_active_circle(conn, ref)
        except Exception:  # noqa: BLE001 - any database error keeps the old behaviour
            logger.warning("circle lookup failed for %s; share stored unchecked", ref)
            conn.rollback()
            return ref
        finally:
            pool.putconn(conn)

    return find


def resource_caller(
    raw_request: Any, *, user_id: str | None, project_id: str, is_admin: bool
) -> ResourceCaller:
    """A guard caller from verified identity parts (+ lazy circle resolver)."""
    return ResourceCaller(
        user_id=user_id or "",
        project_id=project_id or "",
        is_admin=is_admin,
        circles_resolver=circles_resolver_for(raw_request, user_id or ""),
        circle_finder=circle_finder_for(raw_request),
    )


def checked_access(
    raw: Any,
    caller: ResourceCaller,
    *,
    error: Callable[..., PlatformError],
    name: str = "forbidden",
) -> list[str]:
    """validate_access, or raise the family's typed 403 with the reason."""
    try:
        return validate_access(raw, caller)
    except AccessListError as exc:
        raise_typed(error(name, f"access: {exc}"), cause=None)


def graph_read_allowed(
    raw_request: Any,
    *,
    kind: str,
    project_id: str | None,
    owner_id: str,
    stored_access: Any,
    scope: Any,
) -> bool:
    """May the caller read one knowledge graph: the tier ladder OR the row's
    access list (ADR-080). The one rule for the registry routes, K3 graph
    selection and the proposal routes."""
    from umrs.knowledge.access_scope import graph_readable

    if graph_readable(
        kind=kind, project_id=project_id, owner_id=owner_id, caller=scope
    ):
        return True
    caller = resource_caller(
        raw_request,
        user_id=scope.user_id,
        project_id=scope.project_id,
        is_admin=scope.is_admin,
    )
    return access_admits(stored_access, caller)


def graph_meta_readable(raw_request: Any, graph: Any, scope: Any) -> bool:
    """``graph_read_allowed`` for a registry row (``KnowledgeGraphMeta``)."""
    return graph_read_allowed(
        raw_request,
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        stored_access=graph.access,
        scope=scope,
    )
