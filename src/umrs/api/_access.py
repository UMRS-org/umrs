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


def resource_caller(
    raw_request: Any, *, user_id: str | None, project_id: str, is_admin: bool
) -> ResourceCaller:
    """A guard caller from verified identity parts (+ lazy circle resolver)."""
    return ResourceCaller(
        user_id=user_id or "",
        project_id=project_id or "",
        is_admin=is_admin,
        circles_resolver=circles_resolver_for(raw_request, user_id or ""),
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
