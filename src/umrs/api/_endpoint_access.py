"""API-side glue for endpoint access scopes (ADR-078).

The policy lives in ``umrs.orchestrator.endpoint_access`` (pure). This module
binds it to request identity: the caller's groups come from the shared
``umrs_db_pool`` (looked up lazily, only for group scopes) and roster recipes
may carry a default ``access`` list.
"""

from __future__ import annotations

import logging
from typing import Any

from ..orchestrator.endpoint_access import (
    AccessError,
    EndpointCaller,
    validate_access,
)
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)


def _groups_resolver(raw_request: Any, user_id: str):
    def resolve() -> list[str]:
        app = getattr(raw_request, "app", None)
        pool = getattr(getattr(app, "state", None), "umrs_db_pool", None)
        if pool is None or not user_id:
            return []
        from .groups import groups_for_user

        conn = pool.getconn()
        try:
            return groups_for_user(conn, user_id)
        finally:
            pool.putconn(conn)

    return resolve


def caller_for(
    raw_request: Any, *, user_id: str | None, project_id: str, is_admin: bool
) -> EndpointCaller:
    return EndpointCaller(
        user_id=user_id or "",
        project_id=project_id or "",
        is_admin=is_admin,
        groups_resolver=_groups_resolver(raw_request, user_id or ""),
    )


def _roster_access(model: str, variant: str | None) -> Any:
    from ..orchestrator.provision_roster import resolve_model_provision_defaults

    defaults = resolve_model_provision_defaults(model, variant) or {}
    return (defaults.get("_full_entry") or {}).get("access")


def provision_access(request: Any, caller: EndpointCaller) -> list[str]:
    """The access list a provision records.

    An explicit ``request.access`` must be grantable (typed 403 otherwise).
    Without one, the roster recipe's ``access`` applies when the caller may
    grant it; otherwise the endpoint stays with its owning project.
    """
    if request.access is not None:
        try:
            return validate_access(request.access, caller)
        except AccessError as exc:
            raise_typed(
                PlatformError.endpoints("forbidden", f"access: {exc}"), cause=None
            )
    recipe = _roster_access(request.model, request.variant)
    if recipe is None:
        return []
    try:
        return validate_access(recipe, caller)
    except AccessError as exc:
        logger.info(
            "roster access for %s not applied for this caller: %s", request.model, exc
        )
        return []


def checked_access(raw: Any, caller: EndpointCaller) -> list[str]:
    """Validate an access update (typed 403/400 on refusal)."""
    try:
        return validate_access(raw, caller)
    except AccessError as exc:
        raise_typed(PlatformError.endpoints("forbidden", f"access: {exc}"), cause=None)
    return []  # pragma: no cover - raise_typed never returns
