"""Durable tool-approval inbox routes (U3 S1 item 8, ADR-036 (a), WP-05).

Pending tool-grain intents and their decisions are core: research loops
park on them whether or not the r4 companion is installed. Moved from
``companion`` with the same paths (``/umrs/companion/approvals/...``,
mounted through the companion router until U3 S4/S5 mounts it directly),
the same ownership rules and the same typed errors.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from .identity import strict_user_header
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

class ApprovalDecisionRequest(BaseModel):
    decision: str  # once | always | no


class ApprovalsPendingResponse(BaseModel):
    pending: list[dict[str, Any]] = []
    # Where the inbox lives: postgres (durable) | sqlite | process (ADR-036 (a)).
    durability: str = "process"


# ── pending approvals — durable inbox (ADR-036 (a), WP-05) ───────────────────


async def list_pending_approvals(
    request: Request,
    x_user_id: str = Depends(strict_user_header),
) -> ApprovalsPendingResponse:
    """Pending tool-grain intents visible to the caller.

    Owner-scoped: rows of loops the caller owns plus ownerless rows (loops
    started without a gateway identity). ``?loop_id=`` narrows to one loop;
    ``?registry_key=`` keeps the venue-ownership check of the m5 contract.
    """
    registry_key = request.query_params.get("registry_key")
    if registry_key:
        # The m5 venue-ownership check reads the companion worker registry,
        # which D6 retires in U3 S4 (no extension hook carries it: the six
        # hooks do not, by design). TODO(U3-S4): drop with the registry.
        from .companion import _enforce_ownership, _managers

        managers, _ = _managers(request)
        row = managers.get(registry_key)
        _enforce_ownership(row or {}, x_user_id)
    from umrs.knowledge import tool_approvals

    rows = tool_approvals.pending(
        loop_id=request.query_params.get("loop_id") or None,
        user_id=x_user_id,
    )
    return ApprovalsPendingResponse(
        pending=_visible_approvals(rows, request),
        durability=tool_approvals.durability(),
    )


def _loop_project_and_owner(loop_id: str) -> tuple[str, str]:
    """A server-hosted loop's project and owner ('' for what is unknown)."""
    from umrs.api import loop as loop_api

    loop = loop_api.get_loop_store().get(loop_id) if loop_id else None
    metadata = getattr(loop, "metadata", None) or {}
    return (
        str(getattr(loop, "project_id", "") or ""),
        str(metadata.get("owner_user_id") or ""),
    )


def _ownerless_approval_open(record: dict, scope: Any) -> bool:
    """An ownerless intent follows its loop's own access rule: an owned loop's
    owner (or an admin); an ownerless loop's project writers. It was listed
    to and decidable by anyone (H-1, 2026-10-17); a project writer could
    decide an owned loop's ownerless intent (review 2026-10-18)."""
    from umrs.knowledge.access_scope import project_writable

    if scope.is_admin or scope.anonymous:
        return True
    project, owner = _loop_project_and_owner(str(record.get("loop_id") or ""))
    if owner:
        return scope.user_id == owner
    return bool(project) and project_writable(project, scope)


def _visible_approvals(rows: list[dict], request: Request) -> list[dict]:
    """Owned rows as listed; ownerless rows only for their loop's project."""
    if all(row.get("user_id") for row in rows):
        return rows
    from ._scope_identity import request_caller_scope

    scope = request_caller_scope(request)
    return [r for r in rows if r.get("user_id") or _ownerless_approval_open(r, scope)]


def _enforce_approval_ownership(record: dict, x_user_id: str, request: Request) -> None:
    """An approval intent is decidable only by the loop owner; an ownerless
    one by its loop's project. Anyone else's reads as not found, as an
    unknown id does (H-1, 2026-10-18: a non-owner got 401 "approval
    ownership mismatch", which showed that the approval exists)."""
    owner = str(record.get("user_id") or "")
    if owner:
        allowed = owner == x_user_id
    else:
        from ._scope_identity import request_caller_scope

        allowed = _ownerless_approval_open(record, request_caller_scope(request))
    if not allowed:
        raise_typed(
            PlatformError.companion(
                "not_found", f"approval not found: {record.get('id', '')}"
            )
        )


def _flip_loop_to_auto(loop_id: str) -> None:
    try:
        from umrs.api.loop import get_loop_store

        store = get_loop_store()
        loop = store.get(loop_id)
        if loop is not None:
            loop.metadata["mode"] = "auto"
            store.save(loop)
    except Exception:
        logger.warning(
            "always-decision mode flip failed for loop %s", loop_id, exc_info=True
        )


def _notify_approval_listeners(record: dict, decision: str, x_user_id: str) -> None:
    """U3 S3e: extension approval listeners (r4, D7: ``always`` mints a
    ``session:<loop_id>`` trust row). Best effort and isolated: the decision
    itself is already durable."""
    from umrs.extensions import get_extension_context

    for listener in get_extension_context().approval_listeners:
        try:
            listener(record, decision, x_user_id)
        except Exception:
            logger.warning(
                "approval listener %s failed for %s",
                getattr(listener, "__name__", listener),
                record.get("id"),
                exc_info=True,
            )


def _apply_always_decision(record: dict, x_user_id: str) -> None:
    """``always``: flip the loop to auto (core, D1); listeners add the rest.

    Best effort (ADR-036 §3): the decision itself is already durable.
    """
    loop_id = str(record.get("loop_id") or "")
    if loop_id:
        _flip_loop_to_auto(loop_id)


async def record_approval_decision(
    approval_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
    x_user_id: str = Depends(strict_user_header),
) -> dict:
    """Decide one pending intent (durable inbox; loop drivers park on these).

    ``once``/``always`` unblock the parked loop; ``no`` makes its gate fail.
    ``always`` additionally flips the owning loop into ``mode=auto`` and
    mints a ``session:<loop_id>`` trust row. 404 unknown or not the
    caller's, 409 already decided.
    """
    from umrs.knowledge import tool_approvals

    existing = tool_approvals.get(approval_id)
    if existing is None:
        raise_typed(
            PlatformError.companion("not_found", f"approval not found: {approval_id}")
        )
    _enforce_approval_ownership(existing, x_user_id, request)
    try:
        record = tool_approvals.decide(approval_id, body.decision, decided_by=x_user_id)
    except KeyError as exc:
        raise_typed(PlatformError.companion("not_found", str(exc)), cause=exc)
    except ValueError as exc:
        raise_typed(PlatformError.companion("conflict", str(exc)), cause=exc)

    if body.decision == "always":
        _apply_always_decision(record, x_user_id)
    _notify_approval_listeners(record, body.decision, x_user_id)

    return {
        "approval_id": approval_id,
        "decision": body.decision,
        "status": record["status"],
        "loop_id": record.get("loop_id", ""),
        "durability": tool_approvals.durability(),
    }


def register(router: APIRouter) -> None:
    """Add the inbox routes to ``router`` (the companion router today, prefix
    ``/umrs/companion``; a core router after U3 S4/S5). Same paths, methods
    and handlers as the former ``@router`` decorators in ``companion``."""
    router.add_api_route("/approvals/pending", list_pending_approvals, methods=["GET"])
    router.add_api_route(
        "/approvals/{approval_id}/decision", record_approval_decision, methods=["POST"]
    )
