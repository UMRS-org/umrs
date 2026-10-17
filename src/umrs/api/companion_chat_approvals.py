"""Durable chat-turn approval intents (ADR-057 D-3).

A chat turn POSTs an intent; any client of the session's owner lists the
pending rows and answers the decision. Owner-scoping follows the snapshot
row: the session must be mirrored for this uid (404 otherwise) so intents
can never cross owners.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from umrs.companion_trust.chat_turn_approvals import (
    VALID_DECISIONS,
    ChatTurnApprovalsStore,
)

from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/companion/sessions", tags=["companion-sessions"])


class ApprovalIntent(BaseModel):
    turn_id: str = ""
    action: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None


class ApprovalDecision(BaseModel):
    decision: str


def _uid(request: Request) -> str:
    """Verified caller id (401 without it; never invented).

    The header fallback read a bare ``X-User-ID`` (H-1 item 7, 2026-10-17);
    it now goes through the session routes' owner helper, which verifies the
    gateway HMAC in strict mode (mvp keeps the raw header).
    """
    from .companion_sessions import _uid as verified_owner

    identity = getattr(request.state, "identity", None)
    uid = getattr(identity, "user_id", "")
    if uid:
        return str(uid)
    return verified_owner(request)


def _owner_session_or_404(conn: Any, session_id: str, uid: str) -> None:
    cur = conn.cursor()
    # F1 audit-fix (2026-09-21): column is `user_id` (0080 + 0084 index);
    # fake-cursor fakes passed otherwise — real PG raises UndefinedColumn.
    cur.execute(
        "SELECT 1 FROM companion_session_snapshots WHERE session_id = %s AND user_id = %s",
        (session_id, uid),
    )
    if cur.fetchone() is None:
        raise_typed(
            PlatformError.companionsessions("not_found", "snapshot not found")
        )


@router.post("/{session_id}/approvals", response_model=dict[str, Any])
async def request_approval(
    session_id: str, body: ApprovalIntent, request: Request
) -> dict[str, Any]:
    """Raise a durable tool-grain intent for one chat turn.
    # auth: verified
    """
    uid = _uid(request)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.companionsessions("unavailable", "Shared DB pool unavailable in HTTP sidecar")
        )
    conn = pool.getconn()
    try:
        _owner_session_or_404(conn, session_id, uid)
        record = ChatTurnApprovalsStore(conn).request(
            session_id,
            body.turn_id,
            body.action,
            approval_id=body.approval_id,
            user_id=uid,
        )
        return {"session_id": session_id, "approval": record}
    finally:
        pool.putconn(conn)


@router.get("/{session_id}/approvals", response_model=dict[str, Any])
async def list_approvals(
    session_id: str, request: Request, status: str = "pending"
) -> dict[str, Any]:
    """List intents for the owner (pending by default).
    # auth: verified
    """
    uid = _uid(request)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.companionsessions("unavailable", "Shared DB pool unavailable in HTTP sidecar")
        )
    conn = pool.getconn()
    try:
        _owner_session_or_404(conn, session_id, uid)
        store = ChatTurnApprovalsStore(conn)
        records = (
            store.pending(session_id=session_id, user_id=uid)
            if status == "pending"
            else store.pending_decided(session_id=session_id, status=status)
        )
        return {"session_id": session_id, "approvals": records, "count": len(records)}
    finally:
        pool.putconn(conn)


@router.post("/{session_id}/approvals/{approval_id}/decision", response_model=dict[str, Any])
async def decide_approval(
    session_id: str, approval_id: str, body: ApprovalDecision, request: Request
) -> dict[str, Any]:
    """Answer an intent (`once|always|reject`); terminal transitions only.
    # auth: verified
    """
    uid = _uid(request)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.companionsessions("unavailable", "Shared DB pool unavailable in HTTP sidecar")
        )
    if body.decision not in VALID_DECISIONS:
        raise_typed(
            PlatformError.companionsessions(
                "invalid_decision", f"decision must be one of {sorted(VALID_DECISIONS)}"
            )
        )
    conn = pool.getconn()
    try:
        _owner_session_or_404(conn, session_id, uid)
        store = ChatTurnApprovalsStore(conn)
        record = store.get(approval_id)
        if record is None or record["session_id"] != session_id:
            raise_typed(
                PlatformError.companionsessions("not_found", "approval not found")
            )
        try:
            decided = store.decide(approval_id, body.decision, decided_by=uid)
        except ValueError as exc:
            raise_typed(
                PlatformError.companionsessions("conflict", str(exc))
            )
        return {"session_id": session_id, "approval": decided}
    finally:
        pool.putconn(conn)
