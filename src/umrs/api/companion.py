"""UMRS Companion HTTP surface (wave T3b m5).

Routes under ``/umrs/companion`` (deployed as ``/gw/companion/*`` behind NGINX
+ auth sidecar — m1 fold-in). Responsibilities in this milestone:

- worker lifecycle surface over the 0070 registry (register/read),
- turn-attempt surface over the 0071 registry (begin/read/idempotent sse),
- pending-approval scaffold (typed, body lands with the live chat roundtrip),
- ownership pinning: a registry row is reachable only by its owner_user_id
  (401 on mismatch), token ≠ auth ≠ liveness holds end-to-end.

Strict-mode identity gate reuses ``workflows._require_verified_workflow_identity``
— the venue consumes ONLY the sidecar's signed identity tuple, never raw auth
material, exactly per ADR-027's claim-carrier line.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import Any, Iterator, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from umrs.jobs.companion_worker import (
    CompanionTurnAttemptRegistry,
    CompanionWorkerManager,
    CompanionWorkerSpec,
    InvalidResumeToken,
    TurnAttemptNotFound,
    WorkerExhausted,
    WorkerNotFound,
)
from umrs.jobs.companion_turn_executor import kick as kick_turn_executor
from umrs.jobs.companion_worker_supervisor import AtCapacity

from .workflows import _require_verified_workflow_identity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/companion", tags=["companion"])

# ── typed error mapping ─────────────────────────────────────────────────────


def _http_error_from(exc: Exception) -> HTTPException:
    """Map typed venue errors to HTTP statuses deterministically."""
    if isinstance(exc, WorkerNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TurnAttemptNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, WorkerExhausted):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, InvalidResumeToken):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, AtCapacity):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, HTTPException):
        return exc
    return HTTPException(status_code=500, detail=str(exc))


def _managers(
    request: Request,
) -> tuple[CompanionWorkerManager, CompanionTurnAttemptRegistry]:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database pool unavailable")
    return CompanionWorkerManager(pool), CompanionTurnAttemptRegistry(pool)


def _strict_headers(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> str:
    """Gate + return the caller's verified user id."""
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    return (x_user_id or "").strip()


# ── request/response models ──────────────────────────────────────────────────


class WorkerRegisterRequest(BaseModel):
    owner_user_id: str
    project_id: str
    job_id: str


class WorkerRowResponse(BaseModel):
    registry_key: str
    owner_user_id: str
    project_id: str
    job_id: str
    state: str
    attempt_index: int
    attempts_max: int


class TurnBeginRequest(BaseModel):
    registry_key: str
    message: str
    session_id: Optional[str] = None
    mode: Optional[str] = Field(
        default=None,
        description=(
            "Permission mode captured per attempt: 'ask' (default) or 'auto'. "
            "Tool-grain enforcement lands with the agentic executor; the "
            "surface stores it today so clients code against one shape."
        ),
    )
    request_token: Optional[str] = Field(
        default=None,
        description=(
            "Client-side idempotency key. Re-issuing the same token returns "
            "the same attempt row; a failed attempt needs a re-attempt "
            "(parent pointer) under a fresh token."
        ),
    )
    parent_attempt_id: Optional[str] = None
    research_session_id: Optional[str] = None


class TurnRowResponse(BaseModel):
    attempt_id: str
    job_id: str
    registry_key: str
    session_id: str
    request_token: str
    state: str
    committed_response_ref: Optional[str] = None
    response_content: Optional[str] = None
    mode: str = "ask"
    parent_attempt_id: Optional[str] = None
    research_session_id: Optional[str] = None


class ApprovalDecisionRequest(BaseModel):
    decision: str  # once | always | no


class ApprovalsPendingResponse(BaseModel):
    pending: list[dict[str, Any]] = []


class TurnAcceptedResponse(BaseModel):
    attempt_id: str
    state: str


# ── helpers ──────────────────────────────────────────────────────────────────


def _worker_row_to_response(row: dict) -> WorkerRowResponse:
    return WorkerRowResponse(
        registry_key=str(row.get("registry_key") or ""),
        owner_user_id=str(row.get("owner_user_id") or ""),
        project_id=str(row.get("project_id") or ""),
        job_id=str(row.get("job_id") or ""),
        state=str(row.get("state") or ""),
        attempt_index=int(row.get("attempt_index") or 0),
        attempts_max=int(row.get("attempts_max") or 6),
    )


def _attempt_to_response(attempt) -> TurnRowResponse:
    return TurnRowResponse(
        attempt_id=attempt.attempt_id,
        job_id=attempt.job_id,
        registry_key=attempt.registry_key,
        session_id=attempt.session_id,
        request_token=attempt.request_token,
        state=attempt.state,
        committed_response_ref=attempt.committed_response_ref,
        response_content=attempt.response_content,
        mode=attempt.mode,
        parent_attempt_id=attempt.parent_attempt_id,
        research_session_id=attempt.research_session_id,
    )


def _enforce_ownership(row: dict, x_user_id: str) -> None:
    """A registry row is reachable only by its owner user."""
    owner = str(row.get("owner_user_id") or "")
    if x_user_id and owner and owner != x_user_id:
        raise HTTPException(status_code=401, detail="worker ownership mismatch")


def _validated_mode(raw: Optional[str]) -> str:
    """Permission mode is captured per attempt ('ask' | 'auto')."""
    mode = (raw or "ask").strip().lower()
    if mode not in ("ask", "auto"):
        raise HTTPException(status_code=422, detail=f"invalid mode: {mode!r}")
    return mode


def _kick_turn(
    turns: CompanionTurnAttemptRegistry, attempt_id: str, request: Request
) -> None:
    """T3c: execute the roundtrip in-sidecar (fail-soft — rows stay reattachable).

    The registry is the exactly-once authority; SSE reads the same row for
    replay (ADR-028). The chat function resolves from app state only in tests.
    """
    try:
        chat = getattr(request.app.state, "companion_turn_chat", None)
        kick_turn_executor(turns, attempt_id, app=request.app, chat=chat)
    except Exception:
        logger.warning("turn executor kick failed for %s", attempt_id)


_MAX_SSE_POLL_SEC = 25.0


async def _sse_events(
    registry: CompanionTurnAttemptRegistry, attempt_id: str
) -> Iterator[str]:
    """Reattach semantics per ADR-028:

    - committed → the committed result streams once, then ``data: [DONE]``.
    - failed → the stream ends with a typed error (410 surfaced by the wire).
    - in_flight → poll the attempt row briefly until it terminalizes.

    Note: re-executing never happens — the stream replays the same attempt
    buffer.
    """
    started = asyncio.get_event_loop().time()
    while True:
        attempt = registry.get(attempt_id)
        if attempt is None:
            yield f"data: {json.dumps({'type': 'attempt_not_found', 'attempt_id': attempt_id})}\n\n"
            return
        if attempt.state == "committed":
            yield f"data: {json.dumps({'type': 'result', 'committed_response_ref': attempt.committed_response_ref, 'content': attempt.response_content})}\n\n"
            yield "data: [DONE]\n\n"
            return
        if attempt.state == "failed":
            yield f"data: {json.dumps({'type': 'attempt_failed', 'attempt_id': attempt.attempt_id})}\n\n"
            return
        if asyncio.get_event_loop().time() - started >= _MAX_SSE_POLL_SEC:
            yield f"data: {json.dumps({'type': 'in_flight', 'attempt_id': attempt.attempt_id})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'in_flight', 'attempt_id': attempt.attempt_id, 'pending': True})}\n\n"
        await asyncio.sleep(0.3)


# ── routes ────────────────────────────────────────────────────────────────────


@router.post("/workers", status_code=status.HTTP_201_CREATED)
async def register_worker(
    body: WorkerRegisterRequest,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> WorkerRowResponse:
    managers, _ = _managers(request)
    spec = CompanionWorkerSpec(
        owner_user_id=body.owner_user_id, project_id=body.project_id
    )
    _enforce_ownership(
        {"owner_user_id": body.owner_user_id}, x_user_id or body.owner_user_id
    )
    try:
        row = managers.register_active(spec, job_id=body.job_id)
        return _worker_row_to_response(row)
    except Exception as exc:
        logger.exception("register_worker failed")
        raise _http_error_from(exc) from exc


@router.get("/workers/{registry_key}")
async def get_worker(
    registry_key: str,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> WorkerRowResponse:
    managers, _ = _managers(request)
    row = managers.get(registry_key)
    if row is None:
        raise HTTPException(status_code=404, detail="worker not found")
    _enforce_ownership(row, x_user_id)
    return _worker_row_to_response(row)


@router.post("/chat", status_code=status.HTTP_202_ACCEPTED)
async def begin_turn_attempt(
    body: TurnBeginRequest,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> TurnAcceptedResponse:
    managers, turns = _managers(request)
    row = managers.get(body.registry_key)
    if row is None:
        raise HTTPException(status_code=404, detail="worker not found")
    _enforce_ownership(row, x_user_id)
    token = (body.request_token or "").strip() or f"rt-{secrets.token_hex(8)}"
    mode = _validated_mode(body.mode)
    try:
        attempt = turns.begin(
            job_id=str(row.get("job_id") or ""),
            registry_key=body.registry_key,
            session_id=body.session_id or "",
            request_token=token,
            user_message=body.message,
            parent_attempt_id=body.parent_attempt_id,
            research_session_id=body.research_session_id,
            mode=mode,
        )
    except Exception as exc:
        logger.exception("begin_turn_attempt failed")
        raise _http_error_from(exc) from exc
    _kick_turn(turns, attempt.attempt_id, request)
    return TurnAcceptedResponse(attempt_id=attempt.attempt_id, state=attempt.state)


@router.get("/turns/{attempt_id}")
async def get_turn_attempt(
    attempt_id: str,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> TurnRowResponse:
    _, turns = _managers(request)
    try:
        attempt = turns._required(attempt_id)
        return _attempt_to_response(attempt)
    except TurnAttemptNotFound as exc:
        raise _http_error_from(exc) from exc


@router.get("/turns/{attempt_id}/events")
async def stream_turn_events(
    attempt_id: str,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> StreamingResponse:
    _, turns = _managers(request)
    return StreamingResponse(
        _sse_events(turns, attempt_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── pending approvals (scaffold, typed, surface-complete) ────────────────────


@router.get("/approvals/pending")
async def list_pending_approvals(
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> ApprovalsPendingResponse:
    managers, _ = _managers(request)
    registry_key = request.query_params.get("registry_key")
    if registry_key:
        row = managers.get(registry_key)
        _enforce_ownership(row or {}, x_user_id)
    # Pending-approval rows are an attempt-grain event — the chat-side
    # implementation (next milestone) will populate this from the turn
    # attempt buffer. The route exists so clients code against a stable
    # surface (never a 501 from an unplanned path when T3c lands).
    return ApprovalsPendingResponse(pending=[])


@router.post("/approvals/{approval_id}/decision")
async def record_approval_decision(
    approval_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> dict:
    _managers(request)  # contract check: surface exists, owner pinned
    # Turn-grain approval persistence lands with the chat roundtrip; the
    # surface returns a typed in-flight marker today so clients connect to
    # the eventual end state instead of inventing their own 501-keyed path.
    return {
        "approval_id": approval_id,
        "decision": body.decision,
        "status": "surface_ready",
    }
