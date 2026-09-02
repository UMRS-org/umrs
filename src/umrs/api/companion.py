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
import os
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

# Per-user-scope TRUST allowlist (wave T3c): module-level store handle, set
# at sidecar boot the same way the research-session store is.
_trust_store = None


def set_trust_store(store) -> None:
    global _trust_store
    _trust_store = store


def _require_trust_store():
    if _trust_store is None:
        raise HTTPException(
            status_code=503,
            detail="Companion trust store unavailable (no DB pool configured)",
        )
    return _trust_store

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
    job_id: Optional[str] = None


async def _submit_companion_job(request: Request, owner: str, project: str) -> str:
    """Submit the venue's companion_worker durable job (real jobs row the
    companion_turns FK references). Test-injectable via app.state."""
    submitter = getattr(request.app.state, "companion_job_submitter", None)
    if submitter is not None:
        return await submitter(owner, project)
    from umrs.server.durable_job_runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="durable job runtime unavailable")
    from umrs.jobs.models import JobType

    return await runtime.submit_job_sync(
        job_type=JobType.COMPANION_WORKER,
        project_id=project,
        spec={"registry_key": f"{owner}@{project}"},
    )


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


class TrustGrantRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    scope: str = Field("global", min_length=1)
    ttl_sec: int = Field(1800, ge=60, le=86400)


class TrustListResponse(BaseModel):
    rows: list[dict[str, Any]] = []
    count: int = 0


def _validated_scope(raw: str) -> str:
    """Scope grammar: global | project:<id> | session:<id>."""
    scope = raw.strip()
    if scope == "global":
        return scope
    if scope.startswith("project:") and len(scope) > len("project:"):
        return scope
    if scope.startswith("session:") and len(scope) > len("session:"):
        return scope
    raise HTTPException(
        status_code=422,
        detail=f"invalid trust scope (want global|project:<id>|session:<id>): {raw!r}",
    )


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
    turns: CompanionTurnAttemptRegistry,
    attempt_id: str,
    request: Request,
    *,
    registry_key: str = "",
) -> None:
    """T3c: execute the roundtrip in-sidecar (fail-soft — rows stay reattachable).

    The registry is the exactly-once authority; SSE reads the same row for
    replay (ADR-028). The chat function resolves from app state only in tests.
    """
    try:
        chat = getattr(request.app.state, "companion_turn_chat", None)
        kick_turn_executor(
            turns, attempt_id, app=request.app, chat=chat, registry_key=registry_key
        )
    except Exception:
        logger.warning("turn executor kick failed for %s", attempt_id)


_MAX_SSE_POLL_SEC = 25.0


def _stale_ttl_seconds() -> int:
    """Reaper TTL for in_flight attempts (T3c follow-up; env-overridable)."""
    try:
        return max(30, int(os.environ.get("UMRS_COMPANION_TURN_STALE_TTL_SEC", "300")))
    except ValueError:
        return 300


async def _sse_events(
    registry: CompanionTurnAttemptRegistry, attempt_id: str
) -> Iterator[str]:
    """Reattach semantics per ADR-028:

    - committed → the committed result streams once, then ``data: [DONE]``.
    - failed → the stream ends with a typed error (410 surfaced by the wire).
    - in_flight → poll the attempt row briefly until it terminalizes.

    Note: re-executing never happens — the stream replays the same attempt
    buffer. Attempts orphaned past the reaper TTL (sidecar reload) are
    lazily failed on read so clients never poll forever.
    """
    started = asyncio.get_event_loop().time()
    while True:
        attempt = registry.get(attempt_id)
        if attempt is None:
            yield f"data: {json.dumps({'type': 'attempt_not_found', 'attempt_id': attempt_id})}\n\n"
            return
        # Reap once when an OPEN stream surfaces a stale in_flight attempt
        # (sidecar reload orphan): fail terminally instead of polling
        # forever; live attempts unaffected (UPDATE matches nothing when
        # fresh) and terminal states never touch the reaper.
        if attempt.state == "in_flight" and registry.fail_if_stale(
            attempt_id, _stale_ttl_seconds()
        ):
            yield f"data: {json.dumps({'type': 'attempt_failed', 'attempt_id': attempt_id, 'reason': 'stale_reaped'})}\n\n"
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
        job_id = body.job_id or await _submit_companion_job(
            request, body.owner_user_id, body.project_id
        )
        row = managers.register_active(spec, job_id=job_id)
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
    _kick_turn(turns, attempt.attempt_id, request, registry_key=body.registry_key)
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
    from umrs.knowledge import tool_approvals

    return ApprovalsPendingResponse(pending=tool_approvals.pending())


@router.post("/approvals/{approval_id}/decision")
async def record_approval_decision(
    approval_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> dict:
    _managers(request)  # contract check: surface exists, owner pinned
    # Tool-grain approval registry (ADR-028): loop drivers park on these
    # intents; the decision unblocks them. ``always`` additionally flips the
    # owning loop into ``mode=auto`` so it stops gating from then on.
    from umrs.knowledge import tool_approvals

    try:
        record = tool_approvals.decide(approval_id, body.decision, decided_by=x_user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if body.decision == "always":
        try:
            from umrs.api.loop import _loop_store

            loop = _loop_store.get(record["loop_id"])
            if loop is not None:
                loop.metadata["mode"] = "auto"
                _loop_store.save(loop)
        except Exception:  # noqa: BLE001 — mode flip is best-effort
            logger.warning(
                "always-decision mode flip failed for loop %s",
                record.get("loop_id"),
                exc_info=True,
            )

    return {
        "approval_id": approval_id,
        "decision": body.decision,
        "status": record["status"],
        "loop_id": record.get("loop_id", ""),
    }


# ── per-user-scope TRUST allowlist (wave T3c, ADR-035) ───────────────────────


@router.get("/trust", response_model=TrustListResponse)
async def list_trust_rows(
    scope: Optional[str] = None,
    x_user_id: str = Depends(_strict_headers),
) -> TrustListResponse:
    """List the caller's unexpired allowlist rows (owner-pinned)."""
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else None
    rows = store.list(x_user_id, scope=resolved)
    return TrustListResponse(rows=[r.to_dict() for r in rows], count=len(rows))


@router.put("/trust", status_code=status.HTTP_201_CREATED)
async def grant_trust_row(
    body: TrustGrantRequest,
    x_user_id: str = Depends(_strict_headers),
) -> dict:
    """Grant/refresh a trust row for the caller (owner-pinned, TTL-capped)."""
    store = _require_trust_store()
    scope = _validated_scope(body.scope)
    row = store.grant(
        x_user_id, body.tool_name, scope=scope, ttl_sec=body.ttl_sec, created_by=x_user_id
    )
    return row.to_dict()


@router.get("/trust/check")
async def check_trust_row(
    tool_name: str,
    scope: Optional[str] = None,
    x_user_id: str = Depends(_strict_headers),
) -> dict:
    """Autopilot consult: is this tool allowlisted NOW for the caller+scope?

    Returns ``{"allowed": bool, "expires_at": ...}``. The companion calls this
    instead of trusting its own process-global TRUST mode.
    """
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else "global"
    row = store.check(x_user_id, tool_name, scope=resolved)
    if row is None and resolved != "global":
        row = store.check(x_user_id, tool_name, scope="global")
    return {
        "allowed": row is not None,
        "scope": (row.scope if row else resolved),
        "expires_at": (row.expires_at if row else ""),
    }


@router.delete("/trust/{tool_name}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_trust_row(
    tool_name: str,
    scope: Optional[str] = None,
    x_user_id: str = Depends(_strict_headers),
) -> None:
    """Revoke a trust row (owner-pinned)."""
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else "global"
    store.revoke(x_user_id, tool_name, scope=resolved)
    return None
