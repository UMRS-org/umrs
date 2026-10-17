"""UMRS Companion HTTP surface (wave T3b m5).

Routes under ``/umrs/companion`` (deployed as ``/gw/companion/*`` behind NGINX
+ auth sidecar — m1 fold-in). Responsibilities in this milestone:

- worker lifecycle surface over the 0070 registry (register/read),
- turn-attempt surface over the 0071 registry (begin/read/idempotent sse),
- pending-approval inbox over durable rows (ADR-036 (a), WP-05): owner-scoped
  list + terminal decisions that parked loop drivers consume,
- ownership pinning: a registry row is reachable only by its owner_user_id
  (401 on a worker mismatch; another user's approval reads as 404), token ≠
  auth ≠ liveness holds end-to-end.

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
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from umrs.companion_trust.store import SessionArchivedError
from umrs.jobs.companion_turn_executor import kick as kick_turn_executor
from umrs.jobs.companion_worker import (
    CompanionTurnAttemptRegistry,
    CompanionWorkerManager,
    CompanionWorkerSpec,
    InvalidResumeToken,
    TurnAttemptNotFound,
    WorkerExhausted,
    WorkerNotFound,
)
from umrs.jobs.companion_worker_supervisor import AtCapacity
from umrs.shared.scope_grammar import ScopeError, fallback_chain
from umrs.shared.scope_grammar import parse as parse_scope

from .typed_errors import PlatformError, raise_typed
from .workflows import _require_verified_workflow_identity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/companion", tags=["companion"])

# Per-user-scope TRUST allowlist (wave T3c): module-level store handle, set
# at sidecar boot the same way the research-session store is.
# ADR-053 (1.1): singleton home is `umrs.shared.trust_store_access`; these
# are the api-layer re-exports preserving every api caller byte-for-byte.
from umrs.shared.trust_store_access import (
    get_trust_store as get_trust_store,
)
from umrs.shared.trust_store_access import (
    set_trust_store as set_trust_store,
)


def _require_trust_store():
    from umrs.shared.trust_store_access import get_trust_store as _shared_get

    store = _shared_get()
    if store is None:
        raise_typed(
            PlatformError.companion(
                "unavailable",
                "Companion trust store unavailable (no DB pool configured)",
            )
        )
    return store


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
    if isinstance(exc, SessionArchivedError):
        return HTTPException(status_code=410, detail=str(exc))
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
        raise_typed(PlatformError.companion("unavailable", "database pool unavailable"))
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


def _assert_mirrored_session_owner_order(session_id: str, user_id: str) -> None:
    """403 on cross-owner archive (only when a mirrored row identifies the
    real owner; row-absent → caller's tombstone lands under its own name).

    Reads `companion_session_snapshots` directly (0080/0084 lockstep — owner
    column name is `user_id`, same F1-fix lived across routes).
    """
    from umrs.shared.pg import get_default_sync_pool

    pool = get_default_sync_pool()
    if pool is None:
        return  # dev-tier without the pool: local-truth lane rules already block
    conn = pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id FROM companion_session_snapshots WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    finally:
        pool.putconn(conn)
    if row is None:
        return  # never had a mirrored row — caller's own tombstone territory
    if row[0] != user_id:
        raise_typed(
            PlatformError.companion(
                "forbidden", "archive refused: session is not owned by this caller"
            )
        )


def _strict_caller(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> dict[str, str]:
    """Gate + return (user_id, project_id) for scope-authorizing routes."""
    uid = _strict_headers(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    return {"user_id": uid, "project_id": (x_verified_project_id or "").strip()}


# ── request/response models ──────────────────────────────────────────────────


class WorkerRegisterRequest(BaseModel):
    owner_user_id: str
    project_id: str
    job_id: str | None = None


async def _submit_companion_job(request: Request, owner: str, project: str) -> str:
    """Submit the venue's companion_worker durable job (real jobs row the
    companion_turns FK references). Test-injectable via app.state."""
    submitter = getattr(request.app.state, "companion_job_submitter", None)
    if submitter is not None:
        return await submitter(owner, project)
    from umrs.server.durable_job_runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise_typed(
            PlatformError.companion("unavailable", "durable job runtime unavailable")
        )
    from umrs.api.agent_quota import enforce_spawn_budget
    from umrs.jobs.models import JobType

    # ADR-051 D-2: agent CPU-process budget per user — once at spawn,
    # before the durable row is created (typed 503 on the ceiling).
    enforce_spawn_budget(
        getattr(request.app.state, "umrs_db_pool", None), user_id=owner
    )
    return await runtime.submit_job_sync(
        job_type=JobType.COMPANION_WORKER,
        project_id=project,
        user_id=owner,
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
    session_id: str | None = None
    mode: str | None = Field(
        default=None,
        description=(
            "Permission mode captured per attempt: 'ask' (default) or 'auto'. "
            "Tool-grain enforcement lands with the agentic executor; the "
            "surface stores it today so clients code against one shape."
        ),
    )
    request_token: str | None = Field(
        default=None,
        description=(
            "Client-side idempotency key. Re-issuing the same token returns "
            "the same attempt row; a failed attempt needs a re-attempt "
            "(parent pointer) under a fresh token."
        ),
    )
    parent_attempt_id: str | None = None
    research_session_id: str | None = None


class TurnRowResponse(BaseModel):
    attempt_id: str
    job_id: str
    registry_key: str
    session_id: str
    request_token: str
    state: str
    committed_response_ref: str | None = None
    response_content: str | None = None
    mode: str = "ask"
    parent_attempt_id: str | None = None
    research_session_id: str | None = None


class ApprovalDecisionRequest(BaseModel):
    decision: str  # once | always | no


class ApprovalsPendingResponse(BaseModel):
    pending: list[dict[str, Any]] = []
    # Where the inbox lives: postgres (durable) | sqlite | process (ADR-036 (a)).
    durability: str = "process"


class TrustGrantRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    scope: str = Field("system", min_length=1)
    ttl_sec: int = Field(1800, ge=60, le=86400)


class TrustListResponse(BaseModel):
    rows: list[dict[str, Any]] = []
    count: int = 0


def _validated_scope(raw: str) -> str:
    """Canonical scope grammar (umrs.shared.scope_grammar). Legacy 'global'
    is REJECTED on this write path — callers must pass 'system'."""
    try:
        return str(parse_scope(raw, allow_legacy=False))
    except ScopeError as exc:
        raise_typed(PlatformError.companion("unprocessable", str(exc)), cause=exc)


def _authorize_scope_write(scope: str, x_user_id: str, caller_project: str) -> None:
    """Scope-write authorization (matrix row: companion trust).

    Rows are always user-pinned, but scope strings claim a domain and must
    not be mintable for domains the caller doesn't inhabit:
    - project:<pid>  → pid must be the caller's verified project
    - circle:<p/cid> → not resolvable on this trust-write path → 422
    - group:<gid>    → superseded by circle: (ADR-086) → 422
    - user:<uid>     → uid must equal the caller
    - session:<sid>  → allowed (caller-pinned rows)
    - system         → allowed (per-user blanket, still user-pinned)
    """
    claim = parse_scope(scope)
    if claim.kind == "project":
        if caller_project and claim.id == caller_project:
            return
        raise_typed(
            PlatformError.companion(
                "forbidden",
                (
                    f"cannot grant trust scoped to project {claim.id!r} — not the "
                    "caller's verified project"
                ),
            )
        )
    if claim.kind in ("circle", "group"):
        raise_typed(
            PlatformError.companion(
                "unprocessable",
                f"{claim.kind}-scoped trust is not writable on this path"
                " (audience-scoped trust resolves nowhere here)",
            )
        )
    if claim.kind == "user" and claim.id != x_user_id:
        raise_typed(
            PlatformError.companion(
                "forbidden",
                f"cannot grant trust scoped to user {claim.id!r} — not the caller",
            )
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
        raise_typed(
            PlatformError.companion("unauthorized", "worker ownership mismatch")
        )


def _validated_mode(raw: str | None) -> str:
    """Permission mode is captured per attempt ('ask' | 'auto')."""
    mode = (raw or "ask").strip().lower()
    if mode not in ("ask", "auto"):
        raise_typed(PlatformError.companion("unprocessable", f"invalid mode: {mode!r}"))
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


def _writable_worker_project(request: Request, requested: str | None) -> str:
    """The worker's durable job lands in a project the caller may write; the
    requested project was honoured unchecked (H-1, 2026-10-17)."""
    from umrs.knowledge.access_scope import writable_project_for

    from ._scope_identity import request_caller_scope

    project = writable_project_for(requested or "", request_caller_scope(request))
    if project is None:
        raise_typed(
            PlatformError.companion(
                "forbidden", f"no write access to project {requested!r}"
            )
        )
    return project


def _worker_owner(declared: str, x_user_id: str) -> str:
    """The worker's owner: the verified caller. An empty owner defaults to
    the caller and another user is refused (H-1, 2026-10-18: an empty owner
    passed the ownership check and then failed as a 500). An identity-less
    dev (mvp) caller keeps the declared owner."""
    owner = (declared or "").strip() or x_user_id
    _enforce_ownership({"owner_user_id": owner}, x_user_id)
    if not owner:
        raise_typed(
            PlatformError.companion("unprocessable", "owner_user_id is required")
        )
    return owner


def _owned_companion_job(job: Any, owner: str, project_id: str) -> bool:
    from umrs.jobs.models import JobType

    return (
        job is not None
        and job.user_id == owner
        and job.project_id == project_id
        and job.job_type == JobType.COMPANION_WORKER
    )


def _caller_worker_job(job_id: str | None, owner: str, project_id: str, x_user_id: str):
    """A caller-supplied job id, when it is the caller's own companion job in
    that project; another job (or an unknown one) reads as not found (H-1,
    2026-10-18: it was bound unchecked, and the row's turns and the
    supervisor's R4_COMPANION_JOB_ID run under it). An omitted id is
    submitted later; an identity-less dev (mvp) caller keeps its id."""
    if not job_id or not x_user_id:
        return job_id
    from umrs.jobs.store import JobNotFoundError

    from .jobs import get_optional_job_store

    store = get_optional_job_store()
    if store is None:
        raise_typed(PlatformError.companion("unavailable", "job store unavailable"))
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        job = None
    if not _owned_companion_job(job, owner, project_id):
        raise_typed(PlatformError.companion("not_found", f"job not found: {job_id}"))
    return job_id


@router.post("/workers", status_code=status.HTTP_201_CREATED)
async def register_worker(
    body: WorkerRegisterRequest,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> WorkerRowResponse:
    managers, _ = _managers(request)
    owner = _worker_owner(body.owner_user_id, x_user_id)
    project_id = _writable_worker_project(request, body.project_id)
    spec = CompanionWorkerSpec(owner_user_id=owner, project_id=project_id)
    job_id = _caller_worker_job(body.job_id, owner, project_id, x_user_id)
    try:
        job_id = job_id or await _submit_companion_job(request, owner, project_id)
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
        raise_typed(PlatformError.companion("not_found", "worker not found"))
    _enforce_ownership(row, x_user_id)
    return _worker_row_to_response(row)


def _turn_worker(
    managers: CompanionWorkerManager, registry_key: str, x_user_id: str
) -> dict:
    """The worker a turn begins on. Another user's is refused as before; an
    ownerless one reads as not found to an identified caller, as its turns
    would (review 2026-10-18: anyone could begin turns on it)."""
    row = managers.get(registry_key)
    if row is None or (x_user_id and not row.get("owner_user_id")):
        raise_typed(PlatformError.companion("not_found", "worker not found"))
    _enforce_ownership(row, x_user_id)
    return row


@router.post("/chat", status_code=status.HTTP_202_ACCEPTED)
async def begin_turn_attempt(
    body: TurnBeginRequest,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> TurnAcceptedResponse:
    managers, turns = _managers(request)
    row = _turn_worker(managers, body.registry_key, x_user_id)
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


def _owns_turn(managers: CompanionWorkerManager, attempt: Any, x_user_id: str) -> bool:
    """A turn belongs to the owner of its worker row (a missing row: nobody).

    The turn reads returned any attempt's content to any caller (H-1,
    2026-10-17). An identity-less mvp caller keeps the dev behaviour.
    """
    if not x_user_id:
        return True
    row = managers.get(str(attempt.registry_key or "")) or {}
    return str(row.get("owner_user_id") or "") == x_user_id


def _foreign_turn(request: Request, attempt_id: str, x_user_id: str) -> bool:
    """True when the attempt exists but belongs to another user."""
    managers, turns = _managers(request)
    attempt = turns.get(attempt_id)
    return attempt is not None and not _owns_turn(managers, attempt, x_user_id)


async def _attempt_not_found_events(attempt_id: str) -> Iterator[str]:
    """What a missing attempt streams; a foreign one streams the same."""
    yield f"data: {json.dumps({'type': 'attempt_not_found', 'attempt_id': attempt_id})}\n\n"


@router.get("/turns/{attempt_id}")
async def get_turn_attempt(
    attempt_id: str,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> TurnRowResponse:
    managers, turns = _managers(request)
    attempt = turns.get(attempt_id)
    if attempt is None or not _owns_turn(managers, attempt, x_user_id):
        raise _http_error_from(TurnAttemptNotFound(attempt_id))
    return _attempt_to_response(attempt)


@router.get("/turns/{attempt_id}/events")
async def stream_turn_events(
    attempt_id: str,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> StreamingResponse:
    _, turns = _managers(request)
    events = (
        _attempt_not_found_events(attempt_id)
        if _foreign_turn(request, attempt_id, x_user_id)
        else _sse_events(turns, attempt_id)
    )
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── pending approvals — durable inbox (ADR-036 (a), WP-05) ───────────────────


@router.get("/approvals/pending")
async def list_pending_approvals(
    request: Request,
    x_user_id: str = Depends(_strict_headers),
) -> ApprovalsPendingResponse:
    """Pending tool-grain intents visible to the caller.

    Owner-scoped: rows of loops the caller owns plus ownerless rows (loops
    started without a gateway identity). ``?loop_id=`` narrows to one loop;
    ``?registry_key=`` keeps the venue-ownership check of the m5 contract.
    """
    registry_key = request.query_params.get("registry_key")
    if registry_key:
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

    loop = loop_api._loop_store.get(loop_id) if loop_id else None
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
        from umrs.api.loop import _loop_store

        loop = _loop_store.get(loop_id)
        if loop is not None:
            loop.metadata["mode"] = "auto"
            _loop_store.save(loop)
    except Exception:
        logger.warning(
            "always-decision mode flip failed for loop %s", loop_id, exc_info=True
        )


def _mint_session_trust(loop_id: str, action: dict, x_user_id: str) -> None:
    """ADR-036 §3: ``always`` mints a durable ``session:<loop_id>`` trust row."""
    from umrs.shared.trust_store_access import get_trust_store as _shared_get

    if _shared_get() is None:
        return
    tool_name = str(
        action.get("workflow_name") or action.get("kind") or "workflow_trigger"
    )
    try:
        _shared_get().grant(
            x_user_id, tool_name, scope=f"session:{loop_id}", created_by=x_user_id
        )
    except Exception:
        logger.warning(
            "always-decision trust grant failed for loop %s", loop_id, exc_info=True
        )


def _apply_always_decision(record: dict, x_user_id: str) -> None:
    """``always``: flip the loop to auto AND mint a session-scope trust row.

    Both best-effort (ADR-036 §3): the decision itself is already durable.
    """
    loop_id = str(record.get("loop_id") or "")
    if not loop_id:
        return
    _flip_loop_to_auto(loop_id)
    _mint_session_trust(loop_id, record.get("action") or {}, x_user_id)


@router.post("/approvals/{approval_id}/decision")
async def record_approval_decision(
    approval_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
    x_user_id: str = Depends(_strict_headers),
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

    return {
        "approval_id": approval_id,
        "decision": body.decision,
        "status": record["status"],
        "loop_id": record.get("loop_id", ""),
        "durability": tool_approvals.durability(),
    }


# ── per-user-scope TRUST allowlist (wave T3c, ADR-035) ───────────────────────


@router.get("/trust", response_model=TrustListResponse)
async def list_trust_rows(
    scope: str | None = None,
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
    caller: dict = Depends(_strict_caller),
) -> dict:
    """Grant/refresh a trust row for the caller (owner-pinned, TTL-capped,
    scope-write authorized against the verified caller identity)."""
    store = _require_trust_store()
    scope = _validated_scope(body.scope)
    _authorize_scope_write(scope, caller["user_id"], caller["project_id"])
    try:
        row = store.grant(
            caller["user_id"],
            body.tool_name,
            scope=scope,
            ttl_sec=body.ttl_sec,
            created_by=caller["user_id"],
        )
    except SessionArchivedError as exc:
        # Gone: once the tombstone exists, late grant attempts must fail
        # closed (410) — the archived session's scopes are terminal.
        raise_typed(PlatformError.companion("gone", str(exc)), cause=exc)
    return row.to_dict()


@router.get("/trust/check")
async def check_trust_row(
    tool_name: str,
    scope: str | None = None,
    x_user_id: str = Depends(_strict_headers),
) -> dict:
    """Autopilot consult: is this tool allowlisted NOW for the caller+scope?

    Returns ``{"allowed": bool, "expires_at": ...}``. The companion calls this
    instead of trusting its own process-global TRUST mode.
    """
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else "system"
    row = None
    # Precedence walk: session > project > group > user > system (one module,
    # one chain — see umrs.shared.scope_grammar).
    for candidate in fallback_chain(resolved):
        row = store.check(x_user_id, tool_name, scope=candidate)
        if row is not None:
            break
    return {
        "allowed": row is not None,
        "scope": (row.scope if row else resolved),
        "expires_at": (row.expires_at if row else ""),
    }


@router.delete("/trust/{tool_name}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_trust_row(
    tool_name: str,
    scope: str | None = None,
    x_user_id: str = Depends(_strict_headers),
) -> None:
    """Revoke a trust row (owner-pinned)."""
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else "system"
    store.revoke(x_user_id, tool_name, scope=resolved)


@router.post("/sessions/{session_id}/archive")
async def archive_companion_session(
    session_id: str,
    caller: dict = Depends(_strict_caller),
) -> dict:
    """Archive a companion session — tombstone + session-scope trust revoke
    # auth: verified
    + one audit row per revoked grant, ONE transaction (schema v2).

    This IS the client `session archive` lifecycle hook: bridge, CLI, and
    ACP converge here so the cascade is enforced where the trust rows live,
    never in whichever client triggered the archive. A separate bulk-revoke
    verb (if ever added) stays an explicit user action, NOT part of archive.

    F5 audit-block (2026-09-21): when a mirrored snapshot row records its
    own user_id, cross-owner archives refuse: the caller can't permanently
    traverse someone else's session-death OR trust cascade. No probe-row?
    Tombstone lands under this caller's own name only.
    """
    store = _require_trust_store()
    _assert_mirrored_session_owner_order(session_id, caller["user_id"])
    revoked = store.archive_session(session_id, archived_by=caller["user_id"])
    return {
        "session_id": session_id,
        "archived": True,
        "revoked_grants": revoked,
        "revoked_count": len(revoked),
    }
