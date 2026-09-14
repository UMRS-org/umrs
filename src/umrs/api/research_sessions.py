"""UMRS Research Sessions REST API (ADR-025).

Durable investigation umbrella: create/list/get/update sessions and attach
research loops to them. Purely additive — no existing loop API behavior
changes; loop start accepts an optional ``session_id`` and auto-registers
membership when present.

Convention notes: follows the jobs/MCP router pattern (module-level store set
at sidecar boot, sync facade store, HTTP 503 when the store is unavailable).
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from umrs.research_sessions import ResearchSession, SessionLoopLink
from umrs.research_sessions.store import SessionStore

from ._verified_headers import CallerIdentity, get_caller
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

# DB error club for cascade-rollback surfaces (matches store dialects).
import sqlite3 as _sqlite3

_DB_ERROR_TYPES: list[type[Exception]] = [_sqlite3.Error]
try:
    import psycopg as _psycopg

    _DB_ERROR_TYPES.append(_psycopg.Error)
except ImportError:  # pragma: no cover - sqlite-only environments
    pass
_DB_ERROR_TYPES = tuple(_DB_ERROR_TYPES)

router = APIRouter(prefix="/umrs/research/sessions", tags=["research-sessions"])

_store: SessionStore | None = None


def set_session_store(store: SessionStore) -> None:
    global _store
    _store = store


def get_session_store() -> SessionStore | None:
    return _store


def _require_store() -> SessionStore:
    if _store is None:
        raise_typed(
            PlatformError.session(
                "store_unavailable",
                "Research session store unavailable (no DB pool configured)",
            )
        )
    return _store


# ── Models ─────────────────────────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    research_goal: str = Field("", description="Investigation intent")
    title: str = Field("", description="Human label; derived from goal if empty")
    authoritative_runtime: str = Field("umrs", pattern="^(umrs|local|hybrid)$")
    session_id: str = Field(
        "", description="Client-pinned id (idempotency); else generated"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    status: str | None = Field(None, pattern="^(active|paused|completed|archived)$")
    metadata: dict[str, Any] | None = None


class AttachLoopRequest(BaseModel):
    loop_id: str = Field(..., min_length=1)
    venue: str = Field("umrs", pattern="^(umrs|local)$")
    loop_role: str = Field("auto")


class SessionResponse(BaseModel):
    session: dict[str, Any]


class ListSessionsResponse(BaseModel):
    sessions: list[dict[str, Any]]
    count: int


class ListLoopsResponse(BaseModel):
    session_id: str
    loops: list[dict[str, Any]]
    count: int


def _new_session_id() -> str:
    return f"sess-{secrets.token_hex(8)}"


def _derive_title(goal: str) -> str:
    goal = goal.strip()
    if not goal:
        return "Untitled investigation"
    return goal[:64] + ("…" if len(goal) > 64 else "")


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    request: CreateSessionRequest,
    caller: CallerIdentity | None = Depends(get_caller),
) -> SessionResponse:
    """Create a research session (idempotent when client pins session_id)."""
    store = _require_store()

    project_id = request.project_id
    if caller is not None:
        project_id = caller.bind_for_write(request.project_id)
    session_id = request.session_id or _new_session_id()
    existing = store.get(session_id)
    if existing is not None:
        # Idempotent create: same session_id → return the stored session, but
        # only when the caller may actually see it (wave-6 F5).
        if caller is not None:
            caller.assert_can_access(existing.project_id)
        return SessionResponse(session=existing.to_dict())

    title = request.title.strip() or _derive_title(request.research_goal)
    session = ResearchSession(
        session_id=session_id,
        project_id=project_id,
        title=title,
        research_goal=request.research_goal,
        authoritative_runtime=request.authoritative_runtime,
        metadata=request.metadata,
    )
    store.create(session)
    return SessionResponse(session=session.to_dict())


@router.get("", response_model=ListSessionsResponse)
async def list_sessions(
    project_id: str | None = Query(None),
    status: str | None = Query(None),
    caller: CallerIdentity | None = Depends(get_caller),
) -> ListSessionsResponse:
    """List sessions, optionally filtered by project and/or status."""
    store = _require_store()
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    sessions = store.list(project_id=project_id, status=status)
    return ListSessionsResponse(
        sessions=[s.to_dict() for s in sessions], count=len(sessions)
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    caller: CallerIdentity | None = Depends(get_caller),
) -> SessionResponse:
    """Fetch one session with its loop links."""
    store = _require_store()
    session = store.get(session_id)
    if session is None:
        raise_typed(
            PlatformError.session("not_found", f"Session {session_id} not found")
        )
    if caller is not None:
        caller.assert_can_access(session.project_id)
    return SessionResponse(session=session.to_dict())


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    caller: CallerIdentity | None = Depends(get_caller),
) -> SessionResponse:
    """Update title / status / metadata. Status transitions log a marker."""
    store = _require_store()
    if caller is not None:
        existing = store.get(session_id)
        if existing is None:
            raise_typed(
                PlatformError.session("not_found", f"Session {session_id} not found")
            )
        caller.assert_can_access(existing.project_id)

    # ADR-035 item-8 cascade: status=archived routes through the
    # same-transaction archive (session row + session-scope trust revoke +
    # per-grant audit commit together, or not at all). Any other field
    # updates ride along first.
    archiving = request.status == "archived"
    if archiving:
        if request.title is not None or request.metadata is not None:
            store.update(
                session_id,
                title=request.title,
                status=None,
                metadata=request.metadata,
            )
        from .companion import get_trust_store

        try:
            session = store.archive_with_trust_cascade(session_id, get_trust_store())
        except _DB_ERROR_TYPES as exc:
            # Cascade failed AFTER session update was staged — the store's
            # own except-path rolls back, but clients need a clean 500, not
            # a raw DB exception leaking through FastAPI's default handler.
            logger.exception("archive cascade failed for %s", session_id)
            raise_typed(
                PlatformError.session(
                    "archive_cascade_failed",
                    f"archive cascade failed, transaction rolled back: {exc}",
                ),
                cause=exc,
            )
    else:
        session = store.update(
            session_id,
            title=request.title,
            status=request.status,
            metadata=request.metadata,
        )
    if session is None:
        raise_typed(
            PlatformError.session("not_found", f"Session {session_id} not found")
        )
    logger.info(
        "research session %s updated: title=%s status=%s",
        session_id,
        request.title is not None,
        request.status,
    )
    return SessionResponse(session=session.to_dict())


@router.post("/{session_id}/loops", status_code=201)
async def attach_loop(
    session_id: str,
    request: AttachLoopRequest,
    caller: CallerIdentity | None = Depends(get_caller),
) -> dict[str, Any]:
    """Register a loop under a session (also used by loop start auto-register)."""
    store = _require_store()
    existing = store.get(session_id)
    if existing is None:
        raise_typed(
            PlatformError.session("not_found", f"Session {session_id} not found")
        )
    if caller is not None:
        caller.assert_can_access(existing.project_id)

    link = SessionLoopLink(
        session_id=session_id,
        loop_id=request.loop_id,
        venue=request.venue,
        loop_role=request.loop_role,
    )
    store.attach_loop(link)
    return {"session_id": session_id, "loop": link.to_dict()}


@router.get("/{session_id}/loops", response_model=ListLoopsResponse)
async def list_session_loops(
    session_id: str,
    caller: CallerIdentity | None = Depends(get_caller),
) -> ListLoopsResponse:
    """List the loops attached to a session."""
    store = _require_store()
    existing = store.get(session_id)
    if existing is None:
        raise_typed(
            PlatformError.session("not_found", f"Session {session_id} not found")
        )
    if caller is not None:
        caller.assert_can_access(existing.project_id)
    loops = store.list_loops(session_id)
    return ListLoopsResponse(
        session_id=session_id, loops=[lnk.to_dict() for lnk in loops], count=len(loops)
    )


@router.delete("/{session_id}/loops/{loop_id}")
async def detach_loop(
    session_id: str,
    loop_id: str,
    caller: CallerIdentity | None = Depends(get_caller),
) -> dict[str, Any]:
    """Remove a loop from a session (loop itself is NOT stopped)."""
    store = _require_store()
    existing = store.get(session_id)
    if existing is None:
        raise_typed(
            PlatformError.session("not_found", f"Session {session_id} not found")
        )
    if caller is not None:
        caller.assert_can_access(existing.project_id)
    store.detach_loop(session_id, loop_id)
    return {"session_id": session_id, "loop_id": loop_id, "detached": True}
