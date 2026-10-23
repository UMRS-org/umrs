"""Remote sessions act like the user's local session (ADR-099 D-8, M-6).

Two mechanisms, each with an owner side and a worker-rail side:

Agent settings (migration 0128): the owner's client sends the settings its
local bridge applies; the worker reads them before each queued turn.

    PUT  /umrs/companion/sessions/{sid}/venue/settings            owner
    GET  /umrs/companion/sessions/{sid}/rail/settings             rail

Durable ask_user questions (migration 0127): the worker raises a question
and polls it; the owner answers from any attached client.

    POST /umrs/companion/sessions/{sid}/rail/questions            rail
    GET  /umrs/companion/sessions/{sid}/rail/questions/{qid}      rail
    POST /umrs/companion/sessions/{sid}/rail/questions/{qid}/expire  rail
    GET  /umrs/companion/sessions/{sid}/questions?status=pending  owner
    POST /umrs/companion/sessions/{sid}/questions/{qid}/answer    owner

Owner routes take the verified caller (``companion_sessions._uid``) and
answer 404 unless the caller owns the mirrored session. Rail routes take the
lease-token principal (``companion_venue.rail_principal``) and only see their
own session's rows. Questions and answers are conversation content: they are
never audited with their text and never appear in admin views. A settings
write is audited as ``venue.settings.set`` with the modes it sets.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from umrs.companion_trust.chat_turn_questions import (
    MAX_ANSWER_CHARS,
    MAX_QUESTION_CHARS,
    STATUSES,
    ChatTurnQuestionsStore,
    QuestionClosed,
)
from umrs.venues.agent_settings import (
    SettingsInvalid,
    SettingsOwnerConflict,
    VenueAgentSettingsStore,
    validate_agent_settings,
)
from umrs.venues.frontend_service import VenueError

from ._rail_identity import RailPrincipal
from .companion_sessions import _pool, _uid
from .companion_venue import _typed, get_venue_service, rail_principal
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/companion/sessions", tags=["companion-sessions"])

_QUESTION_ID = r"^[A-Za-z0-9_.:-]{1,64}$"
_AUDITED_SETTINGS = (
    "permission_mode",
    "interaction_mode",
    "ask_user_mode",
    "workflow_armed",
)


class RailQuestionBody(BaseModel):
    turn_id: str = Field(default="", max_length=200)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    options: list[str] = Field(default_factory=list, max_length=24)
    question_id: str | None = Field(default=None, pattern=_QUESTION_ID)


class AnswerBody(BaseModel):
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)


# ── wiring ─────────────────────────────────────────────────────────────────


def _settings_store(request: Request) -> VenueAgentSettingsStore:
    """Process-cached store (tests inject ``app.state.companion_venue_settings_store``)."""
    cached = getattr(request.app.state, "companion_venue_settings_store", None)
    if cached is None:
        cached = VenueAgentSettingsStore(_pool(request))
        request.app.state.companion_venue_settings_store = cached
    return cached


def _questions(request: Request, conn: Any) -> ChatTurnQuestionsStore:
    """Questions store on the borrowed connection (tests inject a factory)."""
    factory = getattr(request.app.state, "companion_venue_questions_factory", None)
    if factory is not None:
        return factory(conn)
    return ChatTurnQuestionsStore(conn)


def _with_questions(request: Request, fn: Any) -> Any:
    pool = _pool(request)
    conn = pool.getconn()
    try:
        return fn(_questions(request, conn))
    finally:
        pool.putconn(conn)


def _session_owned(request: Request, uid: str, session_id: str) -> bool:
    try:
        return get_venue_service(request).session_exists(uid, session_id)
    except VenueError as exc:  # no pool: a typed 503, never a bare 500
        _typed(exc)
        return False


def _owner(request: Request, session_id: str) -> str:
    """The verified caller, when they own the mirrored session (404 otherwise)."""
    uid = _uid(request)
    if not _session_owned(request, uid, session_id):
        raise_typed(PlatformError.companionvenue("not_found", "session not found"))
    return uid


def _own_question(
    record: dict[str, Any] | None, session_id: str, owner: str
) -> dict[str, Any]:
    """The record when it is this session's and this owner's; 404 otherwise."""
    if (
        record is None
        or record.get("session_id") != session_id
        or record.get("user_id") != owner
    ):
        raise_typed(PlatformError.companionvenue("not_found", "question not found"))
    return record


# ── agent settings ─────────────────────────────────────────────────────────


def _audit_settings(
    uid: str, session_id: str, rev: int, settings: dict[str, Any]
) -> None:
    from umrs.venues.session_audit import audit_venue

    modes = {key: settings[key] for key in _AUDITED_SETTINGS if key in settings}
    audit_venue(
        "settings.set", actor=uid, session_id=session_id, owner=uid, rev=rev, **modes
    )


@router.put("/{session_id}/venue/settings", response_model=dict[str, Any])
async def put_venue_settings(
    session_id: str, body: dict[str, Any], request: Request
) -> dict[str, Any]:
    """Store the owner's agent settings for this session's worker (new ``rev``).
    # auth: verified
    """
    uid = _owner(request, session_id)
    try:
        settings = validate_agent_settings(body)
    except SettingsInvalid as exc:
        raise_typed(
            PlatformError.companionvenue("bad_request", str(exc), field=exc.field),
            cause=exc,
        )
    try:
        rev = _settings_store(request).put(
            session_id=session_id, user_id=uid, settings=settings
        )
    except SettingsOwnerConflict as exc:
        raise_typed(PlatformError.companionvenue("conflict", str(exc)), cause=exc)
    _audit_settings(uid, session_id, rev, settings)
    return {"session_id": session_id, "rev": rev}


@router.get("/{session_id}/rail/settings", response_model=dict[str, Any])
async def rail_get_settings(
    session_id: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """The worker reads the owner's settings (``rev`` 0 and ``{}`` when none).
    # auth: lease-token
    """
    rev, settings = _settings_store(request).get(session_id, user_id=principal.owner)
    return {"session_id": session_id, "rev": rev, "settings": settings}


# ── questions: worker rail ─────────────────────────────────────────────────


@router.post("/{session_id}/rail/questions", response_model=dict[str, Any])
async def rail_ask_question(
    session_id: str,
    body: RailQuestionBody,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """The worker raises a durable ask_user question for the owner.
    # auth: lease-token
    """

    def ask(store: ChatTurnQuestionsStore) -> dict[str, Any]:
        return store.ask(
            session_id=session_id,
            turn_id=body.turn_id,
            user_id=principal.owner,
            question=body.question,
            options=list(body.options),
            question_id=body.question_id,
        )

    record = _with_questions(request, ask)
    if (
        record.get("session_id") != session_id
        or record.get("user_id") != principal.owner
    ):
        raise_typed(
            PlatformError.companionvenue("conflict", "question id already in use")
        )
    return {"session_id": session_id, "question": record}


@router.get("/{session_id}/rail/questions/{question_id}", response_model=dict[str, Any])
async def rail_get_question(
    session_id: str,
    question_id: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """The worker polls its question (its own session's only; 404 otherwise).
    # auth: lease-token
    """
    record = _with_questions(request, lambda store: store.get(question_id))
    return {
        "session_id": session_id,
        "question": _own_question(record, session_id, principal.owner),
    }


@router.post(
    "/{session_id}/rail/questions/{question_id}/expire", response_model=dict[str, Any]
)
async def rail_expire_question(
    session_id: str,
    question_id: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """The worker gave up waiting: a pending question becomes ``expired``.
    # auth: lease-token
    """

    def expire(store: ChatTurnQuestionsStore) -> dict[str, Any]:
        _own_question(store.get(question_id), session_id, principal.owner)
        return store.expire(question_id)

    return {"question": _with_questions(request, expire)}


# ── questions: owner ───────────────────────────────────────────────────────


@router.get("/{session_id}/questions", response_model=dict[str, Any])
async def list_questions(
    session_id: str, request: Request, status: str = "pending"
) -> dict[str, Any]:
    """The owner lists this session's questions (pending by default; ``all``).
    # auth: verified
    """
    uid = _owner(request, session_id)
    if status != "all" and status not in STATUSES:
        raise_typed(
            PlatformError.companionvenue(
                "bad_request", f"status must be one of {[*STATUSES, 'all']}"
            )
        )
    rows = _with_questions(
        request, lambda store: store.by_status(session_id=session_id, status=status)
    )
    return {"questions": [row for row in rows if row.get("user_id") == uid]}


@router.post(
    "/{session_id}/questions/{question_id}/answer", response_model=dict[str, Any]
)
async def answer_question(
    session_id: str, question_id: str, body: AnswerBody, request: Request
) -> dict[str, Any]:
    """The owner answers a pending question; 409 when it is no longer pending.
    # auth: verified
    """
    uid = _owner(request, session_id)

    def answer(store: ChatTurnQuestionsStore) -> dict[str, Any]:
        _own_question(store.get(question_id), session_id, uid)
        return store.answer(question_id, body.answer, answered_by=uid)

    try:
        record = _with_questions(request, answer)
    except QuestionClosed as exc:
        raise_typed(
            PlatformError.companionvenue("question_closed", str(exc)), cause=exc
        )
    return {"question": record}
