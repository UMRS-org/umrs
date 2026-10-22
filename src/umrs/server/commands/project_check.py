"""ZMQ project check — ADR-097 D-7 (remote-session plan S-1, wave RS-0).

The HTTP rail decides a request's project in the gateway sidecar against
the caller's live memberships (ADR-097 D-3). Before 2026-10-22 the ZMQ rail
took the project from the payload unchecked: a bound CURVE session could
submit a job, list jobs or run a workflow in any project by naming it.

``CommandRegistry.dispatch`` calls :func:`project_refusal` for every
client command. Every project the payload names (top-level ``project_id``,
``params.project_id``, and the ``default`` project a JOB_SUBMIT or
RUN_WORKFLOW without one falls back to) must be one of the acting user's
live memberships, the same ``umrs.api.projects.live_memberships`` the
gateway reads, unless the caller is an admin. The modes mirror the
gateway's ``GATEWAY_PROJECT_MEMBERSHIP`` (here ``UMRS_ZMQ_PROJECT_MEMBERSHIP``),
with the CURVE session's token-bound projects in the role of the token's
own project:

- ``token``: the token's projects only (no lookup);
- ``live`` (default): a live membership, or a token project (drift logged);
  an unreadable membership fails closed unless the project is a token one;
- ``enforce``: a live membership only; an unreadable membership fails closed.

:func:`strip_identity_claim` drops a client-supplied ``_identity_verified``
from the wire: only the server's own in-process HTTP trigger may set it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd

logger = logging.getLogger(__name__)

IDENTITY_CLAIM = "_identity_verified"
MODES = ("token", "live", "enforce")
_CACHE_MAX = 4096

NOT_MEMBER = "project_not_member"
REVOKED = "project_membership_revoked"
UNAVAILABLE = "membership_unavailable"

Refusal = tuple[str, str]
Resolver = Callable[[str], "list[str] | None"]

_cache: dict[str, tuple[float, list[str]]] = {}
_cache_lock = threading.Lock()
_resolver: Resolver | None = None


# ---------------------------------------------------------------------------
# Payload reading
# ---------------------------------------------------------------------------


def wire_body(payload: Any) -> dict[str, Any] | None:
    """The JSON object of a ``(client_id, payload_str, ...)`` payload, else None."""
    if not isinstance(payload, tuple) or len(payload) < 2:
        return None
    raw = payload[1]
    if not isinstance(raw, str) or not raw.lstrip().startswith("{"):
        return None
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return body if isinstance(body, dict) else None


def strip_identity_claim(payload: Any) -> Any:
    """*payload* without a client-supplied ``_identity_verified`` key.

    The flag marks a gateway-verified caller. Only the HTTP trigger route
    sets it, in process (``api.workflows`` → ``handle_workflow_command``
    with ``trusted_identity``); nothing on the wire may.
    """
    body = wire_body(payload)
    if body is None or IDENTITY_CLAIM not in body:
        return payload
    body.pop(IDENTITY_CLAIM)
    logger.warning(
        "ZMQ payload from %r: client-supplied %s dropped", payload[0], IDENTITY_CLAIM
    )
    return (payload[0], json.dumps(body), *payload[2:])


def _as_id(value: Any) -> str | None:
    """A named project id, compared exactly as the handler stores it."""
    if value is None or value == "":
        return None
    return value if isinstance(value, str) else str(value)


def _params(body: dict[str, Any]) -> dict[str, Any]:
    params = body.get("params")
    return params if isinstance(params, dict) else {}


def _fallback_project(cmd: str, body: dict[str, Any]) -> str | None:
    """The project a handler acts in when the payload names none."""
    if cmd == TaskCmd.JOB_SUBMIT and "project_id" not in body:
        return (
            "default"  # JobCommandBackend._submit: payload.get("project_id", "default")
        )
    if cmd == TaskCmd.RUN_WORKFLOW:
        from umrs.server.workflow_registry_handler import _workflow_project_id

        return _workflow_project_id(body.get("params"))
    return None


def named_projects(cmd: str, body: dict[str, Any]) -> list[str]:
    """Every project *body* makes *cmd* act in, sorted."""
    names = {
        _as_id(body.get("project_id")),
        _as_id(_params(body).get("project_id")),
        _fallback_project(cmd, body),
    }
    return sorted(name for name in names if name is not None)


# ---------------------------------------------------------------------------
# Membership lookup (ADR-097 D-1, D-5)
# ---------------------------------------------------------------------------


def _db_live_memberships(user: str) -> list[str] | None:
    """``live_memberships`` over the process's default Postgres pool; None
    when Postgres is not configured or unreadable (fail closed)."""
    from umrs.shared.pg import get_default_sync_pool, pg_configured_from_env

    if not pg_configured_from_env():
        return None
    from umrs.api.projects import ProjectStore, live_memberships

    pool = get_default_sync_pool()
    conn = pool.getconn()
    try:
        return live_memberships(ProjectStore(conn), user)
    finally:
        try:
            conn.rollback()  # a read must not leave the conn idle in transaction
        finally:
            pool.putconn(conn)


def set_membership_resolver(resolver: Resolver | None) -> None:
    """Tests / boot code replace the lookup (None = the default pool). Clears the cache."""
    global _resolver
    _resolver = resolver
    reset_membership_cache()


def reset_membership_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _cache_ttl() -> float:
    try:
        return float(os.environ.get("UMRS_ZMQ_MEMBERSHIP_CACHE_TTL_SEC", "30"))
    except ValueError:
        return 30.0


def _lookup(user: str) -> list[str] | None:
    try:
        return (_resolver or _db_live_memberships)(user)
    except Exception:  # unreadable is not "none"
        logger.warning("ZMQ live memberships unreadable for %s", user, exc_info=True)
        return None


def live_memberships_cached(user: str) -> list[str] | None:
    """The user's live memberships, cached per user like the gateway's
    (ADR-097 D-5: 30 s, bounded). Failures are not cached."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(user)
    if hit and now - hit[0] < _cache_ttl():
        return hit[1]
    value = _lookup(user)
    if value is None:
        return None
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[user] = (now, value)
    return value


# ---------------------------------------------------------------------------
# Decision (ADR-097 D-3, D-4)
# ---------------------------------------------------------------------------


def current_mode() -> str:
    """``UMRS_ZMQ_PROJECT_MEMBERSHIP``; an unset or unknown value is ``live``
    (never a silent "no check": this rail had none before D-7)."""
    mode = os.environ.get("UMRS_ZMQ_PROJECT_MEMBERSHIP", "live").strip().lower()
    return mode if mode in MODES else "live"


def _not_member(project: str) -> Refusal:
    return NOT_MEMBER, f"not a member of project '{project}'"


def _token_mode_refusal(
    project: str, token_projects: tuple[str, ...]
) -> Refusal | None:
    return None if project in token_projects else _not_member(project)


def _unreadable_refusal(
    project: str, token_projects: tuple[str, ...], mode: str
) -> Refusal | None:
    if mode == "enforce" or project not in token_projects:
        return UNAVAILABLE, "project membership could not be checked"
    return None


def decide(
    project: str, live: list[str] | None, token_projects: tuple[str, ...], mode: str
) -> Refusal | None:
    """Why *project* may not be acted in; None = allowed (the gateway's
    ``_membership_refusal`` with the session's token projects)."""
    if mode == "token":
        return _token_mode_refusal(project, token_projects)
    if live is None:
        return _unreadable_refusal(project, token_projects, mode)
    if project in live:
        return None
    if project not in token_projects:
        return _not_member(project)
    if mode == "enforce":
        return REVOKED, f"no longer a member of project '{project}'"
    logger.warning(
        "ZMQ membership drift: token project %s is not a live membership", project
    )
    return None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _is_admin(control_mq: Any, client_id: str) -> bool:
    """The server's own role verdict (CURVE session grants or an admin token
    presented to the server), never a payload claim."""
    get_auth = getattr(control_mq, "get_auth_context", None)
    auth = get_auth(client_id) if callable(get_auth) else None
    return getattr(auth, "is_admin", False) is True


def _session_entry(control_mq: Any, zmq_sender_id: Any) -> Any | None:
    verified = getattr(control_mq, "verified_session", None)
    return verified(zmq_sender_id) if callable(verified) else None


def _caller(
    control_mq: Any, client_id: str, zmq_sender_id: Any
) -> tuple[str, tuple[str, ...]]:
    """(user to look up, the token's projects): the bound CURVE session's,
    else the client_id with no token projects (non-session auth modes)."""
    entry = _session_entry(control_mq, zmq_sender_id)
    if entry is None:
        return client_id, ()
    grants = getattr(entry, "grants", None)
    projects = tuple(getattr(grants, "projects", ()) or ())
    return str(entry.user_id), projects


def project_refusal(
    cmd: str, client_id: str, payload: Any, control_mq: Any, zmq_sender_id: Any
) -> Refusal | None:
    """ADR-097 D-7: the first project *payload* names that the caller may
    not act in, as ``(code, message)``; None when every one is allowed."""
    body = wire_body(payload)
    projects = named_projects(cmd, body) if body is not None else []
    if not projects or _is_admin(control_mq, client_id):
        return None
    user, token_projects = _caller(control_mq, client_id, zmq_sender_id)
    mode = current_mode()
    live = None if mode == "token" else live_memberships_cached(user)
    for project in projects:
        refusal = decide(project, live, token_projects, mode)
        if refusal is not None:
            return refusal
    return None


__all__ = [
    "IDENTITY_CLAIM",
    "MODES",
    "NOT_MEMBER",
    "REVOKED",
    "UNAVAILABLE",
    "current_mode",
    "decide",
    "live_memberships_cached",
    "named_projects",
    "project_refusal",
    "reset_membership_cache",
    "set_membership_resolver",
    "strip_identity_claim",
    "wire_body",
]
