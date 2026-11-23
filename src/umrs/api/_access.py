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
    scopes_admit,
    stored_access_scopes,
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


def circle_finder_for(raw_request: Any) -> Callable[[str], str | None]:
    """``find_active_circle`` on the request's pool. Unreadable (no pool, a
    database error) answers the ref itself: the share is stored unchecked,
    as before the check existed — a validation, not the access boundary."""

    def find(ref: str) -> str | None:
        app = getattr(raw_request, "app", None)
        pool = getattr(getattr(app, "state", None), "umrs_db_pool", None)
        if pool is None:
            return ref
        from .circles import find_active_circle

        conn = pool.getconn()
        try:
            return find_active_circle(conn, ref)
        except Exception:  # noqa: BLE001 - any database error keeps the old behaviour
            logger.warning("circle lookup failed for %s; share stored unchecked", ref)
            conn.rollback()
            return ref
        finally:
            pool.putconn(conn)

    return find


def resource_caller(
    raw_request: Any, *, user_id: str | None, project_id: str, is_admin: bool
) -> ResourceCaller:
    """A guard caller from verified identity parts (+ lazy circle resolver)."""
    return ResourceCaller(
        user_id=user_id or "",
        project_id=project_id or "",
        is_admin=is_admin,
        circles_resolver=circles_resolver_for(raw_request, user_id or ""),
        circle_finder=circle_finder_for(raw_request),
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


# ── KN-S4b (ADR-118 D-8 read side): circles of the graph's own project ────
#
# A circle manager may add ANY user to a circle (circles.add_member checks
# the actor, not the new member), so a ``circle:<project>/<c>`` entry on a
# graph of that project could admit an outsider who was put on the roster
# after the share, past the write-time gate (KN-S4). From 2026-11-24 such an
# entry admits a reader only while the reader is inside the graph's project:
# verified into it, an ADR-097 live member (owned projects included) or an
# admin. ``user:``/``project:``/``system`` entries and circles of OTHER
# projects are unchanged (each was named at write time). Unreadable
# membership answers "outside" (fail closed).
#
# A graph with no project_id (a personal graph, or a legacy row with a NULL
# project) has no "own project" circle, so every entry on it is judged as
# named at write time, like a circle of another project: its owner chose the
# circle and accepted that circle's managers. There is no project to be
# inside of, so the gate has nothing to check (ADR-118 D-8 note, 2026-11-24).

_INSIDERS_ATTR = "kn_s4b_insiders"


def _live_member(raw_request: Any, user: str, project_id: str) -> bool:
    """ADR-097 D-1: is *user* a live member (or owner) of *project_id*?
    No pool, no user or an unreadable record answers False."""
    from .projects import ProjectStore, live_memberships

    app = getattr(raw_request, "app", None)
    pool = getattr(getattr(app, "state", None), "umrs_db_pool", None)
    if pool is None or not user:
        return False
    conn = pool.getconn()
    try:
        memberships = live_memberships(ProjectStore(conn), user)
    except Exception:  # unreadable is outside (fail closed)
        logger.warning("live membership unreadable for %s", user, exc_info=True)
        memberships = None
    finally:
        try:
            conn.rollback()  # a read must not leave the conn idle in transaction
        finally:
            pool.putconn(conn)
    return project_id in (memberships or [])


def _insider_cache(raw_request: Any) -> dict[tuple[str, str], bool]:
    """Per-request memo of membership answers (a listing asks per graph)."""
    state = getattr(raw_request, "state", None)
    cache = getattr(state, _INSIDERS_ATTR, None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    try:
        setattr(state, _INSIDERS_ATTR, cache)
    except (AttributeError, TypeError):
        pass  # no request state: answer uncached
    return cache


def user_in_project(raw_request: Any, user: str, project_id: str) -> bool:
    """ADR-097 D-1: is *user* a live member (or owner) of *project_id*,
    memoised per request. The ONE membership read of the knowledge gates:
    the KN-S4 share gate, the KN-S4b read gate and ``effective_readers``."""
    if not (user and project_id):
        return False
    cache = _insider_cache(raw_request)
    key = (user, project_id)
    if key not in cache:
        cache[key] = _live_member(raw_request, user, project_id)
    return cache[key]


def project_insider(raw_request: Any, caller: ResourceCaller, project_id: str) -> bool:
    """Is the caller inside *project_id*: an admin, verified into it, or a
    live member (its owner included)?"""
    if caller.is_admin or (project_id and caller.project_id == project_id):
        return True
    return user_in_project(raw_request, caller.user_id, project_id)


def _own_project_circle(scope: Any, project_id: str) -> bool:
    """A ``circle:`` entry naming a circle of the graph's own project."""
    return (
        scope.kind == "circle"
        and bool(project_id)
        and scope.id.partition("/")[0] == project_id
    )


def graph_access_admits(
    raw_request: Any, *, project_id: str | None, stored_access: Any, caller: ResourceCaller
) -> bool:
    """ADR-080 access list of one knowledge graph, with KN-S4b: an entry for
    a circle of the graph's own project admits only a project insider."""
    project = project_id or ""
    scopes = stored_access_scopes(stored_access)
    own = [s for s in scopes if _own_project_circle(s, project)]
    rest = [s for s in scopes if s not in own]
    if scopes_admit(rest, caller):
        return True
    if not own or not scopes_admit(own, caller):
        return False
    return project_insider(raw_request, caller, project)


def graph_read_allowed(
    raw_request: Any,
    *,
    kind: str,
    project_id: str | None,
    owner_id: str,
    stored_access: Any,
    scope: Any,
    caller: ResourceCaller | None = None,
) -> bool:
    """May the caller read one knowledge graph: the tier ladder OR the row's
    access list (ADR-080, KN-S4b). The one rule for the registry routes, K3
    graph selection and the proposal routes. ``caller`` lets a loop share one
    circle lookup."""
    from umrs.knowledge.access_scope import graph_readable

    if graph_readable(
        kind=kind, project_id=project_id, owner_id=owner_id, caller=scope
    ):
        return True
    if caller is None:
        caller = resource_caller(
            raw_request,
            user_id=scope.user_id,
            project_id=scope.project_id,
            is_admin=scope.is_admin,
        )
    return graph_access_admits(
        raw_request, project_id=project_id, stored_access=stored_access, caller=caller
    )


def graph_meta_readable(raw_request: Any, graph: Any, scope: Any) -> bool:
    """``graph_read_allowed`` for a registry row (``KnowledgeGraphMeta``)."""
    return graph_read_allowed(
        raw_request,
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        stored_access=graph.access,
        scope=scope,
    )
