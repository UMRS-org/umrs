"""UMRS API routes for the knowledge graph registry (multi-graph Phase 1/3).

Routes (mounted under the gateway ``/gw/knowledge/`` prefix — no nginx edits):

  GET   /umrs/knowledge/graphs            — list (project/status filters, archive-aware)
  POST  /umrs/knowledge/graphs            — create a named graph
  GET   /umrs/knowledge/graphs/{graph_id} — graph detail
  PATCH /umrs/knowledge/graphs/{graph_id} — metadata update / archive (no hard delete)
  GET   /umrs/knowledge/graphs/{graph_id}/proposed — proposed rows awaiting review (K1.5)

Backward compatibility: listing with a ``project_id`` the caller may write
materializes that project's deterministic default graph first (H-1,
2026-10-17: never for another project), so existing projects appear
automatically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from umrs.knowledge.store import KnowledgeStore

from umrs.knowledge.access_scope import (
    CallerScope,
    graph_readable,
    graph_writable,
    project_writable,
)
from umrs.knowledge.graph_registry import (
    GRAPH_STATUSES,
    GraphRegistryStore,
    graph_id_is_default,
)

from ._access import (
    checked_access,
    graph_access_admits,
    resource_caller,
    user_in_project,
)
from ._access import graph_read_allowed as _graph_read_allowed
from ._knowledge_mutation_guard import (
    AGENT_ACTOR,
    enforce_registry_write_authorization,
    request_actor,
)
from ._scope_identity import get_caller_scope
from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge/graphs", tags=["knowledge-graphs"])


def _registry(request: Request) -> GraphRegistryStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.knowledgegraphs("unavailable", "Database unavailable")
        )
    return GraphRegistryStore(pool)


def _shared_read_admits(request: Request, graph: Any, scope: CallerScope) -> bool:
    """ADR-080: does the row's access list admit this caller (beyond the tier
    ladder)? Circle membership resolves lazily against the request pool; a
    circle of the graph's own project admits only a project insider (KN-S4b)."""
    caller = resource_caller(
        request,
        user_id=scope.user_id,
        project_id=scope.project_id,
        is_admin=scope.is_admin,
    )
    return graph_access_admits(
        request, project_id=graph.project_id, stored_access=graph.access, caller=caller
    )


def _stored_project(
    kind: str | None, project_id: str | None, scope: CallerScope
) -> str | None:
    """The project a new graph is stored under.

    Owner-only graphs are not project-bound and system graphs stay global.
    Any other graph created without a project gets the caller's project, the
    one create_graph's write check admitted. Persisting NULL left a graph no
    non-admin could ever update, archive or decide proposals on, its creator
    included (2026-10-09).
    """
    kind_reduced = (kind or "").strip().lower()
    if kind_reduced in ("user", "personal"):
        return None
    if project_id is None and kind_reduced != "system":
        return scope.project_id or None
    return project_id


class AccessUpdate(BaseModel):
    """ADR-080 access list update (validated against the canonical grammar)."""

    access: list[str] = Field(
        default_factory=list,
        description=(
            "user:<id> | project:<id> | circle:<proj>/<id> | system; [] = tier ladder only"
        ),
    )


def _base_graph_list(
    registry: GraphRegistryStore,
    *,
    project_id: str | None,
    status: str | None,
    q: str | None,
    limit: int,
) -> list[Any]:
    """The tier-ladder listing (the caller materializes its default graph)."""
    if q:
        graphs = registry.search_graphs(q, project_id=project_id, limit=limit)
        if status:
            graphs = [g for g in graphs if g.status == status]
        return graphs
    return registry.list_graphs(project_id=project_id, status=status, limit=limit)


class CreateGraphRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    project_id: str | None = None
    kind: str = "research"
    ontology_id: str | None = None
    ontology_version: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class UpdateGraphRequest(BaseModel):
    description: str | None = None
    kind: str | None = None
    ontology_id: str | None = None
    ontology_version: str | None = None
    status: str | None = Field(default=None, description="Lifecycle: active | archived")


@router.get("")
async def list_graphs(
    request: Request,
    project_id: str | None = Query(None),
    status: str | None = Query("active"),
    q: str | None = Query(None, description="Name/description search"),
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    """List registered knowledge graphs with optional filters.

    When ``project_id`` names a project the caller may write, its default
    graph is materialized first so legacy projects are listed without any
    manual migration. That write ran before the scope check, for any named
    project (H-1, 2026-10-17); it now follows a passing check.
    """
    scope = get_caller_scope(request)
    registry = _registry(request)
    if project_id and project_writable(project_id, scope):
        registry.ensure_default_graph(project_id)
    graphs = _base_graph_list(
        registry, project_id=project_id, status=status, q=q, limit=limit
    )
    visible = [
        g
        for g in graphs
        if graph_readable(
            kind=g.kind, project_id=g.project_id, owner_id=g.owner_id, caller=scope
        )
    ]
    seen = {g.graph_id for g in visible}
    visible.extend(
        _shared_visible_graphs(request, registry, status=status, scope=scope, seen=seen)
    )
    return {"graphs": [g.to_dict() for g in visible], "total": len(visible)}


def _shared_visible_graphs(
    request: Request,
    registry: GraphRegistryStore,
    *,
    status: str | None,
    scope: CallerScope,
    seen: set[str],
) -> list[Any]:
    """ADR-080 rows outside the caller's tier whose access admits them."""
    out: list[Any] = []
    for shared in registry.list_shared_graphs(status=status):
        if shared.graph_id in seen:
            continue
        if _shared_read_admits(request, shared, scope):
            out.append(shared)
            seen.add(shared.graph_id)
    return out


@router.post("", status_code=201)
async def create_graph(
    body: CreateGraphRequest,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Create a named, independent knowledge graph.

    Three-tier boundary: system graphs are admin-only; user/personal graphs
    are normalized to the caller's personal namespace; project-like graphs
    require the caller to belong to the target project.
    """
    enforce_registry_write_authorization(request)
    registry = _registry(request)
    scope = CallerScope(
        user_id=identity.user_id or "",
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    project_id = body.project_id
    if not graph_writable(
        kind=body.kind,
        project_id=project_id if project_id is not None else scope.project_id,
        owner_id=scope.user_id,
        caller=scope,
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": (
                    f"Cannot create a '{body.kind}' graph in this namespace — "
                    "system graphs are admin-only, project graphs require "
                    "project membership, personal graphs are owner-only."
                ),
            },
        )
    project_id = _stored_project(body.kind, project_id, scope)
    ontology_version = _checked_ontology_version(
        request, body.ontology_id, body.ontology_version, project_id
    )
    graph = registry.create_graph(
        body.name,
        description=body.description,
        owner_id=identity.user_id or "",
        project_id=project_id,
        kind=body.kind,
        ontology_id=body.ontology_id,
        ontology_version=ontology_version,
        provenance=body.provenance,
    )
    return graph.to_dict()


def _checked_ontology_version(
    request: Request,
    ontology_id: str | None,
    version: str | None,
    project_id: str | None,
) -> str | None:
    """The ontology version to store for a graph binding (ADR-116 D-5).

    The coordinate must be built in or registered live (400
    ``ontology.not_found``); an id without a version pins the latest live
    version. A version without an id is 400 ``ontology.invalid``.
    """
    from . import knowledge_ontologies as ontology_api

    if not ontology_id:
        if version:
            ontology_api.ontology_error(
                "invalid", "ontology_version needs an ontology_id"
            )
        return version
    return ontology_api.pinned_ontology_version(
        request, ontology_id, version, project_id
    )


def _patched_ontology(
    request: Request, body: UpdateGraphRequest, existing: Any
) -> tuple[str | None, str | None]:
    """(ontology_id, ontology_version) a PATCH writes; (None, None) = unchanged.

    ``ontology_id: ""`` unbinds the graph (back to the default vocabulary).
    """
    ontology_id = body.ontology_id
    if ontology_id is None and body.ontology_version is not None:
        ontology_id = existing.ontology_id or None
    if ontology_id is None:
        _checked_ontology_version(request, None, body.ontology_version, None)
        return None, None
    if ontology_id == "":
        return "", ""
    version = _checked_ontology_version(
        request, ontology_id, body.ontology_version, existing.project_id
    )
    return ontology_id, version


@router.get("/{graph_id}")
async def get_graph(graph_id: str, request: Request) -> dict[str, Any]:
    """Fetch one graph by id (or resolve by exact name)."""
    registry = _registry(request)
    graph = registry.resolve_graph(graph_id)
    scope = get_caller_scope(request)
    if graph is None or not _graph_read_allowed(
        request,
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        stored_access=graph.access,
        scope=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    payload = graph.to_dict()
    # ADR-049 B-part-2: traversal surfaces carry the resolved ontology
    # (registry-first); unresolvable references named, never smudged.
    payload["ontology"] = _resolved_ontology_block(request, payload)
    return payload


def _resolved_ontology_block(request: Request, meta: dict[str, Any]) -> dict[str, Any]:
    """Registry-first ontology block for one graph meta (ADR-049 B-part-2).

    Shares the knowledge-query payload shape
    ({graph_id, ontology} | {graph_id, ontology_error}) so consumers have
    ONE language across query and traversal surfaces.
    """
    from .knowledge_query import _ontology_blocks_for

    pool = getattr(request.app.state, "umrs_db_pool", None)
    blocks = _ontology_blocks_for([meta], pool)
    if blocks:
        return blocks[0]
    return {"graph_id": meta.get("graph_id", "?"), "ontology_error": "unresolved"}


@router.get("/{graph_id}/content")
async def get_graph_content(
    graph_id: str,
    request: Request,
    node_limit: int = Query(200, ge=1, le=1000),
    edge_limit: int = Query(400, ge=1, le=2000),
) -> dict[str, Any]:
    """Bounded nodes+edges for one registered graph (webview visualization).

    Read-only; no mutation guard. ``graph_id`` may be an id or exact name
    (same resolution as the detail route). 404 when the graph is unknown
    or outside the caller's tier (existence is not disclosed).
    """
    registry = _registry(request)
    try:
        content = registry.graph_content(
            graph_id, node_limit=node_limit, edge_limit=edge_limit
        )
    except KeyError:
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    scope = get_caller_scope(request)
    meta = registry.resolve_graph(graph_id)
    if not _graph_read_allowed(
        request,
        kind=str(content.get("kind") or ""),
        project_id=content.get("project_id")
        if isinstance(content.get("project_id"), str)
        else None,
        owner_id=str(content.get("owner_id") or ""),
        stored_access=meta.access if meta is not None else [],
        scope=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    return content


def _review_queue(request: Request) -> Any:
    from umrs.knowledge.review_queue import ReviewQueueStore

    return ReviewQueueStore(getattr(request.app.state, "umrs_db_pool", None))


@router.get("/{graph_id}/proposed")
async def list_proposed_rows(
    graph_id: str,
    request: Request,
    limit: int = Query(200, ge=1, le=1000),
    run_id: str | None = Query(None, description="Only the rows of this run"),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Proposed (D-35) nodes and edges of one graph, newest first (K1.5).

    The review inbox reads this to show what waits for a person to
    establish. Read-only; needs the verified caller (401 without), and the
    graph detail route's visibility (404 when the graph is unknown or
    outside the caller's reach). Each row names its origin and run, for
    "establish run" (ADR-084 D-2); ``run_id`` narrows to one run, so a
    run is reachable past the newest ``limit`` rows.
    """
    graph = _registry(request).resolve_graph(graph_id)
    scope = CallerScope(
        user_id=identity.user_id or "",
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    if graph is None or not _graph_read_allowed(
        request,
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        stored_access=graph.access,
        scope=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    rows = _review_queue(request).proposed_rows(graph, limit=limit, run_id=run_id)
    return {"graph_id": graph.graph_id, "project_id": graph.project_id, **rows}


@router.patch("/{graph_id}")
async def update_graph(
    graph_id: str,
    body: UpdateGraphRequest,
    request: Request,
) -> dict[str, Any]:
    """Update metadata or lifecycle status. Archiving is the only 'delete'.

    Tier rule: system graphs are admin-managed, user graphs owner-only,
    project graphs archive-able by project members.
    """
    enforce_registry_write_authorization(request)
    registry = _registry(request)
    existing = registry.get_graph(graph_id)
    if existing is None:
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    scope = get_caller_scope(request)
    if not graph_writable(
        kind=existing.kind,
        project_id=existing.project_id,
        owner_id=existing.owner_id,
        caller=scope,
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": "Graph lifecycle is outside the caller's access tier.",
            },
        )
    # Validate the ontology binding before any write (ADR-116 D-5).
    ontology_id, ontology_version = _patched_ontology(request, body, existing)
    if body.status is not None:
        if body.status not in GRAPH_STATUSES:
            raise_typed(
                PlatformError.knowledgegraphs(
                    "bad_request", f"Invalid status {body.status!r}"
                )
            )
        registry.set_status(graph_id, body.status)
    graph = registry.update_graph(
        graph_id,
        description=body.description,
        kind=body.kind,
        ontology_id=ontology_id,
        ontology_version=ontology_version,
    )
    assert graph is not None
    return graph.to_dict()


@router.get("/{graph_id}/access")
async def get_graph_access(graph_id: str, request: Request) -> dict[str, Any]:
    """The graph's ADR-080 access list (any caller who may read the graph).
    # auth: verified
    """
    registry = _registry(request)
    graph = registry.resolve_graph(graph_id)
    scope = get_caller_scope(request)
    if graph is None or not _graph_read_allowed(
        request,
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        stored_access=graph.access,
        scope=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    return {"graph_id": graph.graph_id, "access": list(graph.access)}


# ── ADR-118 D-8 (K-14): sharing outside the project ─────────────────────────
#
# Any graph writer could replace the access list until 2026-11-24, so one
# call exposed a sensitive project's graph to another project. Now an entry
# the request ADDS that reaches outside the graph's project (another
# project, a circle of another project, a same-project circle whose roster
# holds a non-member, ``system``, a user who is not a live member) needs the project owner or an admin AND a project whose
# ``data_class`` is ``non_sensitive``. Narrowing, keeping the stored entries
# and in-project audiences (the project itself, its circles, its members)
# stay open to every writer. Project-less graphs (personal, system) keep
# their own owner/admin ladder unchanged. Facts that cannot be read fail
# closed: no owner, no class, no membership.

_PROJECT_SHARE_FACTS_SQL = (
    "SELECT owner_id, data_class FROM projects WHERE project_id = %s"
)
_CIRCLE_ROSTER_SQL = (
    "SELECT cm.user_id FROM circle_members cm"
    " JOIN circles c ON c.project_id = cm.project_id AND c.circle_id = cm.circle_id"
    " WHERE cm.project_id = %s AND cm.circle_id = %s AND c.status = 'active'"
    " ORDER BY cm.user_id"
)
_SHARE_RULE = "ADR-118 D-8"


def _pool_fetch(request: Request, sql: str, params: tuple[Any, ...], *, many: bool) -> Any:
    """One read on the request pool; None when the pool is absent or the
    read fails (each caller decides what unreadable means: fail closed)."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return None
    conn = pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall() if many else cur.fetchone()
    except Exception:  # unreadable is answered by the caller (fail closed)
        logger.warning("share gate read failed (%s)", sql.split(" FROM ")[0], exc_info=True)
        return None
    finally:
        pool.putconn(conn)


def _project_share_facts(request: Request, project_id: str) -> tuple[str, str]:
    """(owner_id, data_class) of the graph's project; ``("", "")`` when the
    row is absent or unreadable."""
    row = _pool_fetch(request, _PROJECT_SHARE_FACTS_SQL, (project_id,), many=False)
    if not row:
        return "", ""
    return str(row[0] or ""), str(row[1] or "")


def _user_outside(request: Request, user: str, project_id: str, owner: str) -> bool:
    """A user is inside only as the project owner or an ADR-097 live member
    (``_access.user_in_project``, the one membership read; unreadable is
    outside)."""
    return user != owner and not user_in_project(request, user, project_id)


def _roster_outside(request: Request, ref: str, project_id: str, owner: str) -> bool:
    """A same-project circle is inside only when every member of its active
    roster is inside. Anyone may open a personal circle in any project and a
    circle manager may add any user (KN-S4 review, 2026-11-24), so the
    ``project/`` prefix alone proves nothing. An unreadable roster is outside."""
    roster = _circle_roster(request, ref)
    if roster is None:
        return True
    return any(_user_outside(request, user, project_id, owner) for user in roster)


def _entry_outside(request: Request, entry: str, project_id: str, owner: str) -> bool:
    """Does one access entry reach readers outside the graph's project?"""
    from umrs.shared.access_scopes import parse_access_entry

    scope = parse_access_entry(entry)
    if scope.kind == "project":
        return scope.id.partition("/")[0] != project_id
    if scope.kind == "circle":
        if scope.id.partition("/")[0] != project_id:
            return True
        return _roster_outside(request, scope.id, project_id, owner)
    if scope.kind == "user":
        return _user_outside(request, scope.id, project_id, owner)
    return True  # system: every authenticated caller


def _share_refusal(
    graph_id: str, project_id: str, outside: list[str], data_class: str, *, owner_ok: bool
) -> PlatformError:
    """The typed 403 naming why the outside entries are refused."""
    named = ", ".join(outside)
    if not owner_ok:
        return PlatformError.knowledgegraphs(
            "forbidden",
            f"sharing graph {graph_id} outside project {project_id!r} ({named}) is "
            f"limited to the project owner or an admin ({_SHARE_RULE})",
        )
    shown = data_class or "unknown"
    return PlatformError.knowledgegraphs(
        "forbidden",
        f"project {project_id!r} has data_class {shown!r}: its graphs are not shared "
        f"outside the project ({named}); only a non_sensitive project may ({_SHARE_RULE})",
    )


def _added_entries(access: list[str], stored: list[str]) -> list[str]:
    """Entries the request adds (stored ones are kept, not re-granted)."""
    return [entry for entry in access if entry not in stored]


def _is_project_owner_or_admin(identity: CallerIdentity, owner: str) -> bool:
    """A platform admin, or the caller named as the project row's owner."""
    user = identity.user_id or ""
    return bool(identity.is_admin) or bool(user and user == owner)


def _outside_entries(
    request: Request, added: list[str], project_id: str, owner: str
) -> list[str]:
    """The added entries that reach readers outside the graph's project."""
    return [e for e in added if _entry_outside(request, e, project_id, owner)]


def _enforce_share_gate(
    request: Request, graph: Any, access: list[str], identity: CallerIdentity
) -> None:
    """ADR-118 D-8: refuse added outside-project entries unless the caller
    is the project owner or an admin and the project is non_sensitive."""
    project_id = graph.project_id or ""
    added = _added_entries(access, graph.access)
    if not project_id or not added:
        return
    owner, data_class = _project_share_facts(request, project_id)
    owner_ok = _is_project_owner_or_admin(identity, owner)
    if owner_ok and data_class == "non_sensitive":
        return
    outside = _outside_entries(request, added, project_id, owner)
    if outside:
        raise_typed(
            _share_refusal(graph.graph_id, project_id, outside, data_class, owner_ok=owner_ok)
        )


def _circle_roster(request: Request, ref: str) -> list[str] | None:
    """Members of an active circle (``project/circle``); None = unreadable."""
    project_id, _, circle_id = ref.partition("/")
    rows = _pool_fetch(request, _CIRCLE_ROSTER_SQL, (project_id, circle_id), many=True)
    return None if rows is None else [str(r[0]) for r in rows]


def _circle_readers(request: Request, ref: str, project_id: str) -> dict[str, Any]:
    """A circle entry's readers. On a circle of the graph's OWN project the
    KN-S4b read gate admits only project insiders, so roster members outside
    the project are listed as ``not_admitted``, not as readers."""
    roster = _circle_roster(request, ref)
    if roster is None or not project_id or ref.partition("/")[0] != project_id:
        return {"users": roster}
    admitted = [u for u in roster if user_in_project(request, u, project_id)]
    return {"users": admitted, "not_admitted": [u for u in roster if u not in admitted]}


def _entry_readers(request: Request, entry: str, project_id: str) -> dict[str, Any]:
    """Who one stored access entry admits (circle rosters resolved)."""
    from umrs.shared.access_scopes import parse_access_entry

    scope = parse_access_entry(entry)
    row: dict[str, Any] = {"scope": entry, "kind": scope.kind}
    if scope.kind == "user":
        row["users"] = [scope.id]
    elif scope.kind == "circle":
        row.update(_circle_readers(request, scope.id, project_id))
    elif scope.kind == "project":
        row["audience"] = f"live members of project {scope.id}"
    else:
        row["audience"] = "every authenticated caller"
    return row


def _effective_readers(request: Request, graph: Any) -> dict[str, Any]:
    """ADR-118 D-8: the readers a graph now has — its tier ladder plus each
    access entry, circle rosters included."""
    if graph.project_id:
        tier = f"live members of project {graph.project_id}"
    elif (graph.kind or "").strip().lower() == "system":
        tier = "every authenticated caller"
    else:
        tier = f"its owner {graph.owner_id}"
    return {
        "tier": tier,
        "entries": [
            _entry_readers(request, entry, graph.project_id or "") for entry in graph.access
        ],
    }


@router.put("/{graph_id}/access")
async def set_graph_access(
    graph_id: str,
    body: AccessUpdate,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Replace the graph's access list (write-tier owners or admin; audited).

    The list names who may READ the graph beyond its tier ladder;
    ``system`` entries are admin-only. Lifecycle writes are unaffected.
    Adding an audience outside the graph's project needs the project owner
    or an admin and a ``non_sensitive`` project (ADR-118 D-8, 2026-11-24);
    the response carries ``effective_readers``.
    # auth: verified
    """
    enforce_registry_write_authorization(request)
    registry = _registry(request)
    graph = registry.resolve_graph(graph_id)
    scope = CallerScope(
        user_id=identity.user_id or "",
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    if graph is None or not graph_writable(
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        caller=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    caller = resource_caller(
        request,
        user_id=identity.user_id,
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    access = checked_access(body.access, caller, error=PlatformError.knowledgegraphs)
    _enforce_share_gate(request, graph, access, identity)
    updated = registry.set_access(graph.graph_id, access)
    from .audit_events import emit_audit

    emit_audit(
        "knowledgegraph.access_changed",
        project_id=graph.project_id or "",
        details={"graph_id": graph.graph_id, "access": access, "by": scope.user_id},
    )
    result = updated or graph
    return {**result.to_dict(), "effective_readers": _effective_readers(request, result)}


class EstablishResponse(BaseModel):
    node_id: str | None = None
    edge_id: str | None = None
    status: str
    established_by: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float | None = None


class EstablishBody(BaseModel):
    """D-35 evidence refs on tiered rows: the human's promotion carries
    the evidence that justified it (+ optionally the settled confidence).

    Both optional — absent means the review block records the actor only
    (never an invented ref or confidence)."""

    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float | None = Field(None, ge=0.0, le=1.0)


def _refuse_agent_promotion(request: Request) -> None:
    """D-35: proposed → established is a human decision, never the agent's."""
    if request_actor(request) == AGENT_ACTOR:
        raise_typed(
            PlatformError.knowledgegraphs(
                "forbidden",
                "Knowledge promotion (proposed → established) is a human "
                "decision; agents propose, humans establish.",
            )
        )


# Tier rows the establish routes promote: kind -> (table, key column, label).
_ESTABLISH_ROWS = {
    "node": ("knowledge_nodes", "node_id", "Node"),
    "edge": ("knowledge_edges", "edge_id", "Edge"),
}


async def _knowledge_pool(request: Request) -> Any:
    from . import _asyncpg_pool

    pool = await _asyncpg_pool.knowledge_asyncpg_pool(request)
    if pool is None:
        raise_typed(
            PlatformError.knowledgegraphs("unavailable", "Knowledge store unavailable")
        )
    return pool


def _knowledge_store(pool: Any) -> KnowledgeStore:
    from umrs.knowledge.store import KnowledgeStore

    return KnowledgeStore(pool)


async def _row_placement(pool: Any, kind: str, row_id: str) -> Any:
    """The (project_id, graph_id) of one node/edge row, or None."""
    table, key, _label = _ESTABLISH_ROWS[kind]
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            f"SELECT project_id, graph_id FROM {table} WHERE {key} = $1", row_id
        )


def _row_in_graph(row: Any, graph_id: str) -> bool:
    """Legacy rows (graph_id '') belong to their project's default graph."""
    stored = str(row["graph_id"] or "")
    if stored:
        return stored == graph_id
    return graph_id_is_default(graph_id, str(row["project_id"] or ""))


def _graph_admits_write(
    request: Request, graph_id: str, row_project: str, scope: CallerScope
) -> bool:
    """May the caller write the graph? A default graph not yet materialized
    in the registry follows its project."""
    meta = _registry(request).get_graph(graph_id)
    if meta is None:
        return graph_id_is_default(graph_id, row_project) and project_writable(
            row_project, scope
        )
    return graph_writable(
        kind=meta.kind, project_id=meta.project_id, owner_id=meta.owner_id, caller=scope
    )


async def _require_establishable(
    request: Request,
    pool: Any,
    *,
    graph_id: str,
    kind: str,
    row_id: str,
    identity: CallerIdentity,
) -> None:
    """404 unless the row is in ``graph_id`` and the caller may write it.

    The establish routes ignored ``graph_id`` and promoted any project's
    node or edge (H-1 A.2, 2026-10-17); a foreign row reads as not found.
    """
    scope = CallerScope(
        user_id=identity.user_id or "",
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    row = await _row_placement(pool, kind, row_id)
    if (
        row is None
        or not _row_in_graph(row, graph_id)
        or not _graph_admits_write(
            request, graph_id, str(row["project_id"] or ""), scope
        )
    ):
        label = _ESTABLISH_ROWS[kind][2]
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"{label} {row_id} not found")
        )


@router.post("/{graph_id}/nodes/{node_id}/establish", response_model=EstablishResponse)
async def establish_node(
    graph_id: str,
    node_id: str,
    request: Request,
    body: EstablishBody | None = None,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> EstablishResponse:
    """Promote one node proposed → established (D-35 two-tier knowledge).

    Human-only (agent actor refused by name); idempotent on established
    rows; 404 when the node does not exist, is not in ``graph_id`` or the
    caller may not write that graph. The optional body carries
    the promotion's evidence refs and settled confidence (D-35 tail:
    evidence links on tiered rows).
    """
    _refuse_agent_promotion(request)
    pool = await _knowledge_pool(request)
    await _require_establishable(
        request, pool, graph_id=graph_id, kind="node", row_id=node_id, identity=identity
    )
    store = _knowledge_store(pool)
    established_by = (identity.user_id or "user").strip() or "user"
    refs = list(body.evidence_refs) if body else []
    confidence = body.confidence if body else None
    ok = await store.establish_node(
        node_id,
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )
    if not ok:
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Node {node_id} not found")
        )
    return EstablishResponse(
        node_id=node_id,
        status="established",
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )


@router.post("/{graph_id}/edges/{edge_id}/establish", response_model=EstablishResponse)
async def establish_edge(
    graph_id: str,
    edge_id: str,
    request: Request,
    body: EstablishBody | None = None,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> EstablishResponse:
    """Promote one edge proposed → established (same rules as nodes)."""
    _refuse_agent_promotion(request)
    pool = await _knowledge_pool(request)
    await _require_establishable(
        request, pool, graph_id=graph_id, kind="edge", row_id=edge_id, identity=identity
    )
    store = _knowledge_store(pool)
    established_by = (identity.user_id or "user").strip() or "user"
    refs = list(body.evidence_refs) if body else []
    confidence = body.confidence if body else None
    ok = await store.establish_edge(
        edge_id,
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )
    if not ok:
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Edge {edge_id} not found")
        )
    return EstablishResponse(
        edge_id=edge_id,
        status="established",
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )
