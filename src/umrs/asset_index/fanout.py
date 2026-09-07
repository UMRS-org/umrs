"""Asset-index fan-out (Review F4).

Query-time composition across existing stores. No caching, no persistence —
every call re-reads; the subsystems stay authoritative. Each collector is
independently degradable: an unavailable store yields an IndexWarning and
drops that kind, never a whole response (mirrors the 503-per-surface
convention at the HTTP layer above it).

Lineage derivation per Review F4 (edges come from existing relations):
- artifact → parent artifact: artifact_registry lineage table
- job → artifacts: artifacts carry execution_id
- research_session → loops: session membership links
- loop → artifacts: loop iterations record produced artifact ids
  (ADR-036 D4, WP-05 — finding artifacts + workflow outputs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .models import AssetLineageEdge, AssetRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexWarning:
    """A kind that failed to index, with the reason (503-per-surface shape)."""

    kind: str
    reason: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "reason": self.reason}


# ── caller visibility (read-side authorization, Review F3/F4) ───────────────


def _visible_by_claim(claim: Any, caller: Any) -> bool:
    """Visibility of a parsed scope claim (fail closed for unknown kinds)."""
    if claim.kind == "system":
        return True
    if claim.kind == "project":
        return bool(getattr(caller, "project_id", "")) and claim.id == caller.project_id
    if claim.kind == "user":
        return bool(getattr(caller, "user_id", "")) and claim.id == caller.user_id
    return False


def _visible_row(row: AssetRow, caller: Any | None) -> bool:
    """Whether a caller may see this row.

    caller is a knowledge.access_scope.CallerScope (or None=model all-visible
    for in-process consumers). Non-admin callers see system rows, project
    rows of their verified project, and own user rows; group rows are hidden
    for non-admins until a group-membership authority exists (fail closed).
    """
    if caller is None or getattr(caller, "is_admin", False):
        return True
    if not row.scope:
        return False
    try:
        from umrs.shared.scope_grammar import parse as parse_scope

        return _visible_by_claim(parse_scope(row.scope, allow_legacy=True), caller)
    except Exception:
        return False


def _scope_of(*, scope: str | None = None, project_id: str | None = None,
              owner: str | None = None) -> str:
    """Derive the canonical scope string for a row (umrs.shared.scope_grammar).

    Subsystem rows carry different owner markers; this collapses to the
    canonical grammar (system / project:<id> / group:<id> / user:<id>).
    """
    if scope:
        from umrs.shared.scope_grammar import ScopeError, parse as parse_scope

        try:
            return str(parse_scope(scope, allow_legacy=True))
        except ScopeError:
            pass
    if project_id:
        return f"project:{project_id}"
    if owner:
        return f"user:{owner}"
    return "system"


def _collect(named: str, fn: Callable[[], Iterable[AssetRow]], warns: list[IndexWarning]) -> list[AssetRow]:
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001 — per-kind degradation is the contract
        logger.warning("asset index collector %s failed: %s", named, exc)
        warns.append(IndexWarning(kind=named, reason=str(exc)[:200]))
        return []


# ── per-kind collectors (all duck-typed on store surface) ────────────────────


def _artifact_lineage(store: Any, artifact_id: Any) -> list[AssetLineageEdge]:
    """1-hop lineage from the registry; empty (not failing) on query errors."""
    edges: list[AssetLineageEdge] = []
    try:
        for edge in store.lineage(artifact_id, depth=1):
            edges.append(
                AssetLineageEdge(
                    from_kind="artifact",
                    from_id=str(artifact_id),
                    to_kind="artifact",
                    to_id=str(edge.parent_artifact_id),
                    relation=str(edge.relation or "derived_from"),
                )
            )
    except Exception:  # noqa: BLE001 — lineage is derivable later
        return []
    return edges


def _artifact_row(store: Any, art: Any) -> AssetRow:
    edges = _artifact_lineage(store, art.id)
    if art.execution_id:
        edges.append(
            AssetLineageEdge(
                from_kind="artifact",
                from_id=str(art.id),
                to_kind="job",
                to_id=str(art.execution_id),
                relation="executed_by",
            )
        )
    return AssetRow(
        kind="artifact",
        asset_id=str(art.id),
        scope=_scope_of(project_id=art.project_id, owner=art.owner_id),
        status=getattr(art, "state", "registered") or "registered",
        title=str(art.name or ""),
        owner=str(art.owner_id or ""),
        lineage=tuple(edges),
        portable=True,  # INV-13 bundle export
        metadata={"artifact_type": str(art.artifact_type or ""), "uri": str(art.uri or "")},
    )


def _collect_artifacts(store: Any, project_id: str | None) -> list[AssetRow]:
    from umrs.artifact_registry.models import ArtifactQuery

    query = ArtifactQuery(project_id=project_id, limit=500)
    return [_artifact_row(store, art) for art in store.list(query)]


def _collect_models(store: Any, ctx: Any = None) -> list[AssetRow]:
    """model_registry listing; ctx = AccessContext when strict, else duck call."""
    models = store.list_models(ctx) if ctx is not None else store.list_models()
    rows: list[AssetRow] = []
    for model in models[:500]:
        raw_scope = getattr(model, "scope", "")
        meta_scope = getattr(raw_scope, "value", str(raw_scope))
        proj = str(getattr(model, "project_id", "") or "")
        owner = str(getattr(model, "owner_id", "") or "")
        rows.append(
            AssetRow(
                kind="model",
                asset_id=str(getattr(model, "model_id", "") or getattr(model, "id", "")),
                scope=_scope_of(scope=(f"project:{proj}" if meta_scope == "project" and proj else meta_scope), owner=owner),
                status=getattr(getattr(model, "status", ""), "value", "registered"),
                title=str(getattr(model, "name", "") or getattr(model, "model_id", "")),
                owner=owner,
                portable=True,  # model packages leave via .kb/bundle flows
                metadata={"scope_raw": meta_scope},
            )
        )
    return rows


def _collect_mcp_servers(store: Any) -> list[AssetRow]:
    rows: list[AssetRow] = []
    for server in store.list_servers()[:500]:
        rows.append(
            AssetRow(
                kind="mcp_server",
                asset_id=str(getattr(server, "server_id", "") or getattr(server, "id", "")),
                scope=_scope_of(owner=str(getattr(server, "owner_id", "") or "")),
                status=str(getattr(server, "status", "") or "registered"),
                title=str(getattr(server, "name", "") or ""),
                owner=str(getattr(server, "owner_id", "") or ""),
                metadata={"transport": str(getattr(server, "transport", "") or "")},
            )
        )
    return rows


def _collect_sessions(store: Any, project_id: str | None) -> list[AssetRow]:
    rows: list[AssetRow] = []
    for sess in store.list(project_id=project_id):
        edges = [
            AssetLineageEdge(
                from_kind="research_session",
                from_id=str(sess.session_id),
                to_kind="loop",
                to_id=str(link.loop_id),
                relation="contains",
            )
            for link in sess.loops
        ]
        rows.append(
            AssetRow(
                kind="research_session",
                asset_id=str(sess.session_id),
                scope=_scope_of(project_id=sess.project_id),
                status=str(sess.status),
                title=str(sess.title or sess.research_goal[:64]),
                lineage=tuple(edges),
                metadata={"goal": sess.research_goal},
            )
        )
    return rows


def _loop_artifact_edges(loop: Any) -> list[AssetLineageEdge]:
    """``loop --produced--> artifact`` for every artifact id its iterations recorded."""
    loop_id = str(getattr(loop, "loop_id", ""))
    seen: set[str] = set()
    edges: list[AssetLineageEdge] = []
    for iteration in getattr(loop, "iterations", None) or []:
        for artifact_id in getattr(iteration, "artifact_ids", None) or []:
            aid = str(artifact_id)
            if not aid or aid in seen:
                continue
            seen.add(aid)
            edges.append(
                AssetLineageEdge(
                    from_kind="loop",
                    from_id=loop_id,
                    to_kind="artifact",
                    to_id=aid,
                    relation="produced",
                )
            )
    return edges


def _loop_session_edges(loop: Any) -> list[AssetLineageEdge]:
    sid = str(getattr(loop, "research_session_id", "") or "")
    if not sid:
        return []
    return [
        AssetLineageEdge(
            from_kind="loop",
            from_id=str(getattr(loop, "loop_id", "")),
            to_kind="research_session",
            to_id=sid,
            relation="registered_by",
        )
    ]


def _loop_metadata(loop: Any) -> dict[str, str]:
    meta = getattr(loop, "metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    return {
        "venue": str(getattr(loop, "venue", "") or ""),
        "hosting": str(meta.get("hosting") or ""),
        "durable_job_id": str(meta.get("durable_job_id") or ""),
    }


def _loop_row(loop: Any) -> AssetRow:
    status = getattr(loop, "status", "")
    goal = getattr(loop, "research_goal", "") or getattr(loop, "goal", "") or ""
    return AssetRow(
        kind="loop",
        asset_id=str(getattr(loop, "loop_id", "")),
        scope=_scope_of(project_id=str(getattr(loop, "project_id", "") or "") or None),
        status=str(getattr(status, "value", status) or "unknown"),
        title=str(goal)[:64],
        lineage=tuple(_loop_session_edges(loop) + _loop_artifact_edges(loop)),
        metadata=_loop_metadata(loop),
    )


def _collect_loops(store: Any) -> list[AssetRow]:
    return [_loop_row(loop) for loop in store.list_all()[:500]]


def _job_row(job: Any) -> AssetRow:
    job_type = getattr(job, "job_type", "")
    job_state = getattr(job, "state", "") or getattr(job, "status", "")
    return AssetRow(
        kind="job",
        asset_id=str(getattr(job, "job_id", "") or getattr(job, "id", "")),
        scope=_scope_of(project_id=str(getattr(job, "project_id", "") or "") or None),
        status=getattr(job_state, "value", str(job_state or "unknown")),
        title=getattr(job_type, "value", str(job_type or "")),
        metadata={"authority": str(getattr(job, "execution_authority", "") or "")},
    )


def _collect_jobs(store: Any) -> list[AssetRow]:
    if not hasattr(store, "list_jobs"):
        return []
    return [_job_row(job) for job in store.list_jobs(limit=500)]


_COLLECTOR_PARAMS: dict[str, str] = {
    "artifact": "artifact_store",
    "model": "model_store",
    "mcp_server": "mcp_store",
    "research_session": "session_store",
    "loop": "loop_store",
    "job": "job_store",
}


def build_asset_index(
    *,
    project_id: str | None = None,
    kinds: Iterable[str] | None = None,
    artifact_store: Any = None,
    model_store: Any = None,
    mcp_store: Any = None,
    session_store: Any = None,
    loop_store: Any = None,
    job_store: Any = None,
    model_ctx: Any = None,
    caller: Any = None,
) -> tuple[list[AssetRow], list[IndexWarning]]:
    """Compose the cross-subsystem asset index.

    Every store is optional; a missing optional store simply yields no rows
    of its kind (NOT a warning — absent store ≠ failed store). A store that
    raises produces an IndexWarning and a row gap, never a raised error.

    ``caller`` scopes the result: rows invisible to the caller are filtered
    (see ``_visible_row``). ``None`` = in-process consumer, no filtering.
    """
    warnings: list[IndexWarning] = []
    stores = {
        "artifact": artifact_store,
        "model": model_store,
        "mcp_server": mcp_store,
        "research_session": session_store,
        "loop": loop_store,
        "job": job_store,
    }
    if model_ctx is None and model_store is not None:
        model_ctx = model_access_context(caller=caller)
    builders: dict[str, Callable[[], list[AssetRow]]] = {
        "artifact": lambda: _collect_artifacts(artifact_store, project_id),
        "model": lambda: _collect_models(model_store, model_ctx),
        "mcp_server": lambda: _collect_mcp_servers(mcp_store),
        "research_session": lambda: _collect_sessions(session_store, project_id),
        "loop": lambda: _collect_loops(loop_store),
        "job": lambda: _collect_jobs(job_store),
    }
    wanted = list(kinds) if kinds is not None else list(builders)
    rows: list[AssetRow] = []
    for kind in wanted:
        builder = builders.get(kind)
        if builder is None or stores.get(kind) is None:
            continue
        rows.extend(_collect(kind, builder, warnings))
    return [row for row in rows if _visible_row(row, caller)], warnings


# Model collectors need an AccessContext (strict store). It derives from the
# CALLER (never synthetic-admin in the HTTP path): in-process consumers pass
# no caller and get the full-visibility in-process context.
def model_access_context(
    user_id: str = "__asset_index__", caller: Any = None
) -> Any:
    """AccessContext for read-model fanout mirroring the caller's visibility."""
    from umrs.model_registry.store.acl import AccessContext

    if caller is None:
        return AccessContext(user_id=user_id, is_admin=True)
    return AccessContext(
        user_id=getattr(caller, "user_id", "") or user_id,
        is_admin=bool(getattr(caller, "is_admin", False)),
        project_memberships={getattr(caller, "project_id", "")}
        - {""},
    )
