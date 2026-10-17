"""UMRS API routes for graph-extension proposals (multi-graph Phase 10/11).

Routes:

  POST /umrs/knowledge/graphs/{graph_id}/proposals     — stage a proposal (pending)
  GET  /umrs/knowledge/graphs/{graph_id}/proposals     — list proposals for a graph
  GET  /umrs/knowledge/proposals/{proposal_id}         — proposal detail
  POST /umrs/knowledge/proposals/{proposal_id}/decision — user approve/decline

Only the *decision* authorizes mutation: agent-originated mutations are then
accepted by the server-side guard (``_knowledge_mutation_guard``) and the
mutation call marks the proposal executed with provenance.

KG-05 (2026-10-09): the decision refuses agent actors (ADR-084 D-3, as skills
review does), and every route answers only for a graph the caller may read
(ADR-080) — a decision also needs write access. Before, any caller could
read or decide any project's proposals, and an agent could approve its own
and then pass the mutation guard.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from umrs.knowledge.access_scope import CallerScope, graph_readable, graph_writable
from umrs.knowledge.graph_registry import (
    GraphRegistryStore,
    KnowledgeGraphMeta,
    ProposalStore,
)

from ..shared.access_scopes import access_admits
from ._access import graph_meta_readable, resource_caller
from ._knowledge_mutation_guard import AGENT_ACTOR, request_actor
from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge-proposals"])


class CreateProposalRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=4000)
    missing_information: list[Any] = Field(default_factory=list)
    source_artifacts: list[Any] = Field(default_factory=list)
    proposed_operations: list[Any] = Field(default_factory=list)
    session_id: str = ""


class ProposalDecisionRequest(BaseModel):
    approved: bool


def _proposal_store(request: Request) -> ProposalStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.proposals("unavailable", "Database unavailable"))
    return ProposalStore(pool)


def _scope(identity: CallerIdentity) -> CallerScope:
    return CallerScope(
        user_id=identity.user_id,
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )


def _readable_graph(
    request: Request, graph_id: str, identity: CallerIdentity
) -> KnowledgeGraphMeta | None:
    """The graph, if it exists and the caller may read it (ADR-080)."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.proposals("unavailable", "Database unavailable"))
    scope = _scope(identity)
    candidates = GraphRegistryStore(pool).graph_candidates(graph_id)
    return next((g for g in candidates if graph_meta_readable(request, g, scope)), None)


def _require_graph(
    request: Request, graph_id: str, identity: CallerIdentity
) -> KnowledgeGraphMeta:
    """The readable graph, or not_found: absent and not-yours look the same."""
    graph = _readable_graph(request, graph_id, identity)
    if graph is None:
        raise_typed(PlatformError.proposals("not_found", f"Graph {graph_id} not found"))
    return graph


def _visible_proposal(
    request: Request, store: ProposalStore, proposal_id: str, identity: CallerIdentity
) -> tuple[Any, KnowledgeGraphMeta]:
    """The proposal and its target graph, if the caller may read that graph."""
    proposal = store.get_proposal(proposal_id)
    graph = (
        _readable_graph(request, proposal.target_graph_id, identity)
        if proposal
        else None
    )
    if proposal is None or graph is None:
        raise_typed(
            PlatformError.proposals("not_found", f"Proposal {proposal_id} not found")
        )
    return proposal, graph


def _refuse_agent_decision(request: Request) -> None:
    """Agents propose, humans decide (ADR-084 D-3; skills review precedent)."""
    if request_actor(request) == AGENT_ACTOR:
        raise_typed(
            PlatformError.proposals(
                "forbidden",
                {
                    "error": "proposal_decision_forbidden_for_agent",
                    "detail": "a proposal is decided by a person, never by an agent",
                },
            )
        )


class _PerGraphStats(ProposalStore):
    """Proposal counts per target graph with the graph's access fields.

    One query (review 2026-10-18): the counts are joined with
    ``knowledge_graphs`` so the caller filters in Python instead of one
    registry lookup per graph. Proposals of a graph missing from the registry
    drop out of the inner join, as before.
    """

    def per_graph_counts(self) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT c.target_graph_id, c.accepted, c.declined, c.pending,"
                " g.kind, g.project_id, g.owner_id, g.access"
                " FROM (SELECT target_graph_id,"
                " COUNT(*) FILTER (WHERE status IN ('approved','executed')) AS accepted,"
                " COUNT(*) FILTER (WHERE status = 'declined') AS declined,"
                " COUNT(*) FILTER (WHERE status = 'pending') AS pending"
                " FROM knowledge_graph_proposals GROUP BY target_graph_id) c"
                " JOIN knowledge_graphs g ON g.graph_id = c.target_graph_id"
            )
            return list(cur.fetchall() or [])


def _stored_access(raw: Any) -> Any:
    """The row's access list; a JSON string column decodes like the registry."""
    return json.loads(raw) if isinstance(raw, str) else raw


def _row_readable(row: dict[str, Any], scope: CallerScope, caller: Any) -> bool:
    """``graph_meta_readable`` for a joined stats row, with one shared caller
    so the ADR-080 circle lookup runs at most once per request."""
    if graph_readable(
        kind=row.get("kind") or "",
        project_id=row.get("project_id"),
        owner_id=row.get("owner_id") or "",
        caller=scope,
    ):
        return True
    return access_admits(_stored_access(row.get("access")), caller)


def _readable_stats(request: Request, identity: CallerIdentity) -> dict[str, Any]:
    """``acceptance_stats`` summed over the graphs the caller may read."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.proposals("unavailable", "Database unavailable"))
    scope = _scope(identity)
    caller = resource_caller(
        request,
        user_id=scope.user_id,
        project_id=scope.project_id,
        is_admin=scope.is_admin,
    )
    totals = {"accepted": 0, "declined": 0, "pending": 0}
    for row in _PerGraphStats(pool).per_graph_counts():
        if _row_readable(row, scope, caller):
            for key in totals:
                totals[key] += int(row.get(key) or 0)
    decided = totals["accepted"] + totals["declined"]
    return {
        **totals,
        "decided": decided,
        "acceptance_rate": (totals["accepted"] / decided) if decided else None,
    }


@router.post("/graphs/{graph_id}/proposals", status_code=201)
async def create_proposal(
    graph_id: str,
    body: CreateProposalRequest,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Stage a graph-extension proposal (pending — no mutation yet)."""
    graph = _require_graph(request, graph_id, identity)
    proposal = _proposal_store(request).create_proposal(
        graph.graph_id,
        reason=body.reason,
        missing_information=body.missing_information,
        source_artifacts=body.source_artifacts,
        proposed_operations=body.proposed_operations,
        created_by=identity.user_id or "agent",
        session_id=body.session_id,
    )
    return proposal.to_dict()


@router.get("/graphs/{graph_id}/proposals")
async def list_proposals(
    graph_id: str,
    request: Request,
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """List proposals for one graph (newest first)."""
    graph = _require_graph(request, graph_id, identity)
    proposals = _proposal_store(request).list_proposals(
        target_graph_id=graph.graph_id, status=status, limit=limit
    )
    return {"proposals": [p.to_dict() for p in proposals], "total": len(proposals)}


@router.get("/proposals/stats")
async def proposal_stats(
    request: Request,
    graph_id: str | None = None,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Proposal acceptance rate (D-44): quality signal for the knowledge
    graph's semi-automatic growth — how often the loop's proposals survive
    the scientist's review. rate is null when nothing was ever decided
    (no fabricated 100%).

    Without ``graph_id`` it counted every graph's proposals (H-1 A.2,
    2026-10-17): a non-admin caller now gets the graphs it may read only.
    """
    if graph_id is not None:
        graph_id = _require_graph(request, graph_id, identity).graph_id
    elif not identity.is_admin:
        return _readable_stats(request, identity)
    return _proposal_store(request).acceptance_stats(target_graph_id=graph_id)


@router.get("/proposals/{proposal_id}")
async def get_proposal(
    proposal_id: str,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Fetch one proposal by id."""
    proposal, _graph = _visible_proposal(
        request, _proposal_store(request), proposal_id, identity
    )
    return proposal.to_dict()


@router.post("/proposals/{proposal_id}/decision")
async def decide_proposal(
    proposal_id: str,
    body: ProposalDecisionRequest,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Record the *user's* approval decision for a proposal.

    The actor recorded here is the verified caller identity, not the agent
    session — the decision is the human authorization boundary.
    """
    _refuse_agent_decision(request)
    store = _proposal_store(request)
    proposal, graph = _visible_proposal(request, store, proposal_id, identity)
    if not graph_writable(
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        caller=_scope(identity),
    ):
        raise_typed(
            PlatformError.proposals(
                "forbidden", f"Deciding needs write access to graph {graph.graph_id}"
            )
        )
    if proposal.status != "pending":
        raise_typed(
            PlatformError.proposals(
                "conflict",
                {
                    "error": "proposal_already_decided",
                    "proposal_id": proposal_id,
                    "status": proposal.status,
                },
            ),
        )
    decided_by = identity.user_id or "user"
    updated = store.decide(proposal_id, approved=body.approved, decided_by=decided_by)
    assert updated is not None
    return updated.to_dict()
