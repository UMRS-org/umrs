"""UMRS API routes for graph-extension proposals (multi-graph Phase 10/11).

Routes:

  POST /umrs/knowledge/graphs/{graph_id}/proposals     — stage a proposal (pending)
  GET  /umrs/knowledge/graphs/{graph_id}/proposals     — list proposals for a graph
  GET  /umrs/knowledge/proposals/{proposal_id}         — proposal detail
  POST /umrs/knowledge/proposals/{proposal_id}/decision — user approve/decline

Only the *decision* authorizes mutation: agent-originated mutations are then
accepted by the server-side guard (``_knowledge_mutation_guard``) and the
mutation call marks the proposal executed with provenance.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from .typed_errors import PlatformError, raise_typed

from umrs.knowledge.graph_registry import GraphRegistryStore, ProposalStore

from .jobs import CallerIdentity, get_caller_identity

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


def _require_graph(request: Request, graph_id: str) -> None:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.proposals("unavailable", "Database unavailable"))
    if GraphRegistryStore(pool).resolve_graph(graph_id) is None:
        raise_typed(PlatformError.proposals("not_found", f"Graph {graph_id} not found"))


@router.post("/graphs/{graph_id}/proposals", status_code=201)
async def create_proposal(
    graph_id: str,
    body: CreateProposalRequest,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Stage a graph-extension proposal (pending — no mutation yet)."""
    _require_graph(request, graph_id)
    proposal = _proposal_store(request).create_proposal(
        graph_id,
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
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """List proposals for one graph (newest first)."""
    _require_graph(request, graph_id)
    proposals = _proposal_store(request).list_proposals(
        target_graph_id=graph_id, status=status, limit=limit
    )
    return {"proposals": [p.to_dict() for p in proposals], "total": len(proposals)}


@router.get("/proposals/{proposal_id}")
async def get_proposal(proposal_id: str, request: Request) -> dict[str, Any]:
    """Fetch one proposal by id."""
    proposal = _proposal_store(request).get_proposal(proposal_id)
    if proposal is None:
        raise_typed(PlatformError.proposals("not_found", f"Proposal {proposal_id} not found"))
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
    store = _proposal_store(request)
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise_typed(PlatformError.proposals("not_found", f"Proposal {proposal_id} not found"))
    if proposal.status != "pending":
        raise_typed(
            PlatformError.proposals(
                "conflict",
                { "error": "proposal_already_decided", "proposal_id": proposal_id, "status": proposal.status, }
            ),
        )
    decided_by = identity.user_id or "user"
    updated = store.decide(proposal_id, approved=body.approved, decided_by=decided_by)
    assert updated is not None
    return updated.to_dict()
