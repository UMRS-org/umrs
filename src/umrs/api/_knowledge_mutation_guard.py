"""Server-side authorization for persistent knowledge-graph mutations.

Companion/agent callers identify themselves via the ``x-r4-actor: agent``
header (+ ``x-r4-session-id``). Human-initiated traffic (CLI, webview
buttons, direct API with actor absent or ``user``) keeps the pre-existing
behavior: no proposal required — the human action itself is authorization.

Agent-originated mutations MUST carry an *approved* graph-extension proposal
(migration 0069) targeting the graph they mutate. This is the server-side
enforcement that client modes (TRUST/autopilot) cannot bypass.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request

from umrs.knowledge.graph_registry import ProposalStore

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

AGENT_ACTOR = "agent"


def request_actor(request: Request) -> str:
    """Actor identity: ``user`` unless explicitly marked as an agent."""
    return (request.headers.get("x-r4-actor") or "user").strip().lower() or "user"


def request_session_id(request: Request) -> str:
    """Session id accompanying agent requests (empty when absent)."""
    return (request.headers.get("x-r4-session-id") or "").strip()


def _proposal_store(request: Request) -> ProposalStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.knowledge_guard("unavailable", "Database unavailable")
        )
    return ProposalStore(pool)


def enforce_graph_mutation_authorization(
    request: Request,
    *,
    target_graph_id: str,
    proposal_id: str | None,
) -> None:
    """Raise unless the request may mutate persistent graph content.

    Rules:
      * human actor → allowed (explicit action is the authorization)
      * agent actor → requires an approved proposal for *this* graph

    Raises:
        HTTPException 409: proposal required / not approved / target mismatch
        HTTPException 404: proposal not found
    """
    if request_actor(request) != AGENT_ACTOR:
        return
    if not proposal_id:
        raise_typed(
            PlatformError.knowledge_guard(
                "conflict",
                {
                    "error": "proposal_required",
                    "target_graph_id": target_graph_id,
                    "message": (
                        "Agent-originated knowledge-graph mutation requires an "
                        "approved graph-extension proposal. Create one via "
                        "POST /umrs/knowledge/graphs/{graph_id}/proposals and have "
                        "the user approve it before mutating."
                    ),
                },
            ),
        )
    store = _proposal_store(request)
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise_typed(
            PlatformError.knowledge_guard(
                "not_found", {"error": "proposal_not_found", "proposal_id": proposal_id}
            ),
        )
    if proposal.target_graph_id != target_graph_id:
        raise_typed(
            PlatformError.knowledge_guard(
                "conflict",
                {
                    "error": "proposal_target_mismatch",
                    "proposal_id": proposal_id,
                    "proposal_target": proposal.target_graph_id,
                    "requested_target": target_graph_id,
                },
            ),
        )
    if proposal.status != "approved":
        raise_typed(
            PlatformError.knowledge_guard(
                "conflict",
                {
                    "error": "proposal_not_approved",
                    "proposal_id": proposal_id,
                    "status": proposal.status,
                },
            ),
        )


def enforce_registry_write_authorization(request: Request) -> None:
    """Forbid agent-originated graph-registry writes (create/update/archive).

    Graph lifecycle is a human decision: proposals must reference existing
    graphs, so the proposal flow cannot authorize creation; and an archived
    graph would silently drop it from selection. Agents extend graphs through
    proposals — a human creates or archives them.
    """
    if request_actor(request) != AGENT_ACTOR:
        return
    raise_typed(
        PlatformError.knowledge_guard(
            "forbidden",
            {
                "error": "registry_write_forbidden_for_agent",
                "message": (
                    "Agents may not create, update, or archive knowledge graphs. "
                    "Propose changes via "
                    "POST /umrs/knowledge/graphs/{graph_id}/proposals — a human "
                    "decides; a human manages graph lifecycle."
                ),
            },
        ),
    )


def enforce_human_delete_authorization(request: Request, *, operation: str) -> None:
    """Forbid agent-originated knowledge deletes (ADR-093 MD-16).

    Deleting an artifact destroys rows no proposal can bring back, and the
    proposal flow (0069) authorizes extensions, not removals: an agent holds
    no approved user decision that could cover a delete. Like graph
    lifecycle (:func:`enforce_registry_write_authorization`), it is the
    person's act — the CLI or a direct API call — never a companion tool's.
    """
    if request_actor(request) != AGENT_ACTOR:
        return
    raise_typed(
        PlatformError.knowledge_guard(
            "forbidden",
            {
                "error": "delete_forbidden_for_agent",
                "operation": operation,
                "message": (
                    "Agents may not delete knowledge. Ask the user to run "
                    "`r4 knowledge delete <artifact_id>` themselves."
                ),
            },
        ),
    )


def mark_proposal_executed(
    request: Request,
    proposal_id: str | None,
    *,
    actor: str,
    session_id: str,
    operation: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Record execution provenance on the proposal (audit anchor, 0069).

    Failure-isolated: provenance write problems never fail the mutation
    that already succeeded.
    """
    if not proposal_id:
        return
    try:
        execution: dict[str, Any] = {
            "actor": actor,
            "session_id": session_id,
            "operation": operation,
        }
        if extra:
            execution.update(extra)
        _proposal_store(request).mark_executed(proposal_id, execution)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to record proposal execution %s", proposal_id)
