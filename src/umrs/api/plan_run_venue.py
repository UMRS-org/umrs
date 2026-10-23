"""Server-venue routes of a team plan run (ADR-092 slice 8).

Mounted under the traces family (``/umrs/traces/plan-runs/…``), which the
gateway already proxies (``/gw/traces/*``), the caller's own runs only:

    POST /umrs/traces/plan-runs/{run_id}/venue    start or resume it on the server venue
    GET  /umrs/traces/plan-runs/{run_id}/venue    its record and its worker's state
    POST /umrs/traces/plan-runs/{run_id}/cancel   stop its worker, cancel its record

The logic lives in ``umrs.venues.plan_runs`` (a foundation module); these
routes resolve the verified caller, the venue service and the project's
ADR-067 caps, and project refusals onto typed errors. A sandbox never
reaches them (the rail gw admits only its own run's record verbs).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..jobs import plan_run_records as prr
from ..jobs.store import JobStore
from ..session_policy import PolicyError
from ..venues import plan_runs
from ..venues.frontend_service import VenueError
from .jobs import CallerIdentity, get_caller_identity, get_job_store
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/traces/plan-runs", tags=["execution-traces"])


class PlanRunVenueStartBody(BaseModel):
    """A first start carries the run's definition; a resume may omit it."""

    definition: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    default_source: str = ""
    web_policy: str = ""
    web_allow_domains: list[str] = Field(default_factory=list)


def _venue_service(request: Request) -> Any:
    from .companion_venue import get_venue_service

    return get_venue_service(request)


def _caps(request: Request, project_id: str) -> dict[str, Any]:
    """The project's ADR-067 caps (tests inject ``app.state.plan_run_budget_reader``).

    The loop routes' reader: an unreadable row is uncapped, logged (as a loop
    start reads it).
    """
    reader = getattr(request.app.state, "plan_run_budget_reader", None)
    if reader is None:
        from .loop import _project_budget_caps as reader
    return plan_runs.caps_for(reader(project_id))


def _caller_scope(caller: CallerIdentity) -> tuple[str, str]:
    if caller.is_anonymous or not caller.user_id:
        raise_typed(
            PlatformError.auth("project_required", "A plan run needs a project")
        )
    return caller.user_id, caller.project_id


def _refused(exc: Exception, verb: str) -> None:
    if isinstance(exc, PolicyError):  # ADR-099: the web choice and the limits
        from .companion_venue import policy_typed

        policy_typed(exc)
    if isinstance(exc, VenueError):
        raise_typed(
            PlatformError.companionvenue(exc.code, exc.message, **exc.detail), cause=exc
        )
    if isinstance(exc, prr.PlanRunRecordError):
        raise_typed(PlatformError.jobs(exc.code, exc.message), cause=exc)
    logger.exception("plan run venue %s failed", verb)
    raise_typed(
        PlatformError.companionvenue(
            "unavailable", f"plan run venue {verb} failed: {type(exc).__name__}: {exc}"
        ),
        cause=exc,
    )


@router.post("/{run_id}/venue", status_code=202)
async def start_plan_run_on_venue(
    run_id: str,
    body: PlanRunVenueStartBody,
    request: Request,
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Start (or resume) a team plan run hosted by a frontend-venue worker.
    # auth: verified
    """
    owner, project = _caller_scope(caller)
    try:
        prr.check_run_id(run_id)
        service = _venue_service(request)
        caps = _caps(request, project)
        return await asyncio.to_thread(
            plan_runs.start,
            service,
            job_store,
            owner=owner,
            project=project,
            run_id=run_id,
            definition=body.definition,
            caps=caps,
            source_refs=body.source_refs,
            default_source=body.default_source,
            web_policy=body.web_policy,
            web_allow_domains=body.web_allow_domains,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — every refusal is named
        _refused(exc, "start")


@router.get("/{run_id}/venue")
async def plan_run_venue_status(
    run_id: str,
    request: Request,
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """A server-hosted run's record and its worker session's state.
    # auth: verified
    """
    owner, project = _caller_scope(caller)
    try:
        return plan_runs.status(
            _venue_service(request),
            job_store,
            owner=owner,
            project=project,
            run_id=run_id,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _refused(exc, "status")


@router.post("/{run_id}/cancel")
async def cancel_plan_run_on_venue(
    run_id: str,
    request: Request,
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Stop a server-hosted run's worker and cancel its record.
    # auth: verified
    """
    owner, project = _caller_scope(caller)
    try:
        return await asyncio.to_thread(
            plan_runs.cancel,
            _venue_service(request),
            job_store,
            owner=owner,
            project=project,
            run_id=run_id,
            actor=owner,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _refused(exc, "cancel")


# ── slice 8b: the server-hosted loop's plan-run host ───────────────────────


def build_loop_plan_host(
    pool: Any, *, service: Any = None, job_store: Any = None
) -> Any:
    """The venue verbs and the proposal lane a server-hosted plan loop uses
    (``umrs.knowledge.plan_iteration``), bound at boot over ``pool``.

    The same ``umrs.venues.plan_runs`` verbs the routes call, with the
    project's ADR-067 caps the route would read, and the existing proposal
    lane (migration 0069: a pending proposal a person decides, ADR-050).
    """
    from umrs.knowledge.plan_iteration import PlanRunHost

    from .companion_venue import build_venue_service

    venue = service or build_venue_service(pool)

    def _store() -> Any:
        if job_store is not None:
            return job_store
        from .jobs import get_optional_job_store

        store = get_optional_job_store()
        if store is None:
            raise RuntimeError("durable job store unavailable")
        return store

    def start(*, owner, project, run_id, definition, caps, **venue_args):
        from .loop import _project_budget_caps

        project_caps = plan_runs.caps_for(_project_budget_caps(project))
        merged = _tighter_caps(project_caps, caps)
        return plan_runs.start(
            venue,
            _store(),
            owner=owner,
            project=project,
            run_id=run_id,
            definition=definition,
            caps=merged,
            **venue_args,
        )

    def view(*, owner, project, run_id):
        return plan_runs.status(
            venue, _store(), owner=owner, project=project, run_id=run_id
        )

    def cancel(*, owner, project, run_id):
        return plan_runs.cancel(
            venue, _store(), owner=owner, project=project, run_id=run_id, actor="loop"
        )

    return PlanRunHost(
        start=start,
        view=view,
        cancel=cancel,
        propose=lambda **kw: propose_finding(pool, **kw),
        venue_missing=lambda: venue.config.missing(),
    )


def _tighter_caps(project: dict[str, Any], loop: dict[str, Any]) -> dict[str, Any]:
    walls = [
        int(c["max_wall_clock_sec"])
        for c in (project, loop)
        if c.get("max_wall_clock_sec")
    ]
    return {"max_wall_clock_sec": min(walls)} if walls else {}


def propose_finding(
    pool: Any,
    *,
    project: str,
    owner: str,
    reason: str,
    source_artifacts: list[str],
    operation: dict[str, Any],
) -> dict[str, Any]:
    """Stage one PENDING proposal on the project's first graph (0069 lane).

    Never decided here: a proposal is decided by a person (ADR-050; agents
    and loops propose, the decide route refuses agents).
    """
    from umrs.knowledge.graph_registry import GraphRegistryStore, ProposalStore

    graphs = GraphRegistryStore(pool).list_graphs(project_id=project, limit=1)
    if not graphs:
        return {
            "status": "not_proposed",
            "reason": f"project {project!r} has no knowledge graph to propose into",
        }
    proposal = ProposalStore(pool).create_proposal(
        graphs[0].graph_id,
        reason=reason,
        source_artifacts=list(source_artifacts),
        proposed_operations=[operation],
        created_by=owner or "research_loop",
    )
    return {
        "status": "proposed",
        "graph_id": graphs[0].graph_id,
        "proposal_id": proposal.proposal_id,
        "proposal_status": getattr(proposal, "status", "pending"),
    }
