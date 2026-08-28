"""UMRS Research Loop REST API.

Routes for managing autonomous research loop lifecycle:
- Start / pause / resume / stop a loop
- List loop iterations
- Query loop status and results

Acceptance (TASK-036 / research-workflow spec §12):
  - Loop criteria enforced: unapproved projects refuse to start (returns 403)
  - Loop status, iterations, results always queryable
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.knowledge.loop import (
    AutonomousResearchLoop,
    LoopStatus,
    ResearchLoop,
)
from umrs.knowledge.loop_store import LoopStore
from umrs.knowledge.project import ResearchProject

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/loops", tags=["loops"])

# Store defaults to in-memory (identical behavior to previous dict); the
# server upgrades it to the Postgres-backed LoopStore at boot via
# configure_loop_store(pool). State survives restarts from that point on.
_loop_store: LoopStore = LoopStore.memory()
_loop_engine: AutonomousResearchLoop | None = None


def configure_loop_store(pool: Any) -> None:
    """Upgrade the loop store to Postgres persistence at server boot.

    Memory-resident loops (test runs, non-configured startups) are preserved:
    they are written through to Postgres so ongoing work is not lost.
    """
    global _loop_store
    incoming = LoopStore(pool)
    for live in _loop_store.list_all():
        try:
            incoming.save(live)
        except Exception as exc:
            logger.warning("loop migration on boot skipped %s: %s", live.loop_id, exc)
    _loop_store = incoming
    logger.info(
        "loop store upgraded to Postgres (%d loops carried)",
        len(_loop_store.list_all()),
    )


def configure_loop_store_sync(dsn: str) -> bool:
    """Upgrade to psycopg2-backed persistence (works from any thread).

    Same semantics as configure_loop_store but the connection is psycopg2:
    loop facades must be callable from the REST event loop and from runner
    threads; asyncpg pools are not loop-portable.
    Returns True when the sync pool backed store is now active.
    """
    global _loop_store
    import psycopg2

    conn = psycopg2.connect(dsn)
    incoming = LoopStore.sync_pg(conn)
    for live in _loop_store.list_all():
        try:
            incoming.save(live)
        except Exception as exc:  # noqa: BLE001
            logger.warning("loop migration on boot skipped %s: %s", live.loop_id, exc)
    _loop_store = incoming
    logger.info(
        "loop store upgraded to sync Postgres (%d loops carried)",
        len(_loop_store.list_all()),
    )
    return True


def reregister_loop_drivers() -> int:
    """Restart rehydration: re-spawn drivers for RUNNING loops in the store.

    Driver kind from loop.metadata: ``agent`` marker → agent runner; a
    ``driver=engine`` marker (or neither — legacy engine loops) → engine
    runner. Called from the http sidecar lifespan after recorder wiring and
    store upgrade so durable loops actually resume across restarts (before,
    only the docstring claimed this).
    """
    from umrs.knowledge import agent_loop_runner, engine_loop_runner

    resumed = 0
    for loop in _loop_store.list_all():
        if loop.status not in (LoopStatus.RUNNING, LoopStatus.PAUSED):
            continue
        try:
            if loop.metadata.get("agent"):
                from umrs.knowledge.project import ResearchProject

                project = ResearchProject(project_id=loop.project_id)
                agent_loop_runner.register(
                    loop, executor=_agent_executor_for(project), store=_loop_store
                )
            else:
                engine_loop_runner.register(
                    loop,
                    engine_factory=_engine_factory_from(_get_engine()),
                    store=_loop_store,
                )
            resumed += 1
        except Exception as exc:  # noqa: BLE001 — one loop must not block boot
            logger.warning(
                "loop driver re-registration failed for %s: %s", loop.loop_id, exc
            )
    if resumed:
        logger.info("re-registered %d loop driver(s) after restart", resumed)
    return resumed


def _get_engine() -> AutonomousResearchLoop:
    global _loop_engine
    if _loop_engine is None:
        _loop_engine = AutonomousResearchLoop(workflow_port=_default_workflow_port())
    return _loop_engine


def _default_workflow_port():
    """UMRS-side workflow port when a durable job store is available.

    Returns None without a store: the engine then fails closed in real mode
    (used to silently produce fake RUNNING-not-driven behavior; now the
    runner surfaces the honest error on the loop record)."""
    from umrs.api.jobs import get_optional_job_store

    store = get_optional_job_store()
    if store is None:
        return None
    from umrs.knowledge.umrs_workflow_port import UmrsWorkflowExecutionPort

    return UmrsWorkflowExecutionPort(store)


def get_loop(loop_id: str) -> ResearchLoop:
    loop = _loop_store.get(loop_id)
    if loop is None:
        raise HTTPException(status_code=404, detail=f"Loop {loop_id} not found")
    return loop


# ── Models ─────────────────────────────────────────────────────────────────


class StartLoopRequest(BaseModel):
    project_id: str = Field(..., description="Project ID")
    research_goal: str = Field(..., min_length=1, description="Research goal")
    max_iterations: int = Field(10, ge=1, le=100, description="Max iterations")
    session_id: str = Field(
        "",
        description=(
            "Optional ADR-025 research session id. When set, the new loop is "
            "auto-registered under that session (404 when the session is missing)."
        ),
    )
    agentic: bool = Field(
        False,
        description=(
            "Wave-9 agentic mode: each iteration is one bounded agent turn "
            "(hypothesis/evidence/evaluation/next-step) on the site LLM, "
            "matching the r4 LocalRuntime executor semantics. Runs outside the "
            "engine's phased pipeline."
        ),
    )
    approved: bool = Field(
        False,
        description=(
            "Readiness approval flag. When True the server validates that "
            "the project's readiness report has been approved by a human "
            "before starting the loop."
        ),
    )
    mode: str = Field(
        "auto",
        description=(
            "Tool-grain permission mode: 'auto' (default) triggers workflows "
            "immediately; 'ask' parks each trigger for a decision via the "
            "approvals route. Mirrors the companion turn-attempt vocabulary."
        ),
    )


class LoopStatusResponse(BaseModel):
    loop_id: str
    project_id: str
    research_goal: str
    status: str
    phase: str
    current_iteration: int
    max_iterations: int
    total_hypotheses_tested: int
    total_experiments_run: int
    total_findings: int
    success_rate: float
    started_at: str
    completed_at: str
    metadata: dict[str, Any]


class IterationResponse(BaseModel):
    iteration_id: str
    iteration_number: int
    phase: str
    started_at: str
    completed_at: str
    hypothesis: str
    experiment_id: str
    analysis: str
    evaluation: str
    findings: list[str]
    status: str


class StartLoopResponse(BaseModel):
    loop_id: str
    project_id: str
    status: str
    phase: str
    message: str


class StopLoopResponse(BaseModel):
    loop_id: str
    status: str
    message: str


class ListLoopsResponse(BaseModel):
    loops: list[dict[str, str]]


# ── Readiness enforcement ─────────────────────────────────────────────────


def _enforce_readiness_approval(request: StartLoopRequest) -> None:
    """Check project readiness approval when approved=True.

    Raises HTTPException(403) if not approved and has artifacts.
    Fresh project (no artifacts) — trust the client.
    """
    if not request.approved:
        return

    # #208: fail closed. Previously a construction error set project=None and
    # skipped the gate, and a readiness-check error was swallowed — any
    # transient failure bypassed the 403 and started a loop on an unapproved
    # project with artifacts.
    try:
        project = ResearchProject(project_id=request.project_id)
        readiness = project.check_readiness()
        gate_applies = not readiness.user_approved
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Readiness gate unavailable for project {request.project_id}: "
                f"{exc}. Retry later; loops without a verifiable gate are denied "
                "while approved=False is not explicitly set."
            ),
        ) from exc

    if gate_applies and project is not None:
        has_artifacts = bool(project.knowledge_store.list_artifacts())
        if not has_artifacts:
            return  # Trust the client for fresh projects
        raise HTTPException(
            status_code=403,
            detail=(
                f"Research readiness not approved for project "
                f"{request.project_id}. "
                f"Approve with: r4 project approve {request.project_id}"
            ),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=StartLoopResponse,
    status_code=201,
    summary="Start a new research loop",
)
async def start_loop(request: StartLoopRequest) -> StartLoopResponse:
    """Start a new autonomous research loop.

    **Loop criteria enforcement (spec §12):**
    When approved=True, the engine verifies the project's readiness
    has been approved by a human. Unapproved projects receive 403.
    """
    _enforce_readiness_approval(request)

    engine = _get_engine()
    loop = engine.start_loop(
        project_id=request.project_id,
        research_goal=request.research_goal,
        max_iterations=request.max_iterations,
    )
    loop.metadata["mode"] = "ask" if request.mode == "ask" else "auto"
    # Snapshot the just-started state for the response: the memory store
    # shares this object with the driver thread, which may complete the loop
    # before this endpoint returns — the POST must still report "started".
    status_at_start = loop.status.value
    phase_at_start = loop.phase.value
    if request.session_id:
        # Durable session linkage visible on loop status/list payloads.
        loop.metadata.setdefault("session_id", request.session_id)
    _loop_store.save(loop)

    if request.session_id:
        # ADR-025 D3: auto-register loop membership under the given session.
        try:
            from umrs.api.research_sessions import get_session_store
            from umrs.research_sessions import SessionLoopLink

            store = get_session_store()
            if store is None:
                raise HTTPException(
                    status_code=503,
                    detail="Session store unavailable; cannot register loop under session",
                )
            if store.get(request.session_id) is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {request.session_id} not found",
                )
            store.attach_loop(
                SessionLoopLink(
                    session_id=request.session_id,
                    loop_id=loop.loop_id,
                    venue="umrs",
                    loop_role="auto",
                )
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — session attach must not break loop start
            logger.warning(
                "session attach skipped for loop %s (session %s): %s",
                loop.loop_id,
                request.session_id,
                exc,
            )

    if request.agentic:
        try:
            from umrs.knowledge.agent_loop_runner import register as _register_runner
            from umrs.knowledge.project import ResearchProject

            project = ResearchProject(project_id=request.project_id)
            _register_runner(
                loop, executor=_agent_executor_for(project), store=_loop_store
            )
            loop.metadata["agent"] = "umrs-litellm"
            _loop_store.save(loop)
        except Exception as exc:  # noqa: BLE001 — agentic must not break start
            logger.warning("agentic runner skipped for loop %s: %s", loop.loop_id, exc)
    else:
        # Non-agentic: the server drives the phased engine itself (daemon
        # thread, store-checkpointed, store-polled for pause/stop). Before
        # this, the loop was created RUNNING but only a client driving
        # engine.run could advance it — REST-started loops never iterated.
        try:
            from umrs.knowledge import engine_loop_runner as _engine_runner

            _engine_runner.register(
                loop,
                engine_factory=_engine_factory_from(engine),
                store=_loop_store,
            )
            loop.metadata["driver"] = "engine"
            _loop_store.save(loop)
        except Exception as exc:  # noqa: BLE001 — driver must not break start
            logger.warning("engine runner skipped for loop %s: %s", loop.loop_id, exc)

    return StartLoopResponse(
        loop_id=loop.loop_id,
        project_id=loop.project_id,
        status=status_at_start,
        phase=phase_at_start,
        message=f"Loop {loop.loop_id} started for project {request.project_id}",
    )


def _agent_executor_for(project: Any):
    """Build the UMRS-side agentic iteration executor for one project."""
    from umrs.knowledge.agent_loop_runner import default_executor_for_project

    return default_executor_for_project(project)


def _engine_factory_from(engine: AutonomousResearchLoop):
    """Fresh engine per loop mirroring the app engine's wiring.

    The module-global engine is only used for plan construction (planner is
    stateless); the runner gets its own instance so cancels never cross
    loops (one shared ``_cancel_requested`` flag would bleed across them).
    """

    def _make() -> AutonomousResearchLoop:
        return AutonomousResearchLoop(
            simulated=engine.simulated,
            workflow_port=getattr(engine, "_workflow_port", None),
            knowledge_store=getattr(engine, "_knowledge_store", None),
            graph_navigator=getattr(engine, "_graph_navigator", None),
        )

    return _make


@router.get(
    "",
    response_model=ListLoopsResponse,
    summary="List all research loops",
)
async def list_loops() -> ListLoopsResponse:
    loops = [
        {
            "loop_id": loop_obj.loop_id,
            "status": loop_obj.status.value,
            "phase": loop_obj.phase.value,
        }
        for loop_obj in _loop_store.list_all()
    ]
    return ListLoopsResponse(loops=loops)


@router.get(
    "/{loop_id}",
    response_model=LoopStatusResponse,
    summary="Get loop status",
)
async def get_loop_status(loop_id: str) -> LoopStatusResponse:
    loop = get_loop(loop_id)
    return LoopStatusResponse(
        loop_id=loop.loop_id,
        project_id=loop.project_id,
        research_goal=loop.research_goal,
        status=loop.status.value,
        phase=loop.phase.value,
        current_iteration=loop.current_iteration,
        max_iterations=loop.max_iterations,
        total_hypotheses_tested=loop.total_hypotheses_tested,
        total_experiments_run=loop.total_experiments_run,
        total_findings=loop.total_findings,
        success_rate=loop.success_rate,
        started_at=loop.started_at,
        completed_at=loop.completed_at,
        metadata=loop.metadata,
    )


@router.get(
    "/{loop_id}/iterations",
    response_model=list[IterationResponse],
    summary="List loop iterations",
)
async def list_iterations(
    loop_id: str,
    phase: str | None = Query(None, description="Filter by phase"),
    limit: int = Query(50, ge=1, le=500, description="Max iterations"),
) -> list[IterationResponse]:
    loop = get_loop(loop_id)
    iterations = loop.iterations
    if phase:
        iterations = [i for i in iterations if i.phase.value == phase]

    def _finding_text(f: Any) -> str:
        # Engine findings are dataclasses (statement+confidence); companions
        # write plain strings. Serialize honestly to text for the wire model.
        if isinstance(f, str):
            return f
        return getattr(f, "statement", str(f))

    return [
        IterationResponse(
            iteration_id=i.iteration_id,
            iteration_number=i.iteration_number,
            phase=i.phase.value,
            started_at=i.started_at,
            completed_at=i.completed_at,
            hypothesis=i.hypothesis,
            experiment_id=i.experiment_id,
            analysis=i.analysis,
            evaluation=i.evaluation,
            findings=[_finding_text(f) for f in i.findings],
            status=i.status,
        )
        for i in iterations[-limit:]
    ]


@router.post(
    "/{loop_id}/stop",
    response_model=StopLoopResponse,
    summary="Stop a running loop",
)
async def stop_loop(loop_id: str) -> StopLoopResponse:
    loop = get_loop(loop_id)
    loop.status = LoopStatus.STOPPED
    loop.completed_at = datetime.now(timezone.utc).isoformat()
    _loop_store.save(loop)  # PG-backed stores only persist via save()
    # Stop any live drivers promptly (store poll would catch up anyway).
    from umrs.knowledge.agent_loop_runner import gracefully_stop as _stop_agent
    from umrs.knowledge.engine_loop_runner import gracefully_stop as _stop_engine

    _stop_agent(loop_id)
    _stop_engine(loop_id)
    return StopLoopResponse(
        loop_id=loop_id,
        status=loop.status.value,
        message=f"Loop {loop_id} stopped",
    )


@router.post(
    "/{loop_id}/pause",
    response_model=StopLoopResponse,
    summary="Pause a running loop",
)
async def pause_loop(loop_id: str) -> StopLoopResponse:
    loop = get_loop(loop_id)
    loop.status = LoopStatus.PAUSED
    _loop_store.save(loop)  # PG-backed stores only persist via save()
    return StopLoopResponse(
        loop_id=loop_id,
        status=loop.status.value,
        message=f"Loop {loop_id} paused",
    )


@router.post(
    "/{loop_id}/resume",
    response_model=StartLoopResponse,
    status_code=200,
    summary="Resume a paused loop",
)
async def resume_loop(loop_id: str) -> StartLoopResponse:
    loop = get_loop(loop_id)
    if loop.status != LoopStatus.PAUSED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume loop in status {loop.status.value}; only paused loops can be resumed",
        )
    loop.status = LoopStatus.RUNNING
    _loop_store.save(loop)  # PG-backed stores only persist via save()
    return StartLoopResponse(
        loop_id=loop_id,
        project_id=loop.project_id,
        status=loop.status.value,
        phase=loop.phase.value,
        message=f"Loop {loop_id} resumed",
    )


def clear_loop_store() -> None:
    """Test utility — wipes the loop store. Memory mode only; pool-backed
    store requires explicit per-loop deletes.

    Also aborts live drivers: a zombie runner polling a cleared store sees
    its loop vanish and re-saves it on exit — polluting the next test.
    """
    try:
        from umrs.knowledge import agent_loop_runner, engine_loop_runner

        agent_loop_runner.stop_all()
        engine_loop_runner.stop_all()
    except Exception:  # noqa: BLE001 — reset must never raise
        logger.debug("runner sweep during store clear failed", exc_info=True)
    _loop_store.clear()
