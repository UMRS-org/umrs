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
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from umrs.knowledge import loop_hosting, plan_iteration
from umrs.knowledge.loop import (
    AutonomousResearchLoop,
    LoopPhase,
    LoopStatus,
    ResearchLoop,
)
from umrs.knowledge.loop_store import LoopStore
from umrs.knowledge.project import ResearchProject

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/loops", tags=["loops"])

# Store defaults to in-memory (identical behavior to previous dict); the
# server upgrades it to the Postgres-backed LoopStore at boot via
# configure_loop_store(pool). State survives restarts from that point on.
_loop_store: LoopStore = LoopStore.memory()
_loop_engine: AutonomousResearchLoop | None = None


def _finding_sink():
    """Knowledge-store sink for loop findings when a knowledge DSN exists.

    None (reported on the loop as ``finding_ingest.sink == "none"``) when the
    process has no knowledge database — never a silent no-op.
    """
    from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn
    from umrs.knowledge.finding_ingest import KnowledgeFindingSink

    return KnowledgeFindingSink() if _resolve_knowledge_dsn() else None


def _catalog_pinner():
    """D-43/D-52 findings pin: single-active-catalog append via the jobs pool.

    None when no job service/pool is up (loop_hosting reports the skip);
    never a silent no-op on a wired host.
    """

    from umrs.jobs.service import get_job_service

    svc = get_job_service()
    pool = getattr(svc, "db_pool", None) if svc is not None else None
    if pool is None:
        logger.warning("catalog pinner: no jobs pool — finding pins skipped")
        return None
    return lambda project_id, pins: _pin_findings_onto_catalog(pool, project_id, pins)


def _pin_findings_onto_catalog(pool, project_id: str, pins) -> None:
    """Append kind=finding pins; single-active-catalog rule, deduped.

    Zero catalogs → nothing to associate against; several → ambiguous,
    never silently guessed (pin by hand with `r4 catalog pin`).
    """
    from umrs.scientific_catalogs.store import ScientificCatalogStore

    conn = pool.getconn()
    try:
        store = ScientificCatalogStore(conn)
        catalogs = store.list(project_id=project_id)
        if len(catalogs) != 1:
            return
        catalog_id = catalogs[0]["catalog_id"]
        already = {
            p.get("ref")
            for p in (store.get(catalog_id).get("provenance") or {}).get("pins", [])
        }
        for ref, note in pins:
            if ref in already:
                continue
            store.add_pin(catalog_id, kind="finding", ref=ref, note=note)
    finally:
        pool.putconn(conn)


def _checkpoint_store():
    """Singleton CheckpointStore accessor (ADR-036).

    Rehomed here by ADR-053: the unmounted ``api/checkpoints`` router was
    the previous holder; the jobs store is the authority now. Same process
    singleton; the DSN resolves like the loop store's since H-10.1.
    """
    global _checkpoint_store_singleton
    if _checkpoint_store_singleton is None:
        from umrs.jobs.checkpoint_store import CheckpointStore

        from ._asyncpg_pool import sync_store_dsn

        # H-10.1: the same DSN chain as the loop store (umrs-front sets only
        # the component vars, so DATABASE_URL/UMRS_DB_URL alone meant memory).
        _checkpoint_store_singleton = CheckpointStore(db_url=sync_store_dsn())
    return _checkpoint_store_singleton


_checkpoint_store_singleton = None


def _configure_hosting() -> None:
    """Wire the loop-hosting seam (ADR-036 D1) with late-bound accessors.

    The ``research_loop_job`` backend (jobs/) drives loops through
    ``umrs.knowledge.loop_hosting``; this module owns the store/engine and
    therefore installs the getters (resolved per call — the store is swapped
    to Postgres at boot).
    """
    loop_hosting.configure(
        loop_hosting.HostingWiring(
            store=lambda: _loop_store,
            engine_factory=lambda: _engine_factory_from(_get_engine()),
            agent_executor=lambda project_id: _agent_executor_for(
                ResearchProject(project_id=project_id)
            ),
            checkpoint_store=_checkpoint_store,
            finding_sink=_finding_sink,
            catalog_pinner=_catalog_pinner,
        )
    )


_configure_hosting()


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
        except Exception as exc:  # noqa: BLE001 — one loop must not block boot
            logger.warning("loop migration on boot skipped %s: %s", live.loop_id, exc)
    _loop_store = incoming
    logger.info(
        "loop store upgraded to Postgres (%d loops carried)",
        len(_loop_store.list_all()),
    )


def _apply_project_ask_policy(http_request: Request, request: Any) -> Any:
    """D-33: inherit the project's persisted default when mode is unset.

    Explicit mode wins (auto/ask/profile); '' / absent → the project row's
    ask_mode + pause_triggers; project row absent/unreadable → the request
    unchanged (never invented).
    """
    if (request.mode or "").strip():
        return request
    row = _project_row(http_request, request.project_id)
    updates = _ask_policy_updates(row) if row is not None else None
    return request.model_copy(update=updates) if updates else request


def _project_row(http_request: Request, project_id: str) -> dict[str, Any] | None:
    """The durable project row, or None (no pool / unreadable / absent)."""
    pool = getattr(http_request.app.state, "umrs_db_pool", None)
    if pool is None:
        return None
    conn = pool.getconn()
    try:
        from .projects import ProjectStore

        return ProjectStore(conn).get(project_id)
    except Exception:  # noqa: BLE001 — the default never blocks a start
        return None
    finally:
        pool.putconn(conn)


def _ask_policy_updates(row: dict[str, Any]) -> dict[str, Any] | None:
    """The request fields the project's persisted ask policy supplies."""
    ask_mode = str(row.get("ask_mode") or "").strip()
    if not ask_mode:
        return None
    triggers = [
        str(t).strip() for t in (row.get("pause_triggers") or []) if str(t).strip()
    ]
    updates: dict[str, Any] = {"mode": ask_mode}
    if ask_mode == "profile" and triggers:
        updates["pause_triggers"] = triggers
    return updates


def _apply_loop_mode(loop: Any, request: Any) -> None:
    """Permission mode + pause profile (ADR-066) and the parallel-track
    budget (ADR-066 amendment 2) onto the new loop's metadata."""
    if request.mode == "profile":
        # ADR-066 (D-33): park only when the action's trigger class is named.
        loop.metadata["mode"] = "profile"
        loop.metadata["pause_profile"] = {"triggers": list(request.pause_triggers)}
    else:
        loop.metadata["mode"] = "ask" if request.mode == "ask" else "auto"
    if request.max_tracks > 1:
        loop.metadata["max_tracks"] = request.max_tracks


def _project_budget_caps(project_id: str) -> dict[str, float]:
    """Row from project_budgets (ADR-067) via the loop store's sync conn.

    Empty dict when the store is not PG-backed or the read fails — an
    unreadable budget row never blocks a loop start (fail-open is logged;
    an over-budget loop is stoppable by hand). The read goes through the
    store (``read_row``): a dead session is replaced and the read retried
    once, and its transaction never stays open (ops-watch #532).
    """
    try:
        row = _loop_store.read_row(
            "SELECT max_gpu_hours, max_wall_clock_hours"
            " FROM project_budgets WHERE project_id = %s",
            (project_id,),
        )
    except Exception as exc:  # noqa: BLE001 — budget read is best-effort
        logger.warning("project budget read failed for %s: %s", project_id, exc)
        return {}
    if not row:
        return {}
    return {
        "max_gpu_hours": float(row[0] or 0.0),
        "max_wall_clock_hours": float(row[1] or 0.0),
    }


def configure_loop_store_sync(dsn: str) -> bool:
    """Upgrade to psycopg2-backed persistence (works from any thread).

    Same semantics as configure_loop_store but the connection is psycopg2:
    loop facades must be callable from the REST event loop and from runner
    threads; asyncpg pools are not loop-portable.
    Returns True when the sync pool backed store is now active.
    """
    global _loop_store
    import psycopg2

    from umrs.shared.pg_settings import pg_connect_kwargs

    def _connect():
        return psycopg2.connect(dsn, **pg_connect_kwargs("loop-store"))

    incoming = LoopStore.sync_pg(_connect(), reconnect=_connect)
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
    runner. Called from the http sidecar lifespan after recorder wiring,
    store upgrade and the durable-runtime start so durable loops actually
    resume across restarts (before, only the docstring claimed this).

    Job-hosted loops (``metadata.hosting == "job"``, ADR-036 D1) are NOT
    re-driven here when the durable runtime is up: the reconciler marks
    their attempt LOST and requeues the job, and the scheduler re-hosts the
    loop at its last iteration (one driver, one authority). Without a
    runtime they fall back to thread hosting so they never park forever —
    the flip is recorded on the loop (``hosting_note``).
    """
    runtime_up = _durable_runtime_available()
    resumed = 0
    for loop in _loop_store.list_all():
        if loop.status not in (LoopStatus.RUNNING, LoopStatus.PAUSED):
            continue
        if loop.metadata.get("hosting") == loop_hosting.HOSTING_JOB and runtime_up:
            continue  # reconciler → scheduler → loop_hosting.host resumes it
        try:
            _rehost_on_thread(loop)
            resumed += 1
        except Exception as exc:  # noqa: BLE001 — one loop must not block boot
            logger.warning(
                "loop driver re-registration failed for %s: %s", loop.loop_id, exc
            )
    if resumed:
        logger.info("re-registered %d loop driver(s) after restart", resumed)
    return resumed


def _durable_runtime_available() -> bool:
    try:
        from umrs.server.durable_job_runtime import get_runtime

        return get_runtime() is not None
    except Exception:  # noqa: BLE001 — runtime module optional in slim processes
        return False


def _rehost_on_thread(loop: ResearchLoop) -> None:
    """Re-spawn the in-process driver for one RUNNING/PAUSED loop."""
    from umrs.knowledge import agent_loop_runner, engine_loop_runner

    rewound = loop_hosting.rewind_interrupted_iteration(loop)
    if loop.metadata.get("hosting") == loop_hosting.HOSTING_JOB:
        loop.metadata["hosting"] = loop_hosting.HOSTING_THREAD
        loop.metadata["hosting_note"] = (
            "durable runtime unavailable at boot; thread-hosted"
        )
    if rewound or loop.metadata.get("hosting") == loop_hosting.HOSTING_THREAD:
        _loop_store.save(loop)
    if loop.metadata.get("agent"):
        project = ResearchProject(project_id=loop.project_id)
        agent_loop_runner.register(
            loop, executor=_agent_executor_for(project), store=_loop_store
        )
        return
    engine_loop_runner.register(
        loop,
        engine_factory=_engine_factory_from(_get_engine()),
        store=_loop_store,
        on_iteration=loop_hosting.iteration_hook(),
    )


def _get_engine() -> AutonomousResearchLoop:
    global _loop_engine
    if _loop_engine is None:
        from .rag_spine import SpineKnowledgeStore

        # Evidence comes from the knowledge spine; without a store every
        # real-mode iteration stopped at the evidence gate (2026-10-09).
        _loop_engine = AutonomousResearchLoop(
            workflow_port=_default_workflow_port(),
            knowledge_store=SpineKnowledgeStore(),
        )
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
        raise_typed(PlatformError.loops("not_found", f"Loop {loop_id} not found"))
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
            "approvals route; 'profile' (ADR-066 D-33) parks only actions "
            "whose trigger class is in pause_triggers. Mirrors the companion "
            "turn-attempt vocabulary."
        ),
    )
    max_tracks: int = Field(
        1,
        ge=1,
        le=4,
        description=(
            "ADR-066 amendment 2 (D-34): pursue up to N hypotheses in parallel "
            "tracks (iterations serve active tracks round-robin; a pending "
            "question on one track never idles the others; a rejected track "
            "closes without failing the loop). 1 = the single-branch loop."
        ),
    )
    pause_triggers: list[str] = Field(
        default_factory=list,
        description=(
            "ADR-066 trigger classes for mode='profile' (ignored otherwise): "
            "'cost', 'competing_hypotheses', 'direction_change', "
            "'physical_action' (ADR-066 amendment 3)."
        ),
    )
    workflow_name: str = Field(
        "",
        description=(
            "Registered UMRS workflow every real-mode experiment runs (G-P1-3). "
            "Empty → resolved from the registry: exactly one registered workflow "
            "is used automatically; none or several fail closed at start (400) "
            "listing the candidates. Discover names with `r4 workflow list`."
        ),
    )
    venue: str = Field(
        "frontend",
        description=(
            "Loop venue (WP-05 r2 start-route lane): 'frontend' (default; the "
            "loop driver shares this server process) or 'cluster:<site>' (the "
            "driver runs in the mini-RiVault/Slurm sandbox of that cluster — "
            "engine driver only today). Mirrors companion venue vocabulary."
        ),
    )
    iteration_plan: str = Field(
        "",
        description=(
            "ADR-092 slices 6, 8b: a team-plan template each iteration runs as "
            "one venue-hosted plan run (the plan engine in the sandboxed r4 "
            "worker). Refused by name (422 loops.unprocessable, "
            "plan_iteration_needs_server_venue) when this site has no frontend "
            "venue."
        ),
    )
    iteration_plan_doc: dict[str, Any] = Field(
        default_factory=dict,
        description="The template's plan (JSON), resolved by the client (slice 8b).",
    )
    plan_args: dict[str, Any] = Field(
        default_factory=dict,
        description="The plan's params other than the loop's own (objective, context).",
    )
    plan_sources: list[str] = Field(
        default_factory=list,
        description="Venue sources for the runs' models: pushed sources and/or 'site'.",
    )
    plan_default_source: str = ""
    plan_web_policy: str = ""
    plan_default_model: str = ""
    plan_tool_capabilities: dict[str, Any] = Field(default_factory=dict)


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
    # Workflow the loop's experiments run ("" for simulated/agentic loops).
    workflow_name: str = ""


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
    except Exception as exc:  # noqa: BLE001 — readiness gate failure surfaces typed
        raise_typed(
            PlatformError.loops(
                "unavailable",
                (
                    f"Readiness gate unavailable for project {request.project_id}: "
                    f"{exc}. Retry later; loops without a verifiable gate are denied "
                    "while approved=False is not explicitly set."
                ),
            ),
            cause=exc,
        )

    if gate_applies and project is not None:
        has_artifacts = bool(project.knowledge_store.list_artifacts())
        if not has_artifacts:
            return  # Trust the client for fresh projects
        raise_typed(
            PlatformError.loops(
                "forbidden",
                (
                    f"Research readiness not approved for project "
                    f"{request.project_id}. "
                    f"Approve with: r4 project approve {request.project_id}"
                ),
            ),
        )


# ── Workflow resolution (G-P1-3, WP-07) ─────────────────────────────────────

_WORKFLOW_DISCOVERY_HINT = (
    "Discover registered workflows with `r4 workflow list` (workflow_list tool) "
    "and pass workflow_name at loop start; register one with `r4 workflow upload`."
)


def _registered_workflow_names(owner: str) -> list[str]:
    """Names of registered UMRS workflows the loop owner may run.

    In-process catalog read (the same handler the REST listing uses).
    System-scope workflows are visible to everyone; user-scope ones only to
    their owner. An ownerless loop (no gateway identity) sees all of them.
    """
    import json as _json

    from umrs.core.TaskCmdStatus import TaskCmd
    from umrs.server.workflow_registry_handler import handle_workflow_command

    data = _json.loads(
        handle_workflow_command(TaskCmd.LIST_WORKFLOWS, owner or "_", "")
    )
    if data.get("status") == "error":
        raise RuntimeError(str(data.get("message") or "workflow catalog error"))
    entries = [e for e in (data.get("workflows") or []) if _workflow_visible(e, owner)]
    return sorted({str(e.get("name")).strip() for e in entries})


def _workflow_visible(entry: dict[str, Any], owner: str) -> bool:
    if not str(entry.get("name") or "").strip():
        return False
    if not owner or entry.get("scope", "user") == "system":
        return True
    return str(entry.get("user_id") or "") == owner


def _experiment_workflow(experiment: Any) -> str:
    return str(
        getattr(experiment, "workflow_name", None)
        or (getattr(experiment, "metadata", None) or {}).get("workflow_name")
        or ""
    )


def _plan_workflow(loop: ResearchLoop) -> str | None:
    """The plan's own workflow when every experiment already names one."""
    experiments = list(loop.plan.experiments) if loop.plan else []
    if experiments and all(_experiment_workflow(e) for e in experiments):
        return _experiment_workflow(experiments[0])
    return None


def _catalog_candidates(requested: str, owner: str) -> list[str] | None:
    """Registered names, or None when the catalog is unreachable but a name was given."""
    try:
        return _registered_workflow_names(owner)
    except Exception as exc:  # noqa: BLE001 — reported as 503, never hidden
        if requested:
            logger.warning(
                "workflow catalog unavailable; trusting requested name: %s", exc
            )
            return None
        raise_typed(
            PlatformError.loops(
                "unavailable",
                (
                    f"Workflow catalog unavailable ({exc}); pass workflow_name explicitly "
                    "or retry once the registry is reachable."
                ),
            ),
            cause=exc,
        )


def _pick_workflow(requested: str, candidates: list[str] | None) -> tuple[str, str]:
    """(name, source) or a 400 that names the candidates and the next step."""
    if requested:
        if candidates is not None and requested not in candidates:
            raise_typed(
                PlatformError.loops(
                    "bad_request",
                    (
                        f"workflow_name '{requested}' is not a registered UMRS workflow "
                        f"(registered: {candidates or 'none'}). {_WORKFLOW_DISCOVERY_HINT}"
                    ),
                ),
            )
        return requested, (
            "request" if candidates is not None else "request-unverified"
        )
    names = candidates or []
    if len(names) == 1:
        return names[0], "single-registered-default"
    if not names:
        raise_typed(
            PlatformError.loops(
                "bad_request",
                (
                    "Real-mode loops run each experiment as a registered UMRS workflow "
                    f"and none is registered for this caller. {_WORKFLOW_DISCOVERY_HINT}"
                ),
            ),
        )
    raise_typed(
        PlatformError.loops(
            "bad_request",
            (
                f"Several workflows are registered ({names}); pass workflow_name to "
                f"choose one. {_WORKFLOW_DISCOVERY_HINT}"
            ),
        ),
    )


def _resolve_loop_workflow(
    request: StartLoopRequest,
    engine: AutonomousResearchLoop,
    loop: ResearchLoop,
    owner: str,
) -> str:
    """Pick the registered workflow the loop's experiments run (G-P1-3).

    Simulated engines and agentic loops run no workflows. A plan whose
    experiments all name a workflow keeps it. Otherwise an explicit
    ``workflow_name`` is validated against the catalog, and with none given
    exactly one registered workflow is used automatically — zero or several
    fail closed here (400, candidates named) instead of a RUNNING loop dying
    at its first iteration with a registry-identifier error. The choice is
    written on the loop (``metadata.workflow_name`` + ``workflow_resolution``)
    and onto every experiment lacking one.
    """
    if _runs_no_workflow(request, engine):
        return ""
    planned = _plan_workflow(loop)
    if planned is not None:
        loop.metadata["workflow_resolution"] = {
            "workflow_name": planned,
            "source": "plan",
        }
        return planned
    requested = (request.workflow_name or "").strip()
    candidates = _catalog_candidates(requested, owner)
    name, source = _pick_workflow(requested, candidates)
    _stamp_experiments(loop, name)
    loop.metadata["workflow_name"] = name
    loop.metadata["workflow_resolution"] = {
        "workflow_name": name,
        "source": source,
        "candidates": list(candidates or [])[:20],
    }
    return name


def _runs_no_workflow(
    request: StartLoopRequest, engine: AutonomousResearchLoop
) -> bool:
    """Simulated engines, agentic loops and plan iterations run no workflow."""
    return bool(
        engine.simulated or request.agentic or (request.iteration_plan or "").strip()
    )


def _stamp_experiments(loop: ResearchLoop, name: str) -> None:
    """Every planned experiment lacking a workflow gets the resolved one."""
    for experiment in list(loop.plan.experiments) if loop.plan else []:
        if not _experiment_workflow(experiment):
            experiment.metadata["workflow_name"] = name


def _loop_job_spec(loop: ResearchLoop, *, agentic: bool) -> dict[str, Any]:
    """research_loop_job spec for one loop (WP-05 r2 start leg).

    The cluster venue rides the spec: the store resolves jobs.venue into the
    claim-side scoping class (jobs.venue = 'rccs-front' for cluster:rccs-front);
    the default 'frontend' is deliberately NOT written — byte-identical to
    pre-venue rows for every existing caller.
    """
    spec: dict[str, Any] = {
        "loop_id": loop.loop_id,
        "driver": "agent" if agentic else "engine",
    }
    venue = str(loop.metadata.get("venue") or "").strip()
    if venue and venue != "frontend":
        spec["venue"] = venue
    return spec


def _refuse_plan_iteration(request: StartLoopRequest) -> None:
    """ADR-092 slice 6: a plan as the iteration executor runs in the local
    venue only; here it is refused by name (slice 8 brings the server
    venue), before anything is created — never a plain loop with the plan
    dropped."""
    template = (request.iteration_plan or "").strip()
    if not template:
        return
    from umrs.jobs.research_loop_backend import PLAN_ITERATION_REFUSAL
    from umrs.venues.frontend_config import FrontendVenueConfig
    from umrs.venues.plan_runs import ENABLE_POINTER

    # ADR-092 slice 8b: a server-hosted loop runs each iteration as one
    # venue-hosted plan run; without the venue it is refused by name.
    missing = FrontendVenueConfig.from_env().missing()
    if missing:
        raise_typed(
            PlatformError.loops(
                "unprocessable",
                f"{PLAN_ITERATION_REFUSAL} (this site has no server venue (missing: "
                f"{', '.join(missing)}); {ENABLE_POINTER})",
                reason="plan_iteration_needs_server_venue",
                iteration_plan=template[:120],
                venue_missing=missing,
            )
        )
    problems = _plan_iteration_problems(request)
    if problems:
        raise_typed(
            PlatformError.loops(
                "unprocessable",
                "plan_invalid: " + "; ".join(problems),
                reason="plan_invalid",
                iteration_plan=template[:120],
            )
        )


def _plan_iteration_problems(request: StartLoopRequest) -> list[str]:
    """Why the loop's plan cannot run as its iteration ([] = it can), checked
    with the shared loop table and the stdlib grammar before anything exists."""
    from r4_contracts import loop_plan
    from r4_contracts.team_engine.plan import prepare_plan

    try:
        shape = loop_plan.loop_template(request.iteration_plan)
        args = loop_plan.caller_args(shape, request.plan_args)
    except ValueError as exc:
        return [str(exc)]
    if not request.plan_sources:
        return [
            "plan_sources: a venue source serves the runs' models (a pushed source, or site)"
        ]
    args.update({shape.goal_param: request.research_goal, shape.context_param: ""})
    _program, problems = prepare_plan(
        request.iteration_plan_doc,
        args,
        tool_capabilities=request.plan_tool_capabilities,
    )
    return [f"{p.path}: {p.message}" for p in problems[:10]]


def _attach_iteration_plan(loop: Any, request: StartLoopRequest) -> None:
    """ADR-092 slice 8b: the loop's plan, recorded on the loop; a plan loop
    with no mode asks at every plan run (fail closed, as the local one)."""
    if not (request.iteration_plan or "").strip():
        return
    loop.metadata[plan_iteration.META_KEY] = {
        "template": request.iteration_plan.strip(),
        "doc": dict(request.iteration_plan_doc),
        "args": dict(request.plan_args),
        "sources": list(request.plan_sources),
        "default_source": request.plan_default_source,
        "web_policy": request.plan_web_policy,
        "default_model": request.plan_default_model,
        "tool_capabilities": dict(request.plan_tool_capabilities),
    }
    if not (request.mode or "").strip() or "mode" not in request.model_fields_set:
        loop.metadata["mode"] = "ask"


def _validate_loop_venue(request: StartLoopRequest) -> str:
    """Start-route venue grammar (WP-05 r2 start leg).

    Grammar mirrors the jobs-layer validator exactly — start must fail closed
    with 400 here, matching research_loop_job's own rejection later. Burying
    the rule twice would drift-fronts on vocab additions one day; keep the
    single source of truth and map its errors onto 400.
    """
    venue = (request.venue or "").strip()
    if not venue:
        # absent: byte-identical to pre-venue semantics (today = frontend)
        return "frontend"
    if request.agentic and venue != "frontend":
        raise HTTPException(
            400,
            "research_loop venue accepts the engine driver only on cluster "
            "venues — agentic has cluster-side wiring parked (follow-on)",
        )
    from umrs.jobs.research_loop_backend import validate_research_loop_spec

    try:
        validate_research_loop_spec({"loop_id": "venue-probe", "venue": venue})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return venue


# ── Owner identity (ADR-036 (a): approvals are answerable by the loop owner) ─


def _loop_owner(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> str:
    """Gateway-verified owner of a loop being started ("" when no identity).

    When identity headers are present they must verify like every other
    sealed route (401 on a bad signature in strict mode). Absent headers are
    accepted only outside strict mode (direct sidecar callers, mvp dev): in
    strict mode a start without identity is a 401 (WP-08 — an ownerless
    durable approval inbox would let any caller decide the loop's tool
    grants). mvp keeps the historical ownerless loop.
    """
    if not (x_user_id or "").strip():
        from umrs.api._scope_identity import _strict_mode

        if _strict_mode():
            raise_typed(
                PlatformError.auth(
                    "gateway_identity_required",
                    "loop start requires gateway-verified identity in strict mode",
                ),
            )
        return ""
    from umrs.api.workflows import _require_verified_workflow_identity

    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    return (x_user_id or "").strip()


def _enforce_loop_project(
    request: StartLoopRequest, http_request: Request, owner: str
) -> None:
    """A loop runs in a project its verified starter may write.

    The body names the project and nothing bound it to the caller, so a
    verified user could run a loop, with its knowledge reads, finding writes
    and the project's own model endpoints, in any project (review of the
    loop's registry route, 2026-10-09). Same rule as knowledge ingest's
    project_mismatch: a member of several projects uses that project's token.
    Ownerless loops exist only outside strict mode (``_loop_owner``).
    """
    from umrs.knowledge.access_scope import CallerScope, project_writable

    project_id = (request.project_id or "").strip()
    if not project_id:
        raise_typed(PlatformError.loops("bad_request", "project_id is required"))
    if not owner:
        return
    headers = http_request.headers
    scope = CallerScope(
        user_id=owner,
        project_id=(headers.get("x-verified-project-id") or "").strip(),
        is_admin=(headers.get("x-verified-is-admin") or "").strip().lower()
        in ("1", "true"),
    )
    if not project_writable(project_id, scope):
        raise_typed(
            PlatformError.loops(
                "forbidden",
                f"Cannot start a loop in project {project_id!r} with a token for "
                f"{scope.project_id!r}; log in to that project",
            )
        )


_COMPANION_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _companion_session(request: Request) -> str:
    """The calling companion session (``x-r4-session-id``), when well-formed."""
    sid = (request.headers.get("x-r4-session-id") or "").strip()
    return sid if _COMPANION_SESSION_ID.match(sid) else ""


@dataclass(frozen=True)
class _LoopCaller:
    user_id: str  # "" only outside strict mode (identity-less mvp caller)
    is_admin: bool


def _loop_caller(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> _LoopCaller:
    """The verified caller of a loop route (same envelope as loop start;
    strict mode without identity is a 401)."""
    user = _loop_owner(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    admin = bool(user) and (x_verified_is_admin or "").strip().lower() in ("true", "1")
    return _LoopCaller(user_id=user, is_admin=admin)


def _may_access(loop: ResearchLoop, caller: _LoopCaller) -> bool:
    """Owner, admin, an ownerless (mvp) loop, or an identity-less mvp caller."""
    owner = str(loop.metadata.get("owner_user_id") or "")
    return not owner or not caller.user_id or caller.is_admin or caller.user_id == owner


def _owned_loop(loop_id: str, caller: _LoopCaller) -> ResearchLoop:
    """The loop, when the caller may see it; another user's loop is a 404
    (its existence is not disclosed)."""
    loop = get_loop(loop_id)
    if not _may_access(loop, caller):
        raise_typed(PlatformError.loops("not_found", f"Loop {loop_id} not found"))
    return loop


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=StartLoopResponse,
    status_code=201,
    summary="Start a new research loop",
)
async def start_loop(
    request: StartLoopRequest,
    http_request: Request,
    owner: str = Depends(_loop_owner),
) -> StartLoopResponse:
    """Start a new autonomous research loop.

    **Loop criteria enforcement (spec §12):**
    When approved=True, the engine verifies the project's readiness
    has been approved by a human. Unapproved projects receive 403.

    **Hosting (ADR-036 D1):** with a durable job runtime the loop is hosted
    by a ``research_loop_job`` (``metadata.hosting == "job"``, job id on
    ``metadata.durable_job_id``); otherwise the process drives it on a
    thread (``hosting == "thread"``). Both are visible on the loop record.
    """
    _enforce_loop_project(request, http_request, owner)
    _refuse_plan_iteration(request)
    _enforce_readiness_approval(request)
    # D-33 tail: the project's persisted ask-policy default fills the
    # request when the caller left mode unset; an explicit mode always
    # wins (the persisted default never overrides a caller's choice).
    request = _apply_project_ask_policy(http_request, request)

    engine = _get_engine()
    loop = engine.start_loop(
        project_id=request.project_id,
        research_goal=request.research_goal,
        max_iterations=request.max_iterations,
    )
    _apply_loop_mode(loop, request)
    _attach_iteration_plan(loop, request)
    venue = _validate_loop_venue(request)
    loop.metadata["venue"] = venue
    # D-52 (2026-09-22): one trace id through the loop's jobs, iterations and
    # knowledge writes — the loop id itself is the trace anchor.
    loop.metadata.setdefault("trace_id", loop.loop_id)
    # D-52 join: the companion session that started the loop (header
    # x-r4-session-id; distinct from the ADR-025 research session_id).
    companion_session = _companion_session(http_request)
    if companion_session:
        loop.metadata.setdefault("companion_session_id", companion_session)
    # ADR-067 (gap row D-36): per-project loop budgets — wall-clock cap is
    # enforced by the driver at iteration boundaries; the GPU cap is recorded
    # (leg B: job-cost accounting).
    budget = _project_budget_caps(request.project_id)
    if budget.get("max_wall_clock_hours", 0.0) > 0:
        loop.metadata["max_wall_clock_hours"] = budget["max_wall_clock_hours"]
    if budget.get("max_gpu_hours", 0.0) > 0:
        loop.metadata["max_gpu_hours"] = budget["max_gpu_hours"]
    if request.agentic:
        # ADR-064 (2026-09-22): the engine runner is the one server driver of
        # record; the agent driver is deprecated in place (runs to completion,
        # folds into the engine driver's executor in the follow-up wave).
        loop.metadata["driver_note"] = "agent-deprecated"
    if owner:
        loop.metadata["owner_user_id"] = owner
    workflow_name = _resolve_loop_workflow(request, engine, loop, owner)
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
        _attach_loop_to_session(loop, request.session_id)

    await _host_loop(http_request, loop, engine, agentic=request.agentic)

    return StartLoopResponse(
        loop_id=loop.loop_id,
        project_id=loop.project_id,
        status=status_at_start,
        phase=phase_at_start,
        message=f"Loop {loop.loop_id} started for project {request.project_id}",
        workflow_name=workflow_name,
    )


def _attach_loop_to_session(loop: ResearchLoop, session_id: str) -> None:
    """ADR-025 D3: auto-register loop membership under the given session.

    404/503 propagate (the caller asked for a session); other failures are
    logged — session attach must not break loop start.
    """
    try:
        from umrs.api.research_sessions import get_session_store
        from umrs.research_sessions import SessionLoopLink

        store = get_session_store()
        if store is None:
            raise_typed(
                PlatformError.loops(
                    "unavailable",
                    "Session store unavailable; cannot register loop under session",
                ),
            )
        if store.get(session_id) is None:
            raise_typed(
                PlatformError.session("not_found", f"Session {session_id} not found"),
            )
        store.attach_loop(
            SessionLoopLink(
                session_id=session_id,
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
            session_id,
            exc,
        )


async def _host_loop(
    http_request: Request,
    loop: ResearchLoop,
    engine: AutonomousResearchLoop,
    *,
    agentic: bool,
) -> None:
    """Durable-job hosting when a runtime exists, else the in-process driver."""
    job_id = await _submit_loop_job(http_request, loop, agentic=agentic)
    if not job_id:
        _host_on_thread(loop, engine, agentic=agentic)
        return
    loop.metadata["hosting"] = loop_hosting.HOSTING_JOB
    loop.metadata["durable_job_id"] = job_id
    if agentic:
        loop.metadata["agent"] = "umrs-litellm"
    _loop_store.save(loop)


async def _submit_loop_job(
    http_request: Request, loop: ResearchLoop, *, agentic: bool
) -> str | None:
    """Submit the ``research_loop_job`` hosting this loop (None → thread host).

    Test-injectable via ``app.state.loop_job_submitter`` (mirrors the
    companion venue's submitter seam); otherwise the process durable runtime.
    Submission failures are logged and fall back to thread hosting — the
    loop record says which hosting it got.
    """
    submitter = getattr(http_request.app.state, "loop_job_submitter", None)
    try:
        if submitter is not None:
            return await submitter(loop)
        from umrs.server.durable_job_runtime import get_runtime

        runtime = get_runtime()
        if runtime is None:
            return None
        from umrs.api.agent_quota import enforce_spawn_budget
        from umrs.jobs.models import JobType

        # ADR-051 D-2: agent CPU-process budget per user — once at spawn.
        # A quota refusal aborts loop start with a typed 503 (never a
        # fallback thread masquerading as "not allowed").
        owner = str(loop.metadata.get("owner_user_id") or "") or ""
        enforce_spawn_budget(
            getattr(http_request.app.state, "umrs_db_pool", None), user_id=owner
        )

        return await runtime.submit_job_sync(
            job_type=JobType.RESEARCH_LOOP,
            project_id=loop.project_id,
            user_id=owner,
            spec=_loop_job_spec(loop, agentic=agentic),
        )
    except Exception as exc:  # noqa: BLE001 — hosting fallback, recorded on the loop
        logger.warning(
            "research_loop_job submission failed for %s: %s", loop.loop_id, exc
        )
        loop.metadata["hosting_note"] = f"job submission failed: {str(exc)[:160]}"
        return None


def _host_on_thread(
    loop: ResearchLoop, engine: AutonomousResearchLoop, *, agentic: bool
) -> None:
    """In-process driver (no durable runtime): agent runner or phased engine.

    Non-agentic: the server drives the phased engine itself (daemon thread,
    store-checkpointed, store-polled for pause/stop). Before this, the loop
    was created RUNNING but only a client driving engine.run could advance
    it — REST-started loops never iterated.
    """
    loop.metadata["hosting"] = loop_hosting.HOSTING_THREAD
    try:
        from umrs.knowledge import engine_loop_runner as _engine_runner

        if plan_iteration.is_plan_loop(loop):
            # ADR-092 slice 8b: the plan executor under the same engine driver.
            _engine_runner.register(
                loop,
                engine_factory=plan_iteration.engine_factory(),
                store=_loop_store,
                on_iteration=loop_hosting.iteration_hook(),
            )
            loop.metadata["driver"] = "engine"
        elif agentic:
            # ADR-064 fold: the agent executor hosts under the ENGINE driver
            # (AgenticEngine adapter); the agent runner is no longer spawned.
            from umrs.knowledge.agentic_engine import AgenticEngine

            project = ResearchProject(project_id=loop.project_id)
            engine_factory = lambda: AgenticEngine(
                executor=_agent_executor_for(project)
            )
            _engine_runner.register(
                loop,
                engine_factory=engine_factory,
                store=_loop_store,
                on_iteration=loop_hosting.iteration_hook(),
            )
            loop.metadata["driver"] = "engine"
            loop.metadata["agent"] = "umrs-litellm"
        else:
            _engine_runner.register(
                loop,
                engine_factory=_engine_factory_from(engine),
                store=_loop_store,
                on_iteration=loop_hosting.iteration_hook(),
            )
            loop.metadata["driver"] = "engine"
        _loop_store.save(loop)
    except Exception as exc:  # noqa: BLE001 — driver must not break start
        logger.warning("loop driver skipped for loop %s: %s", loop.loop_id, exc)


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
async def list_loops(
    caller: _LoopCaller = Depends(_loop_caller),
    companion_session_id: str | None = Query(
        None, description="Only loops this companion session started (D-52)"
    ),
) -> ListLoopsResponse:
    loops = [
        {
            "loop_id": loop_obj.loop_id,
            "status": loop_obj.status.value,
            "phase": loop_obj.phase.value,
            "companion_session_id": str(
                loop_obj.metadata.get("companion_session_id") or ""
            ),
            # 2026-10-01: project scope flows to the work pane so the console
            # can show only the current project's loops. Real field from the
            # loop object, never reconstructed client-side.
            "project_id": str(loop_obj.project_id or ""),
        }
        for loop_obj in _loop_store.list_all()
        if _may_access(loop_obj, caller)
        and (
            not companion_session_id
            or loop_obj.metadata.get("companion_session_id") == companion_session_id
        )
    ]
    return ListLoopsResponse(loops=loops)


@router.get(
    "/{loop_id}",
    response_model=LoopStatusResponse,
    summary="Get loop status",
)
async def get_loop_status(
    loop_id: str, caller: _LoopCaller = Depends(_loop_caller)
) -> LoopStatusResponse:
    loop = _owned_loop(loop_id, caller)
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
    caller: _LoopCaller = Depends(_loop_caller),
) -> list[IterationResponse]:
    loop = _owned_loop(loop_id, caller)
    iterations = loop.iterations
    if phase:
        iterations = [i for i in iterations if i.phase.value == phase]

    def _finding_text(f: Any) -> str:
        # Engine findings are dataclasses (statement+confidence), rows
        # reloaded from the store are dicts, companions write plain strings.
        # Serialize honestly to text for the wire model.
        if isinstance(f, str):
            return f
        if isinstance(f, dict):
            return str(f.get("statement") or f.get("finding_id") or f)
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


class AppendIterationRequest(BaseModel):
    """Cluster-venue loop-driver writes one iteration outcome (WP-05 round 2).

    Mirrors the durable loop record's LoopIteration shape; the backend maps it
    onto research_state semantics via loop_store.load/save (never replaces the
    whole record, never invents findings text).
    """

    iteration_id: str = ""
    iteration_number: int = 0
    phase: str = "experiment_execution"
    started_at: str = ""
    completed_at: str = ""
    hypothesis: str = ""
    hypothesis_id: str = ""
    experiment_id: str = ""
    results: dict[str, Any] = Field(default_factory=dict)
    analysis: str = ""
    evaluation: str = ""
    findings: list[str] = Field(default_factory=list)
    workflow_run_id: str = ""
    status: str = "pending"  # pending | running | completed | failed
    completion_state: str | None = None  # "ready" when the driver is concluding


async def _apply_iteration_record(
    loop: ResearchLoop, body: AppendIterationRequest
) -> dict[str, Any]:
    """Map the driver-provided body onto the store-backed loop and save it."""
    from r4_contracts.research import Finding

    from umrs.knowledge.loop import LoopIteration, LoopPhase

    try:
        phase = LoopPhase(body.phase)
    except ValueError as exc:
        raise_typed(
            PlatformError.loops("unprocessable", f"invalid phase {body.phase!r}"),
            cause=exc,
        )

    iteration_id = body.iteration_id or f"it-{body.iteration_number:03d}"
    iteration = LoopIteration(
        iteration_id=iteration_id,
        loop_id=loop.loop_id,
        iteration_number=body.iteration_number or len(loop.iterations) + 1,
        phase=phase,
        started_at=body.started_at,
        completed_at=body.completed_at,
        hypothesis=body.hypothesis,
        hypothesis_id=body.hypothesis_id,
        experiment_id=body.experiment_id,
        workflow_run_id=body.workflow_run_id,
        results=body.results,
        analysis=body.analysis,
        evaluation=body.evaluation,
        findings=[
            Finding(
                finding_id=f"{iteration_id}-f{i}",
                statement=f,
                hypothesis_id=body.hypothesis_id,
            )
            for i, f in enumerate(body.findings)
        ],
        status=body.status,
    )
    loop.iterations.append(iteration)
    loop.current_iteration = min(
        loop.max_iterations,
        max(loop.current_iteration, iteration.iteration_number),
    )
    if body.findings:
        loop.total_findings = loop.total_findings + len(body.findings)
    if body.completion_state == "ready":
        loop.status = LoopStatus.STOPPED
    _loop_store.save(loop)
    return {
        "ok": True,
        "iteration_number": iteration.iteration_number,
        "status": loop.status.value,
        "current_iteration": loop.current_iteration,
    }


@router.post(
    "/{loop_id}/iterations",
    summary="Append one iteration outcome (cluster-venue driver write)",
    response_model=dict[str, Any],
)
async def append_iteration(
    loop_id: str,
    body: AppendIterationRequest,
    caller: _LoopCaller = Depends(_loop_caller),
) -> dict[str, Any]:
    """Cluster-venue loop drivers write their iteration outcome here.

    Owner-scoped via the verified-identity envelope (same envelope as allied
    loop routes through the gateway lanes).
    """
    loop = _owned_loop(loop_id, caller)
    if loop.status == LoopStatus.STOPPED:
        raise_typed(PlatformError.loops("conflict", "loop already finished"))
    return await _apply_iteration_record(loop, body)


@router.get(
    "/{loop_id}/snapshot",
    summary="Checker-safe snapshot of loop runtime state",
)
async def get_loop_snapshot(
    loop_id: str, caller: _LoopCaller = Depends(_loop_caller)
) -> dict[str, Any]:
    """Mirror the loop_hosting.snapshot checker view over HTTP.

    Cluster-venue checkers call this to compute attempt status without a
    direct DSN (loop truth stays server-side; snapshot stays a projection).
    """
    _owned_loop(loop_id, caller)
    from umrs.knowledge import loop_hosting

    if not loop_hosting.is_configured():
        _configure_hosting()
    snap = loop_hosting.snapshot(loop_id)
    if snap is None:
        raise_typed(PlatformError.loops("not_found", f"Loop {loop_id} not found"))
    return snap


class LoopStateUpdateRequest(BaseModel):
    """Whole-status projection write from cluster-venue drivers (WP-05 round 2).

    `save()` semantics over the durable loop record: only the fields drivers
    legitimately change between iterations. Other state (plan, iterations,
    timestamps except completed_at) stays owned by the server.
    """

    status: str | None = None
    phase: str | None = None
    current_iteration: int | None = None
    total_hypotheses_tested: int | None = None
    total_experiments_run: int | None = None
    total_findings: int | None = None
    success_rate: float | None = None
    completed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


@router.put(
    "/{loop_id}/state",
    summary="Status projection write (cluster driver save)",
    response_model=LoopStatusResponse,
)
async def update_loop_state(
    loop_id: str,
    body: LoopStateUpdateRequest,
    caller: _LoopCaller = Depends(_loop_caller),
) -> LoopStatusResponse:
    """Driver calls `save()` after each iteration — mapped to the durable record."""
    loop = _owned_loop(loop_id, caller)
    _apply_loop_enums(loop, body)
    _apply_loop_counters(loop, body)
    if body.error is not None:
        loop.metadata["error"] = body.error
    if body.metadata:
        loop.metadata.update(body.metadata)
    loop.updated_at = datetime.now(timezone.utc).isoformat()
    _loop_store.save(loop)
    return await get_loop_status(loop_id, caller)


def _apply_loop_enums(loop: Any, body: LoopStateUpdateRequest) -> None:
    """Validated status/phase transitions (422 on unknown enum values)."""
    if body.status is not None:
        try:
            loop.status = LoopStatus(body.status)
        except ValueError as exc:
            raise_typed(
                PlatformError.loops("unprocessable", f"invalid status {body.status!r}"),
                cause=exc,
            )
    if body.phase is not None:
        try:
            loop.phase = LoopPhase(body.phase)
        except ValueError as exc:
            raise_typed(
                PlatformError.loops("unprocessable", f"invalid phase {body.phase!r}"),
                cause=exc,
            )


_LOOP_COUNTER_FIELDS = (
    "total_hypotheses_tested",
    "total_experiments_run",
    "total_findings",
    "success_rate",
    "completed_at",
)


def _apply_loop_counters(loop: Any, body: LoopStateUpdateRequest) -> None:
    """Copy the provided (non-None) progress fields; clamp current_iteration."""
    if body.current_iteration is not None:
        loop.current_iteration = max(
            0, min(loop.max_iterations, body.current_iteration)
        )
    for name in _LOOP_COUNTER_FIELDS:
        value = getattr(body, name)
        if value is not None:
            setattr(loop, name, value)


@router.post(
    "/{loop_id}/stop",
    response_model=StopLoopResponse,
    summary="Stop a running loop",
)
async def stop_loop(
    loop_id: str, caller: _LoopCaller = Depends(_loop_caller)
) -> StopLoopResponse:
    loop = _owned_loop(loop_id, caller)
    loop.status = LoopStatus.STOPPED
    loop.completed_at = datetime.now(timezone.utc).isoformat()
    _loop_store.save_preserving_progress(loop)  # never clobbers runner progress (H-1)

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
async def pause_loop(
    loop_id: str, caller: _LoopCaller = Depends(_loop_caller)
) -> StopLoopResponse:
    loop = _owned_loop(loop_id, caller)
    loop.status = LoopStatus.PAUSED
    _loop_store.save_preserving_progress(loop)  # never clobbers runner progress (H-1)

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
async def resume_loop(
    loop_id: str, caller: _LoopCaller = Depends(_loop_caller)
) -> StartLoopResponse:
    loop = _owned_loop(loop_id, caller)
    if loop.status != LoopStatus.PAUSED:
        raise_typed(
            PlatformError.loops(
                "bad_request",
                f"Cannot resume loop in status {loop.status.value}; only paused loops can be resumed",
            ),
        )
    loop.status = LoopStatus.RUNNING
    _loop_store.save_preserving_progress(loop)  # never clobbers runner progress (H-1)

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
    except Exception:  # reset must never raise
        logger.debug("runner sweep during store clear failed", exc_info=True)
    _loop_store.clear()
