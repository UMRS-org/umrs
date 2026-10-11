"""A team plan run hosted on the server venue (ADR-092 slice 8, D-29, D-63).

UMRS hosts a plan run by starting one frontend-venue worker session (the
ADR-046 A2 path: a sandboxed ``r4-companion`` worker on a per-session
``--internal`` network, a lease token as its only credential, model calls
through the egress sidecar to ``umrs-infer``) whose command runs the r4 plan
engine headless for that one run (``r4 venue worker --plan-run <id>``).
UMRS never imports r4 (CONSTITUTION §1): it validates the run's definition
with the stdlib plan grammar it already imports (``r4_contracts``), records
the run, launches the worker and keeps the record.

The run's durable record is slice 7's: one ``research_loop_job``, here with
``hosting: venue``, plus one ``execution_traces`` row per record. Only the
run's own worker writes it (``plan_run_records.check_writer``); its owner
reads it, resumes it (a fresh worker, attempt N+1) or cancels it.

Before any worker starts, the run is refused by name when:

- the site has the frontend venue off or unconfigured
  (``plan_iteration_needs_server_venue``, with the missing settings and the
  pointer to enable it);
- its plan does not validate under the grammar (D-4);
- its wall clock exceeds the project's ADR-067 cap (or is unbounded while
  the project is capped): ``plan_exceeds_project_caps``.

Foundation module: no ``umrs.api`` or ``umrs.server`` imports.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.jobs import plan_run_records as prr

from .frontend_service import FrontendVenueService, VenueError

logger = logging.getLogger(__name__)

REFUSAL_CODE = "plan_iteration_needs_server_venue"
CAPS_CODE = "plan_exceeds_project_caps"
ENABLE_POINTER = (
    "enable the frontend venue on this site: UMRS_VENUE_ENABLED=true plus the "
    "UMRS_VENUE_* settings (deploy: the frontend-venue play, "
    "docs/adr/046-remote-session-credential-egress.md Amendment 2)"
)
MAX_PROBLEMS = 20


def venue_refusal(missing: list[str]) -> VenueError:
    """The named refusal when this site cannot host a plan run."""
    return VenueError(
        "unavailable",
        f"{REFUSAL_CODE}: this server cannot host a team plan run: the frontend venue "
        f"is not enabled or not configured (missing: {', '.join(missing)}); "
        f"{ENABLE_POINTER}",
        reason=REFUSAL_CODE,
        missing=list(missing),
    )


def plan_problems(definition: Any) -> list[str]:
    """Every grammar problem of the definition's bound, expanded plan (D-4)."""
    from r4_contracts.team_engine.plan import validate_plan

    plan = definition.get("plan") if isinstance(definition, dict) else None
    if not isinstance(plan, dict) or not plan:
        return ["definition.plan: the run's bound, expanded plan is required"]
    manifest = definition.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    capabilities = manifest.get("tool_capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    problems = validate_plan(plan, tool_capabilities=capabilities)
    return [f"{p.path}: {p.message}" for p in problems[:MAX_PROBLEMS]]


def caps_for(budget_row: dict[str, Any] | None) -> dict[str, Any]:
    """The ADR-067 caps a plan run is held to: the project's wall clock.

    GPU hours are not a plan concept (D-7: compute goes through UMRS jobs
    and their own caps). ``{}`` = uncapped.
    """
    hours = float((budget_row or {}).get("max_wall_clock_hours") or 0.0)
    return {"max_wall_clock_sec": int(hours * 3600)} if hours > 0 else {}


def caps_problem(definition: dict[str, Any], caps: dict[str, Any]) -> str:
    """Why the plan's own budget exceeds the project's caps ("" = within)."""
    cap = int(caps.get("max_wall_clock_sec") or 0)
    if cap <= 0:
        return ""
    budget = (definition.get("plan") or {}).get("budget") or {}
    own = budget.get("max_wallclock_sec")
    if not isinstance(own, int) or isinstance(own, bool):
        return (
            f"{CAPS_CODE}: the plan's wall clock is unbounded and the project's ADR-067 "
            f"cap is {cap} s; set budget.max_wallclock_sec to at most {cap}"
        )
    if own > cap:
        return (
            f"{CAPS_CODE}: the plan's budget.max_wallclock_sec is {own} s and the "
            f"project's ADR-067 cap is {cap} s"
        )
    return ""


def start(
    service: FrontendVenueService,
    store: Any,
    *,
    owner: str,
    project: str,
    run_id: str,
    definition: Any,
    caps: dict[str, Any],
    source_refs: list[str],
    default_source: str = "",
    web_policy: str = "",
    web_allow_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Start (or resume) ``run_id`` on the server venue; nothing runs on refusal."""
    missing = service.config.missing()
    if missing:
        raise venue_refusal(missing)
    existing = prr.find_run_job(store, project, owner, run_id)
    recorded = (existing.spec or {}).get("definition") if existing is not None else None
    _require_runnable(recorded if existing is not None else definition, caps)
    job, created = prr.venue_run_job(
        store,
        project_id=project,
        user_id=owner,
        run_id=run_id,
        definition=definition,
        caps=caps,
    )
    session_id = prr.venue_session_id(owner, run_id)
    venue = service.start_plan_run(
        owner=owner,
        session_id=session_id,
        run_id=run_id,
        run_job_id=job.job_id,
        source_refs=source_refs,
        default_source=default_source,
        web_policy=web_policy,
        web_allow_domains=web_allow_domains,
        project=project,
    )
    logger.info(
        "plan run %s hosted on the server venue: job=%s session=%s created=%s",
        run_id,
        job.job_id,
        session_id,
        created,
    )
    return {"run_id": run_id, "job_id": job.job_id, "created": created, "venue": venue}


def _require_runnable(definition: Any, caps: dict[str, Any]) -> None:
    problems = plan_problems(definition)
    if problems:
        raise VenueError(
            "bad_request",
            "plan_invalid: the run's plan does not validate: " + "; ".join(problems),
            problems=problems,
        )
    problem = caps_problem(definition, caps)
    if problem:
        raise VenueError("forbidden", problem, reason=CAPS_CODE, caps=dict(caps))


def status(
    service: FrontendVenueService, store: Any, *, owner: str, project: str, run_id: str
) -> dict[str, Any]:
    """The run's record view plus its worker session's state."""
    job = _venue_job(store, owner, project, run_id)
    view = prr.run_view(store, job)
    view["venue"] = service.status(
        owner=owner, session_id=str(job.spec.get("venue_session_id") or "")
    )
    return view


def cancel(
    service: FrontendVenueService,
    store: Any,
    *,
    owner: str,
    project: str,
    run_id: str,
    actor: str = "user",
) -> dict[str, Any]:
    """Stop the run's worker (teardown, lease revoked), then cancel its record.

    The worker first, so nothing it still sends is accepted after the
    cancel; a cancelled record is never reopened (slice 7).
    """
    job = _venue_job(store, owner, project, run_id)
    session_id = str(job.spec.get("venue_session_id") or "")
    stopped = None
    if service.store.get(session_id) is not None:
        stopped = service.stop(owner=owner, session_id=session_id, actor=actor)
    try:
        store.cancel_job(job.job_id, reason=f"plan run cancelled by {actor}")
        cancelled = True
    except Exception as exc:  # noqa: BLE001 — already terminal: the record keeps its outcome
        logger.info("plan run %s record not cancelled: %s", run_id, exc)
        cancelled = False
    return {
        "run_id": run_id,
        "job_id": job.job_id,
        "cancelled": cancelled,
        "job_status": _status_value(store.get_job(job.job_id)),
        "venue": stopped,
    }


def _venue_job(store: Any, owner: str, project: str, run_id: str) -> Any:
    job = prr.find_run_job(store, project, owner, run_id)
    if job is None:
        raise prr.PlanRunRecordError(
            "not_found", f"No durable record of plan run {run_id} for this caller"
        )
    if not prr.is_venue_hosted(job.spec):
        raise prr.PlanRunRecordError(
            "conflict",
            f"hosting_mismatch: run {run_id} is client-hosted; its client cancels it",
        )
    return job


def _status_value(job: Any) -> str:
    return str(getattr(job.status, "value", job.status))


__all__ = [
    "CAPS_CODE",
    "ENABLE_POINTER",
    "REFUSAL_CODE",
    "cancel",
    "caps_for",
    "caps_problem",
    "plan_problems",
    "start",
    "status",
    "venue_refusal",
]
