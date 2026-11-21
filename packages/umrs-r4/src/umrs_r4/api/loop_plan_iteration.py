"""Plan as the loop's iteration executor: the start legs (ADR-092 slice 8b).

U3 S3c: moved out of ``api/loop.py``; U3 S5: into ``umrs_r4`` with
``knowledge/plan_iteration.py`` and the plan-run venue. It registers as loop
iteration kind ``plan`` (``umrs_r4.register``).
"""

from __future__ import annotations

from typing import Any

from umrs.api.typed_errors import PlatformError, raise_typed
from umrs.extensions.context import LoopIterationKind

from umrs_r4.knowledge import plan_iteration


def refuse_plan_iteration(request: Any) -> None:
    """ADR-092 slice 6: a plan as the iteration executor runs in the local
    venue only; here it is refused by name (slice 8 brings the server
    venue), before anything is created — never a plain loop with the plan
    dropped."""
    template = (request.iteration_plan or "").strip()
    if not template:
        return
    from umrs.jobs.research_loop_backend import PLAN_ITERATION_REFUSAL

    from umrs_r4.venues.frontend_config import FrontendVenueConfig
    from umrs_r4.venues.plan_runs import ENABLE_POINTER

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
    problems = plan_iteration_problems(request)
    if problems:
        raise_typed(
            PlatformError.loops(
                "unprocessable",
                "plan_invalid: " + "; ".join(problems),
                reason="plan_invalid",
                iteration_plan=template[:120],
            )
        )


def plan_iteration_problems(request: Any) -> list[str]:
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


def attach_iteration_plan(loop: Any, request: Any) -> None:
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
        "web_allow_domains": list(request.plan_web_allow_domains),
        "default_model": request.plan_default_model,
        "tool_capabilities": dict(request.plan_tool_capabilities),
    }
    if not (request.mode or "").strip() or "mode" not in request.model_fields_set:
        loop.metadata["mode"] = "ask"


def check_plan_web_choice(request: Any, http_request: Any, owner: str) -> None:
    """ADR-099 D-2: a server-hosted plan loop is a non-interactive start; its
    runs' web choice is resolved against the caps at loop start, so a missing
    or too-wide choice is refused up front (typed ``sessionpolicy.*``), never
    as a refused run at iteration 1."""
    if not (request.iteration_plan or "").strip():
        return
    from umrs.api.session_policy_errors import policy_typed
    from umrs.session_policy import PolicyError

    from umrs_r4.api.companion_venue import get_venue_service

    try:
        get_venue_service(http_request).check_web_choice(
            owner=owner,
            project=request.project_id,
            web_policy=request.plan_web_policy,
            web_allow_domains=request.plan_web_allow_domains,
        )
    except PolicyError as exc:
        message = f"research loop runs (plan_web_policy): {exc.message}"
        policy_typed(PolicyError(exc.code, message, **exc.detail))


def validate_start(request: Any, http_request: Any, owner: str) -> None:
    """Refuse an unrunnable plan loop before anything is created."""
    refuse_plan_iteration(request)
    check_plan_web_choice(request, http_request, owner)


def plan_kind() -> LoopIterationKind:
    return LoopIterationKind(
        name="plan",
        matches=plan_iteration.is_plan_loop,
        validate_start=validate_start,
        attach=attach_iteration_plan,
        engine_factory=plan_iteration.engine_factory,
    )
