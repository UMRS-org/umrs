"""A team plan run as a server-hosted loop's iteration (ADR-092 slice 8b).

The existing server-hosted research loop (``research_loop_job``, ADR-036,
driven by ``EngineLoopRunner``) gains one iteration executor kind: each
iteration starts one slice-8 venue-hosted plan run (the plan engine stays
in the sandboxed r4 worker; UMRS only binds the round's params with the
stdlib grammar and records), waits for the outcome its worker recorded,
and applies the loop's decision. Not a new loop, driver or engine: the
runner wires this adapter's ``_iteration_hook`` / ``_cancel_check`` /
``_approval_gate`` exactly as it wires the agent executor's
(``AgenticEngine``).

What the loop adds around one run, all read from the one shared table
(``r4_contracts.loop_plan``, which r4's local loop reads too):

- the run is a ``plan_run`` action, classed ``cost`` (``ask_policy``): the
  runner's gate parks an ask-mode loop on the durable approvals inbox, and
  a profile naming ``cost`` too; a human's no stops the loop, nothing run;
- the plan's wall clock is min(its own, what the loop has left of its
  ADR-067 cap); under a second left, no run starts and the loop stops;
- the loop's own goal and its code-derived record of earlier rounds are
  the plan's ``objective`` and ``context`` (never an agent's words);
- the decision: a pick publishes (the loop completes), no pick or a
  near-tie continues, a run that did not finish (a rejected gate, a spent
  budget, a failure, a stop) stops the loop with its reason;
- a published round stages one PROPOSED finding through the existing
  proposal lane (migration 0069: pending, confidence 0, producer-tagged,
  untrusted; a person decides it, ADR-050) and records a Catalog candidate
  on the iteration; nothing is decided, established or written.

The venue verbs (start, view, cancel) and the proposal lane are wired by the
REST layer at boot (``configure``), so this module stays free of
``umrs.api``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from r4_contracts import loop_plan

from .loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop

logger = logging.getLogger(__name__)

PLAN_RUN = "plan_run"  # the ask-policy action kind (cost)
META_KEY = "iteration_plan"
ROUNDS_KEY = "plan_rounds"
POLL_SECONDS = 5.0
_RUN_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class PlanRunHost:
    """The venue verbs and the proposal lane, bound by the REST layer.

    ``start(owner=, project=, run_id=, definition=, caps=, source_refs=,
    default_source=, web_policy=)``, ``view(owner=, project=, run_id=)``,
    ``cancel(owner=, project=, run_id=)``, ``propose(project=, owner=,
    reason=, source_artifacts=, operation=)`` and ``venue_missing()``.
    """

    start: Callable[..., dict[str, Any]]
    view: Callable[..., dict[str, Any]]
    cancel: Callable[..., dict[str, Any]]
    propose: Callable[..., dict[str, Any]]
    venue_missing: Callable[[], list[str]]


_HOST: PlanRunHost | None = None


def configure(host: PlanRunHost | None) -> None:
    global _HOST
    _HOST = host


def configured_host() -> PlanRunHost | None:
    return _HOST


def is_plan_loop(loop: ResearchLoop) -> bool:
    return isinstance(loop.metadata.get(META_KEY), dict)


# ── one round's definition (bound here with the stdlib grammar) ─────────────


def run_id_for(loop: ResearchLoop, iteration: int) -> str:
    stem = _RUN_ID_UNSAFE.sub("-", loop.loop_id).strip("-.")[:100] or "loop"
    return f"{stem}-it{int(iteration)}"


def round_definition(
    loop: ResearchLoop, iteration: int, remaining_sec: float | None
) -> tuple[str, dict[str, Any], list[str]]:
    """``(run_id, definition, problems)`` of one round's plan run."""
    from r4_contracts.team_engine.plan import canonical_hash, prepare_plan

    meta = loop.metadata[META_KEY]
    shape = loop_plan.loop_template(meta.get("template"))
    args = {
        **loop_plan.caller_args(shape, meta.get("args")),
        shape.goal_param: loop.research_goal,
        shape.context_param: loop_plan.context_text(loop.metadata.get(ROUNDS_KEY) or []),
    }
    raw = loop_plan.plan_within_wallclock(meta.get("doc"), remaining_sec)
    capabilities = dict(meta.get("tool_capabilities") or {})
    program, problems = prepare_plan(raw, args, tool_capabilities=capabilities)
    run_id = run_id_for(loop, iteration)
    if problems:
        return run_id, {}, [f"{p.path}: {p.message}" for p in problems[:10]]
    manifest = {
        "run_id": run_id,
        "engine": "plan",
        "requested": None,
        "hint": None,
        "reason": f"server-hosted research loop {loop.loop_id}, iteration {iteration}",
        "plan_hash": canonical_hash(program),
        "project": loop.project_id,
        "run_name": shape.name,
        "max_tokens": None,
        "concurrency": None,
        "default_model": meta.get("default_model") or None,
        "tool_capabilities": capabilities,
        "project_root": None,
        "loop_id": loop.loop_id,
        "iteration": iteration,
    }
    definition = {
        "plan": program,
        "params": dict(program.get("_bound_params") or {}),
        "manifest": manifest,
    }
    return run_id, definition, []


def run_outcome(view: dict[str, Any]) -> dict[str, Any] | None:
    """The outcome a run's worker recorded, or None while it runs."""
    if view.get("status") == "cancelled":
        return {"status": "cancelled", "error": "the run's record was cancelled"}
    attempts = [a for a in view.get("attempts") or () if isinstance(a, dict)]
    last = attempts[-1] if attempts else None
    if last and last.get("plan_status"):
        summary = last.get("summary") if isinstance(last.get("summary"), dict) else {}
        return {
            "status": last["plan_status"],
            "error": summary.get("error") or last.get("exit_reason") or "",
            "summary": summary,
        }
    venue = view.get("venue") or {}
    ended = venue.get("state") in ("failed", "stopped")
    if ended and not (last and last.get("status") == "running"):
        why = venue.get("reason") or venue["state"]
        return {"status": "failed", "error": f"worker_ended: {why}"}
    return None


# ── the executor: one round ─────────────────────────────────────────────────


class PlanRunIteration:
    """Start one venue-hosted run, wait for its outcome, decide (one round)."""

    def __init__(
        self,
        *,
        host: PlanRunHost | None = None,
        poll_seconds: float = POLL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._host = host
        self._poll = poll_seconds
        self._sleep = sleep

    def host(self) -> PlanRunHost | None:
        return self._host or configured_host()

    def run(
        self,
        loop: ResearchLoop,
        iteration: int,
        remaining_sec: float | None,
        cancel_check: Callable[[], bool],
    ) -> dict[str, Any]:
        host = self.host()
        refused = _refusal(host, remaining_sec)
        if refused:
            return _not_run(iteration, refused)
        run_id, definition, problems = round_definition(loop, iteration, remaining_sec)
        if problems:
            return _not_run(iteration, "plan_invalid: " + "; ".join(problems))
        scope = _scope(loop, run_id)
        try:
            host.start(**scope, definition=definition, caps=_caps(remaining_sec), **_venue(loop))
        except Exception as exc:  # noqa: BLE001 — a refused start stops the loop, named
            return _not_run(iteration, f"plan_refused: {type(exc).__name__}: {exc}")
        outcome = self._wait(host, scope, cancel_check)
        return self._conclude(host, loop, iteration, run_id, outcome)

    def _wait(
        self, host: PlanRunHost, scope: dict[str, str], cancel_check: Callable[[], bool]
    ) -> dict[str, Any]:
        while True:
            outcome = run_outcome(host.view(**scope))
            if outcome is not None:
                return outcome
            if cancel_check():
                host.cancel(**scope)
                return {"status": "cancelled", "error": "the research loop was stopped"}
            self._sleep(self._poll)

    def _conclude(
        self,
        host: PlanRunHost,
        loop: ResearchLoop,
        iteration: int,
        run_id: str,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        shape = loop_plan.loop_template(loop.metadata[META_KEY].get("template"))
        summary = outcome.get("summary") or {}
        review = loop_plan.review_of(summary.get("output"), shape)
        decision = loop_plan.loop_decision(outcome.get("status"), review, outcome.get("error"))
        result: dict[str, Any] = {
            "iteration": iteration,
            "phase": "plan_iteration",
            "plan": {
                "template": shape.name,
                "run_id": run_id,
                "status": outcome.get("status"),
                "venue": "server",
                "error": str(outcome.get("error") or "")[: loop_plan.MAX_REASON] or None,
                "spent_tokens": summary.get("spent_tokens"),
            },
            "review": loop_plan.review_record(review),
            loop_plan.LOOP_DECISION_KEY: decision,
        }
        winner = loop_plan.picked_winner(review)
        if winner is not None and decision["decision"] == loop_plan.DECISION_PUBLISH:
            finding = _propose(host, loop, iteration, run_id, winner)
            result["finding"] = finding
            result["catalog_candidate"] = loop_plan.catalog_candidate(
                loop_id=loop.loop_id,
                iteration=iteration,
                goal=loop.research_goal,
                run_id=run_id,
                winner=winner,
                finding=finding,
            )
        return result


def _refusal(host: PlanRunHost | None, remaining_sec: float | None) -> str:
    if remaining_sec is not None and remaining_sec < 1:
        return "budget_exhausted:wall_clock"
    if host is None:
        return "plan_host_unavailable: this server has no plan-run venue wired"
    missing = host.venue_missing()
    if missing:
        return (
            "plan_iteration_needs_server_venue: the frontend venue is not enabled or "
            f"not configured (missing: {', '.join(missing)})"
        )
    return ""


def _scope(loop: ResearchLoop, run_id: str) -> dict[str, str]:
    return {
        "owner": str(loop.metadata.get("owner_user_id") or ""),
        "project": loop.project_id,
        "run_id": run_id,
    }


def _venue(loop: ResearchLoop) -> dict[str, Any]:
    meta = loop.metadata[META_KEY]
    return {
        "source_refs": list(meta.get("sources") or []),
        "default_source": str(meta.get("default_source") or ""),
        "web_policy": str(meta.get("web_policy") or ""),
    }


def _caps(remaining_sec: float | None) -> dict[str, Any]:
    return {"max_wall_clock_sec": max(1, int(remaining_sec))} if remaining_sec else {}


def _propose(
    host: PlanRunHost, loop: ResearchLoop, iteration: int, run_id: str, winner: dict[str, Any]
) -> dict[str, Any]:
    """The pick as a PROPOSED finding (pending; a person decides it)."""
    operation = loop_plan.finding_operation(
        loop_id=loop.loop_id, iteration=iteration, run_id=run_id, winner=winner
    )
    reason = (
        f"Research loop {loop.loop_id}, iteration {iteration}: the reviewers picked this "
        f"design for the goal {loop.research_goal[:200]!r}. Proposed only; a person "
        "decides (ADR-050)."
    )
    try:
        return host.propose(
            project=loop.project_id,
            owner=str(loop.metadata.get("owner_user_id") or ""),
            reason=reason,
            source_artifacts=[f"team_run:{run_id}"],
            operation=operation,
        )
    except Exception as exc:  # noqa: BLE001 — a failed proposal never fails the loop
        return {"status": "not_proposed", "reason": f"propose failed: {exc}"[:300]}


def _not_run(iteration: int, reason: str) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "phase": "plan_iteration",
        "plan": {"status": "not_run", "venue": "server"},
        loop_plan.LOOP_DECISION_KEY: {
            "decision": loop_plan.DECISION_STOP,
            "reason": reason[: loop_plan.MAX_REASON],
        },
    }


# ── the engine surface the runner drives ────────────────────────────────────


class PlanIterationEngine:
    """The plan executor under the one engine driver (as ``AgenticEngine``)."""

    def __init__(self, *, executor: PlanRunIteration | None = None) -> None:
        self._executor = executor or PlanRunIteration()
        # Wired by EngineLoopRunner.drive before run().
        self._iteration_hook: Callable[[LoopIteration], None] = lambda _it: None
        self._cancel_check: Callable[[], bool] = lambda: False
        self._approval_gate: Callable[[dict], bool | str] = lambda _action: True

    def run(self, loop: ResearchLoop) -> ResearchLoop:
        while (
            loop.status == LoopStatus.RUNNING
            and loop.current_iteration < loop.max_iterations
            and not self._cancel_check()
        ):
            iteration = self._advance_one(loop)
            self._iteration_hook(iteration)
            if self._apply(loop, iteration):
                return loop
        if loop.status == LoopStatus.RUNNING and not self._cancel_check():
            _finish(loop, "max_iterations")
        return loop

    def _advance_one(self, loop: ResearchLoop) -> LoopIteration:
        loop.current_iteration += 1
        number = loop.current_iteration
        remaining = remaining_wallclock_sec(loop)
        meta = loop.metadata[META_KEY]
        action = {
            "kind": PLAN_RUN,
            "iteration": number,
            "loop_id": loop.loop_id,
            "template": meta.get("template"),
            "goal": loop.research_goal[: loop_plan.MAX_REASON],
            "args": dict(meta.get("args") or {}),
            "remaining_wallclock_sec": remaining,
        }
        verdict = self._approval_gate(action)
        if verdict == "deferred":
            return _record(loop, number, {"deferred_gate": PLAN_RUN}, "awaiting_decision")
        if verdict is not True:
            summary = _not_run(number, f"approval_rejected:{PLAN_RUN}")
            return _record(loop, number, summary, "rejected")
        summary = self._executor.run(loop, number, remaining, self._cancel_check)
        return _record(loop, number, summary, "completed")

    def _apply(self, loop: ResearchLoop, iteration: LoopIteration) -> bool:
        """Act on the round's decision; True when the loop has ended."""
        decision = (iteration.results or {}).get(loop_plan.LOOP_DECISION_KEY) or {}
        kind = decision.get("decision")
        rounds = list(loop.metadata.get(ROUNDS_KEY) or [])
        if "review" in iteration.results:
            rounds.append(
                {"iteration": iteration.iteration_number, "review": iteration.results["review"]}
            )
            loop.metadata[ROUNDS_KEY] = rounds[-loop_plan.CONTEXT_ROUNDS :]
        if kind == loop_plan.DECISION_PUBLISH:
            _finish(loop, f"published:{decision.get('reason') or ''}")
            return True
        if kind == loop_plan.DECISION_STOP:
            _finish(loop, str(decision.get("reason") or "stopped"))
            return True
        return False


def remaining_wallclock_sec(loop: ResearchLoop) -> float | None:
    """What the loop has left of its ADR-067 wall clock (None = uncapped)."""
    from .engine_loop_runner import _elapsed_hours

    try:
        cap = float(loop.metadata.get("max_wall_clock_hours") or 0.0)
    except (TypeError, ValueError):
        return None
    if cap <= 0:
        return None
    return max(0.0, (cap - _elapsed_hours(loop)) * 3600.0)


def _record(loop: ResearchLoop, number: int, summary: dict[str, Any], status: str) -> LoopIteration:
    now = datetime.now(timezone.utc).isoformat()
    iteration = LoopIteration(
        iteration_id=f"it-{loop.loop_id}-{number}",
        loop_id=loop.loop_id,
        iteration_number=number,
        phase=LoopPhase.EXPERIMENT_EXECUTION,
        started_at=now,
        completed_at=now,
        results=dict(summary),
        evaluation=str((summary.get(loop_plan.LOOP_DECISION_KEY) or {}).get("reason") or ""),
        findings=[],
        status=status,
    )
    loop.iterations.append(iteration)
    loop.updated_at = now
    return iteration


def _finish(loop: ResearchLoop, reason: str) -> None:
    loop.metadata["state_reason"] = reason[: loop_plan.MAX_REASON]
    loop.phase = LoopPhase.COMPLETED
    loop.status = LoopStatus.STOPPED
    loop.completed_at = datetime.now(timezone.utc).isoformat()


def engine_factory() -> Callable[[], PlanIterationEngine]:
    return lambda: PlanIterationEngine()


__all__ = [
    "META_KEY",
    "PLAN_RUN",
    "PlanIterationEngine",
    "PlanRunHost",
    "PlanRunIteration",
    "configure",
    "configured_host",
    "engine_factory",
    "is_plan_loop",
    "remaining_wallclock_sec",
    "round_definition",
    "run_id_for",
    "run_outcome",
]
