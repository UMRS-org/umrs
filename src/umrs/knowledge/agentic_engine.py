"""Agentic engine adapter — the agent executor under the engine driver (ADR-064).

The agent runner's distinct value (one LLM-decisioned iteration per step)
hosts under ``EngineLoopRunner`` through the engine surface the runner
drives: ``run(loop)`` plus the ``_iteration_hook`` / ``_cancel_check`` /
``_approval_gate`` attributes the runner wires before ``run()``.

Semantics preserved from ``agent_loop_runner`` (= API parity, no behavior
drift): store-checkpoint per iteration via the hook, cooperative cancel
between iterations, completion at ``max_iterations``. The approval gate
fires per agent action when the engine wires it (ask-mode parity with the
engine path).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop

logger = logging.getLogger(__name__)

AgentExecutor = Callable[[str, int, str], dict[str, Any]]


def _deferred_agent_iteration(loop: ResearchLoop) -> LoopIteration:
    """D-34 first leg: an iteration awaiting the durable decision — no
    executor ran, nothing fabricated (the runner's ledger bounds spin)."""
    now = datetime.now(timezone.utc).isoformat()
    return LoopIteration(
        iteration_id=f"it-{loop.loop_id}-{loop.current_iteration}",
        loop_id=loop.loop_id,
        iteration_number=loop.current_iteration,
        phase=LoopPhase.EXPERIMENT_EXECUTION,
        started_at=now,
        completed_at=now,
        results={"deferred_gate": "agent_iteration"},
        findings=[],
        status="awaiting_decision",
    )


class AgenticEngine:
    """Engine-surface wrapper around one agent iteration executor."""

    def __init__(self, *, executor: AgentExecutor) -> None:
        self._executor = executor
        # Wired by EngineLoopRunner.drive before run().
        self._iteration_hook: Callable[[LoopIteration], None] = lambda it: None
        self._cancel_check: Callable[[], bool] = lambda: False
        # True = run, False = decided-no (raises), "deferred" = D-34 first leg
        # (profile-mode pending → iteration awaits the durable decision).
        self._approval_gate: Callable[[dict], bool | str] = lambda action: True

    def run(self, loop: ResearchLoop) -> ResearchLoop:
        """Drive agent-decisioned iterations to completion (or cancel)."""
        while (
            loop.status == LoopStatus.RUNNING
            and loop.current_iteration < loop.max_iterations
            and not self._cancel_check()
        ):
            iteration = self._advance_one(loop)
            self._iteration_hook(iteration)

        if loop.status == LoopStatus.RUNNING and not self._cancel_check():
            loop.phase = LoopPhase.COMPLETED
            loop.status = LoopStatus.STOPPED
            loop.completed_at = datetime.now(timezone.utc).isoformat()
        return loop

    def _advance_gated(self, loop: ResearchLoop, verdict: bool | str) -> LoopIteration:
        """A gated verdict's outcome: False raises (decided-no — today's
        contract); "deferred" records the awaiting_decision iteration (D-34
        first leg — the loop continues on the next ungated iteration)."""
        if verdict == "deferred":
            iteration = _deferred_agent_iteration(loop)
            loop.iterations.append(iteration)
            loop.updated_at = datetime.now(timezone.utc).isoformat()
            return iteration
        raise RuntimeError(
            f"agent iteration {loop.current_iteration} rejected by the approval gate"
        )

    def _advance_one(self, loop: ResearchLoop) -> LoopIteration:
        loop.current_iteration += 1
        action = {
            "kind": "agent_iteration",
            "iteration": loop.current_iteration,
            "loop_id": loop.loop_id,
        }
        verdict = self._approval_gate(action)
        if verdict is not True:
            return self._advance_gated(loop, verdict)
        summary = (
            self._executor(loop.loop_id, loop.current_iteration, loop.research_goal)
            or {}
        )
        hypothesis = summary.get("hypothesis") or ""
        iteration = LoopIteration(
            iteration_id=f"it-{loop.loop_id}-{loop.current_iteration}",
            loop_id=loop.loop_id,
            iteration_number=loop.current_iteration,
            phase=LoopPhase.EXPERIMENT_EXECUTION,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            hypothesis=hypothesis,
            experiment_id="",
            results=dict(summary),
            analysis=str(summary.get("evaluation") or ""),
            evaluation=str(summary.get("evaluation") or ""),
            findings=[],
            status="completed",
        )
        iteration.metadata = summary
        loop.iterations.append(iteration)
        if hypothesis:
            loop.total_hypotheses_tested += 1
        if summary.get("llm"):
            loop.total_experiments_run += 1
        loop.success_rate = (
            loop.total_hypotheses_tested / loop.total_experiments_run
            if loop.total_experiments_run
            else 0.0
        )
        loop.updated_at = datetime.now(timezone.utc).isoformat()
        return iteration
