"""UMRS agent loop runner (Wave 9, ADR-009 parity with r4's LocalRuntime daemon).

A registered agent loop advances on a background thread, one LLM-decisioned
iteration per step, mirroring the r4 ``LocalRuntime`` daemon semantics without
giving up UMRS-side durability:

- ``LoopStore`` is refreshed before and after every iteration (agent state
  never lives longer in memory than in the store).
- ``pause`` / ``stop`` / ``resume`` are honored via the persisted loop record
  — same cooperative protocol as r4's ``_LoopControl``.
- ``AutonomousResearchLoop`` stays in charge of the full engine path
  (simulated/planned loops); the runner owns agent-decisioned loops.
- On UMRS restart, in-flight *agented* loops are re-registered from the store
  (durability crossing process boundaries — matching the durable-honest pause
  contract used everywhere else).

Runner registration is module-global so the REST router spawns it via
``register(loop, ...)``; tests may drive it synchronously via ``kick()`` +
``run_agentic(loop)`` without touching threads.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .agent_executor import UmrsAgentIterationExecutor
from .engine_loop_runner import save_keeping_control
from .loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop

logger = logging.getLogger(__name__)


class AgentLoopRunner(threading.Thread):
    """Drive one loop through ``executor(loop_id, iteration_number, goal)``.

    Cheap on purpose: a daemon thread per agented loop; persistence via the
    ``LoopStore`` injected at registration. Tests: call ``kick(loop)`` in the
    main thread instead of spawning the thread.
    """

    def __init__(
        self,
        loop: ResearchLoop,
        *,
        executor: Callable[[str, int, str], dict[str, Any]],
        store: Any,
        poll_seconds: float = 0.5,
    ) -> None:
        super().__init__(name=f"agent-loop-{loop.loop_id}", daemon=True)
        self._loop = loop
        self._executor = executor
        self._store = store
        self._poll_seconds = poll_seconds
        self._control = threading.Event()  # set => cooperative early stop

    # -- loop semantics (mirrors r4 LocalRuntime daemon) -------------------

    def abort(self) -> None:
        """Mark the runner for early exit at the next iteration boundary."""
        self._control.set()

    def run(self) -> None:  # Thread.run
        try:
            loop_out = self.run_agentic(self._loop)
            save_keeping_control(self._store, loop_out)
            logger.info(
                "agent loop %s finished status=%s (%d iterations)",
                loop_out.loop_id,
                loop_out.status.value,
                len(loop_out.iterations),
            )
        except Exception as exc:  # noqa: BLE001 — never strand iterations untracked
            logger.error("agent loop %s failed: %s", self._loop.loop_id, exc)
            self._loop.status = LoopStatus.ERROR
            self._loop.metadata["error"] = str(exc)
            save_keeping_control(self._store, self._loop)

    def kick(self, loop: ResearchLoop) -> ResearchLoop:
        """In-thread driver for tests (no threads, deterministic)."""
        return self.run_agentic(loop)

    def run_agentic(self, loop: ResearchLoop) -> ResearchLoop:
        while (
            loop.status == LoopStatus.RUNNING
            and loop.current_iteration < loop.max_iterations
            and not self._control.is_set()
        ):
            loop = self._advance_one(loop)
            save_keeping_control(self._store, loop)
            advance, loop = self._service_control(loop)
            if not advance:
                break

        if loop.status == LoopStatus.RUNNING and not self._control.is_set():
            loop.phase = LoopPhase.COMPLETED
            loop.status = LoopStatus.STOPPED
            loop.completed_at = datetime.now(timezone.utc).isoformat()
        return loop

    def _service_control(self, loop: ResearchLoop) -> tuple[bool, ResearchLoop]:
        """Refresh the loop from the store and handle stop/pause/resume.

        Returns (should_advance, loop): False when the loop should exit before
        the next iteration (vanished, stopped, or resumed anymore)."""
        pending = self._poll_loop(loop.loop_id)
        if pending is None:  # loop vanished from the store — abort politely
            return False, self._finish(loop)
        if pending.status == LoopStatus.STOPPED:
            return False, self._finish(loop)
        if pending.status == LoopStatus.PAUSED:
            self._wait_resumed(pending.loop_id)
            loop = self._refresh(loop.loop_id) or loop
            return loop.status == LoopStatus.RUNNING, loop
        return True, loop

    @staticmethod
    def _finish(loop: ResearchLoop) -> ResearchLoop:
        loop.status = LoopStatus.STOPPED
        loop.completed_at = datetime.now(timezone.utc).isoformat()
        return loop

    # -- single iteration ---------------------------------------------------

    def _advance_one(self, loop: ResearchLoop) -> ResearchLoop:
        loop.current_iteration += 1
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
        iteration.metadata = summary  # W8 requires hint durable on the iteration
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
        return loop

    # -- control protocol -----------------------------------------------------

    def _poll_loop(self, loop_id: str) -> ResearchLoop | None:
        try:
            return self._store.get(loop_id)
        except Exception as exc:  # noqa: BLE001 — treat poll failure as missing
            logger.warning("agent loop poll failed for %s: %s", loop_id, exc)
            return None

    def _refresh(self, loop_id: str) -> ResearchLoop | None:
        return self._poll_loop(loop_id)

    def _wait_resumed(self, loop_id: str) -> None:
        """Hold the loop until drive resumes (or stop wins)."""
        while not self._control.is_set():
            time.sleep(self._poll_seconds)
            pending = self._poll_loop(loop_id)
            if pending is None or pending.status in (
                LoopStatus.RUNNING,
                LoopStatus.STOPPED,
            ):
                return


# ── Global registration table (REST router hooks up here) ─────────────────


_RUNNERS: dict[str, AgentLoopRunner] = {}
_LOCK = threading.Lock()


def register(
    loop: ResearchLoop,
    *,
    executor: Callable[[str, int, str], dict[str, Any]],
    store: Any,
    poll_seconds: float = 0.5,
) -> AgentLoopRunner:
    """Spawn a daemon runner for ``loop``. Returns the runner for testing.

    Registering twice is idempotent (callers ask for the live runner with
    ``get(loop_id)`` instead of duplicating).
    """
    with _LOCK:
        existing = _RUNNERS.get(loop.loop_id)
        if existing is not None and existing.is_alive():
            return existing
        runner = AgentLoopRunner(
            loop, executor=executor, store=store, poll_seconds=poll_seconds
        )
        _RUNNERS[loop.loop_id] = runner
    runner.start()
    return runner


def get(loop_id: str) -> AgentLoopRunner | None:
    """Live runner (or None) — tests + lifecycle ops use this."""
    with _LOCK:
        runner = _RUNNERS.get(loop_id)
        return runner if (runner and runner.is_alive()) else None


def gracefully_stop(loop_id: str) -> None:
    """Cooperatively halt the runner for ``loop_id`` (no record mutation)."""
    runner = get(loop_id)
    if runner is not None:
        runner.abort()


def stop_all(timeout_seconds: float = 5.0) -> None:
    """Abort every live runner, join it, then clear the registry.

    Joining matters: an aborted-but-still-finishing thread writes its final
    checkpoint into the store — which races the test/store reset that
    typically follows stop_all (resurrected loops in the next test).
    """
    with _LOCK:
        runners = list(_RUNNERS.values())
    for runner in runners:
        if runner.is_alive():
            runner.abort()
    deadline = time.monotonic() + timeout_seconds
    for runner in runners:
        if runner.is_alive():
            runner.join(timeout=max(0.0, deadline - time.monotonic()))
    with _LOCK:
        for runner in runners:
            if not runner.is_alive():
                _RUNNERS.pop(runner._loop.loop_id, None)


def default_executor_for_project(project: Any, model: str | None = None):
    """Default agent executor factory (knowledge-graph-liteLLM wiring)."""
    return UmrsAgentIterationExecutor(project, model=model)


def kick(
    loop: ResearchLoop,
    *,
    executor: Callable[[str, int, str], dict[str, Any]],
    store: Any,
) -> ResearchLoop:
    """Synchronous driver for tests: advance to completion in the caller's
    thread (never spawns, deterministic)."""
    runner = AgentLoopRunner(loop, executor=executor, store=store, poll_seconds=0)
    return runner.kick(loop)
