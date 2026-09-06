"""UMRS engine loop runner — server-side driver for non-agentic loops.

Closes the REST gap where ``POST /umrs/loops`` created a RUNNING loop whose
iterations were only advanced by a client (companion) calling run itself.
With this runner, the server drives the phased engine path
(``AutonomousResearchLoop.run``) on a daemon thread, mirroring the agent
runner's durability contract:

- ``LoopStore`` is the loop record of truth: saved after every engine
  iteration (the engine's ``iteration_hook`` writes the checkpoint), and
  polled between iterations for cooperative control.
- ``pause`` / ``resume`` / ``stop`` ride the same store protocol as the
  agent runner — the driver waits while PAUSED, and cancels between
  iterations when the record turns STOPPED (or vanishes).
- One fresh engine per loop: the module-global REST engine is only used for
  plan construction, so a cancel on one loop never bleeds into another.

Tests drive deterministically via ``kick(loop, engine=..., store=...)``;
no threads are spawned outside ``register()``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .loop import AutonomousResearchLoop, LoopIteration, LoopStatus, ResearchLoop

logger = logging.getLogger(__name__)

EngineFactory = Callable[[], AutonomousResearchLoop]
IterationHook = Callable[[ResearchLoop, LoopIteration], None]


class EngineLoopRunner(threading.Thread):
    """Drive one loop through the phased engine to completion.

    Cheap on purpose: a daemon thread per non-agentic loop, persistence via
    the injected ``LoopStore``. The engine's built-in cancel_check is wired
    to the store poll so stop/pause are honored between iterations.
    """

    def __init__(
        self,
        loop: ResearchLoop,
        *,
        engine_factory: EngineFactory,
        store: Any,
        poll_seconds: float = 0.5,
        on_iteration: Optional[IterationHook] = None,
    ) -> None:
        super().__init__(name=f"engine-loop-{loop.loop_id}", daemon=True)
        self._loop = loop
        self._engine_factory = engine_factory
        self._store = store
        self._poll_seconds = poll_seconds
        self._on_iteration = on_iteration
        self._abort = threading.Event()

    def abort(self) -> None:
        """Cooperative early stop at the next iteration boundary."""
        self._abort.set()

    def run(self) -> None:  # Thread.run
        try:
            loop_out = self.drive(self._loop)
            # Vanish guard: a store wipe/delete while finishing means the
            # record is gone on purpose — do not resurrect it with this save
            # (test resets, explicit deletes in prod).
            if self._poll(loop_out.loop_id) is not None:
                self._store.save(loop_out)
            logger.info(
                "engine loop %s finished status=%s (%d iterations)",
                loop_out.loop_id,
                loop_out.status.value,
                len(loop_out.iterations),
            )
        except Exception as exc:  # noqa: BLE001 — never strand iterations untracked
            logger.error("engine loop %s failed: %s", self._loop.loop_id, exc)
            self._loop.status = LoopStatus.ERROR
            self._loop.metadata["error"] = str(exc)
            self._loop.completed_at = datetime.now(timezone.utc).isoformat()
            self._store.save(self._loop)

    # -- control protocol (store is the source of truth) --------------------

    def _poll(self, loop_id: str) -> Optional[ResearchLoop]:
        try:
            return self._store.get(loop_id)
        except Exception as exc:  # noqa: BLE001 — treat poll failure as missing
            logger.warning("engine loop poll failed for %s: %s", loop_id, exc)
            return None

    def _cancel_check(self) -> bool:
        """Engine-facing cancel probe: waits out pauses, cancels on stop."""
        while True:
            if self._abort.is_set():
                return True
            pending = self._poll(self._loop.loop_id)
            if pending is None or pending.status == LoopStatus.STOPPED:
                return True
            if pending.status == LoopStatus.PAUSED:
                time.sleep(self._poll_seconds)
                continue
            return False

    def _make_approval_gate(self, loop: ResearchLoop) -> Callable[[dict], bool]:
        """Ask-mode action gate: request approval, park until decided.

        auto-mode returns True immediately (current behavior). On rejection
        or driver abort returns False; the engine raises on False. Intents
        live in the durable approval inbox (ADR-036 (a)) keyed by the loop
        owner + hosting job/attempt; a restarted loop re-requests under the
        same id from its metadata marker and finds any decision recorded
        while it was down (request_approval is idempotent under approval_id).
        """

        def _request(action: dict, approval_id: Optional[str]) -> dict:
            from . import tool_approvals

            return tool_approvals.request_approval(
                loop.loop_id,
                int(action.get("iteration", 0)),
                action,
                approval_id=approval_id,
                user_id=str(loop.metadata.get("owner_user_id") or ""),
                job_id=str(loop.metadata.get("durable_job_id") or ""),
                attempt_id=str(loop.metadata.get("attempt_id") or ""),
            )

        def _gate(action: dict) -> bool:
            if loop.metadata.get("mode", "auto") != "ask":
                return True
            from . import tool_approvals

            record = _request(action, loop.metadata.get("awaiting_approval") or None)
            loop.metadata["awaiting_approval"] = record["id"]
            self._store.save(loop)
            logger.info(
                "loop %s parked for approval %s (action=%s)",
                loop.loop_id,
                record["id"],
                action.get("kind"),
            )
            while True:
                if self._abort.is_set():
                    return False
                current = tool_approvals.get(record["id"])
                if current is None:
                    current = _request(action, record["id"])
                status = current.get("status")
                if status == "approved":
                    loop.metadata.pop("awaiting_approval", None)
                    self._store.save(loop)
                    return True
                if status == "rejected":
                    loop.metadata.pop("awaiting_approval", None)
                    self._store.save(loop)
                    return False
                time.sleep(self._poll_seconds)

        return _gate

    # -- drive --------------------------------------------------------------

    def _checkpoint(self, loop: ResearchLoop, iteration: LoopIteration) -> None:
        """Engine iteration hook: persist, then run the hosting-side hook.

        The hook (loop_hosting: durable-job checkpoint + finding minting)
        is best-effort — it may enrich the iteration and re-save, but it
        never breaks the loop's own persistence.
        """
        self._store.save(loop)
        if self._on_iteration is None:
            return
        try:
            self._on_iteration(loop, iteration)
        except Exception as exc:  # noqa: BLE001 — hosting hook must not kill the loop
            logger.warning("iteration hook failed for %s: %s", loop.loop_id, exc)

    def drive(self, loop: ResearchLoop) -> ResearchLoop:
        engine = self._engine_factory()

        # Checkpoint the live record after each engine iteration.
        engine._iteration_hook = lambda it: self._checkpoint(loop, it)  # noqa: SLF001
        # Cooperative cancel/pause via the store poll.
        engine._cancel_check = self._cancel_check  # noqa: SLF001
        # Tool-grain gate: ask-mode loops park at each workflow trigger until
        # a decision lands in the durable approval inbox (ADR-036 (a)).
        engine._approval_gate = self._make_approval_gate(loop)  # noqa: SLF001

        result = engine.run(loop)
        if result.status == LoopStatus.RUNNING:
            # Belt-and-braces: run() already finalizes, but a cooperative
            # cancel leaves the record RUNNING with cancel_requested set.
            result.status = LoopStatus.STOPPED
            result.completed_at = datetime.now(timezone.utc).isoformat()
        result.updated_at = datetime.now(timezone.utc).isoformat()
        return result


# ── Global registration table (REST router hooks up here) ─────────────────


_RUNNERS: dict[str, EngineLoopRunner] = {}
_LOCK = threading.Lock()


def register(
    loop: ResearchLoop,
    *,
    engine_factory: EngineFactory,
    store: Any,
    poll_seconds: float = 0.5,
    on_iteration: Optional[IterationHook] = None,
) -> EngineLoopRunner:
    """Spawn a daemon engine runner for ``loop``. Idempotent when alive."""
    with _LOCK:
        existing = _RUNNERS.get(loop.loop_id)
        if existing is not None and existing.is_alive():
            return existing
        runner = EngineLoopRunner(
            loop,
            engine_factory=engine_factory,
            store=store,
            poll_seconds=poll_seconds,
            on_iteration=on_iteration,
        )
        _RUNNERS[loop.loop_id] = runner
    runner.start()
    return runner


def get(loop_id: str) -> Optional[EngineLoopRunner]:
    """Live runner (or None)."""
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
