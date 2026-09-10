"""Loop hosting — the seam between the ``research_loop_job`` kind and drivers.

ADR-036 D1/D2 (WP-05 round 1): a server-side research loop is hosted by a
durable job (one job per loop; attempt = one hosted run). The scheduler's
backend calls :func:`host` when an attempt is submitted; the status checker
reads :func:`snapshot`. Both live in ``jobs/`` and must not import the REST
layer, so the REST layer (``umrs.api.loop``) *configures* this registry with
getters for the loop store, engine factory, agent executor, checkpoint store
and finding sink — resolved at call time, because the store is upgraded to
Postgres at boot.

Resume semantics (ADR-036 (b)): ``host`` never resets a loop. A LOST attempt
is requeued by the reconciler, the scheduler submits a new attempt, and the
driver continues from the stored ``current_iteration`` — the campaign resumes
at its last iteration checkpoint instead of restarting. Terminal loops
(STOPPED/ERROR) are not re-driven; the checker reports them.

Per-iteration hooks (both hosting modes):
- findings of a completed iteration are minted as knowledge artifacts with
  provenance (``finding_ingest``), ids recorded on the iteration;
- job-hosted loops additionally write one ``checkpoints`` row per iteration
  (``CheckpointStore``) so Job/Attempt accounting sees progress.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import finding_ingest
from .loop import LoopIteration, LoopStatus, ResearchLoop

logger = logging.getLogger(__name__)

HOSTING_JOB = "job"
HOSTING_THREAD = "thread"


@dataclass
class HostingWiring:
    """Late-bound accessors supplied by the REST layer."""

    store: Callable[[], Any]
    engine_factory: Callable[[], Callable[[], Any]]
    agent_executor: Callable[[str], Callable[..., Any]]
    checkpoint_store: Callable[[], Any | None]
    finding_sink: Callable[[], Any | None]


_WIRING: HostingWiring | None = None
_LOCK = threading.Lock()


def configure(wiring: HostingWiring | None) -> None:
    """Install (or reset) the hosting wiring."""
    global _WIRING
    with _LOCK:
        _WIRING = wiring


def _wiring() -> HostingWiring:
    with _LOCK:
        if _WIRING is None:
            raise RuntimeError(
                "loop hosting not configured (umrs.api.loop wires it at import)"
            )
        return _WIRING


def is_configured() -> bool:
    with _LOCK:
        return _WIRING is not None


def _safe(getter: Callable[[], Any], label: str) -> Any:
    try:
        return getter()
    except Exception as exc:  # noqa: BLE001 — accessors degrade to None, reported
        logger.warning("loop hosting %s unavailable: %s", label, exc)
        return None


# ── iteration hooks ─────────────────────────────────────────────────────────


def _mint_findings(loop: ResearchLoop, iteration: LoopIteration, store: Any) -> None:
    sink = _safe(_wiring().finding_sink, "finding sink")
    outcome = finding_ingest.mint_iteration(loop, iteration, sink=sink)
    if outcome.minted or outcome.errors or iteration.artifact_ids:
        store.save(loop)


def _checkpoint_iteration(
    loop: ResearchLoop, iteration: LoopIteration, *, job_id: str, attempt_id: str
) -> None:
    ckpt = _safe(_wiring().checkpoint_store, "checkpoint store")
    if ckpt is None or not job_id:
        return
    progress = (
        iteration.iteration_number / loop.max_iterations if loop.max_iterations else 0.0
    )
    ckpt.save_checkpoint(
        job_id,
        attempt_id,
        "iteration",
        iteration=iteration.iteration_number,
        progress=min(1.0, max(0.0, progress)),
        state_inline={
            "loop_id": loop.loop_id,
            "iteration_id": iteration.iteration_id,
            "status": iteration.status,
            "phase": iteration.phase.value,
            "artifact_ids": list(iteration.artifact_ids),
            "finding_ids": [
                str(
                    f.get("finding_id")
                    if isinstance(f, dict)
                    else getattr(f, "finding_id", "")
                )
                for f in iteration.findings
            ],
        },
        metadata={"kind": "research_loop_iteration"},
    )


def iteration_hook(
    *, job_id: str = "", attempt_id: str = ""
) -> Callable[[ResearchLoop, LoopIteration], None]:
    """Hosting-side hook run by the engine driver after each checkpoint."""

    def _hook(loop: ResearchLoop, iteration: LoopIteration) -> None:
        store = _wiring().store()
        _mint_findings(loop, iteration, store)
        if iteration.status == "completed":
            _checkpoint_iteration(loop, iteration, job_id=job_id, attempt_id=attempt_id)

    return _hook


# ── hosting ─────────────────────────────────────────────────────────────────


def rewind_interrupted_iteration(loop: ResearchLoop) -> dict[str, int] | None:
    """Make a resumed loop redo the iteration that was in flight when it died.

    The engine increments ``current_iteration`` when an iteration *starts* and
    the record is checkpointed mid-iteration (approval park, post-trigger), so
    a loop killed mid-iteration comes back with ``current_iteration`` one past
    its persisted iterations — and would resume as if that iteration had
    completed (live finding, WP-07: a mid-wait loop resumed straight to
    "completed" with zero iterations). Rewind to the persisted count; the
    ``awaiting_approval`` marker is kept so a re-run gate re-requests the same
    durable intent (already decided → continues; pending → parks again).
    """
    persisted = len(loop.iterations)
    if loop.current_iteration <= persisted:
        return None
    rewind = {"from": loop.current_iteration, "to": persisted}
    loop.current_iteration = persisted
    loop.metadata["resume_rewind"] = rewind
    return rewind


def _register_driver(
    loop: ResearchLoop, store: Any, *, job_id: str, attempt_id: str
) -> str:
    from . import agent_loop_runner, engine_loop_runner

    wiring = _wiring()
    if loop.metadata.get("agent"):
        agent_loop_runner.register(
            loop, executor=wiring.agent_executor(loop.project_id), store=store
        )
        return "agent"
    engine_loop_runner.register(
        loop,
        engine_factory=wiring.engine_factory(),
        store=store,
        on_iteration=iteration_hook(job_id=job_id, attempt_id=attempt_id),
    )
    return "engine"


def host(loop_id: str, *, job_id: str, attempt_id: str) -> str:
    """Drive ``loop_id`` under durable job ``job_id`` / ``attempt_id``.

    Returns the driver kind (``engine`` | ``agent``) or ``terminal`` when the
    loop already ended (no driver spawned). Raises ``LookupError`` when the
    loop record does not exist — the attempt then fails with a clear reason.
    Resumes from the stored ``current_iteration``; never resets the campaign.
    """
    store = _wiring().store()
    loop = store.get(loop_id)
    if loop is None:
        raise LookupError(f"research loop not found: {loop_id}")
    if loop.status in (LoopStatus.STOPPED, LoopStatus.ERROR):
        return "terminal"
    rewind_interrupted_iteration(loop)
    loop.metadata["hosting"] = HOSTING_JOB
    loop.metadata["durable_job_id"] = job_id
    loop.metadata["attempt_id"] = attempt_id
    store.save(loop)
    kind = _register_driver(loop, store, job_id=job_id, attempt_id=attempt_id)
    logger.info(
        "research loop %s hosted by job %s attempt %s (driver=%s, resume at iteration %d)",
        loop_id,
        job_id,
        attempt_id,
        kind,
        loop.current_iteration,
    )
    return kind


def driver_alive(loop_id: str) -> bool:
    from . import agent_loop_runner, engine_loop_runner

    return (
        engine_loop_runner.get(loop_id) is not None
        or agent_loop_runner.get(loop_id) is not None
    )


def snapshot(loop_id: str) -> dict[str, Any] | None:
    """Checker-facing view of a hosted loop (None when the record is gone)."""
    loop = _wiring().store().get(loop_id)
    if loop is None:
        return None
    return {
        "status": loop.status.value,
        "phase": loop.phase.value,
        "current_iteration": loop.current_iteration,
        "max_iterations": loop.max_iterations,
        "total_findings": loop.total_findings,
        "error": str(loop.metadata.get("error") or ""),
        "driver_alive": driver_alive(loop_id),
        "updated_at": loop.updated_at or "",
    }


__all__ = [
    "HOSTING_JOB",
    "HOSTING_THREAD",
    "HostingWiring",
    "configure",
    "driver_alive",
    "host",
    "is_configured",
    "iteration_hook",
    "rewind_interrupted_iteration",
    "snapshot",
]
