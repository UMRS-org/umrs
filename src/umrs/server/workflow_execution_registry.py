"""In-process workflow execution registry.

Provides a thread-safe registry for tracking the status of workflow
executions launched via ``POST /umrs/workflows/{name}/trigger``.

The registry is intentionally light-weight (in-memory, single-process)
to unblock the perception → reasoning end-to-end story without
prematurely committing to a specific durable-job schema.  When the
durable Job/Attempt machinery is wired in, a thin adapter can replace
this module without changing callers.

Lifecycle:
    submitted -> running -> succeeded | failed | cancelled

Cancelling a submitted-but-not-yet-running execution marks it as
cancelled before the worker thread starts; cancelling a running
execution sets the cancel flag (cooperatively checked by the runner).
"""

from __future__ import annotations

import builtins
import json
import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Terminal statuses (no further updates expected).
_TERMINAL = {"succeeded", "failed", "cancelled"}


@dataclass
class StepProgress:
    """Per-step progress entry recorded during execution."""

    name: str
    status: str = "pending"  # pending | running | succeeded | failed | skipped
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    progress: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "progress": self.progress,
            "extra": self.extra,
        }


@dataclass
class ExecutionRecord:
    """One workflow execution tracked in memory."""

    execution_id: str
    workflow_name: str
    user_id: str
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "submitted"
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    progress: float = 0.0
    steps: dict[str, StepProgress] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    durable_job_id: str | None = None
    cancel_requested: bool = False
    identity_provenance: str = "unverified"
    # Manual approval gates (W5). Decisions are serializable audit data;
    # the wait events are transient threading primitives (never serialized).
    gate_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _gate_events: dict[str, threading.Event] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "execution_id": self.execution_id,
                "workflow_name": self.workflow_name,
                "user_id": self.user_id,
                "status": self.status,
                "submitted_at": self.submitted_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "progress": self.progress,
                "params": dict(self.params),
                "steps": [step.to_dict() for step in self.steps.values()],
                "artifacts": list(self.artifacts),
                "durable_job_id": self.durable_job_id,
                "cancel_requested": self.cancel_requested,
                "identity_provenance": self.identity_provenance,
                "gates": {k: dict(v) for k, v in self.gate_decisions.items()},
            }

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    def update(self, **fields_: Any) -> None:
        with self._lock:
            for key, value in fields_.items():
                setattr(self, key, value)

    def set_step(self, name: str, **fields_: Any) -> None:
        with self._lock:
            step = self.steps.setdefault(name, StepProgress(name=name))
            for key, value in fields_.items():
                setattr(step, key, value)

    def add_artifact(self, key: str) -> None:
        with self._lock:
            if key not in self.artifacts:
                self.artifacts.append(key)

    def set_durable_job_id(self, job_id: str) -> None:
        self.update(durable_job_id=job_id)

    # ------------------------------------------------------------------
    # Manual approval gates (W5)
    # ------------------------------------------------------------------

    def prepare_gate(self, step_name: str) -> threading.Event:
        """Create (or fetch) the wait event for a manual gate step."""
        with self._lock:
            return self._gate_events.setdefault(step_name, threading.Event())

    def resolve_gate(
        self,
        step_name: str,
        *,
        approved: bool,
        approver: str | None,
        comment: str | None,
    ) -> None:
        """Store an approval decision and release the waiting gate thread.

        The decision is appended to ``results["gate_audit"]`` so gate
        outcomes live in the durable record's audit trail. ``update()``
        is invoked with no fields purely so durable subclasses fire
        their persist hook. First decision wins — late duplicate calls
        are no-ops.
        """
        decision = {
            "step": step_name,
            "approved": bool(approved),
            "approver": approver,
            "comment": comment,
            "decided_at": time.time(),
        }
        with self._lock:
            if step_name in self.gate_decisions:
                return
            self.gate_decisions[step_name] = decision
            self.results.setdefault("gate_audit", []).append(dict(decision))
        self.update()  # persist hook (see DurableExecutionRecord)
        event = self._gate_events.get(step_name)
        if event is not None:
            event.set()


class WorkflowExecutionRegistry:
    """Thread-safe in-memory registry of workflow executions."""

    def __init__(self, max_records: int = 1000) -> None:
        self._records: dict[str, ExecutionRecord] = {}
        self._max = max_records
        self._lock = threading.Lock()
        self._workers: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        workflow_name: str,
        user_id: str,
        params: dict[str, Any] | None = None,
        execution_id: str | None = None,
        durable_job_id: str | None = None,
        identity_provenance: str = "unverified",
    ) -> ExecutionRecord:
        record = ExecutionRecord(
            execution_id=execution_id or f"wf-{uuid.uuid4().hex[:12]}",
            workflow_name=workflow_name,
            user_id=user_id,
            params=dict(params or {}),
            durable_job_id=durable_job_id,
            identity_provenance=identity_provenance,
        )
        with self._lock:
            self._records[record.execution_id] = record
            self._gc_locked()
        return record

    def get(self, execution_id: str) -> ExecutionRecord | None:
        with self._lock:
            return self._records.get(execution_id)

    def list(self, *, user_id: str | None = None) -> builtins.list[ExecutionRecord]:
        with self._lock:
            records = list(self._records.values())
        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        records.sort(key=lambda r: r.submitted_at, reverse=True)
        return records

    def cancel(self, execution_id: str) -> ExecutionRecord | None:
        record = self.get(execution_id)
        if record is None:
            return None
        with record._lock:
            if record.status in _TERMINAL:
                return record
            record.cancel_requested = True
            if record.status == "submitted":
                record.status = "cancelled"
                record.finished_at = time.time()
        return record

    def resolve_approval(
        self,
        execution_id: str,
        step_name: str,
        *,
        approved: bool,
        approver: str | None,
        comment: str | None,
    ) -> ExecutionRecord | None:
        """Forward a manual-gate decision to a tracked execution (W5)."""
        record = self.get(execution_id)
        if record is None:
            return None
        record.resolve_gate(
            step_name,
            approved=approved,
            approver=approver,
            comment=comment,
        )
        return record

    # ------------------------------------------------------------------
    # Worker dispatch
    # ------------------------------------------------------------------

    def launch(
        self,
        record: ExecutionRecord,
        runner: Callable[[ExecutionRecord], None],
    ) -> None:
        """Launch ``runner(record)`` in a background daemon thread.

        ``runner`` should respect ``record.cancel_requested`` and is
        responsible for transitioning the record into a terminal state.
        Exceptions raised by ``runner`` are converted into a ``failed``
        terminal state with the traceback stored in ``record.error``.
        """

        def _wrapped() -> None:
            try:
                if record.cancel_requested:
                    record.update(
                        status="cancelled",
                        finished_at=time.time(),
                    )
                    return
                runner(record)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Workflow execution %s crashed", record.execution_id)
                record.update(
                    status="failed",
                    error=f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}",
                    finished_at=time.time(),
                )
            finally:
                with self._lock:
                    self._workers.pop(record.execution_id, None)

        thread = threading.Thread(
            target=_wrapped,
            name=f"workflow-runner-{record.execution_id}",
            daemon=True,
        )
        with self._lock:
            self._workers[record.execution_id] = thread
        thread.start()

    def join(self, execution_id: str, timeout: float | None = None) -> bool:
        """Wait for ``execution_id`` to leave the running state.

        Returns ``True`` if it reached a terminal state within the
        timeout, ``False`` otherwise.
        """
        deadline = None if timeout is None else time.time() + timeout
        while True:
            record = self.get(execution_id)
            if record is None:
                return False
            if record.is_terminal():
                return True
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gc_locked(self) -> None:
        if len(self._records) <= self._max:
            return
        # Drop the oldest terminal records first, then oldest overall.
        terminals = sorted(
            (r for r in self._records.values() if r.is_terminal()),
            key=lambda r: r.finished_at or r.submitted_at,
        )
        to_remove = max(0, len(self._records) - self._max)
        for record in terminals[:to_remove]:
            self._records.pop(record.execution_id, None)


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------

_registry: WorkflowExecutionRegistry | None = None
_registry_lock = threading.Lock()


def get_execution_registry() -> WorkflowExecutionRegistry:
    """Return the process-wide execution registry (W1: durable when available).

    On first build, resolves through ``umrs.workflow.pg_registry``
    ``build_execution_registry`` -- Postgres-backed when the environment
    provides a database, otherwise the classic in-memory registry.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                from ..workflow.pg_registry import build_execution_registry

                _registry = build_execution_registry()
    return _registry


def reset_execution_registry_for_tests() -> None:
    """Drop the singleton so tests can start with an empty registry."""
    global _registry
    with _registry_lock:
        _registry = None


def serialise_record(record: ExecutionRecord) -> str:
    """JSON-encode an execution record (used by ZMQ handlers)."""
    return json.dumps(record.to_dict(), default=str)
