"""UMRS-side WorkflowExecutionPort (ADR-009 real-mode composition, P2L).

The only production adapter of ``WorkflowExecutionPort`` previously lived
client-side (``r4.research.workflow_executor.R4ClientWorkflowExecutor`` over
HTTP). UMRS's own REST loop driver needs the same capability in-process:
this port maps the protocol onto the durable job runtime plus the
in-process workflow execution registry.

Semantics:
  - ``trigger_as`` creates one durable ``WORKFLOW_RUN`` job for the loop's
    verified owner, with the workflow resolved owner-scoped up front (ADR-074
    D-8: a ``user`` workflow only for its owner, ``system`` for anyone) and
    its ``source_path`` in the spec; the scheduler/monitor then routes it to
    ``WorkflowRunBackend`` exactly like a client-triggered run
    (attempt-scoped outputs, commit pointer intact). The owner never comes
    from ``params``: the experiment planner writes those. ``trigger`` (the
    protocol entry, no verified owner) runs only a ``system`` workflow.
  - ``wait`` blocks (the caller — the engine runner thread — is designed
    for blocking) until the execution registry reports a terminal status.
  - ``results`` mirrors the registry record, augmented with ``durable_job_id``
    and attempt artifacts, because the engine requires the durable linkage.
  - ``cancel`` cooperatively cancels both the registry record and the job.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from umrs.jobs.models import JobType
from umrs.knowledge.execution_port import WorkflowExecutionPort
from umrs.shared.workflow_execution_access import VERIFIED_PROJECT_SPEC_KEY

logger = logging.getLogger(__name__)

_TERMINAL = {"succeeded", "failed", "cancelled"}


def _first(*values, default):
    """First truthy value, else ``default`` (keeps result assembly flat)."""
    for value in values:
        if value:
            return value
    return default


def _owner_source_path(workflow_name: str, owner: str, verified: str | None) -> str:
    """The registered source of a workflow *owner* may run (ADR-074 D-8,
    ADR-080; *verified* admits ``project:`` shares), or a RuntimeError by name."""
    from umrs.server.workflow_registry_handler import resolve_workflow_for_caller

    entry = (
        resolve_workflow_for_caller(workflow_name, owner, caller_project=verified) or {}
    )
    source_path = str(entry.get("source_path") or "")
    if not source_path:
        raise RuntimeError(
            f"workflow {workflow_name!r} is not one the loop's owner "
            f"{owner or '(none)'!r} may run: a user workflow runs only for "
            "its owner or those its access list admits, a system workflow "
            "for anyone (ADR-074 D-8, ADR-080)"
        )
    return source_path


class UmrsWorkflowExecutionPort(WorkflowExecutionPort):
    """WorkflowExecutionPort backed by JobStore + execution registry."""

    def __init__(self, job_store, poll_interval: float = 2.0) -> None:
        self._store = job_store
        self._poll = poll_interval
        # execution_id -> job_id for jobs this process submitted (registry
        # records don't carry it until the backend writes them).
        self._jobs: dict[str, str] = {}

    # -- registration --------------------------------------------------------

    def trigger(self, workflow_name: str, params: dict) -> str:
        return self.trigger_as(workflow_name, params, owner="")

    def trigger_as(
        self,
        workflow_name: str,
        params: dict,
        *,
        owner: str,
        owner_project: str | None = None,
    ) -> str:
        """Start *workflow_name* as *owner* (the loop's verified owner; ""
        for an ownerless loop). *owner_project* is the loop's project when
        the loop recorded it as the starter's gateway-verified project
        (``ResearchLoop`` metadata, :func:`umrs.knowledge.loop.verified_loop_project`);
        it admits an ADR-080 ``project:`` share, and only when it is also the
        run's own project. A workflow the owner may not run fails here by
        name, before any job exists that ``wait`` would wait out."""
        params = {k: v for k, v in params.items() if k != "user_id"}
        project_id = str(params.get("project_id") or "research")
        verified = owner_project if owner and owner_project == project_id else None
        source_path = _owner_source_path(workflow_name, owner, verified)
        execution_id = f"exec-{uuid4().hex}"
        spec = {
            "workflow_name": workflow_name,
            "params": params,
            "execution_id": execution_id,
            "source_path": source_path,
        }
        if verified:
            spec[VERIFIED_PROJECT_SPEC_KEY] = verified
        job_id = self._store.create_job(
            project_id=project_id,
            user_id=owner or "research-loop",
            job_type=JobType.WORKFLOW_RUN,
            spec=spec,
            # research iterations are deterministic replays — re-executing a
            # crashed attempt is expected, so retry semantics stay explicit.
            max_attempts=1,
        )
        # create_job leaves the row PLANNED; the scheduler only claims QUEUED
        # jobs (JobService.submit_job enqueues right after creating). Live
        # finding (WP-07 walkthrough): without this every server-driven
        # real-mode loop blocked forever in wait() on a job nobody ran.
        self._store.enqueue_job(job_id)
        self._jobs[execution_id] = job_id
        logger.info(
            "research loop workflow triggered: %s → job %s (exec %s)",
            workflow_name,
            job_id,
            execution_id,
        )
        return execution_id

    # -- status / wait / results ---------------------------------------------

    def status(self, execution_id: str) -> dict:
        record = self._record(execution_id)
        if record is None:
            return {"execution_id": execution_id, "status": "unknown"}
        data = record.to_dict()
        data.setdefault("durable_job_id", self._jobs.get(execution_id, ""))
        return data

    def wait(self, execution_id: str, *, timeout: float | None = None) -> dict:
        deadline = time.monotonic() + (timeout if timeout is not None else 3600.0)
        while time.monotonic() < deadline:
            record = self._record(execution_id)
            if record is not None:
                status = str(getattr(record, "status", "") or "").lower()
                if status in _TERMINAL:
                    data = record.to_dict()
                    data["durable_job_id"] = self._jobs.get(execution_id, "")
                    return data
            time.sleep(self._poll)
        return {
            "execution_id": execution_id,
            "status": "timeout",
            "error": f"workflow wait exceeded {timeout or 3600.0}s",
            "durable_job_id": self._jobs.get(execution_id, ""),
        }

    def results(self, execution_id: str) -> dict:
        record = self._record(execution_id)
        if record is None:
            return {"execution_id": execution_id, "status": "unknown"}
        data = record.to_dict()
        results = _first(data.get("results"), default={})
        return {
            "status": data.get("status"),
            "outputs": _first(results.get("outputs"), data.get("outputs"), default={}),
            "metrics": _first(results.get("metrics"), default={}),
            "artifacts": _first(
                data.get("artifacts"), results.get("artifacts"), default=[]
            ),
            "error": _first(data.get("error"), results.get("error"), default=""),
            "durable_job_id": self._jobs.get(execution_id, ""),
        }

    def cancel(self, execution_id: str) -> dict:
        from umrs.jobs.workflow_run import cancel_job_and_run

        job_id = self._jobs.get(execution_id, "")
        if not job_id:
            return {"status": "unknown", "detail": "job not registered in this process"}
        try:
            job, _attempts = self._store.get_job_with_attempts(job_id)
            # The durable job first, then the run (also when the job had
            # ended; never after another store error: peer review F2).
            cancel_job_and_run(self._store, job)
        except Exception as exc:  # noqa: BLE001 — cancel is best-effort
            return {"status": "error", "detail": str(exc)}
        return {
            "status": "cancelled",
            "execution_id": execution_id,
            "durable_job_id": job_id,
        }

    # -- internals -------------------------------------------------------------

    def _record(self, execution_id: str):
        """Registry record lookup (None when not yet tracked this process)."""
        try:
            from umrs.server.workflow_execution_registry import get_execution_registry
        except Exception:  # noqa: BLE001
            return None
        return get_execution_registry().get(execution_id)


__all__ = ["UmrsWorkflowExecutionPort"]
