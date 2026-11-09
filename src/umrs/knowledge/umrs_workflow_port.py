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


def _owner_entry(workflow_name: str, owner: str, verified: str | None) -> dict:
    """The registered version of a workflow *owner* may run (ADR-074 D-8,
    ADR-080; *verified* admits ``project:`` shares; ``name@X`` names a
    version, else the default, ADR-113 D-3), or a RuntimeError by name."""
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
    return entry


def _refuse_unplanned_run(store, workflow_name: str, project_id: str) -> None:
    """ADR-105 D-4: a project with ``require_plan`` starts runs only through
    plan -> commit, and a loop iteration commits no plan. RuntimeError by
    name (this port's convention), before any job exists; an unreadable
    policy refuses too (fail closed). A store without a pool has no policy."""
    from umrs.workflow_plans import (
        PlanPolicyUnavailable,
        plan_required_message,
        read_project_plan_policy,
    )

    try:
        policy = read_project_plan_policy(getattr(store, "db_pool", None), project_id)
    except PlanPolicyUnavailable as exc:
        raise RuntimeError(str(exc)) from exc
    if policy.require_plan:
        raise RuntimeError(
            "research loop run refused: "
            + plan_required_message(project_id, workflow_name)
        )


def _grant_target(entry: dict) -> tuple[dict, str]:
    import json
    import os

    source = str(entry.get("source_path") or "")
    with open(os.path.join(source, "workflow.json"), encoding="utf-8") as handle:
        return json.load(handle), source


def _user_account_identity(
    entry: dict, owner: str, project: str | None, *, any_mode: bool = False
):
    """The run identity, or None without HPC work or on a prior refusal."""
    from umrs.orchestrator.hpc_identity import HpcAccessRefused
    from umrs.server.hpc_user_work import is_user_account, workflow_user_work_identity

    try:
        workflow_json, source = _grant_target(entry)
        identity = workflow_user_work_identity(
            workflow_json, source, user_id=owner, project_id=project
        )
    except (OSError, ValueError, HpcAccessRefused):
        return None
    return identity if any_mode or is_user_account(identity) else None


def _grant_run(
    entry: dict, owner: str, project: str | None, params: dict, grant_id: str | None
):
    """ADR-112 wave C (K12): a loop iteration on a site that runs the
    owner's work under their own HPC account runs under a standing grant:
    the one the loop recorded, else the one that covers the run. None: the
    run does not use the person's account. A recorded grant that ended
    stops the loop (:class:`~umrs.knowledge.loop.GrantEnded`)."""
    from types import SimpleNamespace

    from umrs.server.hpc_user_work import is_user_account
    from umrs.server.run_intent_admission import admit_run

    identity = _user_account_identity(entry, owner, project, any_mode=True)
    if identity is None:
        return None
    workflow_json, source = _grant_target(entry)
    run_project = project or (
        "" if is_user_account(identity) else str(params.get("project_id") or "")
    )
    run = {
        "workflow": entry.get("name"),
        "version": entry.get("version") or "",
        "project": run_project,
        "principal": owner,
    }
    target = SimpleNamespace(
        entry=entry, source_path=source, workflow_json=workflow_json
    )
    admission, error = admit_run(
        identity, {"grant_id": grant_id} if grant_id else {}, params, run, None, target
    )
    if error:
        raise _grant_refusal(error, grant_id)
    return admission.grant


def _grant_refusal(error: str, grant_id: str | None) -> Exception:
    """A recorded grant that ended is ``GrantEnded``; anything else is a
    RuntimeError by name (this port's convention)."""
    import json

    from umrs.knowledge.loop import GrantEnded

    body = json.loads(error)
    ended = body.get("code") == "hpcaccess.grant_not_found" or body.get("reason") in (
        "expired",
        "cancelled",
    )
    if grant_id and ended:
        return GrantEnded(grant_id, str(body.get("message") or ""))
    return RuntimeError(
        f"research loop run refused: {body.get('message')} ({body.get('code')})"
    )


def _run_spec(
    entry: dict, workflow_name: str, params: dict, verified: str | None, gated
) -> dict:
    """The iteration's durable job spec; a grant run (wave C) keeps its run
    spec's params and its ``run_id`` as the execution id."""
    spec = {
        "workflow_name": entry.get("name") or workflow_name,
        "params": params,
        "execution_id": f"exec-{uuid4().hex}",
        # ADR-113 D-4: the iteration pins the version it resolved.
        "source_path": entry["source_path"],
        "workflow_version": entry.get("version"),
        "source_sha256": entry.get("source_sha256"),
    }
    if verified:
        spec[VERIFIED_PROJECT_SPEC_KEY] = verified
    if gated is not None:
        spec.update(
            params=dict(gated.body["params"]),
            execution_id=gated.run_id,
            intent=gated.spec(),
        )
    return spec


class UmrsWorkflowExecutionPort(WorkflowExecutionPort):
    """WorkflowExecutionPort backed by JobStore + execution registry."""

    def __init__(self, job_store, poll_interval: float = 2.0) -> None:
        self._store = job_store
        self._poll = poll_interval
        # execution_id -> job_id for jobs this process submitted (registry
        # records don't carry it until the backend writes them).
        self._jobs: dict[str, str] = {}
        # ADR-112 wave C (K12): execution_id -> the grant its run uses.
        self._grants: dict[str, str] = {}

    def grant_for(self, execution_id: str) -> str:
        """The standing grant *execution_id* runs under ("" for none)."""
        return self._grants.get(execution_id, "")

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
        grant_id: str | None = None,
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
        entry = _owner_entry(workflow_name, owner, verified)
        _refuse_unplanned_run(self._store, workflow_name, project_id)
        gated = _grant_run(entry, owner, verified, params, grant_id)
        spec = _run_spec(entry, workflow_name, params, verified, gated)
        execution_id = spec["execution_id"]
        if gated is not None:
            self._grants[execution_id] = gated.grant_id
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
