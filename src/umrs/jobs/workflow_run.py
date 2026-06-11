"""Durable workflow_run job adapter hooks."""

from __future__ import annotations

from collections.abc import Callable

from .models import Job, JobAttempt
from .monitor import AttemptCheckResult
from .scheduler import OrchestratorBackend


class WorkflowRunBackend(OrchestratorBackend):
    """Adapter hook for routing workflow_run jobs through JobService."""

    def __init__(self, submitter: Callable[[Job, JobAttempt], str] | None = None):
        self._submitter = submitter

    @property
    def backend_name(self) -> str:
        return "workflow-run"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        if self._submitter is not None:
            return self._submitter(job, attempt)
        execution_id = job.spec.get("execution_id", attempt.attempt_id)
        workflow_name = job.spec.get("workflow_name")
        if not isinstance(workflow_name, str) or not workflow_name.strip():
            raise ValueError("workflow_run spec.workflow_name is required")

        from umrs.server.workflow_registry_handler import _launch_workflow_from_catalog

        _launch_workflow_from_catalog(
            workflow_name=workflow_name,
            client_id=job.user_id,
            params=job.spec.get("params", {}),
            execution_id=execution_id,
            durable_job_id=job.job_id,
            source_path=job.spec.get("source_path"),
        )
        return f"workflow_run|execution_id={execution_id}"


class WorkflowRunStatusChecker:
    """Report workflow_run attempt status from the execution registry."""

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        execution_id = _execution_id_from_attempt_or_job(attempt, job)
        from umrs.server.workflow_execution_registry import get_execution_registry

        record = get_execution_registry().get(execution_id)
        if record is None:
            return AttemptCheckResult.running(
                detail=f"waiting for workflow execution {execution_id}"
            )

        data = record.to_dict()
        metrics = {
            "execution_id": execution_id,
            "progress": data.get("progress", 0.0),
            "step_count": len(data.get("steps", [])),
        }
        status = data.get("status")
        if status == "succeeded":
            return AttemptCheckResult.succeeded(metrics=metrics, commit=True)
        if status == "failed":
            return AttemptCheckResult.failed(data.get("error") or "workflow failed")
        if status == "cancelled":
            return AttemptCheckResult.failed("workflow cancelled")
        return AttemptCheckResult.running(detail=f"workflow {status or 'running'}")


def _execution_id_from_attempt_or_job(attempt: JobAttempt, job: Job) -> str:
    if (
        isinstance(job.spec.get("execution_id"), str)
        and job.spec["execution_id"].strip()
    ):
        return job.spec["execution_id"].strip()
    if attempt.backend_ref and "execution_id=" in attempt.backend_ref:
        return attempt.backend_ref.rsplit("execution_id=", 1)[1]
    return attempt.attempt_id


def cancel_workflow_execution_for_job(job: Job) -> None:
    """Cooperatively cancel the in-process workflow record for a durable job."""
    execution_id = job.spec.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id.strip():
        return

    from umrs.server.workflow_execution_registry import get_execution_registry

    get_execution_registry().cancel(execution_id.strip())
