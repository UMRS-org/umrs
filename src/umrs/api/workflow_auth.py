"""Workflow execution authorization helpers.

Enforces (user_id, project_id) ownership for workflow executions across
HTTP and ZMQ surfaces.  Preserves 404 for absent executions and returns
structured 403 for existing but inaccessible executions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..jobs.models import Job, JobStatus, JobType
from ..jobs.store import JobStore
from ..server.workflow_execution_registry import ExecutionRecord
from .jobs import CallerIdentity
from .typed_errors import PlatformError, raise_typed

# Canonical workflow lifecycle statuses (minimal contract skeleton).
WORKFLOW_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
WORKFLOW_ACTIVE_STATUSES = frozenset({"submitted", "running"})


def strict_auth_mode() -> bool:
    return os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() == "strict"


@dataclass
class ZmqWorkflowCaller:
    """Caller identity for direct ZMQ workflow commands."""

    user_id: str
    project_id: str | None
    is_admin: bool = False
    project_provenance: str = "unverified"


def authorization_denied_detail(
    *,
    resource: str = "workflow_execution",
    remediation: str | None = None,
) -> dict[str, str]:
    return {
        "error": "authorization_denied",
        "code": "workflow_access_denied",
        "message": f"Caller cannot access this {resource}",
        "remediation": remediation
        or "Verify Gateway-verified project context and execution ownership",
    }


def zmq_access_denied_response(
    *,
    resource: str = "workflow_execution",
    remediation: str | None = None,
) -> str:
    import json

    detail = authorization_denied_detail(resource=resource, remediation=remediation)
    return json.dumps(
        {
            "status": "error",
            "code": detail["code"],
            "message": detail["message"],
            "remediation": detail["remediation"],
        }
    )


def execution_project_id(
    *,
    record: ExecutionRecord | None = None,
    job: Job | None = None,
    params: dict[str, Any] | None = None,
) -> str | None:
    """Resolve project scope from durable job, execution record, or params."""
    if job is not None and job.project_id:
        return job.project_id
    if record is not None:
        raw = (
            record.params.get("project_id") if isinstance(record.params, dict) else None
        )
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if isinstance(params, dict):
        raw = params.get("project_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def execution_identity_provenance(
    *,
    record: ExecutionRecord | None = None,
    job: Job | None = None,
) -> str:
    if job is not None:
        return "gateway_verified"
    if record is not None:
        provenance = getattr(record, "identity_provenance", None)
        if isinstance(provenance, str) and provenance:
            return provenance
    return "unverified"


def execution_owner_user_id(
    *,
    record: ExecutionRecord | None = None,
    job: Job | None = None,
) -> str | None:
    if record is not None:
        return record.user_id
    if job is not None:
        return job.user_id
    return None


def can_access_execution(
    caller_user_id: str,
    caller_project_id: str | None,
    *,
    is_admin: bool = False,
    caller_project_provenance: str = "gateway_verified",
    record: ExecutionRecord | None = None,
    job: Job | None = None,
) -> bool:
    """Return True when caller may read or mutate the execution."""
    if is_admin:
        return True

    owner_user_id = execution_owner_user_id(record=record, job=job)
    if owner_user_id is None:
        return False
    if owner_user_id != caller_user_id:
        return False

    project_id = execution_project_id(record=record, job=job)
    provenance = execution_identity_provenance(record=record, job=job)

    if provenance == "unverified":
        return False

    if strict_auth_mode():
        if provenance != "gateway_verified":
            return False
        if caller_project_provenance != "gateway_verified":
            return False

    if caller_project_id is None or project_id is None:
        return not strict_auth_mode()

    return project_id == caller_project_id


def caller_from_identity(identity: CallerIdentity) -> tuple[str, str | None, bool, str]:
    return (
        identity.user_id,
        identity.project_id,
        identity.is_admin,
        "gateway_verified",
    )


def zmq_caller_from_payload(
    client_id: str, payload: dict[str, Any] | None
) -> ZmqWorkflowCaller:
    """The caller of a ZMQ workflow-status command.

    ``_identity_verified`` reaches this payload only from the server's own
    verified path: the command registry drops it from every wire payload and
    ``handle_workflow_command`` drops it unless ``trusted_identity`` is set
    (ADR-097 D-7, 2026-10-22; before that a ZMQ client could set it).
    """
    payload = payload if isinstance(payload, dict) else {}
    project_id = payload.get("project_id")
    if not isinstance(project_id, str) or not project_id.strip():
        params = payload.get("params")
        if isinstance(params, dict):
            raw = params.get("project_id")
            if isinstance(raw, str) and raw.strip():
                project_id = raw.strip()
    else:
        project_id = project_id.strip()

    provenance = (
        "gateway_verified"
        if payload.get("_identity_verified") and project_id
        else ("mvp_claim" if project_id and not strict_auth_mode() else "unverified")
    )
    return ZmqWorkflowCaller(
        user_id=client_id,
        project_id=project_id,
        project_provenance=provenance,
    )


def enforce_http_execution_access(
    identity: CallerIdentity,
    *,
    record: ExecutionRecord | None,
    job: Job | None,
    execution_id: str,
) -> None:
    """Raise 404/403 when the caller cannot access an execution."""
    if record is None and job is None:
        raise_typed(
            PlatformError.workflowauth(
                "not_found", f"Execution {execution_id} not found"
            ),
        )

    user_id, project_id, is_admin, provenance = caller_from_identity(identity)
    if can_access_execution(
        user_id,
        project_id,
        is_admin=is_admin,
        caller_project_provenance=provenance,
        record=record,
        job=job,
    ):
        return

    raise_typed(
        PlatformError.workflowauth("forbidden", authorization_denied_detail()),
    )


def check_zmq_execution_access(
    caller: ZmqWorkflowCaller,
    *,
    record: ExecutionRecord | None,
    job: Job | None,
) -> str | None:
    """Return a JSON error string when access is denied, else None."""
    if record is None and job is None:
        return None

    if can_access_execution(
        caller.user_id,
        caller.project_id,
        is_admin=caller.is_admin,
        caller_project_provenance=caller.project_provenance,
        record=record,
        job=job,
    ):
        return None

    return zmq_access_denied_response()


def find_workflow_job_by_execution_id(
    store: JobStore | None,
    execution_id: str,
) -> Job | None:
    if store is None:
        return None
    try:
        jobs = store.list_jobs(job_type=JobType.WORKFLOW_RUN, limit=1000)
    except TypeError:
        jobs = store.list_jobs(job_type=JobType.WORKFLOW_RUN)
    for job in jobs:
        if job.spec.get("execution_id") == execution_id:
            return job
    return None


def resolve_execution_context(
    execution_id: str,
    *,
    store: JobStore | None = None,
    registry: Any | None = None,
) -> tuple[ExecutionRecord | None, Job | None]:
    if registry is None:
        from ..server.workflow_execution_registry import get_execution_registry

        registry = get_execution_registry()
    record = registry.get(execution_id)
    job = find_workflow_job_by_execution_id(store, execution_id)
    return record, job


def workflow_job_status(job_status: JobStatus) -> str:
    return {
        JobStatus.PLANNED: "submitted",
        JobStatus.QUEUED: "submitted",
        JobStatus.RUNNING: "running",
        JobStatus.COMPLETED: "succeeded",
        JobStatus.FAILED: "failed",
        JobStatus.CANCELLED: "cancelled",
    }[job_status]


def execution_payload_from_job(job: Job) -> dict[str, Any]:
    spec = job.spec
    payload = job.to_dict() if hasattr(job, "to_dict") else dict(job)
    return {
        "execution_id": spec.get("execution_id"),
        "workflow_name": spec.get("workflow_name"),
        "user_id": job.user_id,
        "status": workflow_job_status(job.status),
        "submitted_at": job.created_at.timestamp() if job.created_at else None,
        "started_at": None,
        "finished_at": (
            job.last_state_change_at.timestamp()
            if job.status
            in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            and job.last_state_change_at
            else None
        ),
        "error": None,
        "progress": 1.0 if job.status == JobStatus.COMPLETED else 0.0,
        "params": spec.get("params", {}),
        "steps": [],
        "artifacts": [],
        "durable_job_id": job.job_id,
        "durable_job": payload,
        "cancel_requested": job.status == JobStatus.CANCELLED,
    }
