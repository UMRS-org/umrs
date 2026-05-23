"""Workflow Registry REST API.

Provides:
- GET /umrs/workflows - List registered workflows with optional filters
- POST /umrs/workflows/{workflow_name}/trigger - Accept workflow execution

This router is intentionally thin. It bridges the HTTP sidecar to the
existing workflow registry command handlers used by the ZMQ server.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from ..core.TaskCmdStatus import TaskCmd
from ..jobs.models import JobStatus, JobType
from ..jobs.store import InvalidTransitionError, JobNotFoundError, JobStore
from ..server.file_handling import handle_upload_workflow
from ..server.job_tracking_handler import DurableJobCommandBackend
from ..server.workflow_registry_handler import handle_workflow_command, register_workflow
from .jobs import get_optional_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/workflows", tags=["workflows"])


class WorkflowTriggerRequest(BaseModel):
    """Workflow trigger request body."""

    user_id: str = Field(..., min_length=1, description="Workflow owner or caller")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow runtime parameters",
    )


class WorkflowUploadResponse(BaseModel):
    """Workflow upload response body."""

    status: str = Field(..., description="Operation result")
    workflow_name: str = Field(..., description="Registered workflow name")
    accepted: bool = Field(..., description="Whether the upload was accepted")
    source_path: str = Field(..., description="Extracted workflow directory")
    message: str = Field(..., description="Human-readable result message")


def _parse_command_response(raw: str, *, action: str) -> dict[str, Any]:
    """Parse a workflow command response and map JSON errors to HTTP errors."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON from workflow command handler during %s", action)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Invalid workflow handler response during {action}",
        ) from exc

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unexpected workflow handler response during {action}",
        )

    if data.get("status") != "error":
        return data

    message = str(data.get("message", "Workflow operation failed"))
    lower_message = message.lower()

    if "not found" in lower_message or "missing at" in lower_message:
        code = status.HTTP_404_NOT_FOUND
    elif "invalid json" in lower_message or "required" in lower_message:
        code = status.HTTP_400_BAD_REQUEST
    else:
        code = status.HTTP_500_INTERNAL_SERVER_ERROR

    raise HTTPException(status_code=code, detail=message)


def _raise_upload_error(message: str) -> None:
    """Map upload failures to HTTP errors."""
    lower_message = message.lower()
    if "too large" in lower_message:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    elif (
        "not a valid zip" in lower_message
        or "workflow.json" in lower_message
        or "invalid workflow" in lower_message
        or "unsafe path" in lower_message
        or "escapes extracted directory" in lower_message
        or "missing after extraction" in lower_message
    ):
        code = status.HTTP_400_BAD_REQUEST
    else:
        code = status.HTTP_500_INTERNAL_SERVER_ERROR

    raise HTTPException(status_code=code, detail=message)


def _strict_auth_mode() -> bool:
    return os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"


def _header_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _verified_workflow_user(user_id: str | None, x_user_id: str | None) -> str:
    verified_user_id = _header_str(x_user_id)
    if verified_user_id:
        return verified_user_id
    if _strict_auth_mode():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified user identity",
        )
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id is required")
    return user_id.strip()


def _verified_workflow_params(
    params: dict[str, Any],
    x_verified_project_id: str | None,
) -> dict[str, Any]:
    verified_project_id = _header_str(x_verified_project_id)
    resolved_params = dict(params)
    if verified_project_id:
        resolved_params["project_id"] = verified_project_id
        return resolved_params
    if _strict_auth_mode():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified project identity",
        )
    return resolved_params


def _require_verified_workflow_identity(
    x_user_id: str | None,
    x_verified_project_id: str | None,
) -> None:
    if _header_str(x_user_id) and _header_str(x_verified_project_id):
        return
    if _strict_auth_mode():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified workflow identity",
        )


def _workflow_job_status(job_status: JobStatus) -> str:
    return {
        JobStatus.PLANNED: "submitted",
        JobStatus.QUEUED: "submitted",
        JobStatus.RUNNING: "running",
        JobStatus.COMPLETED: "succeeded",
        JobStatus.FAILED: "failed",
        JobStatus.CANCELLED: "cancelled",
    }[job_status]


def _workflow_job_payload(job) -> dict[str, Any]:
    return job.to_dict() if hasattr(job, "to_dict") else dict(job)


def _find_workflow_job_by_execution_id(
    store: JobStore | None,
    execution_id: str,
):
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


def _get_job_for_record(store: JobStore | None, record) -> Any:
    if store is None or not record.durable_job_id:
        return None
    try:
        return store.get_job(record.durable_job_id)
    except JobNotFoundError:
        return None


def _execution_payload(record, job=None) -> dict[str, Any]:
    execution = record.to_dict()
    if job is not None:
        execution["durable_job_id"] = job.job_id
        execution["durable_job"] = _workflow_job_payload(job)
        durable_status = _workflow_job_status(job.status)
        if job.status in {
            JobStatus.PLANNED,
            JobStatus.QUEUED,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        } or execution.get("status") == "submitted":
            execution["status"] = durable_status
    return execution


def _execution_payload_from_job(job) -> dict[str, Any]:
    spec = job.spec
    return {
        "execution_id": spec.get("execution_id"),
        "workflow_name": spec.get("workflow_name"),
        "user_id": job.user_id,
        "status": _workflow_job_status(job.status),
        "submitted_at": job.created_at.timestamp() if job.created_at else None,
        "started_at": None,
        "finished_at": (
            job.last_state_change_at.timestamp()
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            and job.last_state_change_at
            else None
        ),
        "error": None,
        "progress": 1.0 if job.status == JobStatus.COMPLETED else 0.0,
        "params": spec.get("params", {}),
        "steps": [],
        "artifacts": [],
        "durable_job_id": job.job_id,
        "durable_job": _workflow_job_payload(job),
        "cancel_requested": job.status == JobStatus.CANCELLED,
    }


@router.get("")
async def list_workflows(
    scope: str | None = Query(None, description="Optional workflow scope filter"),
    name: str | None = Query(None, description="Optional workflow-name filter"),
    user_id: str | None = Query(None, description="Optional owner filter"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
):
    _require_verified_workflow_identity(x_user_id, x_verified_project_id)
    resolved_user_id = _header_str(x_user_id) or user_id
    payload = {
        key: value
        for key, value in (("scope", scope), ("name", name), ("user_id", resolved_user_id))
        if value is not None
    }
    raw = handle_workflow_command(
        TaskCmd.LIST_WORKFLOWS,
        resolved_user_id or "_",
        json.dumps(payload) if payload else "",
    )
    return _parse_command_response(raw, action="list")


@router.post(
    "/upload",
    response_model=WorkflowUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_workflow(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    workflow_name: str | None = Form(None),
    scope: str = Form("user"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
):
    _require_verified_workflow_identity(x_user_id, x_verified_project_id)
    resolved_user_id = _verified_workflow_user(user_id, x_user_id)

    # Governance quota check
    project_id = x_verified_project_id or ""
    try:
        from .governance import check_quota
        check_quota(project_id, "workflows")
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise
        pass  # governance not active

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename required")

    resolved_workflow_name = workflow_name or Path(file.filename).stem
    if not resolved_workflow_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="workflow_name is required")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    ok, dest, err = handle_upload_workflow(resolved_user_id, temp_path, resolved_workflow_name)
    if not ok:
        _raise_upload_error(err or "Workflow upload failed")

    workflow_json_path = os.path.join(dest, "workflow.json")
    try:
        with open(workflow_json_path, encoding="utf-8") as handle:
            workflow_json = json.load(handle)
        version = str(workflow_json.get("version", "0.1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load extracted workflow.json: {exc}",
        ) from exc

    result = register_workflow(
        workflow_name=resolved_workflow_name,
        version=version,
        user_id=resolved_user_id,
        source_path=dest,
        workflow_json=workflow_json,
        scope=scope,
        description=workflow_json.get("description", ""),
    )
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(result.get("message", "Workflow registration failed")),
        )

    # Governance tracking
    from .governance import record_resource_created
    record_resource_created(
        "workflows", resolved_workflow_name, resolved_user_id,
        project_id=project_id, tier=scope,
    )

    return WorkflowUploadResponse(
        status="ok",
        workflow_name=resolved_workflow_name,
        accepted=True,
        source_path=dest,
        message=f"Workflow '{resolved_workflow_name}' uploaded and registered",
    )


@router.post("/{workflow_name}/trigger")
async def trigger_workflow(
    workflow_name: str,
    request: WorkflowTriggerRequest,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
):
    user_id = _verified_workflow_user(request.user_id, x_user_id)
    params = _verified_workflow_params(request.params, x_verified_project_id)
    store = get_optional_job_store()
    job_backend = DurableJobCommandBackend(store) if store is not None else None
    raw = handle_workflow_command(
        TaskCmd.RUN_WORKFLOW,
        user_id,
        json.dumps(
            {
                "workflow_name": workflow_name,
                "params": params,
            }
        ),
        job_backend=job_backend,
    )
    return _parse_command_response(raw, action="trigger")


# ---------------------------------------------------------------------------
# Execution status / results / cancel
# ---------------------------------------------------------------------------


@router.get("/executions/{execution_id}")
async def get_execution_status(
    execution_id: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
):
    """Return workflow execution status, backed by durable job state when present."""
    from ..server.workflow_execution_registry import get_execution_registry

    _require_verified_workflow_identity(x_user_id, x_verified_project_id)
    store = get_optional_job_store()
    registry = get_execution_registry()
    record = registry.get(execution_id)
    if record is not None:
        job = _get_job_for_record(store, record)
        return {"status": "ok", "execution": _execution_payload(record, job)}

    job = _find_workflow_job_by_execution_id(store, execution_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )
    return {"status": "ok", "execution": _execution_payload_from_job(job)}


@router.get("/executions/{execution_id}/results")
async def get_execution_results(
    execution_id: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
):
    """Return the structured outputs of a finished execution.

    For unfinished executions, returns the partial results gathered so
    far (so clients can render progressive UIs without retrying).
    """
    from ..server.workflow_execution_registry import get_execution_registry
    from ..server.workflow_runner import list_artifacts

    _require_verified_workflow_identity(x_user_id, x_verified_project_id)
    store = get_optional_job_store()
    registry = get_execution_registry()
    record = registry.get(execution_id)
    if record is None:
        job = _find_workflow_job_by_execution_id(store, execution_id)
        if job is not None:
            execution = _execution_payload_from_job(job)
            return {
                "status": "ok",
                "execution_id": execution_id,
                "execution_status": execution["status"],
                "progress": execution["progress"],
                "error": execution["error"],
                "results": {"steps": {}, "final": None},
                "artifacts": [],
                "durable_job_id": job.job_id,
                "durable_job": execution["durable_job"],
            }
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )

    job = _get_job_for_record(store, record)
    execution = _execution_payload(record, job)

    artifacts = [
        {"key": key, "size_bytes": size}
        for key, size in list_artifacts(record)
    ]
    return {
        "status": "ok",
        "execution_id": record.execution_id,
        "execution_status": execution["status"],
        "progress": execution["progress"],
        "error": execution["error"],
        "results": {
            "steps": record.results.get("steps", {}),
            "final": record.results.get("final"),
        },
        "artifacts": artifacts,
        "durable_job_id": execution.get("durable_job_id"),
        "durable_job": execution.get("durable_job"),
    }


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(
    execution_id: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
):
    """Request cancellation of a workflow execution."""
    from ..server.workflow_execution_registry import get_execution_registry

    _require_verified_workflow_identity(x_user_id, x_verified_project_id)
    store = get_optional_job_store()
    registry = get_execution_registry()
    record = registry.cancel(execution_id)
    job = None
    if record is not None:
        job = _get_job_for_record(store, record)
    if job is None:
        job = _find_workflow_job_by_execution_id(store, execution_id)
    if job is not None and store is not None:
        with contextlib.suppress(InvalidTransitionError):
            store.cancel_job(job.job_id, reason="workflow_execution_cancelled")

    if record is None and job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )
    if record is not None:
        return {"status": "ok", "execution": _execution_payload(record, job)}
    return {"status": "ok", "execution": _execution_payload_from_job(job)}
