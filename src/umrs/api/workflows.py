"""Workflow Registry REST API.

Provides:
- GET /umrs/workflows - List registered workflows with optional filters
- POST /umrs/workflows/{workflow_name}/trigger - Accept workflow execution

This router is intentionally thin. It bridges the HTTP sidecar to the
existing workflow registry command handlers used by the ZMQ server.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from ..core.TaskCmdStatus import TaskCmd
from ..server.file_handling import handle_upload_workflow
from ..server.workflow_registry_handler import handle_workflow_command
from ..server.workflow_registry_handler import register_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/workflows", tags=["workflows"])


class WorkflowTriggerRequest(BaseModel):
    """Workflow trigger request body."""

    user_id: str = Field(..., min_length=1, description="Workflow owner or caller")
    params: Dict[str, Any] = Field(
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


def _parse_command_response(raw: str, *, action: str) -> Dict[str, Any]:
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


@router.get("")
async def list_workflows(
    scope: Optional[str] = Query(None, description="Optional workflow scope filter"),
    name: Optional[str] = Query(None, description="Optional workflow-name filter"),
    user_id: Optional[str] = Query(None, description="Optional owner filter"),
):
    payload = {
        key: value
        for key, value in (("scope", scope), ("name", name), ("user_id", user_id))
        if value is not None
    }
    raw = handle_workflow_command(
        TaskCmd.LIST_WORKFLOWS,
        user_id or "_",
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
    workflow_name: Optional[str] = Form(None),
    scope: str = Form("user"),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename required")

    resolved_workflow_name = workflow_name or Path(file.filename).stem
    if not resolved_workflow_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="workflow_name is required")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    ok, dest, err = handle_upload_workflow(user_id, temp_path, resolved_workflow_name)
    if not ok:
        _raise_upload_error(err or "Workflow upload failed")

    workflow_json_path = os.path.join(dest, "workflow.json")
    try:
        with open(workflow_json_path, "r", encoding="utf-8") as handle:
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
        user_id=user_id,
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
):
    raw = handle_workflow_command(
        TaskCmd.RUN_WORKFLOW,
        request.user_id,
        json.dumps(
            {
                "workflow_name": workflow_name,
                "params": request.params,
            }
        ),
    )
    return _parse_command_response(raw, action="trigger")


# ---------------------------------------------------------------------------
# Execution status / results / cancel
# ---------------------------------------------------------------------------


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Return the in-memory status of a workflow execution."""
    from ..server.workflow_execution_registry import get_execution_registry

    registry = get_execution_registry()
    record = registry.get(execution_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )
    return {"status": "ok", "execution": record.to_dict()}


@router.get("/executions/{execution_id}/results")
async def get_execution_results(execution_id: str):
    """Return the structured outputs of a finished execution.

    For unfinished executions, returns the partial results gathered so
    far (so clients can render progressive UIs without retrying).
    """
    from ..server.workflow_execution_registry import get_execution_registry
    from ..server.workflow_runner import list_artifacts

    registry = get_execution_registry()
    record = registry.get(execution_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )

    artifacts = [
        {"key": key, "size_bytes": size}
        for key, size in list_artifacts(record)
    ]
    return {
        "status": "ok",
        "execution_id": record.execution_id,
        "execution_status": record.status,
        "progress": record.progress,
        "error": record.error,
        "results": {
            "steps": record.results.get("steps", {}),
            "final": record.results.get("final"),
        },
        "artifacts": artifacts,
    }


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Request cancellation of a workflow execution."""
    from ..server.workflow_execution_registry import get_execution_registry

    registry = get_execution_registry()
    record = registry.cancel(execution_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )
    return {"status": "ok", "execution": record.to_dict()}