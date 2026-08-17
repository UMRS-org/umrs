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

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from ..core.zmq_correlation import dumps_with_correlation_id
from pydantic import BaseModel, Field, model_validator

from ..core.TaskCmdStatus import TaskCmd
from ..jobs.models import JobStatus
from ..jobs.store import InvalidTransitionError, JobNotFoundError, JobStore
from ..server.file_handling import handle_upload_workflow
from ..server.job_tracking_handler import DurableJobCommandBackend
from ..server.workflow_registry_handler import (
    _resolve_workflow_catalog_entry,
    handle_workflow_command,
    register_workflow,
)
from .jobs import CallerIdentity, get_caller_identity, get_optional_job_store
from .workflow_auth import (
    enforce_http_execution_access,
    execution_payload_from_job,
    find_workflow_job_by_execution_id,
    resolve_execution_context,
    strict_auth_mode,
    workflow_job_status,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/workflows", tags=["workflows"])


class WorkflowTriggerRequest(BaseModel):
    """Workflow trigger request body (UMRS + public Gateway contract)."""

    user_id: str | None = Field(
        default=None,
        min_length=1,
        description="Workflow owner or caller (MVP mode only)",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy UMRS runtime parameters",
    )
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Public contract runtime parameters",
    )
    input_files: list[str] | None = Field(
        default=None,
        description="Optional input file keys",
    )
    project_id: str | None = Field(
        default=None,
        description="Project context (MVP mode only; strict uses verified headers)",
    )

    def resolved_params(self) -> dict[str, Any]:
        merged = dict(self.params)
        if isinstance(self.parameters, dict):
            merged.update(self.parameters)
        if self.input_files:
            merged["input_files"] = list(self.input_files)
        if self.project_id:
            merged.setdefault("project_id", self.project_id)
        return merged

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "parameters" not in data and isinstance(data.get("params"), dict):
            return data
        return data


class WorkflowUploadResponse(BaseModel):
    """Workflow upload response body."""

    status: str = Field(..., description="Operation result")
    workflow_name: str = Field(..., description="Registered workflow name")
    accepted: bool = Field(..., description="Whether the upload was accepted")
    source_path: str = Field(..., description="Extracted workflow directory")
    message: str = Field(..., description="Human-readable result message")


def _schema_from_item_list(items: list[Any]) -> dict[str, Any]:
    """Build a JSON-schema object from an item-list parameter declaration."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for item in items:
        entry = _schema_entry_from_item(item)
        if entry is None:
            continue
        name, spec, is_required = entry
        properties[name] = spec
        if is_required:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _schema_entry_from_item(item: Any) -> tuple[str, dict[str, Any], bool] | None:
    """Normalize one parameter item into (name, spec, required) or None."""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None
    spec = {
        key: value for key, value in item.items() if key not in {"name", "required"}
    }
    return name, spec, bool(item.get("required"))


def _workflow_parameter_schema(workflow_json: dict[str, Any]) -> dict[str, Any]:
    for key in ("parameter_schema", "parameters_schema", "parameters"):
        candidate = workflow_json.get(key)
        if isinstance(candidate, dict) and candidate.get("type"):
            return candidate
        if isinstance(candidate, list):
            return _schema_from_item_list(candidate)
    return {"type": "object", "properties": {}}


def _read_entry_workflow_json(entry: dict[str, Any]) -> dict[str, Any]:
    """Read a catalog entry's workflow.json ({} when unavailable)."""
    source_path = entry.get("source_path", "")
    wf_json_path = os.path.join(source_path, "workflow.json")
    if source_path and os.path.isfile(wf_json_path):
        with open(wf_json_path, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _describe_workflow_entry(entry: dict[str, Any]) -> dict[str, Any]:
    workflow_json = _read_entry_workflow_json(entry)
    return {
        "status": "ok",
        "contract_version": "1.0.0",
        "workflow_name": entry.get("name"),
        "version": entry.get("version") or workflow_json.get("version", ""),
        "description": entry.get("description") or workflow_json.get("description", ""),
        "scope": entry.get("scope", "user"),
        "user_id": entry.get("user_id", ""),
        "parameter_schema": _workflow_parameter_schema(workflow_json),
        "steps": entry.get("steps", []),
        "requirements": entry.get("requirements", []),
        # W4: declared event subscriptions (additive; empty by default)
        "triggers": workflow_json.get("triggers", []),
    }


def _validate_params_against_schema(
    params: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Minimal JSON-schema enforcement for required top-level properties."""
    if schema.get("type") != "object":
        return
    required = schema.get("required")
    if not isinstance(required, list):
        return
    missing = [
        name for name in required if isinstance(name, str) and name not in params
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "validation_error",
                "code": "workflow_parameters_invalid",
                "message": f"Missing required workflow parameters: {', '.join(missing)}",
                "missing": missing,
            },
        )


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
    return strict_auth_mode()


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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="user_id is required"
        )
    return user_id.strip()


def _declared_workflow_project(resolved_params: dict[str, Any]) -> str:
    value = resolved_params.get("project_id")
    if isinstance(value, str):
        return value.strip()
    return ""


def _checked_membership_set(
    x_verified_project_memberships: str | None,
) -> set[str]:
    """Caller membership set honored for cross-project authorization.

    Wave-6 F1b: the memberships header is NOT part of the v1 HMAC signature
    tuple — an attacker with any one valid gateway session (or none at all
    behind the sidecar port in strict mode) could forge
    ``X-Verified-Project-Memberships`` and claim victim-project scope. Under
    strict mode only the HMAC-covered primary project (or admin) may
    authorize a foreign declared project_id. Memberships-based authorization
    stays available in non-strict (mvp) mode until the handshake tuple is
    extended to v2.
    """
    if _strict_auth_mode():
        return set()
    memberships_raw = _header_str(x_verified_project_memberships) or ""
    return {item.strip() for item in memberships_raw.split(",") if item.strip()}


def _authorize_declared_workflow_project(
    declared_project_id: str,
    verified_project_id: str | None,
    x_verified_project_memberships: str | None,
    x_verified_is_admin: str | None,
) -> None:
    """Workflow-declared project_id is authoritative — but only when the
    caller is authorized for it (admin role, or the declared project is in
    the caller's verified membership set). Anything else is a loud 403,
    never a silent rewrite (Phase P0-1).
    """
    is_admin = _header_str(x_verified_is_admin) == "true"
    allowed = {p for p in (verified_project_id,) if p}
    allowed |= _checked_membership_set(x_verified_project_memberships)
    if is_admin or not verified_project_id:
        return
    if declared_project_id not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"project_id '{declared_project_id}' is outside the "
                "caller's verified projects"
            ),
        )


def _verified_workflow_params(
    params: dict[str, Any],
    x_verified_project_id: str | None,
    x_verified_project_memberships: str | None = None,
    x_verified_is_admin: str | None = None,
) -> dict[str, Any]:
    verified_project_id = _header_str(x_verified_project_id)
    resolved_params = dict(params)
    declared_project_id = _declared_workflow_project(resolved_params)
    if declared_project_id:
        _authorize_declared_workflow_project(
            declared_project_id,
            verified_project_id,
            x_verified_project_memberships,
            x_verified_is_admin,
        )
        return resolved_params
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
    x_verified_is_admin: str | None = None,
    x_verified_hmac: str | None = None,
    x_verified_workflow_author: str | None = None,
) -> None:
    """Strict-mode workflow identity gate.

    Wave-6 F1: this resolver previously passed on presence alone — any raw
    `X-User-ID` + `X-Verified-Project-ID` headers were trusted against strict,
    bypassing the signature requirement every other sealed resolver uses. Now:
    presence alone is insufficient in strict mode; the request must carry a
    valid gateway HMAC over the complete identity tuple.
    """
    if not (_header_str(x_user_id) and _header_str(x_verified_project_id)):
        if _strict_auth_mode():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Gateway-verified workflow identity",
            )
        return
    if not _strict_auth_mode():
        return
    from ._verified_headers import handshake_secret, verified_headers_trusted

    if not verified_headers_trusted(
        strict=True,
        secret=handshake_secret(),
        presented_hmac=_header_str(x_verified_hmac),
        user_id=_header_str(x_user_id),
        project_id=_header_str(x_verified_project_id),
        is_admin=_header_str(x_verified_is_admin) == "true",
        is_workflow_author=_header_str(x_verified_workflow_author) == "true",
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gateway-verified identity signature rejected",
        )


def _resolve_workflow_caller(
    x_user_id: str | None,
    x_verified_project_id: str | None,
    x_verified_is_admin: str | None,
    x_verified_hmac: str | None,
    x_verified_workflow_author: str | None,
):
    from ._verified_headers import resolve_caller

    _require_verified_workflow_identity(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    identity = resolve_caller(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified caller identity",
        )
    return identity


def _workflow_job_status(job_status: JobStatus) -> str:
    return workflow_job_status(job_status)


def _workflow_job_payload(job) -> dict[str, Any]:
    return job.to_dict() if hasattr(job, "to_dict") else dict(job)


def _find_workflow_job_by_execution_id(
    store: JobStore | None,
    execution_id: str,
):
    return find_workflow_job_by_execution_id(store, execution_id)


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
        if (
            job.status
            in {
                JobStatus.PLANNED,
                JobStatus.QUEUED,
                JobStatus.CANCELLED,
                JobStatus.FAILED,
            }
            or execution.get("status") == "submitted"
        ):
            execution["status"] = durable_status
    return execution


def _execution_payload_from_job(job) -> dict[str, Any]:
    return execution_payload_from_job(job)


@router.get("")
async def list_workflows(
    scope: str | None = Query(None, description="Optional workflow scope filter"),
    name: str | None = Query(None, description="Optional workflow-name filter"),
    user_id: str | None = Query(None, description="Optional owner filter"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    resolved_user_id = _header_str(x_user_id) or user_id
    payload = {
        key: value
        for key, value in (
            ("scope", scope),
            ("name", name),
            ("user_id", resolved_user_id),
        )
        if value is not None
    }
    raw = handle_workflow_command(
        TaskCmd.LIST_WORKFLOWS,
        resolved_user_id or "_",
        dumps_with_correlation_id(payload) if payload else "",
    )
    return _parse_command_response(raw, action="list")


@router.get("/capabilities")
async def workflow_capabilities(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
):
    """Read-only effective UMRS workflow capabilities for the verified caller."""
    from .workflow_capability import capabilities_payload

    identity = _resolve_workflow_caller(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    return capabilities_payload(identity)


class GateDecisionRequest(BaseModel):
    """Manual gate approval decision (W5)."""

    approved: bool = Field(..., description="True = approve, False = reject")
    approver: str | None = Field(default=None, max_length=255)
    comment: str | None = Field(default=None, max_length=2000)


@router.get("/{workflow_name}/triggers")
async def list_workflow_triggers(
    workflow_name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
):
    """W4: return the workflow's declared event trigger subscriptions."""
    _require_verified_workflow_identity(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    caller = _header_str(x_user_id) or ""
    entry = _resolve_workflow_catalog_entry(workflow_name, caller)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_name}' not found",
        )
    workflow_json = _read_entry_workflow_json(entry)
    return {
        "status": "ok",
        "workflow_name": workflow_name,
        "triggers": workflow_json.get("triggers", []),
    }


@router.get("/{workflow_name}")
async def describe_workflow(
    workflow_name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
):
    """Return declared parameter schema and metadata for a workflow."""
    _require_verified_workflow_identity(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    caller = _header_str(x_user_id) or ""
    entry = _resolve_workflow_catalog_entry(workflow_name, caller)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_name}' not found",
        )
    return _describe_workflow_entry(entry)


def _check_workflow_quota(project_id: str) -> None:
    """Best-effort governance quota check (no-op when inactive)."""
    try:
        from .governance import check_quota

        check_quota(project_id, "workflows")
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise
        pass  # governance not active


def _resolved_upload_name(filename: str, workflow_name: str | None) -> str:
    """Validate filename/name form inputs and return the final name."""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Filename required"
        )
    resolved = workflow_name or Path(filename).stem
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="workflow_name is required"
        )
    return resolved


def _load_extracted_workflow_json(dest: str) -> tuple[str, dict[str, Any]]:
    """Read version + payload from the extracted workflow.json."""
    workflow_json_path = os.path.join(dest, "workflow.json")
    try:
        with open(workflow_json_path, encoding="utf-8") as handle:
            workflow_json = json.load(handle)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load extracted workflow.json: {exc}",
        ) from exc
    return str(workflow_json.get("version", "0.1")), workflow_json


def _register_uploaded_workflow(
    *,
    resolved_workflow_name: str,
    version: str,
    resolved_user_id: str,
    dest: str,
    workflow_json: dict[str, Any],
    scope: str,
    project_id: str,
) -> None:
    """Register the workflow in the catalog + governance ledger."""
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
        "workflows",
        resolved_workflow_name,
        resolved_user_id,
        project_id=project_id,
        tier=scope,
    )


@router.post(
    "/upload",
    response_model=WorkflowUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_workflow(
    file: UploadFile = File(...),
    user_id: str | None = Form(None),
    workflow_name: str | None = Form(None),
    scope: str = Form("user"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
):
    from .workflow_capability import authorize_workflow_upload

    identity = _resolve_workflow_caller(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    authorize_workflow_upload(identity, scope)
    resolved_user_id = _verified_workflow_user(user_id, x_user_id)

    project_id = x_verified_project_id or ""
    _check_workflow_quota(project_id)
    resolved_workflow_name = _resolved_upload_name(file.filename or "", workflow_name)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    ok, dest, err = handle_upload_workflow(
        resolved_user_id, temp_path, resolved_workflow_name
    )
    if not ok:
        _raise_upload_error(err or "Workflow upload failed")

    version, workflow_json = _load_extracted_workflow_json(dest)
    _register_uploaded_workflow(
        resolved_workflow_name=resolved_workflow_name,
        version=version,
        resolved_user_id=resolved_user_id,
        dest=dest,
        workflow_json=workflow_json,
        scope=scope,
        project_id=project_id,
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
    x_verified_project_memberships: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    user_id = _verified_workflow_user(request.user_id or "", x_user_id)
    params = _verified_workflow_params(
        request.resolved_params(),
        x_verified_project_id,
        x_verified_project_memberships,
        x_verified_is_admin,
    )
    entry = _resolve_workflow_catalog_entry(workflow_name, user_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_name}' not found",
        )
    describe = _describe_workflow_entry(entry)
    schema = describe.get("parameter_schema", {})
    _validate_params_against_schema(params, schema)
    store = get_optional_job_store()
    job_backend = DurableJobCommandBackend(store) if store is not None else None
    # Wave-6 F1: identity only marked verified when presence AND (strict mode,
    # when active) HMAC trust was established. The mouth-to-mouth mark stays
    # False for legacy unsigned paths so the durable layer never uprates a
    # forged tuple into `gateway_verified` truth.
    identity_verified = bool(
        _header_str(x_user_id) and _header_str(x_verified_project_id)
    )
    if identity_verified and _strict_auth_mode():
        from ._verified_headers import handshake_secret, verified_headers_trusted

        identity_verified = verified_headers_trusted(
            strict=True,
            secret=handshake_secret(),
            presented_hmac=_header_str(x_verified_hmac),
            user_id=_header_str(x_user_id),
            project_id=_header_str(x_verified_project_id),
            is_admin=_header_str(x_verified_is_admin) == "true",
            is_workflow_author=_header_str(x_verified_workflow_author) == "true",
        )
    raw = handle_workflow_command(
        TaskCmd.RUN_WORKFLOW,
        user_id,
        json.dumps(
            {
                "workflow_name": workflow_name,
                "params": params,
                "_identity_verified": identity_verified,
            }
        ),
        job_backend=job_backend,
        # Wave-6 F1: the durable trust bridge reflects the verified flag,
        # never hardcoded True. Under strict mode + missing/invalid HMAC,
        # `_identity_verified` is False, so this no longer uprates a forged
        # tuple into `gateway_verified` truth downstream.
        trusted_identity=identity_verified,
    )
    return _parse_command_response(raw, action="trigger")


# ---------------------------------------------------------------------------
# Execution status / results / cancel
# ---------------------------------------------------------------------------


@router.get("/executions/{execution_id}")
async def get_execution_status(
    execution_id: str,
    identity: CallerIdentity = Depends(get_caller_identity),
):
    """Return workflow execution status, backed by durable job state when present."""
    store = get_optional_job_store()
    record, job = resolve_execution_context(execution_id, store=store)
    enforce_http_execution_access(
        identity,
        record=record,
        job=job,
        execution_id=execution_id,
    )
    if record is not None:
        linked_job = _get_job_for_record(store, record) if job is None else job
        return {"status": "ok", "execution": _execution_payload(record, linked_job)}
    return {"status": "ok", "execution": _execution_payload_from_job(job)}


@router.get("/executions/{execution_id}/results")
async def get_execution_results(
    execution_id: str,
    identity: CallerIdentity = Depends(get_caller_identity),
):
    """Return the structured outputs of a finished execution.

    For unfinished executions, returns the partial results gathered so
    far (so clients can render progressive UIs without retrying).
    """
    from ..server.workflow_runner import list_artifacts

    store = get_optional_job_store()
    record, job = resolve_execution_context(execution_id, store=store)
    enforce_http_execution_access(
        identity,
        record=record,
        job=job,
        execution_id=execution_id,
    )
    if record is None:
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

    linked_job = _get_job_for_record(store, record) if job is None else job
    execution = _execution_payload(record, linked_job)

    artifacts = [
        {"key": key, "size_bytes": size} for key, size in list_artifacts(record)
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
    identity: CallerIdentity = Depends(get_caller_identity),
):
    """Request cancellation of a workflow execution."""
    from ..server.workflow_execution_registry import get_execution_registry

    store = get_optional_job_store()
    record, job = resolve_execution_context(execution_id, store=store)
    enforce_http_execution_access(
        identity,
        record=record,
        job=job,
        execution_id=execution_id,
    )

    registry = get_execution_registry()
    cancelled_record = registry.cancel(execution_id) if record is not None else None
    if cancelled_record is not None:
        record = cancelled_record
    linked_job = job
    if record is not None and linked_job is None:
        linked_job = _get_job_for_record(store, record)
    if linked_job is not None and store is not None:
        with contextlib.suppress(InvalidTransitionError):
            store.cancel_job(linked_job.job_id, reason="workflow_execution_cancelled")

    if record is not None:
        return {"status": "ok", "execution": _execution_payload(record, linked_job)}
    return {"status": "ok", "execution": _execution_payload_from_job(linked_job)}


# ---------------------------------------------------------------------------
# Manual gate approval (W5)
# ---------------------------------------------------------------------------


def _gate_state_for(record: Any, step_name: str) -> str:
    """Classify a step's gate state for approve-route validation."""
    step = record.steps.get(step_name)
    if step is None:
        return "unknown_step"
    if record.gate_decisions.get(step_name) is not None:
        return "already_decided"
    if step.status != "waiting_approval":
        return "not_waiting"
    return "waiting"


def _reject_gate_decision(state: str, execution_id: str, step_name: str) -> None:
    """Raise the HTTP error matching a non-waiting gate state."""
    if state == "unknown_step":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Step '{step_name}' is not part of execution '{execution_id}'"),
        )
    detail = (
        f"Step '{step_name}' in execution '{execution_id}' already has a "
        "recorded gate decision"
        if state == "already_decided"
        else (
            f"Step '{step_name}' in execution '{execution_id}' is not "
            "waiting for approval"
        )
    )
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


@router.post("/executions/{execution_id}/steps/{step_name}/approve")
async def approve_execution_gate(
    execution_id: str,
    step_name: str,
    request: GateDecisionRequest,
    identity: CallerIdentity = Depends(get_caller_identity),
):
    """W5: approve or reject a waiting manual-gate step.

    Releases the parked worker thread; the decision lands in the
    execution's ``results.gate_audit`` trail (durable via the record's
    persist hook). Non-waiting steps return 409 with the reason.
    """
    from ..server.workflow_execution_registry import get_execution_registry

    store = get_optional_job_store()
    record, job = resolve_execution_context(execution_id, store=store)
    enforce_http_execution_access(
        identity,
        record=record,
        job=job,
        execution_id=execution_id,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Execution '{execution_id}' is not tracked by the in-process registry"
            ),
        )

    gate_state = _gate_state_for(record, step_name)
    if gate_state != "waiting":
        _reject_gate_decision(gate_state, execution_id, step_name)

    registry = get_execution_registry()
    approver = request.approver or getattr(identity, "user_id", None) or record.user_id
    updated = registry.resolve_approval(
        execution_id,
        step_name,
        approved=request.approved,
        approver=approver,
        comment=request.comment,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' is not tracked by the registry",
        )

    linked_job = _get_job_for_record(store, record) if job is None else job
    return {"status": "ok", "execution": _execution_payload(updated, linked_job)}
