"""Workflow Registry REST API.

Provides:
- GET /umrs/workflows - List registered workflows with optional filters
- POST /umrs/workflows/{workflow_name}/trigger - Accept workflow execution

This router is intentionally thin. It bridges the HTTP sidecar to the
existing workflow registry command handlers used by the ZMQ server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
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
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, model_validator

from ..core.TaskCmdStatus import TaskCmd
from ..core.zmq_correlation import dumps_with_correlation_id
from ..jobs.models import JobStatus
from ..jobs.store import JobNotFoundError, JobStore
from ..server.file_handling import (
    MAX_UPLOAD_BYTES,
    handle_upload_workflow,
    workflow_name_refusal,
)
from ..server.job_tracking_handler import DurableJobCommandBackend
from ..server.workflow_registry_handler import (
    _resolve_workflow_catalog_entry,
    handle_workflow_command,
    register_workflow,
    resolve_workflow_for_caller,
    set_workflow_access,
    unregister_workflow,
    workflow_access_entries,
)
from ._access import checked_access, resource_caller
from ._verified_headers import get_verified_memberships
from .jobs import CallerIdentity, get_caller_identity, get_optional_job_store
from .typed_errors import PlatformError, raise_typed
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
        """Legacy 'params' alias for 'parameters' (R6.7a, deprecated 2026-08-23).

        No-op back-compat accept: emits a once-per-process deprecation
        warning; removal is scheduled for two releases out (R1 pattern).
        """
        from ._identity import warn_legacy_params_once

        warn_legacy_params_once("workflows.params_alias")
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
        raise_typed(
            PlatformError.workflows2(
                "bad_request",
                {
                    "error": "validation_error",
                    "code": "workflow_parameters_invalid",
                    "message": f"Missing required workflow parameters: {', '.join(missing)}",
                    "missing": missing,
                },
            ),
        )


def _parse_command_response(raw: str, *, action: str) -> dict[str, Any]:
    """Parse a workflow command response and map JSON errors to HTTP errors."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON from workflow command handler during %s", action)
        raise_typed(
            PlatformError.workflows(
                "bad_gateway", f"Invalid workflow handler response during {action}"
            ),
            cause=exc,
        )

    if not isinstance(data, dict):
        raise_typed(
            PlatformError.workflows(
                "bad_gateway", f"Unexpected workflow handler response during {action}"
            )
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


_MAX_WORKFLOW_UPLOAD_BYTES = MAX_UPLOAD_BYTES
_SPOOL_CHUNK = 1024 * 1024


async def _spool_upload(file: UploadFile, dest_path: str) -> None:
    """Copy the uploaded zip to *dest_path* in chunks, refusing an oversize
    part by its size before reading it (2026-10-21: ``await file.read()``
    held a 600 MiB upload in memory before the 500 MiB cap refused it)."""
    size = getattr(file, "size", None)
    if isinstance(size, int) and size > _MAX_WORKFLOW_UPLOAD_BYTES:
        _raise_upload_error(
            f"Zip file too large ({size} bytes, max {_MAX_WORKFLOW_UPLOAD_BYTES})"
        )
    await asyncio.to_thread(_copy_part, file.file, dest_path)


def _copy_part(source: Any, dest_path: str) -> None:
    with open(dest_path, "wb") as dest:
        shutil.copyfileobj(source, dest, _SPOOL_CHUNK)


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
        raise_typed(
            PlatformError.auth(
                "gateway_identity_required", "Missing Gateway-verified user identity"
            )
        )
    if not user_id or not user_id.strip():
        raise_typed(PlatformError.workflows("bad_request", "user_id is required"))
    return user_id.strip()


def _declared_workflow_project(resolved_params: dict[str, Any]) -> str:
    value = resolved_params.get("project_id")
    if isinstance(value, str):
        return value.strip()
    return ""


def _authorize_declared_workflow_project(
    declared_project_id: str,
    verified_project_id: str | None,
    memberships: frozenset[str] | None,
    x_verified_is_admin: str | None,
) -> None:
    """Workflow-declared project_id is authoritative — but only when the
    caller is authorized for it (admin role, or the declared project is in
    the caller's verified membership set). Anything else is a loud 403,
    never a silent rewrite (Phase P0-1).

    ``memberships`` comes from ``get_verified_memberships``: in strict mode
    the gateway list only when its own HMAC verified (wave-6 F1b ignored the
    unsigned list outright; H-1 2026-10-18 honours the signed one), in mvp
    the list as presented.
    """
    is_admin = _header_str(x_verified_is_admin) == "true"
    allowed = {p for p in (verified_project_id,) if p}
    allowed |= set(memberships or ())
    if is_admin or not verified_project_id:
        return
    if declared_project_id not in allowed:
        raise_typed(
            PlatformError.workflows(
                "forbidden",
                (
                    f"project_id '{declared_project_id}' is outside the "
                    "caller's verified projects"
                ),
            )
        )


def _verified_workflow_params(
    params: dict[str, Any],
    x_verified_project_id: str | None,
    memberships: frozenset[str] | None = None,
    x_verified_is_admin: str | None = None,
) -> dict[str, Any]:
    verified_project_id = _header_str(x_verified_project_id)
    resolved_params = dict(params)
    declared_project_id = _declared_workflow_project(resolved_params)
    if declared_project_id:
        _authorize_declared_workflow_project(
            declared_project_id,
            verified_project_id,
            memberships,
            x_verified_is_admin,
        )
        return resolved_params
    if verified_project_id:
        resolved_params["project_id"] = verified_project_id
        return resolved_params
    if _strict_auth_mode():
        raise_typed(
            PlatformError.auth(
                "gateway_identity_required", "Missing Gateway-verified project identity"
            )
        )
    return resolved_params


def _require_verified_workflow_identity(
    x_user_id: str | None,
    x_verified_project_id: str | None,
    x_verified_is_admin: str | None = None,
    x_verified_hmac: str | None = None,
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
            raise_typed(
                PlatformError.auth(
                    "gateway_identity_required",
                    "Missing Gateway-verified workflow identity",
                )
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
    ):
        raise_typed(
            PlatformError.auth(
                "gateway_signature_rejected",
                "Gateway-verified identity signature rejected",
            )
        )


def _verified_users_may_upload() -> bool:
    """``UMRS_WORKFLOW_UPLOAD_ROLE``: ``verified`` (default, ADR-079 D-1) lets
    any gateway-verified user upload user-scope workflows; ``workflow_author``
    restores the ADR-027 role gate."""
    policy = os.environ.get("UMRS_WORKFLOW_UPLOAD_ROLE", "verified").strip().lower()
    return policy != "workflow_author"


def _scope_policy_decides(
    scope: str, is_admin: bool, headers_present: bool, deny: Any
) -> bool:
    """``scope=system`` needs admin (*deny* raises); True when the ADR-079
    D-1 policy already admits the upload: a gateway-verified researcher
    uploads their own user-scope workflows (maintainer 2026-09-25) — the code
    never runs in the server (ADR-074) and runs only for its owner (D-8)."""
    if scope == "system" and not is_admin:
        deny("system scope requires admin")
    return headers_present and _verified_users_may_upload()


def _enforce_workflow_upload_role(
    *,
    scope: str,
    actor: str,
    project_id: str,
    is_admin: bool,
    presented_author: str | None,
    presented_author_hmac: str | None,
) -> None:
    """ADR-027 role gate on workflow uploads, audited either way.

    - ``scope=system`` requires admin (aligning with ModelScope/SkillScope).
    - Any other upload requires a trusted ``workflow_author`` claim or admin.
    - Claim trust is fail-closed: the wf signature (own basis) must verify;
      an unsigned or forged author header reads as not-author.

    Enforcement applies when the request carries a verified identity (the
    strict production path) or strict mode is active; the header-less compat
    path keeps legacy behavior (the compat deployments carry no role
    carrier yet and would otherwise soft-lock uploads).
    """
    from ._verified_headers import handshake_secret, workflow_author_trusted
    from .audit_events import emit_audit

    def _deny(reason: str) -> None:
        emit_audit(
            "workflow_upload",
            actor=actor,
            project_id=project_id,
            outcome="denied",
            scope=scope,
            details={"reason": reason},
        )
        raise_typed(PlatformError.workflows("forbidden", reason))

    headers_present = bool(actor and project_id)
    if not (_strict_auth_mode() or headers_present):
        return
    if _scope_policy_decides(scope, is_admin, headers_present, _deny):
        return
    author_ok = workflow_author_trusted(
        strict=_strict_auth_mode(),
        secret=handshake_secret(),
        presented_author=presented_author,
        presented_author_hmac=presented_author_hmac,
        user_id=actor,
        project_id=project_id,
    )
    if not (author_ok or is_admin):
        _deny("workflow uploads require the workflow_author role (or admin)")


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


def _list_workflows_payload(
    scope: str | None,
    name: str | None,
    resolved_user_id: str | None,
    *,
    caller_id: str | None,
    verified_project: str | None,
    admin_all: bool,
) -> dict[str, Any]:
    if admin_all:
        # Explicit admin audit view (mirrors ZMQ ``all_users: true``,
        # 2026-10-06): no owner/share scoping — filters only.
        pairs: tuple[tuple[str, Any], ...] = (("scope", scope), ("name", name))
    else:
        pairs = (
            ("scope", scope),
            ("name", name),
            ("user_id", resolved_user_id),
            # ADR-080: entries whose access list admits the caller (shared).
            ("include_shared_for", caller_id),
            ("verified_project", verified_project),
            # The owner filter above dropped the system-scope workflows every
            # user may run (review 2026-10-04).
            ("include_system", True if resolved_user_id else None),
        )
    return {key: value for key, value in pairs if value is not None}


@router.get("")
async def list_workflows(
    scope: str | None = Query(None, description="Optional workflow scope filter"),
    name: str | None = Query(None, description="Optional workflow-name filter"),
    user_id: str | None = Query(None, description="Optional owner filter"),
    all_users: bool = Query(
        False,
        description="Admin-only: list the full catalog across owners",
    ),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    resolved_user_id = _header_str(x_user_id) or user_id
    payload = _list_workflows_payload(
        scope,
        name,
        resolved_user_id,
        caller_id=_header_str(x_user_id),
        verified_project=_header_str(x_verified_project_id),
        admin_all=all_users and _header_str(x_verified_is_admin) == "true",
    )
    raw = handle_workflow_command(
        TaskCmd.LIST_WORKFLOWS,
        resolved_user_id or "_",
        dumps_with_correlation_id(payload) if payload else "",
    )
    return _parse_command_response(raw, action="list")


class GateDecisionRequest(BaseModel):
    """Manual gate approval decision (W5)."""

    approved: bool = Field(..., description="True = approve, False = reject")
    approver: str | None = Field(default=None, max_length=255)
    comment: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Run definitions — persisted named invocation templates (replay without LLM)
# ---------------------------------------------------------------------------


class RunDefinitionRequest(BaseModel):
    """Create/update a run definition."""

    name: str = Field(..., min_length=1, max_length=128)
    workflow_name: str = Field(..., min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


def _resolve_definition_for_caller(
    name: str, caller: str, is_admin: bool
) -> dict[str, Any]:
    """Owner-first, then admin-any: admins see any definition by name."""
    from ..server.run_definition_registry import get_definition

    record = get_definition(name, caller)
    if record is not None:
        return record
    if is_admin:
        for candidate in _all_definitions():
            if candidate.get("name") == name:
                return candidate
    raise_typed(
        PlatformError.workflows("not_found", f"Run definition '{name}' not found")
    )


def _all_definitions() -> list[dict[str, Any]]:
    from ..server.run_definition_registry import list_definitions

    return list_definitions()


@router.put("/run-definitions")
async def save_run_definition(
    request: RunDefinitionRequest,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    from r4_contracts.workflows import RunDefinitionRecord

    from ..server.run_definition_registry import save_definition

    caller = _header_str(x_user_id) or ""
    workflow_entry = resolve_workflow_for_caller(
        request.workflow_name,
        caller,
        is_admin=_header_str(x_verified_is_admin) == "true",
        caller_project=_header_str(x_verified_project_id),
    )
    if workflow_entry is None:
        raise_typed(
            PlatformError.workflows(
                "not_found",
                (
                    f"Workflow '{request.workflow_name}' not found; "
                    "run definitions must reference a registered workflow"
                ),
            )
        )
    try:
        record_dict = save_definition(
            name=request.name,
            workflow_name=request.workflow_name,
            params=request.params,
            user_id=caller,
            project_id=_header_str(x_verified_project_id) or "",
            description=request.description,
        )
    except ValueError as exc:
        raise_typed(PlatformError.workflows("bad_request", str(exc)), cause=exc)
    return {
        "status": "ok",
        "definition": RunDefinitionRecord.from_dict(record_dict).to_dict(),
    }


@router.get("/run-definitions")
async def list_run_definitions(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    from ..server.run_definition_registry import list_definitions

    caller = _header_str(x_user_id) or ""
    is_admin = _header_str(x_verified_is_admin) == "true"
    # Params may embed sensitive invocation details: owner-scoped by default,
    # admins list everything.
    records = list_definitions(None if is_admin else caller)
    return {"status": "ok", "count": len(records), "definitions": records}


@router.get("/run-definitions/{name}")
async def get_run_definition(
    name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    record = _resolve_definition_for_caller(
        name, caller, _header_str(x_verified_is_admin) == "true"
    )
    return {"status": "ok", "definition": record}


@router.delete("/run-definitions/{name}")
async def delete_run_definition(
    name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    from ..server.run_definition_registry import delete_definition

    caller = _header_str(x_user_id) or ""
    is_admin = _header_str(x_verified_is_admin) == "true"
    record = _resolve_definition_for_caller(name, caller, is_admin)
    owner = str(record.get("user_id", ""))
    # delete_definition is owner-scoped; admins deleting another user's
    # definition delete under the record's owner identity.
    victim = owner if is_admin else caller
    removed = delete_definition(name, victim)
    if removed == 0:
        raise_typed(
            PlatformError.workflows("not_found", f"Run definition '{name}' not found")
        )
    return {
        "status": "ok",
        "name": name,
        "message": f"Run definition '{name}' deleted",
    }


@router.post("/run-definitions/{name}/trigger")
async def trigger_run_definition(
    name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    memberships: frozenset[str] = Depends(get_verified_memberships),
):
    """Replay one persisted (workflow_name, params) — exact same inputs.

    No param overrides are accepted on purpose: a run definition is the
    reproducibility record; overriding would defeat it.
    """
    caller = _header_str(x_user_id) or ""
    record = _resolve_definition_for_caller(
        name, caller, _header_str(x_verified_is_admin) == "true"
    )
    request = WorkflowTriggerRequest(
        user_id=str(record.get("user_id") or caller),
        params=dict(record.get("params") or {}),
    )
    return await trigger_workflow(
        workflow_name=str(record["workflow_name"]),
        request=request,
        x_user_id=x_user_id,
        x_verified_project_id=x_verified_project_id,
        x_verified_is_admin=x_verified_is_admin,
        x_verified_hmac=x_verified_hmac,
        memberships=memberships,
    )


@router.get("/{workflow_name}/triggers")
async def list_workflow_triggers(
    workflow_name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """W4: return the workflow's declared event trigger subscriptions."""
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    entry = resolve_workflow_for_caller(
        workflow_name,
        caller,
        is_admin=_header_str(x_verified_is_admin) == "true",
        caller_project=_header_str(x_verified_project_id),
    )
    if entry is None:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )
    workflow_json = _read_entry_workflow_json(entry)
    return {
        "status": "ok",
        "workflow_name": workflow_name,
        "triggers": workflow_json.get("triggers", []),
    }


# ---------------------------------------------------------------------------
# Workflow package — definition + registered code for the console detail view
# ---------------------------------------------------------------------------

_PACKAGE_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}
_PACKAGE_FILE_CAP = 32 * 1024  # per-file bytes
_PACKAGE_TOTAL_CAP = 200 * 1024  # summed read across files
_PACKAGE_MAX_FILES = 24


def _package_file_row(root: str, path: str) -> dict[str, Any]:
    """One code file row: content capped honestly (`truncated` flags it)."""
    full = os.path.join(root, path)
    size = os.path.getsize(full)
    with open(full, encoding="utf-8", errors="replace") as handle:
        raw = handle.read(_PACKAGE_FILE_CAP + 1)
    truncated = len(raw) > _PACKAGE_FILE_CAP
    content = raw[:_PACKAGE_FILE_CAP]
    return {
        "path": path,
        "bytes": size,
        "lines": content.count("\n")
        + (0 if content.endswith("\n") or not content else 1),
        "content": content,
        "truncated": truncated,
    }


def _collect_package_files(source_path: str) -> tuple[list[dict[str, Any]], bool]:
    """Walk the workflow's registered source dir under budget.

    Returns (rows, truncated) — skipped dirs, the file-count cap and the
    total-bytes cap all mark ``truncated=True``, never a silent drop.
    """
    rows: list[dict[str, Any]] = []
    budget = _PACKAGE_TOTAL_CAP
    truncated = False
    for dirpath, dirnames, filenames in os.walk(source_path):
        dirnames[:] = [d for d in sorted(dirnames) if d not in _PACKAGE_SKIP_DIRS]
        for filename in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, filename), source_path)
            if len(rows) >= _PACKAGE_MAX_FILES or budget <= 0:
                truncated = True
                break
            row = _package_file_row(source_path, rel)
            budget -= len(row["content"])
            rows.append(row)
    return rows, truncated


@router.get("/{workflow_name}/package")
async def get_workflow_package(
    workflow_name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Definition + registered CODE for the console definition card.
    # auth: verified

    Resolved exactly like describe (ADR-074 D-8, ADR-080): own or system
    entry, one whose access list admits the caller (``project:`` scopes only
    for the gateway-verified project), another owner's only for admins.
    Without the verified project a project-shared workflow listed in /work
    opened as a 404 (live 2026-10-23). ``files`` carries registered sources
    with honest caps (``truncated`` flags); a missing source dir ships an
    empty list, never an invented listing.
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    entry = resolve_workflow_for_caller(
        workflow_name,
        caller,
        is_admin=_header_str(x_verified_is_admin) == "true",
        caller_project=_header_str(x_verified_project_id),
    )
    if entry is None:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )
    return _package_payload(entry)


def _package_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """The ``/package`` body for a resolved catalog *entry* (split out of the
    route for the complexity ratchet, 2026-10-02; behaviour unchanged)."""
    payload = _describe_workflow_entry(entry)
    workflow_json = _read_entry_workflow_json(entry)
    # Rich definition fields for the graph card: the REAL workflow.json
    # carries depends_on/config detail catalog rows flatten away — it wins
    # whenever it declares steps; the catalog row set is the fallback.
    if isinstance(workflow_json.get("steps"), list) and workflow_json["steps"]:
        payload["steps"] = workflow_json["steps"]
    payload["resources"] = workflow_json.get("resources", {})
    payload["execution_mode"] = workflow_json.get("execution_mode")
    payload["output"] = workflow_json.get("output")
    payload["source_path"] = str(entry.get("source_path") or "")
    source_path = payload["source_path"]
    if source_path and os.path.isdir(source_path):
        files, truncated = _collect_package_files(source_path)
        payload["files"] = files
        payload["files_truncated"] = truncated or any(f["truncated"] for f in files)
    else:
        payload["files"] = []
        payload["files_truncated"] = False
    return payload


@router.delete("/{workflow_name}")
async def delete_workflow(
    workflow_name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Delete a registered workflow.

    Authorization: caller must OWN the registration, unless the verified
    caller is an admin. System-scope workflows always require admin — a
    shared workflow must never be deletable by an arbitrary project member.
    Governance ledger (ownership + quota) is updated best-effort.
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    is_admin = _header_str(x_verified_is_admin) == "true"

    entry = _resolve_workflow_catalog_entry(workflow_name, caller)
    if entry is None:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )
    owner = str(entry.get("user_id", ""))
    scope = str(entry.get("scope", "user"))
    if scope == "system" and not is_admin:
        raise_typed(
            PlatformError.auth(
                "admin_required", "Only admins may delete system workflows"
            )
        )
    if owner and owner != caller and not is_admin:
        raise_typed(
            PlatformError.workflows(
                "forbidden", f"Workflow '{workflow_name}' is owned by another user"
            )
        )

    result = unregister_workflow(
        workflow_name,
        user_id=owner,
        scope=scope,
    )
    if result["removed"] == 0:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )

    from .governance import record_resource_deleted

    record_resource_deleted(
        "workflows", workflow_name, project_id=_header_str(x_verified_project_id) or ""
    )
    return {
        "status": "ok",
        "workflow_name": workflow_name,
        "removed": result["removed"],
        "message": f"Workflow '{workflow_name}' deleted",
    }


@router.get("/{workflow_name}")
async def describe_workflow(
    workflow_name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Return declared parameter schema and metadata for a workflow.

    Owner-scoped (ADR-074 D-8): the caller's own or a system workflow;
    another owner's only for admins.
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    entry = resolve_workflow_for_caller(
        workflow_name,
        caller,
        is_admin=_header_str(x_verified_is_admin) == "true",
        caller_project=_header_str(x_verified_project_id),
    )
    if entry is None:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )
    return _describe_workflow_entry(entry)


class WorkflowAccessUpdate(BaseModel):
    """ADR-080 access list update (canonical scope grammar)."""

    access: list[str] = Field(
        default_factory=list,
        description=(
            "user:<id> | project:<id> | circle:<proj>/<id> | system; [] = owner/system scopes only"
        ),
    )


def _access_target_owner(
    workflow_name: str, caller: str, owner: str | None, is_admin: bool
) -> str:
    """Whose entries the access update touches: the caller's, or another
    owner's when both requested and admin (typed 403 otherwise)."""
    if not owner or owner == caller:
        return caller
    if not is_admin:
        raise_typed(
            PlatformError.workflows(
                "forbidden", "only an admin may change another owner's access"
            )
        )
    return owner


@router.get("/{workflow_name}/access")
async def get_workflow_access(
    workflow_name: str,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """The workflow's ADR-080 access list (any caller who may run/describe it).
    # auth: verified
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    entry = resolve_workflow_for_caller(
        workflow_name,
        caller,
        is_admin=_header_str(x_verified_is_admin) == "true",
        caller_project=_header_str(x_verified_project_id),
    )
    if entry is None:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )
    return {
        "workflow_name": workflow_name,
        "owner": entry.get("user_id", ""),
        "access": list(entry.get("access") or []),
    }


@router.put("/{workflow_name}/access")
async def put_workflow_access(
    workflow_name: str,
    body: WorkflowAccessUpdate,
    request: Request,
    owner: str | None = Query(None, description="Admin-only: another owner's entries"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Replace the access list on the caller's entries of *workflow_name*
    (all versions; owner or admin; audited).

    The list names who beyond the owner may RUN and describe the workflow
    (ADR-080); lifecycle (delete/re-upload) is unchanged. ``system``
    entries are admin-only; entries at ``system`` scope require admin to
    re-scope (the delete rule).
    # auth: verified
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    is_admin = _header_str(x_verified_is_admin) == "true"
    target_owner = _access_target_owner(workflow_name, caller, owner, is_admin)
    entries = workflow_access_entries(workflow_name, target_owner)
    if not entries:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )
    if any(str(e.get("scope", "user")) == "system" for e in entries) and not is_admin:
        raise_typed(
            PlatformError.auth(
                "admin_required", "Only admins may re-scope system workflows"
            )
        )
    guard = resource_caller(
        request,
        user_id=caller,
        project_id=_header_str(x_verified_project_id) or "",
        is_admin=is_admin,
    )
    access = checked_access(body.access, guard, error=PlatformError.workflows)
    updated = set_workflow_access(workflow_name, target_owner, access)
    from .audit_events import emit_audit

    emit_audit(
        "workflow.access_changed",
        project_id=_header_str(x_verified_project_id) or "",
        details={
            "workflow_name": workflow_name,
            "owner": target_owner,
            "access": access,
            "by": caller,
        },
    )
    return {
        "status": "ok",
        "workflow_name": workflow_name,
        "owner": target_owner,
        "access": access,
        "updated": len(updated),
    }


def _check_workflow_quota(project_id: str) -> None:
    """Best-effort governance quota check (no-op when inactive)."""
    try:
        from .governance import check_quota

        check_quota(project_id, "workflows")
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise
        # governance not active


def _resolved_upload_name(filename: str, workflow_name: str | None) -> str:
    """Validate filename/name form inputs and return the final name."""
    if not filename:
        raise_typed(PlatformError.workflows("bad_request", "Filename required"))
    resolved = workflow_name or Path(filename).stem
    if not resolved:
        raise_typed(PlatformError.workflows("bad_request", "workflow_name is required"))
    refusal = workflow_name_refusal(resolved)
    if refusal:
        raise_typed(PlatformError.workflows("bad_request", refusal))
    return resolved


def _load_extracted_workflow_json(dest: str) -> tuple[str, dict[str, Any]]:
    """Read version + payload from the extracted workflow.json."""
    workflow_json_path = os.path.join(dest, "workflow.json")
    try:
        with open(workflow_json_path, encoding="utf-8") as handle:
            workflow_json = json.load(handle)
    except Exception as exc:
        raise_typed(
            PlatformError.workflows(
                "failure", f"Failed to load extracted workflow.json: {exc}"
            ),
            cause=exc,
        )
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
        raise_typed(
            PlatformError.workflows(
                "failure", str(result.get("message", "Workflow registration failed"))
            ),
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


def _upload_message(workflow_name: str, notice: str | None) -> str:
    message = f"Workflow '{workflow_name}' uploaded and registered"
    return f"{message}; {notice}" if notice else message


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
    x_verified_workflow_author: str | None = Header(None),
    x_verified_workflow_author_hmac: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    _enforce_workflow_upload_role(
        scope=scope,
        actor=_header_str(x_user_id),
        project_id=_header_str(x_verified_project_id),
        is_admin=_header_str(x_verified_is_admin) == "true",
        presented_author=x_verified_workflow_author,
        presented_author_hmac=x_verified_workflow_author_hmac,
    )
    resolved_user_id = _verified_workflow_user(user_id, x_user_id)

    project_id = x_verified_project_id or ""
    _check_workflow_quota(project_id)
    resolved_workflow_name = _resolved_upload_name(file.filename or "", workflow_name)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
    await _spool_upload(file, temp_path)

    ok, dest, notice = handle_upload_workflow(
        resolved_user_id, temp_path, resolved_workflow_name
    )
    if not ok:
        _raise_upload_error(notice or "Workflow upload failed")

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
    from .audit_events import emit_audit

    emit_audit(
        "workflow_upload",
        actor=resolved_user_id,
        project_id=project_id,
        outcome="success",
        scope=scope,
        details={"workflow": resolved_workflow_name, "version": version},
    )
    return WorkflowUploadResponse(
        status="ok",
        workflow_name=resolved_workflow_name,
        accepted=True,
        source_path=dest,
        message=_upload_message(resolved_workflow_name, notice),
    )


@router.post("/{workflow_name}/trigger")
async def trigger_workflow(
    workflow_name: str,
    request: WorkflowTriggerRequest,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    memberships: frozenset[str] = Depends(get_verified_memberships),
):
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    user_id = _verified_workflow_user(request.user_id or "", x_user_id)
    params = _verified_workflow_params(
        request.resolved_params(),
        x_verified_project_id,
        memberships,
        x_verified_is_admin,
    )
    # ADR-074 D-8: owner-scoped, the same resolution the run below uses.
    is_admin = _header_str(x_verified_is_admin) == "true"
    entry = resolve_workflow_for_caller(
        workflow_name,
        user_id,
        is_admin=is_admin,
        caller_project=_header_str(x_verified_project_id),
    )
    if entry is None:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
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
        caller_is_admin=is_admin,
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
    from ..jobs.workflow_run import cancel_workflow_execution
    from ..server.step_reaper import durable_job_store

    # The sidecar's job store, else the durable runtime's (peer review F2).
    store = durable_job_store()
    record, job = resolve_execution_context(execution_id, store=store)
    enforce_http_execution_access(
        identity,
        record=record,
        job=job,
        execution_id=execution_id,
    )

    linked_job = job
    if record is not None and linked_job is None:
        linked_job = _get_job_for_record(store, record)
    # The durable job, then the registry record whatever the lookup above
    # found: a dispatch may have registered it since (peer review F2).
    cancelled_record = cancel_workflow_execution(
        execution_id, store=store, job=linked_job
    )
    if cancelled_record is not None:
        record = cancelled_record

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
        raise_typed(
            PlatformError.workflows(
                "not_found",
                (f"Step '{step_name}' is not part of execution '{execution_id}'"),
            )
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
    raise_typed(PlatformError.workflows("conflict", detail))


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
        raise_typed(
            PlatformError.workflows(
                "not_found",
                (
                    f"Execution '{execution_id}' is not tracked by the in-process registry"
                ),
            )
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
        raise_typed(
            PlatformError.workflows(
                "not_found",
                f"Execution '{execution_id}' is not tracked by the registry",
            )
        )

    linked_job = _get_job_for_record(store, record) if job is None else job

    # ADR-034 Phase 2: conserve the HITL verdict (best-effort, never blocks).
    try:
        from umrs.a2l.recorder import get_recorder

        await get_recorder().record_decision(
            decided_by=approver,
            decision="gate_approved" if request.approved else "gate_rejected",
            rationale=request.comment or "",
            job_id=getattr(linked_job, "job_id", "") if linked_job else "",
            payload={
                "kind": "gate_verdict",
                "execution_id": execution_id,
                "step_name": step_name,
                "workflow_name": getattr(record, "workflow_name", ""),
            },
        )
    except Exception:
        logger.warning("a2l gate decision conservation failed", exc_info=True)

    return {"status": "ok", "execution": _execution_payload(updated, linked_job)}
