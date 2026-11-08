"""Workflow Registry REST API.

Provides:
- GET /umrs/workflows - List registered workflows with optional filters
- GET /umrs/workflows/{workflow_name}/export - The registered source as a zip
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
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import Draft202012Validator, validator_for
from pydantic import BaseModel, Field, model_validator
from referencing.exceptions import Unresolvable

from ..core.TaskCmdStatus import TaskCmd
from ..core.zmq_correlation import dumps_with_correlation_id
from ..jobs.models import JobStatus
from ..jobs.store import JobNotFoundError, JobStore
from ..server.file_handling import (
    MAX_UPLOAD_BYTES,
    UploadOutcome,
    upload_workflow_bundle,
    workflow_name_refusal,
)
from ..server.job_tracking_handler import DurableJobCommandBackend
from ..server.workflow_registry_handler import (
    ResolvedWorkflow,
    _resolve_workflow_catalog_entry,
    _workflow_run_context,
    adopt_workflow,
    commit_workflow_run,
    ensure_catalog_layout,
    handle_workflow_command,
    register_workflow,
    require_code,
    resolve_workflow_for_caller,
    resolve_workflow_version,
    set_trigger_version,
    set_workflow_access,
    unregister_workflow,
    workflow_access_entries,
)
from ..server.workflow_versions import (
    ERROR_HTTP_STATUS,
    WorkflowVersionError,
    requested_version,
    sort_versions_desc,
    split_version_ref,
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
from .workflow_export import ExportResult, ExportTooLarge, build_export_zip
from .workflow_models import (
    WorkflowCommitResponse,
    WorkflowDescribeResponse,
    WorkflowExecutionResponse,
    WorkflowListResponse,
    WorkflowPlanResponse,
    WorkflowResultsResponse,
    WorkflowTriggerResponse,
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
    version: str | None = Field(
        default=None,
        description=(
            "The workflow version to run (ADR-113 D-3); the default version "
            "when absent. `name@X` in the path names one too."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
            "ADR-112 B2: base64 of the signed intent payload (from "
            "POST .../intent-proposal), for a run under your own HPC account"
        ),
    )
    signature: str | None = Field(
        default=None, description="The intent's armored SSH signature"
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


class WorkflowPlanRequest(BaseModel):
    """``POST /umrs/workflows/{name}/plan`` body (ADR-105 D-4): the trigger's
    inputs; nothing runs and nothing is stored."""

    user_id: str | None = Field(
        default=None, min_length=1, description="Caller (MVP mode only)"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Runtime parameters"
    )
    input_files: list[str] | None = Field(
        default=None, description="Optional input file keys"
    )
    project_id: str | None = Field(
        default=None,
        description="Project context (MVP mode only; strict uses verified headers)",
    )
    version: str | None = Field(
        default=None, description="The workflow version to plan (ADR-113 D-3)"
    )

    def as_trigger(self) -> WorkflowTriggerRequest:
        return WorkflowTriggerRequest(
            user_id=self.user_id,
            parameters=dict(self.parameters),
            input_files=self.input_files,
            project_id=self.project_id,
            version=self.version,
        )


class IntentProposalRequest(BaseModel):
    """``POST /umrs/workflows/{name}/intent-proposal`` (ADR-112 B2, plan §5
    B2.0 §10.3): the run to sign; nothing runs and nothing is stored."""

    user_id: str | None = Field(
        default=None, min_length=1, description="Caller (MVP mode only)"
    )
    project_id: str | None = Field(
        default=None, description="Project (strict uses the verified headers)"
    )
    version: str | None = Field(default=None, description="The version (ADR-113 D-3)")
    params: dict[str, Any] = Field(default_factory=dict, description="Runtime parameters")
    input_files: list[str] | None = Field(default=None, description="Optional input file keys")
    inputs: dict[str, list[dict[str, Any]]] | None = Field(
        default=None,
        description="Per step: objects you chose, {bucket, key, sha256, size} (listed)",
    )

    def as_trigger(self) -> WorkflowTriggerRequest:
        return WorkflowTriggerRequest(
            user_id=self.user_id,
            parameters=dict(self.params),
            input_files=self.input_files,
            project_id=self.project_id,
            version=self.version,
        )


class WorkflowCommitRequest(BaseModel):
    """``POST /umrs/workflows/{name}/commit`` body: a plan's ``signed`` dict
    and ``digest`` exactly as ``.../plan`` returned them."""

    signed: dict[str, Any] = Field(..., description="The plan's signed dict, unchanged")
    digest: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="The plan's digest (64 lowercase hex characters)",
    )
    auto: bool = Field(
        default=False,
        description=(
            "An unattended commit: refused unless the project's auto-commit "
            "threshold is set and the re-resolved max_gpu_hours is within it"
        ),
    )


class WorkflowUploadResponse(BaseModel):
    """Workflow upload response body."""

    status: str = Field(
        ..., description="'ok', or 'unchanged' when this version is already registered"
    )
    workflow_name: str = Field(..., description="Registered workflow name")
    accepted: bool = Field(..., description="Whether the upload was accepted")
    source_path: str = Field(..., description="Extracted workflow directory")
    message: str = Field(..., description="Human-readable result message")
    version: str = Field(default="", description="The registered version (ADR-113)")
    source_sha256: str = Field(
        default="", description="Digest of the version's files (ADR-113 D-2)"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Advice, e.g. a version-like suffix in the name (ADR-113 D-1)",
    )


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


def _describe_workflow_entry(
    entry: dict[str, Any], versions: tuple[str, ...] | list[str] = ()
) -> dict[str, Any]:
    """The describe body of one resolved version. ADR-113: the row and its
    own directory are one version, so catalog and workflow.json agree."""
    workflow_json = _read_entry_workflow_json(entry)
    return {
        "status": "ok",
        "contract_version": "1.0.0",
        "workflow_name": entry.get("name"),
        "version": entry.get("version") or workflow_json.get("version", ""),
        "versions": list(versions) or [str(entry.get("version") or "")],
        "source_sha256": entry.get("source_sha256"),
        "code": entry.get("code") or "available",
        "description": entry.get("description") or workflow_json.get("description", ""),
        "scope": entry.get("scope", "user"),
        "user_id": entry.get("user_id", ""),
        "parameter_schema": _workflow_parameter_schema(workflow_json),
        "steps": entry.get("steps", []),
        "requirements": entry.get("requirements", []),
        # W4: declared event subscriptions (additive; empty by default)
        "triggers": workflow_json.get("triggers", []),
    }


#: Keys the platform puts into the run parameters itself: ``input_files``
#: (from the trigger body) and ``project_id`` (body, or the verified header).
#: A workflow schema that does not declare them never sees them, so an
#: ``additionalProperties: false`` schema keeps accepting the runs it did
#: before full validation (ADR-105 D-3; the r4 pre-check mirrors this rule).
#: ``mcp_tools`` is what ``r4 workflow run --mcp-tool`` injects.
_PLATFORM_INJECTED_PARAMS = frozenset({"input_files", "project_id", "mcp_tools"})
#: At most this many schema violations go back in one refusal.
_MAX_SCHEMA_ERRORS = 20

#: Regex keywords are not evaluated on the server: workflow authors are
#: ordinary users and Python ``re`` holds the GIL, so one backtracking
#: ``pattern`` would stall the sidecar for everyone (ReDoS; review of
#: ADR-105 S1). A level with ``patternProperties`` also loses its
#: ``additionalProperties`` / ``unevaluatedProperties`` check, which depends
#: on the patterns, so no valid run is refused for it.
_REGEX_KEYWORDS = ("pattern", "patternProperties")
_PATTERN_DEPENDENT = ("additionalProperties", "unevaluatedProperties")
#: Keywords whose value is a map of name -> subschema.
_SCHEMA_MAPS = ("properties", "$defs", "definitions", "dependentSchemas")
#: Keywords whose value is a subschema or a list of subschemas.
_SCHEMA_SLOTS = (
    "items",
    "prefixItems",
    "additionalItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "allOf",
    "anyOf",
    "oneOf",
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "propertyNames",
)
_SCALAR_PARSERS: dict[str, Any] = {
    "integer": int,
    "number": float,
    "boolean": lambda text: {"true": True, "false": False}[text.strip().lower()],
}


def _regex_free(node: Any) -> Any:
    """A copy of a schema (or list of schemas) without regex keywords."""
    if isinstance(node, list):
        return [_regex_free(item) for item in node]
    if not isinstance(node, dict):
        return node
    return _regex_free_children(_without_regex_keywords(node))


def _without_regex_keywords(node: dict[str, Any]) -> dict[str, Any]:
    drop = set(_REGEX_KEYWORDS)
    if "patternProperties" in node:
        drop |= set(_PATTERN_DEPENDENT)
    return {key: value for key, value in node.items() if key not in drop}


def _regex_free_children(out: dict[str, Any]) -> dict[str, Any]:
    for key in _SCHEMA_MAPS:
        if isinstance(out.get(key), dict):
            out[key] = {name: _regex_free(sub) for name, sub in out[key].items()}
    for key in _SCHEMA_SLOTS:
        if key in out:
            out[key] = _regex_free(out[key])
    return out


def _coerce_scalar(value: Any, spec: Any) -> Any:
    """A string parsed as the scalar type *spec* declares (else *value*)."""
    if not isinstance(value, str) or not isinstance(spec, dict):
        return value
    declared = spec.get("type")
    for name in declared if isinstance(declared, list) else [declared]:
        parser = _SCALAR_PARSERS.get(name)
        try:
            return parser(value) if parser else value
        except (ValueError, KeyError):
            continue
    return value


def _coerced_view(params: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Top-level string values coerced to their declared scalar type.

    ``r4 workflow run --param k=v`` and the TUI's trigger line send every
    value as text; before D-3 only ``required`` was checked, so those runs
    passed. Validation sees the coerced copy; the run gets *params* unchanged.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return params
    return {
        key: _coerce_scalar(value, properties.get(key)) for key, value in params.items()
    }


def _schema_declared_names(schema: dict[str, Any]) -> set[str]:
    """Top-level parameter names a schema declares (properties or required)."""
    properties = schema.get("properties")
    required = schema.get("required")
    names = set(properties) if isinstance(properties, dict) else set()
    if isinstance(required, list):
        names |= {name for name in required if isinstance(name, str)}
    return names


def _params_validation_view(
    params: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    """*params* without the platform-injected keys the schema does not declare."""
    declared = _schema_declared_names(schema)
    return {
        key: value
        for key, value in params.items()
        if key in declared or key not in _PLATFORM_INJECTED_PARAMS
    }


def _missing_required_params(
    params: dict[str, Any], schema: dict[str, Any]
) -> list[str]:
    """Top-level ``required`` names absent from *params* (the pre-D-3 check)."""
    if schema.get("type") != "object":
        return []
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [name for name in required if isinstance(name, str) and name not in params]


def _schema_error_path(error: ValidationError) -> str:
    """RFC 6901 JSON Pointer to the offending value ("" is the whole object)."""
    return "".join(
        "/" + str(part).replace("~", "~0").replace("/", "~1")
        for part in error.absolute_path
    )


def _schema_violations(
    params: dict[str, Any], schema: dict[str, Any]
) -> list[dict[str, str]] | None:
    """Up to ``_MAX_SCHEMA_ERRORS`` ``{path, message}`` rows; ``None`` when the
    workflow's own schema cannot be used (invalid, or an unresolvable $ref)."""
    validator_cls = validator_for(schema, default=Draft202012Validator)
    try:
        validator_cls.check_schema(schema)
        checked = _regex_free(schema)
        view = _coerced_view(params, checked)
        found = list(
            islice(validator_cls(checked).iter_errors(view), _MAX_SCHEMA_ERRORS)
        )
    except (SchemaError, Unresolvable, RecursionError) as exc:
        logger.warning(
            "Workflow parameter_schema is not valid JSON Schema (%s); "
            "falling back to the required-keys check",
            getattr(exc, "message", exc),
        )
        return None
    rows = [{"path": _schema_error_path(err), "message": err.message} for err in found]
    return sorted(rows, key=lambda row: (row["path"], row["message"]))


def _parameters_invalid_message(
    missing: list[str], errors: list[dict[str, str]]
) -> str:
    if missing:
        return f"Missing required workflow parameters: {', '.join(missing)}"
    first = errors[0]
    where = f" at '{first['path']}'" if first["path"] else ""
    return f"Workflow parameters do not match the workflow schema{where}: {first['message']}"


def _validate_params_against_schema(
    params: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Full JSON Schema validation of the run parameters (ADR-105 D-3).

    The refusal is the typed 400 ``workflow_parameters_invalid`` with
    ``missing`` (top-level required names, the pre-D-3 contract) and
    ``errors`` (``{path, message}`` rows). A workflow whose own schema is not
    valid JSON Schema is not blocked: the old required-keys check applies.
    """
    if not isinstance(schema, dict):
        return
    view = _params_validation_view(params, schema)
    errors = _schema_violations(view, schema)
    if errors is None:
        errors = []
        missing = _missing_required_params(params, schema)
    else:
        missing = _missing_required_params(view, schema)
    if not (missing or errors):
        return
    raise_typed(
        PlatformError.workflows2(
            "bad_request",
            {
                "error": "validation_error",
                "code": "workflow_parameters_invalid",
                "message": _parameters_invalid_message(missing, errors),
                "missing": missing,
                "errors": errors,
            },
        ),
    )


def _raise_version_error(exc: WorkflowVersionError) -> NoReturn:
    """A version refusal (ADR-113) as its typed HTTP error."""
    raise_typed(
        PlatformError(code=exc.code, message=str(exc), http_status=exc.http_status),
        cause=exc,
    )


def _resolve_visible(
    workflow_name: str,
    caller: str,
    *,
    is_admin: bool,
    caller_project: str | None,
    version: str | None = None,
) -> ResolvedWorkflow:
    """The version a caller may see (ADR-074 D-8, ADR-080, ADR-113 D-3), or
    the typed 404 (``workflows.not_found``; an unknown version of a known
    name is ``workflows.version_not_found`` listing the versions)."""
    try:
        resolved = resolve_workflow_version(
            workflow_name,
            caller,
            is_admin=is_admin,
            caller_project=caller_project,
            version=version,
        )
    except WorkflowVersionError as exc:
        _raise_version_error(exc)
    if resolved is None:
        name = split_version_ref(workflow_name)[0]
        raise_typed(
            PlatformError.workflows("not_found", f"Workflow '{name}' not found")
        )
    return resolved


def _typed_handler_error(data: dict[str, Any]) -> None:
    """A handler error carrying an ADR-113 version code or an ADR-112
    ``hpcaccess.*`` code: its typed error."""
    code = data.get("code")
    if isinstance(code, str) and code in ERROR_HTTP_STATUS:
        _raise_version_error(
            WorkflowVersionError(code, str(data.get("message") or code))
        )
    _raise_hpcaccess(code, data)


def _raise_hpcaccess(code: Any, data: dict[str, Any]) -> None:
    from r4_contracts.errors import HPCACCESS_ERRORS

    if not isinstance(code, str) or code not in HPCACCESS_ERRORS:
        return
    detail = {k: data[k] for k in ("site", "reason") if isinstance(data.get(k), str)}
    raise_typed(
        PlatformError.hpcaccess(
            code.split(".", 1)[1], str(data.get("message") or code), **detail
        )
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
    _typed_handler_error(data)

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
        _pinned_from_spec(execution, job)
    return execution


def _pinned_from_spec(execution: dict[str, Any], job: Any) -> None:
    """ADR-113 D-4: a record without the pinned version takes the job's."""
    spec = getattr(job, "spec", None) or {}
    for key in ("workflow_version", "source_sha256"):
        if not execution.get(key) and spec.get(key):
            execution[key] = spec[key]


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


@router.get(
    "",
    operation_id="workflow_list",
    response_model=WorkflowListResponse,
    response_model_exclude_unset=True,
)
async def list_workflows(
    scope: str | None = Query(None, description="Optional workflow scope filter"),
    name: str | None = Query(None, description="Optional workflow-name filter"),
    user_id: str | None = Query(None, description="Optional owner filter"),
    all_users: bool = Query(
        False,
        description="Admin-only: list the full catalog across owners",
    ),
    all_versions: bool = Query(
        False,
        description=(
            "One row per registered version (ADR-113); by default one row per "
            "name with the default `version` and every `versions`"
        ),
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
    if all_versions:
        payload["all_versions"] = True
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
    version: str | None = Field(
        default=None,
        description=(
            "The workflow version the definition pins (ADR-113 D-4); the "
            "default version at save time when absent"
        ),
    )


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


def _definition_payload(record_dict: dict[str, Any]) -> dict[str, Any]:
    """The contract record plus the version pin (ADR-113 D-4)."""
    from r4_contracts.workflows import RunDefinitionRecord

    payload = RunDefinitionRecord.from_dict(record_dict).to_dict()
    for key in ("workflow_version", "source_sha256"):
        payload[key] = record_dict.get(key)
    return payload


def _definition_target(
    request: RunDefinitionRequest, caller: str, is_admin: bool, project: str | None
) -> dict[str, Any]:
    """The workflow version a run definition pins: run definitions must
    reference a registered workflow, and the version resolved now (the one
    asked for, else the default) is the one the definition replays."""
    try:
        resolved = resolve_workflow_version(
            request.workflow_name,
            caller,
            is_admin=is_admin,
            caller_project=project,
            version=request.version,
        )
    except WorkflowVersionError as exc:
        _raise_version_error(exc)
    if resolved is None:
        raise_typed(
            PlatformError.workflows(
                "not_found",
                (
                    f"Workflow '{request.workflow_name}' not found; "
                    "run definitions must reference a registered workflow"
                ),
            )
        )
    return resolved.entry


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
    from ..server.run_definition_registry import save_definition

    caller = _header_str(x_user_id) or ""
    entry = _definition_target(
        request,
        caller,
        _header_str(x_verified_is_admin) == "true",
        _header_str(x_verified_project_id),
    )
    try:
        record_dict = save_definition(
            name=request.name,
            workflow_name=str(entry.get("name")),
            params=request.params,
            user_id=caller,
            project_id=_header_str(x_verified_project_id) or "",
            description=request.description,
            workflow_version=entry.get("version"),
            source_sha256=entry.get("source_sha256"),
        )
    except ValueError as exc:
        raise_typed(PlatformError.workflows("bad_request", str(exc)), cause=exc)
    return {"status": "ok", "definition": _definition_payload(record_dict)}


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


def _refuse_changed_definition(
    record: dict[str, Any], caller: str, is_admin: bool, project: str | None
) -> None:
    """A definition replays the files it pinned (ADR-113 D-4): when its
    version now has other files (re-uploaded after a delete), refuse."""
    pinned = record.get("source_sha256")
    if not pinned:
        return
    entry = _resolve_visible(
        str(record["workflow_name"]),
        caller,
        is_admin=is_admin,
        caller_project=project,
        version=record.get("workflow_version"),
    ).entry
    if entry.get("source_sha256") != pinned:
        _raise_version_error(
            WorkflowVersionError(
                "workflows.source_mismatch",
                f"Run definition '{record.get('name')}' pinned other files for "
                f"{record['workflow_name']}@{record.get('workflow_version')}",
            )
        )


@router.post("/run-definitions/{name}/trigger")
async def trigger_run_definition(
    name: str,
    scope_request: Request,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_r4_session_id: str | None = Header(None, alias="x-r4-session-id"),
    memberships: frozenset[str] = Depends(get_verified_memberships),
):
    """Replay one persisted (workflow_name, version, params) — exact same
    inputs, the version the definition pinned (ADR-113 D-4; a definition
    saved before ADR-113 runs the default version).

    No param overrides are accepted on purpose: a run definition is the
    reproducibility record; overriding would defeat it.
    """
    caller = _header_str(x_user_id) or ""
    is_admin = _header_str(x_verified_is_admin) == "true"
    record = _resolve_definition_for_caller(name, caller, is_admin)
    _refuse_changed_definition(
        record, caller, is_admin, _header_str(x_verified_project_id)
    )
    request = WorkflowTriggerRequest(
        user_id=str(record.get("user_id") or caller),
        params=dict(record.get("params") or {}),
        version=record.get("workflow_version") or None,
    )
    return await trigger_workflow(
        workflow_name=str(record["workflow_name"]),
        request=request,
        x_user_id=x_user_id,
        x_verified_project_id=x_verified_project_id,
        x_verified_is_admin=x_verified_is_admin,
        x_verified_hmac=x_verified_hmac,
        x_r4_session_id=x_r4_session_id,
        memberships=memberships,
        scope_request=scope_request,
    )


def _trigger_entry(resolved: ResolvedWorkflow, caller: str, is_admin: bool, project):
    """The row whose triggers fire (ADR-113 D-7): the owner's pin when it
    resolves, else the default."""
    pin = resolved.entry.get("trigger_version")
    if not pin or str(pin) == str(resolved.entry.get("version")):
        return resolved.entry
    pinned = resolve_workflow_for_caller(
        str(resolved.entry.get("name")),
        caller,
        is_admin=is_admin,
        caller_project=project,
        version=str(pin),
    )
    return pinned or resolved.entry


@router.get("/{workflow_name}/triggers")
async def list_workflow_triggers(
    workflow_name: str,
    version: str | None = Query(None, description="One version's triggers"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """W4: the workflow's declared event trigger subscriptions: those of the
    version that fires (the owner's ``trigger_version`` pin, else the
    default; ADR-113 D-7), or of ``?version=``."""
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    is_admin = _header_str(x_verified_is_admin) == "true"
    project = _header_str(x_verified_project_id)
    resolved = _resolve_visible(
        workflow_name,
        caller,
        is_admin=is_admin,
        caller_project=project,
        version=version,
    )
    entry = (
        resolved.entry
        if version or "@" in workflow_name
        else _trigger_entry(resolved, caller, is_admin, project)
    )
    workflow_json = _read_entry_workflow_json(entry)
    return {
        "status": "ok",
        "workflow_name": entry.get("name"),
        "version": entry.get("version"),
        "trigger_version": entry.get("trigger_version"),
        "triggers_disabled": bool(entry.get("triggers_disabled")),
        "adopted_as": entry.get("adopted_as"),
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


class WorkflowTriggerPin(BaseModel):
    """``PUT /umrs/workflows/{name}/triggers`` body (ADR-113 D-7)."""

    version: str | None = Field(
        default=None,
        description="The version whose ingest triggers fire; null = the default version",
    )


@router.put("/{workflow_name}/triggers")
async def put_workflow_triggers(
    workflow_name: str,
    body: WorkflowTriggerPin,
    owner: str | None = Query(None, description="Admin-only: another owner's workflow"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Pin (or unpin) the version whose ingest triggers fire for the caller's
    workflow (owner, or admin with ``?owner=``; audited). One (owner, name)
    fires once per event (ADR-113 D-7).
    # auth: verified
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    is_admin = _header_str(x_verified_is_admin) == "true"
    target_owner = _access_target_owner(workflow_name, caller, owner, is_admin)
    try:
        updated = set_trigger_version(workflow_name, target_owner, body.version)
    except WorkflowVersionError as exc:
        _raise_version_error(exc)
    if not updated:
        raise_typed(
            PlatformError.workflows(
                "not_found", f"Workflow '{workflow_name}' not found"
            )
        )
    from .audit_events import emit_audit

    emit_audit(
        "workflow.trigger_version_changed",
        project_id=_header_str(x_verified_project_id) or "",
        details={
            "workflow_name": workflow_name,
            "owner": target_owner,
            "trigger_version": body.version,
            "by": caller,
        },
    )
    return {
        "status": "ok",
        "workflow_name": workflow_name,
        "owner": target_owner,
        "trigger_version": body.version,
        "updated": len(updated),
    }


class WorkflowAdoptRequest(BaseModel):
    """``POST /umrs/workflows/adopt`` body (ADR-113 D-8)."""

    from_name: str = Field(..., min_length=1, description="The name-versioned workflow")
    name: str = Field(..., min_length=1, description="The stable name to register as")
    version: str = Field(..., min_length=1, description="SemVer to register as")


@router.post("/adopt")
async def adopt_workflow_route(
    body: WorkflowAdoptRequest,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Re-register a name-versioned workflow's files as (name, version):
    access and triggers carried over, the old name's triggers disabled
    (ADR-113 D-8). Owner or admin; audited.
    # auth: verified
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    try:
        result = adopt_workflow(
            body.from_name,
            body.name,
            body.version,
            caller=caller,
            is_admin=_header_str(x_verified_is_admin) == "true",
        )
    except LookupError as exc:
        raise_typed(PlatformError.workflows("not_found", str(exc)), cause=exc)
    except PermissionError as exc:
        raise_typed(PlatformError.workflows("forbidden", str(exc)), cause=exc)
    except WorkflowVersionError as exc:
        _raise_version_error(exc)
    from .audit_events import emit_audit

    emit_audit(
        "workflow.adopted",
        actor=caller,
        project_id=_header_str(x_verified_project_id) or "",
        details=result,
    )
    return {"status": "ok", **result}


@router.get("/{workflow_name}/package")
async def get_workflow_package(
    workflow_name: str,
    version: str | None = Query(None, description="One version (ADR-113)"),
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
    empty list, never an invented listing. ``?version=`` (or ``name@X``)
    shows one version, else the default (ADR-113 D-3).
    """
    resolved = _visible_workflow_entry(
        workflow_name,
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        version,
    )
    return _package_payload(resolved.entry, resolved.versions)


def _visible_workflow_entry(
    workflow_name: str,
    x_user_id: str | None,
    x_verified_project_id: str | None,
    x_verified_is_admin: str | None,
    x_verified_hmac: str | None,
    version: str | None = None,
) -> ResolvedWorkflow:
    """The version ``/package`` and ``/export`` show the caller, or the
    typed 404: verified identity, then the describe resolution (own, system,
    an ADR-080 share for the verified project, another owner's for admins)."""
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    return _resolve_visible(
        workflow_name,
        _header_str(x_user_id) or "",
        is_admin=_header_str(x_verified_is_admin) == "true",
        caller_project=_header_str(x_verified_project_id),
        version=version,
    )


def _built_export(workflow_name: str, entry: dict[str, Any]) -> ExportResult:
    """Build the export archive; the size cap is a typed 413."""
    try:
        return build_export_zip(workflow_name, entry)
    except ExportTooLarge as exc:
        raise_typed(
            PlatformError(
                code="workflows.export_too_large",
                message=str(exc),
                http_status=413,
            ),
            cause=exc,
        )


_EXPORT_CHUNK = 1024 * 1024


def _opened_export(
    workflow_name: str, entry: dict[str, Any]
) -> tuple[ExportResult, Any]:
    """Build the archive, open it and unlink it at once (review fix
    2026-11-07): nothing is left in TMPDIR whatever happens to the response
    (bad or unsatisfiable Range, cancelled send); the open handle keeps the
    bytes alive until :func:`_stream_export` closes it."""
    result = _built_export(workflow_name, entry)
    try:
        handle = open(result.path, "rb")  # noqa: SIM115 - closed by _stream_export
    finally:
        os.unlink(result.path)
    return result, handle


def _stream_export(handle: Any) -> Iterator[bytes]:
    """The open (already unlinked) archive in chunks; always closed."""
    try:
        yield from iter(lambda: handle.read(_EXPORT_CHUNK), b"")
    finally:
        handle.close()


def _export_response(workflow_name: str, handle: Any) -> StreamingResponse:
    """The whole archive, never a range: ``application/zip`` with the
    ``FileResponse`` style ``Content-Disposition`` and an exact length."""
    filename = f"{workflow_name}.zip"
    quoted = quote(filename)
    disposition = (
        f'attachment; filename="{filename}"'
        if quoted == filename
        else f"attachment; filename*=utf-8''{quoted}"
    )
    headers = {
        "Content-Disposition": disposition,
        "Content-Length": str(os.fstat(handle.fileno()).st_size),
    }
    return StreamingResponse(
        _stream_export(handle), media_type="application/zip", headers=headers
    )


@router.get(
    "/{workflow_name}/export",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/zip": {}}}},
)
async def export_workflow(
    workflow_name: str,
    version: str | None = Query(None, description="One version (ADR-113)"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """The workflow's registered source as a zip (2026-11-07).
    # auth: verified

    Whoever may open ``/package`` may export (same identity gate, same
    resolution: project members get a ``project:`` share through the
    verified project); everyone else gets the same typed 404. The zip holds
    the source tree (symlinks, hard links and secrets-shaped paths skipped
    and listed), ``workflow.json`` with a relative ``source.path`` and
    ``EXPORT.json`` (sha256 per file); container images are not included. Above
    ``MAX_EXPORT_BYTES`` it is a typed 413. Audited as ``workflow.exported``.
    The temp archive is unlinked before the response starts and streamed
    whole (no Range support), so no request outcome leaves it on disk.
    ``?version=`` (or ``name@X``) exports one version, else the default.
    """
    entry = _visible_workflow_entry(
        workflow_name,
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        version,
    ).entry
    name = str(entry.get("name") or split_version_ref(workflow_name)[0])
    result, handle = await asyncio.to_thread(_opened_export, name, entry)
    caller = _header_str(x_user_id) or ""
    project_id = _header_str(x_verified_project_id) or ""
    from .audit_events import emit_audit

    emit_audit(
        "workflow.exported",
        actor=caller,
        project_id=project_id,
        details={
            "workflow": name,
            "version": entry.get("version"),
            "owner": str(entry.get("user_id", "")),
            "caller": caller,
            "project": project_id,
            "files": result.files,
            "bytes": result.bytes,
            "excluded": result.excluded,
        },
    )
    return _export_response(name, handle)


def _package_payload(
    entry: dict[str, Any], versions: tuple[str, ...] | list[str] = ()
) -> dict[str, Any]:
    """The ``/package`` body for a resolved catalog *entry* (split out of the
    route for the complexity ratchet, 2026-10-02): the definition and the
    files of that version's own directory (ADR-113)."""
    payload = _describe_workflow_entry(entry, versions)
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


def _deletable_entry(
    workflow_name: str, caller: str, is_admin: bool
) -> tuple[str, str]:
    """(owner, scope) of the registration the caller may delete, or the
    typed 404/403: the caller must OWN it unless an admin; system scope is
    admin-only."""
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
    return owner, scope


def _refuse_nothing_deleted(name: str, version: str | None, owner: str, scope: str):
    """No row matched: an unknown version names the versions there are."""
    if version:
        from ..server.workflow_registry_handler import workflow_access_entries as rows

        available = sort_versions_desc(
            e.get("version")
            for e in rows(name, owner)
            if e.get("scope", "user") == scope
        )
        _raise_version_error(
            WorkflowVersionError(
                "workflows.version_not_found",
                f"Workflow '{name}' has no version {version}; available "
                f"versions: {', '.join(available)}",
            )
        )
    raise_typed(PlatformError.workflows("not_found", f"Workflow '{name}' not found"))


@router.delete("/{workflow_name}")
async def delete_workflow(
    workflow_name: str,
    version: str | None = Query(
        None, description="Delete one version (ADR-113 D-6); all versions when absent"
    ),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Delete a registered workflow: one version (``?version=`` or
    ``name@X``), else every version of the name (ADR-113 D-6). A version's
    files go once no queued or running run references it.

    Authorization: caller must OWN the registration, unless the verified
    caller is an admin. System-scope workflows always require admin — a
    shared workflow must never be deletable by an arbitrary project member.
    Governance ledger (ownership + quota) is updated best-effort.
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _header_str(x_user_id) or ""
    try:
        name, version = requested_version(workflow_name, version)
    except WorkflowVersionError as exc:
        _raise_version_error(exc)
    owner, scope = _deletable_entry(
        name, caller, _header_str(x_verified_is_admin) == "true"
    )
    result = unregister_workflow(name, user_id=owner, scope=scope, version=version)
    if result["removed"] == 0:
        _refuse_nothing_deleted(name, version, owner, scope)

    from .governance import record_resource_deleted

    record_resource_deleted(
        "workflows", name, project_id=_header_str(x_verified_project_id) or ""
    )
    return {
        "status": "ok",
        "workflow_name": name,
        "removed": result["removed"],
        "versions": result["versions"],
        "files_deferred": result["deferred"],
        "message": f"Workflow '{name}'"
        + (f" version {version}" if version else "")
        + " deleted",
    }


@router.get(
    "/{workflow_name}",
    operation_id="workflow_describe",
    response_model=WorkflowDescribeResponse,
    response_model_exclude_unset=True,
)
async def describe_workflow(
    workflow_name: str,
    version: str | None = Query(
        None, description="One version (ADR-113 D-3); the default when absent"
    ),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
):
    """Return declared parameter schema and metadata for a workflow version.

    Owner-scoped (ADR-074 D-8): the caller's own or a system workflow;
    another owner's only for admins. ``?version=`` (or ``name@X``) names a
    version; an unknown one is ``workflows.version_not_found`` listing the
    versions. ``versions`` lists every version, highest first.
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    resolved = _resolve_visible(
        workflow_name,
        _header_str(x_user_id) or "",
        is_admin=_header_str(x_verified_is_admin) == "true",
        caller_project=_header_str(x_verified_project_id),
        version=version,
    )
    return _describe_workflow_entry(resolved.entry, resolved.versions)


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
    outcome: UploadOutcome,
    resolved_user_id: str,
    workflow_json: dict[str, Any],
    scope: str,
    project_id: str,
) -> None:
    """Register the uploaded version in the catalog + governance ledger."""
    result = register_workflow(
        workflow_name=resolved_workflow_name,
        version=outcome.version,
        user_id=resolved_user_id,
        source_path=str(outcome.dest),
        workflow_json=workflow_json,
        scope=scope,
        description=workflow_json.get("description", ""),
        source_sha256=outcome.source_sha256,
    )
    if result.get("status") == "error":
        _typed_handler_error(result)
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


def _refuse_upload(outcome: UploadOutcome) -> NoReturn:
    """A typed upload refusal (ADR-113 version codes), else the legacy map."""
    if outcome.code:
        _raise_version_error(WorkflowVersionError(outcome.code, str(outcome.message)))
    _raise_upload_error(outcome.message or "Workflow upload failed")
    raise AssertionError("unreachable")  # _raise_upload_error always raises


def _upload_response(
    workflow_name: str, outcome: UploadOutcome, response: Response
) -> WorkflowUploadResponse:
    if outcome.unchanged:
        # ADR-113 D-2: the same bundle again is a no-op success.
        response.status_code = status.HTTP_200_OK
    return WorkflowUploadResponse(
        status="unchanged" if outcome.unchanged else "ok",
        workflow_name=workflow_name,
        accepted=True,
        source_path=str(outcome.dest),
        message=_upload_message(workflow_name, outcome.message),
        version=outcome.version,
        source_sha256=outcome.source_sha256,
        warnings=list(outcome.warnings),
    )


@router.post(
    "/upload",
    response_model=WorkflowUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_workflow(
    response: Response,
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
    """Upload one workflow version (ADR-113 D-1/D-2): ``workflow.json`` must
    carry a SemVer ``version`` (else ``workflows.invalid_version``); the
    version is stored once, immutably — the same bundle again answers 200
    ``unchanged``, other content under that version is 409
    ``workflows.version_exists``."""
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

    # D-8: a catalog from before ADR-113 moves to per-version dirs first.
    ensure_catalog_layout()
    outcome = upload_workflow_bundle(
        resolved_user_id, temp_path, resolved_workflow_name
    )
    if not outcome.ok:
        _refuse_upload(outcome)

    _version, workflow_json = _load_extracted_workflow_json(str(outcome.dest))
    _register_uploaded_workflow(
        resolved_workflow_name=resolved_workflow_name,
        outcome=outcome,
        resolved_user_id=resolved_user_id,
        workflow_json=workflow_json,
        scope=scope,
        project_id=project_id,
    )
    from .audit_events import emit_audit

    emit_audit(
        "workflow_upload",
        actor=resolved_user_id,
        project_id=project_id,
        outcome="unchanged" if outcome.unchanged else "success",
        scope=scope,
        details={
            "workflow": resolved_workflow_name,
            "version": outcome.version,
            "source_sha256": outcome.source_sha256,
        },
    )
    return _upload_response(resolved_workflow_name, outcome, response)


def _run_workflow_payload(
    workflow_name: str,
    params: dict[str, Any],
    identity_verified: bool,
    session_header: Any,
    version: str | None = None,
    signed: tuple[str | None, str | None] = (None, None),
) -> dict[str, Any]:
    """The RUN_WORKFLOW payload; a well-formed ``x-r4-session-id`` rides as
    ``_origin_session_id`` so the durable job carries ``spec.origin.session_id``
    (ADR-105 D-5). Absent or malformed: the payload is unchanged. *version*
    is the version the trigger resolved (ADR-113 D-3)."""
    from ..jobs.job_events import valid_session_id

    payload: dict[str, Any] = {
        "workflow_name": workflow_name,
        "version": version,
        "params": params,
        "_identity_verified": identity_verified,
    }
    sid = valid_session_id(session_header)
    if sid is not None:
        payload["_origin_session_id"] = sid
    # ADR-112 B2: the signed intent and its signature, checked by the handler
    # (the same rules as a ZMQ RUN_WORKFLOW that carries them).
    payload.update({k: v for k, v in zip(("intent", "signature"), signed) if v})
    return payload


@router.post(
    "/{workflow_name}/trigger",
    operation_id="workflow_run",
    response_model=WorkflowTriggerResponse,
    response_model_exclude_unset=True,
)
async def trigger_workflow(
    workflow_name: str,
    request: WorkflowTriggerRequest,
    scope_request: Request,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_r4_session_id: str | None = Header(None, alias="x-r4-session-id"),
    memberships: frozenset[str] = Depends(get_verified_memberships),
):
    """Start a run. Refused (typed 403 ``workflows.plan_required``) when the
    project requires plan -> commit (ADR-105 D-4); a work level's request
    needs its scope's ``workflow:<name>`` grant (ADR-108 D-3). A trigger that
    carries a signed intent (ADR-112 B2) is not refused ``plan_required``:
    the person's signature over the exact run is the explicit approval a
    plan asks for, and the dispatch admits the intent or refuses it (plan
    §5 B2.0 §10.3, fixer amendment 2026-11-08; a plan cannot carry one).
    # auth: verified
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _WorkflowHeaders(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    resolved = _resolve_trigger(workflow_name, request, caller, memberships)
    _refuse_unscoped_trigger(scope_request, resolved.name, resolved)
    _refuse_hpc_user_work_trigger(workflow_name, resolved)
    if not resolved.signed[0]:
        _refuse_unplanned_trigger(workflow_name, resolved.project_id)
    return _dispatch_trigger(resolved, x_r4_session_id)


@router.post("/{workflow_name}/intent-proposal", operation_id="workflow_intent_proposal")
def propose_workflow_intent(
    workflow_name: str,
    request: IntentProposalRequest,
    scope_request: Request,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    memberships: frozenset[str] = Depends(get_verified_memberships),
):
    """The server's proposed intent for a run under your own HPC account
    (ADR-112 B2): ``{intent_b64, intent_id, site_profile_sha256, summary}``.
    Refused ``hpcaccess.intent_invalid`` where the run does not resolve to
    your own account, ``hpcaccess.gate_version`` where your gate or its
    site profile is not current. A plain ``def`` route: the live ``hello``
    and the input hashing block, so FastAPI runs it in its threadpool.
    # auth: verified
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _WorkflowHeaders(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    resolved = _resolve_trigger(workflow_name, request.as_trigger(), caller, memberships)
    _refuse_unscoped_trigger(scope_request, resolved.name, resolved)
    return _proposal(workflow_name, resolved, request.inputs)


def _proposal(
    workflow_name: str, resolved: _ResolvedTrigger, inputs: Any
) -> dict[str, Any]:
    from ..hpc_accounts.intents import IntentInvalid
    from ..hpc_accounts.site_profile import SiteProfileUnavailable
    from ..server.intent_proposal import GateStale
    from ..server.step_executor import StepExecutionError

    try:
        return _proposal_or_raise(workflow_name, resolved, inputs)
    except IntentInvalid as exc:
        raise_typed(
            PlatformError.hpcaccess("intent_invalid", str(exc), reason=exc.reason), cause=exc
        )
    except (GateStale, SiteProfileUnavailable) as exc:
        reason = getattr(exc, "reason", "site_profile_unavailable")
        raise_typed(
            PlatformError.hpcaccess("gate_version", f"{reason}: {exc}", reason=reason), cause=exc
        )
    except StepExecutionError as exc:
        raise_typed(_proposal_step_error(exc), cause=exc)


def _proposal_step_error(exc: Any) -> PlatformError:
    """A gate that refused or cannot be reached (``gate_refused`` /
    ``not_enrolled``), or a stage plan the run could not have (``intent_invalid``)."""
    from ..server.gated_step_lane import GateFailure

    if isinstance(exc, GateFailure):
        name = "not_enrolled" if exc.name == "not_enrolled" else "gate_refused"
        return PlatformError.hpcaccess(name, exc.message, gate=exc.name)
    return PlatformError.hpcaccess("intent_invalid", exc.message, reason="stage_plan")


def _proposal_or_raise(
    workflow_name: str, resolved: _ResolvedTrigger, inputs: Any
) -> dict[str, Any]:
    from ..hpc_accounts.intents import IntentInvalid
    from ..server.hpc_user_work import is_user_account
    from ..server.intent_proposal import propose

    try:
        source_path, workflow_json, _ = _workflow_run_context(
            workflow_name,
            resolved.user_id,
            source_path=str(resolved.entry.get("source_path") or ""),
        )
    except ValueError as exc:
        raise_typed(PlatformError.workflows("not_found", str(exc)), cause=exc)
    identity = _refuse_hpc_user_work(workflow_json, source_path, resolved)
    if not is_user_account(identity):
        raise IntentInvalid("not_user_account", "this run does not run under your own HPC account")
    return propose(resolved, workflow_json, source_path, identity, inputs)


def _refuse_unscoped_trigger(
    http_request: Request | None, workflow_name: str, resolved: _ResolvedTrigger
) -> None:
    """A scoped trigger needs ``workflow:<name>`` (or ``workflow:*``)."""
    if http_request is None:
        return
    from umrs.api.scope_tokens import require_grant, scope_of

    claims = scope_of(http_request, user=resolved.user_id, project=resolved.project_id)
    require_grant(claims, f"workflow:{workflow_name}")


# ---------------------------------------------------------------------------
# Plan -> commit (ADR-105 D-4): a plan is signed, never stored
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkflowHeaders:
    """The gateway-verified identity headers of one workflow request."""

    x_user_id: str | None
    x_verified_project_id: str | None
    x_verified_is_admin: str | None
    x_verified_hmac: str | None

    @property
    def is_admin(self) -> bool:
        return _header_str(self.x_verified_is_admin) == "true"

    def identity_verified(self) -> bool:
        """Wave-6 F1: verified only when presence AND (strict mode) the HMAC
        hold; legacy unsigned paths stay False, so the durable layer never
        uprates a forged tuple into ``gateway_verified`` truth."""
        user, project = (
            _header_str(self.x_user_id),
            _header_str(self.x_verified_project_id),
        )
        if not (user and project):
            return False
        if not _strict_auth_mode():
            return True
        from ._verified_headers import handshake_secret, verified_headers_trusted

        return verified_headers_trusted(
            strict=True,
            secret=handshake_secret(),
            presented_hmac=_header_str(self.x_verified_hmac),
            user_id=user,
            project_id=project,
            is_admin=self.is_admin,
        )


@dataclass(frozen=True)
class _ResolvedTrigger:
    """A trigger's verified caller, parameters and catalog entry."""

    user_id: str
    params: dict[str, Any]
    entry: dict[str, Any]
    is_admin: bool
    identity_verified: bool
    #: ADR-112 B2: the trigger body's signed intent and signature, if any.
    signed: tuple[str | None, str | None] = (None, None)

    @property
    def name(self) -> str:
        """The workflow's name (never ``name@X``)."""
        return str(self.entry.get("name") or "")

    @property
    def version(self) -> str | None:
        """The resolved version (ADR-113 D-3)."""
        value = self.entry.get("version")
        return None if value in (None, "") else str(value)

    @property
    def project_id(self) -> str:
        """The project the durable job is recorded under (``default`` when
        none), as ``workflow_registry_handler._workflow_project_id``."""
        value = self.params.get("project_id")
        return value.strip() if isinstance(value, str) and value.strip() else "default"

    @property
    def verified_project(self) -> str | None:
        """The run's project when the identity verified (the header project,
        or a declared one checked against the verified memberships), else
        None; never ``default`` (as ``_verified_caller_project``)."""
        if not self.identity_verified or self.project_id == "default":
            return None
        return self.project_id


def _resolve_trigger(
    workflow_name: str,
    request: WorkflowTriggerRequest,
    caller: _WorkflowHeaders,
    memberships: frozenset[str] | None,
) -> _ResolvedTrigger:
    """Identity, project authorization, owner-scoped resolution of the
    version asked for (else the default, ADR-113 D-3) and the full
    parameter-schema validation: the one path trigger, plan and commit share.
    Each route also calls ``_require_verified_workflow_identity`` itself, so
    the H-1 scope guard sees the resolver in the handler."""
    _require_verified_workflow_identity(
        caller.x_user_id,
        caller.x_verified_project_id,
        caller.x_verified_is_admin,
        caller.x_verified_hmac,
    )
    user_id = _verified_workflow_user(request.user_id or "", caller.x_user_id)
    params = _verified_workflow_params(
        request.resolved_params(),
        caller.x_verified_project_id,
        memberships,
        caller.x_verified_is_admin,
    )
    # ADR-074 D-8: owner-scoped, the same resolution the run uses.
    entry = _resolve_visible(
        workflow_name,
        user_id,
        is_admin=caller.is_admin,
        caller_project=_header_str(caller.x_verified_project_id),
        version=request.version,
    ).entry
    try:
        require_code(entry)
    except WorkflowVersionError as exc:
        _raise_version_error(exc)
    schema = _describe_workflow_entry(entry).get("parameter_schema", {})
    _validate_params_against_schema(params, schema)
    return _ResolvedTrigger(
        user_id,
        params,
        entry,
        caller.is_admin,
        caller.identity_verified(),
        (request.intent, request.signature),
    )


def _dispatch_trigger(
    resolved: _ResolvedTrigger,
    session_header: Any,
    *,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """RUN_WORKFLOW through the registry handler, for exactly the version
    resolved (ADR-113 D-3); a plan commit carries its plan id as the durable
    job's idempotency key."""
    store = get_optional_job_store()
    job_backend = DurableJobCommandBackend(store) if store is not None else None
    payload = json.dumps(
        _run_workflow_payload(
            resolved.name,
            resolved.params,
            resolved.identity_verified,
            session_header,
            resolved.version,
            resolved.signed,
        )
    )
    # Wave-6 F1: the durable trust bridge reflects the verified flag, never
    # hardcoded True.
    if idempotency_key is None:
        raw = handle_workflow_command(
            TaskCmd.RUN_WORKFLOW,
            resolved.user_id,
            payload,
            job_backend=job_backend,
            trusted_identity=resolved.identity_verified,
            caller_is_admin=resolved.is_admin,
        )
        return _parse_command_response(raw, action="trigger")
    raw = commit_workflow_run(
        resolved.user_id,
        payload,
        job_backend,
        trusted_identity=resolved.identity_verified,
        caller_is_admin=resolved.is_admin,
        idempotency_key=idempotency_key,
    )
    return _parse_command_response(raw, action="commit")


def _plan_refusal(name: str, http_status: int, message: str) -> NoReturn:
    """A typed ``workflows.<name>`` refusal of the plan/commit family."""
    raise_typed(
        PlatformError(
            code=f"workflows.{name}", message=message, http_status=http_status
        )
    )


def _project_plan_policy(project_id: str) -> Any:
    """The project's plan policy over the job store's ``umrs_db`` pool; an
    unreadable policy is a typed 503 (fail closed)."""
    from ..workflow_plans import (
        POLICY_UNAVAILABLE_MESSAGE,
        PlanPolicyUnavailable,
        read_project_plan_policy,
    )

    pool = getattr(get_optional_job_store(), "db_pool", None)
    try:
        return read_project_plan_policy(pool, project_id)
    except PlanPolicyUnavailable:
        # read_project_plan_policy logged the database error; the reply is generic.
        _plan_refusal("policy_unavailable", 503, POLICY_UNAVAILABLE_MESSAGE)


def _refuse_unplanned_trigger(workflow_name: str, project_id: str) -> None:
    from ..workflow_plans import plan_required_message

    if _project_plan_policy(project_id).require_plan:
        _plan_refusal(
            "plan_required", 403, plan_required_message(project_id, workflow_name)
        )


def _quote_resolved(workflow_name: str, resolved: _ResolvedTrigger) -> _Quoted:
    """The run's quote; a workflow this site cannot run is a typed 409."""
    from ..workflow_plans import quote_workflow

    try:
        source_path, workflow_json, _ = _workflow_run_context(
            workflow_name,
            resolved.user_id,
            source_path=str(resolved.entry.get("source_path") or ""),
        )
    except ValueError as exc:
        raise_typed(PlatformError.workflows("not_found", str(exc)), cause=exc)
    quote = quote_workflow(workflow_json, source_path)
    if quote.refusal:
        _plan_refusal("lane_unavailable", 409, quote.refusal)
    identity = _refuse_hpc_user_work(workflow_json, source_path, resolved)
    from ..server.hpc_user_work import is_user_account

    return _Quoted(quote, is_user_account(identity))


@dataclass(frozen=True)
class _Quoted:
    """A run's quote, and whether it needs a signed intent (ADR-112 B2: it
    runs under the caller's own HPC account; a commit cannot start it)."""

    quote: Any
    requires_intent: bool


def _refuse_hpc_user_work(
    workflow_json: dict[str, Any], source_path: str, resolved: _ResolvedTrigger
) -> Any:
    """Typed 403 ``hpcaccess.*`` when the run would use this site's HPC lane
    and the site's ``hpc_access`` does not admit the caller's verified
    project (ADR-112 A.3.4). Uses the verified project, never the claimed
    ``params.project_id``; the admin bit is the caller's user record, the
    same source the runner reads, so an admitted run is not refused later.
    Returns the lane's identity (None: the run does not use the HPC lane);
    a ``user_account`` one needs a signed intent (B2, checked at dispatch)."""
    from ..orchestrator.hpc_identity import HpcAccessRefused, platform_error
    from ..server.hpc_user_work import workflow_user_work_identity

    try:
        return workflow_user_work_identity(
            workflow_json,
            source_path,
            user_id=resolved.user_id,
            project_id=resolved.verified_project,
        )
    except HpcAccessRefused as refused:
        raise_typed(platform_error(refused), cause=refused)


def _refuse_hpc_user_work_trigger(
    workflow_name: str, resolved: _ResolvedTrigger
) -> None:
    """:func:`_refuse_hpc_user_work` for a trigger, before it is accepted. A
    workflow that does not load is left to the dispatch, which reports it."""
    try:
        source_path, workflow_json, _ = _workflow_run_context(
            workflow_name,
            resolved.user_id,
            source_path=str(resolved.entry.get("source_path") or ""),
        )
    except ValueError:
        return
    _refuse_hpc_user_work(workflow_json, source_path, resolved)


_PLATFORM_PARAM_KEYS = ("input_files", "project_id")


def _plan_fields(resolved: _ResolvedTrigger, quote: Any) -> dict[str, Any]:
    """The values a plan signs (``workflow_plans.SIGNED_FIELDS`` minus
    ``expires_at``): the resolved version and the digest of its files
    (ADR-113 D-4)."""
    params = resolved.params
    return {
        "project_id": resolved.project_id,
        "user_id": resolved.user_id,
        "workflow_name": resolved.name,
        "version": resolved.entry.get("version"),
        "source_sha256": resolved.entry.get("source_sha256"),
        "parameters": {
            k: v for k, v in params.items() if k not in _PLATFORM_PARAM_KEYS
        },
        "input_files": params.get("input_files"),
        "lane": quote.lane,
        "partition": quote.partition,
        "account": quote.account,
        "gpu_steps": quote.gpu_steps,
        "max_wall_seconds": quote.max_wall_seconds,
        "max_gpu_hours": quote.max_gpu_hours,
    }


def _plan_quota(user_id: str) -> dict[str, Any]:
    """The caller's agent units (reported; a workflow run is not one)."""
    from .agent_quota import agent_max_processes, count_active_agent_units

    pool = getattr(get_optional_job_store(), "db_pool", None)
    try:
        active: int | None = count_active_agent_units(pool, user_id=user_id)
    except Exception:  # a report, never a refusal
        logger.warning("agent unit count failed for %s", user_id, exc_info=True)
        active = None
    return {"active_units": active, "max_units": agent_max_processes()}


@router.post(
    "/{workflow_name}/plan",
    operation_id="workflow_plan",
    response_model=WorkflowPlanResponse,
    response_model_exclude_unset=True,
)
async def plan_workflow(
    workflow_name: str,
    request: WorkflowPlanRequest,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    memberships: frozenset[str] = Depends(get_verified_memberships),
):
    """Resolve and sign a run plan (ADR-105 D-4). The same identity,
    authorization, resolution and parameter validation as trigger; nothing
    is written (no job, no catalog change). Commit before ``expires_at``.
    # auth: verified
    """
    from ..workflow_plans import sign_plan

    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _WorkflowHeaders(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    resolved = _resolve_trigger(
        workflow_name, request.as_trigger(), caller, memberships
    )
    quoted = _quote_resolved(workflow_name, resolved)
    quote = quoted.quote
    policy = _project_plan_policy(resolved.project_id)
    signed, digest, plan_id = sign_plan(_plan_fields(resolved, quote))
    return {
        "status": "ok",
        "plan_id": plan_id,
        "digest": digest,
        "expires_at": signed["expires_at"],
        "signed": signed,
        "workflow_name": resolved.name,
        "version": signed["version"],
        "source_sha256": signed["source_sha256"],
        "steps": [step.to_dict() for step in quote.steps],
        "lane": quote.lane,
        "partition": quote.partition,
        "account": quote.account,
        "gpu_steps": quote.gpu_steps,
        "max_wall_seconds": quote.max_wall_seconds,
        "max_gpu_hours": signed["max_gpu_hours"],
        "quota": _plan_quota(resolved.user_id),
        "auto_commit_max_gpu_hours": policy.auto_commit_max_gpu_hours,
        "require_plan": policy.require_plan,
        "auto_commit_allowed": policy.auto_commit_allows(quote.max_gpu_hours),
        "requires_intent": quoted.requires_intent,
    }


def _verified_plan(workflow_name: str, body: WorkflowCommitRequest) -> dict[str, Any]:
    """The signed dict once its digest, expiry and workflow hold."""
    from ..workflow_plans import plan_expired, verify_plan

    signed = body.signed
    if not verify_plan(signed, body.digest):
        _plan_refusal("plan_invalid", 403, "plan signature does not verify")
    if plan_expired(signed):
        _plan_refusal(
            "plan_expired",
            409,
            f"plan expired at {signed.get('expires_at')}; plan again",
        )
    if signed.get("workflow_name") != split_version_ref(workflow_name)[0]:
        _plan_refusal("plan_mismatch", 403, "the plan was made for another workflow")
    return signed


def _trigger_from_plan(signed: dict[str, Any]) -> WorkflowTriggerRequest:
    """The trigger request a plan was made from (its signed inputs, for the
    version it signed: a newer upload meanwhile does not change it)."""
    parameters = dict(signed.get("parameters") or {})
    if signed.get("input_files") is not None:
        parameters["input_files"] = signed["input_files"]
    project = signed.get("project_id")
    version = signed.get("version")
    return WorkflowTriggerRequest(
        user_id=signed.get("user_id") or None,
        parameters=parameters,
        project_id=None if project in (None, "", "default") else project,
        version=str(version) if version not in (None, "") else None,
    )


def _refuse_foreign_plan(signed: dict[str, Any], resolved: _ResolvedTrigger) -> None:
    if signed.get("user_id") != resolved.user_id:
        _plan_refusal("plan_mismatch", 403, "the plan was made for another user")
    if signed.get("project_id") != resolved.project_id:
        _plan_refusal("plan_mismatch", 403, "the plan was made for another project")


def _refuse_stale_plan(signed: dict[str, Any], fresh: dict[str, Any]) -> None:
    from ..workflow_plans import stale_fields

    changed = stale_fields(signed, fresh)
    if changed:
        _plan_refusal(
            "plan_stale",
            409,
            f"the plan no longer matches ({', '.join(changed)} changed); plan again",
        )


def _refuse_auto_commit(project_id: str, max_gpu_hours: float) -> None:
    """Fail closed: an ``auto`` commit needs a threshold the quote is within."""
    policy = _project_plan_policy(project_id)
    if policy.auto_commit_allows(max_gpu_hours):
        return
    limit = policy.auto_commit_max_gpu_hours
    threshold = "no auto-commit threshold" if limit is None else f"threshold {limit:g}"
    _plan_refusal(
        "auto_commit_refused",
        403,
        (
            f"auto commit refused: project '{project_id}' has {threshold} GPU hours; "
            f"this plan's max_gpu_hours is {max_gpu_hours:g}"
        ),
    )


@router.post(
    "/{workflow_name}/commit",
    operation_id="workflow_commit",
    response_model=WorkflowCommitResponse,
    response_model_exclude_unset=True,
)
async def commit_workflow(
    workflow_name: str,
    request: WorkflowCommitRequest,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_r4_session_id: str | None = Header(None, alias="x-r4-session-id"),
    memberships: frozenset[str] = Depends(get_verified_memberships),
):
    """Start the run a plan describes (ADR-105 D-4): verify the signature,
    expiry and caller, re-resolve (a changed target is ``plan_stale``), check
    ``auto`` against the project's threshold, then trigger with the plan id
    as the idempotency key: a second commit answers the run already started.
    The project's ``require_plan`` does not apply here.
    # auth: verified
    """
    _require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    caller = _WorkflowHeaders(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    signed = _verified_plan(workflow_name, request)
    resolved = _resolve_trigger(
        workflow_name, _trigger_from_plan(signed), caller, memberships
    )
    _refuse_foreign_plan(signed, resolved)
    quote = _quote_resolved(workflow_name, resolved).quote
    _refuse_stale_plan(signed, _plan_fields(resolved, quote))
    if request.auto:
        _refuse_auto_commit(resolved.project_id, quote.max_gpu_hours)
    if get_optional_job_store() is None:
        _plan_refusal(
            "commit_unavailable", 503, "a plan commit needs the durable job store"
        )
    from ..workflow_plans import plan_id_for

    plan_id = plan_id_for(request.digest)
    payload = _dispatch_trigger(resolved, x_r4_session_id, idempotency_key=plan_id)
    payload["plan_id"] = plan_id
    return payload


# ---------------------------------------------------------------------------
# Execution status / results / cancel
# ---------------------------------------------------------------------------


@router.get(
    "/executions/{execution_id}",
    operation_id="workflow_status",
    response_model=WorkflowExecutionResponse,
    response_model_exclude_unset=True,
)
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


@router.get(
    "/executions/{execution_id}/results",
    operation_id="workflow_results",
    response_model=WorkflowResultsResponse,
    response_model_exclude_unset=True,
)
async def get_execution_results(
    execution_id: str,
    identity: CallerIdentity = Depends(get_caller_identity),
):
    """Return the structured outputs of a finished execution.

    For unfinished executions, returns the partial results gathered so
    far (so clients can render progressive UIs without retrying).
    ``workflow_version`` and ``source_sha256`` name what the run pinned
    (ADR-113 D-4).
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
            "workflow_version": execution.get("workflow_version"),
            "source_sha256": execution.get("source_sha256"),
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
        "workflow_version": execution.get("workflow_version"),
        "source_sha256": execution.get("source_sha256"),
        "results": {
            "steps": record.results.get("steps", {}),
            "final": record.results.get("final"),
        },
        "artifacts": artifacts,
        "durable_job_id": execution.get("durable_job_id"),
        "durable_job": execution.get("durable_job"),
    }


@router.post(
    "/executions/{execution_id}/cancel",
    operation_id="workflow_cancel",
    response_model=WorkflowExecutionResponse,
    response_model_exclude_unset=True,
)
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
