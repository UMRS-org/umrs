"""
Workflow Registry Handler — bridges ZMQ protocol to workflow registry operations.

Provides handle_workflow_command() for the server to call from its ZMQ loop.
Uses a JSON file catalog for lightweight workflow registration (mirrors the
model_registry_handler.py pattern).

Supported commands (via TaskCmd):
  LIST_WORKFLOWS  — list all registered workflows (with optional scope/user filter)

Also provides register_workflow() for internal use (auto-registration on upload).

Catalog file: <resolved UMRS config dir>/workflow_catalog.json
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from umrs.shared.config import get_resolved_umrs_subdir

if TYPE_CHECKING:
    from umrs.server.job_tracking_handler import DurableJobCommandBackend

logger = logging.getLogger(__name__)

_CATALOG_LOCK = threading.Lock()


def _catalog_path() -> str:
    """Return path to the JSON workflow catalog file.

    ``UMRS_WORKFLOW_STATE_DIR`` (set in umrs-front compose to a bind-mounted
    dir) relocates the catalog off container tmpfs so registrations survive
    restarts; default preserves the legacy config-dir location.
    """
    state_dir = os.environ.get("UMRS_WORKFLOW_STATE_DIR", "").strip()
    if state_dir:
        return os.path.join(state_dir, "workflow_catalog.json")
    return get_resolved_umrs_subdir("workflow_catalog.json")


def _load_catalog() -> dict[str, Any]:
    """Load workflow catalog from JSON file."""
    path = _catalog_path()
    if not os.path.exists(path):
        return {"workflows": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(catalog: dict[str, Any]) -> None:
    """Save workflow catalog to JSON file.

    All four umrs-front cluster servers share one state dir; writes go
    tmp-then-os.replace so the four process writers never truncate each
    other's concurrent read.
    """
    path = _catalog_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def register_workflow(
    workflow_name: str,
    version: str,
    user_id: str,
    source_path: str,
    workflow_json: dict[str, Any],
    scope: str = "user",
    description: str | None = None,
) -> dict[str, Any]:
    """Register a workflow in the catalog.

    Called internally by the upload handler (auto-registration) or
    potentially via a future REGISTER_WORKFLOW command.

    Args:
        workflow_name: Unique name for the workflow.
        version: Version string (e.g. "1.0.0", "0.1").
        user_id: Client/user who owns this workflow.
        source_path: Absolute path to the extracted workflow directory.
        workflow_json: Parsed contents of workflow.json.
        scope: "user" (private) or "system" (shared).
        description: Optional human-readable description.

    Returns:
        Dict with "status" ("ok" or "error") and details.
    """
    # Extract step metadata from workflow_json
    steps_meta: list[dict[str, str]] = []
    for step in workflow_json.get("steps", []):
        steps_meta.append(
            {
                "name": step.get("name", ""),
                "entrypoint": step.get("entrypoint", ""),
            }
        )

    entry = {
        "name": workflow_name,
        "version": version,
        "user_id": user_id,
        "source_path": source_path,
        "scope": scope,
        "description": description or workflow_json.get("description", ""),
        "steps": steps_meta,
        "requirements": workflow_json.get("requirements", []),
        "registered_at": datetime.now().isoformat(),
    }

    with _CATALOG_LOCK:
        catalog = _load_catalog()
        # Check for duplicate (name, version, scope, user_id)
        for existing in catalog["workflows"]:
            if (
                existing["name"] == workflow_name
                and existing.get("version") == version
                and existing.get("scope") == scope
                and existing.get("user_id") == user_id
            ):
                # Update in place (re-upload overwrites)
                existing.update(entry)
                _save_catalog(catalog)
                logger.info(
                    "Updated workflow registration: %s v%s (user=%s, scope=%s)",
                    workflow_name,
                    version,
                    user_id,
                    scope,
                )
                return {
                    "status": "ok",
                    "message": f"Workflow '{workflow_name}' v{version} updated",
                    "workflow": entry,
                }
        catalog["workflows"].append(entry)
        _save_catalog(catalog)

    logger.info(
        "Registered workflow: %s v%s (user=%s, scope=%s, %d steps)",
        workflow_name,
        version,
        user_id,
        scope,
        len(steps_meta),
    )
    return {
        "status": "ok",
        "message": f"Workflow '{workflow_name}' v{version} registered",
        "workflow": entry,
    }


def unregister_workflow(
    workflow_name: str,
    *,
    user_id: str,
    scope: str | None = None,
) -> dict[str, Any]:
    """Remove a workflow registration from the catalog.

    Deletes exactly the entries that match ``(name, user_id, scope?)`` —
    ownership is decided by the caller (REST layer checks admin OR owner;
    system scope requires admin), this function enforces the match.

    Returns ``{"status": "ok", "removed": n}``; ``removed == 0`` when no
    matching registration exists (caller maps to 404).
    """
    with _CATALOG_LOCK:
        catalog = _load_catalog()
        keep: list[dict[str, Any]] = []
        removed = 0
        for entry in catalog.get("workflows", []):
            if (
                entry.get("name") == workflow_name
                and entry.get("user_id") == user_id
                and (scope is None or entry.get("scope", "user") == scope)
            ):
                removed += 1
                continue
            keep.append(entry)
        if removed:
            catalog["workflows"] = keep
            _save_catalog(catalog)

    if removed:
        logger.info(
            "Unregistered workflow: %s (user=%s, scope=%s, removed=%d)",
            workflow_name,
            user_id,
            scope or "*",
            removed,
        )
    return {"status": "ok", "removed": removed}


def _handle_list_workflows(client_id: str, payload_str: str) -> str:
    """List registered workflows, optionally filtered by scope or user.

    Payload JSON (all optional):
      - scope: "user" | "system" — filter by scope
      - user_id: str — filter by owner
      - name: str — filter by workflow name (substring match)
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        payload = {}

    with _CATALOG_LOCK:
        catalog = _load_catalog()
    workflows = catalog.get("workflows", [])

    # Apply filters
    scope_filter = payload.get("scope") if isinstance(payload, dict) else None
    user_filter = payload.get("user_id") if isinstance(payload, dict) else None
    name_filter = payload.get("name") if isinstance(payload, dict) else None

    if scope_filter:
        workflows = [w for w in workflows if w.get("scope") == scope_filter]
    if user_filter:
        workflows = [w for w in workflows if w.get("user_id") == user_filter]
    if name_filter:
        workflows = [
            w for w in workflows if name_filter.lower() in w.get("name", "").lower()
        ]

    return json.dumps(
        {
            "status": "ok",
            "count": len(workflows),
            "workflows": workflows,
        },
        default=str,
    )


def _handle_unregister_workflow(client_id: str, payload_str: str) -> str:
    """ZMQ command path mirroring the REST DELETE.

    Payload JSON:
      - name (required): workflow name
      - scope (optional): only delete entries with this scope

    Authorization (admin/owner) is enforced by the authenticated caller
    identity that reaches this handler — same trust as REGISTER/UPLOAD.
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"status": "error", "message": f"invalid JSON: {exc}"})

    name = str(payload.get("name", "")).strip()
    if not name:
        return json.dumps({"status": "error", "message": "name is required"})
    result = unregister_workflow(
        name,
        user_id=client_id,
        scope=payload.get("scope") if isinstance(payload, dict) else None,
    )
    if result["removed"] == 0:
        return json.dumps({"status": "error", "message": f"Workflow '{name}' not found"})
    return json.dumps(result)


def _workflow_identity_provenance(
    params: dict[str, Any],
    *,
    identity_verified: bool = False,
) -> str:
    from umrs.api.workflow_auth import strict_auth_mode

    project_id = _workflow_project_id(params)
    if project_id == "default":
        return "unverified"
    if strict_auth_mode():
        return "gateway_verified" if identity_verified else "unverified"
    return "mvp_claim"


def _workflow_project_id(params: Any) -> str:
    if isinstance(params, dict) and isinstance(params.get("project_id"), str):
        project_id = params["project_id"].strip()
        if project_id:
            return project_id
    return "default"


def _workflow_run_context(
    workflow_name: str,
    client_id: str,
    source_path: str | None = None,
) -> tuple[str, dict[str, Any], int]:
    if source_path is None:
        entry = _resolve_workflow_catalog_entry(workflow_name, client_id)
        if entry is None:
            raise ValueError(f"Workflow '{workflow_name}' not found in registry")
        source_path = entry.get("source_path", "")

    if not source_path or not os.path.isdir(source_path):
        raise ValueError(f"Workflow source path not found: {source_path}")

    wf_json_path = os.path.join(source_path, "workflow.json")
    if not os.path.isfile(wf_json_path):
        raise ValueError(f"workflow.json missing at {source_path}")

    try:
        with open(wf_json_path, encoding="utf-8") as handle:
            workflow_json = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Failed to load workflow.json: {exc}") from exc

    return source_path, workflow_json, len(workflow_json.get("steps", []))


def _resolve_workflow_catalog_entry(
    workflow_name: str,
    client_id: str,
) -> dict[str, Any] | None:
    """Resolve a workflow by caller-owned entry, then system, then any-name.

    Scopes ("user"/"system") are advisory — durable-jobs run under varying
    runtime identities (upload client, scheduler-side execution engine,
    deployment umbrella). A registered workflow shall be resolvable by
    everyone; otherwise different legacies of upload user vs execution user
    resurface as spurious "not found in registry" errors deep in the chain.
    """
    with _CATALOG_LOCK:
        catalog = _load_catalog()

    owned: dict[str, Any] | None = None
    system: dict[str, Any] | None = None
    any_name: dict[str, Any] | None = None
    for workflow_entry in catalog.get("workflows", []):
        if workflow_entry.get("name") != workflow_name:
            continue
        if any_name is None:
            any_name = workflow_entry
        if workflow_entry.get("user_id") == client_id:
            owned = workflow_entry
            break
        if workflow_entry.get("scope") == "system" and system is None:
            system = workflow_entry

    return owned or system or any_name


def _launch_workflow_from_catalog(
    *,
    workflow_name: str,
    client_id: str,
    params: dict[str, Any],
    execution_id: str | None = None,
    durable_job_id: str | None = None,
    source_path: str | None = None,
    identity_verified: bool = False,
):
    source_path, workflow_json, step_count = _workflow_run_context(
        workflow_name,
        client_id,
        source_path=source_path,
    )

    logger.info(
        "Running workflow '%s' for user '%s' with %d steps (params=%s, durable_job_id=%s)",
        workflow_name,
        client_id,
        step_count,
        params,
        durable_job_id,
    )

    from umrs.server.step_router import build_registry_backed_router
    from umrs.server.workflow_execution_registry import get_execution_registry
    from umrs.server.workflow_runner import run_workflow_record

    registry = get_execution_registry()
    record = registry.create(
        workflow_name=workflow_name,
        user_id=client_id,
        params=params,
        execution_id=execution_id,
        durable_job_id=durable_job_id,
        identity_provenance=_workflow_identity_provenance(
            params,
            identity_verified=identity_verified,
        ),
    )

    project_id = _workflow_project_id(params)
    router = build_registry_backed_router(project_id=project_id)

    def _runner(record_to_run):
        run_workflow_record(
            record_to_run,
            source_path=source_path,
            workflow_json=workflow_json,
            router=router,
            project_id=project_id,
            user_id=client_id,
        )

    registry.launch(record, _runner)
    return record, source_path, step_count


def _submit_durable_workflow_run(
    *,
    client_id: str,
    workflow_name: str,
    params: dict[str, Any],
    execution_id: str,
    source_path: str,
    step_count: int,
    job_backend: DurableJobCommandBackend,
) -> str:
    raw = job_backend.submit(
        client_id,
        json.dumps(
            {
                "project_id": _workflow_project_id(params),
                "job_type": "workflow_run",
                "spec": {
                    "workflow_name": workflow_name,
                    "execution_id": execution_id,
                    "params": params,
                    "source_path": source_path,
                    # Auto-ingest (TASK-035): completed workflow outputs flow
                    # into the knowledge pipeline. "step" "*" marks run-level
                    # provenance (not a single named step).
                    "ingest_results": True,
                    "workflow": workflow_name,
                    "step": "*",
                },
                "max_attempts": 4,
                "restart_policy": "on_crash",
            }
        ),
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return json.dumps(
            {"status": "error", "message": f"Invalid durable job response: {exc}"}
        )

    if data.get("status") == "error":
        return raw

    job = data.get("job") if isinstance(data.get("job"), dict) else {}
    durable_job_id = job.get("job_id")
    return json.dumps(
        {
            "status": "ok",
            "action": "run-workflow",
            "workflow_name": workflow_name,
            "source_path": source_path,
            "steps": step_count,
            "params": params,
            "execution_id": execution_id,
            "durable_job_id": durable_job_id,
            "durable_job": job,
            "message": (
                f"Workflow '{workflow_name}' execution queued as durable job "
                f"{durable_job_id or '<unknown>'} ({step_count} steps)"
            ),
        },
        default=str,
    )


def _handle_run_workflow(
    client_id: str,
    payload_str: str,
    job_backend: DurableJobCommandBackend | None = None,
    *,
    trusted_identity: bool = False,
) -> str:
    """Trigger execution of a registered workflow.

    Payload JSON:
      - workflow_name: str — name of the workflow to run
      - params: dict — additional runtime parameters (e.g. bucket alias)
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "Invalid JSON payload"})

    workflow_name = payload.get("workflow_name", "")
    params = payload.get("params", {})

    if not workflow_name:
        return json.dumps({"status": "error", "message": "workflow_name is required"})
    if not isinstance(params, dict):
        return json.dumps(
            {"status": "error", "message": "workflow params must be an object"}
        )

    identity_verified = trusted_identity and bool(payload.get("_identity_verified"))

    try:
        source_path, _, step_count = _workflow_run_context(workflow_name, client_id)
    except ValueError as exc:
        return json.dumps({"status": "error", "message": str(exc)})

    if job_backend is not None:
        execution_id = f"wf-{uuid.uuid4().hex[:12]}"
        return _submit_durable_workflow_run(
            client_id=client_id,
            workflow_name=workflow_name,
            params=params,
            execution_id=execution_id,
            source_path=source_path,
            step_count=step_count,
            job_backend=job_backend,
        )

    record, source_path, step_count = _launch_workflow_from_catalog(
        workflow_name=workflow_name,
        client_id=client_id,
        params=params,
        source_path=source_path,
        identity_verified=identity_verified,
    )

    return json.dumps(
        {
            "status": "ok",
            "action": "run-workflow",
            "workflow_name": workflow_name,
            "source_path": source_path,
            "steps": step_count,
            "params": params,
            "execution_id": record.execution_id,
            "durable_job_id": record.durable_job_id,
            "message": f"Workflow '{workflow_name}' execution accepted ({step_count} steps)",
        },
        default=str,
    )


def handle_workflow_command(
    cmd: str,
    client_id: str,
    payload_str: str,
    job_backend: DurableJobCommandBackend | None = None,
    *,
    trusted_identity: bool = False,
) -> str:
    """Dispatch a workflow registry command and return a JSON response string.

    Called from the server main loop when a LIST_WORKFLOWS command is received.

    ``trusted_identity`` must be set ONLY by server-side callers that already
    verified identity (the HTTP trigger route after HMAC signature validation).
    For ZMQ-exposed commands the payload's ``_identity_verified`` flag is a
    client-supplied claim and is ignored unless ``trusted_identity`` is set
    (wave-4 audit fix).
    """
    from umrs.server.correlation import correlation_scope

    try:
        _payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        _payload = {}

    with correlation_scope(
        (_payload or {}).get("_request_id") or (_payload or {}).get("_correlation_id")
    ):
        return _dispatch_workflow_command(
            cmd, client_id, payload_str, job_backend, trusted_identity
        )


def _dispatch_workflow_command(
    cmd,
    client_id,
    payload_str,
    job_backend,
    trusted_identity,
):  # pragma: no cover - factored from handle_workflow_command body
    """Command dispatch under a fresh correlation scope (Q-05/G-001)."""
    from umrs.core.TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.LIST_WORKFLOWS:
            return _handle_list_workflows(client_id, payload_str)
        elif cmd == TaskCmd.RUN_WORKFLOW:
            return _handle_run_workflow(
                client_id,
                payload_str,
                job_backend=job_backend,
                trusted_identity=trusted_identity,
            )
        elif cmd == TaskCmd.WORKFLOW_STATUS:
            return _handle_workflow_status(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_RESULTS:
            return _handle_workflow_results(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_CANCEL:
            return _handle_workflow_cancel(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_ERRORS:
            return _handle_workflow_errors(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_LOGS:
            return _handle_workflow_logs(client_id, payload_str)
        elif cmd == TaskCmd.UNREGISTER_WORKFLOW:
            return _handle_unregister_workflow(client_id, payload_str)
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Unknown workflow command: {cmd}",
                }
            )
    except Exception as e:
        logger.exception("Workflow command '%s' failed", cmd)
        return json.dumps({"status": "error", "message": str(e)})


def _parse_execution_id(payload_str: str) -> tuple[str | None, dict[str, Any]]:
    """Parse execution_id from payload. Returns (execution_id, payload)."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(payload, dict):
        return None, {}
    return payload.get("execution_id"), payload


def _zmq_execution_access(
    client_id: str,
    payload: dict[str, Any],
    execution_id: str,
) -> tuple[Any | None, Any | None, str | None]:
    """Resolve execution context and return an access-denied JSON string when needed."""
    from umrs.api.jobs import get_optional_job_store
    from umrs.api.workflow_auth import (
        check_zmq_execution_access,
        resolve_execution_context,
        zmq_caller_from_payload,
    )

    store = get_optional_job_store()
    record, job = resolve_execution_context(execution_id, store=store)
    if record is None and job is None:
        return None, None, None
    caller = zmq_caller_from_payload(client_id, payload)
    denied = check_zmq_execution_access(caller, record=record, job=job)
    if denied is not None:
        return record, job, denied
    return record, job, None


def _handle_workflow_status(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_STATUS — show execution status for a workflow run."""
    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )
    if record is not None:
        return json.dumps(
            {
                "status": "ok",
                "execution": record.to_dict(),
            },
            default=str,
        )
    from umrs.api.workflow_auth import execution_payload_from_job

    return json.dumps(
        {
            "status": "ok",
            "execution": execution_payload_from_job(job),
        },
        default=str,
    )


def _handle_workflow_results(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_RESULTS — fetch results for a completed workflow run."""
    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )
    if record is None:
        from umrs.api.workflow_auth import execution_payload_from_job

        return json.dumps(
            {
                "status": "ok",
                "execution": execution_payload_from_job(job),
                "results": {"steps": {}, "final": None},
                "artifacts": [],
            },
            default=str,
        )

    return json.dumps(
        {
            "status": "ok",
            "execution": record.to_dict(),
            "results": record.results,
            "artifacts": record.artifacts,
        },
        default=str,
    )


def _handle_workflow_cancel(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_CANCEL — cancel a running workflow execution."""
    from umrs.server.workflow_execution_registry import get_execution_registry

    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )

    registry = get_execution_registry()
    record = registry.cancel(execution_id) if record is not None else None
    from umrs.api.jobs import get_optional_job_store
    from umrs.jobs.store import InvalidTransitionError

    store = get_optional_job_store()
    if job is not None and store is not None:
        try:
            store.cancel_job(job.job_id, reason="workflow_execution_cancelled")
        except InvalidTransitionError:
            pass
        job = store.get_job(job.job_id)

    if record is None and job is not None:
        from umrs.api.workflow_auth import execution_payload_from_job

        return json.dumps(
            {
                "status": "ok",
                "message": f"Execution '{execution_id}' cancelled",
                "execution": execution_payload_from_job(job),
            },
            default=str,
        )

    if record is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "message": f"Execution '{execution_id}' cancelled",
            "execution": record.to_dict(),
        },
        default=str,
    )


def _handle_workflow_errors(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_ERRORS — show error logs for a failed workflow execution.

    This command surfaces error logs for failed Attempts, including:
    - Workflow-level error message
    - Per-step error details for failed steps
    - Error log entries (if available)
    """

    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    tail = 50
    if isinstance(payload, dict):
        tail_val = payload.get("tail")
        if isinstance(tail_val, int) and tail_val > 0:
            tail = tail_val

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )

    # Collect error logs from the execution record
    error_logs: list[dict[str, Any]] = []

    # Workflow-level error
    if record.error:
        error_logs.append(
            {
                "timestamp": datetime.fromtimestamp(
                    record.finished_at or record.submitted_at
                ).isoformat(),
                "level": "ERROR",
                "source": "workflow",
                "message": record.error,
            }
        )

    # Per-step errors
    for step_name, step in record.steps.items():
        if step.status == "failed" and step.error:
            error_logs.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        step.finished_at or step.started_at or record.submitted_at
                    ).isoformat(),
                    "level": "ERROR",
                    "source": f"step:{step_name}",
                    "message": step.error,
                }
            )

    # Limit to tail
    error_logs = error_logs[-tail:]

    return json.dumps(
        {
            "status": "ok",
            "execution": record.to_dict(),
            "error_logs": error_logs,
            "count": len(error_logs),
        },
        default=str,
    )


def _handle_workflow_logs(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_LOGS — stream execution logs for a workflow run.

    Returns a structured log timeline including:
    - Workflow lifecycle events (submitted, started, finished)
    - Per-step events (started, routing, succeeded/failed)
    - Workflow-level error
    - Per-step errors

    The CLI can poll this endpoint with ``--follow`` to stream updates
    until the workflow reaches a terminal state.
    """
    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    tail = 200
    if isinstance(payload, dict):
        tail_val = payload.get("tail")
        if isinstance(tail_val, int) and tail_val > 0:
            tail = tail_val

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )
    if record is None:
        from umrs.api.workflow_auth import execution_payload_from_job

        return json.dumps(
            {
                "status": "ok",
                "execution": execution_payload_from_job(job),
                "logs": [],
                "count": 0,
            },
            default=str,
        )

    # Build log timeline
    log_entries: list[dict[str, Any]] = []

    # Workflow lifecycle
    log_entries.append(
        {
            "timestamp": datetime.fromtimestamp(record.submitted_at).isoformat(),
            "level": "INFO",
            "source": "workflow",
            "message": f"Workflow '{record.workflow_name}' submitted",
        }
    )
    if record.started_at:
        log_entries.append(
            {
                "timestamp": datetime.fromtimestamp(record.started_at).isoformat(),
                "level": "INFO",
                "source": "workflow",
                "message": "Workflow execution started",
            }
        )
    if record.finished_at:
        log_entries.append(
            {
                "timestamp": datetime.fromtimestamp(record.finished_at).isoformat(),
                "level": "INFO" if record.status == "succeeded" else "ERROR",
                "source": "workflow",
                "message": f"Workflow finished with status '{record.status}'",
            }
        )

    # Per-step events
    for step_name, step in record.steps.items():
        if step.started_at:
            log_entries.append(
                {
                    "timestamp": datetime.fromtimestamp(step.started_at).isoformat(),
                    "level": "INFO",
                    "source": f"step:{step_name}",
                    "message": f"Step '{step_name}' started",
                }
            )
        if step.finished_at:
            level = "ERROR" if step.status == "failed" else "INFO"
            log_entries.append(
                {
                    "timestamp": datetime.fromtimestamp(step.finished_at).isoformat(),
                    "level": level,
                    "source": f"step:{step_name}",
                    "message": (
                        f"Step '{step_name}' {step.status}"
                        + (f": {step.error}" if step.error else "")
                    ),
                }
            )

    # Sort by timestamp
    log_entries.sort(key=lambda e: e.get("timestamp", ""))

    # Limit to tail
    log_entries = log_entries[-tail:]

    return json.dumps(
        {
            "status": "ok",
            "execution": record.to_dict(),
            "logs": log_entries,
            "count": len(log_entries),
        },
        default=str,
    )
