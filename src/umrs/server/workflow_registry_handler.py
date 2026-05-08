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
    """Return path to the JSON workflow catalog file."""
    return get_resolved_umrs_subdir("workflow_catalog.json")


def _load_catalog() -> dict[str, Any]:
    """Load workflow catalog from JSON file."""
    path = _catalog_path()
    if not os.path.exists(path):
        return {"workflows": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(catalog: dict[str, Any]) -> None:
    """Save workflow catalog to JSON file."""
    path = _catalog_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)


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
        steps_meta.append({
            "name": step.get("name", ""),
            "entrypoint": step.get("entrypoint", ""),
        })

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
                    workflow_name, version, user_id, scope,
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
        workflow_name, version, user_id, scope, len(steps_meta),
    )
    return {
        "status": "ok",
        "message": f"Workflow '{workflow_name}' v{version} registered",
        "workflow": entry,
    }


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
        workflows = [w for w in workflows if name_filter.lower() in w.get("name", "").lower()]

    return json.dumps({
        "status": "ok",
        "count": len(workflows),
        "workflows": workflows,
    }, default=str)


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
        with _CATALOG_LOCK:
            catalog = _load_catalog()

        entry = None
        for workflow_entry in catalog.get("workflows", []):
            if workflow_entry["name"] == workflow_name:
                if workflow_entry.get("user_id") == client_id:
                    entry = workflow_entry
                    break
                if entry is None:
                    entry = workflow_entry

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


def _launch_workflow_from_catalog(
    *,
    workflow_name: str,
    client_id: str,
    params: dict[str, Any],
    execution_id: str | None = None,
    durable_job_id: str | None = None,
    source_path: str | None = None,
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
                },
                "max_attempts": 4,
                "restart_policy": "on_crash",
            }
        ),
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return json.dumps({"status": "error", "message": f"Invalid durable job response: {exc}"})

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
        return json.dumps({"status": "error", "message": "workflow params must be an object"})

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
    )

    return json.dumps({
        "status": "ok",
        "action": "run-workflow",
        "workflow_name": workflow_name,
        "source_path": source_path,
        "steps": step_count,
        "params": params,
        "execution_id": record.execution_id,
        "durable_job_id": record.durable_job_id,
        "message": f"Workflow '{workflow_name}' execution accepted ({step_count} steps)",
    }, default=str)


def handle_workflow_command(
    cmd: str,
    client_id: str,
    payload_str: str,
    job_backend: DurableJobCommandBackend | None = None,
) -> str:
    """Dispatch a workflow registry command and return a JSON response string.

    Called from the server main loop when a LIST_WORKFLOWS command is received.
    """
    from umrs.core.TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.LIST_WORKFLOWS:
            return _handle_list_workflows(client_id, payload_str)
        elif cmd == TaskCmd.RUN_WORKFLOW:
            return _handle_run_workflow(client_id, payload_str, job_backend=job_backend)
        else:
            return json.dumps({
                "status": "error",
                "message": f"Unknown workflow command: {cmd}",
            })
    except Exception as e:
        logger.exception("Workflow command '%s' failed", cmd)
        return json.dumps({"status": "error", "message": str(e)})
