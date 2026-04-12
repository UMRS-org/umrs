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
from datetime import datetime
from typing import Any, Dict, List, Optional

from umrs.shared.config import get_resolved_umrs_subdir

logger = logging.getLogger(__name__)

_CATALOG_LOCK = threading.Lock()


def _catalog_path() -> str:
    """Return path to the JSON workflow catalog file."""
    return get_resolved_umrs_subdir("workflow_catalog.json")


def _load_catalog() -> Dict[str, Any]:
    """Load workflow catalog from JSON file."""
    path = _catalog_path()
    if not os.path.exists(path):
        return {"workflows": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(catalog: Dict[str, Any]) -> None:
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
    workflow_json: Dict[str, Any],
    scope: str = "user",
    description: Optional[str] = None,
) -> Dict[str, Any]:
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
    steps_meta: List[Dict[str, str]] = []
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


def _handle_run_workflow(client_id: str, payload_str: str) -> str:
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

    with _CATALOG_LOCK:
        catalog = _load_catalog()

    # Find the workflow entry
    entry = None
    for w in catalog.get("workflows", []):
        if w["name"] == workflow_name:
            # Prefer user's own workflow, fall back to any match
            if w.get("user_id") == client_id:
                entry = w
                break
            if entry is None:
                entry = w

    if entry is None:
        return json.dumps({
            "status": "error",
            "message": f"Workflow '{workflow_name}' not found in registry",
        })

    source_path = entry.get("source_path", "")
    if not source_path or not os.path.isdir(source_path):
        return json.dumps({
            "status": "error",
            "message": f"Workflow source path not found: {source_path}",
        })

    # Load workflow.json from the registered source
    wf_json_path = os.path.join(source_path, "workflow.json")
    if not os.path.isfile(wf_json_path):
        return json.dumps({
            "status": "error",
            "message": f"workflow.json missing at {source_path}",
        })

    try:
        with open(wf_json_path, "r", encoding="utf-8") as fh:
            wf_data = json.load(fh)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to load workflow.json: {e}"})

    steps = wf_data.get("steps", [])
    logger.info(
        "Running workflow '%s' for user '%s' with %d steps (params=%s)",
        workflow_name, client_id, len(steps), params,
    )

    return json.dumps({
        "status": "ok",
        "action": "run-workflow",
        "workflow_name": workflow_name,
        "source_path": source_path,
        "steps": len(steps),
        "params": params,
        "message": f"Workflow '{workflow_name}' execution accepted ({len(steps)} steps)",
    }, default=str)


def handle_workflow_command(cmd: str, client_id: str, payload_str: str) -> str:
    """Dispatch a workflow registry command and return a JSON response string.

    Called from the server main loop when a LIST_WORKFLOWS command is received.
    """
    from umrs.core.TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.LIST_WORKFLOWS:
            return _handle_list_workflows(client_id, payload_str)
        elif cmd == TaskCmd.RUN_WORKFLOW:
            return _handle_run_workflow(client_id, payload_str)
        else:
            return json.dumps({
                "status": "error",
                "message": f"Unknown workflow command: {cmd}",
            })
    except Exception as e:
        logger.exception("Workflow command '%s' failed", cmd)
        return json.dumps({"status": "error", "message": str(e)})
