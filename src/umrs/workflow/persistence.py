"""Workflow persistence layer.

Handles saving, loading, and version migration of workflow definitions.
Backward compatible with schema v1 (no outputs section).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import (
    WORKFLOW_SCHEMA_VERSION,
    WorkflowDefinition,
    generate_workflow_id,
)

logger = logging.getLogger(__name__)

# Persistence directory (created on first use)
_DEFAULT_WF_DIR = Path.home() / ".umrs" / "workflows"


def _ensure_dir(dir_path: Path) -> Path:
    """Ensure the directory exists, create if needed."""
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _workflow_path(dir_path: Path, workflow_id: str) -> Path:
    """Get the file path for a workflow definition."""
    return _ensure_dir(dir_path) / f"{workflow_id}.json"


def save_workflow(
    workflow: WorkflowDefinition,
    dir_path: Path | None = None,
) -> Path:
    """Save a workflow definition to disk.

    Args:
        workflow: The workflow definition to save.
        dir_path: Directory to save to (defaults to ~/.umrs/workflows).

    Returns:
        Path to the saved file.
    """
    dir_path = dir_path or _DEFAULT_WF_DIR
    path = _workflow_path(dir_path, workflow.id)

    data = workflow.to_dict()
    data["_schema_version"] = workflow.version

    path.write_text(json.dumps(data, indent=2))
    logger.debug("Saved workflow '%s' (%s) -> %s", workflow.id, workflow.name, path)
    return path


def load_workflow(
    workflow_id: str,
    dir_path: Path | None = None,
) -> WorkflowDefinition | None:
    """Load a workflow definition from disk.

    Handles backward-compatible loading: if the file has no outputs section,
    the loaded workflow will have empty outputs lists (correct for v1).

    Args:
        workflow_id: The workflow ID to load.
        dir_path: Directory to load from (defaults to ~/.umrs/workflows).

    Returns:
        The workflow definition, or None if not found.
    """
    dir_path = dir_path or _DEFAULT_WF_DIR
    path = _workflow_path(dir_path, workflow_id)

    if not path.exists():
        logger.debug("Workflow file not found: %s", path)
        return None

    try:
        data = json.loads(path.read_text())

        # Handle schema version (v1 has none or explicit v1)
        schema_version = data.pop("_schema_version", None) or data.get(
            "version", "1.0.0"
        )
        if schema_version != WORKFLOW_SCHEMA_VERSION:
            logger.info(
                "Migrating workflow '%s' from schema %s to %s",
                workflow_id,
                schema_version,
                WORKFLOW_SCHEMA_VERSION,
            )
            data = _migrate_schema(data, schema_version)

        workflow = WorkflowDefinition.from_dict(data)
        logger.debug(
            "Loaded workflow '%s' (%s) from %s", workflow_id, workflow.name, path
        )
        return workflow

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Failed to load workflow '%s': %s", workflow_id, exc)
        return None


def list_workflows(
    dir_path: Path | None = None,
) -> list[WorkflowDefinition]:
    """List all saved workflow definitions.

    Args:
        dir_path: Directory to list from (defaults to ~/.umrs/workflows).

    Returns:
        List of workflow definitions sorted by updated_at descending.
    """
    dir_path = dir_path or _DEFAULT_WF_DIR
    if not dir_path.exists():
        return []

    workflows: list[WorkflowDefinition] = []
    for path in dir_path.glob("*.json"):
        wf = load_workflow(path.stem, dir_path)
        if wf is not None:
            workflows.append(wf)

    workflows.sort(key=lambda w: w.updated_at, reverse=True)
    return workflows


def delete_workflow(
    workflow_id: str,
    dir_path: Path | None = None,
) -> bool:
    """Delete a workflow definition.

    Args:
        workflow_id: The workflow ID to delete.
        dir_path: Directory to delete from (defaults to ~/.umrs/workflows).

    Returns:
        True if deleted, False if not found.
    """
    dir_path = dir_path or _DEFAULT_WF_DIR
    path = _workflow_path(dir_path, workflow_id)

    if path.exists():
        path.unlink()
        logger.debug("Deleted workflow '%s' from %s", workflow_id, path)
        return True

    return False


def _migrate_schema(data: dict[str, Any], from_version: str) -> dict[str, Any]:
    """Migrate workflow data to the latest schema version.

    Currently handles v1 -> v2 (adding outputs section).
    """
    if from_version == "1.0.0" or from_version.startswith("1."):
        # v1 -> v2: ensure outputs sections exist
        for step in data.get("steps", []):
            if "outputs" not in step:
                step["outputs"] = []
        if "outputs" not in data:
            data["outputs"] = []
        data["_schema_version"] = WORKFLOW_SCHEMA_VERSION
        data["version"] = WORKFLOW_SCHEMA_VERSION

    return data


def generate_id() -> str:
    """Generate a new workflow ID."""
    return generate_workflow_id()


def validate_workflow_data(data: dict[str, Any]) -> list[str]:
    """Validate workflow data before upload.

    Checks for required fields and output reference integrity.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []

    # Required fields
    if not data.get("id"):
        errors.append("Missing 'id' field")
    if not data.get("name"):
        errors.append("Missing 'name' field")

    # Steps validation
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        errors.append("'steps' must be a list")
    else:
        step_ids = set()
        for step in steps:
            if not step.get("id"):
                errors.append("Step missing 'id'")
            if not step.get("action"):
                errors.append(f"Step '{step.get('id', '?')}' missing 'action'")
            step_ids.add(step.get("id", ""))

            # Validate output references
            for output in step.get("outputs", []):
                if output.get("ingest") and not output.get("path_or_ref"):
                    errors.append(
                        f"Step '{step.get('id', '?')}' output '{output.get('name', '?')}' "
                        f"has ingest=true but no path_or_ref"
                    )

    # Workflow-level outputs validation
    for output in data.get("outputs", []):
        if not output.get("name"):
            errors.append("Workflow output missing 'name'")

    return errors
