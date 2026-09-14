"""Registered run definitions (named workflow invocation templates).

Purpose: token efficiency + reproducibility. A user's proven
``(workflow_name, params)`` pair is persisted server-side once; re-running it
never requires an LLM to re-derive parameters, and the exact same inputs are
replayed every time.

Storage mirrors the workflow catalog: a JSON catalog shared across the four
umrs-front cluster servers via the same state dir (``UMRS_WORKFLOW_STATE_DIR``
when set, else the resolved umrs config dir), written tmp-then-replace so
concurrent writers never truncate each other (same write discipline as
``workflow_registry_handler._save_catalog``).

Ownership model: ``(name, user_id)`` is the identity. Reads, triggers, and
deletes are owner-only; admins ( callers with the verified admin header) are
handled at the route layer by passing the record's owner as ``user_id``.

Records are plain dicts matching ``r4_contracts.workflows.RunDefinitionRecord``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_CATALOG_LOCK = threading.Lock()


def _definitions_path() -> str:
    """Resolve the run-definition catalog path (workflow-catalog parity)."""
    from umrs.shared.config import get_resolved_umrs_subdir

    state_dir = os.environ.get("UMRS_WORKFLOW_STATE_DIR", "").strip()
    if state_dir:
        return os.path.join(state_dir, "run_definition_catalog.json")
    return get_resolved_umrs_subdir("run_definition_catalog.json")


def _load_definitions() -> dict[str, Any]:
    path = _definitions_path()
    if not os.path.exists(path):
        return {"definitions": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_definitions(catalog: dict[str, Any]) -> None:
    path = _definitions_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def save_definition(
    *,
    name: str,
    workflow_name: str,
    params: dict[str, Any],
    user_id: str,
    project_id: str = "",
    description: str = "",
) -> dict[str, Any]:
    """Create or update one run definition (upsert by (name, user_id))."""
    if not name or not workflow_name:
        raise ValueError("name and workflow_name are required")
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    now = datetime.now().isoformat()
    with _CATALOG_LOCK:
        catalog = _load_definitions()
        for existing in catalog.get("definitions", []):
            if existing.get("name") == name and existing.get("user_id") == user_id:
                existing.update(
                    {
                        "workflow_name": workflow_name,
                        "params": params,
                        "project_id": project_id,
                        "description": description,
                        "updated_at": now,
                    }
                )
                _save_definitions(catalog)
                logger.info("Updated run definition: %s (user=%s)", name, user_id)
                return dict(existing)
        record = {
            "name": name,
            "workflow_name": workflow_name,
            "params": params,
            "user_id": user_id,
            "project_id": project_id,
            "description": description,
            "created_at": now,
            "updated_at": now,
        }
        catalog.setdefault("definitions", []).append(record)
        _save_definitions(catalog)
    logger.info(
        "Registered run definition: %s (workflow=%s, user=%s)",
        name,
        workflow_name,
        user_id,
    )
    return dict(record)


def list_definitions(user_id: str | None = None) -> list[dict[str, Any]]:
    """List run definitions, optionally owner-filtered."""
    with _CATALOG_LOCK:
        catalog = _load_definitions()
    entries = catalog.get("definitions", [])
    if user_id:
        entries = [d for d in entries if d.get("user_id") == user_id]
    return [dict(d) for d in entries]


def get_definition(name: str, user_id: str) -> dict[str, Any] | None:
    """Owner-scoped lookup (ownership check lives HERE, not just the route)."""
    with _CATALOG_LOCK:
        catalog = _load_definitions()
    for entry in catalog.get("definitions", []):
        if entry.get("name") == name and entry.get("user_id") == user_id:
            return dict(entry)
    return None


def delete_definition(name: str, user_id: str) -> int:
    """Owner-scoped delete; returns number of removed records (0 = not found)."""
    with _CATALOG_LOCK:
        catalog = _load_definitions()
        keep = [
            d
            for d in catalog.get("definitions", [])
            if not (d.get("name") == name and d.get("user_id") == user_id)
        ]
        removed = len(catalog.get("definitions", [])) - len(keep)
        if removed:
            catalog["definitions"] = keep
            _save_definitions(catalog)
    if removed:
        logger.info("Deleted run definition: %s (user=%s)", name, user_id)
    return removed
