"""
Model Registry Handler — bridges ZMQ protocol to model registry operations.

Provides handle_model_command() for the server to call from its ZMQ loop.
Uses a JSON file catalog for lightweight model registration (no PostgreSQL
required). When the full model_registry.store is available (PostgreSQL),
it can be used instead.

Supported commands (via TaskCmd):
  MODEL_REGISTER  — register a local model path
  MODEL_LIST      — list all registered models

NOTE: MODEL_DOWNLOAD was removed. Downloads now run client-side (on the
login node which has internet access), then the client sends MODEL_REGISTER.
This avoids blocking the server ZMQ loop and supports air-gapped compute nodes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict

from .config import get_default_umrs_cfg_dir

logger = logging.getLogger(__name__)

_CATALOG_LOCK = threading.Lock()


def _catalog_path() -> str:
    """Return path to the JSON model catalog file."""
    cfg_dir = get_default_umrs_cfg_dir()
    return os.path.join(cfg_dir, "model_catalog.json")


def _load_catalog() -> Dict[str, Any]:
    """Load model catalog from JSON file."""
    path = _catalog_path()
    if not os.path.exists(path):
        return {"models": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(catalog: Dict[str, Any]) -> None:
    """Save model catalog to JSON file."""
    path = _catalog_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)


def _handle_register(client_id: str, payload_str: str) -> str:
    """Register a local model path in the catalog."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})

    name = payload.get("name", "").strip()
    path = payload.get("path", "").strip()
    model_type = payload.get("model_type", "embedding")
    scope = payload.get("scope", "system")
    description = payload.get("description")
    huggingface_id = payload.get("huggingface_id")

    if not name:
        return json.dumps({"status": "error", "message": "Model name is required"})
    if not path:
        return json.dumps({"status": "error", "message": "Model path is required"})
    if not os.path.exists(path):
        return json.dumps({"status": "error", "message": f"Path does not exist: {path}"})

    entry = {
        "name": name,
        "storage_ref": os.path.abspath(path),
        "model_type": model_type,
        "scope": scope,
        "description": description,
        "huggingface_id": huggingface_id,
        "registered_by": client_id,
        "registered_at": datetime.now().isoformat(),
    }

    with _CATALOG_LOCK:
        catalog = _load_catalog()
        # Check for duplicate name in same scope
        for existing in catalog["models"]:
            if existing["name"] == name and existing.get("scope") == scope:
                return json.dumps({
                    "status": "error",
                    "message": f"Model '{name}' already registered in scope '{scope}'",
                })
        catalog["models"].append(entry)
        _save_catalog(catalog)

    logger.info("Registered model: %s at %s (scope=%s)", name, path, scope)
    return json.dumps({
        "status": "ok",
        "message": f"Model '{name}' registered at {path}",
        "model": entry,
    }, default=str)


def _handle_list(client_id: str, payload_str: str) -> str:
    """List all registered models."""
    with _CATALOG_LOCK:
        catalog = _load_catalog()
    models = catalog.get("models", [])
    return json.dumps({
        "status": "ok",
        "count": len(models),
        "models": models,
    }, default=str)


def handle_model_command(cmd: str, client_id: str, payload_str: str) -> str:
    """
    Dispatch a model registry command and return a JSON response string.

    Called from the server main loop when a MODEL_* command is received.

    Args:
        cmd: TaskCmd value (MODEL_REGISTER, MODEL_LIST)
        client_id: Username / client identifier
        payload_str: JSON-encoded payload

    Returns:
        JSON string to send back to client via ZMQ
    """
    from .TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.MODEL_REGISTER:
            return _handle_register(client_id, payload_str)
        elif cmd == TaskCmd.MODEL_LIST:
            return _handle_list(client_id, payload_str)
        else:
            return json.dumps({"status": "error", "message": f"Unknown model command: {cmd}"})
    except Exception as e:
        logger.exception("Error handling model command %s", cmd)
        return json.dumps({"status": "error", "message": str(e)})
