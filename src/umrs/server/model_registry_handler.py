"""
Model Registry Handler — bridges ZMQ protocol to model registry operations.

Provides handle_model_command() for the server to call from its ZMQ loop.
Uses a JSON file catalog for lightweight model registration (no PostgreSQL
required). When the full model_registry.store is available (PostgreSQL),
it can be used instead.

Supported commands (via TaskCmd):
  MODEL_REGISTER  — register a local model path
  MODEL_LIST      — list all registered models
    MODEL_DOWNLOAD  — download from HuggingFace in a background thread, then register

model download has two modes:
  - Client-side (default):  client downloads locally, then sends MODEL_REGISTER
  - Server-side (--server-side): client sends MODEL_DOWNLOAD, server downloads
    in a background thread and replies immediately with "accepted"
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict

from umrs.shared.config import get_resolved_model_staging_root, get_resolved_umrs_subdir

logger = logging.getLogger(__name__)

_CATALOG_LOCK = threading.Lock()


def _catalog_path() -> str:
    """Return path to the JSON model catalog file."""
    return get_resolved_umrs_subdir("model_catalog.json")


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
    """List registered models, optionally filtered by scope."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        payload = {}

    with _CATALOG_LOCK:
        catalog = _load_catalog()
    models = catalog.get("models", [])

    scope_filter = payload.get("scope") if isinstance(payload, dict) else None
    if scope_filter:
        models = [m for m in models if m.get("scope") == scope_filter]

    return json.dumps({
        "status": "ok",
        "count": len(models),
        "models": models,
    }, default=str)


def _handle_download(client_id: str, payload_str: str) -> str:
    """Accept a server-side download request, run it in a background thread.

    Returns immediately with an "accepted" response.  The download runs
    asynchronously; upon completion the model is auto-registered.
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})

    repo_id = payload.get("repo_id", "").strip()
    target_name = payload.get("name", "").strip()
    model_type = payload.get("model_type", "embedding")
    scope = payload.get("scope", "system")

    if not repo_id:
        return json.dumps({"status": "error", "message": "repo_id is required"})
    if not target_name:
        target_name = repo_id.split("/")[-1]

    staging_root = os.environ.get("UMRS_MODEL_STAGING_ROOT")
    if not staging_root:
        staging_root = get_resolved_model_staging_root()
    target_path = os.path.join(staging_root, target_name)

    def _bg_download():
        """Background thread: download from HF, then register."""
        try:
            if os.path.exists(target_path) and os.listdir(target_path):
                logger.info("Model already at %s, registering only", target_path)
            else:
                from huggingface_hub import snapshot_download
                os.makedirs(target_path, exist_ok=True)
                logger.info("Downloading %s → %s ...", repo_id, target_path)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    token=os.environ.get("HF_TOKEN"),
                )
                logger.info("Download complete: %s", target_path)

            register_payload = json.dumps({
                "name": target_name,
                "path": os.path.abspath(target_path),
                "model_type": model_type,
                "scope": scope,
                "huggingface_id": repo_id,
                "description": f"Downloaded from {repo_id}",
            })
            result = _handle_register(client_id, register_payload)
            logger.info("Server-side download result: %s", result)
        except Exception:
            logger.exception("Server-side download failed for %s", repo_id)

    t = threading.Thread(target=_bg_download, daemon=True,
                         name=f"model-dl-{target_name}")
    t.start()

    return json.dumps({
        "status": "ok",
        "message": f"Download accepted: {repo_id} → {target_path} (running in background)",
    })


def handle_model_command(cmd: str, client_id: str, payload_str: str) -> str:
    """
    Dispatch a model registry command and return a JSON response string.

    Called from the server main loop when a MODEL_* command is received.

    Args:
        cmd: TaskCmd value (MODEL_REGISTER, MODEL_LIST, MODEL_DOWNLOAD)
        client_id: Username / client identifier
        payload_str: JSON-encoded payload

    Returns:
        JSON string to send back to client via ZMQ
    """
    from umrs.core.TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.MODEL_REGISTER:
            return _handle_register(client_id, payload_str)
        elif cmd == TaskCmd.MODEL_LIST:
            return _handle_list(client_id, payload_str)
        elif cmd == TaskCmd.MODEL_DOWNLOAD:
            return _handle_download(client_id, payload_str)
        else:
            return json.dumps({"status": "error", "message": f"Unknown model command: {cmd}"})
    except Exception as e:
        logger.exception("Error handling model command %s", cmd)
        return json.dumps({"status": "error", "message": str(e)})
