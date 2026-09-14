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
from typing import Any

from umrs.model_registry.models import ModelScope, ModelType, ServingProfile
from umrs.model_registry.store import (
    AccessContext,
    ACLEnforcer,
    DuplicateModelError,
    ModelStore,
    PathNotAllowedError,
    PathValidator,
    PermissionDeniedError,
)
from umrs.shared.auth import create_auth_context_from_request
from umrs.shared.config import get_resolved_umrs_subdir

logger = logging.getLogger(__name__)

_CATALOG_LOCK = threading.Lock()
_ACL = ACLEnforcer()


class DurableModelCommandBackend:
    """PostgreSQL-backed adapter for ZMQ model commands."""

    def __init__(self, db_pool):
        self._db_pool = db_pool
        conn = self._db_pool.getconn()
        try:
            from umrs.model_registry import (
                ensure_schema as ensure_model_registry_schema,
            )

            ensure_model_registry_schema(conn)
            conn.commit()
        finally:
            self._db_pool.putconn(conn)

    def close(self) -> None:
        if self._db_pool is not None:
            self._db_pool.closeall()

    def _with_store(self, callback):
        conn = self._db_pool.getconn()
        try:
            store = ModelStore(conn, path_validator=PathValidator())
            return callback(store)
        finally:
            self._db_pool.putconn(conn)

    def register(self, client_id: str, payload_str: str) -> str:
        try:
            payload = _parse_register_payload(payload_str)
            validated = _validate_register_request(client_id, payload)
            scope = validated["scope"]
            model_type = ModelType(
                str(payload.get("model_type", "embedding")).strip().lower()
            )
            serving_profile = None
            if isinstance(payload.get("serving_profile"), dict):
                serving_profile = ServingProfile.from_dict(payload["serving_profile"])

            model = self._with_store(
                lambda store: store.create_model(
                    name=validated["name"],
                    scope=scope,
                    storage_ref=validated["normalized_path"],
                    model_type=model_type,
                    owner_id=validated["access_ctx"].user_id
                    if scope == ModelScope.USER
                    else None,
                    project_id=validated["project_id"]
                    if scope == ModelScope.PROJECT
                    else None,
                    huggingface_id=payload.get("huggingface_id"),
                    description=payload.get("description"),
                    serving_profile=serving_profile,
                    ctx=validated["access_ctx"],
                )
            )
        except (
            DuplicateModelError,
            PathNotAllowedError,
            PermissionDeniedError,
            ValueError,
        ) as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        logger.info(
            "Registered model in durable registry: %s at %s (scope=%s)",
            model.name,
            model.storage_ref,
            model.scope.value,
        )
        return json.dumps(
            {
                "status": "ok",
                "message": f"Model '{model.name}' registered at {model.storage_ref}",
                "model": model.to_dict(),
            },
            default=str,
        )

    def list_models(self, client_id: str, payload_str: str) -> str:
        try:
            payload = _parse_list_payload(payload_str)
            access_ctx = _build_access_context(client_id, payload)
            scope = _parse_scope(payload) if payload.get("scope") else None
            models = self._with_store(
                lambda store: store.list_models(
                    ctx=access_ctx,
                    scope=scope,
                )
            )
        except (PermissionDeniedError, ValueError) as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        return json.dumps(
            {
                "status": "ok",
                "count": len(models),
                "models": [model.to_dict() for model in models],
            },
            default=str,
        )


# ADR-053: JSON catalog readers live in umrs.shared.model_catalog (foundation);
# maintain the legacy private names as thin delegations so callers inside the
# server tree are byte-identical.
from umrs.shared.model_catalog import (  # noqa: F401 — _catalog_path re-exported for legacy import sites
    _catalog_path,
    load_catalog,
    save_catalog,
)


def _load_catalog() -> dict[str, Any]:
    """Legacy alias — the shared reader (ADR-053)."""
    return load_catalog()


def _save_catalog(catalog: dict[str, Any]) -> None:
    """Legacy alias — the shared writer (ADR-053)."""
    save_catalog(catalog)


def _project_memberships_path() -> str:
    return os.environ.get(
        "UMRS_MODEL_PROJECT_MEMBERSHIPS_FILE",
        get_resolved_umrs_subdir("model_project_memberships.json"),
    )


def _load_project_memberships() -> dict[str, set[str]]:
    raw_json = os.environ.get("UMRS_MODEL_PROJECT_MEMBERSHIPS")
    raw_data: Any = None
    if raw_json:
        raw_data = json.loads(raw_json)
    else:
        path = _project_memberships_path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                raw_data = json.load(handle)

    return _normalize_project_memberships(raw_data)


def _normalize_project_memberships(raw_data: Any) -> dict[str, set[str]]:
    if raw_data is None:
        return {}
    if isinstance(raw_data, dict) and isinstance(raw_data.get("users"), dict):
        raw_data = raw_data["users"]
    if not isinstance(raw_data, dict):
        raise ValueError("Project membership mapping must be a JSON object")

    memberships: dict[str, set[str]] = {}
    for username, projects in raw_data.items():
        memberships[str(username)] = _normalize_project_list(projects)
    return memberships


def _normalize_project_list(projects: Any) -> set[str]:
    if isinstance(projects, str):
        return {projects}
    if isinstance(projects, list):
        return {str(project_id) for project_id in projects}
    raise ValueError(
        "Project memberships must map usernames to a string or list of strings"
    )


def _build_access_context(client_id: str, payload: dict[str, Any]) -> AccessContext:
    auth_ctx = create_auth_context_from_request(
        client_id,
        token=payload.get("admin_token"),
        project_id=payload.get("project_id"),
    )
    memberships = _load_project_memberships().get(auth_ctx.username or client_id, set())
    return AccessContext(
        user_id=auth_ctx.username or client_id,
        is_admin=auth_ctx.is_admin,
        project_memberships=memberships,
    )


def _parse_scope(payload: dict[str, Any]) -> ModelScope:
    raw_scope = str(payload.get("scope", "user")).strip().lower()
    try:
        return ModelScope(raw_scope)
    except ValueError as exc:
        raise ValueError(f"Invalid model scope: {raw_scope}") from exc


def _catalog_identity(entry: dict[str, Any], scope: ModelScope) -> str | None:
    if scope == ModelScope.USER:
        return entry.get("owner_id") or entry.get("registered_by")
    if scope == ModelScope.PROJECT:
        return entry.get("project_id")
    return None


def _entry_scope_key(
    scope: ModelScope, owner_id: str | None, project_id: str | None
) -> str | None:
    if scope == ModelScope.USER:
        return owner_id
    if scope == ModelScope.PROJECT:
        return project_id
    return None


def _find_duplicate(
    catalog: dict[str, Any],
    name: str,
    scope: ModelScope,
    owner_id: str | None,
    project_id: str | None,
) -> bool:
    expected_key = _entry_scope_key(scope, owner_id, project_id)
    for existing in catalog.get("models", []):
        if existing.get("name") != name or existing.get("scope") != scope.value:
            continue
        if scope == ModelScope.SYSTEM:
            return True
        if _catalog_identity(existing, scope) == expected_key:
            return True
    return False


def _parse_register_payload(payload_str: str) -> dict[str, Any]:
    try:
        return json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def _parse_list_payload(payload_str: str) -> dict[str, Any]:
    try:
        return json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        return {}


def _validate_register_request(
    client_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    name = payload.get("name", "").strip()
    path = payload.get("path", "").strip()
    project_id = payload.get("project_id")
    scope = _parse_scope(payload)
    access_ctx = _build_access_context(client_id, payload)
    _ACL.enforce_create(scope, access_ctx, project_id)

    if not name:
        raise ValueError("Model name is required")
    if not path:
        raise ValueError("Model path is required")

    normalized_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(normalized_path):
        logger.warning(
            "Path not reachable from UMRS server (may be on HPC filesystem): %s",
            normalized_path,
        )

    PathValidator().validate_scope_path(
        normalized_path,
        scope,
        owner_id=access_ctx.user_id if scope == ModelScope.USER else None,
        project_id=project_id if scope == ModelScope.PROJECT else None,
    )
    if scope == ModelScope.PROJECT:
        PathValidator(["/shared/projects/{project_id}/models/*"]).validate_scope_path(
            normalized_path, scope, project_id=project_id
        )
    return {
        "access_ctx": access_ctx,
        "name": name,
        "normalized_path": normalized_path,
        "project_id": project_id,
        "scope": scope,
    }


def _build_catalog_entry(
    *,
    client_id: str,
    payload: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any]:
    scope = validated["scope"]
    access_ctx = validated["access_ctx"]
    project_id = validated["project_id"]
    return {
        "name": validated["name"],
        "storage_ref": validated["normalized_path"],
        "model_type": payload.get("model_type", "embedding"),
        "scope": scope.value,
        "owner_id": access_ctx.user_id if scope == ModelScope.USER else None,
        "project_id": project_id if scope == ModelScope.PROJECT else None,
        "description": payload.get("description"),
        "huggingface_id": payload.get("huggingface_id"),
        "registered_by": client_id,
        "registered_at": datetime.now().isoformat(),
    }


def _resolve_target_path(
    *,
    scope: ModelScope,
    owner_id: str,
    project_id: str | None,
    model_name: str,
    staging_dir: str | None,
    validator: PathValidator,
) -> str:
    if staging_dir:
        root_dir = os.path.abspath(os.path.expanduser(staging_dir))
    elif scope == ModelScope.SYSTEM:
        root_dir = os.path.abspath(
            os.environ.get("UMRS_MODEL_STAGING_ROOT", "/shared/models/system")
        )
    elif scope == ModelScope.PROJECT:
        root_dir = os.path.join("/shared/projects", project_id or "", "models")
    else:
        root_dir = os.path.join("/shared/users", owner_id, "models")

    target_path = os.path.abspath(os.path.join(root_dir, model_name))
    validator.validate_scope_path(
        target_path,
        scope,
        owner_id=owner_id if scope == ModelScope.USER else None,
        project_id=project_id if scope == ModelScope.PROJECT else None,
    )
    return target_path


def _is_visible(entry: dict[str, Any], access_ctx: AccessContext) -> bool:
    if access_ctx.is_admin:
        return True
    scope = entry.get("scope")
    if scope == ModelScope.SYSTEM.value:
        return True
    if scope == ModelScope.USER.value:
        owner_id = entry.get("owner_id") or entry.get("registered_by")
        return owner_id == access_ctx.user_id
    if scope == ModelScope.PROJECT.value:
        return entry.get("project_id") in access_ctx.project_memberships
    return False


def _visible_models(
    models: list[dict[str, Any]], access_ctx: AccessContext
) -> list[dict[str, Any]]:
    return [model for model in models if _is_visible(model, access_ctx)]


def _filter_models_by_scope(
    models: list[dict[str, Any]], scope_filter: str | None
) -> list[dict[str, Any]]:
    if not scope_filter:
        return models
    return [model for model in models if model.get("scope") == scope_filter]


def _handle_register(
    client_id: str, payload_str: str, backend: DurableModelCommandBackend | None = None
) -> str:
    """Register a local model path in the catalog."""
    if backend is not None:
        return backend.register(client_id, payload_str)

    try:
        payload = _parse_register_payload(payload_str)
        validated = _validate_register_request(client_id, payload)
    except (PathNotAllowedError, PermissionDeniedError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})

    entry = _build_catalog_entry(
        client_id=client_id, payload=payload, validated=validated
    )
    scope = validated["scope"]
    project_id = validated["project_id"]

    with _CATALOG_LOCK:
        catalog = _load_catalog()
        if _find_duplicate(
            catalog, entry["name"], scope, entry.get("owner_id"), project_id
        ):
            scope_label = (
                scope.value
                if scope != ModelScope.PROJECT
                else f"{scope.value}/{project_id}"
            )
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Model '{entry['name']}' already registered in scope '{scope_label}'",
                }
            )
        catalog["models"].append(entry)
        _save_catalog(catalog)

    logger.info(
        "Registered model: %s at %s (scope=%s)",
        entry["name"],
        entry["storage_ref"],
        scope.value,
    )
    return json.dumps(
        {
            "status": "ok",
            "message": f"Model '{entry['name']}' registered at {entry['storage_ref']}",
            "model": entry,
        },
        default=str,
    )


def _handle_list(
    client_id: str, payload_str: str, backend: DurableModelCommandBackend | None = None
) -> str:
    """List registered models, optionally filtered by scope."""
    if backend is not None:
        return backend.list_models(client_id, payload_str)

    payload = _parse_list_payload(payload_str)

    with _CATALOG_LOCK:
        catalog = _load_catalog()
    models = catalog.get("models", [])

    access_ctx = _build_access_context(client_id, payload)
    models = _visible_models(models, access_ctx)
    models = _filter_models_by_scope(models, payload.get("scope"))

    return json.dumps(
        {
            "status": "ok",
            "count": len(models),
            "models": models,
        },
        default=str,
    )


def _handle_download(
    client_id: str,
    payload_str: str,
    backend: DurableModelCommandBackend | None = None,
) -> str:
    """Accept a server-side download request, run it in a background thread.

    Returns immediately with an "accepted" response. The download runs
    asynchronously; upon completion the model is auto-registered.
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})

    repo_id = payload.get("repo_id", "").strip()
    target_name = payload.get("name", "").strip()
    project_id = payload.get("project_id")
    staging_dir = payload.get("staging_dir")

    if not repo_id:
        return json.dumps({"status": "error", "message": "repo_id is required"})
    if not target_name:
        target_name = repo_id.split("/")[-1]

    try:
        scope = _parse_scope(payload)
        access_ctx = _build_access_context(client_id, payload)
        _ACL.enforce_create(scope, access_ctx, project_id)
        target_path = _resolve_target_path(
            scope=scope,
            owner_id=access_ctx.user_id,
            project_id=project_id,
            model_name=target_name,
            staging_dir=staging_dir,
            validator=PathValidator(),
        )
    except (PermissionDeniedError, PathNotAllowedError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})

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

            register_payload = json.dumps(
                {
                    "name": target_name,
                    "path": os.path.abspath(target_path),
                    "model_type": "embedding",
                    "scope": scope.value,
                    "project_id": project_id,
                    "huggingface_id": repo_id,
                    "description": f"Downloaded from {repo_id}",
                    "admin_token": payload.get("admin_token"),
                }
            )
            result = _handle_register(client_id, register_payload, backend=backend)
            logger.info("Server-side download result: %s", result)
        except Exception:
            logger.exception("Server-side download failed for %s", repo_id)

    t = threading.Thread(
        target=_bg_download, daemon=True, name=f"model-dl-{target_name}"
    )
    t.start()

    return json.dumps(
        {
            "status": "ok",
            "message": f"Download accepted: {repo_id} → {target_path} (running in background)",
        }
    )


def build_durable_model_backend_from_env() -> DurableModelCommandBackend | None:
    """Best-effort bootstrap of the durable model backend from process env."""
    try:
        from umrs.server.job_tracking_handler import build_durable_job_db_pool_from_env

        db_pool = build_durable_job_db_pool_from_env()
        logger.info("Durable ZMQ model backend enabled")
        return DurableModelCommandBackend(db_pool)
    except Exception:
        logger.exception("Durable ZMQ model backend unavailable")
        return None


def handle_model_command(
    cmd: str,
    client_id: str,
    payload_str: str,
    backend: DurableModelCommandBackend | None = None,
) -> str:
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
            return _handle_register(client_id, payload_str, backend=backend)
        elif cmd == TaskCmd.MODEL_LIST:
            return _handle_list(client_id, payload_str, backend=backend)
        elif cmd == TaskCmd.MODEL_DOWNLOAD:
            return _handle_download(client_id, payload_str, backend=backend)
        else:
            return json.dumps(
                {"status": "error", "message": f"Unknown model command: {cmd}"}
            )
    except Exception as e:
        logger.exception("Error handling model command %s", cmd)
        return json.dumps({"status": "error", "message": str(e)})
