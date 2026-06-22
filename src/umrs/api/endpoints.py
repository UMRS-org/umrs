"""
Endpoint Registry REST API.

Provides:
- GET /umrs/endpoints/active - Query active endpoints for Gateway discovery
- POST /umrs/endpoints - Register an endpoint (internal use)
- DELETE /umrs/endpoints/{job_id} - Remove an endpoint (internal use)
- GET /umrs/endpoints/{job_id}/status - Get endpoint status
- PUT /umrs/endpoints/{job_id}/used - Mark endpoint as used (for TTL extension)
- GET /umrs/endpoints/stats - Registry statistics

Authentication:
- project_id is extracted from Authorization header (JWT claims)
- For MVP: Accept X-Project-ID header for testing
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator

from ..orchestrator.endpoint_registry import (
    EndpointRegistry,
    RegisteredEndpoint,
    EndpointStatus,
    get_registry,
)
from ..orchestrator.endpoint_mounts import validate_mount_source
from ..orchestrator.readiness import get_readiness_poller
from ..shared.auth import JWTError, resolve_jwt_secret, validate_jwt

logger = logging.getLogger(__name__)


def _strict_auth_mode() -> bool:
    """Return True when MVP header/default project fallback is disabled."""
    return os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"


def _raise_strict_project_auth_error() -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "UMRS_AUTH_MODE=strict requires a verified project claim; "
            "X-Project-ID and default-project fallbacks are disabled."
        ),
    )


def _header_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bearer_token(authorization: str | None) -> str | None:
    value = _header_str(authorization)
    if not value:
        return None
    parts = value.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def _jwt_claims(
    authorization: str | None,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    token = _bearer_token(authorization)
    if not token:
        if required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization bearer token",
            )
        return None
    if token.count(".") != 2:
        if required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization bearer token is not a JWT",
            )
        return None
    try:
        return validate_jwt(token, resolve_jwt_secret())
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def _claim_values(claims: dict[str, Any], *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        value = claims.get(name)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            values.extend(
                item.strip() for item in value if isinstance(item, str) and item.strip()
            )
    return values


def _claim_project_id(claims: dict[str, Any]) -> str | None:
    values = _claim_values(claims, "project_id", "project", "tenant_id", "tenant")
    return values[0] if values else None


def _claim_project_memberships(claims: dict[str, Any]) -> set[str]:
    memberships = set(
        _claim_values(
            claims,
            "project_memberships",
            "project_ids",
            "projects",
            "allowed_projects",
        )
    )
    project_id = _claim_project_id(claims)
    if project_id:
        memberships.add(project_id)
    return memberships


def _claim_is_admin(claims: dict[str, Any] | None) -> bool:
    if not claims:
        return False
    return "admin" in set(_claim_values(claims, "roles", "groups"))


def _claim_can_access_project(claims: dict[str, Any], project_id: str) -> bool:
    if _claim_is_admin(claims):
        return True
    memberships = _claim_project_memberships(claims)
    return "*" in memberships or project_id in memberships


def _forbid_if_project_mismatch(
    endpoint: RegisteredEndpoint,
    project_id: str | None,
) -> None:
    if project_id is not None and endpoint.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Endpoint not found",
        )


def _resolve_authenticated_project_id(
    authorization: str | None,
    x_user_id: str | None,
    x_verified_project_id: str | None,
    x_project_id: str | None,
    *,
    optional: bool = False,
) -> str | None:
    strict = _strict_auth_mode()
    claims = _jwt_claims(authorization, required=False)
    verified_user_id = _header_str(x_user_id)
    verified_project_id = _header_str(x_verified_project_id)
    requested_project_id = verified_project_id or _header_str(x_project_id)

    if strict and verified_project_id:
        return verified_project_id

    if claims:
        return _project_id_authorized_by_claims(claims, requested_project_id)

    if strict:
        if optional:
            return None
        _raise_strict_project_auth_error()

    return _legacy_project_id(authorization, x_project_id, optional=optional)


def _project_id_authorized_by_claims(
    claims: dict[str, Any],
    requested_project_id: str | None,
) -> str:
    effective_project_id = requested_project_id or _claim_project_id(claims)
    if not effective_project_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT is missing project_id claim",
        )
    if not _claim_can_access_project(claims, effective_project_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requested project is not allowed by JWT project memberships",
        )
    return effective_project_id


def _legacy_project_id(
    authorization: str | None,
    x_project_id: str | None,
    *,
    optional: bool,
) -> str | None:
    project_id = _header_str(x_project_id)
    if project_id:
        return project_id
    if authorization and authorization.startswith("Bearer "):
        if optional:
            return "default-project"
        logger.warning("JWT project_id extraction not implemented, using default")
        return "default-project"
    return None


def _header_bool(value: str | None) -> bool:
    normalized = _header_str(value)
    return normalized is not None and normalized.lower() in {"1", "true", "yes", "on"}


def _caller_is_admin(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
) -> bool:
    claims = _jwt_claims(authorization, required=False)
    if claims is not None:
        return _claim_is_admin(claims)
    return _header_str(x_user_id) is not None and _header_bool(x_verified_is_admin)


def _require_internal_endpoint_writer(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
) -> None:
    if not _strict_auth_mode():
        return
    if not _caller_is_admin(authorization, x_user_id, x_verified_is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Endpoint registry mutation requires admin JWT in strict mode",
        )


def _enforce_provision_authorization(
    request: "ProvisionRequest",
    *,
    is_admin: bool,
) -> None:
    if request.tier == "system" and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins may provision system-tier endpoints",
        )


def _derive_reasoning_parser(model_name: str) -> str:
    """Mirror serve.slurm's derive_reasoning_parser() for forced reasoning.

    Used only when the caller requests ``reasoning="on"`` so the parser is
    set explicitly even if a site disables the shell-side auto-derivation.
    Returns an empty string when the model family has no reasoning parser.
    """
    name_lc = (model_name or "").lower()
    if "glm" in name_lc:
        return "glm45"
    if "gemma" in name_lc and "4" in name_lc:
        return "gemma4"
    if "kimi" in name_lc:
        return "kimi_k2"
    return ""


router = APIRouter(prefix="/umrs/endpoints", tags=["endpoints"])

_DEFAULT_PROVISION_MODEL_PATH = "/models/rivault-gpt-oss"


# Backend job statuses considered "alive enough" to reuse a warm endpoint
# (registry says READY but the underlying job must also be in a non-terminal
# state).  Imported lazily inside the helper to avoid an import cycle.
async def _is_backend_job_alive(service: Any, job_id: str) -> bool:
    """Return True if the backend reports the job as alive.

    Used to guard against stale warm-endpoint reuse: an endpoint may be
    cached as READY in the registry while its Slurm/K8s job has already
    terminated.  Returns False on any verification error so the caller
    falls through to a fresh provisioning path (fail-safe default).
    """
    from ..orchestrator.base import JobStatus

    backend = getattr(service, "backend", None)
    if backend is None:
        return False
    try:
        info = await backend.get_job_status(job_id)
    except Exception as exc:  # noqa: BLE001 - defensive, treat as dead
        logger.warning(
            "Backend get_job_status(%s) failed during warm-endpoint check: %s",
            job_id,
            exc,
        )
        return False
    status_value = getattr(info, "status", None)
    return status_value in (
        JobStatus.PENDING,
        JobStatus.STARTING,
        JobStatus.READY,
        JobStatus.RUNNING,
    )


async def _backend_job_liveness(service: Any, job_id: str) -> bool | None:
    """Return backend liveness as alive / dead / unknown.

    ``True`` means the backend positively reports a non-terminal job.
    ``False`` means the backend positively reports a terminal or unknown job.
    ``None`` means we could not verify backend state and should preserve the
    registry row rather than deleting a potentially valid endpoint.
    """
    from ..orchestrator.base import JobStatus

    backend = getattr(service, "backend", None)
    if backend is None:
        return None

    try:
        info = await backend.get_job_status(job_id)
    except Exception as exc:  # noqa: BLE001 - preserve on verification errors
        logger.warning(
            "Backend get_job_status(%s) failed during active-endpoint verification: %s",
            job_id,
            exc,
        )
        return None

    status_value = getattr(info, "status", None)
    return status_value in (
        JobStatus.PENDING,
        JobStatus.STARTING,
        JobStatus.READY,
        JobStatus.RUNNING,
    )


async def _filter_live_ready_endpoints(
    registry: EndpointRegistry,
    endpoints: list[Any],
) -> list[Any]:
    """Return only endpoints whose backing jobs are still alive.

    ``/umrs/endpoints/active`` is a discovery surface, so returning a stale
    READY record after a UMRS restart is worse than returning nothing. When we
    can initialize the orchestration service, verify every READY/DEGRADED row
    against the backend and remove ghosts from the registry backend.
    """
    if not endpoints:
        return endpoints

    from ..orchestrator.service import get_orchestration_service

    try:
        service = get_orchestration_service()
    except Exception as exc:  # noqa: BLE001 - keep discovery available
        logger.warning(
            "Could not initialize orchestration service during active endpoint verification: %s",
            exc,
        )
        return endpoints

    live_endpoints: list[Any] = []
    for endpoint in endpoints:
        is_alive = await _backend_job_liveness(service, endpoint.job_id)
        if is_alive is None:
            live_endpoints.append(endpoint)
            continue
        if is_alive:
            live_endpoints.append(endpoint)
            continue

        logger.warning(
            "Removing stale active endpoint %s/%s (job_id=%s): backend job no longer alive",
            endpoint.project_id,
            endpoint.model,
            endpoint.job_id,
        )
        registry.remove(endpoint.job_id)

    return live_endpoints


_REFRESHABLE_ENDPOINT_STATUSES = {
    EndpointStatus.PENDING,
    EndpointStatus.PROVISIONING,
    EndpointStatus.STARTING,
}


_READINESS_DETAIL_RULES = (
    (
        ("connection error", "connection refused"),
        "starting LiteLLM / waiting for service port",
    ),
    (
        ("chat probe timeout", "request timeout"),
        "loading model weights / waiting for inference warmup",
    ),
    (
        ("http 404", "http 500", "http 502", "http 503"),
        "starting LiteLLM / waiting for proxy readiness",
    ),
    (
        ("no models in response", "expected model"),
        "registering served model / loading model",
    ),
    (
        ("chat probe http", "chat probe error"),
        "endpoint reachable; waiting for inference readiness",
    ),
)


def _short_readiness_error(error: str | None, *, limit: int = 96) -> str | None:
    if not error:
        return None
    cleaned = " ".join(error.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _readiness_status_detail(endpoint: Any, error: str | None) -> str:
    target = f"{endpoint.host}:{endpoint.port}"
    if not error:
        return f"waiting for endpoint readiness ({target})"

    lowered = error.lower()
    for needles, detail in _READINESS_DETAIL_RULES:
        if any(needle in lowered for needle in needles):
            return detail

    return f"waiting for endpoint readiness ({target}): {_short_readiness_error(error, limit=72)}"


def _endpoint_can_refresh_readiness(endpoint: Any) -> bool:
    return (
        endpoint.status in _REFRESHABLE_ENDPOINT_STATUSES
        and bool(endpoint.host)
        and endpoint.host != "pending"
    )


async def _refresh_endpoint_readiness(
    registry: EndpointRegistry,
    endpoint: Any,
) -> Any:
    if not _endpoint_can_refresh_readiness(endpoint):
        return endpoint

    readiness_probe = str(
        endpoint.metadata.get("readiness_probe")
        or _resolve_readiness_probe_mode(endpoint.model)
    )
    readiness_path = endpoint.metadata.get("readiness_path")

    try:
        is_ready, is_degraded, models, error = await get_readiness_poller().check_ready_detail(
            endpoint.host,
            endpoint.port,
            probe_model=endpoint.metadata.get("served_model_name") or endpoint.model,
            probe_kind=readiness_probe,
            probe_path=str(readiness_path) if readiness_path else None,
        )
    except Exception as exc:  # noqa: BLE001 - status lookup must stay best-effort
        logger.warning(
            "Endpoint readiness refresh failed for %s: %s", endpoint.job_id, exc
        )
        return endpoint

    if is_ready:
        registry.update_metadata(
            endpoint.job_id,
            {
                "detail": "endpoint ready",
                "readiness_error": None,
                "readiness_models": models,
            },
        )
        return registry.mark_ready(endpoint.job_id) or endpoint

    if is_degraded:
        registry.update_metadata(
            endpoint.job_id,
            {
                "detail": f"degraded: {_readiness_status_detail(endpoint, error)}",
                "readiness_error": _short_readiness_error(error),
                "readiness_models": models,
            },
        )
        return registry.mark_degraded(endpoint.job_id) or endpoint

    registry.update_metadata(
        endpoint.job_id,
        {
            "detail": _readiness_status_detail(endpoint, error),
            "readiness_error": _short_readiness_error(error),
        },
    )
    return registry.get_by_job_id(endpoint.job_id) or endpoint


# =============================================================================
# Request/Response Models
# =============================================================================


class EndpointResponse(BaseModel):
    """Endpoint data for API responses."""

    project_id: str
    model: str
    job_id: str
    host: str
    port: int
    status: str
    created_at: str
    ready_at: str | None = None
    last_used_at: str | None = None
    expires_at: str | None = None
    backend: str
    error_message: str | None = None
    litellm_url: str
    owner_type: str | None = None
    owner_id: str | None = None
    tier: str | None = None
    served_model_name: str | None = None
    model_type: str | None = None
    serving_backend: str | None = None
    service_kind: str | None = None
    service_url: str | None = None
    invoke_base_path: str | None = None
    readiness_probe: str | None = None
    readiness_path: str | None = None
    launch_profiles: list[str] | None = None
    container_image: str | None = None
    vision_adapter: str | None = None
    vision_task: str | None = None
    output_format: str | None = None
    detail: str | None = None
    job_status: str | None = None
    max_model_len: int | None = None
    variant: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "project_id": "proj-123",
                "model": "rivault-gpt-oss-static",
                "job_id": "slurm-456",
                "host": "compute-node-01",
                "port": 11434,
                "status": "ready",
                "created_at": "2024-01-15T10:30:00Z",
                "ready_at": "2024-01-15T10:31:00Z",
                "last_used_at": "2024-01-15T10:35:00Z",
                "expires_at": "2024-01-15T11:30:00Z",
                "backend": "slurm",
                "error_message": None,
                "litellm_url": "http://compute-node-01:11434/v1",
                "owner_type": "user",
                "owner_id": None,
                "tier": "user",
                "detail": "starting container / loading model",
                "job_status": "starting",
            }
        }


class EndpointsListResponse(BaseModel):
    """Response for endpoint listing."""

    endpoints: list[EndpointResponse]
    total: int


class EndpointRegisterRequest(BaseModel):
    """Request to register a new endpoint."""

    project_id: str = Field(..., description="Project/tenant identifier")
    model: str = Field(..., description="Model name (e.g., rivault-gpt-oss-static)")
    job_id: str = Field(..., description="Backend job identifier")
    host: str = Field(..., description="Hostname or IP address")
    port: int = Field(..., ge=1, le=65535, description="LiteLLM port")
    backend: str = Field(
        "unknown", description="Backend type (slurm/kubernetes/compose)"
    )
    ttl_sec: int | None = Field(
        None,
        ge=0,
        description="TTL in seconds (0 = no expiry; None = server default)",
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class EndpointStatusUpdate(BaseModel):
    """Request to update endpoint status."""

    status: str = Field(..., description="New status")
    error_message: str | None = Field(None, description="Error message if failed")


class LoraSpec(BaseModel):
    """LoRA adapter specification for multi-lora endpoints."""

    name: str = Field(
        ...,
        min_length=1,
        description="Adapter name (served as `model` over OpenAI API)",
    )
    source: str = Field(
        ..., min_length=1, description="Filesystem path or registered LoRA model name"
    )


class EndpointMountSpec(BaseModel):
    """Read-only host mount requested for a provisioned endpoint."""

    source: str = Field(
        ..., min_length=1, description="Host/shared filesystem source path"
    )
    target: str = Field(..., min_length=1, description="Container target path")

    @model_validator(mode="after")
    def _validate_paths(self) -> "EndpointMountSpec":
        try:
            self.source = validate_mount_source(self.source)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if not self.target.startswith("/"):
            raise ValueError("mount target must be an absolute container path")
        return self


class EndpointMounts(BaseModel):
    """Mount bundle for mini-RiVault MCP and RAG side services."""

    mcp_servers: list[EndpointMountSpec] = Field(default_factory=list)
    project_data: EndpointMountSpec | None = None
    rag_index: EndpointMountSpec | None = None

    @model_validator(mode="after")
    def _validate_mount_targets(self) -> "EndpointMounts":
        for mount in self.mcp_servers:
            if not _target_under(mount.target, "/mcp-servers"):
                raise ValueError("mcp_servers targets must be under /mcp-servers")
        if self.project_data and not _target_under(
            self.project_data.target, "/project"
        ):
            raise ValueError("project_data target must be under /project")
        if self.rag_index and not _target_under(self.rag_index.target, "/rag-index"):
            raise ValueError("rag_index target must be under /rag-index")
        return self


def _target_under(target: str, root: str) -> bool:
    normalized = target.rstrip("/") or "/"
    return normalized == root or normalized.startswith(root + "/")


class ProvisionRequest(BaseModel):
    """Request to provision a new endpoint (U-202)."""

    project_id: str | None = Field(
        None,
        description=(
            "Project/tenant identifier. Optional when the Gateway provides "
            "X-Project-ID or X-Verified-Project-ID."
        ),
    )
    model: str = Field(..., description="Model name (primary; equals models[0])")
    models: list[str] | None = Field(
        None,
        description=(
            "Optional list of base models for multi-model endpoints. "
            "Defaults to ``[model]``. The first entry must equal ``model``."
        ),
    )
    serving_mode: str = Field(
        "single",
        description="One of: single, multi-lora, co-located",
    )
    serving_backend: str | None = Field(
        None,
        description="Serving backend override: vllm or bentoml",
    )
    service_kind: str | None = Field(
        None,
        description="Service API kind override: openai or bentoml_vision",
    )
    launch_profiles: list[str] | None = Field(
        None,
        description="Mini-RiVault profiles to activate for this endpoint",
    )
    co_locate_vllm: bool = Field(
        False,
        description="For BentoML vision, also launch the vLLM profile",
    )
    container_image: str | None = Field(None, description="Container image override")
    service_port: int | None = Field(
        None, ge=1, le=65535, description="External service port override"
    )
    readiness_probe: str | None = Field(
        None, description="Readiness probe override: models, chat, or http"
    )
    readiness_path: str | None = Field(
        None, description="HTTP readiness path for non-OpenAI services"
    )
    invoke_base_path: str | None = Field(
        None, description="Base path for invoking the service"
    )
    vision_adapter: str | None = Field(None, description="Vision adapter name")
    vision_task: str | None = Field(None, description="Default vision task")
    output_format: str | None = Field(None, description="Default vision result format")
    loras: list[LoraSpec] | None = Field(
        None,
        description="LoRA adapters (multi-lora mode only)",
    )
    tier: str = Field("user", description="Deployment tier: system, project, or user")
    mounts: EndpointMounts | None = Field(
        None,
        description="Read-only MCP/RAG mounts for user/project mini-RiVault instances",
    )
    egress_domains: list[str] = Field(
        default_factory=list,
        description="Domain allowlist for EgressGuard-enabled side services",
    )
    model_path: str | None = Field(
        None,
        description="Optional path to model files. If omitted, resolve from the registered model catalog.",
    )
    ttl_sec: int = Field(
        3600,
        ge=0,
        description="TTL in seconds (0 = no expiry / never auto-recycle)",
    )
    gpu_count: int = Field(1, ge=1, description="Number of GPUs per node")
    node_count: int = Field(
        1,
        ge=1,
        description="Number of Slurm nodes (>1 for multi-node pipeline parallelism)",
    )
    max_model_len: int = Field(4096, ge=512, description="Max context length")
    prefer_warm: bool = Field(True, description="Prefer reusing existing endpoint")
    variant: str | None = Field(
        None,
        description=(
            "Deployment variant name (e.g. '128k', 'tep-128k'). "
            "Differentiates multiple endpoints for the same model with "
            "different parameters (ctx length, node count, etc.). "
            "When set, warm-reuse only considers endpoints with the same variant."
        ),
    )
    wait: bool = Field(True, description="Wait for readiness before responding")
    partition: str | None = Field(
        None, description="Override Slurm partition (must be in site candidates)"
    )
    nodelist: str | None = Field(
        None, description="Slurm --nodelist hint (e.g. 'ai-h200-brc-00')"
    )
    parallel_strategy: str | None = Field(
        None,
        description=(
            "Multi-node parallelism layout: 'tp_pp' (default; TP=GPUs/node, "
            "PP=node count) or 'tep' (tensor+expert parallel: TP=world size, "
            "PP=1, --enable-expert-parallel). Forwarded to serve.slurm via "
            "VLLM_PARALLEL_STRATEGY."
        ),
    )
    reasoning: str | None = Field(
        None,
        description=(
            "Reasoning-parser control for thinking models: 'auto' (default; "
            "server derives the parser from the model name), 'on' (force the "
            "model's native reasoning parser), or 'off' (disable reasoning "
            "splitting). Forwarded to serve.slurm via VLLM_REASONING_PARSER."
        ),
    )

    @model_validator(mode="after")
    def _validate_multi_model(self) -> "ProvisionRequest":
        # Normalize/validate models[]
        if self.models is None:
            self.models = [self.model]
        if not self.models:
            raise ValueError("models must be a non-empty list")
        if any(not m for m in self.models):
            raise ValueError("models entries must be non-empty strings")
        if self.models[0] != self.model:
            raise ValueError("models[0] must equal model")

        if self.serving_mode not in ("single", "multi-lora", "co-located"):
            raise ValueError(
                "serving_mode must be one of: single, multi-lora, co-located"
            )
        if self.serving_backend and self.serving_backend not in ("vllm", "bentoml"):
            raise ValueError("serving_backend must be one of: vllm, bentoml")
        if self.service_kind and self.service_kind not in ("openai", "bentoml_vision"):
            raise ValueError("service_kind must be one of: openai, bentoml_vision")
        if self.readiness_probe and self.readiness_probe not in (
            "models",
            "chat",
            "http",
            "bentoml",
        ):
            raise ValueError(
                "readiness_probe must be one of: models, chat, http, bentoml"
            )
        if self.tier not in ("system", "project", "user"):
            raise ValueError("tier must be one of: system, project, user")
        if self.reasoning is not None and self.reasoning not in ("auto", "on", "off"):
            raise ValueError("reasoning must be one of: auto, on, off")
        if self.serving_mode == "single" and len(self.models) != 1:
            raise ValueError("serving_mode=single requires exactly one model")
        if self.serving_mode == "co-located" and len(self.models) < 2:
            raise ValueError("serving_mode=co-located requires at least 2 models")

        if self.loras:
            if self.serving_mode != "multi-lora":
                raise ValueError("loras requires serving_mode=multi-lora")
            if len(self.models) != 1:
                raise ValueError("multi-lora requires exactly one base model")
            names = [lora.name for lora in self.loras]
            if len(set(names)) != len(names):
                raise ValueError("LoRA names must be unique")
        elif self.serving_mode == "multi-lora":
            raise ValueError("serving_mode=multi-lora requires a non-empty loras list")
        return self


class ProvisionResponse(BaseModel):
    """Response for provision request."""

    status: str = Field(..., description="Status: ready, provisioning, failed")
    job_id: str | None = Field(None, description="Job ID if provisioning started")
    endpoint: EndpointResponse | None = Field(None, description="Endpoint if ready")
    retry_after: int | None = Field(None, description="Seconds to wait before retry")
    error: str | None = Field(None, description="Error message if failed")


class RegistryStatsResponse(BaseModel):
    """Registry statistics."""

    total: int
    by_status: dict[str, int]
    by_backend: dict[str, int]
    ready_count: int


class EndpointHealthInfo(BaseModel):
    """Health information for a single endpoint."""

    job_id: str
    project_id: str
    model: str
    status: str
    host: str
    port: int
    healthy: bool
    last_health_check: str | None = None
    uptime_seconds: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Comprehensive health response (U-204)."""

    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    registry_ok: bool
    total_endpoints: int
    healthy_endpoints: int
    unhealthy_endpoints: int
    provisioning_endpoints: int
    endpoints: list[EndpointHealthInfo] = Field(default_factory=list)
    message: str | None = None


# =============================================================================
# Dependencies
# =============================================================================


def get_project_id(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
) -> str:
    """
    Extract project_id from request headers.

    Production: Extract from JWT claims in Authorization header.
    MVP/Testing: Accept X-Project-ID header.

    Raises:
        HTTPException: If project_id cannot be determined.
    """
    project_id = _resolve_authenticated_project_id(
        authorization,
        x_user_id,
        x_verified_project_id,
        x_project_id,
    )
    if project_id:
        return project_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing project_id. Provide X-Project-ID header or valid JWT.",
    )


def get_project_id_optional(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
) -> str | None:
    """Get project_id if available, None otherwise (for admin endpoints)."""
    return _resolve_authenticated_project_id(
        authorization,
        x_user_id,
        x_verified_project_id,
        x_project_id,
        optional=True,
    )


def _get_requesting_user(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
) -> str | None:
    """Extract the requesting username from JWT ``sub`` or verified header.

    Returns the username string, or None if the caller is unauthenticated
    (non-strict mode).
    """
    claims = _jwt_claims(authorization, required=False)
    if claims is not None:
        return claims.get("sub") or claims.get("user_id")
    return _header_str(x_user_id)


def _resolve_provision_project_id(
    request: ProvisionRequest,
    authenticated_project_id: str | None,
) -> str:
    """Resolve the project that owns an endpoint provision request."""
    if authenticated_project_id is not None:
        if request.project_id and request.project_id != authenticated_project_id:
            logger.warning(
                "Ignoring endpoint provision body project_id=%s in favor of authenticated project_id=%s",
                request.project_id,
                authenticated_project_id,
            )
        return authenticated_project_id

    if _strict_auth_mode():
        _raise_strict_project_auth_error()

    body_project_id = _header_str(request.project_id)
    if body_project_id:
        return body_project_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing project_id. Provide X-Project-ID header or project_id in the request body.",
    )


def _resolve_served_model_name(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
) -> str | None:
    """Resolve served_model_name from model catalog metadata."""
    entry = model_entry or _resolve_registered_model_entry(model_name)
    if not entry:
        return model_name

    serving_profile = entry.get("serving_profile") or {}
    served_model_name = serving_profile.get("served_model_name")
    if served_model_name:
        return str(served_model_name)

    # Default to the catalog alias. Hugging Face IDs identify the source model,
    # not necessarily the runtime API name exposed by vLLM.
    return model_name


def _resolve_registered_model_entry(model_name: str) -> dict[str, Any] | None:
    """Resolve a registered model entry from the authoritative UMRS catalog."""
    try:
        from ..model_registry.store import AccessContext
        from .models import get_model_store

        store = get_model_store()
        matches = [
            entry.to_dict()
            for entry in store.list_models(
                AccessContext(user_id="", is_admin=True),
                limit=500,
            )
            if entry.name == model_name
        ]
        if matches:
            scope_order = {"system": 0, "project": 1, "user": 2}
            matches.sort(
                key=lambda entry: scope_order.get(str(entry.get("scope", "system")), 99)
            )
            return matches[0]
    except Exception as e:
        logger.debug(
            f"Failed to resolve model entry from model store for {model_name}: {e}"
        )

    try:
        from umrs.server.model_registry_handler import _load_catalog

        matches = [
            entry
            for entry in _load_catalog().get("models", [])
            if entry.get("name") == model_name
        ]
        if not matches:
            return None

        scope_order = {"system": 0, "project": 1, "user": 2}
        matches.sort(
            key=lambda entry: scope_order.get(str(entry.get("scope", "system")), 99)
        )
        return matches[0]
    except Exception as e:
        logger.debug(f"Failed to resolve model entry for {model_name}: {e}")
    return None


def _resolve_requested_model_path(
    model_name: str,
    model_path: str | None,
    model_entry: dict[str, Any] | None = None,
) -> str | None:
    """Resolve the actual model storage path for endpoint provisioning."""
    normalized_path = model_path.strip() if model_path else None
    if normalized_path and normalized_path != _DEFAULT_PROVISION_MODEL_PATH:
        return _validated_model_path_override(normalized_path)

    entry = model_entry or _resolve_registered_model_entry(model_name)
    if entry and entry.get("storage_ref"):
        return entry["storage_ref"]

    if model_name.startswith("/"):
        return _validated_model_path_override(model_name)

    return None


def _validated_model_path_override(model_path: str) -> str:
    if "://" in model_path and not model_path.startswith("file://"):
        return model_path
    try:
        return validate_mount_source(model_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_path: {exc}",
        ) from exc


def _serving_profile(model_entry: dict[str, Any] | None) -> dict[str, Any]:
    profile = (model_entry or {}).get("serving_profile") or {}
    return profile if isinstance(profile, dict) else {}


def _resolve_model_type(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
) -> str:
    entry = model_entry or _resolve_registered_model_entry(model_name)
    return str((entry or {}).get("model_type") or "llm").lower()


def _resolve_serving_backend(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_backend = _serving_profile(entry).get("serving_backend")
    if profile_backend:
        return str(profile_backend).lower()
    if _resolve_model_type(model_name, entry) == "vision":
        return "bentoml"
    return "vllm"


def _resolve_service_kind(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_kind = _serving_profile(entry).get("service_kind")
    if profile_kind:
        return str(profile_kind)
    backend = serving_backend or _resolve_serving_backend(model_name, entry)
    if backend == "bentoml" or _resolve_model_type(model_name, entry) == "vision":
        return "bentoml_vision"
    return "openai"


def _profile_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.replace(",", " ").split() if part]
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if str(part)]
    return []


def _active_site_profiles() -> list[str]:
    profiles: list[str] = []
    try:
        from ..orchestrator.site_config import get_site_config

        profiles.extend(get_site_config().runtime.profiles)
    except Exception:
        pass
    profiles.extend(_profile_list(os.environ.get("RIVAULT_PROFILES")))
    return list(dict.fromkeys(profiles))


def _with_site_support_profiles(profiles: list[str]) -> list[str]:
    resolved = list(profiles)
    if (
        resolved
        and "harness" in resolved
        and "litellm-db" not in resolved
        and "litellm-db" in _active_site_profiles()
    ):
        resolved.append("litellm-db")
    return resolved


def _resolve_launch_profiles(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    explicit: list[str] | None = None,
    co_locate_vllm: bool = False,
) -> list[str]:
    if explicit:
        return _profile_list(explicit)
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_profiles = _profile_list(_serving_profile(entry).get("launch_profiles"))
    if profile_profiles:
        return _with_site_support_profiles(profile_profiles)
    backend = serving_backend or _resolve_serving_backend(model_name, entry)
    if backend == "bentoml" or _resolve_model_type(model_name, entry) == "vision":
        profiles = ["harness", "vision-bentoml"]
        if co_locate_vllm:
            profiles.insert(1, "vllm")
        return _with_site_support_profiles(profiles)
    # LLM / embedding models are served by vLLM only. Return an explicit vllm
    # profile set (instead of an empty list) so the orchestrator always sets
    # RIVAULT_PROFILES for the job. Otherwise the Slurm wrapper falls back to a
    # site-wide default that also starts the BentoML vision sidecar, launching
    # the wrong service stack for a text model and starving vLLM of GPU memory.
    return _with_site_support_profiles(["harness", "vllm"])


def _normalize_readiness_probe(value: Any) -> str | None:
    if not value:
        return None
    normalized = str(value)
    return "http" if normalized == "bentoml" else normalized


def _is_bentoml_vision_service(
    model_name: str,
    model_entry: dict[str, Any] | None,
    *,
    serving_backend: str | None = None,
    service_kind: str | None = None,
) -> bool:
    entry = model_entry or _resolve_registered_model_entry(model_name)
    backend = serving_backend or _resolve_serving_backend(model_name, entry)
    kind = service_kind or _resolve_service_kind(
        model_name, entry, serving_backend=backend
    )
    return (
        backend == "bentoml"
        or kind == "bentoml_vision"
        or _resolve_model_type(model_name, entry) == "vision"
    )


def _resolve_readiness_probe_mode(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    explicit: str | None = None,
) -> str:
    """Resolve the readiness probe mode for a provisioned endpoint."""
    explicit_probe = _normalize_readiness_probe(explicit)
    if explicit_probe:
        return explicit_probe
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_probe = _normalize_readiness_probe(
        _serving_profile(entry).get("readiness_probe")
    )
    if profile_probe:
        return profile_probe
    model_type = _resolve_model_type(model_name, entry)
    if _is_bentoml_vision_service(model_name, entry, serving_backend=serving_backend):
        return "http"
    if model_type == "embedding":
        return "models"
    return "chat"


def _resolve_readiness_path(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    service_kind: str | None = None,
    readiness_probe: str | None = None,
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_path = _serving_profile(entry).get("readiness_path")
    if profile_path:
        return str(profile_path)
    probe = readiness_probe or _resolve_readiness_probe_mode(
        model_name,
        entry,
        serving_backend=serving_backend,
    )
    if probe == "http" and _is_bentoml_vision_service(
        model_name,
        entry,
        serving_backend=serving_backend,
        service_kind=service_kind,
    ):
        return "/readyz"
    return None


def _enrich_endpoint_response(ep_dict: dict[str, Any]) -> dict[str, Any]:
    """Add served_model_name to endpoint dict if not already present."""
    if not ep_dict.get("served_model_name"):
        served = _resolve_served_model_name(ep_dict["model"])
        if served:
            ep_dict["served_model_name"] = served
    return ep_dict


# Mapping of capability tag -> (model_entry path, expected truthy value)
# ``None`` for the second slot means "boolean: any truthy value is OK".
_CAPABILITY_TAG_MAP: dict[str, Any] = {
    "vision": ("capabilities.supports_vision", None),
    "tool_calling": ("capabilities.supports_tool_calling", None),
    "structured_output": ("capabilities.supports_structured_output", None),
    "embedding": ("model_type", "embedding"),
    # ``gpu`` is treated as always-satisfied: every served model in this
    # platform runs on GPU. Kept here so workflows can declare it without
    # being filtered out.
    "gpu": (None, None),
}


def _entry_field(entry: dict[str, Any], dotted: str) -> Any:
    cur: Any = entry
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _capability_satisfied(entry: dict[str, Any] | None, tag: str) -> bool:
    """Return True if the model entry advertises the given capability tag."""
    if entry is None:
        return False
    if tag not in _CAPABILITY_TAG_MAP:
        # Unknown tag: never matches (caller gets an empty list).
        return False
    field_path, expected = _CAPABILITY_TAG_MAP[tag]
    if field_path is None:
        # Always-satisfied tag (e.g. "gpu").
        return True
    actual = _entry_field(entry, field_path)
    if expected is None:
        return bool(actual)
    return actual == expected


def _endpoint_passes_capability_filter(
    ep_dict: dict[str, Any],
    capabilities: list[str] | None,
    model_type: str | None,
) -> bool:
    """Return True if the endpoint's served model satisfies the filters."""
    if not capabilities and not model_type:
        return True
    entry = _resolve_registered_model_entry(ep_dict.get("model", ""))
    if model_type:
        actual_mt = (entry or {}).get("model_type")
        if actual_mt != model_type:
            return False
    if capabilities:
        for tag in capabilities:
            if not _capability_satisfied(entry, tag):
                return False
    return True


def _parse_capabilities_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return tags or None


def _is_all_models_query(model: str | None) -> bool:
    """Return True when the caller is requesting all visible models."""
    if model is None:
        return True
    return model.strip() in {"", "*"}


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/active",
    response_model=EndpointsListResponse,
    summary="Query active endpoints",
    description="""
    Query active (ready) endpoints for a project.

    This is the primary discovery endpoint for Gateway to find where to route requests.
    Returns only endpoints with status=ready and sufficient TTL remaining.

    **Usage by Gateway:**
    1. Extract project_id from incoming request JWT
    2. Call GET /umrs/endpoints/active?model=rivault-gpt-oss-static
    3. Use returned litellm_url to proxy the request
    """,
)
async def get_active_endpoints(
    model: str | None = Query(
        None, description="Filter by model name (catalog or served name)"
    ),
    capabilities: str | None = Query(
        None,
        description=(
            "Comma-separated capability tags the served model must advertise. "
            "Known tags: vision, tool_calling, structured_output, embedding, gpu."
        ),
    ),
    model_type: str | None = Query(
        None,
        description="Filter by ModelType: 'llm', 'vision', or 'embedding'.",
    ),
    project_id: str = Depends(get_project_id),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointsListResponse:
    """Get active (ready) endpoints for the project."""

    normalized_model = model.strip() if model else None
    capability_tags = _parse_capabilities_csv(capabilities)
    normalized_model_type = model_type.strip() if model_type else None

    # If specific model requested, try to get that endpoint
    if not _is_all_models_query(normalized_model):
        endpoint = registry.get(project_id, normalized_model, prefer_warm=True)
        if not endpoint:
            # Try matching by served_model_name in metadata
            all_eps = registry.list_all(
                project_id=project_id,
                status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
            )
            for ep in all_eps:
                served = ep.metadata.get("served_model_name")
                if not served:
                    served = _resolve_served_model_name(ep.model)
                if served == normalized_model:
                    endpoint = ep
                    break
        if endpoint:
            live = await _filter_live_ready_endpoints(registry, [endpoint])
            if not live:
                return EndpointsListResponse(endpoints=[], total=0)
            endpoint = live[0]
            ep_dict = _enrich_endpoint_response(endpoint.to_dict())
            if not _endpoint_passes_capability_filter(
                ep_dict, capability_tags, normalized_model_type
            ):
                return EndpointsListResponse(endpoints=[], total=0)
            return EndpointsListResponse(
                endpoints=[EndpointResponse(**ep_dict)],
                total=1,
            )
        return EndpointsListResponse(endpoints=[], total=0)

    # List all ready endpoints for the project
    endpoints = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )
    endpoints = await _filter_live_ready_endpoints(registry, endpoints)

    enriched = [_enrich_endpoint_response(e.to_dict()) for e in endpoints]
    filtered = [
        ep
        for ep in enriched
        if _endpoint_passes_capability_filter(
            ep, capability_tags, normalized_model_type
        )
    ]

    return EndpointsListResponse(
        endpoints=[EndpointResponse(**e) for e in filtered],
        total=len(filtered),
    )


@router.get(
    "",
    response_model=EndpointsListResponse,
    summary="List all endpoints",
    description="List all endpoints (for monitoring/admin). Optionally filter by project.",
)
async def list_endpoints(
    project_id: str | None = Depends(get_project_id_optional),
    status_filter: str | None = Query(None, description="Comma-separated statuses"),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointsListResponse:
    """List all endpoints, optionally filtered.

    F-012: READY/DEGRADED endpoints are verified against the backend and
    ghost entries are pruned, same as the ``/active`` route.  This ensures
    ``r4 endpoint list --all`` never shows stale READY rows.
    """

    statuses = None
    if status_filter:
        try:
            statuses = [EndpointStatus(s.strip()) for s in status_filter.split(",")]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter: {e}",
            ) from e

    endpoints = registry.list_all(project_id=project_id, status_filter=statuses)

    # F-012: Prune ghost READY/DEGRADED endpoints at list time so the
    # user-facing ``r4 endpoint list`` output is always honest.
    ready_eps = [
        e
        for e in endpoints
        if e.status in (EndpointStatus.READY, EndpointStatus.DEGRADED)
    ]
    if ready_eps:
        live_ready = await _filter_live_ready_endpoints(registry, ready_eps)
        live_job_ids = {e.job_id for e in live_ready}
        endpoints = [
            e
            for e in endpoints
            if e.status not in (EndpointStatus.READY, EndpointStatus.DEGRADED)
            or e.job_id in live_job_ids
        ]

    return EndpointsListResponse(
        endpoints=[EndpointResponse(**e.to_dict()) for e in endpoints],
        total=len(endpoints),
    )


@router.post(
    "",
    response_model=EndpointResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an endpoint",
    description="Register a new endpoint (internal use by UMRS orchestrator).",
)
async def register_endpoint(
    request: EndpointRegisterRequest,
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Register a new endpoint."""

    endpoint = registry.register(
        project_id=request.project_id,
        model=request.model,
        job_id=request.job_id,
        host=request.host,
        port=request.port,
        backend=request.backend,
        ttl_sec=request.ttl_sec,
        metadata=request.metadata,
    )

    return EndpointResponse(**endpoint.to_dict())


@router.put(
    "/{job_id}/status",
    response_model=EndpointResponse,
    summary="Update endpoint status",
)
async def update_endpoint_status(
    job_id: str,
    request: EndpointStatusUpdate,
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Update endpoint status (internal use)."""

    try:
        new_status = EndpointStatus(request.status)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {request.status}",
        ) from None

    endpoint = registry.update_status(job_id, new_status, request.error_message)
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Endpoint not found: {job_id}",
        )

    return EndpointResponse(**endpoint.to_dict())


@router.put(
    "/{job_id}/used",
    response_model=EndpointResponse,
    summary="Mark endpoint as used",
    description="Update last_used_at timestamp (for activity tracking).",
)
async def mark_endpoint_used(
    job_id: str,
    project_id: str | None = Depends(get_project_id_optional),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Mark endpoint as used (update last_used_at)."""

    existing = registry.get_by_job_id(job_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Endpoint not found: {job_id}",
        )
    _forbid_if_project_mismatch(existing, project_id)

    endpoint = registry.mark_used(job_id)
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Endpoint not found: {job_id}",
        )

    return EndpointResponse(**endpoint.to_dict())


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an endpoint",
)
async def remove_endpoint(
    job_id: str,
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> None:
    """Remove an endpoint from the registry."""

    endpoint = registry.remove(job_id)
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Endpoint not found: {job_id}",
        )


@router.get(
    "/stats",
    response_model=RegistryStatsResponse,
    summary="Get registry statistics",
)
async def get_registry_stats(
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> RegistryStatsResponse:
    """Get registry statistics."""
    return RegistryStatsResponse(**registry.stats())


# =============================================================================
# Health check endpoints (U-204)
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Registry health check (U-204)",
    description="""
    Comprehensive health check for the endpoint registry.

    Returns:
    - Overall status (healthy/degraded/unhealthy)
    - Counts by status
    - Individual endpoint health details

    Use this to monitor the health of all active endpoints.
    """,
)
async def health_check(
    include_details: bool = Query(False, description="Include per-endpoint details"),
    registry: EndpointRegistry = Depends(get_registry),
) -> HealthResponse:
    """Comprehensive health check endpoint."""
    from datetime import datetime

    # Get all endpoints
    all_endpoints = registry.list_all()

    # Count by health status
    healthy = 0
    unhealthy = 0
    provisioning = 0

    endpoint_details = []

    for ep in all_endpoints:
        if ep.status == EndpointStatus.READY:
            healthy += 1
            is_healthy = True
        elif ep.status == EndpointStatus.PENDING:
            provisioning += 1
            is_healthy = True  # Still healthy, just starting
        elif ep.status in (EndpointStatus.FAILED, EndpointStatus.TERMINATED):
            unhealthy += 1
            is_healthy = False
        else:
            is_healthy = ep.is_usable

        if include_details:
            uptime = None
            if ep.ready_at:
                uptime = (datetime.utcnow() - ep.ready_at).total_seconds()

            endpoint_details.append(
                EndpointHealthInfo(
                    job_id=ep.job_id,
                    project_id=ep.project_id,
                    model=ep.model,
                    status=ep.status.value,
                    host=ep.host,
                    port=ep.port,
                    healthy=is_healthy,
                    last_health_check=ep.last_used_at.isoformat()
                    if ep.last_used_at
                    else None,
                    uptime_seconds=uptime,
                    error=ep.error_message,
                )
            )

    # Determine overall status
    total = len(all_endpoints)
    if total == 0:
        overall_status = "healthy"
        message = "No endpoints registered"
    elif unhealthy == 0:
        overall_status = "healthy"
        message = None
    elif unhealthy < total:
        overall_status = "degraded"
        message = f"{unhealthy} endpoint(s) unhealthy"
    else:
        overall_status = "unhealthy"
        message = "All endpoints unhealthy"

    return HealthResponse(
        status=overall_status,
        registry_ok=True,
        total_endpoints=total,
        healthy_endpoints=healthy,
        unhealthy_endpoints=unhealthy,
        provisioning_endpoints=provisioning,
        endpoints=endpoint_details if include_details else [],
        message=message,
    )


@router.get(
    "/{job_id}",
    response_model=EndpointResponse,
    summary="Get endpoint by job_id",
)
async def get_endpoint(
    job_id: str,
    project_id: str | None = Depends(get_project_id_optional),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Get endpoint details by job_id."""

    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Endpoint not found: {job_id}",
        )
    _forbid_if_project_mismatch(endpoint, project_id)

    endpoint = await _refresh_endpoint_readiness(registry, endpoint)

    return EndpointResponse(**endpoint.to_dict())


# =============================================================================
# Provisioning endpoints (U-202, U-203, U-205)
# =============================================================================


@router.post(
    "/provision",
    response_model=ProvisionResponse,
    responses={
        200: {"description": "Endpoint ready", "model": ProvisionResponse},
        202: {"description": "Provisioning in progress", "model": ProvisionResponse},
        503: {"description": "Service unavailable"},
    },
    summary="Provision an endpoint (U-202)",
    description="""
    Request provisioning of a new inference endpoint.

    Response codes (U-205):
    - 200 OK: Endpoint is ready and usable
    - 202 Accepted: Provisioning in progress, check Retry-After header
    - 503 Service Unavailable: Backend unavailable, retry later

    If provisioning, includes `retry_after` hint for when to check again.
    The actual HTTP response includes a `Retry-After` header when status is "provisioning".
    """,
)
async def provision_endpoint(
    request: ProvisionRequest,
    project_id: str | None = Depends(get_project_id_optional),
    is_admin: bool = Depends(_caller_is_admin),
    requesting_user: str | None = Depends(_get_requesting_user),
    registry: EndpointRegistry = Depends(get_registry),
):
    """Provision a new endpoint or return existing one."""
    from fastapi.responses import JSONResponse

    from ..orchestrator.service import (
        ProvisionRequest as ServiceProvisionRequest,
    )
    from ..orchestrator.service import (
        get_orchestration_service,
    )

    effective_project_id = _resolve_provision_project_id(request, project_id)
    _enforce_provision_authorization(request, is_admin=is_admin)

    # Governance: quota check before provisioning
    from .governance import check_quota

    check_quota(effective_project_id, "endpoints")

    # Validate that the requesting user is registered and (optionally) has
    # a Slurm account.  When ``require_user_slurm_account`` is False (the
    # default) users without a per-user Slurm account can still submit jobs
    # — the backend falls back to the site's master Slurm account.
    if requesting_user:
        try:
            from ..admin.user_store import UserStore, UserStoreError

            _user_store = UserStore()
            _user_record = _user_store.resolve_user(requesting_user)
            if _user_record.disabled:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User '{_user_record.username}' is disabled.",
                )
            _site_name: str | None = None
            _require_acct: bool = False
            try:
                from ..orchestrator.site_config import load_site_config as _load_sc

                _sc = _load_sc()
                _site_name = _sc.name
                _require_acct = getattr(
                    _sc.scheduler, "require_user_slurm_account", False
                )
            except FileNotFoundError:
                pass
            if _require_acct:
                _acct = _user_record.resolve_slurm_account(_site_name)
                if not _acct:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=(
                            f"User '{_user_record.username}' has no Slurm account configured. "
                            "Ask an admin to run: "
                            f"umrs-server admin user set-slurm-account {_user_record.username} <account>"
                        ),
                    )
        except HTTPException:
            raise
        except UserStoreError as _use:
            # User record not found in the store — reject rather than proceed.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User '{requesting_user}' is not registered. "
                "Ask an admin to run: "
                f"umrs-server admin user add {requesting_user}",
            ) from _use
        except Exception:
            # User store infrastructure failure (I/O error, corrupt file, etc.).
            # Log and proceed to avoid blocking provisioning when the store is
            # temporarily unreachable (e.g. NFS hiccup in development).
            logger.warning(
                "User store unavailable while verifying user %s; proceeding without validation",
                requesting_user,
                exc_info=True,
            )

    # Check for existing warm endpoint first
    if request.prefer_warm:
        existing = registry.get(
            effective_project_id,
            request.model,
            prefer_warm=True,
            min_ttl_sec=60,
            variant=request.variant,
        )
        if existing and existing.is_usable:
            # Verify the backend job is actually still alive before
            # handing the cached endpoint back to the caller. The registry
            # caches the last-known status; if the underlying Slurm/K8s
            # job has terminated since (walltime, manual cancel, crash),
            # `is_usable` would still return True and we would return a
            # ghost endpoint that no longer responds. Probe the backend
            # and demote stale entries before reusing.
            service = get_orchestration_service()
            if await _is_backend_job_alive(service, existing.job_id):
                return ProvisionResponse(
                    status="ready",
                    job_id=existing.job_id,
                    endpoint=EndpointResponse(**existing.to_dict()),
                )
            logger.warning(
                "Stale warm endpoint detected for %s/%s (job_id=%s, variant=%s); "
                "marking failed and provisioning fresh",
                effective_project_id,
                request.model,
                existing.job_id,
                request.variant,
            )
            registry.mark_failed(
                existing.job_id,
                "stale registry entry: backend job no longer alive",
            )

    # Check if there's already a provisioning endpoint for this
    # project/model/variant combination. Multiple endpoints for the
    # same model are allowed when they have different variants, so
    # we only block on exact (model, variant) match.
    provisioning = registry.list_all(
        project_id=effective_project_id,
        status_filter=[EndpointStatus.PENDING, EndpointStatus.PROVISIONING],
    )
    for ep in provisioning:
        if ep.model == request.model and ep.variant == request.variant:
            # Already provisioning this exact variant - return 202 with Retry-After
            response = ProvisionResponse(
                status="provisioning",
                job_id=ep.job_id,
                retry_after=30,  # Check again in 30 seconds
            )
            return JSONResponse(
                content=response.model_dump(),
                status_code=status.HTTP_202_ACCEPTED,
                headers={"Retry-After": "30"},
            )

    # Trigger new provisioning
    service = get_orchestration_service()
    model_entry = _resolve_registered_model_entry(request.model)
    resolved_model_path = _resolve_requested_model_path(
        request.model,
        request.model_path,
        model_entry=model_entry,
    )
    if not resolved_model_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unable to resolve model path for '{request.model}'. "
                "Register the model in the catalog or provide model_path explicitly."
            ),
        )

    env_vars: dict[str, str] = {}
    model_type = _resolve_model_type(request.model, model_entry)
    serving_backend = _resolve_serving_backend(
        request.model,
        model_entry=model_entry,
        explicit=request.serving_backend,
    )
    service_kind = _resolve_service_kind(
        request.model,
        model_entry=model_entry,
        serving_backend=serving_backend,
        explicit=request.service_kind,
    )
    launch_profiles = _resolve_launch_profiles(
        request.model,
        model_entry=model_entry,
        serving_backend=serving_backend,
        explicit=request.launch_profiles,
        co_locate_vllm=request.co_locate_vllm,
    )
    readiness_probe = _resolve_readiness_probe_mode(
        request.model,
        model_entry=model_entry,
        serving_backend=serving_backend,
        explicit=request.readiness_probe,
    )
    readiness_path = _resolve_readiness_path(
        request.model,
        model_entry=model_entry,
        serving_backend=serving_backend,
        service_kind=service_kind,
        readiness_probe=readiness_probe,
        explicit=request.readiness_path,
    )
    serving_profile = _serving_profile(model_entry)
    container_image = request.container_image or serving_profile.get("container_image")
    service_port = request.service_port or serving_profile.get("service_port")
    invoke_base_path = request.invoke_base_path or serving_profile.get(
        "invoke_base_path"
    )
    vision_adapter = request.vision_adapter or serving_profile.get("vision_adapter")
    vision_task = request.vision_task or serving_profile.get("vision_task")
    output_format = request.output_format or serving_profile.get("output_format")

    # Wire ServingProfile vLLM parameters as env-var defaults.
    # API request fields take precedence; ServingProfile fills in catalog values.
    profile_tp = serving_profile.get("tensor_parallel_size")
    if profile_tp and int(profile_tp) > 1:
        env_vars.setdefault("VLLM_TP", str(profile_tp))
    profile_gpu_util = serving_profile.get("gpu_memory_utilization")
    if profile_gpu_util:
        env_vars.setdefault("VLLM_GPU_MEM_UTIL", str(profile_gpu_util))
    profile_dtype = serving_profile.get("dtype")
    if profile_dtype and profile_dtype != "auto":
        env_vars.setdefault("VLLM_DTYPE", str(profile_dtype))
    profile_quantization = serving_profile.get("quantization")
    if profile_quantization:
        env_vars.setdefault("VLLM_QUANTIZATION", str(profile_quantization))
    profile_max_model_len = serving_profile.get("max_model_len")
    if profile_max_model_len:
        env_vars.setdefault("VLLM_MAX_MODEL_LEN", str(profile_max_model_len))

    # Apply recommended_gpu_count from ServingProfile when the API request
    # uses the default value (1).  This lets the model catalog specify GPU
    # requirements without every caller needing to set gpu_count explicitly.
    profile_gpu_count = serving_profile.get("recommended_gpu_count")
    effective_gpu_count = request.gpu_count
    if request.gpu_count == 1 and profile_gpu_count and int(profile_gpu_count) > 1:
        effective_gpu_count = int(profile_gpu_count)
        logger.info(
            "Using recommended_gpu_count=%d from model catalog for '%s'",
            effective_gpu_count,
            request.model,
        )

    # Apply recommended max_model_len from ServingProfile when the API request
    # uses the default value (4096).
    effective_max_model_len = request.max_model_len
    if request.max_model_len == 4096 and profile_max_model_len:
        effective_max_model_len = int(profile_max_model_len)

    # Pipeline parallelism (multi-node): forward from ServingProfile → VLLM_PP env.
    profile_pp = serving_profile.get("pipeline_parallel_size")
    if profile_pp and int(profile_pp) > 1:
        env_vars.setdefault("VLLM_PP", str(profile_pp))

    # Multi-node parallelism strategy (tp_pp default | tep). API request field
    # takes precedence; ServingProfile supplies a catalog default. Forwarded to
    # serve.slurm via VLLM_PARALLEL_STRATEGY (opt-in; tp_pp is unchanged behaviour).
    effective_parallel_strategy = request.parallel_strategy or serving_profile.get(
        "parallel_strategy"
    )
    if effective_parallel_strategy:
        env_vars.setdefault(
            "VLLM_PARALLEL_STRATEGY", str(effective_parallel_strategy).lower()
        )

    # Reasoning-parser control (auto | on | off). 'auto' (default) lets
    # serve.slurm derive the parser from the model name. 'off' disables it via
    # the 'none' sentinel (serve.slurm clears VLLM_REASONING_PARSER). 'on'
    # forces the model's native parser so a thinking model still splits its
    # <think> block even if shell-side auto-derivation is disabled site-wide.
    if request.reasoning == "off":
        env_vars["VLLM_REASONING_PARSER"] = "none"
    elif request.reasoning == "on":
        _forced_parser = _derive_reasoning_parser(request.model)
        if _forced_parser:
            env_vars["VLLM_REASONING_PARSER"] = _forced_parser

    # Apply recommended_node_count from ServingProfile when the API request
    # uses the default value (1).
    profile_node_count = serving_profile.get("recommended_node_count")
    effective_node_count = request.node_count
    if request.node_count == 1 and profile_node_count and int(profile_node_count) > 1:
        effective_node_count = int(profile_node_count)
        logger.info(
            "Using recommended_node_count=%d from model catalog for '%s'",
            effective_node_count,
            request.model,
        )
    if effective_node_count > 1:
        env_vars.setdefault("SLURM_NODES", str(effective_node_count))

    # Per-gateway pool-roster sizing defaults. When this deployment publishes a
    # pool roster (UMRS_PROVISION_ROSTER), use it to size single-endpoint
    # provisions for known large models. This is what routes models like
    # DeepSeek-V4-Pro to the 4n4gpu-p partition with the right node/GPU/context
    # and TTL even when provisioned individually (not via `r4 pool up`) — without
    # it the request falls back to 1 node / 1 GPU and Slurm holds the job
    # PD (MaxMemPerLimit) forever. Roster values fill ONLY fields the caller
    # left at their defaults, so explicit API/CLI flags always win.
    from ..orchestrator.provision_roster import resolve_model_provision_defaults

    effective_partition = request.partition
    effective_ttl_sec = request.ttl_sec
    roster_defaults = resolve_model_provision_defaults(
        request.model, variant=request.variant
    )
    if roster_defaults:
        if effective_partition is None and roster_defaults.get("partition"):
            effective_partition = str(roster_defaults["partition"])
        roster_nodes = roster_defaults.get("node_count")
        if effective_node_count == 1 and roster_nodes and int(roster_nodes) > 1:
            effective_node_count = int(roster_nodes)
            env_vars.setdefault("SLURM_NODES", str(effective_node_count))
        roster_gpus = roster_defaults.get("gpu_count")
        if effective_gpu_count == 1 and roster_gpus and int(roster_gpus) > 1:
            effective_gpu_count = int(roster_gpus)
        roster_mml = roster_defaults.get("max_model_len")
        if effective_max_model_len == 4096 and roster_mml:
            effective_max_model_len = int(roster_mml)
            env_vars.setdefault("VLLM_MAX_MODEL_LEN", str(effective_max_model_len))
        roster_strategy = roster_defaults.get("parallel_strategy")
        if not effective_parallel_strategy and roster_strategy:
            env_vars.setdefault("VLLM_PARALLEL_STRATEGY", str(roster_strategy).lower())
        roster_ttl = roster_defaults.get("ttl_sec")
        if request.ttl_sec == 3600 and roster_ttl is not None:
            effective_ttl_sec = int(roster_ttl)
        for _k, _v in (roster_defaults.get("env_vars") or {}).items():
            env_vars.setdefault(str(_k), str(_v))
        logger.info(
            "Applied pool-roster sizing for '%s': partition=%s nodes=%d gpus=%d "
            "max_model_len=%d ttl_sec=%s",
            request.model,
            effective_partition,
            effective_node_count,
            effective_gpu_count,
            effective_max_model_len,
            effective_ttl_sec,
        )

    request_metadata: dict[str, Any] = {
        "model_type": model_type,
        "serving_backend": serving_backend,
        "service_kind": service_kind,
        "readiness_probe": readiness_probe,
    }
    if launch_profiles:
        request_metadata["launch_profiles"] = launch_profiles
        env_vars["RIVAULT_PROFILES"] = " ".join(launch_profiles)
    if readiness_path:
        request_metadata["readiness_path"] = readiness_path
    if service_port:
        request_metadata["service_port"] = int(service_port)
    if invoke_base_path:
        request_metadata["invoke_base_path"] = str(invoke_base_path)
    if container_image:
        request_metadata["container_image"] = str(container_image)
    if vision_adapter:
        request_metadata["vision_adapter"] = str(vision_adapter)
    if vision_task:
        request_metadata["vision_task"] = str(vision_task)
    if output_format:
        request_metadata["output_format"] = str(output_format)

    if serving_backend == "bentoml" or service_kind == "bentoml_vision":
        env_vars.setdefault("VISION_REQUIRE_PRESIGNED_URLS", "true")
        env_vars.setdefault("TRANSFORMERS_OFFLINE", "1")
        env_vars.setdefault("HF_HUB_OFFLINE", "1")
        if container_image:
            env_vars.setdefault("BENTOML_VISION_IMAGE", str(container_image))

    request_metadata.setdefault("owner_type", request.tier)
    request_metadata.setdefault("tier", request_metadata["owner_type"])
    served_model_name = _resolve_served_model_name(
        request.model,
        model_entry=model_entry,
    )
    if served_model_name and served_model_name != request.model:
        request_metadata["served_model_name"] = served_model_name
        env_vars["VLLM_SERVED_MODEL_NAME"] = served_model_name

    # Multi-model fields (forwarded via metadata so orchestrator backends
    # can read them without a service-layer schema change yet). Phase 3
    # will surface these as first-class ProvisionRequest fields.
    request_metadata["serving_mode"] = request.serving_mode
    request_metadata["models"] = list(request.models or [request.model])
    if request.loras:
        request_metadata["loras"] = [
            {"name": lora.name, "source": lora.source} for lora in request.loras
        ]

    # Create provision request
    service_request = ServiceProvisionRequest(
        project_id=effective_project_id,
        model=request.model,
        model_path=resolved_model_path,
        ttl_sec=effective_ttl_sec,
        gpu_count=effective_gpu_count,
        node_count=effective_node_count,
        max_model_len=effective_max_model_len,
        tier=request.tier,
        mounts=request.mounts.model_dump(exclude_none=True) if request.mounts else {},
        egress_domains=list(request.egress_domains),
        prefer_warm=request.prefer_warm,
        wait_for_ready=request.wait,
        partition_override=effective_partition,
        nodelist_hint=request.nodelist,
        min_gpu_memory_gb=serving_profile.get("min_gpu_memory_gb"),
        env_vars=env_vars,
        metadata=request_metadata,
        requesting_user=requesting_user,
        variant=request.variant,
    )

    try:
        result = await service.provision(service_request)

        if result.success and result.endpoint and result.endpoint.is_usable:
            # Ready immediately (unlikely for real provisioning)
            from .audit_events import emit_audit

            emit_audit(
                "endpoint.created",
                user_id=requesting_user or "",
                project_id=effective_project_id,
                details={
                    "model": request.model,
                    "tier": request.tier,
                    "job_id": result.job_id,
                },
            )
            from .governance import record_resource_created

            record_resource_created(
                "endpoints",
                result.job_id,
                requesting_user or "",
                project_id=effective_project_id,
                tier=request.tier,
                metadata={"model": request.model},
            )
            return ProvisionResponse(
                status="ready",
                job_id=result.job_id,
                endpoint=EndpointResponse(**result.endpoint.to_dict())
                if result.endpoint
                else None,
            )
        elif result.success:
            # Provisioning started
            from .audit_events import emit_audit

            emit_audit(
                "endpoint.provisioning",
                user_id=requesting_user or "",
                project_id=effective_project_id,
                details={
                    "model": request.model,
                    "tier": request.tier,
                    "job_id": result.job_id,
                },
            )
            from .governance import record_resource_created

            record_resource_created(
                "endpoints",
                result.job_id,
                requesting_user or "",
                project_id=effective_project_id,
                tier=request.tier,
                metadata={"model": request.model},
            )
            response = ProvisionResponse(
                status="provisioning",
                job_id=result.job_id,
                retry_after=30,
            )
            return JSONResponse(
                content=response.model_dump(),
                status_code=status.HTTP_202_ACCEPTED,
                headers={"Retry-After": "30"},
            )
        else:
            from .audit_events import emit_audit

            emit_audit(
                "endpoint.provision_failed",
                user_id=requesting_user or "",
                project_id=effective_project_id,
                outcome="error",
                details={"model": request.model, "error": result.error},
            )
            return ProvisionResponse(
                status="failed",
                job_id=result.job_id,
                error=result.error,
            )

    except Exception as e:
        logger.exception(f"Provision failed: {e}")
        from .audit_events import emit_audit

        emit_audit(
            "endpoint.provision_failed",
            user_id=requesting_user or "",
            project_id=effective_project_id,
            outcome="error",
            details={"model": request.model, "error": str(e)},
        )
        return ProvisionResponse(
            status="failed",
            error=str(e),
        )


@router.delete(
    "/{job_id}/deprovision",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deprovision an endpoint (U-203)",
    description="Stop and clean up an endpoint.",
)
async def deprovision_endpoint(
    job_id: str,
    force: bool = Query(False, description="Force cleanup even if in use"),
    project_id: str | None = Depends(get_project_id_optional),
    registry: EndpointRegistry = Depends(get_registry),
) -> None:
    """Deprovision (stop and cleanup) an endpoint."""
    from ..orchestrator.service import get_orchestration_service

    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Endpoint not found: {job_id}",
        )
    _forbid_if_project_mismatch(endpoint, project_id)

    service = get_orchestration_service()

    success = await service.deprovision(job_id, force=force)
    if not success:
        # deprovision() returns False either because the backend cancel was
        # issued but the job is still settling (e.g. Slurm COMPLETING), or
        # because of an unexpected failure. A job that is still tearing down is
        # not a server error — report it as "accepted, stopping" (202) so the
        # client does not surface a spurious 500. The janitor finalizes the
        # registry removal once the backend job actually exits.
        from fastapi.responses import JSONResponse

        still = registry.get_by_job_id(job_id)
        cleanup_status = (still.metadata or {}).get("cleanup_status") if still else None
        if still is not None and cleanup_status:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "job_id": job_id,
                    "status": "stopping",
                    "detail": (
                        "Deprovision initiated; backend job still stopping "
                        f"({cleanup_status})."
                    ),
                },
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deprovision endpoint: {job_id}",
        )

    from .audit_events import emit_audit

    emit_audit(
        "endpoint.deprovisioned",
        project_id=endpoint.project_id if endpoint else "",
        details={"job_id": job_id, "force": force},
    )
    from .governance import record_resource_deleted

    record_resource_deleted(
        "endpoints",
        job_id,
        project_id=endpoint.project_id if endpoint else "",
    )
