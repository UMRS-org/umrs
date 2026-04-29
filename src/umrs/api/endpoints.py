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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator

from ..orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    get_registry,
)

logger = logging.getLogger(__name__)

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
            job_id, exc,
        )
        return False
    status_value = getattr(info, "status", None)
    return status_value in (
        JobStatus.PENDING,
        JobStatus.STARTING,
        JobStatus.READY,
        JobStatus.RUNNING,
    )


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
    ready_at: Optional[str] = None
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    backend: str
    error_message: Optional[str] = None
    litellm_url: str
    served_model_name: Optional[str] = None
    detail: Optional[str] = None
    job_status: Optional[str] = None

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
                "detail": "starting container / loading model",
                "job_status": "starting",
            }
        }


class EndpointsListResponse(BaseModel):
    """Response for endpoint listing."""

    endpoints: List[EndpointResponse]
    total: int


class EndpointRegisterRequest(BaseModel):
    """Request to register a new endpoint."""

    project_id: str = Field(..., description="Project/tenant identifier")
    model: str = Field(..., description="Model name (e.g., rivault-gpt-oss-static)")
    job_id: str = Field(..., description="Backend job identifier")
    host: str = Field(..., description="Hostname or IP address")
    port: int = Field(..., ge=1, le=65535, description="LiteLLM port")
    backend: str = Field("unknown", description="Backend type (slurm/kubernetes/compose)")
    ttl_sec: Optional[int] = Field(None, ge=60, description="TTL in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EndpointStatusUpdate(BaseModel):
    """Request to update endpoint status."""

    status: str = Field(..., description="New status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class LoraSpec(BaseModel):
    """LoRA adapter specification for multi-lora endpoints."""

    name: str = Field(..., min_length=1, description="Adapter name (served as `model` over OpenAI API)")
    source: str = Field(..., min_length=1, description="Filesystem path or registered LoRA model name")


class ProvisionRequest(BaseModel):
    """Request to provision a new endpoint (U-202)."""

    project_id: str = Field(..., description="Project/tenant identifier")
    model: str = Field(..., description="Model name (primary; equals models[0])")
    models: Optional[List[str]] = Field(
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
    loras: Optional[List[LoraSpec]] = Field(
        None,
        description="LoRA adapters (multi-lora mode only)",
    )
    model_path: Optional[str] = Field(
        None,
        description="Optional path to model files. If omitted, resolve from the registered model catalog.",
    )
    ttl_sec: int = Field(3600, ge=60, description="TTL in seconds")
    gpu_count: int = Field(1, ge=1, description="Number of GPUs")
    max_model_len: int = Field(4096, ge=512, description="Max context length")
    prefer_warm: bool = Field(True, description="Prefer reusing existing endpoint")
    wait: bool = Field(True, description="Wait for readiness before responding")
    partition: Optional[str] = Field(None, description="Override Slurm partition (must be in site candidates)")
    nodelist: Optional[str] = Field(None, description="Slurm --nodelist hint (e.g. 'ai-h200-brc-00')")

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
        if self.serving_mode == "single" and len(self.models) != 1:
            raise ValueError("serving_mode=single requires exactly one model")
        if self.serving_mode == "co-located" and len(self.models) < 2:
            raise ValueError("serving_mode=co-located requires at least 2 models")

        if self.loras:
            if self.serving_mode != "multi-lora":
                raise ValueError("loras requires serving_mode=multi-lora")
            if len(self.models) != 1:
                raise ValueError("multi-lora requires exactly one base model")
            names = [l.name for l in self.loras]
            if len(set(names)) != len(names):
                raise ValueError("LoRA names must be unique")
        elif self.serving_mode == "multi-lora":
            raise ValueError(
                "serving_mode=multi-lora requires a non-empty loras list"
            )
        return self


class ProvisionResponse(BaseModel):
    """Response for provision request."""

    status: str = Field(..., description="Status: ready, provisioning, failed")
    job_id: Optional[str] = Field(None, description="Job ID if provisioning started")
    endpoint: Optional[EndpointResponse] = Field(None, description="Endpoint if ready")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")
    error: Optional[str] = Field(None, description="Error message if failed")


class RegistryStatsResponse(BaseModel):
    """Registry statistics."""

    total: int
    by_status: Dict[str, int]
    by_backend: Dict[str, int]
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
    last_health_check: Optional[str] = None
    uptime_seconds: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Comprehensive health response (U-204)."""

    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    registry_ok: bool
    total_endpoints: int
    healthy_endpoints: int
    unhealthy_endpoints: int
    provisioning_endpoints: int
    endpoints: List[EndpointHealthInfo] = Field(default_factory=list)
    message: Optional[str] = None


# =============================================================================
# Dependencies
# =============================================================================


def get_project_id(
    authorization: Optional[str] = Header(None),
    x_project_id: Optional[str] = Header(None),
) -> str:
    """
    Extract project_id from request headers.

    Production: Extract from JWT claims in Authorization header.
    MVP/Testing: Accept X-Project-ID header.

    Raises:
        HTTPException: If project_id cannot be determined.
    """
    # MVP: Accept X-Project-ID header for testing
    if x_project_id:
        return x_project_id

    # TODO: Production - decode JWT and extract project_id claim
    if authorization and authorization.startswith("Bearer "):
        # For now, just return a default for testing
        # In production: decode JWT and extract project_id
        logger.warning("JWT project_id extraction not implemented, using default")
        return "default-project"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing project_id. Provide X-Project-ID header or valid JWT.",
    )


def get_project_id_optional(
    authorization: Optional[str] = Header(None),
    x_project_id: Optional[str] = Header(None),
) -> Optional[str]:
    """Get project_id if available, None otherwise (for admin endpoints)."""
    if x_project_id:
        return x_project_id
    if authorization and authorization.startswith("Bearer "):
        return "default-project"
    return None


def _resolve_served_model_name(
    model_name: str,
    model_entry: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
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


def _resolve_registered_model_entry(model_name: str) -> Optional[Dict[str, Any]]:
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
            matches.sort(key=lambda entry: scope_order.get(str(entry.get("scope", "system")), 99))
            return matches[0]
    except Exception as e:
        logger.debug(f"Failed to resolve model entry from model store for {model_name}: {e}")

    try:
        from umrs.server.model_registry_handler import _load_catalog

        matches = [
            entry for entry in _load_catalog().get("models", []) if entry.get("name") == model_name
        ]
        if not matches:
            return None

        scope_order = {"system": 0, "project": 1, "user": 2}
        matches.sort(key=lambda entry: scope_order.get(str(entry.get("scope", "system")), 99))
        return matches[0]
    except Exception as e:
        logger.debug(f"Failed to resolve model entry for {model_name}: {e}")
    return None


def _resolve_requested_model_path(
    model_name: str,
    model_path: Optional[str],
    model_entry: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve the actual model storage path for endpoint provisioning."""
    normalized_path = model_path.strip() if model_path else None
    if normalized_path and normalized_path != _DEFAULT_PROVISION_MODEL_PATH:
        return normalized_path

    entry = model_entry or _resolve_registered_model_entry(model_name)
    if entry and entry.get("storage_ref"):
        return entry["storage_ref"]

    if model_name.startswith("/"):
        return model_name

    return None


def _resolve_readiness_probe_mode(
    model_name: str,
    model_entry: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve the readiness probe mode for a provisioned endpoint."""
    entry = model_entry or _resolve_registered_model_entry(model_name)
    model_type = str(entry.get("model_type", "")).lower() if entry else ""
    if model_type == "embedding":
        return "models"
    return "chat"


def _enrich_endpoint_response(ep_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add served_model_name to endpoint dict if not already present."""
    if not ep_dict.get("served_model_name"):
        served = _resolve_served_model_name(ep_dict["model"])
        if served:
            ep_dict["served_model_name"] = served
    return ep_dict


# Mapping of capability tag -> (model_entry path, expected truthy value)
# ``None`` for the second slot means "boolean: any truthy value is OK".
_CAPABILITY_TAG_MAP: Dict[str, Any] = {
    "vision": ("capabilities.supports_vision", None),
    "tool_calling": ("capabilities.supports_tool_calling", None),
    "structured_output": ("capabilities.supports_structured_output", None),
    "embedding": ("model_type", "embedding"),
    # ``gpu`` is treated as always-satisfied: every served model in this
    # platform runs on GPU. Kept here so workflows can declare it without
    # being filtered out.
    "gpu": (None, None),
}


def _entry_field(entry: Dict[str, Any], dotted: str) -> Any:
    cur: Any = entry
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _capability_satisfied(entry: Optional[Dict[str, Any]], tag: str) -> bool:
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
    ep_dict: Dict[str, Any],
    capabilities: Optional[List[str]],
    model_type: Optional[str],
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


def _parse_capabilities_csv(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return tags or None


def _is_all_models_query(model: Optional[str]) -> bool:
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
    model: Optional[str] = Query(None, description="Filter by model name (catalog or served name)"),
    capabilities: Optional[str] = Query(
        None,
        description=(
            "Comma-separated capability tags the served model must advertise. "
            "Known tags: vision, tool_calling, structured_output, embedding, gpu."
        ),
    ),
    model_type: Optional[str] = Query(
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
    project_id: Optional[str] = Depends(get_project_id_optional),
    status_filter: Optional[str] = Query(None, description="Comma-separated statuses"),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointsListResponse:
    """List all endpoints, optionally filtered."""

    statuses = None
    if status_filter:
        try:
            statuses = [EndpointStatus(s.strip()) for s in status_filter.split(",")]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter: {e}",
            )

    endpoints = registry.list_all(project_id=project_id, status_filter=statuses)

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
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Update endpoint status (internal use)."""

    try:
        new_status = EndpointStatus(request.status)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {request.status}",
        )

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
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Mark endpoint as used (update last_used_at)."""

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
                    last_health_check=ep.last_used_at.isoformat() if ep.last_used_at else None,
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
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Get endpoint details by job_id."""

    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Endpoint not found: {job_id}",
        )

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
    registry: EndpointRegistry = Depends(get_registry),
):
    """Provision a new endpoint or return existing one."""
    from fastapi.responses import JSONResponse

    from ..orchestrator.service import (
        ProvisionRequest as ServiceProvisionRequest,
        get_orchestration_service,
    )

    # Check for existing warm endpoint first
    if request.prefer_warm:
        existing = registry.get(
            request.project_id,
            request.model,
            prefer_warm=True,
            min_ttl_sec=60,
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
                "Stale warm endpoint detected for %s/%s (job_id=%s); "
                "marking failed and provisioning fresh",
                request.project_id, request.model, existing.job_id,
            )
            registry.mark_failed(
                existing.job_id,
                "stale registry entry: backend job no longer alive",
            )

    # Check if there's already a provisioning endpoint for this project/model
    provisioning = registry.list_all(
        project_id=request.project_id,
        status_filter=[EndpointStatus.PENDING],
    )
    for ep in provisioning:
        if ep.model == request.model:
            # Already provisioning - return 202 with Retry-After
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

    request_metadata: Dict[str, Any] = {
        # This route provisions interactive inference endpoints. Require a
        # successful chat path before exposing the endpoint as ready.
        "readiness_probe": "chat"
    }
    served_model_name = _resolve_served_model_name(
        request.model,
        model_entry=model_entry,
    )
    if served_model_name and served_model_name != request.model:
        request_metadata["served_model_name"] = served_model_name

    # Multi-model fields (forwarded via metadata so orchestrator backends
    # can read them without a service-layer schema change yet). Phase 3
    # will surface these as first-class ProvisionRequest fields.
    request_metadata["serving_mode"] = request.serving_mode
    request_metadata["models"] = list(request.models or [request.model])
    if request.loras:
        request_metadata["loras"] = [
            {"name": l.name, "source": l.source} for l in request.loras
        ]

    # Create provision request
    service_request = ServiceProvisionRequest(
        project_id=request.project_id,
        model=request.model,
        model_path=resolved_model_path,
        ttl_sec=request.ttl_sec,
        gpu_count=request.gpu_count,
        max_model_len=request.max_model_len,
        prefer_warm=request.prefer_warm,
        wait_for_ready=request.wait,
        partition_override=request.partition,
        nodelist_hint=request.nodelist,
        metadata=request_metadata,
    )

    try:
        result = await service.provision(service_request)

        if result.success and result.endpoint and result.endpoint.is_usable:
            # Ready immediately (unlikely for real provisioning)
            return ProvisionResponse(
                status="ready",
                job_id=result.job_id,
                endpoint=EndpointResponse(**result.endpoint.to_dict()) if result.endpoint else None,
            )
        elif result.success:
            # Provisioning started
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
            return ProvisionResponse(
                status="failed",
                job_id=result.job_id,
                error=result.error,
            )

    except Exception as e:
        logger.exception(f"Provision failed: {e}")
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

    service = get_orchestration_service()

    success = await service.deprovision(job_id, force=force)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deprovision endpoint: {job_id}",
        )
