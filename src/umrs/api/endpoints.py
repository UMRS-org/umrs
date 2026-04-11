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
from pydantic import BaseModel, Field

from ..orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    get_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/endpoints", tags=["endpoints"])


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


class ProvisionRequest(BaseModel):
    """Request to provision a new endpoint (U-202)."""

    project_id: str = Field(..., description="Project/tenant identifier")
    model: str = Field(..., description="Model name")
    model_path: str = Field("/models/rivault-gpt-oss", description="Path to model files")
    ttl_sec: int = Field(3600, ge=60, description="TTL in seconds")
    gpu_count: int = Field(1, ge=1, description="Number of GPUs")
    max_model_len: int = Field(4096, ge=512, description="Max context length")
    prefer_warm: bool = Field(True, description="Prefer reusing existing endpoint")


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


def _resolve_served_model_name(model_name: str) -> Optional[str]:
    """Resolve served_model_name from model catalog (huggingface_id)."""
    try:
        from umrs.server.model_registry_handler import _load_catalog

        catalog = _load_catalog()
        for entry in catalog.get("models", []):
            if entry.get("name") == model_name:
                hf_id = entry.get("huggingface_id")
                if hf_id:
                    return hf_id
    except Exception as e:
        logger.debug(f"Failed to resolve served_model_name for {model_name}: {e}")
    return None


def _enrich_endpoint_response(ep_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add served_model_name to endpoint dict if not already present."""
    if not ep_dict.get("served_model_name"):
        served = _resolve_served_model_name(ep_dict["model"])
        if served:
            ep_dict["served_model_name"] = served
    return ep_dict


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
    project_id: str = Depends(get_project_id),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointsListResponse:
    """Get active (ready) endpoints for the project."""

    # If specific model requested, try to get that endpoint
    if model:
        endpoint = registry.get(project_id, model, prefer_warm=True)
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
                if served == model:
                    endpoint = ep
                    break
        if endpoint:
            ep_dict = _enrich_endpoint_response(endpoint.to_dict())
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

    return EndpointsListResponse(
        endpoints=[EndpointResponse(**_enrich_endpoint_response(e.to_dict())) for e in endpoints],
        total=len(endpoints),
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
            return ProvisionResponse(
                status="ready",
                job_id=existing.job_id,
                endpoint=EndpointResponse(**existing.to_dict()),
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

    # Create provision request
    service_request = ServiceProvisionRequest(
        project_id=request.project_id,
        model=request.model,
        model_path=request.model_path,
        ttl_sec=request.ttl_sec,
        gpu_count=request.gpu_count,
        max_model_len=request.max_model_len,
        prefer_warm=request.prefer_warm,
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
