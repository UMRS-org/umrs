"""
Model Registry REST API (U-811 ~ U-820).

Provides:
- GET /umrs/models - List models visible to user
- GET /umrs/models/system - List system models (U-817)
- GET /umrs/models/{model_id} - Get model details
- POST /umrs/models - Register a new model
- POST /umrs/models/stage - Request model download (creates job)
- POST /umrs/models/{name}/download - Download to user space (U-818)
- DELETE /umrs/models/{model_id} - Delete model entry

Policy endpoints:
- GET /umrs/models/policies - List policies (U-819)
- POST /umrs/models/policies - Create policy (U-820)
- GET /umrs/models/policies/{policy_id} - Get policy
- PATCH /umrs/models/policies/{policy_id} - Update policy
- DELETE /umrs/models/policies/{policy_id} - Delete policy

See DEVELOPMENT_OBJECTIVES.md Section 1.9.2 for design details.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..jobs.models import JobType, RestartPolicy
from ..jobs.store import DuplicateJobError, JobStore
from ..model_registry.models import (
    ModelCapabilities,
    ModelEntry,
    ModelPolicy,
    ModelScope,
    ModelStatus,
    ModelType,
    ServingProfile,
)
from ..model_registry.store import (
    AccessContext,
    DuplicateModelError,
    ModelNotFoundError,
    ModelStore,
    PathNotAllowedError,
    PermissionDeniedError,
)
from .jobs import get_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/models", tags=["models"])


# =============================================================================
# Dependency Injection
# =============================================================================

# Global model store instance (set during app startup)
_model_store: ModelStore | None = None


def set_model_store(store: ModelStore) -> None:
    """Set the global model store instance."""
    global _model_store
    _model_store = store


def get_model_store() -> ModelStore:
    """Get the global model store instance."""
    if _model_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model store not initialized",
        )
    return _model_store


def get_optional_model_store() -> ModelStore | None:
    """Return the configured model store, if available."""
    return _model_store


# =============================================================================
# Request/Response Models
# =============================================================================


class ServingProfileRequest(BaseModel):
    """Serving configuration for vLLM/LiteLLM or BentoML vision."""

    serving_backend: str | None = Field(
        None, description="Serving backend: vllm or bentoml"
    )
    service_kind: str | None = Field(
        None, description="Service API kind: openai or bentoml_vision"
    )
    launch_profiles: list[str] = Field(
        default_factory=list, description="Mini-RiVault profiles to activate"
    )
    container_image: str | None = Field(
        None, description="Container image override for this service"
    )
    service_port: int | None = Field(
        None, ge=1, le=65535, description="External service port"
    )
    readiness_probe: str | None = Field(
        None, description="Readiness probe kind: models, chat, or http"
    )
    readiness_path: str | None = Field(
        None, description="HTTP readiness path for non-OpenAI services"
    )
    invoke_base_path: str | None = Field(
        None, description="Base path for invoking the service"
    )

    tensor_parallel_size: int = Field(
        1, ge=1, le=8, description="Number of GPUs for tensor parallelism"
    )
    max_model_len: int | None = Field(None, description="Maximum context length")
    gpu_memory_utilization: float = Field(
        0.9, ge=0.1, le=1.0, description="GPU memory fraction"
    )
    dtype: str = Field("auto", description="Data type: auto, float16, bfloat16")
    quantization: str | None = Field(None, description="Quantization: awq, gptq, None")
    served_model_name: str | None = Field(
        None, description="Override model name in API"
    )
    min_gpu_memory_gb: float | None = Field(
        None, description="Minimum GPU memory required"
    )
    recommended_gpu_count: int = Field(
        1, ge=1, description="Recommended number of GPUs"
    )
    vision_adapter: str | None = Field(
        None, description="Vision adapter name, e.g. sam3 or hmae"
    )
    vision_task: str | None = Field(None, description="Default vision task")
    output_format: str | None = Field(
        None, description="Default output envelope/artifact format"
    )
    co_located: bool | None = Field(
        None, description="Whether this profile expects vLLM co-location"
    )


class CapabilitiesRequest(BaseModel):
    """Model capability metadata."""

    supports_tool_calling: bool = Field(
        False, description="Whether the model supports tool/function calling"
    )
    supports_vision: bool = Field(
        False, description="Whether the model supports vision/image input"
    )
    supports_structured_output: bool = Field(
        False, description="Whether the model supports structured (JSON) output"
    )
    supports_reasoning: bool = Field(
        False, description="Whether the model supports thinking/reasoning output"
    )
    max_output_tokens: int | None = Field(None, description="Maximum output tokens")
    max_position_embeddings: int | None = Field(
        None, description="Maximum context length (positions)"
    )
    model_family: str | None = Field(
        None, description="Model family: qwen, llama, mistral, etc."
    )
    reasoning_effort_levels: list[str] | None = Field(
        None, description="Allowed reasoning_effort values for this model"
    )
    reasoning_effort_default: str | None = Field(
        None, description="Default reasoning_effort for this model"
    )


class ModelCreateRequest(BaseModel):
    """Request to register a new model."""

    name: str = Field(
        ..., min_length=1, max_length=256, description="Model name (unique per scope)"
    )
    scope: str = Field(..., description="Scope: system, project, user")
    storage_ref: str = Field(..., description="Filesystem path to model weights")
    model_type: str = Field("llm", description="Type: llm, embedding, or vision")

    owner_id: str | None = Field(
        None, description="Owner user ID (required for user scope)"
    )
    project_id: str | None = Field(
        None, description="Project ID (required for project scope)"
    )

    huggingface_id: str | None = Field(None, description="Source HuggingFace model ID")
    description: str | None = Field(
        None, max_length=1000, description="Model description"
    )
    serving_profile: ServingProfileRequest | None = Field(
        None, description="Serving configuration"
    )
    capabilities: CapabilitiesRequest | None = Field(
        None, description="Model capabilities"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "qwen-7b-team",
                "scope": "project",
                "storage_ref": "/shared/projects/proj-123/models/qwen-7b-team",
                "model_type": "llm",
                "project_id": "proj-123",
                "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
                "description": "Team fine-tuned Qwen model for code review",
                "serving_profile": {
                    "tensor_parallel_size": 1,
                    "max_model_len": 4096,
                    "gpu_memory_utilization": 0.9,
                },
            }
        }


class ModelStageRequest(BaseModel):
    """Request to download/stage a model (creates a job)."""

    huggingface_id: str = Field(..., description="HuggingFace model ID to download")
    name: str = Field(
        ..., min_length=1, max_length=256, description="Name for the model"
    )
    scope: str = Field("user", description="Target scope: project or user")
    model_type: str = Field("llm", description="Type: llm, embedding, or vision")

    project_id: str | None = Field(None, description="Project ID (for project scope)")
    description: str | None = Field(None, description="Model description")
    serving_profile: ServingProfileRequest | None = Field(
        None, description="Serving configuration"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "qwen-7b-myproject",
                "scope": "project",
                "model_type": "llm",
                "project_id": "proj-123",
                "description": "Qwen 7B for our project",
            }
        }


class ModelUpdateRequest(BaseModel):
    """Request to update model metadata."""

    description: str | None = Field(None, description="New description")
    serving_profile: ServingProfileRequest | None = Field(
        None, description="New serving profile"
    )
    status: str | None = Field(None, description="New status: available, deprecated")
    capabilities: CapabilitiesRequest | None = Field(
        None, description="Model capabilities"
    )


class ModelResponse(BaseModel):
    """Model data for API responses."""

    model_id: str
    name: str
    scope: str
    owner_id: str | None
    project_id: str | None
    storage_ref: str
    model_ref: dict[str, Any] | None = None
    model_type: str
    huggingface_id: str | None
    description: str | None
    serving_profile: dict[str, Any]
    capabilities: dict[str, Any] = {}
    execution_authority_default: str = "compose"
    status: str
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_entry(cls, entry: ModelEntry) -> "ModelResponse":
        """Create response from ModelEntry."""
        return cls(
            model_id=entry.model_id,
            name=entry.name,
            scope=entry.scope.value,
            owner_id=entry.owner_id,
            project_id=entry.project_id,
            storage_ref=entry.storage_ref,
            model_ref=entry.model_ref.to_dict() if entry.model_ref else None,
            model_type=entry.model_type.value,
            huggingface_id=entry.huggingface_id,
            description=entry.description,
            serving_profile=entry.serving_profile.to_dict(),
            capabilities=entry.capabilities.to_dict(),
            execution_authority_default=entry.execution_authority_default.value,
            status=entry.status.value,
            created_at=entry.created_at.isoformat() if entry.created_at else None,
            updated_at=entry.updated_at.isoformat() if entry.updated_at else None,
        )


class ModelListResponse(BaseModel):
    """Response for model list endpoint."""

    models: list[ModelResponse]
    total: int
    limit: int
    offset: int


class ModelStageResponse(BaseModel):
    """Response for model staging request."""

    job_id: str
    message: str
    model_name: str
    target_scope: str


class ModelDownloadRequest(BaseModel):
    """Request to download a system model to user space (U-818)."""

    target_scope: str = Field("user", description="Target scope: project or user")
    project_id: str | None = Field(
        None, description="Target project (for project scope)"
    )
    custom_name: str | None = Field(None, description="Custom name for the copy")

    class Config:
        json_schema_extra = {
            "example": {
                "target_scope": "project",
                "project_id": "proj-123",
                "custom_name": "qwen-7b-team-copy",
            }
        }


class ModelDownloadResponse(BaseModel):
    """Response for model download request."""

    job_id: str
    message: str
    source_model: str
    target_name: str
    target_scope: str


# =============================================================================
# Policy Request/Response Models (U-819, U-820)
# =============================================================================


class PolicyCreateRequest(BaseModel):
    """Request to create a model policy."""

    model_name: str = Field(..., description="Model name this policy applies to")
    allowed_projects: list[str] = Field(
        default_factory=list, description="Allowed project IDs"
    )
    denied_projects: list[str] = Field(
        default_factory=list, description="Denied project IDs"
    )
    max_concurrent_endpoints: int | None = Field(
        None, description="Max concurrent endpoints"
    )
    max_tokens_per_day: int | None = Field(None, description="Daily token quota")
    priority: int = Field(0, description="Scheduling priority")
    warm_pool_min: int = Field(0, description="Minimum warm endpoints")
    warm_pool_max: int = Field(0, description="Maximum warm endpoints")
    warm_pool_ttl_seconds: int = Field(3600, description="Warm endpoint TTL")
    enable_warm_pool: bool = Field(False, description="Enable warm pooling")
    is_system_model: bool = Field(False, description="Mark as system model")
    default_serving_profile: ServingProfileRequest | None = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "qwen-7b-instruct",
                "allowed_projects": [],
                "is_system_model": True,
                "enable_warm_pool": True,
                "warm_pool_min": 1,
                "warm_pool_max": 3,
            }
        }


class PolicyUpdateRequest(BaseModel):
    """Request to update a model policy."""

    allowed_projects: list[str] | None = Field(None)
    denied_projects: list[str] | None = Field(None)
    max_concurrent_endpoints: int | None = Field(None)
    enable_warm_pool: bool | None = Field(None)
    warm_pool_min: int | None = Field(None)
    warm_pool_max: int | None = Field(None)


class PolicyResponse(BaseModel):
    """Policy data for API responses."""

    policy_id: str
    model_name: str
    allowed_projects: list[str]
    denied_projects: list[str]
    max_concurrent_endpoints: int | None
    max_tokens_per_day: int | None
    priority: int
    warm_pool_min: int
    warm_pool_max: int
    warm_pool_ttl_seconds: int
    enable_warm_pool: bool
    is_system_model: bool
    default_serving_profile: dict[str, Any]
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_policy(cls, policy: ModelPolicy) -> "PolicyResponse":
        """Create response from ModelPolicy."""
        return cls(
            policy_id=policy.policy_id,
            model_name=policy.model_name,
            allowed_projects=policy.allowed_projects,
            denied_projects=policy.denied_projects,
            max_concurrent_endpoints=policy.max_concurrent_endpoints,
            max_tokens_per_day=policy.max_tokens_per_day,
            priority=policy.priority,
            warm_pool_min=policy.warm_pool_min,
            warm_pool_max=policy.warm_pool_max,
            warm_pool_ttl_seconds=policy.warm_pool_ttl_seconds,
            enable_warm_pool=policy.enable_warm_pool,
            is_system_model=policy.is_system_model,
            default_serving_profile=policy.default_serving_profile.to_dict(),
            created_at=policy.created_at.isoformat() if policy.created_at else None,
            updated_at=policy.updated_at.isoformat() if policy.updated_at else None,
        )


class PolicyListResponse(BaseModel):
    """Response for policy list endpoint."""

    policies: list[PolicyResponse]
    total: int
    limit: int
    offset: int


class PolicyCheckResponse(BaseModel):
    """Response for policy access check."""

    allowed: bool
    reason: str | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _strict_auth_mode() -> bool:
    return os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"


def get_access_context(
    user_id: str = Query("", description="User ID (legacy, ignored in strict mode)"),
    is_admin: bool = Query(
        False, description="Admin flag (legacy, ignored in strict mode)"
    ),
    project_memberships: str | None = Query(
        None, description="Comma-separated project IDs (legacy)"
    ),
    x_user_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_project_memberships: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
) -> AccessContext:
    """
    Build access context from Gateway-verified headers, falling back to
    query params in MVP/testing mode.

    In production (strict mode) the identity is derived entirely from
    ``X-User-ID``, ``X-Verified-Is-Admin``, and
    ``X-Verified-Project-Memberships`` / ``X-Verified-Project-ID`` headers
    set by the Gateway after JWT validation.
    """
    header_user = (x_user_id or "").strip() or None
    header_admin = (x_verified_is_admin or "").strip()
    header_memberships = (
        (x_verified_project_memberships or "").strip()
        or (x_verified_project_id or "").strip()
        or None
    )

    if header_user:
        admin = header_admin in ("true", "1")
        memberships: set[str] = set()
        if header_memberships:
            memberships = {
                p.strip() for p in header_memberships.split(",") if p.strip()
            }
        return AccessContext(
            user_id=header_user,
            is_admin=admin,
            project_memberships=memberships,
        )

    if _strict_auth_mode():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified caller identity for model access",
        )

    # MVP / testing fallback — use query params
    memberships_set: set[str] = set()
    if project_memberships:
        memberships_set = {
            p.strip() for p in project_memberships.split(",") if p.strip()
        }

    return AccessContext(
        user_id=user_id,
        is_admin=is_admin,
        project_memberships=memberships_set,
    )


def serving_profile_from_request(
    req: ServingProfileRequest | None,
) -> ServingProfile | None:
    """Convert request to ServingProfile."""
    if req is None:
        return None
    return ServingProfile(
        serving_backend=req.serving_backend,
        service_kind=req.service_kind,
        launch_profiles=list(req.launch_profiles),
        container_image=req.container_image,
        service_port=req.service_port,
        readiness_probe=req.readiness_probe,
        readiness_path=req.readiness_path,
        invoke_base_path=req.invoke_base_path,
        tensor_parallel_size=req.tensor_parallel_size,
        max_model_len=req.max_model_len,
        gpu_memory_utilization=req.gpu_memory_utilization,
        dtype=req.dtype,
        quantization=req.quantization,
        served_model_name=req.served_model_name,
        min_gpu_memory_gb=req.min_gpu_memory_gb,
        recommended_gpu_count=req.recommended_gpu_count,
        vision_adapter=req.vision_adapter,
        vision_task=req.vision_task,
        output_format=req.output_format,
        co_located=req.co_located,
    )


def capabilities_from_request(
    req: CapabilitiesRequest | None,
) -> ModelCapabilities | None:
    """Convert request to ModelCapabilities."""
    if req is None:
        return None
    return ModelCapabilities(
        supports_tool_calling=req.supports_tool_calling,
        supports_vision=req.supports_vision,
        supports_structured_output=req.supports_structured_output,
        supports_reasoning=req.supports_reasoning,
        max_output_tokens=req.max_output_tokens,
        max_position_embeddings=req.max_position_embeddings,
        model_family=req.model_family,
        reasoning_effort_levels=req.reasoning_effort_levels,
        reasoning_effort_default=req.reasoning_effort_default,
    )


def _resolve_job_project_id(
    caller_project_id: str | None,
    requested_project_id: str | None,
    ctx: AccessContext,
) -> str:
    """Resolve the durable-job project partition used for job API access."""
    return requested_project_id or caller_project_id or ctx.user_id


def _submit_model_download_job(
    job_store: JobStore,
    *,
    project_id: str,
    ctx: AccessContext,
    spec: dict[str, Any],
) -> str:
    """Create and enqueue a durable model_download job."""
    try:
        job_id = job_store.create_job(
            project_id=project_id,
            user_id=ctx.user_id,
            job_type=JobType.MODEL_DOWNLOAD,
            spec=spec,
            priority=0,
            max_attempts=2,
            restart_policy=RestartPolicy.ON_CRASH,
        )
    except DuplicateJobError as exc:
        logger.info("Reusing existing model download job %s", exc.existing_job_id)
        return exc.existing_job_id

    job_store.enqueue_job(job_id)
    return job_id


def _build_stage_job_spec(model: ModelEntry) -> dict[str, Any]:
    """Build the durable job spec for Hugging Face model staging."""
    return {
        "source": "huggingface",
        "source_id": model.huggingface_id,
        "target_path": model.storage_ref,
        "model_name": model.name,
        "scope": model.scope.value,
        "model_id": model.model_id,
        "model_type": model.model_type.value,
        "metadata": {
            "model_id": model.model_id,
            "owner_id": model.owner_id,
            "project_id": model.project_id,
            "model_type": model.model_type.value,
            "description": model.description,
            "huggingface_id": model.huggingface_id,
            "serving_profile": model.serving_profile.to_dict(),
            "operation": "stage",
        },
    }


def _build_copy_job_spec(
    source_model: ModelEntry,
    target_model: ModelEntry,
) -> dict[str, Any]:
    """Build the durable job spec for system-model copy requests."""
    return {
        "source": "local",
        "source_id": source_model.storage_ref,
        "target_path": target_model.storage_ref,
        "model_name": target_model.name,
        "scope": target_model.scope.value,
        "model_id": target_model.model_id,
        "model_type": target_model.model_type.value,
        "metadata": {
            "model_id": target_model.model_id,
            "owner_id": target_model.owner_id,
            "project_id": target_model.project_id,
            "model_type": target_model.model_type.value,
            "description": target_model.description,
            "huggingface_id": target_model.huggingface_id,
            "serving_profile": target_model.serving_profile.to_dict(),
            "operation": "copy",
            "source_model_id": source_model.model_id,
            "source_model_name": source_model.name,
        },
    }


def _rollback_staged_model(
    store: ModelStore,
    model_id: str,
    ctx: AccessContext,
) -> None:
    """Best-effort cleanup when model job submission fails after catalog creation."""
    try:
        store.delete_model(model_id, ctx)
    except Exception:
        logger.exception(
            "Failed to roll back staged model %s after job submission error", model_id
        )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("", response_model=ModelListResponse)
async def list_models(
    scope: str | None = Query(None, description="Filter by scope"),
    model_type: str | None = Query(None, description="Filter by type"),
    status: str | None = Query(None, description="Filter by status"),
    project_id: str | None = Query(None, description="Filter by project"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> ModelListResponse:
    """
    List models visible to the current user (U-811).

    Returns system models, user's project models, and user's own models.
    """
    # Parse enum filters
    scope_enum = ModelScope(scope) if scope else None
    type_enum = ModelType(model_type) if model_type else None
    status_enum = ModelStatus(status) if status else None

    models = store.list_models(
        ctx=ctx,
        scope=scope_enum,
        model_type=type_enum,
        status=status_enum,
        project_id=project_id,
        limit=limit,
        offset=offset,
    )

    # Apply deployment-scoped visibility (e.g. frontier-only on AI4S).
    from .model_visibility import is_model_visible

    models = [m for m in models if is_model_visible(getattr(m, "name", None))]

    return ModelListResponse(
        models=[ModelResponse.from_entry(m) for m in models],
        total=len(models),  # TODO: Add count query for accurate total
        limit=limit,
        offset=offset,
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> ModelResponse:
    """
    Get model details by ID (U-812).

    Requires read access to the model.
    """
    try:
        model = store.get_model(model_id, ctx)
        return ModelResponse.from_entry(model)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    request: ModelCreateRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> ModelResponse:
    """
    Register a new model in the catalog (U-813).

    The model files must already exist at storage_ref.
    For downloading models, use POST /umrs/models/stage instead.
    """
    try:
        scope = ModelScope(request.scope)
        model_type = ModelType(request.model_type)
        serving_profile = serving_profile_from_request(request.serving_profile)
        capabilities = capabilities_from_request(request.capabilities)
        owner_id = request.owner_id or ctx.user_id

        if scope == ModelScope.SYSTEM:
            owner_id = None

        # Governance quota check (before creation)
        project_id_for_quota = request.project_id or ""
        if project_id_for_quota:
            from .governance import check_quota as _check_model_quota

            _check_model_quota(project_id_for_quota, "models")

        model = store.create_model(
            name=request.name,
            scope=scope,
            storage_ref=request.storage_ref,
            model_type=model_type,
            owner_id=owner_id,
            project_id=request.project_id,
            huggingface_id=request.huggingface_id,
            description=request.description,
            serving_profile=serving_profile,
            capabilities=capabilities,
            ctx=ctx,
        )

        # Governance tracking
        from .governance import record_resource_created

        record_resource_created(
            "models",
            model.model_id,
            ctx.user_id,
            project_id=request.project_id or "",
            tier=scope.value,
        )

        logger.info(
            "User %s created model %s (%s) in %s scope",
            ctx.user_id,
            model.model_id,
            model.name,
            scope.value,
        )

        return ModelResponse.from_entry(model)

    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
    except PathNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except DuplicateModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.post(
    "/stage", response_model=ModelStageResponse, status_code=status.HTTP_202_ACCEPTED
)
async def stage_model(
    request: ModelStageRequest,
    caller_project_id: str | None = Query(
        None,
        description="Caller active project ID for durable job scoping",
    ),
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
    job_store: JobStore = Depends(get_job_store),
) -> ModelStageResponse:
    """
    Request model download/staging (U-814).

    Creates a model_download job to fetch the model from HuggingFace.
    Returns immediately with job_id for tracking.

    The model entry will be created with status='staging' and updated
    to 'available' when the download completes.
    """
    try:
        scope = ModelScope(request.scope)

        # Validate scope permissions
        if scope == ModelScope.SYSTEM:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot stage system models via API. Contact admin.",
            )

        if (
            scope == ModelScope.PROJECT
            and request.project_id not in ctx.project_memberships
        ):
            if not ctx.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not a member of project {request.project_id}",
                )

        # Compute target storage path based on scope
        if scope == ModelScope.PROJECT:
            storage_ref = f"/shared/projects/{request.project_id}/models/{request.name}"
        else:  # USER
            storage_ref = f"/shared/users/{ctx.user_id}/models/{request.name}"

        # Create model entry with staging status
        serving_profile = serving_profile_from_request(request.serving_profile)
        model = store.create_model(
            name=request.name,
            scope=scope,
            storage_ref=storage_ref,
            model_type=ModelType(request.model_type),
            owner_id=ctx.user_id if scope == ModelScope.USER else None,
            project_id=request.project_id if scope == ModelScope.PROJECT else None,
            huggingface_id=request.huggingface_id,
            description=request.description,
            serving_profile=serving_profile,
            status=ModelStatus.STAGING,
            ctx=ctx,
        )

        job_project_id = _resolve_job_project_id(
            caller_project_id=caller_project_id,
            requested_project_id=model.project_id,
            ctx=ctx,
        )

        try:
            job_id = _submit_model_download_job(
                job_store,
                project_id=job_project_id,
                ctx=ctx,
                spec=_build_stage_job_spec(model),
            )
        except Exception:
            _rollback_staged_model(store, model.model_id, ctx)
            raise

        logger.info(
            "User %s requested staging of %s as %s (job %s)",
            ctx.user_id,
            request.huggingface_id,
            model.name,
            job_id,
        )

        return ModelStageResponse(
            job_id=job_id,
            message=f"Model staging initiated. Track progress with: r4 job describe {job_id}",
            model_name=model.name,
            target_scope=scope.value,
        )

    except DuplicateModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    request: ModelUpdateRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> ModelResponse:
    """
    Update model metadata.

    Only description, serving_profile, and status can be updated.
    """
    try:
        serving_profile = serving_profile_from_request(request.serving_profile)
        status_enum = ModelStatus(request.status) if request.status else None
        capabilities = capabilities_from_request(request.capabilities)

        model = store.update_model(
            model_id=model_id,
            ctx=ctx,
            description=request.description,
            serving_profile=serving_profile,
            status=status_enum,
            capabilities=capabilities,
        )

        return ModelResponse.from_entry(model)

    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    delete_files: bool = Query(False, description="Also remove model files from disk"),
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> None:
    """
    Delete a model entry (U-816).

    When delete_files=true, also removes the model directory from disk.
    """
    try:
        model = store.delete_model(model_id, ctx)

        # Governance tracking
        from .governance import record_resource_deleted

        record_resource_deleted("models", model_id)

        logger.info("User %s deleted model %s", ctx.user_id, model_id)

        if delete_files and model.storage_ref:
            import pathlib
            import shutil

            storage_path = pathlib.Path(model.storage_ref)
            if storage_path.is_dir():
                shutil.rmtree(storage_path)
                logger.info(
                    "Removed model files at %s for model %s",
                    model.storage_ref,
                    model_id,
                )
            elif storage_path.is_file():
                storage_path.unlink()
                logger.info(
                    "Removed model file at %s for model %s",
                    model.storage_ref,
                    model_id,
                )
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


# =============================================================================
# Model Publish Endpoint (U-815)
# =============================================================================


class ModelDetectResponse(BaseModel):
    """Response for capability detection."""

    model_id: str
    name: str
    detected: dict[str, Any] = Field(default_factory=dict)
    updated: bool = False
    message: str = ""


@router.post("/{model_id}/detect", response_model=ModelDetectResponse)
async def detect_model_capabilities(
    model_id: str,
    dry_run: bool = Query(False, description="Preview without updating"),
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> ModelDetectResponse:
    """
    Detect model capabilities from files on the server filesystem.

    Reads config.json, tokenizer_config.json, and generation_config.json
    from the model's storage_path to infer capabilities.
    """
    from ..model_registry.detect import detect_capabilities

    try:
        model = store.get_model(model_id)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        ) from None

    if not model.storage_ref:
        return ModelDetectResponse(
            model_id=model.model_id,
            name=model.name,
            message="No storage path configured",
        )

    caps = detect_capabilities(model.storage_ref, model_name=model.name)
    if caps is None:
        return ModelDetectResponse(
            model_id=model.model_id,
            name=model.name,
            message=f"No config files found at {model.storage_ref}",
        )

    detected = caps.to_dict()

    if dry_run:
        return ModelDetectResponse(
            model_id=model.model_id,
            name=model.name,
            detected=detected,
            updated=False,
            message="Dry run — capabilities not saved",
        )

    try:
        store.update_capabilities(model.model_id, caps)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update capabilities: {e}",
        ) from e

    return ModelDetectResponse(
        model_id=model.model_id,
        name=model.name,
        detected=detected,
        updated=True,
        message="Capabilities updated",
    )


class ModelPublishRequest(BaseModel):
    """Request to publish a user model to project scope."""

    target_project_id: str = Field(..., description="Target project ID to publish to")
    new_name: str | None = Field(
        None, description="Override model name in target scope"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target_project_id": "proj-123",
                "new_name": "my-finetune-v1-published",
            }
        }


@router.post(
    "/{model_id}/publish",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
)
async def publish_model(
    model_id: str,
    request: ModelPublishRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> ModelResponse:
    """Publish a user-scoped model to project scope by creating a project copy."""
    try:
        source_model = store.get_model(model_id, ctx)

        if source_model.scope != ModelScope.USER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only user-scoped models can be published to project scope",
            )

        if not ctx.is_admin and source_model.owner_id != ctx.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the model owner can publish it",
            )

        if (
            not ctx.is_admin
            and request.target_project_id not in ctx.project_memberships
            and request.target_project_id not in ctx.admin_project_ids
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User is not a member of project '{request.target_project_id}'",
            )

        published_name = request.new_name or source_model.name
        published_model = store.create_model(
            name=published_name,
            scope=ModelScope.PROJECT,
            storage_ref=source_model.storage_ref,
            model_type=source_model.model_type,
            owner_id=None,
            project_id=request.target_project_id,
            huggingface_id=source_model.huggingface_id,
            description=f"Published from user model: {source_model.description or source_model.name}",
            serving_profile=source_model.serving_profile,
            status=source_model.status,
            model_ref=source_model.model_ref,
            execution_authority_default=source_model.execution_authority_default,
            ctx=ctx,
        )

        logger.info(
            "User %s published model %s to project %s as %s",
            ctx.user_id,
            model_id,
            request.target_project_id,
            published_model.model_id,
        )

        return ModelResponse.from_entry(published_model)

    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        ) from None
    except DuplicateModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


@router.get("/{model_id}/path", response_model=dict[str, str])
async def resolve_model_path(
    model_id: str,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> dict[str, str]:
    """
    Resolve model ID to storage path (for endpoint provisioning).

    This is used internally by UMRS when provisioning endpoints.
    Returns the validated storage_ref for the model.
    """
    try:
        path = store.resolve_model_path(model_id, ctx)
        return {"model_id": model_id, "storage_ref": path}
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


# =============================================================================
# System Model Endpoints (U-817)
# =============================================================================


@router.get("/system", response_model=ModelListResponse)
async def list_system_models(
    model_type: str | None = Query(None, description="Filter by type"),
    status: str | None = Query("available", description="Filter by status"),
    prefer_system: bool = Query(True, description="Include system models first"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    store: ModelStore = Depends(get_model_store),
) -> ModelListResponse:
    """
    List system-level models available to all users (U-817).

    These are admin-managed models that can be used by any project.
    Returns models sorted by name with system models first if prefer_system=true.
    """
    # Create a minimal access context for reading system models (public access)
    ctx = AccessContext(user_id="__system_query__", is_admin=False)

    type_enum = ModelType(model_type) if model_type else None
    status_enum = ModelStatus(status) if status else None

    models = store.list_models(
        ctx=ctx,
        scope=ModelScope.SYSTEM,
        model_type=type_enum,
        status=status_enum,
        limit=limit,
        offset=offset,
    )

    return ModelListResponse(
        models=[ModelResponse.from_entry(m) for m in models],
        total=len(models),
        limit=limit,
        offset=offset,
    )


class SystemModelCreateRequest(BaseModel):
    """Request to register a system-level model with policy (U-817)."""

    name: str = Field(..., min_length=1, max_length=256, description="Model name")
    storage_ref: str = Field(..., description="Filesystem path to model weights")
    model_type: str = Field("llm", description="Type: llm, embedding, or vision")
    huggingface_id: str | None = Field(None, description="Source HuggingFace model ID")
    description: str | None = Field(
        None, max_length=1000, description="Model description"
    )
    serving_profile: ServingProfileRequest | None = Field(
        None, description="Serving configuration"
    )
    execution_authority_default: str = Field(
        "compose", description="Default authority: slurm, k8s, compose"
    )

    # Policy settings
    warm_pool_min: int = Field(0, ge=0, description="Minimum warm endpoints")
    warm_pool_max: int = Field(1, ge=0, description="Maximum warm endpoints")
    warm_pool_ttl_seconds: int = Field(3600, ge=60, description="Warm endpoint TTL")
    enable_warm_pool: bool = Field(False, description="Enable warm pooling")
    allowed_projects: list[str] = Field(
        default_factory=list, description="Allowed project IDs (empty = all)"
    )
    denied_projects: list[str] = Field(
        default_factory=list, description="Denied project IDs"
    )
    max_concurrent_endpoints: int | None = Field(
        None, description="Max concurrent endpoints"
    )
    priority: int = Field(0, description="Scheduling priority")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "qwen-7b-instruct",
                "storage_ref": "/shared/models/Qwen2.5-7B-Instruct",
                "model_type": "llm",
                "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
                "description": "Qwen 7B Instruct - system model for all users",
                "execution_authority_default": "slurm",
                "enable_warm_pool": True,
                "warm_pool_min": 1,
                "warm_pool_max": 3,
            }
        }


class SystemModelCreateResponse(BaseModel):
    """Response for system model creation."""

    model: ModelResponse
    policy: PolicyResponse


@router.post(
    "/system",
    response_model=SystemModelCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_system_model(
    request: SystemModelCreateRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> SystemModelCreateResponse:
    """
    Register a system-level model with associated policy (U-817).

    Admin-only endpoint. Creates both the model entry and its access policy
    in a single atomic transaction.

    System models are available to all users and can have warm pools enabled
    for faster first-request latency.
    """
    if not ctx.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to create system models",
        )

    try:
        from ..model_registry.models import ExecutionAuthority

        serving_profile = serving_profile_from_request(request.serving_profile)
        exec_auth = ExecutionAuthority(request.execution_authority_default)

        model, policy = store.create_system_model(
            name=request.name,
            storage_ref=request.storage_ref,
            model_type=ModelType(request.model_type),
            huggingface_id=request.huggingface_id,
            description=request.description,
            serving_profile=serving_profile,
            execution_authority_default=exec_auth,
            warm_pool_min=request.warm_pool_min,
            warm_pool_max=request.warm_pool_max,
            warm_pool_ttl_seconds=request.warm_pool_ttl_seconds,
            enable_warm_pool=request.enable_warm_pool,
            allowed_projects=request.allowed_projects,
            denied_projects=request.denied_projects,
            max_concurrent_endpoints=request.max_concurrent_endpoints,
            priority=request.priority,
            ctx=ctx,
        )

        logger.info(
            "Admin %s created system model %s (%s) with policy %s",
            ctx.user_id,
            model.model_id,
            model.name,
            policy.policy_id,
        )

        return SystemModelCreateResponse(
            model=ModelResponse.from_entry(model),
            policy=PolicyResponse.from_policy(policy),
        )

    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
    except PathNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except DuplicateModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.post(
    "/{model_name}/download",
    response_model=ModelDownloadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def download_model_to_user_space(
    model_name: str,
    request: ModelDownloadRequest,
    caller_project_id: str | None = Query(
        None,
        description="Caller active project ID for durable job scoping",
    ),
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
    job_store: JobStore = Depends(get_job_store),
) -> ModelDownloadResponse:
    """
    Download/copy a system model to user/project space (U-818).

    This creates a job to copy the model weights from the system model
    to the user's or project's space, creating a new model entry.

    Use case: User wants their own copy of a system model to fine-tune
    or use with custom serving settings.
    """
    try:
        # Get the source system model
        source_model = store.get_model_by_name(
            name=model_name,
            scope=ModelScope.SYSTEM,
            ctx=ctx,
        )

        # Check policy access
        if request.project_id:
            allowed, reason = store.check_policy_access(model_name, request.project_id)
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=reason,
                )

        target_scope = ModelScope(request.target_scope)
        target_name = request.custom_name or f"{model_name}-copy"

        # Validate target scope permissions
        if target_scope == ModelScope.SYSTEM:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot download to system scope. Use project or user scope.",
            )

        if target_scope == ModelScope.PROJECT:
            if not request.project_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="project_id required for project scope",
                )
            if request.project_id not in ctx.project_memberships and not ctx.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not a member of project {request.project_id}",
                )
            storage_ref = f"/shared/projects/{request.project_id}/models/{target_name}"
        else:  # USER
            storage_ref = f"/shared/users/{ctx.user_id}/models/{target_name}"

        target_model = store.create_model(
            name=target_name,
            scope=target_scope,
            storage_ref=storage_ref,
            model_type=source_model.model_type,
            owner_id=ctx.user_id if target_scope == ModelScope.USER else None,
            project_id=request.project_id
            if target_scope == ModelScope.PROJECT
            else None,
            huggingface_id=source_model.huggingface_id,
            description=source_model.description or f"Copy of system/{model_name}",
            serving_profile=source_model.serving_profile,
            execution_authority_default=source_model.execution_authority_default,
            status=ModelStatus.STAGING,
            ctx=ctx,
        )

        job_project_id = _resolve_job_project_id(
            caller_project_id=caller_project_id,
            requested_project_id=target_model.project_id,
            ctx=ctx,
        )

        try:
            job_id = _submit_model_download_job(
                job_store,
                project_id=job_project_id,
                ctx=ctx,
                spec=_build_copy_job_spec(source_model, target_model),
            )
        except Exception:
            _rollback_staged_model(store, target_model.model_id, ctx)
            raise

        logger.info(
            "User %s requested copy of system/%s to %s/%s (job %s)",
            ctx.user_id,
            model_name,
            target_scope.value,
            target_name,
            job_id,
        )

        return ModelDownloadResponse(
            job_id=job_id,
            message=f"Model copy initiated. Track progress with: r4 job describe {job_id}",
            source_model=f"system/{model_name}",
            target_name=target_name,
            target_scope=target_scope.value,
        )

    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System model not found: {model_name}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


# =============================================================================
# Model Policy by Name Endpoints (U-819, U-820)
# =============================================================================


class ModelPolicyUpdateRequest(BaseModel):
    """Request to update a model's policy by model name (U-820)."""

    allowed_projects: list[str] | None = Field(
        None, description="New allowed projects list"
    )
    denied_projects: list[str] | None = Field(
        None, description="New denied projects list"
    )
    max_concurrent_endpoints: int | None = Field(
        None, description="Max concurrent endpoints"
    )
    enable_warm_pool: bool | None = Field(
        None, description="Enable/disable warm pooling"
    )
    warm_pool_min: int | None = Field(None, ge=0, description="Minimum warm endpoints")
    warm_pool_max: int | None = Field(None, ge=0, description="Maximum warm endpoints")


@router.get("/{model_name}/policy", response_model=PolicyResponse)
async def get_model_policy(
    model_name: str,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> PolicyResponse:
    """
    Get the policy for a model by name (U-819).

    Returns the access control and quota settings for the specified model.
    Admin or project owner can view policies.
    """
    policy = store.get_policy_by_model_name(model_name)

    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No policy found for model: {model_name}",
        )

    return PolicyResponse.from_policy(policy)


@router.patch("/{model_name}/policy", response_model=PolicyResponse)
async def update_model_policy(
    model_name: str,
    request: ModelPolicyUpdateRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> PolicyResponse:
    """
    Update the policy for a model by name (U-820).

    Admin-only endpoint. Updates access control and quotas for the specified model.
    """
    if not ctx.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to update model policies",
        )

    try:
        policy = store.update_policy_by_model_name(
            model_name=model_name,
            allowed_projects=request.allowed_projects,
            denied_projects=request.denied_projects,
            max_concurrent_endpoints=request.max_concurrent_endpoints,
            enable_warm_pool=request.enable_warm_pool,
            warm_pool_min=request.warm_pool_min,
            warm_pool_max=request.warm_pool_max,
            ctx=ctx,
        )

        logger.info("Admin %s updated policy for model %s", ctx.user_id, model_name)

        return PolicyResponse.from_policy(policy)

    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No policy found for model: {model_name}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


# =============================================================================
# Policy Collection Endpoints
# =============================================================================

policy_router = APIRouter(prefix="/umrs/models/policies", tags=["policies"])


@policy_router.get("", response_model=PolicyListResponse)
async def list_policies(
    is_system_model: bool | None = Query(
        None, description="Filter by system model flag"
    ),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    store: ModelStore = Depends(get_model_store),
) -> PolicyListResponse:
    """
    List model policies (U-819).

    Policies define access control, quotas, and warm pool settings for models.
    """
    policies = store.list_policies(
        is_system_model=is_system_model,
        limit=limit,
        offset=offset,
    )

    return PolicyListResponse(
        policies=[PolicyResponse.from_policy(p) for p in policies],
        total=len(policies),
        limit=limit,
        offset=offset,
    )


@policy_router.post(
    "", response_model=PolicyResponse, status_code=status.HTTP_201_CREATED
)
async def create_policy(
    request: PolicyCreateRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> PolicyResponse:
    """
    Create a model policy (U-820).

    Admin only. Defines access control and quotas for a model.
    """
    try:
        serving_profile = serving_profile_from_request(request.default_serving_profile)

        policy = store.create_policy(
            model_name=request.model_name,
            allowed_projects=request.allowed_projects,
            denied_projects=request.denied_projects,
            max_concurrent_endpoints=request.max_concurrent_endpoints,
            max_tokens_per_day=request.max_tokens_per_day,
            priority=request.priority,
            warm_pool_min=request.warm_pool_min,
            warm_pool_max=request.warm_pool_max,
            warm_pool_ttl_seconds=request.warm_pool_ttl_seconds,
            enable_warm_pool=request.enable_warm_pool,
            is_system_model=request.is_system_model,
            default_serving_profile=serving_profile,
            ctx=ctx,
        )

        logger.info(
            "Admin %s created policy %s for model %s",
            ctx.user_id,
            policy.policy_id,
            policy.model_name,
        )

        return PolicyResponse.from_policy(policy)

    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


@policy_router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(
    policy_id: str,
    store: ModelStore = Depends(get_model_store),
) -> PolicyResponse:
    """Get a policy by ID."""
    try:
        policy = store.get_policy(policy_id)
        return PolicyResponse.from_policy(policy)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy not found: {policy_id}",
        ) from None


@policy_router.get("/check/{model_name}", response_model=PolicyCheckResponse)
async def check_policy_access(
    model_name: str,
    project_id: str = Query(..., description="Project requesting access"),
    store: ModelStore = Depends(get_model_store),
) -> PolicyCheckResponse:
    """
    Check if a project has access to a model based on policy.

    Returns allowed=True if access is permitted, with optional reason if denied.
    """
    allowed, reason = store.check_policy_access(model_name, project_id)
    return PolicyCheckResponse(allowed=allowed, reason=reason)


@policy_router.patch("/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: str,
    request: PolicyUpdateRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> PolicyResponse:
    """
    Update a model policy.

    Admin only.
    """
    try:
        policy = store.update_policy(
            policy_id=policy_id,
            allowed_projects=request.allowed_projects,
            denied_projects=request.denied_projects,
            max_concurrent_endpoints=request.max_concurrent_endpoints,
            enable_warm_pool=request.enable_warm_pool,
            warm_pool_min=request.warm_pool_min,
            warm_pool_max=request.warm_pool_max,
            ctx=ctx,
        )

        return PolicyResponse.from_policy(policy)

    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy not found: {policy_id}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e


@policy_router.delete("/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_policy(
    policy_id: str,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> None:
    """
    Delete a model policy.

    Admin only.
    """
    try:
        store.delete_policy(policy_id, ctx)
        logger.info("Admin %s deleted policy %s", ctx.user_id, policy_id)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy not found: {policy_id}",
        ) from None
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
