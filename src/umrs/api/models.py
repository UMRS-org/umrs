"""
Model Registry REST API (U-811 ~ U-816).

Provides:
- GET /umrs/models - List models visible to user
- GET /umrs/models/{model_id} - Get model details
- POST /umrs/models - Register a new model
- POST /umrs/models/stage - Request model download (creates job)
- DELETE /umrs/models/{model_id} - Delete model entry

See DEVELOPMENT_OBJECTIVES.md Section 1.9.2 for design details.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..model_registry.models import (
    ModelEntry,
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/models", tags=["models"])


# =============================================================================
# Dependency Injection
# =============================================================================

# Global model store instance (set during app startup)
_model_store: Optional[ModelStore] = None


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


# =============================================================================
# Request/Response Models
# =============================================================================


class ServingProfileRequest(BaseModel):
    """Serving configuration for vLLM/LiteLLM."""

    tensor_parallel_size: int = Field(1, ge=1, le=8, description="Number of GPUs for tensor parallelism")
    max_model_len: Optional[int] = Field(None, description="Maximum context length")
    gpu_memory_utilization: float = Field(0.9, ge=0.1, le=1.0, description="GPU memory fraction")
    dtype: str = Field("auto", description="Data type: auto, float16, bfloat16")
    quantization: Optional[str] = Field(None, description="Quantization: awq, gptq, None")
    served_model_name: Optional[str] = Field(None, description="Override model name in API")
    min_gpu_memory_gb: Optional[float] = Field(None, description="Minimum GPU memory required")
    recommended_gpu_count: int = Field(1, ge=1, description="Recommended number of GPUs")


class ModelCreateRequest(BaseModel):
    """Request to register a new model."""

    name: str = Field(..., min_length=1, max_length=256, description="Model name (unique per scope)")
    scope: str = Field(..., description="Scope: system, project, user")
    storage_ref: str = Field(..., description="Filesystem path to model weights")
    model_type: str = Field("llm", description="Type: llm or embedding")

    owner_id: Optional[str] = Field(None, description="Owner user ID (required for user scope)")
    project_id: Optional[str] = Field(None, description="Project ID (required for project scope)")

    huggingface_id: Optional[str] = Field(None, description="Source HuggingFace model ID")
    description: Optional[str] = Field(None, max_length=1000, description="Model description")
    serving_profile: Optional[ServingProfileRequest] = Field(None, description="Serving configuration")

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
    name: str = Field(..., min_length=1, max_length=256, description="Name for the model")
    scope: str = Field("user", description="Target scope: project or user")
    model_type: str = Field("llm", description="Type: llm or embedding")

    project_id: Optional[str] = Field(None, description="Project ID (for project scope)")
    description: Optional[str] = Field(None, description="Model description")
    serving_profile: Optional[ServingProfileRequest] = Field(None, description="Serving configuration")

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

    description: Optional[str] = Field(None, description="New description")
    serving_profile: Optional[ServingProfileRequest] = Field(None, description="New serving profile")
    status: Optional[str] = Field(None, description="New status: available, deprecated")


class ModelResponse(BaseModel):
    """Model data for API responses."""

    model_id: str
    name: str
    scope: str
    owner_id: Optional[str]
    project_id: Optional[str]
    storage_ref: str
    model_type: str
    huggingface_id: Optional[str]
    description: Optional[str]
    serving_profile: Dict[str, Any]
    status: str
    created_at: Optional[str]
    updated_at: Optional[str]

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
            model_type=entry.model_type.value,
            huggingface_id=entry.huggingface_id,
            description=entry.description,
            serving_profile=entry.serving_profile.to_dict(),
            status=entry.status.value,
            created_at=entry.created_at.isoformat() if entry.created_at else None,
            updated_at=entry.updated_at.isoformat() if entry.updated_at else None,
        )


class ModelListResponse(BaseModel):
    """Response for model list endpoint."""

    models: List[ModelResponse]
    total: int
    limit: int
    offset: int


class ModelStageResponse(BaseModel):
    """Response for model staging request."""

    job_id: str
    message: str
    model_name: str
    target_scope: str


# =============================================================================
# Helper Functions
# =============================================================================


def get_access_context(
    user_id: str = Query(..., description="User ID"),
    is_admin: bool = Query(False, description="Admin flag"),
    project_memberships: Optional[str] = Query(
        None, description="Comma-separated project IDs"
    ),
) -> AccessContext:
    """
    Build access context from request headers/params.

    In production, this would come from JWT claims via Gateway.
    For now, accept as query params for testing.
    """
    memberships: Set[str] = set()
    if project_memberships:
        memberships = {p.strip() for p in project_memberships.split(",") if p.strip()}

    return AccessContext(
        user_id=user_id,
        is_admin=is_admin,
        project_memberships=memberships,
    )


def serving_profile_from_request(
    req: Optional[ServingProfileRequest],
) -> Optional[ServingProfile]:
    """Convert request to ServingProfile."""
    if req is None:
        return None
    return ServingProfile(
        tensor_parallel_size=req.tensor_parallel_size,
        max_model_len=req.max_model_len,
        gpu_memory_utilization=req.gpu_memory_utilization,
        dtype=req.dtype,
        quantization=req.quantization,
        served_model_name=req.served_model_name,
        min_gpu_memory_gb=req.min_gpu_memory_gb,
        recommended_gpu_count=req.recommended_gpu_count,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("", response_model=ModelListResponse)
async def list_models(
    scope: Optional[str] = Query(None, description="Filter by scope"),
    model_type: Optional[str] = Query(None, description="Filter by type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
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
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


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

        model = store.create_model(
            name=request.name,
            scope=scope,
            storage_ref=request.storage_ref,
            model_type=model_type,
            owner_id=request.owner_id or ctx.user_id,
            project_id=request.project_id,
            huggingface_id=request.huggingface_id,
            description=request.description,
            serving_profile=serving_profile,
            ctx=ctx,
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
        )
    except PathNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except DuplicateModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/stage", response_model=ModelStageResponse, status_code=status.HTTP_202_ACCEPTED)
async def stage_model(
    request: ModelStageRequest,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
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

        if scope == ModelScope.PROJECT and request.project_id not in ctx.project_memberships:
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

        # TODO: Create model_download job via JobStore
        # For now, return a placeholder job_id
        # In the full implementation:
        # job_id = await job_store.create_job(
        #     project_id=request.project_id or ctx.user_id,
        #     user_id=ctx.user_id,
        #     job_type="model_download",
        #     spec={
        #         "model_id": model.model_id,
        #         "huggingface_id": request.huggingface_id,
        #         "target_path": storage_ref,
        #         "model_type": request.model_type,
        #     },
        # )
        job_id = f"job-staging-{model.model_id[-8:]}"

        logger.info(
            "User %s requested staging of %s as %s (job %s)",
            ctx.user_id,
            request.huggingface_id,
            model.name,
            job_id,
        )

        return ModelStageResponse(
            job_id=job_id,
            message=f"Model staging initiated. Track progress with: aisci job describe {job_id}",
            model_name=model.name,
            target_scope=scope.value,
        )

    except DuplicateModelError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


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

        model = store.update_model(
            model_id=model_id,
            ctx=ctx,
            description=request.description,
            serving_profile=serving_profile,
            status=status_enum,
        )

        return ModelResponse.from_entry(model)

    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> None:
    """
    Delete a model entry (U-816).

    Note: This only removes the catalog entry, not the actual model files.
    """
    try:
        store.delete_model(model_id, ctx)
        logger.info("User %s deleted model %s", ctx.user_id, model_id)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get("/{model_id}/path", response_model=Dict[str, str])
async def resolve_model_path(
    model_id: str,
    ctx: AccessContext = Depends(get_access_context),
    store: ModelStore = Depends(get_model_store),
) -> Dict[str, str]:
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
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
