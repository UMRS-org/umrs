"""
Pipeline and RAG Profile REST API (X-001, X-002).

Provides:
- POST /umrs/pipelines - Create pipeline definition
- GET /umrs/pipelines - List pipelines
- GET /umrs/pipelines/{pipeline_id} - Get pipeline details
- PATCH /umrs/pipelines/{pipeline_id} - Update pipeline
- DELETE /umrs/pipelines/{pipeline_id} - Delete pipeline

- POST /umrs/rag-profiles - Create RAG profile
- GET /umrs/rag-profiles - List RAG profiles
- GET /umrs/rag-profiles/{profile_id} - Get RAG profile details
- PATCH /umrs/rag-profiles/{profile_id} - Update RAG profile
- DELETE /umrs/rag-profiles/{profile_id} - Delete RAG profile

- GET /umrs/components/chunkers - List available chunkers
- GET /umrs/components/embedders - List available embedders
- GET /umrs/components/rerankers - List available rerankers

See DEVELOPMENT_OBJECTIVES.md Section 11.1 for design details.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from ..rag.models import (
    ChunkerSpec,
    ChunkerType,
    EmbedderSpec,
    EmbedderType,
    PipelineDefinition,
    PipelineStatus,
    RAGProfile,
    RerankerSpec,
    RerankerType,
)
from ..rag.registry import (
    get_chunker_registry,
    get_component_validator,
    get_embedder_registry,
    get_reranker_registry,
)
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs", tags=["pipelines"])


# =============================================================================
# In-Memory Store (Replace with DB in production)
# =============================================================================


class PipelineStore:
    """In-memory store for pipeline definitions."""

    def __init__(self):
        self._pipelines: dict[str, PipelineDefinition] = {}
        self._rag_profiles: dict[str, RAGProfile] = {}

    # Pipeline operations
    def create_pipeline(self, pipeline: PipelineDefinition) -> PipelineDefinition:
        """Create a new pipeline definition."""
        if pipeline.pipeline_id in self._pipelines:
            raise ValueError(f"Pipeline {pipeline.pipeline_id} already exists")

        # Check for duplicate names in project
        for p in self._pipelines.values():
            if p.project_id == pipeline.project_id and p.name == pipeline.name:
                raise ValueError(
                    f"Pipeline named '{pipeline.name}' already exists in project"
                )

        # If this is default, unset other defaults
        if pipeline.is_default:
            for p in self._pipelines.values():
                if p.project_id == pipeline.project_id and p.is_default:
                    p.is_default = False

        self._pipelines[pipeline.pipeline_id] = pipeline
        return pipeline

    def get_pipeline(self, pipeline_id: str) -> PipelineDefinition | None:
        """Get pipeline by ID."""
        return self._pipelines.get(pipeline_id)

    def list_pipelines(
        self,
        project_id: str | None = None,
        owner_id: str | None = None,
        status_filter: PipelineStatus | None = None,
    ) -> list[PipelineDefinition]:
        """List pipelines with optional filters."""
        return _filter_store_items(
            self._pipelines.values(), project_id, owner_id, status_filter
        )

    def update_pipeline(
        self,
        pipeline_id: str,
        updates: dict[str, Any],
    ) -> PipelineDefinition | None:
        """Update pipeline fields."""
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(pipeline, key):
                setattr(pipeline, key, value)

        pipeline.updated_at = datetime.now()
        return pipeline

    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete pipeline."""
        if pipeline_id in self._pipelines:
            del self._pipelines[pipeline_id]
            return True
        return False

    # RAG Profile operations
    def create_rag_profile(self, profile: RAGProfile) -> RAGProfile:
        """Create a new RAG profile."""
        if profile.profile_id in self._rag_profiles:
            raise ValueError(f"RAG profile {profile.profile_id} already exists")

        # Check for duplicate names in project
        for p in self._rag_profiles.values():
            if p.project_id == profile.project_id and p.name == profile.name:
                raise ValueError(
                    f"RAG profile named '{profile.name}' already exists in project"
                )

        # If this is default, unset other defaults
        if profile.is_default:
            for p in self._rag_profiles.values():
                if p.project_id == profile.project_id and p.is_default:
                    p.is_default = False

        self._rag_profiles[profile.profile_id] = profile
        return profile

    def get_rag_profile(self, profile_id: str) -> RAGProfile | None:
        """Get RAG profile by ID."""
        return self._rag_profiles.get(profile_id)

    def list_rag_profiles(
        self,
        project_id: str | None = None,
        owner_id: str | None = None,
        status_filter: PipelineStatus | None = None,
    ) -> list[RAGProfile]:
        """List RAG profiles with optional filters."""
        return _filter_store_items(
            self._rag_profiles.values(), project_id, owner_id, status_filter
        )

    def update_rag_profile(
        self,
        profile_id: str,
        updates: dict[str, Any],
    ) -> RAGProfile | None:
        """Update RAG profile fields."""
        profile = self._rag_profiles.get(profile_id)
        if not profile:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.updated_at = datetime.now()
        return profile

    def delete_rag_profile(self, profile_id: str) -> bool:
        """Delete RAG profile."""
        if profile_id in self._rag_profiles:
            del self._rag_profiles[profile_id]
            return True
        return False


def _store_item_matches(
    item,
    project_id: str | None,
    owner_id: str | None,
    status_filter: PipelineStatus | None,
) -> bool:
    """Predicate for store-item listing (project/owner/status)."""
    if project_id and item.project_id != project_id:
        return False
    if owner_id and item.owner_id != owner_id:
        return False
    return not (status_filter and item.status != status_filter)


def _filter_store_items(
    items,
    project_id: str | None,
    owner_id: str | None,
    status_filter: PipelineStatus | None,
) -> list:
    """Shared store-item filtering (project, owner, status)."""
    return [
        p for p in items if _store_item_matches(p, project_id, owner_id, status_filter)
    ]


# Global store instance
_pipeline_store: PipelineStore | None = None


def get_pipeline_store() -> PipelineStore:
    """Get global pipeline store singleton."""
    global _pipeline_store
    if _pipeline_store is None:
        _pipeline_store = PipelineStore()
    return _pipeline_store


def set_pipeline_store(store: PipelineStore) -> None:
    """Set the pipeline store (for testing)."""
    global _pipeline_store
    _pipeline_store = store


# =============================================================================
# Request/Response Models - Pipeline
# =============================================================================


class ChunkerSpecRequest(BaseModel):
    """Chunker configuration request."""

    chunker_type: str = Field(
        "token", description="Chunker type: token, sentence, etc."
    )
    chunk_size: int = Field(512, ge=10, le=10000, description="Chunk size")
    chunk_overlap: int = Field(50, ge=0, le=1000, description="Chunk overlap")
    separator: str | None = Field(None, description="Custom separator")
    min_chunk_size: int = Field(100, ge=1, description="Minimum chunk size")
    max_chunk_size: int | None = Field(None, description="Maximum chunk size")
    respect_boundaries: bool = Field(True, description="Respect sentence boundaries")
    language: str = Field("en", description="Language code")
    code_language: str | None = Field(
        None, description="Code language for code chunker"
    )


class EmbedderSpecRequest(BaseModel):
    """Embedder configuration request."""

    embedder_type: str = Field("sentence_transformer", description="Embedder type")
    model_name: str = Field("all-MiniLM-L6-v2", description="Model name")
    model_path: str | None = Field(None, description="Local model path")
    dimension: int = Field(384, ge=1, description="Embedding dimension")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size")
    normalize: bool = Field(True, description="Normalize embeddings")
    device: str = Field("auto", description="Device: auto, cuda, cpu")
    max_seq_length: int | None = Field(None, description="Max sequence length")
    instruction: str | None = Field(None, description="Instruction prefix")


class PipelineCreateRequest(BaseModel):
    """Request to create a pipeline definition (X-001)."""

    name: str = Field(..., min_length=1, max_length=256, description="Pipeline name")
    chunker: ChunkerSpecRequest | None = Field(None, description="Chunker config")
    embedder: EmbedderSpecRequest | None = Field(None, description="Embedder config")
    description: str | None = Field(None, max_length=1000, description="Description")
    is_default: bool = Field(False, description="Set as default for project")
    supported_extensions: list[str] | None = Field(
        None, description="Supported file extensions"
    )
    extract_metadata: bool = Field(True, description="Extract metadata from files")
    metadata_fields: list[str] | None = Field(
        None, description="Metadata fields to extract"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "my-custom-pipeline",
                "description": "Custom pipeline for scientific papers",
                "chunker": {
                    "chunker_type": "sentence",
                    "chunk_size": 5,
                    "chunk_overlap": 1,
                },
                "embedder": {
                    "model_name": "BAAI/bge-base-en-v1.5",
                    "dimension": 768,
                },
                "supported_extensions": [".pdf", ".txt", ".md"],
            }
        }
    )


class PipelineUpdateRequest(BaseModel):
    """Request to update a pipeline definition."""

    name: str | None = Field(None, min_length=1, max_length=256)
    description: str | None = Field(None, max_length=1000)
    status: str | None = Field(None, description="Status: active, draft, deprecated")
    is_default: bool | None = Field(None)
    chunker: ChunkerSpecRequest | None = Field(None)
    embedder: EmbedderSpecRequest | None = Field(None)
    supported_extensions: list[str] | None = Field(None)


class PipelineResponse(BaseModel):
    """Pipeline definition response."""

    pipeline_id: str
    name: str
    project_id: str
    owner_id: str
    description: str | None
    status: str
    is_default: bool
    chunker: dict[str, Any]
    embedder: dict[str, Any]
    supported_extensions: list[str]
    extract_metadata: bool
    metadata_fields: list[str]
    created_at: str
    updated_at: str


# =============================================================================
# Request/Response Models - RAG Profile
# =============================================================================


class RerankerSpecRequest(BaseModel):
    """Reranker configuration request."""

    reranker_type: str = Field("cross_encoder", description="Reranker type")
    model_name: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", description="Model name"
    )
    model_path: str | None = Field(None, description="Local model path")
    top_k: int = Field(5, ge=1, le=100, description="Top K results")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size")
    score_threshold: float | None = Field(None, description="Score threshold")
    device: str = Field("auto", description="Device: auto, cuda, cpu")


class RAGProfileCreateRequest(BaseModel):
    """Request to create a RAG profile (X-002)."""

    name: str = Field(..., min_length=1, max_length=256, description="Profile name")
    search_type: str = Field(
        "hybrid", description="Search type: vector, keyword, hybrid"
    )
    vector_weight: float = Field(0.7, ge=0, le=1, description="Vector search weight")
    keyword_weight: float = Field(0.3, ge=0, le=1, description="Keyword search weight")
    top_k_retrieval: int = Field(
        20, ge=1, le=100, description="Initial retrieval count"
    )
    reranker: RerankerSpecRequest | None = Field(None, description="Reranker config")
    max_context_tokens: int = Field(
        4096, ge=256, le=32768, description="Max context tokens"
    )
    context_template: str | None = Field(None, description="Custom context template")
    include_sources: bool = Field(True, description="Include source citations")
    default_model: str = Field(
        "rivault-gpt-oss-static", description="Default LLM model"
    )
    system_prompt: str | None = Field(None, description="System prompt for LLM")
    temperature: float = Field(0.3, ge=0, le=2, description="LLM temperature")
    max_tokens: int = Field(1024, ge=1, le=8192, description="Max LLM tokens")
    description: str | None = Field(None, max_length=1000, description="Description")
    is_default: bool = Field(False, description="Set as default for project")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "scientific-qa",
                "description": "RAG profile for scientific Q&A",
                "search_type": "hybrid",
                "vector_weight": 0.8,
                "keyword_weight": 0.2,
                "top_k_retrieval": 30,
                "reranker": {
                    "model_name": "BAAI/bge-reranker-base",
                    "top_k": 10,
                },
                "max_context_tokens": 8192,
            }
        }
    )


class RAGProfileUpdateRequest(BaseModel):
    """Request to update a RAG profile."""

    name: str | None = Field(None, min_length=1, max_length=256)
    description: str | None = Field(None, max_length=1000)
    status: str | None = Field(None)
    is_default: bool | None = Field(None)
    search_type: str | None = Field(None)
    vector_weight: float | None = Field(None, ge=0, le=1)
    keyword_weight: float | None = Field(None, ge=0, le=1)
    top_k_retrieval: int | None = Field(None, ge=1, le=100)
    reranker: RerankerSpecRequest | None = Field(None)
    max_context_tokens: int | None = Field(None, ge=256, le=32768)
    default_model: str | None = Field(None)
    temperature: float | None = Field(None, ge=0, le=2)


class RAGProfileResponse(BaseModel):
    """RAG profile response."""

    profile_id: str
    name: str
    project_id: str
    owner_id: str
    description: str | None
    status: str
    is_default: bool
    search_type: str
    vector_weight: float
    keyword_weight: float
    top_k_retrieval: int
    reranker: dict[str, Any]
    max_context_tokens: int
    context_template: str | None
    include_sources: bool
    default_model: str
    system_prompt: str | None
    temperature: float
    max_tokens: int
    created_at: str
    updated_at: str


# =============================================================================
# Helper Functions
# =============================================================================


def extract_user_project(
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
) -> tuple[str, str]:
    """Extract user and project IDs from headers."""
    user_id = x_user_id or "anonymous"
    project_id = x_project_id or "default"
    return user_id, project_id


def chunker_request_to_spec(req: ChunkerSpecRequest | None) -> ChunkerSpec:
    """Convert request to ChunkerSpec."""
    if not req:
        return ChunkerSpec()
    return ChunkerSpec(
        chunker_type=ChunkerType(req.chunker_type),
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        separator=req.separator,
        min_chunk_size=req.min_chunk_size,
        max_chunk_size=req.max_chunk_size,
        respect_boundaries=req.respect_boundaries,
        language=req.language,
        code_language=req.code_language,
    )


def embedder_request_to_spec(req: EmbedderSpecRequest | None) -> EmbedderSpec:
    """Convert request to EmbedderSpec."""
    if not req:
        return EmbedderSpec()
    return EmbedderSpec(
        embedder_type=EmbedderType(req.embedder_type),
        model_name=req.model_name,
        model_path=req.model_path,
        dimension=req.dimension,
        batch_size=req.batch_size,
        normalize=req.normalize,
        device=req.device,
        max_seq_length=req.max_seq_length,
        instruction=req.instruction,
    )


def reranker_request_to_spec(req: RerankerSpecRequest | None) -> RerankerSpec:
    """Convert request to RerankerSpec."""
    if not req:
        return RerankerSpec()
    return RerankerSpec(
        reranker_type=RerankerType(req.reranker_type),
        model_name=req.model_name,
        model_path=req.model_path,
        top_k=req.top_k,
        batch_size=req.batch_size,
        score_threshold=req.score_threshold,
        device=req.device,
    )


def pipeline_to_response(pipeline: PipelineDefinition) -> PipelineResponse:
    """Convert PipelineDefinition to response model."""
    return PipelineResponse(
        pipeline_id=pipeline.pipeline_id,
        name=pipeline.name,
        project_id=pipeline.project_id,
        owner_id=pipeline.owner_id,
        description=pipeline.description,
        status=pipeline.status.value,
        is_default=pipeline.is_default,
        chunker=pipeline.chunker.to_dict(),
        embedder=pipeline.embedder.to_dict(),
        supported_extensions=pipeline.supported_extensions,
        extract_metadata=pipeline.extract_metadata,
        metadata_fields=pipeline.metadata_fields,
        created_at=pipeline.created_at.isoformat(),
        updated_at=pipeline.updated_at.isoformat(),
    )


def rag_profile_to_response(profile: RAGProfile) -> RAGProfileResponse:
    """Convert RAGProfile to response model."""
    return RAGProfileResponse(
        profile_id=profile.profile_id,
        name=profile.name,
        project_id=profile.project_id,
        owner_id=profile.owner_id,
        description=profile.description,
        status=profile.status.value,
        is_default=profile.is_default,
        search_type=profile.search_type,
        vector_weight=profile.vector_weight,
        keyword_weight=profile.keyword_weight,
        top_k_retrieval=profile.top_k_retrieval,
        reranker=profile.reranker.to_dict(),
        max_context_tokens=profile.max_context_tokens,
        context_template=profile.context_template,
        include_sources=profile.include_sources,
        default_model=profile.default_model,
        system_prompt=profile.system_prompt,
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        created_at=profile.created_at.isoformat(),
        updated_at=profile.updated_at.isoformat(),
    )


# =============================================================================
# Pipeline Endpoints (X-001)
# =============================================================================


@router.post(
    "/pipelines", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED
)
async def create_pipeline(
    request: PipelineCreateRequest,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> PipelineResponse:
    """
    Create a new pipeline definition (X-001).

    Pipeline definitions configure how documents are processed during ingestion:
    - Chunking strategy (token, sentence, semantic, etc.)
    - Embedding model selection
    - Metadata extraction rules
    """
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    # Governance quota check
    try:
        from .governance import check_quota

        check_quota(project_id, "pipelines")
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise
        # governance not active

    # Validate components
    _validate_pipeline_components(request)

    pipeline = _pipeline_definition_from_request(
        request, user_id=user_id, project_id=project_id
    )

    try:
        pipeline = store.create_pipeline(pipeline)
    except ValueError as e:
        raise_typed(PlatformError.pipelines("conflict", str(e)), cause=e)

    # Governance tracking
    from .governance import record_resource_created

    record_resource_created(
        "pipelines",
        pipeline.pipeline_id,
        user_id,
        project_id=project_id,
    )

    logger.info(f"Created pipeline {pipeline.pipeline_id} for project {project_id}")
    return pipeline_to_response(pipeline)


@router.get("/pipelines", response_model=list[PipelineResponse])
async def list_pipelines(
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    status_filter: str | None = Query(None, description="Filter by status"),
    store: PipelineStore = Depends(get_pipeline_store),
) -> list[PipelineResponse]:
    """List pipeline definitions for the current project."""
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    status_enum = PipelineStatus(status_filter) if status_filter else None
    pipelines = store.list_pipelines(project_id=project_id, status_filter=status_enum)

    return [pipeline_to_response(p) for p in pipelines]


@router.get("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: str,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> PipelineResponse:
    """Get pipeline definition by ID."""
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    pipeline = store.get_pipeline(pipeline_id)
    if not pipeline:
        raise_typed(
            PlatformError.pipelines("not_found", f"Pipeline {pipeline_id} not found")
        )

    # Check project access
    if pipeline.project_id != project_id:
        raise_typed(
            PlatformError.pipelines("forbidden", "Access denied to this pipeline")
        )

    return pipeline_to_response(pipeline)


@router.patch("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: str,
    request: PipelineUpdateRequest,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> PipelineResponse:
    """Update pipeline definition."""
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    pipeline = store.get_pipeline(pipeline_id)
    if not pipeline:
        raise_typed(
            PlatformError.pipelines("not_found", f"Pipeline {pipeline_id} not found")
        )

    if pipeline.project_id != project_id:
        raise_typed(
            PlatformError.pipelines("forbidden", "Access denied to this pipeline")
        )

    updates = _pipeline_updates(request)
    pipeline = store.update_pipeline(pipeline_id, updates)
    return pipeline_to_response(pipeline)


@router.delete("/pipelines/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pipeline(
    pipeline_id: str,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> None:
    """Delete pipeline definition."""
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    pipeline = store.get_pipeline(pipeline_id)
    if not pipeline:
        raise_typed(
            PlatformError.pipelines("not_found", f"Pipeline {pipeline_id} not found")
        )

    if pipeline.project_id != project_id:
        raise_typed(
            PlatformError.pipelines("forbidden", "Access denied to this pipeline")
        )

    store.delete_pipeline(pipeline_id)

    # Governance tracking
    from .governance import record_resource_deleted

    record_resource_deleted("pipelines", pipeline_id, project_id)

    logger.info(f"Deleted pipeline {pipeline_id}")


def _validate_pipeline_components(request: PipelineCreateRequest) -> None:
    """Validate chunker/embedder specs against the component registry."""
    validator = get_component_validator()
    if request.chunker:
        errors = validator.validate_chunker(request.chunker.chunker_type)
        if errors:
            raise_typed(PlatformError.pipelines("bad_spec", errors[0]))
    if request.embedder:
        errors = validator.validate_embedder(request.embedder.model_name)
        if errors:
            raise_typed(PlatformError.pipelines("bad_spec", errors[0]))


def _pipeline_definition_from_request(
    request: PipelineCreateRequest,
    *,
    user_id: str,
    project_id: str,
) -> PipelineDefinition:
    """Build a PipelineDefinition from a create request."""
    return PipelineDefinition.create(
        name=request.name,
        project_id=project_id,
        owner_id=user_id,
        chunker=chunker_request_to_spec(request.chunker),
        embedder=embedder_request_to_spec(request.embedder),
        description=request.description,
        is_default=request.is_default,
        supported_extensions=request.supported_extensions
        or [".txt", ".md", ".pdf", ".html", ".json"],
        extract_metadata=request.extract_metadata,
        metadata_fields=request.metadata_fields
        or ["title", "author", "date", "source"],
    )


def _pipeline_updates(request: PipelineUpdateRequest) -> dict[str, Any]:
    """Collect non-None update fields (converted specs where applicable)."""
    updates: dict[str, Any] = {}
    for attr, value in (
        ("name", request.name),
        ("description", request.description),
        ("supported_extensions", request.supported_extensions),
    ):
        if value is not None:
            updates[attr] = value
    if request.status is not None:
        updates["status"] = PipelineStatus(request.status)
    if request.is_default is not None:
        updates["is_default"] = request.is_default
    if request.chunker is not None:
        updates["chunker"] = chunker_request_to_spec(request.chunker)
    if request.embedder is not None:
        updates["embedder"] = embedder_request_to_spec(request.embedder)
    return updates


# =============================================================================
# RAG Profile Endpoints (X-002) — RETIRED (R3, 2026-08-22)
# =============================================================================


def _rag_profiles_gone() -> None:
    """rag-profiles CRUD retired (owner decision 2026-08-22).

    Profiles persisted to an ephemeral in-memory store, were never read at
    runtime, and raged nothing — retirement returns 410 Gone instead of a
    deceptive 2xx on a dead store. Router removal scheduled for the next
    release.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=(
            "rag-profiles CRUD is retired: profiles were ephemeral "
            "(in-memory), never consulted at ingest/search time, and are not "
            "part of the runtime path. Use site config for retrieval tuning; "
            "this router family is removed in the next release."
        ),
    )


@router.post(
    "/rag-profiles",
    response_model=RAGProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_rag_profile(
    request: RAGProfileCreateRequest,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> RAGProfileResponse:
    """
    Create a new RAG profile (X-002).

    RAG profiles configure how queries are processed:
    - Search strategy (vector, keyword, hybrid)
    - Reranking configuration
    - Context assembly parameters
    - LLM integration settings
    """
    _rag_profiles_gone()
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    # Governance quota check
    try:
        from .governance import check_quota

        check_quota(project_id, "rag_profiles")
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise
        # governance not active

    # Validate reranker
    validator = get_component_validator()
    if request.reranker:
        errors = validator.validate_reranker(request.reranker.model_name)
        if errors:
            raise_typed(PlatformError.pipelines("bad_spec", errors[0]))

    profile = RAGProfile.create(
        name=request.name,
        project_id=project_id,
        owner_id=user_id,
        search_type=request.search_type,
        vector_weight=request.vector_weight,
        keyword_weight=request.keyword_weight,
        top_k_retrieval=request.top_k_retrieval,
        reranker=reranker_request_to_spec(request.reranker),
        max_context_tokens=request.max_context_tokens,
        context_template=request.context_template,
        include_sources=request.include_sources,
        default_model=request.default_model,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        description=request.description,
        is_default=request.is_default,
    )

    try:
        profile = store.create_rag_profile(profile)
    except ValueError as e:
        raise_typed(PlatformError.pipelines("conflict", str(e)), cause=e)

    # Governance tracking
    from .governance import record_resource_created

    record_resource_created(
        "rag_profiles",
        profile.profile_id,
        user_id,
        project_id=project_id,
    )

    logger.info(f"Created RAG profile {profile.profile_id} for project {project_id}")
    return rag_profile_to_response(profile)


@router.get("/rag-profiles", response_model=list[RAGProfileResponse])
async def list_rag_profiles(
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    status_filter: str | None = Query(None, description="Filter by status"),
    store: PipelineStore = Depends(get_pipeline_store),
) -> list[RAGProfileResponse]:
    """List RAG profiles for the current project."""
    _rag_profiles_gone()
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    status_enum = PipelineStatus(status_filter) if status_filter else None
    profiles = store.list_rag_profiles(project_id=project_id, status_filter=status_enum)

    return [rag_profile_to_response(p) for p in profiles]


@router.get("/rag-profiles/{profile_id}", response_model=RAGProfileResponse)
async def get_rag_profile(
    profile_id: str,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> RAGProfileResponse:
    """Get RAG profile by ID."""
    _rag_profiles_gone()
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    profile = store.get_rag_profile(profile_id)
    if not profile:
        raise_typed(
            PlatformError.pipelines(
                "rag_profile_not_found", f"RAG profile {profile_id} not found"
            )
        )

    if profile.project_id != project_id:
        raise_typed(
            PlatformError.pipelines("forbidden", "Access denied to this profile")
        )

    return rag_profile_to_response(profile)


@router.patch("/rag-profiles/{profile_id}", response_model=RAGProfileResponse)
async def update_rag_profile(
    profile_id: str,
    request: RAGProfileUpdateRequest,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> RAGProfileResponse:
    """Update RAG profile."""
    _rag_profiles_gone()
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    profile = store.get_rag_profile(profile_id)
    if not profile:
        raise_typed(
            PlatformError.pipelines(
                "rag_profile_not_found", f"RAG profile {profile_id} not found"
            )
        )

    if profile.project_id != project_id:
        raise_typed(
            PlatformError.pipelines("forbidden", "Access denied to this profile")
        )

    updates = _rag_profile_updates_from_request(request)

    profile = store.update_rag_profile(profile_id, updates)
    return rag_profile_to_response(profile)


_RAG_PROFILE_UPDATE_FIELD_SPECS = (
    ("name", "plain"),
    ("description", "plain"),
    ("status", "enum"),
    ("is_default", "plain"),
    ("search_type", "plain"),
    ("vector_weight", "plain"),
    ("keyword_weight", "plain"),
    ("top_k_retrieval", "plain"),
    ("reranker", "spec"),
    ("max_context_tokens", "plain"),
    ("default_model", "plain"),
    ("temperature", "plain"),
)


def _rag_profile_updates_from_request(request: RAGProfileUpdateRequest) -> dict:
    """Build the updates dict from a RAGProfileUpdateRequest."""
    updates = {}
    for attr, kind in _RAG_PROFILE_UPDATE_FIELD_SPECS:
        value = getattr(request, attr)
        if value is None:
            continue
        if kind == "enum":
            updates[attr] = PipelineStatus(value)
        elif kind == "spec":
            updates[attr] = reranker_request_to_spec(value)
        else:
            updates[attr] = value
    return updates


@router.delete("/rag-profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rag_profile(
    profile_id: str,
    x_user_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    store: PipelineStore = Depends(get_pipeline_store),
) -> None:
    """Delete RAG profile."""
    _rag_profiles_gone()
    user_id, project_id = extract_user_project(x_user_id, x_project_id)

    profile = store.get_rag_profile(profile_id)
    if not profile:
        raise_typed(
            PlatformError.pipelines(
                "rag_profile_not_found", f"RAG profile {profile_id} not found"
            )
        )

    if profile.project_id != project_id:
        raise_typed(
            PlatformError.pipelines("forbidden", "Access denied to this profile")
        )

    store.delete_rag_profile(profile_id)

    # Governance tracking
    from .governance import record_resource_deleted

    record_resource_deleted("rag_profiles", profile_id, project_id)

    logger.info(f"Deleted RAG profile {profile_id}")


# =============================================================================
# Component Registry Endpoints (X-003, X-004, X-005)
# =============================================================================


@router.get("/components/chunkers")
async def list_chunkers() -> list[dict[str, Any]]:
    """List available chunking methods (X-003)."""
    registry = get_chunker_registry()
    return [entry.to_dict() for entry in registry.list_all()]


@router.get("/components/embedders")
async def list_embedders(
    language: str | None = Query(None, description="Filter by language"),
    gpu_only: bool | None = Query(None, description="Filter by GPU requirement"),
) -> list[dict[str, Any]]:
    """List available embedding models (X-004)."""
    registry = get_embedder_registry()

    embedders = registry.list_all()
    if language:
        embedders = [
            e
            for e in embedders
            if language in e.languages or "multilingual" in e.languages
        ]
    if gpu_only is not None:
        embedders = [e for e in embedders if e.requires_gpu == gpu_only]

    return [entry.to_dict() for entry in embedders]


@router.get("/components/rerankers")
async def list_rerankers() -> list[dict[str, Any]]:
    """List available reranking methods (X-005)."""
    registry = get_reranker_registry()
    return [entry.to_dict() for entry in registry.list_all()]
