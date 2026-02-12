"""
Pipeline and RAG Profile management module.

Provides:
- PipelineDefinition: Custom ingestion pipeline configuration (X-001)
- RAGProfile: RAG search and retrieval configuration (X-002)
- ChunkerRegistry: Available chunking methods (X-003)
- EmbedderRegistry: Available embedding models (X-004)
- RerankerRegistry: Available reranking methods (X-005)
"""

from .models import (
    ChunkerSpec,
    ChunkerType,
    EmbedderSpec,
    EmbedderType,
    PipelineDefinition,
    PipelineStatus,
    RAGProfile,
    RerankerSpec,
    RerankerType,
    generate_pipeline_id,
    generate_rag_profile_id,
)
from .registry import (
    ChunkerEntry,
    ChunkerRegistry,
    ComponentValidator,
    EmbedderEntry,
    EmbedderRegistry,
    RerankerEntry,
    RerankerRegistry,
    get_chunker_registry,
    get_component_validator,
    get_embedder_registry,
    get_reranker_registry,
)

__all__ = [
    # Models
    "ChunkerSpec",
    "ChunkerType",
    "EmbedderSpec",
    "EmbedderType",
    "PipelineDefinition",
    "PipelineStatus",
    "RAGProfile",
    "RerankerSpec",
    "RerankerType",
    "generate_pipeline_id",
    "generate_rag_profile_id",
    # Registry entries
    "ChunkerEntry",
    "EmbedderEntry",
    "RerankerEntry",
    # Registries
    "ChunkerRegistry",
    "EmbedderRegistry",
    "RerankerRegistry",
    "ComponentValidator",
    # Global accessors
    "get_chunker_registry",
    "get_embedder_registry",
    "get_reranker_registry",
    "get_component_validator",
]
