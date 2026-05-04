"""
RAG (Retrieval-Augmented Generation) module.

Provides:
- Search: BM25, HybridSearcher, CrossEncoderReranker, ContextAssembler, RAGPipeline
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
from .search import (
    BM25,
    ContextAssembler,
    CrossEncoderReranker,
    HybridSearcher,
    RAGContentScanner,
    RAGContext,
    RAGPipeline,
    SearchResult,
    get_rag_scanner,
)

__all__ = [
    # Search
    "BM25",
    "ContextAssembler",
    "CrossEncoderReranker",
    "HybridSearcher",
    "RAGContext",
    "RAGContentScanner",
    "RAGPipeline",
    "SearchResult",
    "get_rag_scanner",
    # Pipeline / RAG config models
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
