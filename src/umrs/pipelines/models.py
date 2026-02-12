"""
Pipeline and RAG Profile data models (X-001 ~ X-005).

Provides:
- PipelineDefinition: Custom ingestion pipeline configuration
- RAGProfile: RAG search and retrieval configuration
- ChunkerSpec: Chunking configuration
- EmbedderSpec: Embedding model configuration
- RerankerSpec: Reranking model configuration

See DEVELOPMENT_OBJECTIVES.md Section 11.1 for design details.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class ChunkerType(Enum):
    """Supported chunking methods (X-003)."""

    TOKEN = "token"  # Token-based splitting
    SENTENCE = "sentence"  # Sentence-based splitting
    PARAGRAPH = "paragraph"  # Paragraph-based splitting
    RECURSIVE = "recursive"  # Recursive character splitting
    SEMANTIC = "semantic"  # Semantic-aware splitting
    CODE = "code"  # Code-aware splitting (respects functions, classes)


class EmbedderType(Enum):
    """Supported embedding models (X-004)."""

    SENTENCE_TRANSFORMER = "sentence_transformer"  # Local SentenceTransformers
    HUGGINGFACE = "huggingface"  # HuggingFace transformers
    OPENAI = "openai"  # OpenAI embeddings (via Gateway)
    COHERE = "cohere"  # Cohere embeddings
    CUSTOM = "custom"  # Custom model path


class RerankerType(Enum):
    """Supported reranking methods (X-005)."""

    NONE = "none"  # No reranking
    CROSS_ENCODER = "cross_encoder"  # Cross-encoder reranking
    BM25 = "bm25"  # BM25 score reranking
    COHERE = "cohere"  # Cohere reranker
    CUSTOM = "custom"  # Custom reranker model


class PipelineStatus(Enum):
    """Pipeline definition status."""

    ACTIVE = "active"  # Ready to use
    DRAFT = "draft"  # Not yet validated
    DEPRECATED = "deprecated"  # Marked for removal
    DISABLED = "disabled"  # Temporarily disabled


# =============================================================================
# Component Specifications
# =============================================================================


@dataclass
class ChunkerSpec:
    """Chunking configuration for a pipeline."""

    chunker_type: ChunkerType = ChunkerType.TOKEN
    chunk_size: int = 512  # Tokens or characters per chunk
    chunk_overlap: int = 50  # Overlap between chunks
    separator: Optional[str] = None  # Custom separator for splitting
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: Optional[int] = None  # Maximum chunk size

    # Advanced options
    respect_boundaries: bool = True  # Don't split mid-sentence
    language: str = "en"  # Language for sentence detection
    code_language: Optional[str] = None  # For code chunker

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunker_type": self.chunker_type.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "respect_boundaries": self.respect_boundaries,
            "language": self.language,
            "code_language": self.code_language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkerSpec":
        """Create from dictionary."""
        return cls(
            chunker_type=ChunkerType(data.get("chunker_type", "token")),
            chunk_size=data.get("chunk_size", 512),
            chunk_overlap=data.get("chunk_overlap", 50),
            separator=data.get("separator"),
            min_chunk_size=data.get("min_chunk_size", 100),
            max_chunk_size=data.get("max_chunk_size"),
            respect_boundaries=data.get("respect_boundaries", True),
            language=data.get("language", "en"),
            code_language=data.get("code_language"),
        )


@dataclass
class EmbedderSpec:
    """Embedding configuration for a pipeline."""

    embedder_type: EmbedderType = EmbedderType.SENTENCE_TRANSFORMER
    model_name: str = "all-MiniLM-L6-v2"  # Model identifier
    model_path: Optional[str] = None  # Local model path
    dimension: int = 384  # Embedding dimension
    batch_size: int = 32  # Batch size for embedding
    normalize: bool = True  # Normalize embeddings to unit length

    # Advanced options
    device: str = "auto"  # cuda, cpu, or auto
    max_seq_length: Optional[int] = None  # Max sequence length
    instruction: Optional[str] = None  # Instruction for instruction-tuned models

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedder_type": self.embedder_type.value,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbedderSpec":
        """Create from dictionary."""
        return cls(
            embedder_type=EmbedderType(data.get("embedder_type", "sentence_transformer")),
            model_name=data.get("model_name", "all-MiniLM-L6-v2"),
            model_path=data.get("model_path"),
            dimension=data.get("dimension", 384),
            batch_size=data.get("batch_size", 32),
            normalize=data.get("normalize", True),
            device=data.get("device", "auto"),
            max_seq_length=data.get("max_seq_length"),
            instruction=data.get("instruction"),
        )


@dataclass
class RerankerSpec:
    """Reranking configuration for a RAG profile."""

    reranker_type: RerankerType = RerankerType.CROSS_ENCODER
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_path: Optional[str] = None  # Local model path
    top_k: int = 5  # Number of documents to keep after reranking
    batch_size: int = 32  # Batch size for reranking

    # Advanced options
    score_threshold: Optional[float] = None  # Minimum score threshold
    device: str = "auto"  # cuda, cpu, or auto

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reranker_type": self.reranker_type.value,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "score_threshold": self.score_threshold,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankerSpec":
        """Create from dictionary."""
        return cls(
            reranker_type=RerankerType(data.get("reranker_type", "cross_encoder")),
            model_name=data.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            model_path=data.get("model_path"),
            top_k=data.get("top_k", 5),
            batch_size=data.get("batch_size", 32),
            score_threshold=data.get("score_threshold"),
            device=data.get("device", "auto"),
        )


# =============================================================================
# Pipeline Definition (X-001)
# =============================================================================


def generate_pipeline_id() -> str:
    """Generate unique pipeline ID."""
    return f"pipe-{uuid.uuid4().hex[:12]}"


def generate_rag_profile_id() -> str:
    """Generate unique RAG profile ID."""
    return f"rag-{uuid.uuid4().hex[:12]}"


@dataclass
class PipelineDefinition:
    """
    Custom ingestion pipeline definition (X-001).

    Defines how documents are processed before storage:
    - Chunking strategy
    - Embedding model
    - Metadata extraction
    - File type handling
    """

    pipeline_id: str
    name: str
    project_id: str
    owner_id: str

    # Pipeline components
    chunker: ChunkerSpec = field(default_factory=ChunkerSpec)
    embedder: EmbedderSpec = field(default_factory=EmbedderSpec)

    # Configuration
    description: Optional[str] = None
    status: PipelineStatus = PipelineStatus.ACTIVE
    is_default: bool = False  # Default pipeline for project

    # Supported file types
    supported_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".html", ".json"]
    )

    # Metadata extraction
    extract_metadata: bool = True
    metadata_fields: List[str] = field(
        default_factory=lambda: ["title", "author", "date", "source"]
    )

    # Pre/post processing hooks (plugin references)
    pre_processors: List[str] = field(default_factory=list)
    post_processors: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now())
    updated_at: datetime = field(default_factory=lambda: datetime.now())

    @classmethod
    def create(
        cls,
        name: str,
        project_id: str,
        owner_id: str,
        **kwargs: Any,
    ) -> "PipelineDefinition":
        """Create a new pipeline definition."""
        return cls(
            pipeline_id=generate_pipeline_id(),
            name=name,
            project_id=project_id,
            owner_id=owner_id,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "project_id": self.project_id,
            "owner_id": self.owner_id,
            "chunker": self.chunker.to_dict(),
            "embedder": self.embedder.to_dict(),
            "description": self.description,
            "status": self.status.value,
            "is_default": self.is_default,
            "supported_extensions": self.supported_extensions,
            "extract_metadata": self.extract_metadata,
            "metadata_fields": self.metadata_fields,
            "pre_processors": self.pre_processors,
            "post_processors": self.post_processors,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineDefinition":
        """Create from dictionary."""
        return cls(
            pipeline_id=data["pipeline_id"],
            name=data["name"],
            project_id=data["project_id"],
            owner_id=data["owner_id"],
            chunker=ChunkerSpec.from_dict(data.get("chunker", {})),
            embedder=EmbedderSpec.from_dict(data.get("embedder", {})),
            description=data.get("description"),
            status=PipelineStatus(data.get("status", "active")),
            is_default=data.get("is_default", False),
            supported_extensions=data.get(
                "supported_extensions", [".txt", ".md", ".pdf", ".html", ".json"]
            ),
            extract_metadata=data.get("extract_metadata", True),
            metadata_fields=data.get(
                "metadata_fields", ["title", "author", "date", "source"]
            ),
            pre_processors=data.get("pre_processors", []),
            post_processors=data.get("post_processors", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(),
        )


# =============================================================================
# RAG Profile (X-002)
# =============================================================================


@dataclass
class RAGProfile:
    """
    RAG search and retrieval configuration (X-002).

    Defines how RAG queries are processed:
    - Search strategy (vector, keyword, hybrid)
    - Reranking configuration
    - Context assembly parameters
    - LLM integration settings
    """

    profile_id: str
    name: str
    project_id: str
    owner_id: str

    # Search configuration
    search_type: str = "hybrid"  # vector, keyword, hybrid
    vector_weight: float = 0.7  # Weight for vector search (hybrid)
    keyword_weight: float = 0.3  # Weight for BM25 (hybrid)
    top_k_retrieval: int = 20  # Documents to retrieve initially

    # Reranker configuration
    reranker: RerankerSpec = field(default_factory=RerankerSpec)

    # Context assembly
    max_context_tokens: int = 4096  # Maximum tokens in context
    context_template: Optional[str] = None  # Custom context template
    include_sources: bool = True  # Include source citations

    # LLM integration
    default_model: str = "rivault-gpt-oss-static"
    system_prompt: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1024

    # Configuration
    description: Optional[str] = None
    status: PipelineStatus = PipelineStatus.ACTIVE
    is_default: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now())
    updated_at: datetime = field(default_factory=lambda: datetime.now())

    @classmethod
    def create(
        cls,
        name: str,
        project_id: str,
        owner_id: str,
        **kwargs: Any,
    ) -> "RAGProfile":
        """Create a new RAG profile."""
        return cls(
            profile_id=generate_rag_profile_id(),
            name=name,
            project_id=project_id,
            owner_id=owner_id,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "project_id": self.project_id,
            "owner_id": self.owner_id,
            "search_type": self.search_type,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "top_k_retrieval": self.top_k_retrieval,
            "reranker": self.reranker.to_dict(),
            "max_context_tokens": self.max_context_tokens,
            "context_template": self.context_template,
            "include_sources": self.include_sources,
            "default_model": self.default_model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "description": self.description,
            "status": self.status.value,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGProfile":
        """Create from dictionary."""
        return cls(
            profile_id=data["profile_id"],
            name=data["name"],
            project_id=data["project_id"],
            owner_id=data["owner_id"],
            search_type=data.get("search_type", "hybrid"),
            vector_weight=data.get("vector_weight", 0.7),
            keyword_weight=data.get("keyword_weight", 0.3),
            top_k_retrieval=data.get("top_k_retrieval", 20),
            reranker=RerankerSpec.from_dict(data.get("reranker", {})),
            max_context_tokens=data.get("max_context_tokens", 4096),
            context_template=data.get("context_template"),
            include_sources=data.get("include_sources", True),
            default_model=data.get("default_model", "rivault-gpt-oss-static"),
            system_prompt=data.get("system_prompt"),
            temperature=data.get("temperature", 0.3),
            max_tokens=data.get("max_tokens", 1024),
            description=data.get("description"),
            status=PipelineStatus(data.get("status", "active")),
            is_default=data.get("is_default", False),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(),
        )
