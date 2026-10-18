"""
Component registries for Pipeline and RAG (X-003 ~ X-005).

Provides:
- ChunkerRegistry: Available chunking methods (X-003)
- EmbedderRegistry: Available embedding models (X-004)
- RerankerRegistry: Available reranking methods (X-005)
- ComponentValidator: Validates component configurations

These registries define the supported options and can be extended
via plugins for custom implementations.

Current runtime reach (R6.3, 2026-08-23): these registries feed ONLY the
read-only catalog surface ``GET /umrs/components/{chunkers|embedders|rerankers}``
(via ``umrs.api.pipelines``). RAG ingestion and query in ``rag_service`` never
consume them, and the ``/umrs/rag-profiles*`` CRUD (410 after wave-1 R3;
deleted 2026-10-18 with its last client) was the only other consumer —
treat these as catalog/test-seed data, not as execution-affecting
configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .models import ChunkerType, EmbedderType, RerankerType

logger = logging.getLogger(__name__)


# =============================================================================
# Registry Entry Types
# =============================================================================


@dataclass
class ChunkerEntry:
    """Registry entry for a chunker."""

    chunker_type: ChunkerType
    name: str
    description: str
    default_chunk_size: int = 512
    default_overlap: int = 50
    supports_languages: list[str] = field(default_factory=lambda: ["en"])
    requires_model: bool = False
    is_builtin: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunker_type": self.chunker_type.value,
            "name": self.name,
            "description": self.description,
            "default_chunk_size": self.default_chunk_size,
            "default_overlap": self.default_overlap,
            "supports_languages": self.supports_languages,
            "requires_model": self.requires_model,
            "is_builtin": self.is_builtin,
        }


@dataclass
class EmbedderEntry:
    """Registry entry for an embedding model."""

    embedder_type: EmbedderType
    model_name: str
    display_name: str
    description: str
    dimension: int
    max_seq_length: int = 512
    requires_gpu: bool = False
    model_size_mb: int = 0
    languages: list[str] = field(default_factory=lambda: ["en"])
    is_builtin: bool = True
    model_path: str | None = None  # For pre-staged models

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedder_type": self.embedder_type.value,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "description": self.description,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "requires_gpu": self.requires_gpu,
            "model_size_mb": self.model_size_mb,
            "languages": self.languages,
            "is_builtin": self.is_builtin,
            "model_path": self.model_path,
        }


@dataclass
class RerankerEntry:
    """Registry entry for a reranker."""

    reranker_type: RerankerType
    model_name: str
    display_name: str
    description: str
    requires_gpu: bool = False
    model_size_mb: int = 0
    is_builtin: bool = True
    model_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reranker_type": self.reranker_type.value,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "description": self.description,
            "requires_gpu": self.requires_gpu,
            "model_size_mb": self.model_size_mb,
            "is_builtin": self.is_builtin,
            "model_path": self.model_path,
        }


# =============================================================================
# Chunker Registry (X-003)
# =============================================================================


class ChunkerRegistry:
    """
    Registry of supported chunking methods (X-003).

    Maintains a list of available chunkers with their configurations
    and validation rules.
    """

    def __init__(self):
        self._chunkers: dict[str, ChunkerEntry] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in chunkers."""
        builtins = [
            ChunkerEntry(
                chunker_type=ChunkerType.TOKEN,
                name="token",
                description="Split text by token count using tiktoken",
                default_chunk_size=512,
                default_overlap=50,
            ),
            ChunkerEntry(
                chunker_type=ChunkerType.SENTENCE,
                name="sentence",
                description="Split text by sentences using spaCy",
                default_chunk_size=5,  # Number of sentences
                default_overlap=1,
                supports_languages=["en", "de", "fr", "es", "ja", "zh"],
                requires_model=True,
            ),
            ChunkerEntry(
                chunker_type=ChunkerType.PARAGRAPH,
                name="paragraph",
                description="Split text by paragraphs (double newlines)",
                default_chunk_size=1,  # Paragraphs
                default_overlap=0,
            ),
            ChunkerEntry(
                chunker_type=ChunkerType.RECURSIVE,
                name="recursive",
                description="Recursively split by multiple separators",
                default_chunk_size=1000,
                default_overlap=100,
            ),
            ChunkerEntry(
                chunker_type=ChunkerType.SEMANTIC,
                name="semantic",
                description="Semantically-aware chunking using embeddings",
                default_chunk_size=512,
                default_overlap=50,
                requires_model=True,
            ),
            ChunkerEntry(
                chunker_type=ChunkerType.CODE,
                name="code",
                description="Code-aware chunking respecting functions/classes",
                default_chunk_size=100,  # Lines
                default_overlap=10,
                supports_languages=[
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "cpp",
                ],
            ),
        ]

        for chunker in builtins:
            self._chunkers[chunker.name] = chunker

    def register(self, entry: ChunkerEntry) -> None:
        """Register a custom chunker."""
        self._chunkers[entry.name] = entry
        logger.info(f"Registered chunker: {entry.name}")

    def get(self, name: str) -> ChunkerEntry | None:
        """Get chunker by name."""
        return self._chunkers.get(name)

    def list_all(self) -> list[ChunkerEntry]:
        """List all registered chunkers."""
        return list(self._chunkers.values())

    def list_names(self) -> list[str]:
        """List all chunker names."""
        return list(self._chunkers.keys())

    def is_valid(self, name: str) -> bool:
        """Check if chunker name is valid."""
        return name in self._chunkers


# =============================================================================
# Embedder Registry (X-004)
# =============================================================================


class EmbedderRegistry:
    """
    Registry of supported embedding models (X-004).

    Maintains a list of available embedding models with their
    configurations and resource requirements.
    """

    def __init__(self):
        self._embedders: dict[str, EmbedderEntry] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in embedding models."""
        builtins = [
            EmbedderEntry(
                embedder_type=EmbedderType.SENTENCE_TRANSFORMER,
                model_name="all-MiniLM-L6-v2",
                display_name="MiniLM-L6",
                description="Fast, lightweight English embeddings",
                dimension=384,
                max_seq_length=256,
                requires_gpu=False,
                model_size_mb=80,
                languages=["en"],
            ),
            EmbedderEntry(
                embedder_type=EmbedderType.SENTENCE_TRANSFORMER,
                model_name="all-mpnet-base-v2",
                display_name="MPNet Base",
                description="High-quality English embeddings",
                dimension=768,
                max_seq_length=384,
                requires_gpu=False,
                model_size_mb=420,
                languages=["en"],
            ),
            EmbedderEntry(
                embedder_type=EmbedderType.SENTENCE_TRANSFORMER,
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                display_name="Multilingual MiniLM",
                description="Multilingual embeddings for 50+ languages",
                dimension=384,
                max_seq_length=128,
                requires_gpu=False,
                model_size_mb=450,
                languages=["multilingual"],
            ),
            EmbedderEntry(
                embedder_type=EmbedderType.SENTENCE_TRANSFORMER,
                model_name="BAAI/bge-small-en-v1.5",
                display_name="BGE Small EN",
                description="Efficient BGE embeddings for English",
                dimension=384,
                max_seq_length=512,
                requires_gpu=False,
                model_size_mb=120,
                languages=["en"],
            ),
            EmbedderEntry(
                embedder_type=EmbedderType.SENTENCE_TRANSFORMER,
                model_name="BAAI/bge-base-en-v1.5",
                display_name="BGE Base EN",
                description="High-quality BGE embeddings for English",
                dimension=768,
                max_seq_length=512,
                requires_gpu=False,
                model_size_mb=440,
                languages=["en"],
            ),
            EmbedderEntry(
                embedder_type=EmbedderType.SENTENCE_TRANSFORMER,
                model_name="intfloat/e5-small-v2",
                display_name="E5 Small",
                description="Efficient E5 embeddings",
                dimension=384,
                max_seq_length=512,
                requires_gpu=False,
                model_size_mb=120,
                languages=["en"],
            ),
            EmbedderEntry(
                embedder_type=EmbedderType.SENTENCE_TRANSFORMER,
                model_name="intfloat/e5-base-v2",
                display_name="E5 Base",
                description="High-quality E5 embeddings",
                dimension=768,
                max_seq_length=512,
                requires_gpu=False,
                model_size_mb=440,
                languages=["en"],
            ),
        ]

        for embedder in builtins:
            self._embedders[embedder.model_name] = embedder

    def register(self, entry: EmbedderEntry) -> None:
        """Register a custom embedder."""
        self._embedders[entry.model_name] = entry
        logger.info(f"Registered embedder: {entry.model_name}")

    def get(self, model_name: str) -> EmbedderEntry | None:
        """Get embedder by model name."""
        return self._embedders.get(model_name)

    def list_all(self) -> list[EmbedderEntry]:
        """List all registered embedders."""
        return list(self._embedders.values())

    def list_names(self) -> list[str]:
        """List all embedder model names."""
        return list(self._embedders.keys())

    def is_valid(self, model_name: str) -> bool:
        """Check if embedder model name is valid."""
        return model_name in self._embedders

    def filter_by_language(self, language: str) -> list[EmbedderEntry]:
        """Filter embedders by supported language."""
        return [
            e
            for e in self._embedders.values()
            if language in e.languages or "multilingual" in e.languages
        ]

    def filter_by_gpu(self, requires_gpu: bool) -> list[EmbedderEntry]:
        """Filter embedders by GPU requirement."""
        return [e for e in self._embedders.values() if e.requires_gpu == requires_gpu]


# =============================================================================
# Reranker Registry (X-005)
# =============================================================================


class RerankerRegistry:
    """
    Registry of supported reranking methods (X-005).

    Maintains a list of available rerankers with their
    configurations and resource requirements.
    """

    def __init__(self):
        self._rerankers: dict[str, RerankerEntry] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in rerankers."""
        builtins = [
            RerankerEntry(
                reranker_type=RerankerType.NONE,
                model_name="none",
                display_name="No Reranking",
                description="Skip reranking, use initial retrieval scores",
                requires_gpu=False,
            ),
            RerankerEntry(
                reranker_type=RerankerType.CROSS_ENCODER,
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                display_name="MiniLM Cross-Encoder",
                description="Fast cross-encoder for English reranking",
                requires_gpu=False,
                model_size_mb=80,
            ),
            RerankerEntry(
                reranker_type=RerankerType.CROSS_ENCODER,
                model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
                display_name="TinyBERT Cross-Encoder",
                description="Ultra-fast cross-encoder for English",
                requires_gpu=False,
                model_size_mb=17,
            ),
            RerankerEntry(
                reranker_type=RerankerType.CROSS_ENCODER,
                model_name="BAAI/bge-reranker-base",
                display_name="BGE Reranker Base",
                description="BGE reranker for English",
                requires_gpu=False,
                model_size_mb=440,
            ),
            RerankerEntry(
                reranker_type=RerankerType.CROSS_ENCODER,
                model_name="BAAI/bge-reranker-large",
                display_name="BGE Reranker Large",
                description="High-quality BGE reranker for English",
                requires_gpu=True,
                model_size_mb=1340,
            ),
            RerankerEntry(
                reranker_type=RerankerType.BM25,
                model_name="bm25",
                display_name="BM25 Reranking",
                description="Keyword-based BM25 score reranking",
                requires_gpu=False,
            ),
        ]

        for reranker in builtins:
            self._rerankers[reranker.model_name] = reranker

    def register(self, entry: RerankerEntry) -> None:
        """Register a custom reranker."""
        self._rerankers[entry.model_name] = entry
        logger.info(f"Registered reranker: {entry.model_name}")

    def get(self, model_name: str) -> RerankerEntry | None:
        """Get reranker by model name."""
        return self._rerankers.get(model_name)

    def list_all(self) -> list[RerankerEntry]:
        """List all registered rerankers."""
        return list(self._rerankers.values())

    def list_names(self) -> list[str]:
        """List all reranker model names."""
        return list(self._rerankers.keys())

    def is_valid(self, model_name: str) -> bool:
        """Check if reranker model name is valid."""
        return model_name in self._rerankers


# =============================================================================
# Component Validator
# =============================================================================


class ComponentValidator:
    """
    Validates pipeline and RAG profile component configurations.

    Ensures all referenced chunkers, embedders, and rerankers are
    registered and properly configured.
    """

    def __init__(
        self,
        chunker_registry: ChunkerRegistry | None = None,
        embedder_registry: EmbedderRegistry | None = None,
        reranker_registry: RerankerRegistry | None = None,
    ):
        self.chunker_registry = chunker_registry or ChunkerRegistry()
        self.embedder_registry = embedder_registry or EmbedderRegistry()
        self.reranker_registry = reranker_registry or RerankerRegistry()

    def validate_chunker(self, chunker_type: str) -> list[str]:
        """Validate chunker type. Returns list of errors."""
        errors = []
        if not self.chunker_registry.is_valid(chunker_type):
            valid_names = self.chunker_registry.list_names()
            errors.append(
                f"Invalid chunker type '{chunker_type}'. "
                f"Valid options: {', '.join(valid_names)}"
            )
        return errors

    def validate_embedder(self, model_name: str) -> list[str]:
        """Validate embedder model. Returns list of errors."""
        errors = []
        if not self.embedder_registry.is_valid(model_name):
            valid_names = self.embedder_registry.list_names()
            errors.append(
                f"Invalid embedder model '{model_name}'. "
                f"Valid options: {', '.join(valid_names[:5])}..."
            )
        return errors

    def validate_reranker(self, model_name: str) -> list[str]:
        """Validate reranker model. Returns list of errors."""
        errors = []
        if not self.reranker_registry.is_valid(model_name):
            valid_names = self.reranker_registry.list_names()
            errors.append(
                f"Invalid reranker model '{model_name}'. "
                f"Valid options: {', '.join(valid_names)}"
            )
        return errors


# =============================================================================
# Global Registry Instances
# =============================================================================


_chunker_registry: ChunkerRegistry | None = None
_embedder_registry: EmbedderRegistry | None = None
_reranker_registry: RerankerRegistry | None = None


def get_chunker_registry() -> ChunkerRegistry:
    """Get global chunker registry singleton."""
    global _chunker_registry
    if _chunker_registry is None:
        _chunker_registry = ChunkerRegistry()
    return _chunker_registry


def get_embedder_registry() -> EmbedderRegistry:
    """Get global embedder registry singleton."""
    global _embedder_registry
    if _embedder_registry is None:
        _embedder_registry = EmbedderRegistry()
    return _embedder_registry


def get_reranker_registry() -> RerankerRegistry:
    """Get global reranker registry singleton."""
    global _reranker_registry
    if _reranker_registry is None:
        _reranker_registry = RerankerRegistry()
    return _reranker_registry


def get_component_validator() -> ComponentValidator:
    """Get component validator with global registries."""
    return ComponentValidator(
        chunker_registry=get_chunker_registry(),
        embedder_registry=get_embedder_registry(),
        reranker_registry=get_reranker_registry(),
    )
