"""Knowledge ingestion package — unified ingestion pipeline for scientific artifacts.

This package implements the unified knowledge ingestion pipeline that
discovers, parses, extracts metadata, summarizes, generates embeddings,
registers into UMRS, updates catalog, and indexes for RAG.

The pipeline supports:
  - PDF, Markdown, HTML, Jupyter Notebook, Python, C++, CUDA, JSON, CSV, YAML,
    Images, Slides

Requirements:
  - Incremental indexing
  - Parallel processing
  - Progress reporting
  - Resume capability
  - Duplicate detection
"""

from __future__ import annotations

# Parsers are always available (stdlib-only, no heavy deps)
from .parse_code import CodeParseResult, parse_code
from .parse_notebook import NotebookParseResult, parse_notebook

__all__ = [
    # Parsers (always available)
    "NotebookParseResult",
    "parse_notebook",
    "CodeParseResult",
    "parse_code",
    # Ingestion pipeline (lazy import — requires botocore)
    "KnowledgeIngester",
    "IngestionResult",
    "IngestionError",
    "ingest_object",
    # UMRS API adapters (Phase 3)
    "UMRSScientificMemoryAdapter",
    "UMRSKnowledgeModelAdapter",
    "UMRSKnowledgeGraphAdapter",
    "UMRSScientificExperimentRegistryAdapter",
    "UMRSEvidenceAdapter",
    "UMRSSearchAdapter",
    "UMRSEmbeddingsAdapter",
]


def __getattr__(name: str):
    """Lazy import for ingestion pipeline (avoids botocore dependency for parsers)."""
    if name in {"KnowledgeIngester", "IngestionResult", "IngestionError", "ingest_object"}:
        from .ingest import (
            IngestionError,
            IngestionResult,
            KnowledgeIngester,
            ingest_object,
        )
        return locals()[name]
    if name in {
        "UMRSScientificMemoryAdapter",
        "UMRSKnowledgeModelAdapter",
        "UMRSKnowledgeGraphAdapter",
        "UMRSScientificExperimentRegistryAdapter",
        "UMRSEvidenceAdapter",
        "UMRSSearchAdapter",
        "UMRSEmbeddingsAdapter",
    }:
        from .adapters import (
            UMRSEmbeddingsAdapter,
            UMRSEvidenceAdapter,
            UMRSKnowledgeGraphAdapter,
            UMRSKnowledgeModelAdapter,
            UMRSScientificExperimentRegistryAdapter,
            UMRSScientificMemoryAdapter,
            UMRSSearchAdapter,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

