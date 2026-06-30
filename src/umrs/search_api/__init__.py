"""Search API — unified search across catalog, knowledge graph, embeddings, and RAG.

Phase 1, Step 2.6 of the RC1 migration plan.

UMRS owns the unified search interface. R4 consumes this API.

This module provides a unified search interface that aggregates results
from multiple UMRS subsystems:

  - search_catalog() — search the artifact catalog (grep-based)
  - search_knowledge_graph() — search the knowledge graph (typed relationships)
  - search_embeddings() — search embeddings (vector similarity)
  - search_rag() — search RAG indexes (hybrid BM25 + vector)
  - search() — unified search across all subsystems

Each search returns typed results with source attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

__all__ = [
    "SearchScope",
    "SearchDepth",
    "SearchResultType",
    "SearchResult",
    "SearchResults",
    "CatalogSearchable",
    "KnowledgeGraphSearchable",
    "EmbeddingsSearchable",
    "RAGSearchable",
    "UnifiedSearch",
]


# ── Search Enums ─────────────────────────────────────────────────────────


class SearchScope(str, Enum):
    """Scope of search."""

    CATALOG = "catalog"               # Only catalog search
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Only knowledge graph search
    EMBEDDINGS = "embeddings"         # Only embeddings search
    RAG = "rag"                       # Only RAG search
    ALL = "all"                       # All subsystems


class SearchDepth(str, Enum):
    """Depth of search."""

    SHALLOW = "shallow"               # Quick, single-hop
    MODERATE = "moderate"             # Multi-hop with refinement
    DEEP = "deep"                     # Exhaustive multi-hop


class SearchResultType(str, Enum):
    """Type of search result."""

    CATALOG_HIT = "catalog_hit"
    KNOWLEDGE_NODE = "knowledge_node"
    EMBEDDING_MATCH = "embedding_match"
    RAG_RESULT = "rag_result"


# ── Search Result Models ─────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search result from any subsystem.

    Attributes:
        result_id: Unique identifier for the result
        result_type: Type of result (catalog hit, knowledge node, etc.)
        title: Title of the result
        content: Content of the result (may be truncated)
        score: Relevance score (0.0–1.0)
        source_id: ID of the source object
        source_type: Type of the source object
        metadata: Additional metadata
    """

    result_id: str = ""
    result_type: SearchResultType = SearchResultType.CATALOG_HIT
    title: str = ""
    content: str = ""
    score: float = 0.0
    source_id: str = ""
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.result_id:
            self.result_id = f"res-{uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "result_type": self.result_type.value,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }


@dataclass
class SearchResults:
    """Results from a unified search across multiple subsystems.

    Attributes:
        query: The search query
        scope: The search scope used
        depth: The search depth used
        results: List of search results
        total_results: Total number of results
        searched_at: When the search was performed
        duration_seconds: How long the search took
    """

    query: str = ""
    scope: SearchScope = SearchScope.ALL
    depth: SearchDepth = SearchDepth.MODERATE
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    searched_at: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "scope": self.scope.value,
            "depth": self.depth.value,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "searched_at": self.searched_at,
            "duration_seconds": self.duration_seconds,
        }


# ── Searchable Protocols ─────────────────────────────────────────────────


class CatalogSearchable(Protocol):
    """Protocol for catalog search."""

    def search_catalog(self, query: str, *, project_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        ...


class KnowledgeGraphSearchable(Protocol):
    """Protocol for knowledge graph search."""

    def search_knowledge_graph(self, query: str, *, project_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        ...


class EmbeddingsSearchable(Protocol):
    """Protocol for embeddings search."""

    def search_embeddings(self, query: str, *, project_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        ...


class RAGSearchable(Protocol):
    """Protocol for RAG search."""

    def search_rag(self, query: str, *, project_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        ...


# ── Unified Search ───────────────────────────────────────────────────────


class UnifiedSearch:
    """Unified search across catalog, knowledge graph, embeddings, and RAG.

    Usage:
        search = UnifiedSearch(
            catalog=catalog,
            knowledge_graph=graph,
            embeddings=embeddings,
            rag=rag,
        )
        results = search.search("machine learning", scope=SearchScope.ALL)
    """

    def __init__(
        self,
        *,
        catalog: CatalogSearchable | None = None,
        knowledge_graph: KnowledgeGraphSearchable | None = None,
        embeddings: EmbeddingsSearchable | None = None,
        rag: RAGSearchable | None = None,
    ) -> None:
        self._catalog = catalog
        self._knowledge_graph = knowledge_graph
        self._embeddings = embeddings
        self._rag = rag

    def search(
        self,
        query: str,
        *,
        scope: SearchScope = SearchScope.ALL,
        depth: SearchDepth = SearchDepth.MODERATE,
        project_id: str | None = None,
        limit: int = 20,
    ) -> SearchResults:
        """Perform a unified search across configured subsystems.

        Args:
            query: The search query
            scope: Which subsystems to search
            depth: Search depth
            project_id: Filter by project ID
            limit: Maximum number of results per subsystem

        Returns:
            SearchResults with aggregated results from all searched subsystems
        """
        start = datetime.now(timezone.utc)
        all_results: list[SearchResult] = []

        # Determine which subsystems to search
        scopes = self._get_scopes(scope)

        # Search each subsystem
        for s in scopes:
            if s == SearchScope.CATALOG and self._catalog:
                results = self._catalog.search_catalog(
                    query, project_id=project_id, limit=limit
                )
                all_results.extend(
                    self._convert_results(results, SearchResultType.CATALOG_HIT)
                )
            elif s == SearchScope.KNOWLEDGE_GRAPH and self._knowledge_graph:
                results = self._knowledge_graph.search_knowledge_graph(
                    query, project_id=project_id, limit=limit
                )
                all_results.extend(
                    self._convert_results(results, SearchResultType.KNOWLEDGE_NODE)
                )
            elif s == SearchScope.EMBEDDINGS and self._embeddings:
                results = self._embeddings.search_embeddings(
                    query, project_id=project_id, limit=limit
                )
                all_results.extend(
                    self._convert_results(results, SearchResultType.EMBEDDING_MATCH)
                )
            elif s == SearchScope.RAG and self._rag:
                results = self._rag.search_rag(
                    query, project_id=project_id, limit=limit
                )
                all_results.extend(
                    self._convert_results(results, SearchResultType.RAG_RESULT)
                )

        # Sort by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Limit total results
        all_results = all_results[:limit]

        duration = (datetime.now(timezone.utc) - start).total_seconds()

        return SearchResults(
            query=query,
            scope=scope,
            depth=depth,
            results=all_results,
            total_results=len(all_results),
            searched_at=start.isoformat(),
            duration_seconds=duration,
        )

    def search_catalog(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the artifact catalog."""
        if not self._catalog:
            return []
        return self._catalog.search_catalog(
            query, project_id=project_id, limit=limit
        )

    def search_knowledge_graph(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the knowledge graph."""
        if not self._knowledge_graph:
            return []
        return self._knowledge_graph.search_knowledge_graph(
            query, project_id=project_id, limit=limit
        )

    def search_embeddings(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the embeddings."""
        if not self._embeddings:
            return []
        return self._embeddings.search_embeddings(
            query, project_id=project_id, limit=limit
        )

    def search_rag(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the RAG indexes."""
        if not self._rag:
            return []
        return self._rag.search_rag(
            query, project_id=project_id, limit=limit
        )

    def _get_scopes(self, scope: SearchScope) -> list[SearchScope]:
        """Get list of scopes to search based on the requested scope."""
        if scope == SearchScope.ALL:
            return [
                SearchScope.CATALOG,
                SearchScope.KNOWLEDGE_GRAPH,
                SearchScope.EMBEDDINGS,
                SearchScope.RAG,
            ]
        return [scope]

    def _convert_results(
        self,
        results: list[dict[str, Any]],
        result_type: SearchResultType,
    ) -> list[SearchResult]:
        """Convert raw search results to SearchResult objects."""
        converted: list[SearchResult] = []
        for result in results:
            converted.append(
                SearchResult(
                    result_type=result_type,
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    source_id=result.get("source_id", ""),
                    source_type=result.get("source_type", ""),
                    metadata=result.get("metadata", {}),
                )
            )
        return converted
