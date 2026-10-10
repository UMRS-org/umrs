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

ADR-093 D-6 (C7, 2026-10-10): ``scope=all`` fuses its lanes by weighted
reciprocal-rank fusion instead of sorting raw scores, which are not comparable
across lanes (a catalogue token count, a cosine, an IDF sum, a matched
fraction). A hit found by several lanes is one result carrying
``metadata.lanes`` and ``metadata.lane_score``; catalogue rows are held to a
quota so they cannot crowd passages out (KGC-03); ``total_results`` counts
every distinct hit before the cut. A single-lane scope keeps its lane's own
order and raw score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

__all__ = [
    "LANE_WEIGHTS",
    "RRF_K",
    "CatalogSearchable",
    "EmbeddingsSearchable",
    "KnowledgeGraphSearchable",
    "RAGSearchable",
    "SearchDepth",
    "SearchResult",
    "SearchResultType",
    "SearchResults",
    "SearchScope",
    "UnifiedSearch",
    "catalog_quota",
    "fuse_lanes",
]


# ── Search Enums ─────────────────────────────────────────────────────────


class SearchScope(str, Enum):
    """Scope of search."""

    CATALOG = "catalog"  # Only catalog search
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Only knowledge graph search
    EMBEDDINGS = "embeddings"  # Only embeddings search
    RAG = "rag"  # Only RAG search
    ALL = "all"  # All subsystems


class SearchDepth(str, Enum):
    """Depth of search."""

    SHALLOW = "shallow"  # Quick, single-hop
    MODERATE = "moderate"  # Multi-hop with refinement
    DEEP = "deep"  # Exhaustive multi-hop


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

    def search_catalog(
        self, query: str, *, project_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]: ...


class KnowledgeGraphSearchable(Protocol):
    """Protocol for knowledge graph search."""

    def search_knowledge_graph(
        self, query: str, *, project_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]: ...


class EmbeddingsSearchable(Protocol):
    """Protocol for embeddings search."""

    def search_embeddings(
        self, query: str, *, project_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]: ...


class RAGSearchable(Protocol):
    """Protocol for RAG search."""

    def search_rag(
        self, query: str, *, project_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]: ...


# ── Lane fusion (ADR-093 D-6, C7) ────────────────────────────────────────

#: Reciprocal-rank fusion constant (Cormack et al.; ADR-085 D-1).
RRF_K = 60

#: Lane weights: passages first, graph nodes next, catalogue rows last.
LANE_WEIGHTS: Mapping[SearchScope, float] = {
    SearchScope.EMBEDDINGS: 1.0,
    SearchScope.RAG: 1.0,
    SearchScope.KNOWLEDGE_GRAPH: 0.7,
    SearchScope.CATALOG: 0.5,
}

# Fusion visits lanes heaviest first, so equal fused scores keep that order.
_FUSION_ORDER: tuple[SearchScope, ...] = (
    SearchScope.EMBEDDINGS,
    SearchScope.RAG,
    SearchScope.KNOWLEDGE_GRAPH,
    SearchScope.CATALOG,
)

_RESULT_TYPES: Mapping[SearchScope, SearchResultType] = {
    SearchScope.CATALOG: SearchResultType.CATALOG_HIT,
    SearchScope.KNOWLEDGE_GRAPH: SearchResultType.KNOWLEDGE_NODE,
    SearchScope.EMBEDDINGS: SearchResultType.EMBEDDING_MATCH,
    SearchScope.RAG: SearchResultType.RAG_RESULT,
}


def catalog_quota(limit: int) -> int:
    """At most this many catalogue rows in a fused page of ``limit`` (KGC-03)."""
    return max(2, limit // 4)


@dataclass
class _Fused:
    """One source found by one or more lanes."""

    result: SearchResult
    contribution: float  # the representative lane's weight / (k + rank)
    fused: float = 0.0
    lane_score: dict[str, float] = field(default_factory=dict)
    lane_rank: dict[str, int] = field(default_factory=dict)

    def add(
        self, lane: SearchScope, rank: int, hit: SearchResult, share: float
    ) -> None:
        self.fused += share
        self.lane_score[lane.value] = hit.score
        self.lane_rank[lane.value] = rank
        if share > self.contribution:  # the strongest lane speaks for the hit
            self.result, self.contribution = hit, share

    def to_result(self, attainable: float) -> SearchResult:
        metadata = {
            **self.result.metadata,
            "lanes": list(self.lane_score),
            "lane_score": dict(self.lane_score),
            "lane_rank": dict(self.lane_rank),
            "rrf": self.fused,
        }
        score = self.fused / attainable if attainable > 0 else 0.0
        return replace(self.result, score=round(score, 6), metadata=metadata)


def _ranked(hits: list[SearchResult]) -> list[SearchResult]:
    """A lane's hits best first, each source once (its best rank)."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for hit in sorted(hits, key=lambda h: h.score, reverse=True):
        key = hit.source_id or hit.result_id  # no id: never merged with another
        if key not in seen:
            seen.add(key)
            out.append(hit)
    return out


def fuse_lanes(
    lanes: Mapping[SearchScope, list[SearchResult]],
    *,
    k: int = RRF_K,
    weights: Mapping[SearchScope, float] = LANE_WEIGHTS,
) -> list[SearchResult]:
    """Weighted RRF over the lanes that ran, deduplicated by ``source_id``.

    ``score`` is the fused value over the most a source could attain (rank 1
    in every lane that ran), so 1.0 means every lane put it first; the raw
    RRF sum is ``metadata.rrf`` and each lane's own score ``metadata.lane_score``.
    """
    entries: dict[str, _Fused] = {}
    for lane in (s for s in _FUSION_ORDER if s in lanes):
        for rank, hit in enumerate(_ranked(lanes[lane]), start=1):
            share = weights[lane] / (k + rank)
            key = hit.source_id or hit.result_id
            entry = entries.setdefault(key, _Fused(result=hit, contribution=share))
            entry.add(lane, rank, hit, share)
    attainable = sum(weights[lane] for lane in lanes) / (k + 1)
    ordered = sorted(entries.values(), key=lambda e: e.fused, reverse=True)
    return [entry.to_result(attainable) for entry in ordered]


def _page(ranked: list[SearchResult], limit: int) -> list[SearchResult]:
    """The first ``limit`` fused results, catalogue rows held to their quota.

    Catalogue rows past the quota go behind every result of the other lanes:
    they can no longer push a passage or a node off the page (KGC-03), and
    they still fill a page the other lanes cannot (a project with files but
    no knowledge nodes keeps its catalogue answers).
    """
    quota = catalog_quota(limit)
    page: list[SearchResult] = []
    held: list[SearchResult] = []
    for result in ranked:
        if result.result_type != SearchResultType.CATALOG_HIT:
            page.append(result)
        elif quota > 0:
            quota -= 1
            page.append(result)
        else:
            held.append(result)
    return (page + held)[:limit]


def _single_lane(lanes: Mapping[SearchScope, list[SearchResult]]) -> list[SearchResult]:
    """One scope asked for: its own order and raw score (thresholds keep working)."""
    hits = [hit for found in lanes.values() for hit in found]
    hits.sort(key=lambda r: r.score, reverse=True)
    return hits


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
            limit: Maximum number of results per subsystem, and of the page

        Returns:
            SearchResults: for ``scope=all`` the lanes fused by weighted RRF
            (catalogue rows held to ``catalog_quota``); for one scope that
            lane's hits by raw score. ``total_results`` counts every distinct
            hit before the page is cut.
        """
        start = datetime.now(timezone.utc)
        lanes = self._run_lanes(query, scope, project_id=project_id, limit=limit)
        if scope == SearchScope.ALL:
            ranked = fuse_lanes(lanes)
            page = _page(ranked, limit)
        else:
            ranked = _single_lane(lanes)
            page = ranked[:limit]
        duration = (datetime.now(timezone.utc) - start).total_seconds()

        return SearchResults(
            query=query,
            scope=scope,
            depth=depth,
            results=page,
            total_results=len(ranked),
            searched_at=start.isoformat(),
            duration_seconds=duration,
        )

    def _lane_search(self, lane: SearchScope) -> Any:
        """The bound search call of a configured lane, or None."""
        backends = {
            SearchScope.CATALOG: (self._catalog, "search_catalog"),
            SearchScope.KNOWLEDGE_GRAPH: (
                self._knowledge_graph,
                "search_knowledge_graph",
            ),
            SearchScope.EMBEDDINGS: (self._embeddings, "search_embeddings"),
            SearchScope.RAG: (self._rag, "search_rag"),
        }
        backend, method = backends[lane]
        return getattr(backend, method) if backend else None

    def _run_lanes(
        self, query: str, scope: SearchScope, *, project_id: str | None, limit: int
    ) -> dict[SearchScope, list[SearchResult]]:
        """Every configured lane in scope -> its converted hits (empty lists kept)."""
        lanes: dict[SearchScope, list[SearchResult]] = {}
        for lane in self._get_scopes(scope):
            search = self._lane_search(lane)
            if search is None:
                continue
            raw = search(query, project_id=project_id, limit=limit)
            lanes[lane] = self._convert_results(raw, _RESULT_TYPES[lane])
        return lanes

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
        return self._catalog.search_catalog(query, project_id=project_id, limit=limit)

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
        return self._rag.search_rag(query, project_id=project_id, limit=limit)

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
