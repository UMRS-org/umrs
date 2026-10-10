"""Tests for the search_api subsystem (Phase 1, Step 2.6)."""

from __future__ import annotations

import pytest
from umrs.search_api import (
    SearchDepth,
    SearchResult,
    SearchResults,
    SearchResultType,
    SearchScope,
    UnifiedSearch,
)

# ── Enum Tests ────────────────────────────────────────────────────────────


def test_search_scope_enum():
    """Test SearchScope enum values."""
    assert SearchScope.CATALOG == "catalog"
    assert SearchScope.KNOWLEDGE_GRAPH == "knowledge_graph"
    assert SearchScope.EMBEDDINGS == "embeddings"
    assert SearchScope.RAG == "rag"
    assert SearchScope.ALL == "all"


def test_search_depth_enum():
    """Test SearchDepth enum values."""
    assert SearchDepth.SHALLOW == "shallow"
    assert SearchDepth.MODERATE == "moderate"
    assert SearchDepth.DEEP == "deep"


def test_search_result_type_enum():
    """Test SearchResultType enum values."""
    assert SearchResultType.CATALOG_HIT == "catalog_hit"
    assert SearchResultType.KNOWLEDGE_NODE == "knowledge_node"
    assert SearchResultType.EMBEDDING_MATCH == "embedding_match"
    assert SearchResultType.RAG_RESULT == "rag_result"


# ── SearchResult Tests ────────────────────────────────────────────────────


def test_search_result_defaults():
    """Test SearchResult defaults."""
    result = SearchResult()

    assert result.result_id != ""  # Auto-generated
    assert result.result_type == SearchResultType.CATALOG_HIT
    assert result.title == ""
    assert result.content == ""
    assert result.score == 0.0
    assert result.source_id == ""
    assert result.source_type == ""
    assert result.metadata == {}


def test_search_result_to_dict():
    """Test SearchResult to_dict."""
    result = SearchResult(
        result_type=SearchResultType.RAG_RESULT,
        title="Machine Learning Paper",
        content="This paper presents...",
        score=0.95,
        source_id="obj-001",
        source_type="paper",
    )

    data = result.to_dict()

    assert data["result_type"] == "rag_result"
    assert data["title"] == "Machine Learning Paper"
    assert data["content"] == "This paper presents..."
    assert data["score"] == 0.95
    assert data["source_id"] == "obj-001"
    assert data["source_type"] == "paper"


# ── SearchResults Tests ──────────────────────────────────────────────────


def test_search_results_defaults():
    """Test SearchResults defaults."""
    results = SearchResults()

    assert results.query == ""
    assert results.scope == SearchScope.ALL
    assert results.depth == SearchDepth.MODERATE
    assert results.results == []
    assert results.total_results == 0
    assert results.searched_at == ""
    assert results.duration_seconds == 0.0


def test_search_results_to_dict():
    """Test SearchResults to_dict."""
    results = SearchResults(
        query="machine learning",
        scope=SearchScope.ALL,
        depth=SearchDepth.MODERATE,
        results=[
            SearchResult(
                title="Result 1",
                score=0.9,
            ),
        ],
        total_results=1,
    )

    data = results.to_dict()

    assert data["query"] == "machine learning"
    assert data["scope"] == "all"
    assert data["depth"] == "moderate"
    assert data["total_results"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Result 1"


# ── UnifiedSearch Tests ──────────────────────────────────────────────────


class MockCatalogSearch:
    """Mock catalog search."""

    def search_catalog(self, query: str, *, project_id=None, limit=20):
        return [
            {
                "title": f"Catalog result for '{query}'",
                "content": "Catalog content",
                "score": 0.8,
                "source_id": "cat-001",
                "source_type": "dataset",
            }
        ]


class MockKnowledgeGraphSearch:
    """Mock knowledge graph search."""

    def search_knowledge_graph(self, query: str, *, project_id=None, limit=20):
        return [
            {
                "title": f"Graph result for '{query}'",
                "content": "Graph content",
                "score": 0.9,
                "source_id": "node-001",
                "source_type": "paper",
            }
        ]


class MockEmbeddingsSearch:
    """Mock embeddings search."""

    def search_embeddings(self, query: str, *, project_id=None, limit=20):
        return [
            {
                "title": f"Embedding result for '{query}'",
                "content": "Embedding content",
                "score": 0.7,
                "source_id": "emb-001",
                "source_type": "paper",
            }
        ]


class MockRAGSearch:
    """Mock RAG search."""

    def search_rag(self, query: str, *, project_id=None, limit=20):
        return [
            {
                "title": f"RAG result for '{query}'",
                "content": "RAG content",
                "score": 0.85,
                "source_id": "rag-001",
                "source_type": "paper",
            }
        ]


def test_unified_search_init():
    """Test UnifiedSearch initialization."""
    search = UnifiedSearch()
    assert search is not None


def test_unified_search_no_subsystems():
    """Test unified search with no subsystems configured."""
    search = UnifiedSearch()
    results = search.search("test query")

    assert results.total_results == 0
    assert len(results.results) == 0


def test_unified_search_catalog_only():
    """Test unified search with only catalog configured."""
    search = UnifiedSearch(catalog=MockCatalogSearch())
    results = search.search("test query", scope=SearchScope.CATALOG)

    assert results.total_results == 1
    assert results.results[0].title == "Catalog result for 'test query'"
    assert results.results[0].result_type == SearchResultType.CATALOG_HIT


def test_unified_search_all_subsystems():
    """Test unified search with all subsystems configured."""
    search = UnifiedSearch(
        catalog=MockCatalogSearch(),
        knowledge_graph=MockKnowledgeGraphSearch(),
        embeddings=MockEmbeddingsSearch(),
        rag=MockRAGSearch(),
    )
    results = search.search("test query", scope=SearchScope.ALL)

    assert results.total_results == 4
    # Results should be sorted by score descending
    assert results.results[0].score >= results.results[1].score
    # Check that all result types are present
    result_types = {r.result_type for r in results.results}
    assert SearchResultType.CATALOG_HIT in result_types
    assert SearchResultType.KNOWLEDGE_NODE in result_types
    assert SearchResultType.EMBEDDING_MATCH in result_types
    assert SearchResultType.RAG_RESULT in result_types


def test_unified_search_limit():
    """Test that limit is respected."""
    search = UnifiedSearch(
        catalog=MockCatalogSearch(),
        knowledge_graph=MockKnowledgeGraphSearch(),
        embeddings=MockEmbeddingsSearch(),
        rag=MockRAGSearch(),
    )
    results = search.search("test query", scope=SearchScope.ALL, limit=2)

    assert len(results.results) == 2


def test_unified_search_individual_methods():
    """Test individual search methods."""
    search = UnifiedSearch(
        catalog=MockCatalogSearch(),
        knowledge_graph=MockKnowledgeGraphSearch(),
        embeddings=MockEmbeddingsSearch(),
        rag=MockRAGSearch(),
    )

    # Test individual search methods
    catalog_results = search.search_catalog("test")
    assert len(catalog_results) == 1

    graph_results = search.search_knowledge_graph("test")
    assert len(graph_results) == 1

    embeddings_results = search.search_embeddings("test")
    assert len(embeddings_results) == 1

    rag_results = search.search_rag("test")
    assert len(rag_results) == 1


def test_unified_search_individual_methods_no_subsystem():
    """Test individual search methods with no subsystem configured."""
    search = UnifiedSearch()

    assert search.search_catalog("test") == []
    assert search.search_knowledge_graph("test") == []
    assert search.search_embeddings("test") == []
    assert search.search_rag("test") == []


def test_unified_search_scope_filtering():
    """Test that scope filtering works correctly."""
    search = UnifiedSearch(
        catalog=MockCatalogSearch(),
        knowledge_graph=MockKnowledgeGraphSearch(),
    )

    # Search only catalog
    results = search.search("test", scope=SearchScope.CATALOG)
    assert results.total_results == 1
    assert results.results[0].result_type == SearchResultType.CATALOG_HIT

    # Search only knowledge graph
    results = search.search("test", scope=SearchScope.KNOWLEDGE_GRAPH)
    assert results.total_results == 1
    assert results.results[0].result_type == SearchResultType.KNOWLEDGE_NODE

    # Search all (both catalog and knowledge graph)
    results = search.search("test", scope=SearchScope.ALL)
    assert results.total_results == 2


def test_unified_search_results_sorted_by_score():
    """``scope=all`` sorts by fused score (ADR-093 C7: weighted RRF, not raw).

    Each lane put its one hit first, so the lane weights decide: graph 0.7
    over catalogue 0.5, normalised by the 1.2 both lanes could attain.
    """
    search = UnifiedSearch(
        catalog=MockCatalogSearch(),  # raw score 0.8
        knowledge_graph=MockKnowledgeGraphSearch(),  # raw score 0.9
    )
    results = search.search("test", scope=SearchScope.ALL)

    assert [r.result_type for r in results.results] == [
        SearchResultType.KNOWLEDGE_NODE,
        SearchResultType.CATALOG_HIT,
    ]
    assert results.results[0].score == pytest.approx(0.7 / 1.2, abs=1e-6)
    assert results.results[1].score == pytest.approx(0.5 / 1.2, abs=1e-6)
    assert results.results[0].metadata["lane_score"] == {"knowledge_graph": 0.9}


def test_unified_search_duration_tracked():
    """Test that search duration is tracked."""
    search = UnifiedSearch(catalog=MockCatalogSearch())
    results = search.search("test", scope=SearchScope.CATALOG)

    assert results.duration_seconds >= 0.0
    assert results.searched_at != ""
