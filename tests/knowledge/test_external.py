"""Tests for the external knowledge integration module (Phase 8)."""

from __future__ import annotations

import json
from unittest.mock import patch

from umrs.knowledge.external import (
    ArxivSource,
    CrossRefSource,
    ExternalKnowledgeSearcher,
    ExternalReference,
    ExternalSearchResult,
    ExternalSourceType,
    SearchStatus,
    SemanticScholarSource,
    search_arxiv,
    search_external,
)

# ── Enum Tests ─────────────────────────────────────────────────────────────


def test_external_source_type_enum():
    """Test ExternalSourceType enum values."""
    assert ExternalSourceType.ARXIV == "arxiv"
    assert ExternalSourceType.SEMANTIC_SCHOLAR == "semantic_scholar"
    assert ExternalSourceType.CROSSREF == "crossref"
    assert ExternalSourceType.PUBMED == "pubmed"
    assert ExternalSourceType.GOOGLE_SCHOLAR == "google_scholar"
    assert ExternalSourceType.WEB == "web"
    assert ExternalSourceType.GITHUB == "github"


def test_search_status_enum():
    """Test SearchStatus enum values."""
    assert SearchStatus.SUCCESS == "success"
    assert SearchStatus.NO_RESULTS == "no_results"
    assert SearchStatus.ERROR == "error"
    assert SearchStatus.RATE_LIMITED == "rate_limited"
    assert SearchStatus.TIMEOUT == "timeout"


# ── Model Tests ────────────────────────────────────────────────────────────


def test_external_reference_defaults():
    """Test ExternalReference defaults."""
    ref = ExternalReference(source_type=ExternalSourceType.ARXIV)
    assert ref.source_type == ExternalSourceType.ARXIV
    for name, expected in (
        ("title", ""),
        ("authors", []),
        ("abstract", ""),
        ("doi", ""),
        ("arxiv_id", ""),
        ("pubmed_id", ""),
        ("url", ""),
        ("published_date", ""),
        ("keywords", []),
        ("citations", 0),
    ):
        assert getattr(ref, name) == expected, name


def test_external_reference_to_dict():
    """Test ExternalReference to_dict."""
    ref = ExternalReference(
        source_type=ExternalSourceType.ARXIV,
        title="Test Paper",
        authors=["Author A", "Author B"],
        abstract="Test abstract",
        arxiv_id="2401.12345",
        url="http://arxiv.org/abs/2401.12345",
    )

    data = ref.to_dict()
    assert data["source_type"] == "arxiv"
    assert data["title"] == "Test Paper"
    assert data["authors"] == ["Author A", "Author B"]
    assert data["abstract"] == "Test abstract"
    assert data["arxiv_id"] == "2401.12345"
    assert data["url"] == "http://arxiv.org/abs/2401.12345"


def test_external_search_result_defaults():
    """Test ExternalSearchResult defaults."""
    result = ExternalSearchResult(
        query="test",
        source_type=ExternalSourceType.ARXIV,
    )

    assert result.query == "test"
    assert result.source_type == ExternalSourceType.ARXIV
    assert result.status == SearchStatus.SUCCESS
    assert result.references == []
    assert result.total_results == 0
    assert result.error_message == ""


def test_external_search_result_to_dict():
    """Test ExternalSearchResult to_dict."""
    result = ExternalSearchResult(
        query="transformer",
        source_type=ExternalSourceType.SEMANTIC_SCHOLAR,
        status=SearchStatus.SUCCESS,
        total_results=2,
    )

    data = result.to_dict()
    assert data["query"] == "transformer"
    assert data["source_type"] == "semantic_scholar"
    assert data["status"] == "success"
    assert data["total_results"] == 2


# ── ArxivSource Tests ──────────────────────────────────────────────────────


def test_arxiv_source_init():
    """Test ArxivSource initialization."""
    source = ArxivSource()
    assert source.BASE_URL == "http://export.arxiv.org/api/query"


@patch.object(ArxivSource, "_fetch_url")
def test_arxiv_source_search_success(mock_fetch):
    """Test successful arXiv search."""
    mock_fetch.return_value = """
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/2401.12345v1</id>
            <title>Attention Is All You Need</title>
            <author><name>Vaswani, A.</name></author>
            <author><name>Shazeer, N.</name></author>
            <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</summary>
            <published>2017-06-12T17:00:00Z</published>
        </entry>
    </feed>
    """

    source = ArxivSource()
    result = source.search("attention mechanism", max_results=5)

    assert result.status == SearchStatus.SUCCESS
    assert result.total_results == 1
    assert len(result.references) == 1
    assert result.references[0].title == "Attention Is All You Need"
    assert result.references[0].arxiv_id == "2401.12345v1"
    assert "Vaswani, A." in result.references[0].authors


@patch.object(ArxivSource, "_fetch_url")
def test_arxiv_source_search_no_results(mock_fetch):
    """Test arXiv search with no results."""
    mock_fetch.return_value = """
    <feed xmlns="http://www.w3.org/2005/Atom">
    </feed>
    """

    source = ArxivSource()
    result = source.search("nonexistent topic", max_results=5)

    assert result.status == SearchStatus.NO_RESULTS
    assert result.total_results == 0


@patch.object(ArxivSource, "_fetch_url")
def test_arxiv_source_search_error(mock_fetch):
    """Test arXiv search with error."""
    mock_fetch.return_value = None

    source = ArxivSource()
    result = source.search("test", max_results=5)

    assert result.status == SearchStatus.ERROR
    assert "Failed to fetch" in result.error_message


# ── SemanticScholarSource Tests ─────────────────────────────────────────────


def test_semantic_scholar_source_init():
    """Test SemanticScholarSource initialization."""
    source = SemanticScholarSource()
    assert source.BASE_URL == "https://api.semanticscholar.org/graph/v1/paper/search"


@patch.object(SemanticScholarSource, "_fetch_url")
def test_semantic_scholar_search_success(mock_fetch):
    """Test successful Semantic Scholar search."""
    mock_fetch.return_value = json.dumps(
        {
            "data": [
                {
                    "title": "Attention Is All You Need",
                    "authors": [{"name": "Vaswani, A."}],
                    "abstract": "The dominant sequence transduction models...",
                    "year": 2017,
                    "citationCount": 100000,
                    "externalIds": {
                        "DOI": "10.48550/arXiv.1706.03762",
                        "ArXiv": "1706.03762",
                    },
                }
            ]
        }
    )

    source = SemanticScholarSource()
    result = source.search("attention mechanism", max_results=5)

    assert result.status == SearchStatus.SUCCESS
    assert result.total_results == 1
    assert len(result.references) == 1
    assert result.references[0].title == "Attention Is All You Need"
    assert result.references[0].citations == 100000
    assert result.references[0].arxiv_id == "1706.03762"
    assert result.references[0].doi == "10.48550/arXiv.1706.03762"


@patch.object(SemanticScholarSource, "_fetch_url")
def test_semantic_scholar_search_error(mock_fetch):
    """Test Semantic Scholar search with error."""
    mock_fetch.side_effect = Exception("Connection error")

    source = SemanticScholarSource()
    result = source.search("test", max_results=5)

    assert result.status == SearchStatus.ERROR
    assert "Connection error" in result.error_message


# ── CrossRefSource Tests ───────────────────────────────────────────────────


def test_crossref_source_init():
    """Test CrossRefSource initialization."""
    source = CrossRefSource()
    assert source.BASE_URL == "https://api.crossref.org/works"


@patch.object(CrossRefSource, "_fetch_url")
def test_crossref_search_success(mock_fetch):
    """Test successful CrossRef search."""
    mock_fetch.return_value = json.dumps(
        {
            "message": {
                "items": [
                    {
                        "DOI": "10.48550/arXiv.1706.03762",
                        "title": ["Attention Is All You Need"],
                        "author": [
                            {"given": "Ashish", "family": "Vaswani"},
                        ],
                        "abstract": "The dominant sequence transduction models...",
                        "published": {
                            "date-parts": [[2017, 6, 12]],
                        },
                    }
                ]
            }
        }
    )

    source = CrossRefSource()
    result = source.search("attention mechanism", max_results=5)

    assert result.status == SearchStatus.SUCCESS
    assert result.total_results == 1
    assert len(result.references) == 1
    assert result.references[0].title == "Attention Is All You Need"
    assert result.references[0].doi == "10.48550/arXiv.1706.03762"
    assert "Ashish" in result.references[0].authors[0]


@patch.object(CrossRefSource, "_fetch_url")
def test_crossref_search_no_results(mock_fetch):
    """Test CrossRef search with no results."""
    mock_fetch.return_value = json.dumps(
        {
            "message": {
                "items": [],
            },
        }
    )

    source = CrossRefSource()
    result = source.search("nonexistent topic", max_results=5)

    assert result.status == SearchStatus.NO_RESULTS
    assert result.total_results == 0


# ── ExternalKnowledgeSearcher Tests ────────────────────────────────────────


def test_external_knowledge_searcher_init():
    """Test ExternalKnowledgeSearcher initialization."""
    searcher = ExternalKnowledgeSearcher()
    assert ExternalSourceType.ARXIV in searcher._sources
    assert ExternalSourceType.SEMANTIC_SCHOLAR in searcher._sources
    assert ExternalSourceType.CROSSREF in searcher._sources


def test_external_knowledge_searcher_search_unavailable():
    """Test searching an unavailable source."""
    searcher = ExternalKnowledgeSearcher()
    result = searcher.search("test", ExternalSourceType.PUBMED)

    assert result.status == SearchStatus.ERROR
    assert "not available" in result.error_message


@patch.object(ArxivSource, "_fetch_url")
def test_external_knowledge_searcher_search_arxiv(mock_fetch):
    """Test searching arXiv via the searcher."""
    mock_fetch.return_value = """
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/2401.12345v1</id>
            <title>Test Paper</title>
            <author><name>Test Author</name></author>
            <summary>Test abstract</summary>
            <published>2024-01-15T00:00:00Z</published>
        </entry>
    </feed>
    """

    searcher = ExternalKnowledgeSearcher()
    result = searcher.search("test", ExternalSourceType.ARXIV)

    assert result.status == SearchStatus.SUCCESS
    assert result.total_results == 1
    assert result.references[0].title == "Test Paper"


# ── Convenience Function Tests ─────────────────────────────────────────────


@patch.object(ArxivSource, "_fetch_url")
def test_search_arxiv_convenience(mock_fetch):
    """Test search_arxiv convenience function."""
    mock_fetch.return_value = """
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/2401.12345v1</id>
            <title>Test Paper</title>
            <author><name>Test Author</name></author>
            <summary>Test abstract</summary>
        </entry>
    </feed>
    """

    result = search_arxiv("test", max_results=5)

    assert result.source_type == ExternalSourceType.ARXIV
    assert result.status == SearchStatus.SUCCESS
    assert result.total_results == 1


def test_search_external_convenience():
    """Test search_external convenience function."""
    # D5: CiNii + J-STAGE joined the default source pack; WEKO3 stays opt-in
    # because it needs a per-university base_url.
    from umrs.knowledge.external import CiNiiSource, JStageSource

    with (
        patch.object(ArxivSource, "_fetch_url", return_value=None),
        patch.object(SemanticScholarSource, "_fetch_url", return_value=None),
        patch.object(CrossRefSource, "_fetch_url", return_value=None),
        patch.object(CiNiiSource, "_fetch_url", return_value=None),
        patch.object(JStageSource, "_fetch_url", return_value=None),
    ):
        results = search_external("test", max_results_per_source=2)

    assert len(results) == 5  # arXiv, Semantic Scholar, CrossRef, CiNii, J-STAGE
    assert all(r.status == SearchStatus.ERROR for r in results)
