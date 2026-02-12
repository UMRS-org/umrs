"""
Tests for RAG Pipeline (U-303 to U-306).

Tests:
- BM25 scoring
- Hybrid search (RRF fusion)
- Cross-encoder reranking
- Context assembly
- RAG pipeline integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    content: str
    score: float
    metadata: dict


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        MockSearchResult(
            content="Machine learning is a subset of artificial intelligence.",
            score=0.95,
            metadata={"source": "doc1.txt", "chunk_id": 1},
        ),
        MockSearchResult(
            content="Deep learning uses neural networks with many layers.",
            score=0.88,
            metadata={"source": "doc2.txt", "chunk_id": 2},
        ),
        MockSearchResult(
            content="Natural language processing enables computers to understand text.",
            score=0.75,
            metadata={"source": "doc3.txt", "chunk_id": 3},
        ),
    ]


# =============================================================================
# BM25 Tests
# =============================================================================


class TestBM25:
    """Tests for BM25 keyword scoring."""

    def test_bm25_initialization(self, sample_documents):
        """Test BM25 index creation."""
        from umrs.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        assert bm25.corpus is not None
        assert len(bm25.corpus) == 5
        assert bm25.doc_lens is not None
        assert bm25.avgdl > 0

    def test_bm25_tokenize(self):
        """Test BM25 tokenization."""
        from umrs.rag_pipeline import BM25

        bm25 = BM25()
        tokens = bm25.tokenize("Machine learning is great!")

        assert "machine" in tokens
        assert "learning" in tokens
        assert "great" in tokens
        # Stopwords should be removed
        assert "is" not in tokens

    def test_bm25_score(self, sample_documents):
        """Test BM25 scoring."""
        from umrs.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        scores = bm25.score("machine learning")

        assert len(scores) == 5
        # First document mentions "machine learning" directly
        assert scores[0] > scores[3]  # Higher than computer vision doc

    def test_bm25_search(self, sample_documents):
        """Test BM25 search with top-k."""
        from umrs.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        results = bm25.search("neural networks", top_k=2)

        assert len(results) == 2
        # Results should be (index, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Scores should be sorted descending
        assert results[0][1] >= results[1][1]

    def test_bm25_empty_query(self, sample_documents):
        """Test BM25 with empty query."""
        from umrs.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        scores = bm25.score("")

        assert len(scores) == 5
        assert all(s == 0.0 for s in scores)


# =============================================================================
# HybridSearcher Tests
# =============================================================================


class TestHybridSearcher:
    """Tests for hybrid search with RRF fusion."""

    @pytest.fixture
    def mock_vector_search(self):
        """Mock vector search function."""
        async def search(query: str, top_k: int):
            return [
                MockSearchResult("Doc A", 0.95, {"source": "a.txt"}),
                MockSearchResult("Doc B", 0.85, {"source": "b.txt"}),
                MockSearchResult("Doc C", 0.75, {"source": "c.txt"}),
            ][:top_k]
        return search

    def test_rrf_score_calculation(self):
        """Test Reciprocal Rank Fusion score calculation."""
        from umrs.rag_pipeline import HybridSearcher

        searcher = HybridSearcher(
            vector_search_fn=AsyncMock(),
            documents=[],
            vector_weight=0.6,
            bm25_weight=0.4,
        )

        # RRF formula: 1 / (k + rank)
        # Default k=60
        score = searcher._rrf_score(rank=1, k=60)
        expected = 1.0 / (60 + 1)

        assert abs(score - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_hybrid_search(self, sample_documents, mock_vector_search):
        """Test hybrid search combines vector and BM25 results."""
        from umrs.rag_pipeline import HybridSearcher

        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            documents=sample_documents,
        )

        results = await searcher.search("machine learning", top_k=3)

        assert len(results) <= 3
        # Results should have fused scores
        assert all(hasattr(r, "score") for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_deduplication(self, sample_documents, mock_vector_search):
        """Test that hybrid search deduplicates results."""
        from umrs.rag_pipeline import HybridSearcher

        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            documents=sample_documents,
        )

        results = await searcher.search("neural networks", top_k=5)

        # Check no duplicate content
        contents = [r.content for r in results]
        assert len(contents) == len(set(contents))


# =============================================================================
# CrossEncoderReranker Tests
# =============================================================================


class TestCrossEncoderReranker:
    """Tests for cross-encoder reranking."""

    def test_reranker_initialization(self):
        """Test reranker initialization (lazy loading)."""
        from umrs.rag_pipeline import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Model should not be loaded until first use
        assert reranker._model is None

    def test_reranker_with_mock_model(self, sample_search_results):
        """Test reranking with mocked model."""
        from umrs.rag_pipeline import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8]
        reranker._model = mock_model

        reranked = reranker.rerank(
            query="What is machine learning?",
            results=sample_search_results,
            top_k=2,
        )

        assert len(reranked) == 2
        # Should be sorted by new scores
        assert reranked[0].score >= reranked[1].score

    def test_reranker_empty_results(self):
        """Test reranking with empty results."""
        from umrs.rag_pipeline import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        reranked = reranker.rerank(
            query="test query",
            results=[],
            top_k=5,
        )

        assert len(reranked) == 0


# =============================================================================
# ContextAssembler Tests
# =============================================================================


class TestContextAssembler:
    """Tests for context assembly."""

    def test_default_template(self, sample_search_results):
        """Test context assembly with default template."""
        from umrs.rag_pipeline import ContextAssembler

        assembler = ContextAssembler()

        context = assembler.assemble(
            query="What is machine learning?",
            results=sample_search_results,
        )

        assert "What is machine learning?" in context
        assert "Machine learning is a subset" in context
        assert "[1]" in context  # Source reference

    def test_custom_template(self, sample_search_results):
        """Test context assembly with custom template."""
        from umrs.rag_pipeline import ContextAssembler

        template = "Query: {query}\n\nContext:\n{context}"
        assembler = ContextAssembler(template=template)

        context = assembler.assemble(
            query="test query",
            results=sample_search_results,
        )

        assert "Query: test query" in context
        assert "Context:" in context

    def test_token_budget(self, sample_search_results):
        """Test token budget enforcement."""
        from umrs.rag_pipeline import ContextAssembler

        # Very small budget
        assembler = ContextAssembler(max_tokens=50)

        context = assembler.assemble(
            query="test",
            results=sample_search_results,
        )

        # Should truncate to fit budget
        token_count = assembler.count_tokens(context)
        assert token_count <= 50 + 10  # Some margin for template

    def test_count_tokens(self):
        """Test token counting."""
        from umrs.rag_pipeline import ContextAssembler

        assembler = ContextAssembler()

        # Simple approximation: ~4 chars per token
        text = "This is a test sentence with some words."
        count = assembler.count_tokens(text)

        assert count > 0
        assert count < len(text)  # Should be less than char count

    def test_get_sources(self, sample_search_results):
        """Test source extraction."""
        from umrs.rag_pipeline import ContextAssembler

        assembler = ContextAssembler()

        sources = assembler.get_sources(sample_search_results)

        assert "doc1.txt" in sources
        assert "doc2.txt" in sources
        assert len(sources) == 3


# =============================================================================
# RAGPipeline Tests
# =============================================================================


class TestRAGPipeline:
    """Tests for complete RAG pipeline."""

    @pytest.fixture
    def mock_gateway_client(self):
        """Mock Gateway client."""
        client = AsyncMock()
        client.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "Test response"}}]
        })
        return client

    @pytest.fixture
    def mock_vector_search(self, sample_search_results):
        """Mock vector search."""
        async def search(query: str, top_k: int):
            return sample_search_results[:top_k]
        return search

    @pytest.mark.asyncio
    async def test_pipeline_query(
        self,
        sample_documents,
        mock_gateway_client,
        mock_vector_search,
    ):
        """Test full pipeline query."""
        from umrs.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            gateway_client=mock_gateway_client,
            vector_search_fn=mock_vector_search,
            documents=sample_documents,
        )

        response = await pipeline.query("What is machine learning?")

        assert response.answer is not None
        assert response.sources is not None
        assert response.context is not None
        mock_gateway_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_with_reranking(
        self,
        sample_documents,
        mock_gateway_client,
        mock_vector_search,
    ):
        """Test pipeline with reranking enabled."""
        from umrs.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            gateway_client=mock_gateway_client,
            vector_search_fn=mock_vector_search,
            documents=sample_documents,
            use_reranking=True,
        )

        # Mock the reranker
        pipeline.reranker._model = MagicMock()
        pipeline.reranker._model.predict.return_value = [0.9, 0.8, 0.7]

        response = await pipeline.query("test query")

        assert response.answer is not None

    @pytest.mark.asyncio
    async def test_pipeline_search_only(
        self,
        sample_documents,
        mock_gateway_client,
        mock_vector_search,
    ):
        """Test pipeline search without LLM generation."""
        from umrs.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            gateway_client=mock_gateway_client,
            vector_search_fn=mock_vector_search,
            documents=sample_documents,
        )

        results = await pipeline.search("neural networks", top_k=3)

        assert len(results) <= 3
        # LLM should not be called
        mock_gateway_client.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_custom_system_prompt(
        self,
        sample_documents,
        mock_gateway_client,
        mock_vector_search,
    ):
        """Test pipeline with custom system prompt."""
        from umrs.rag_pipeline import RAGPipeline

        custom_prompt = "You are a helpful science assistant."
        pipeline = RAGPipeline(
            gateway_client=mock_gateway_client,
            vector_search_fn=mock_vector_search,
            documents=sample_documents,
            system_prompt=custom_prompt,
        )

        await pipeline.query("test")

        call_args = mock_gateway_client.chat.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])

        # Check system prompt is included
        system_messages = [m for m in messages if m.get("role") == "system"]
        assert len(system_messages) > 0
        assert custom_prompt in system_messages[0]["content"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_without_model(self, sample_documents):
        """Test end-to-end flow without actual model loading."""
        from umrs.rag_pipeline import (
            BM25,
            HybridSearcher,
            ContextAssembler,
        )

        # Setup BM25
        bm25 = BM25()
        bm25.fit(sample_documents)

        # Mock vector search
        async def mock_vector_search(query: str, top_k: int):
            # Return docs based on simple keyword matching
            results = []
            for i, doc in enumerate(sample_documents):
                if any(word in doc.lower() for word in query.lower().split()):
                    results.append(MockSearchResult(
                        content=doc,
                        score=0.8 - i * 0.1,
                        metadata={"source": f"doc{i}.txt"},
                    ))
            return results[:top_k]

        # Setup hybrid searcher
        searcher = HybridSearcher(
            vector_search_fn=mock_vector_search,
            documents=sample_documents,
        )

        # Search
        results = await searcher.search("machine learning neural", top_k=3)

        assert len(results) > 0

        # Assemble context
        assembler = ContextAssembler(max_tokens=500)
        context = assembler.assemble(
            query="What is machine learning?",
            results=results,
        )

        assert "machine learning" in context.lower() or len(results) == 0
        sources = assembler.get_sources(results)
        assert isinstance(sources, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
