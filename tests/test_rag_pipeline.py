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
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_documents():
    """Sample documents as (doc_id, text) tuples for BM25."""
    return [
        ("doc1", "Machine learning is a subset of artificial intelligence."),
        ("doc2", "Deep learning uses neural networks with many layers."),
        ("doc3", "Natural language processing enables computers to understand text."),
        ("doc4", "Computer vision allows machines to interpret images."),
        ("doc5", "Reinforcement learning trains agents through rewards."),
    ]


@pytest.fixture
def sample_documents_with_embeddings():
    """Sample documents with pre-computed vector similarity scores."""
    return [
        ("doc1", "Machine learning is a subset of artificial intelligence.", 0.92),
        ("doc2", "Deep learning uses neural networks with many layers.", 0.88),
        ("doc3", "Natural language processing enables computers to understand text.", 0.81),
        ("doc4", "Computer vision allows machines to interpret images.", 0.75),
        ("doc5", "Reinforcement learning trains agents through rewards.", 0.70),
    ]


@pytest.fixture
def sample_search_results():
    """Sample SearchResult objects for testing."""
    from umrs.models.rag_pipeline import SearchResult
    return [
        SearchResult(
            document_id="doc1",
            chunk_id="chunk1",
            content="Machine learning is a subset of artificial intelligence.",
            score=0.95,
            metadata={"source": "doc1.txt"},
        ),
        SearchResult(
            document_id="doc2",
            chunk_id="chunk2",
            content="Deep learning uses neural networks with many layers.",
            score=0.88,
            metadata={"source": "doc2.txt"},
        ),
        SearchResult(
            document_id="doc3",
            chunk_id="chunk3",
            content="Natural language processing enables computers to understand text.",
            score=0.75,
            metadata={"source": "doc3.txt"},
        ),
    ]


# =============================================================================
# BM25 Tests
# =============================================================================


class TestBM25:
    """Tests for BM25 keyword scoring."""

    def test_bm25_initialization(self, sample_documents):
        """Test BM25 index creation."""
        from umrs.models.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        assert bm25._documents is not None
        assert len(bm25._documents) == 5
        assert bm25._doc_lengths is not None
        assert bm25._avg_doc_length > 0

    def test_bm25_tokenize(self):
        """Test BM25 tokenization."""
        from umrs.models.rag_pipeline import BM25

        bm25 = BM25()
        # _tokenize is private, but we test it for completeness
        tokens = bm25._tokenize("Machine learning is great!")

        assert "machine" in tokens
        assert "learning" in tokens
        assert "great" in tokens

    def test_bm25_score(self, sample_documents):
        """Test BM25 scoring."""
        from umrs.models.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        results = bm25.score("machine learning", top_k=5)

        assert len(results) == 5
        # Results are (doc_id, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # First doc mentions "machine learning" directly
        doc_ids = [r[0] for r in results]
        assert "doc1" in doc_ids[:2]  # Should be in top results

    def test_bm25_score_top_k(self, sample_documents):
        """Test BM25 scoring with top-k limit."""
        from umrs.models.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        results = bm25.score("neural networks", top_k=2)

        assert len(results) == 2
        # Results should be (doc_id, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Scores should be sorted descending
        assert results[0][1] >= results[1][1]

    def test_bm25_empty_query(self, sample_documents):
        """Test BM25 with empty query."""
        from umrs.models.rag_pipeline import BM25

        bm25 = BM25()
        bm25.fit(sample_documents)

        results = bm25.score("", top_k=5)

        assert len(results) == 5
        # All scores should be 0 for empty query
        assert all(r[1] == 0.0 for r in results)


# =============================================================================
# HybridSearcher Tests
# =============================================================================


class TestHybridSearcher:
    """Tests for hybrid search with RRF fusion."""

    def test_rrf_score_calculation(self):
        """Test Reciprocal Rank Fusion score calculation."""
        from umrs.models.rag_pipeline import HybridSearcher

        searcher = HybridSearcher(
            vector_weight=0.6,
            bm25_weight=0.4,
            rrf_k=60,
        )

        # RRF formula: 1 / (k + rank)
        score = searcher._rrf_score(rank=1)
        expected = 1.0 / (60 + 1)

        assert abs(score - expected) < 0.0001

    def test_hybrid_search(self, sample_documents_with_embeddings):
        """Test hybrid search combines vector and BM25 results."""
        from umrs.models.rag_pipeline import HybridSearcher

        searcher = HybridSearcher()

        results = searcher.search(
            query="machine learning",
            documents=sample_documents_with_embeddings,
            top_k=3,
        )

        assert len(results) <= 3
        # Results should be SearchResult objects with scores
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "document_id") for r in results)
        # vector_score should reflect pre-computed scores, not all 1.0
        assert all(r.vector_score is not None and r.vector_score < 1.0 for r in results)

    def test_hybrid_search_deduplication(self, sample_documents_with_embeddings):
        """Test that hybrid search returns unique documents."""
        from umrs.models.rag_pipeline import HybridSearcher

        searcher = HybridSearcher()

        results = searcher.search(
            query="neural networks",
            documents=sample_documents_with_embeddings,
            top_k=5,
        )

        # Check no duplicate document IDs
        doc_ids = [r.document_id for r in results]
        assert len(doc_ids) == len(set(doc_ids))

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
        from umrs.models.rag_pipeline import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Model should not be loaded until first use
        assert reranker._model is None

    def test_reranker_with_mock_model(self, sample_search_results):
        """Test reranking with mocked model."""
        from umrs.models.rag_pipeline import CrossEncoderReranker

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
        # Should be sorted by rerank scores
        assert reranked[0].rerank_score >= reranked[1].rerank_score

    def test_reranker_empty_results(self):
        """Test reranking with empty results."""
        from umrs.models.rag_pipeline import CrossEncoderReranker

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
        from umrs.models.rag_pipeline import ContextAssembler

        assembler = ContextAssembler(enable_injection_scanning=False)

        context = assembler.assemble(
            query="What is machine learning?",
            results=sample_search_results,
        )

        # Returns RAGContext object
        assert hasattr(context, "query")
        assert hasattr(context, "context_text")
        assert "Machine learning is a subset" in context.context_text
        assert "[1]" in context.context_text  # Source reference

    def test_custom_template(self, sample_search_results):
        """Test context assembly with custom template."""
        from umrs.models.rag_pipeline import ContextAssembler

        template = "Query info:\n\n{chunks}\n\n---\nRefs: {sources}"
        assembler = ContextAssembler(
            context_template=template,
            enable_injection_scanning=False,
        )

        context = assembler.assemble(
            query="test query",
            results=sample_search_results,
        )

        assert "Refs:" in context.context_text

    def test_token_budget(self, sample_search_results):
        """Test token budget enforcement."""
        from umrs.models.rag_pipeline import ContextAssembler

        # Very small budget
        assembler = ContextAssembler(
            max_tokens=50,
            enable_injection_scanning=False,
        )

        context = assembler.assemble(
            query="test",
            results=sample_search_results,
        )

        # Should truncate to fit budget
        token_estimate = assembler._estimate_tokens(context.context_text)
        # Allow some margin for template overhead
        assert token_estimate <= 100

    def test_estimate_tokens(self):
        """Test token estimation."""
        from umrs.models.rag_pipeline import ContextAssembler

        assembler = ContextAssembler()

        # ~4 chars per token by default
        text = "This is a test sentence with some words."
        count = assembler._estimate_tokens(text)

        assert count > 0
        assert count < len(text)  # Should be less than char count

    def test_sources_in_context(self, sample_search_results):
        """Test source extraction in context."""
        from umrs.models.rag_pipeline import ContextAssembler

        assembler = ContextAssembler(enable_injection_scanning=False)

        context = assembler.assemble(
            query="test",
            results=sample_search_results,
        )

        # Sources should be included in the RAGContext
        assert len(context.sources) == 3
        source_ids = [s["source"] for s in context.sources]
        assert "doc1.txt" in source_ids
        assert "doc2.txt" in source_ids


# =============================================================================
# RAGPipeline Tests
# =============================================================================


class TestRAGPipeline:
    """Tests for complete RAG pipeline."""

    def test_pipeline_query(self, sample_documents_with_embeddings):
        """Test full pipeline query."""
        from umrs.models.rag_pipeline import RAGPipeline, ContextAssembler

        # Create pipeline with injection scanning disabled for test
        assembler = ContextAssembler(enable_injection_scanning=False)
        pipeline = RAGPipeline(context_assembler=assembler)

        context = pipeline.query(
            query="What is machine learning?",
            documents=sample_documents_with_embeddings,
            top_k=3,
            rerank=False,  # Skip reranking to avoid model loading
        )

        assert context is not None
        assert hasattr(context, "query")
        assert hasattr(context, "context_text")
        assert hasattr(context, "sources")
        assert context.query == "What is machine learning?"

    def test_pipeline_with_reranking(self, sample_documents_with_embeddings):
        """Test pipeline with reranking enabled."""
        from umrs.models.rag_pipeline import RAGPipeline, CrossEncoderReranker, ContextAssembler

        # Create reranker with mocked model
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

        assembler = ContextAssembler(enable_injection_scanning=False)
        pipeline = RAGPipeline(reranker=reranker, context_assembler=assembler)

        context = pipeline.query(
            query="test query",
            documents=sample_documents_with_embeddings,
            top_k=3,
            rerank=True,
        )

        assert context is not None
        assert len(context.sources) > 0

    def test_pipeline_top_k(self, sample_documents_with_embeddings):
        """Test pipeline respects top_k parameter."""
        from umrs.models.rag_pipeline import RAGPipeline, ContextAssembler

        assembler = ContextAssembler(enable_injection_scanning=False)
        pipeline = RAGPipeline(context_assembler=assembler)

        context = pipeline.query(
            query="neural networks",
            documents=sample_documents_with_embeddings,
            top_k=2,
            rerank=False,
        )

        # Should have at most 2 sources
        assert len(context.sources) <= 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline."""

    def test_end_to_end_without_model(self, sample_documents_with_embeddings):
        """Test end-to-end flow without actual model loading."""
        from umrs.models.rag_pipeline import (
            BM25,
            HybridSearcher,
            ContextAssembler,
        )

        # Test BM25 on its own
        bm25_docs = [(doc_id, text) for doc_id, text, _ in sample_documents_with_embeddings]
        bm25 = BM25()
        bm25.fit(bm25_docs)

        bm25_results = bm25.score("machine learning neural", top_k=3)
        assert len(bm25_results) == 3

        # Test hybrid searcher
        searcher = HybridSearcher()

        results = searcher.search(
            query="machine learning neural",
            documents=sample_documents_with_embeddings,
            top_k=3,
        )

        assert len(results) > 0

        # Test context assembler
        assembler = ContextAssembler(max_tokens=500, enable_injection_scanning=False)
        context = assembler.assemble(
            query="What is machine learning?",
            results=results,
        )

        assert context.context_text is not None
        assert isinstance(context.sources, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
