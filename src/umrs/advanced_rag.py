"""
Advanced RAG Module (U-303, U-304, U-305).

Provides:
- Hybrid search: Combine vector similarity with BM25 keyword matching
- Reranking: Cross-encoder reranking for improved relevance
- Context assembly: Format retrieved chunks for LLM prompts

This module extends the existing UMRS RAG capabilities with production-quality
retrieval features suitable for scientific computing workloads.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with scores and metadata."""

    content: str
    doc_id: str
    chunk_id: int
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "metadata": self.metadata,
        }


@dataclass
class AssembledContext:
    """Assembled context ready for LLM consumption."""

    text: str
    chunks: list[RetrievedChunk]
    total_tokens: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "chunks": [c.to_dict() for c in self.chunks],
            "total_tokens": self.total_tokens,
            "truncated": self.truncated,
        }


# =============================================================================
# BM25 Implementation (U-303)
# =============================================================================


class BM25:
    """
    BM25 (Best Matching 25) ranking function.

    Implementation of the Okapi BM25 algorithm for keyword-based retrieval.
    Used alongside vector search for hybrid retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 parameters.

        Args:
            k1: Term frequency saturation parameter (typical: 1.2-2.0)
            b: Document length normalization (0=none, 1=full)
            epsilon: Floor for IDF to avoid negative values
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Corpus statistics (populated by fit())
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._corpus_size: int = 0
        self._tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r"\w+", text.lower())

    def fit(self, corpus: list[str]) -> BM25:
        """
        Fit BM25 on a corpus of documents.

        Args:
            corpus: List of document strings

        Returns:
            Self for chaining
        """
        self._tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self._corpus_size = len(corpus)
        self._doc_lengths = [len(doc) for doc in self._tokenized_corpus]
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._corpus_size if self._corpus_size else 0
        )

        # Compute document frequencies
        self._doc_freqs = {}
        for doc in self._tokenized_corpus:
            seen = set()
            for token in doc:
                if token not in seen:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen.add(token)

        # Compute IDF values
        self._idf = {}
        for token, df in self._doc_freqs.items():
            idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1)
            self._idf[token] = max(idf, self.epsilon)

        return self

    def score(self, query: str, doc_idx: int) -> float:
        """
        Compute BM25 score for a query against a document.

        Args:
            query: Query string
            doc_idx: Index of document in corpus

        Returns:
            BM25 score
        """
        if doc_idx >= len(self._tokenized_corpus):
            return 0.0

        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenized_corpus[doc_idx]
        doc_length = self._doc_lengths[doc_idx]

        # Count term frequencies in document
        tf: dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue

            idf = self._idf.get(token, self.epsilon)
            term_freq = tf[token]

            # BM25 formula
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * doc_length / self._avg_doc_length
            )
            score += idf * numerator / denominator

        return score

    def score_batch(self, query: str) -> list[float]:
        """
        Score all documents in corpus against a query.

        Args:
            query: Query string

        Returns:
            List of scores for each document
        """
        return [self.score(query, i) for i in range(self._corpus_size)]


# =============================================================================
# Hybrid Search (U-303)
# =============================================================================


class HybridSearcher:
    """
    Hybrid search combining vector similarity and BM25 keyword matching.

    Uses Reciprocal Rank Fusion (RRF) to combine scores from both methods.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_weight: Weight for vector similarity scores
            keyword_weight: Weight for BM25 keyword scores
            rrf_k: RRF constant (typical: 60)
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self._bm25: Optional[BM25] = None

    def fit_keyword_index(self, documents: list[str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of document strings
        """
        self._bm25 = BM25().fit(documents)

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _rrf_score(self, ranks: list[int]) -> float:
        """Compute Reciprocal Rank Fusion score."""
        return sum(1 / (self.rrf_k + r) for r in ranks)

    def search(
        self,
        query: str,
        vector_results: list[tuple[int, float]],  # [(doc_idx, score), ...]
        top_k: int = 10,
    ) -> list[tuple[int, float, float, float]]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            vector_results: Vector search results as (doc_idx, score) pairs
            top_k: Number of results to return

        Returns:
            List of (doc_idx, final_score, vector_score, keyword_score) tuples
        """
        if not self._bm25:
            # No keyword index, return vector results only
            return [
                (idx, score, score, 0.0)
                for idx, score in vector_results[:top_k]
            ]

        # Get BM25 scores for all documents
        keyword_scores = self._bm25.score_batch(query)

        # Build result set from vector results
        doc_indices = set(idx for idx, _ in vector_results)

        # Add top BM25 results to candidate set
        keyword_ranked = sorted(
            enumerate(keyword_scores), key=lambda x: x[1], reverse=True
        )
        for idx, _ in keyword_ranked[:top_k * 2]:
            doc_indices.add(idx)

        # Compute combined scores
        results = []
        for idx in doc_indices:
            # Get vector score
            vector_score = 0.0
            for v_idx, v_score in vector_results:
                if v_idx == idx:
                    vector_score = v_score
                    break

            # Get keyword score
            kw_score = keyword_scores[idx] if idx < len(keyword_scores) else 0.0

            # Weighted combination
            final_score = (
                self.vector_weight * vector_score
                + self.keyword_weight * kw_score
            )

            results.append((idx, final_score, vector_score, kw_score))

        # Sort by final score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# =============================================================================
# Reranking (U-304)
# =============================================================================


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improved relevance scoring.

    Uses a cross-encoder model to jointly encode query-document pairs
    for more accurate relevance scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device for inference ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Reranking will be disabled."
            )
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self._model = None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks using cross-encoder.

        Args:
            query: Query string
            chunks: List of retrieved chunks
            top_k: Number of results to return (None = all)

        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []

        self._load_model()

        if self._model is None:
            # Model not available, return original order
            for chunk in chunks:
                chunk.rerank_score = chunk.final_score
            return chunks

        # Prepare query-document pairs
        pairs = [(query, chunk.content) for chunk in chunks]

        # Score with cross-encoder
        try:
            scores = self._model.predict(pairs, batch_size=self.batch_size)

            # Update chunks with rerank scores
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
                chunk.final_score = float(score)  # Update final score

            # Sort by rerank score
            chunks.sort(key=lambda x: x.rerank_score, reverse=True)

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Keep original scores
            for chunk in chunks:
                chunk.rerank_score = chunk.final_score

        if top_k:
            return chunks[:top_k]
        return chunks


# =============================================================================
# Context Assembly (U-305)
# =============================================================================


class ContextAssembler:
    """
    Assemble retrieved chunks into context for LLM prompts.

    Handles:
    - Token budget management
    - Chunk ordering strategies
    - Metadata inclusion
    - Deduplication
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        chunk_separator: str = "\n\n---\n\n",
        include_metadata: bool = True,
        include_scores: bool = False,
        tokenizer: Optional[Callable[[str], int]] = None,
    ):
        """
        Initialize context assembler.

        Args:
            max_tokens: Maximum tokens in assembled context
            chunk_separator: Separator between chunks
            include_metadata: Include chunk metadata in context
            include_scores: Include relevance scores in context
            tokenizer: Function to count tokens (default: word count / 0.75)
        """
        self.max_tokens = max_tokens
        self.chunk_separator = chunk_separator
        self.include_metadata = include_metadata
        self.include_scores = include_scores
        self.tokenizer = tokenizer or self._default_tokenizer

    def _default_tokenizer(self, text: str) -> int:
        """Default token estimation (words / 0.75)."""
        return int(len(text.split()) / 0.75)

    def _format_chunk(
        self,
        chunk: RetrievedChunk,
        index: int,
    ) -> str:
        """Format a single chunk for inclusion in context."""
        parts = []

        # Header with index
        parts.append(f"[Document {index + 1}]")

        # Metadata
        if self.include_metadata and chunk.metadata:
            meta_parts = []
            if "source" in chunk.metadata:
                meta_parts.append(f"Source: {chunk.metadata['source']}")
            if "title" in chunk.metadata:
                meta_parts.append(f"Title: {chunk.metadata['title']}")
            if "date" in chunk.metadata:
                meta_parts.append(f"Date: {chunk.metadata['date']}")
            if meta_parts:
                parts.append(" | ".join(meta_parts))

        # Relevance score
        if self.include_scores:
            parts.append(f"Relevance: {chunk.final_score:.3f}")

        # Content
        parts.append("")
        parts.append(chunk.content)

        return "\n".join(parts)

    def _deduplicate(
        self,
        chunks: list[RetrievedChunk],
        similarity_threshold: float = 0.9,
    ) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks based on content similarity."""
        if len(chunks) <= 1:
            return chunks

        # Simple deduplication based on exact content match
        seen_content = set()
        deduplicated = []

        for chunk in chunks:
            # Normalize content for comparison
            normalized = " ".join(chunk.content.lower().split())

            # Check for duplicates
            is_duplicate = False
            for seen in seen_content:
                # Simple Jaccard-like similarity
                words1 = set(normalized.split())
                words2 = set(seen.split())
                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_content.add(normalized)
                deduplicated.append(chunk)

        return deduplicated

    def assemble(
        self,
        chunks: list[RetrievedChunk],
        deduplicate: bool = True,
    ) -> AssembledContext:
        """
        Assemble chunks into context.

        Args:
            chunks: Retrieved and optionally reranked chunks
            deduplicate: Remove near-duplicate chunks

        Returns:
            AssembledContext with formatted text
        """
        if not chunks:
            return AssembledContext(
                text="No relevant documents found.",
                chunks=[],
                total_tokens=0,
                truncated=False,
            )

        # Deduplicate if requested
        if deduplicate:
            chunks = self._deduplicate(chunks)

        # Assemble context within token budget
        assembled_parts = []
        included_chunks = []
        total_tokens = 0
        truncated = False

        for i, chunk in enumerate(chunks):
            formatted = self._format_chunk(chunk, i)
            chunk_tokens = self.tokenizer(formatted)

            # Check if adding this chunk exceeds budget
            separator_tokens = (
                self.tokenizer(self.chunk_separator) if assembled_parts else 0
            )
            if total_tokens + chunk_tokens + separator_tokens > self.max_tokens:
                truncated = True
                break

            assembled_parts.append(formatted)
            included_chunks.append(chunk)
            total_tokens += chunk_tokens + separator_tokens

        # Join parts
        text = self.chunk_separator.join(assembled_parts)

        return AssembledContext(
            text=text,
            chunks=included_chunks,
            total_tokens=total_tokens,
            truncated=truncated,
        )


# =============================================================================
# Unified RAG Pipeline
# =============================================================================


class AdvancedRAGPipeline:
    """
    Unified advanced RAG pipeline combining all components.

    Provides a single interface for:
    1. Hybrid search (vector + keyword)
    2. Cross-encoder reranking
    3. Context assembly
    """

    def __init__(
        self,
        hybrid_searcher: Optional[HybridSearcher] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        assembler: Optional[ContextAssembler] = None,
        enable_reranking: bool = True,
    ):
        """
        Initialize advanced RAG pipeline.

        Args:
            hybrid_searcher: Custom hybrid searcher (default: new instance)
            reranker: Custom reranker (default: new instance)
            assembler: Custom assembler (default: new instance)
            enable_reranking: Enable cross-encoder reranking
        """
        self.hybrid_searcher = hybrid_searcher or HybridSearcher()
        self.reranker = reranker or CrossEncoderReranker()
        self.assembler = assembler or ContextAssembler()
        self.enable_reranking = enable_reranking

    def search_and_assemble(
        self,
        query: str,
        documents: list[str],
        vector_results: list[tuple[int, float]],
        metadata: Optional[list[dict]] = None,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
    ) -> AssembledContext:
        """
        Full RAG pipeline: search, rerank, assemble.

        Args:
            query: Search query
            documents: Corpus of documents
            vector_results: Vector search results as (doc_idx, score) pairs
            metadata: Optional metadata for each document
            top_k: Number of results after hybrid search
            rerank_top_k: Number of results after reranking (default: top_k)

        Returns:
            AssembledContext ready for LLM
        """
        # Build keyword index if not done
        self.hybrid_searcher.fit_keyword_index(documents)

        # Hybrid search
        hybrid_results = self.hybrid_searcher.search(
            query=query,
            vector_results=vector_results,
            top_k=top_k,
        )

        # Create chunk objects
        chunks = []
        for doc_idx, final_score, vector_score, keyword_score in hybrid_results:
            if doc_idx < len(documents):
                chunk = RetrievedChunk(
                    content=documents[doc_idx],
                    doc_id=str(doc_idx),
                    chunk_id=doc_idx,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    final_score=final_score,
                    metadata=metadata[doc_idx] if metadata else {},
                )
                chunks.append(chunk)

        # Rerank if enabled
        if self.enable_reranking and chunks:
            chunks = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=rerank_top_k or top_k,
            )

        # Assemble context
        return self.assembler.assemble(chunks)


# =============================================================================
# Factory Functions
# =============================================================================


def create_advanced_rag_pipeline(
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_context_tokens: int = 4096,
    enable_reranking: bool = True,
) -> AdvancedRAGPipeline:
    """
    Create an advanced RAG pipeline with custom settings.

    Args:
        vector_weight: Weight for vector similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
        reranker_model: HuggingFace model for cross-encoder
        max_context_tokens: Maximum tokens in assembled context
        enable_reranking: Enable cross-encoder reranking

    Returns:
        Configured AdvancedRAGPipeline
    """
    hybrid_searcher = HybridSearcher(
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )

    reranker = CrossEncoderReranker(
        model_name=reranker_model,
    ) if enable_reranking else None

    assembler = ContextAssembler(
        max_tokens=max_context_tokens,
    )

    return AdvancedRAGPipeline(
        hybrid_searcher=hybrid_searcher,
        reranker=reranker,
        assembler=assembler,
        enable_reranking=enable_reranking,
    )
