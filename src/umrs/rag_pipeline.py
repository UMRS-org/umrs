"""
Advanced RAG Pipeline (U-303 to U-306).

Provides:
- Hybrid search (BM25 + vector similarity)
- Cross-encoder reranking
- Context assembly for LLM prompts
- RAG → LLM integration
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SearchResult:
    """Single search result."""

    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RAGContext:
    """Assembled RAG context for LLM."""

    query: str
    context_text: str
    sources: list[dict[str, Any]]
    total_chunks: int
    total_tokens_estimate: int


# =============================================================================
# BM25 Implementation (U-303)
# =============================================================================


class BM25:
    """
    BM25 scoring for keyword-based retrieval.

    Implements Okapi BM25 algorithm for text ranking.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._documents: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_ids: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        # Remove punctuation, lowercase, split on whitespace
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """
        Fit BM25 on a corpus.

        Args:
            documents: List of (doc_id, text) tuples
        """
        self._documents = []
        self._doc_ids = []
        self._doc_lengths = []
        self._doc_freqs = {}

        for doc_id, text in documents:
            tokens = self._tokenize(text)
            self._documents.append(tokens)
            self._doc_ids.append(doc_id)
            self._doc_lengths.append(len(tokens))

            # Count document frequency for each term
            seen_terms = set()
            for token in tokens:
                if token not in seen_terms:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen_terms.add(token)

        # Calculate average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0.0

        # Calculate IDF for each term
        n_docs = len(self._documents)
        for term, df in self._doc_freqs.items():
            # IDF with smoothing
            self._idf[term] = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Score documents against query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = []

        for i, (doc_tokens, doc_length) in enumerate(
            zip(self._documents, self._doc_lengths)
        ):
            score = 0.0

            # Count term frequencies in document
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1

            # Calculate BM25 score
            for token in query_tokens:
                if token not in tf:
                    continue

                idf = self._idf.get(token, 0.0)
                term_freq = tf[token]

                # BM25 scoring formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_length / self._avg_doc_length
                )
                score += idf * numerator / denominator

            scores.append((self._doc_ids[i], score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =============================================================================
# Hybrid Search (U-303)
# =============================================================================


class HybridSearcher:
    """
    Hybrid search combining BM25 and vector similarity.

    Uses Reciprocal Rank Fusion (RRF) to combine scores.
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_weight: Weight for vector similarity scores
            bm25_weight: Weight for BM25 scores
            rrf_k: RRF constant (higher = more emphasis on top ranks)
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.bm25 = BM25()

    def _rrf_score(self, rank: int) -> float:
        """Calculate Reciprocal Rank Fusion score."""
        return 1.0 / (self.rrf_k + rank)

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        documents: list[tuple[str, str, list[float]]],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query text
            query_embedding: Query vector embedding
            documents: List of (doc_id, text, embedding) tuples
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Fit BM25 on documents
        bm25_docs = [(doc_id, text) for doc_id, text, _ in documents]
        self.bm25.fit(bm25_docs)

        # Get BM25 scores
        bm25_results = self.bm25.score(query, top_k=top_k * 2)
        bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results)}

        # Calculate vector similarity scores
        query_vec = np.array(query_embedding)
        vector_scores = []

        for doc_id, text, embedding in documents:
            doc_vec = np.array(embedding)
            # Cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
            )
            vector_scores.append((doc_id, text, similarity))

        # Sort by vector similarity
        vector_scores.sort(key=lambda x: x[2], reverse=True)
        vector_ranks = {doc_id: rank for rank, (doc_id, _, _) in enumerate(vector_scores)}

        # Combine scores using RRF
        combined_scores: dict[str, float] = {}
        doc_texts: dict[str, str] = {}
        doc_vector_scores: dict[str, float] = {}
        doc_bm25_scores: dict[str, float] = {}

        for doc_id, text, vec_score in vector_scores:
            doc_texts[doc_id] = text
            doc_vector_scores[doc_id] = vec_score

            # RRF combination
            vec_rank = vector_ranks.get(doc_id, len(documents))
            bm25_rank = bm25_ranks.get(doc_id, len(documents))

            rrf_vec = self._rrf_score(vec_rank) * self.vector_weight
            rrf_bm25 = self._rrf_score(bm25_rank) * self.bm25_weight

            combined_scores[doc_id] = rrf_vec + rrf_bm25

        for doc_id, bm25_score in bm25_results:
            doc_bm25_scores[doc_id] = bm25_score

        # Sort by combined score
        ranked_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )[:top_k]

        # Build results
        results = []
        for doc_id in ranked_ids:
            results.append(
                SearchResult(
                    document_id=doc_id,
                    chunk_id=doc_id,  # For now, chunk_id == doc_id
                    content=doc_texts[doc_id],
                    score=combined_scores[doc_id],
                    vector_score=doc_vector_scores.get(doc_id),
                    bm25_score=doc_bm25_scores.get(doc_id),
                )
            )

        return results


# =============================================================================
# Cross-Encoder Reranking (U-304)
# =============================================================================


class CrossEncoderReranker:
    """
    Cross-encoder reranking for improved relevance.

    Uses a cross-encoder model to score query-document pairs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Reranking will use original scores."
                )
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Search query
            results: Initial search results
            top_k: Number of results to return (None = all)

        Returns:
            Reranked search results
        """
        if not results:
            return results

        self._load_model()

        if self._model is None:
            # Fallback: return original results
            return results[:top_k] if top_k else results

        # Score query-document pairs
        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)

        # Update results with rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)

        return results[:top_k] if top_k else results


# =============================================================================
# Context Assembly (U-305)
# =============================================================================


class ContextAssembler:
    """
    Assemble retrieved chunks into LLM context.

    Handles:
    - Token budget management
    - Source deduplication
    - Context formatting
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        chars_per_token: float = 4.0,
        context_template: Optional[str] = None,
    ):
        """
        Initialize context assembler.

        Args:
            max_tokens: Maximum tokens in assembled context
            chars_per_token: Estimated characters per token
            context_template: Template for formatting context
        """
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.context_template = context_template or self._default_template()

    def _default_template(self) -> str:
        """Default context template."""
        return """Based on the following relevant information:

{chunks}

---
Sources: {sources}"""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return int(len(text) / self.chars_per_token)

    async def assemble(
        self,
        query: str,
        results: list[SearchResult],
    ) -> RAGContext:
        """
        Assemble search results into RAG context.

        Args:
            query: Original query
            results: Search results to include

        Returns:
            Assembled RAG context
        """
        chunks = []
        sources = []
        seen_sources = set()
        total_chars = 0
        max_chars = int(self.max_tokens * self.chars_per_token)

        for i, result in enumerate(results):
            # Check token budget
            chunk_chars = len(result.content)
            if total_chars + chunk_chars > max_chars:
                # Try to fit partial content
                remaining = max_chars - total_chars
                if remaining > 200:  # Only include if meaningful amount
                    truncated = result.content[:remaining] + "..."
                    chunks.append(f"[{i + 1}] {truncated}")
                    total_chars += len(truncated)
                break

            chunks.append(f"[{i + 1}] {result.content}")
            total_chars += chunk_chars

            # Track unique sources
            source = result.metadata.get("source", result.document_id)
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append(
                    {
                        "id": result.document_id,
                        "source": source,
                        "score": result.score,
                    }
                )

        # Format context
        chunks_text = "\n\n".join(chunks)
        sources_text = ", ".join(s["source"] for s in sources)

        context_text = self.context_template.format(
            chunks=chunks_text,
            sources=sources_text,
        )

        return RAGContext(
            query=query,
            context_text=context_text,
            sources=sources,
            total_chunks=len(chunks),
            total_tokens_estimate=self._estimate_tokens(context_text),
        )


# =============================================================================
# RAG Pipeline (U-306)
# =============================================================================


class RAGPipeline:
    """
    Complete RAG pipeline for query → context → LLM.

    Combines hybrid search, reranking, and context assembly.
    """

    def __init__(
        self,
        hybrid_searcher: Optional[HybridSearcher] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        context_assembler: Optional[ContextAssembler] = None,
        gateway_url: Optional[str] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            hybrid_searcher: Hybrid search component
            reranker: Cross-encoder reranker
            context_assembler: Context assembler
            gateway_url: Platform Gateway URL for LLM calls
        """
        self.hybrid_searcher = hybrid_searcher or HybridSearcher()
        self.reranker = reranker or CrossEncoderReranker()
        self.context_assembler = context_assembler or ContextAssembler()
        self.gateway_url = gateway_url

    async def query(
        self,
        query: str,
        query_embedding: list[float],
        documents: list[tuple[str, str, list[float]]],
        top_k: int = 5,
        rerank: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> RAGContext:
        """
        Execute RAG query pipeline.

        Args:
            query: User query
            query_embedding: Query vector embedding
            documents: Corpus documents (doc_id, text, embedding)
            top_k: Number of chunks for context
            rerank: Whether to apply reranking
            metadata: Additional metadata to include

        Returns:
            Assembled RAG context
        """
        # Step 1: Hybrid search
        search_results = await self.hybrid_searcher.search(
            query=query,
            query_embedding=query_embedding,
            documents=documents,
            top_k=top_k * 2 if rerank else top_k,
        )

        # Step 2: Rerank (optional)
        if rerank and search_results:
            search_results = await self.reranker.rerank(
                query=query,
                results=search_results,
                top_k=top_k,
            )
        else:
            search_results = search_results[:top_k]

        # Add metadata to results
        if metadata:
            for result in search_results:
                result.metadata.update(metadata)

        # Step 3: Assemble context
        context = await self.context_assembler.assemble(
            query=query,
            results=search_results,
        )

        return context

    async def query_with_llm(
        self,
        query: str,
        context: RAGContext,
        model: str = "rivault-gpt-oss-static",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Execute RAG query with LLM generation.

        Args:
            query: User query
            context: Assembled RAG context
            model: LLM model name
            system_prompt: System prompt for LLM
            max_tokens: Max tokens in response
            temperature: LLM temperature

        Returns:
            LLM response with sources
        """
        if not self.gateway_url:
            raise ValueError("gateway_url required for LLM integration")

        import aiohttp

        default_system = (
            "You are a helpful assistant. Answer questions based on the provided "
            "context. If the context doesn't contain relevant information, say so. "
            "Cite sources when possible using [1], [2], etc."
        )

        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {
                "role": "user",
                "content": f"{context.context_text}\n\nQuestion: {query}",
            },
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.gateway_url}/gw/llm/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                result = await response.json()

        return {
            "answer": result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", ""),
            "sources": context.sources,
            "model": result.get("model", model),
            "usage": result.get("usage", {}),
        }
