"""
Advanced RAG Pipeline (U-303 to U-306, T-216).

Provides:
- Hybrid search (BM25 + vector similarity)
- Cross-encoder reranking
- Context assembly for LLM prompts
- RAG → LLM integration
- T-216: Injection scanning in RAG retrieval
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


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
            zip(self._documents, self._doc_lengths, strict=False)
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

    def search(
        self,
        query: str,
        documents: list[tuple[str, str, float]],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query text
            documents: List of (doc_id, text, vector_score) tuples.
                       vector_score is a pre-computed cosine similarity in [0, 1].
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

        # Use pre-computed vector similarity scores
        vector_scores = [(doc_id, text, score) for doc_id, text, score in documents]

        # Sort by vector similarity
        vector_scores.sort(key=lambda x: x[2], reverse=True)
        vector_ranks = {
            doc_id: rank for rank, (doc_id, _, _) in enumerate(vector_scores)
        }

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
# Cross-Encoder Reranking (U-304; ADR-093 D-7)
# =============================================================================

RERANKER_MODEL_ENV = "UMRS_RAG_RERANKER_MODEL"
RERANKER_THREADS_ENV = "UMRS_RAG_RERANKER_THREADS"
#: MiniLM takes 512 tokens; 384 holds a query plus a 256-token chunk and its
#: header, and caps the cost of the rare long passage.
RERANKER_MAX_LENGTH = 384
RERANKER_BATCH_SIZE = 16
DEFAULT_RERANKER_THREADS = 2
#: Where ``huggingface_hub.snapshot_download(local_dir=...)`` records the
#: commit each file was downloaded at (first line of the ``.metadata`` file).
_DOWNLOAD_METADATA = (
    Path(".cache") / "huggingface" / "download" / "config.json.metadata"
)


class RerankerUnavailable(RuntimeError):
    """The cross-encoder cannot score; the message says why, by name."""


def reranker_staged(model_dir: str | os.PathLike[str]) -> bool:
    """A directory a cross-encoder loads from offline (the embedder's rule)."""
    path = Path(model_dir)
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def reranker_unavailable_reason(model: str) -> str | None:
    """Why ``model`` cannot be loaded offline, or ``None`` when it can.

    Only a staged directory loads. A hub id is refused by name: under
    ``HF_HUB_OFFLINE=1`` it can never load, and without it a load would be a
    runtime download (CONSTITUTION offline rule).
    """
    name = (model or "").strip()
    if not name:
        return f"not configured ({RERANKER_MODEL_ENV} is empty)"
    path = Path(name)
    if path.is_dir():
        if reranker_staged(path):
            return None
        return f"not staged: {name} lacks config.json or model.safetensors"
    if not path.is_absolute():
        return (
            f"refused: {name!r} is a hub id, not a staged directory "
            "(stage it first: standalone.sh reranker pull)"
        )
    return f"not staged: {name} does not exist"


def staged_revision(model_dir: str | os.PathLike[str]) -> str:
    """The hub commit a staged directory was downloaded at, else ``unknown``."""
    try:
        lines = (Path(model_dir) / _DOWNLOAD_METADATA).read_text().splitlines()
    except OSError:
        return "unknown"
    return (lines[0].strip() if lines else "") or "unknown"


def reranker_threads() -> int:
    """``UMRS_RAG_RERANKER_THREADS`` (default 2: the rag container's CPUs)."""
    try:
        return max(
            1, int(os.environ.get(RERANKER_THREADS_ENV, DEFAULT_RERANKER_THREADS))
        )
    except ValueError:
        return DEFAULT_RERANKER_THREADS


def _load_cross_encoder(path: str, device: str, max_length: int) -> Any:
    """A sentence-transformers CrossEncoder that returns RAW logits.

    The library's default activation for a one-label model is a sigmoid; the
    sigmoid belongs to the cut (``knowledge/retrieval/cutoff.py``), so the
    identity is set here. ``local_files_only``: nothing downloads, ever.
    """
    import torch
    from sentence_transformers import CrossEncoder

    kwargs = {"device": device, "max_length": max_length, "local_files_only": True}
    try:
        return CrossEncoder(path, activation_fn=torch.nn.Identity(), **kwargs)
    except TypeError:  # sentence-transformers < 4 names it differently
        return CrossEncoder(
            path, default_activation_function=torch.nn.Identity(), **kwargs
        )


def _set_torch_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def _sort_key(result: SearchResult) -> float:
    return result.rerank_score if result.rerank_score is not None else float("-inf")


class CrossEncoderReranker:
    """
    Cross-encoder reranking for improved relevance.

    Uses a cross-encoder model to score query-document pairs. ADR-093 D-7:
    loads only a staged directory (a hub id is refused by name), caches a
    load failure instead of retrying it on every call, returns raw logits,
    and runs at ``max_length`` 384, batch 16, ``UMRS_RAG_RERANKER_THREADS``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        *,
        max_length: int = RERANKER_MAX_LENGTH,
        batch_size: int = RERANKER_BATCH_SIZE,
        threads: int | None = None,
    ):
        """
        Initialize reranker.

        Args:
            model_name: staged model directory; ``None`` reads
                ``UMRS_RAG_RERANKER_MODEL`` (empty = no reranker)
            device: Device to run model on ('cpu' or 'cuda')
            max_length: token limit per (query, passage) pair
            batch_size: pairs per forward pass
            threads: torch threads; ``None`` reads ``UMRS_RAG_RERANKER_THREADS``
        """
        if model_name is None:
            model_name = os.environ.get(RERANKER_MODEL_ENV, "")
        self.model_name = model_name.strip()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.threads = reranker_threads() if threads is None else threads
        self._model = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._warned: set[str] = set()

    @property
    def load_error(self) -> str | None:
        """The cached reason a load attempt failed (never retried)."""
        return self._load_error

    @property
    def revision(self) -> str:
        return staged_revision(self.model_name) if self.model_name else "unknown"

    def unavailable_reason(self) -> str | None:
        """``None`` when the model is loaded or loadable, else why not."""
        if self._model is not None:
            return None
        return self._load_error or reranker_unavailable_reason(self.model_name)

    def _load_model(self) -> None:
        """Lazy load from the staged directory; a failure is cached."""
        with self._lock:
            if self._model is not None or self._load_error is not None:
                return
            if reranker_unavailable_reason(self.model_name) is not None:
                return
            try:
                self._model = _load_cross_encoder(
                    self.model_name, self.device, self.max_length
                )
                _set_torch_threads(self.threads)
            except Exception as exc:  # ImportError, OSError, bad weights: all final
                self._load_error = f"load failed: {type(exc).__name__}: {exc}"
                logger.error("cross-encoder %s: %s", self.model_name, self._load_error)
                return
            logger.info("Loaded cross-encoder: %s (%s)", self.model_name, self.revision)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Raw logits, one per passage, in input order.

        Raises :class:`RerankerUnavailable` with the reason when the model is
        not configured, not staged, a hub id, or failed to load.
        """
        self._load_model()
        if self._model is None:
            raise RerankerUnavailable(self.unavailable_reason() or "not loaded")
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        logits = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(value) for value in np.asarray(logits, dtype=float).reshape(-1)]

    def _mark_unavailable(self, results: list[SearchResult], reason: str) -> None:
        """The legacy routes keep their order, but never silently."""
        if reason not in self._warned:
            self._warned.add(reason)
            logger.warning("cross-encoder rerank skipped: %s", reason)
        for result in results:
            result.metadata["rerank"] = f"unavailable: {reason}"

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Search query
            results: Initial search results
            top_k: Number of results to return (None = all)

        Returns:
            Reranked search results. When the model is unavailable the input
            order is kept and every result says so in
            ``metadata["rerank"]``.
        """
        if not results:
            return results

        try:
            scores = self.score(query, [r.content for r in results])
        except RerankerUnavailable as exc:
            self._mark_unavailable(results, str(exc))
            return results[:top_k] if top_k else results

        for result, score in zip(results, scores, strict=False):
            result.rerank_score = score

        # Raw logits are signed: an unscored result sorts last, not at 0.
        results.sort(key=_sort_key, reverse=True)

        return results[:top_k] if top_k else results


# =============================================================================
# Content Scanning for RAG (T-216)
# =============================================================================


class RAGContentScanner:
    """
    Scanner for detecting prompt injection in RAG chunks (T-216).

    Lightweight scanner optimized for RAG content that can be enabled/disabled.
    """

    # High-confidence injection patterns for RAG
    INJECTION_PATTERNS = [
        (
            re.compile(
                r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b",
                re.IGNORECASE,
            ),
            0.9,
        ),
        (
            re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior)\b", re.IGNORECASE),
            0.85,
        ),
        (
            re.compile(
                r"\boverride\s+(?:your|the|all)\s+(?:instructions?|rules?)\b",
                re.IGNORECASE,
            ),
            0.9,
        ),
        (
            re.compile(r"\benter\s+(?:developer|admin|root)\s+mode\b", re.IGNORECASE),
            0.95,
        ),
        (re.compile(r"\[system\]", re.IGNORECASE), 0.85),
        (re.compile(r'"role"\s*:\s*"system"', re.IGNORECASE), 0.9),
    ]

    def __init__(
        self,
        block_threshold: float = 0.8,
        warn_threshold: float = 0.5,
    ):
        """Initialize scanner with thresholds."""
        self.block_threshold = block_threshold
        self.warn_threshold = warn_threshold

    def scan_chunk(
        self,
        content: str,
        document_id: str = "",
    ) -> tuple[str, float, bool]:
        """
        Scan a RAG chunk for injection.

        Args:
            content: Chunk content
            document_id: Source document ID

        Returns:
            Tuple of (processed_content, risk_score, is_blocked)
        """
        max_score = 0.0

        for pattern, score in self.INJECTION_PATTERNS:
            if pattern.search(content):
                max_score = max(max_score, score)
                if max_score >= self.block_threshold:
                    break

        is_blocked = max_score >= self.block_threshold

        if is_blocked:
            logger.warning(
                "RAG chunk blocked due to injection risk",
                extra={
                    "document_id": document_id,
                    "risk_score": max_score,
                },
            )
            return (
                f"[RAG chunk blocked: injection risk {max_score:.2f}]",
                max_score,
                True,
            )

        if max_score >= self.warn_threshold:
            logger.warning(
                "RAG chunk has elevated injection risk",
                extra={
                    "document_id": document_id,
                    "risk_score": max_score,
                },
            )

        return content, max_score, False


# Default scanner instance
_rag_scanner: RAGContentScanner | None = None


def get_rag_scanner() -> RAGContentScanner:
    """Get or create default RAG scanner."""
    global _rag_scanner
    if _rag_scanner is None:
        _rag_scanner = RAGContentScanner()
    return _rag_scanner


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
    - T-216: Injection scanning of chunks
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        chars_per_token: float = 4.0,
        context_template: str | None = None,
        enable_injection_scanning: bool = True,
        scanner: RAGContentScanner | None = None,
    ):
        """
        Initialize context assembler.

        Args:
            max_tokens: Maximum tokens in assembled context
            chars_per_token: Estimated characters per token
            context_template: Template for formatting context
            enable_injection_scanning: Enable T-216 injection scanning
            scanner: Custom scanner instance
        """
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.context_template = context_template or self._default_template()
        self.enable_injection_scanning = enable_injection_scanning
        self.scanner = scanner

    def _default_template(self) -> str:
        """Default context template."""
        return """Based on the following relevant information:

{chunks}

---
Sources: {sources}"""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return int(len(text) / self.chars_per_token)

    def assemble(
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
            content = result.content

            # T-216: Scan chunk for injection if enabled
            if self.enable_injection_scanning:
                scanner = self.scanner or get_rag_scanner()
                content, risk_score, is_blocked = scanner.scan_chunk(
                    content,
                    document_id=result.document_id,
                )
                if is_blocked:
                    # Skip blocked chunks but track for audit
                    result.metadata["injection_blocked"] = True
                    result.metadata["injection_risk_score"] = risk_score
                    continue
                elif risk_score > 0:
                    result.metadata["injection_risk_score"] = risk_score

            # Check token budget
            chunk_chars = len(content)
            if total_chars + chunk_chars > max_chars:
                # Try to fit partial content
                remaining = max_chars - total_chars
                if remaining > 200:  # Only include if meaningful amount
                    truncated = content[:remaining] + "..."
                    chunks.append(f"[{i + 1}] {truncated}")
                    total_chars += len(truncated)
                break

            chunks.append(f"[{i + 1}] {content}")
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
        hybrid_searcher: HybridSearcher | None = None,
        reranker: CrossEncoderReranker | None = None,
        context_assembler: ContextAssembler | None = None,
        gateway_url: str | None = None,
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

    def query(
        self,
        query: str,
        documents: list[tuple[str, str, float]],
        top_k: int = 5,
        rerank: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> RAGContext:
        """
        Execute RAG query pipeline.

        Args:
            query: User query
            documents: Corpus documents (doc_id, text, vector_score)
            top_k: Number of chunks for context
            rerank: Whether to apply reranking
            metadata: Additional metadata to include

        Returns:
            Assembled RAG context
        """
        # Step 1: Hybrid search
        search_results = self.hybrid_searcher.search(
            query=query,
            documents=documents,
            top_k=top_k * 2 if rerank else top_k,
        )

        # Step 2: Rerank (optional)
        if rerank and search_results:
            search_results = self.reranker.rerank(
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
        context = self.context_assembler.assemble(
            query=query,
            results=search_results,
        )

        return context

    async def query_with_llm(
        self,
        query: str,
        context: RAGContext,
        model: str = "rivault-gpt-oss-static",
        system_prompt: str | None = None,
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

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.gateway_url}/gw/llm/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response,
        ):
            result = await response.json()

        return {
            "answer": result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", ""),
            "sources": context.sources,
            "model": result.get("model", model),
            "usage": result.get("usage", {}),
        }
