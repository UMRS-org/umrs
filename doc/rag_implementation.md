# RAG Implementation

Technical reference for the Retrieval-Augmented Generation (RAG) subsystem in UMRS.

## Architecture Overview

```
                         ┌─────────────────────────────────┐
                         │         FastAPI Endpoints        │
                         │  /rag/search   /rag/qa   /rag/query  │
                         └──────────┬──────────────────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
              vector (default)   hybrid           reranking
                   │                │                │
                   ▼                ▼                ▼
            ┌──────────┐   ┌──────────────┐   ┌──────────────┐
            │ pgvector │   │ HybridSearcher│   │CrossEncoder  │
            │ cosine   │   │ BM25 + RRF    │   │ Reranker     │
            └──────────┘   └──────────────┘   └──────────────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │     PostgreSQL      │
                         │  pgvector + HNSW    │
                         └─────────────────────┘
```

**Source files:**

| File | Role |
|---|---|
| `src/rag_service.py` | FastAPI endpoints, retriever, score conversion |
| `src/umrs/rag_pipeline.py` | BM25, hybrid search, reranking, context assembly |
| `src/umrs/sql.py` | Table creation, HNSW index management |

## Search Methods

### 1. Vector Search (default)

Pure cosine similarity via pgvector.

- **Operator:** `<=>` (cosine distance, range `[0, 2]`)
- **Score conversion:** `similarity = max(0, 1 - distance)` → range `[0, 1]`
- **Index:** HNSW (`vector_cosine_ops`, `m=16`, `ef_construction=64`)

Request:
```json
{"query": "fault detection", "top_k": 5}
```

### 2. Hybrid Search (BM25 + Vector RRF)

Combines keyword relevance (BM25) with semantic similarity using Reciprocal Rank Fusion.

- **BM25:** Okapi BM25 (`k1=1.5`, `b=0.75`) with whitespace tokenization
- **RRF formula:** `score = Σ weight_i / (k + rank_i)` with `k=60`
- **Weights:** vector=0.6, BM25=0.4

The hybrid searcher operates on the candidate pool already retrieved by pgvector. It receives pre-computed cosine similarity scores (not raw embeddings) to avoid redundant computation.

Request:
```json
{"query": "fault detection", "top_k": 5, "search_method": "hybrid"}
```

Response metadata includes `hybrid_score`, `bm25_score`, and `vector_score` for transparency.

### 3. Cross-Encoder Reranking

Applies a cross-encoder model to re-score query-document pairs for higher precision.

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable)
- **Loading:** Lazy — loaded on first reranking request
- **Scores:** Raw cross-encoder logits (not bounded to `[0, 1]`)

Request:
```json
{"query": "fault detection", "top_k": 5, "enable_reranking": true}
```

Reranking can be combined with either vector or hybrid search. When enabled, `top_k * 3` candidates are fetched from pgvector to give the reranker a larger pool.

## Score Semantics

The top-level `score` field always reflects the **final ranking score** — the value that determined the result ordering. Metadata fields provide the breakdown:

| Mode | `score` = | Metadata |
|---|---|---|
| vector | cosine similarity (0–1) | — |
| hybrid | RRF combined score | `hybrid_score`, `bm25_score`, `vector_score` |
| rerank | cross-encoder logit | `rerank_score` |

## HNSW Index

Created automatically by `sql.py` when a table stores embeddings:

```sql
CREATE INDEX IF NOT EXISTS idx_{table}_embedding_hnsw
  ON {table} USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

- **New tables:** Index created during `configure_table()`.
- **Existing tables:** Checked on startup; created if missing (upgrade path).

## API Endpoints

### `POST /rag/search`

Retrieves documents by semantic (and optionally hybrid/reranked) search.

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural language query |
| `top_k` | int | 5 | Number of results |
| `search_method` | string | `"vector"` | `"vector"` or `"hybrid"` |
| `enable_reranking` | bool | `false` | Apply cross-encoder reranking |
| `user` | string | null | Filter by UMRS user |
| `bucket_alias` | string | null | Filter by bucket |
| `xp_name` | string | null | Filter by experiment |

### `POST /rag/qa`

Same retrieval parameters as `/rag/search`, plus LLM-generated answer.

### `POST /rag/query`

Gateway endpoint — forwards to UMRS ZMQ server.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `UMRS_RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `UMRS_RAG_RERANKER_DEVICE` | `cpu` | Inference device (`cpu` / `cuda`) |
| `UMRS_RAG_LLM_MODEL` | `microsoft/Phi-3-mini-4k-instruct` | LLM for QA answers |

## Pipeline Module (`rag_pipeline.py`)

Self-contained module with no database dependencies. All classes operate on in-memory data structures:

- **`BM25`** — Okapi BM25 scoring on tokenized text
- **`HybridSearcher`** — Combines BM25 ranks + pre-computed vector scores via RRF
- **`CrossEncoderReranker`** — Lazy-loaded cross-encoder for query-document pair scoring
- **`ContextAssembler`** — Builds token-budgeted context strings for LLM prompts; includes injection scanning (T-216)
- **`RAGPipeline`** — Orchestrates the above into a single `query()` call

## Tests

20 unit tests in `tests/test_rag_pipeline.py`:

```
TestBM25                   — tokenization, scoring, top-k, empty query
TestHybridSearcher         — RRF calculation, hybrid search, deduplication
TestCrossEncoderReranker   — initialization, mock model, empty results
TestContextAssembler       — templates, token budget, sources
TestRAGPipeline            — full pipeline, reranking, top-k
TestRAGPipelineIntegration — end-to-end without model loading
```

Run: `PYTHONPATH=src pytest tests/test_rag_pipeline.py -v`
