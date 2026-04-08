# Demo 05: RAG Semantic Search

Demonstrates UMRS RAG (Retrieval-Augmented Generation) search over embedded text data.

## Prerequisites

1. UMRS server running and reachable
2. Embeddings computed → `demos/02_text_embedding/`
3. RAG service enabled in the deployment environment before startup; on native site deployments, set `UMRS_RAG_ENABLE=true` in `deploy/<site>/.env` and restart the site stack with `./start.sh`.

## What it does

| Step | Description | Endpoint |
|------|-------------|----------|
| 1 | Check UMRS server + RAG service readiness | `ping` + `GET /docs` |
| 2 | Verify embeddings exist in pgvector | Direct DB query |
| 3 | **Vector search** — cosine similarity via pgvector HNSW | `POST /rag/search` |
| 4 | **Hybrid search** — BM25 + vector with Reciprocal Rank Fusion | `POST /rag/search` + `search_method: hybrid` |
| 5 | **Reranking** — vector search + cross-encoder reranking | `POST /rag/search` + `enable_reranking: true` |
| 6 | **RAG Q&A** — LLM-backed question answering (optional) | `POST /rag/qa` |

## Usage

```bash
bash run_demo.sh
```

## RAG API

The RAG service listens on port **8081**. On native site deployments it runs
as a managed login-node process; on containerized local dev it may run inside
the compose stack.

### `/rag/search` — Semantic search (no LLM needed)

```bash
curl -s -X POST http://localhost:8081/rag/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "seismic wave propagation methods", "top_k": 5}'
```

Parameters:
- `query` — natural language search query
- `top_k` — number of results (default: 5)
- `search_method` — `"vector"` (default) or `"hybrid"` (BM25 + vector RRF)
- `enable_reranking` — `true` to apply cross-encoder reranking
- `user`, `bucket_alias`, `xp_name` — optional filters to scope the search

### `/rag/qa` — Question answering (requires LLM)

```bash
curl -s -X POST http://localhost:8081/rag/qa \
  -H 'Content-Type: application/json' \
  -d '{"query": "How is ML used in geophysics?", "top_k": 5}'
```

Additional parameters:
- `system_prompt` — custom system prompt for the LLM
- `max_context_chunks` — max retrieved chunks to feed to LLM (default: 8)
- `enable_query_rewrite` — LLM rewrites query into sub-queries before retrieval (default: true)

> **Note**: `/rag/qa` requires a running LLM backend (vLLM via Mini-Rivault).
> The `/rag/search` endpoints work without an LLM.

### Interactive docs

Open `http://localhost:8081/docs` in a browser (or via port-forward) for the Swagger UI.

## Architecture

```
User query
  │
  ▼
RAG Service (port 8081)
  │
  ├─ /rag/search ─── Embed query (SentenceTransformer or vLLM)
  │                    │
  │                    ├─ Vector search (pgvector cosine HNSW)
  │                    ├─ BM25 search (hybrid mode)
  │                    └─ Cross-encoder rerank (optional)
  │                    │
  │                    └─► Ranked results
  │
  └─ /rag/qa ──────── Embed + retrieve (same as /search)
                       │
                       ├─ Query rewrite (LLM generates sub-queries)
                       ├─ Multi-query retrieval
                       └─ LLM generates answer with context
                       │
                       └─► Answer + citations
```
