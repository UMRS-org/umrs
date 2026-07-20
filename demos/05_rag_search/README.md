# Demo 05: RAG Semantic Search

Demonstrates UMRS RAG (Retrieval-Augmented Generation) search over embedded text data.

The demo now also prints the mini-RiVault RAG mount shape used by endpoint
provisioning: project data at `/project` and a prebuilt index at `/rag-index`.

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
# See the numbered steps below in this README
```

By default the demo scopes searches to Demo 02's text-file corpus by sending:

- `bucket_alias=user1-textfiles`
- `modality=text_file`

Override those defaults when needed:

```bash
RAG_BUCKET_ALIAS=my-bucket RAG_MODALITY=text
# Use your own bucket and modality
```

## Running Manually

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/05_rag_search"

RAG_URL="${RAG_URL:-http://127.0.0.1:8081}"
RAG_BUCKET_ALIAS="${RAG_BUCKET_ALIAS:-user1-textfiles}"
RAG_MODALITY="${RAG_MODALITY:-text_file}"

# 1. Verify UMRS and RAG connectivity.
umrs ping
curl -sf "${RAG_URL%/}/docs" >/dev/null && echo "RAG docs OK"

# 2. Check that embeddings already exist (run Demo 02 first if needed).
umrs status all | head -5 || true

# 3. Vector search (same style as the script).
curl -sf -X POST "${RAG_URL%/}/rag/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"What machine learning methods are used for fault detection in seismic data?\",\"top_k\":3,\"bucket_alias\":\"$RAG_BUCKET_ALIAS\",\"modality\":\"$RAG_MODALITY\"}" \
  | python3 -m json.tool

curl -sf -X POST "${RAG_URL%/}/rag/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"How do seismic P-waves and S-waves differ?\",\"top_k\":3,\"bucket_alias\":\"$RAG_BUCKET_ALIAS\",\"modality\":\"$RAG_MODALITY\"}" \
  | python3 -m json.tool

# 4. Hybrid search (BM25 + vector fusion).
curl -sf -X POST "${RAG_URL%/}/rag/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"What is the finite difference method for wave propagation?\",\"top_k\":3,\"bucket_alias\":\"$RAG_BUCKET_ALIAS\",\"modality\":\"$RAG_MODALITY\",\"search_method\":\"hybrid\"}" \
  | python3 -m json.tool

curl -sf -X POST "${RAG_URL%/}/rag/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"How does GPU acceleration improve HPC simulations?\",\"top_k\":3,\"bucket_alias\":\"$RAG_BUCKET_ALIAS\",\"modality\":\"$RAG_MODALITY\",\"search_method\":\"hybrid\"}" \
  | python3 -m json.tool

# 5. Vector search + reranking.
curl -sf -X POST "${RAG_URL%/}/rag/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"What is RiVault and how does it provide secure inference?\",\"top_k\":3,\"bucket_alias\":\"$RAG_BUCKET_ALIAS\",\"modality\":\"$RAG_MODALITY\",\"enable_reranking\":true}" \
  | python3 -m json.tool

curl -sf -X POST "${RAG_URL%/}/rag/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"Compare numerical methods for seismic modeling\",\"top_k\":3,\"bucket_alias\":\"$RAG_BUCKET_ALIAS\",\"modality\":\"$RAG_MODALITY\",\"enable_reranking\":true}" \
  | python3 -m json.tool

# 6. Optional RAG Q&A (requires a chat-capable LLM backend).
curl -sf -X POST "${RAG_URL%/}/rag/qa" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"Explain how machine learning is applied to geophysical data analysis, citing specific methods.\",\"top_k\":5,\"bucket_alias\":\"$RAG_BUCKET_ALIAS\",\"modality\":\"$RAG_MODALITY\",\"enable_reranking\":true}" \
  | python3 -m json.tool
```

## RAG API

The RAG service listens on port **8081**. On native site deployments it runs
as a managed login-node process; on containerized local dev it may run inside
the compose stack.

### `/rag/search` — Semantic search (no LLM needed)

```bash
curl -s -X POST "${RAG_URL%/}/rag/search" \
  -H 'Content-Type: application/json' \
  -d '{"query": "seismic wave propagation methods", "top_k": 5, "bucket_alias": "user1-textfiles", "modality": "text_file"}'
```

Parameters:
- `query` — natural language search query
- `top_k` — number of results (default: 5)
- `search_method` — `"vector"` (default) or `"hybrid"` (BM25 + vector RRF)
- `enable_reranking` — `true` to apply cross-encoder reranking
- `user`, `bucket_alias`, `xp_name` — optional filters to scope the search

### `/rag/qa` — Question answering (requires LLM)

```bash
curl -s -X POST "${RAG_URL%/}/rag/qa" \
  -H 'Content-Type: application/json' \
  -d '{"query": "How is ML used in geophysics?", "top_k": 5, "bucket_alias": "user1-textfiles", "modality": "text_file"}'
```

Additional parameters:
- `system_prompt` — custom system prompt for the LLM
- `max_context_chunks` — max retrieved chunks to feed to LLM (default: 8)
- `enable_query_rewrite` — LLM rewrites query into sub-queries before retrieval (default: true)

> **Note**: `/rag/qa` requires a chat-capable LLM backend (for example, a
> Mini-RiVault stack whose LiteLLM `/v1/models` endpoint exposes an
> instruction/chat model). On multi-experience deployments you should also
> scope requests with `bucket_alias` and `modality`, otherwise retrieval may
> fail with an ambiguity error. The `/rag/search` endpoints still work with an
> embedding-only backend.

### Interactive docs

Open `${RAG_URL%/}/docs` in a browser (or via port-forward) for the Swagger UI.

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
