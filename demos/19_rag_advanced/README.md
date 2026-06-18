# Demo 19: RAG Advanced — Search Mode Comparison

Compares the RAG search modes that are currently exposed by the API:
**vector**, **hybrid (RRF)**, and **reranked hybrid**. Requires embedded text
data (run Demo 02 first).

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556, 9000, 8081)
- RAG service enabled and running on port 8081
- Text data already embedded (run [Demo 02](../02_text_embedding/README.md) first)
- `umrs-client` installed and in PATH
- `curl` and `python3` available

## Quick Start

```bash
cd demos/19_rag_advanced
# Follow the step-by-step commands in README.md
```

With a custom query:

```bash
# Follow the step-by-step commands in README.md "How is machine learning used in materials science?"
```

## What This Demo Does

1. **Pre-flight** — verifies server and RAG service connectivity
2. **Vector-only search** — cosine similarity search via pgvector HNSW index
3. **Hybrid search** — combines vector + BM25 using Reciprocal Rank Fusion
4. **Reranked hybrid search** — hybrid search followed by cross-encoder reranking
5. **Compare results** — prints a summary table comparing all modes

## Running Manually

This demo queries the RAG service directly via `curl` rather than through
`umrs-client`, since the RAG API provides the fine-grained search control.

```bash
# 1. Verify server and RAG service connectivity
umrs-client ping
RAG_URL="${RAG_URL:-http://127.0.0.1:8081}"
curl -sf "${RAG_URL%/}/docs" >/dev/null && echo "RAG service OK"

# 2. Vector-only search
#    Uses pgvector's HNSW index for cosine similarity.
#    Best for semantic meaning — finds conceptually similar content
#    even when exact keywords don't match.
curl -s -X POST "${RAG_URL%/}/rag/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?", "top_k": 3, "user": "user1", "bucket_alias": "user1-textfiles", "modality": "text_file"}'

# 3. Hybrid search (Reciprocal Rank Fusion)
#    Combines vector and BM25 results using RRF to get the best of
#    both semantic and lexical matching. Usually provides the most
#    balanced results.
curl -s -X POST "${RAG_URL%/}/rag/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?", "top_k": 3, "user": "user1", "bucket_alias": "user1-textfiles", "modality": "text_file", "search_method": "hybrid"}'

# 4. Reranked hybrid search
#    Starts with hybrid search, then applies a cross-encoder model
#    to rerank the results. Most accurate but slower due to the
#    additional reranking pass.
curl -s -X POST "${RAG_URL%/}/rag/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?", "top_k": 3, "user": "user1", "bucket_alias": "user1-textfiles", "modality": "text_file", "search_method": "hybrid", "enable_reranking": true}'
```

## Search Modes Compared

| Mode | Method | Strengths | Trade-offs |
|------|--------|-----------|------------|
| **Vector** | Cosine similarity (pgvector HNSW) | Semantic understanding, concept matching | May miss exact terms |
| **Hybrid** | RRF fusion of vector + BM25 | Balanced semantic + lexical | Slightly slower than single mode |
| **Reranked** | Hybrid + cross-encoder reranking | Most accurate ranking | Slowest, requires reranker model |

## RAG API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rag/search` | POST | Semantic search (no LLM needed) |
| `/rag/qa` | POST | Question answering (requires LLM) |
| `/docs` | GET | Interactive Swagger documentation |

### `/rag/search` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *(required)* | Natural language search query |
| `top_k` | int | `5` | Number of results to return |
| `search_method` | string | `"vector"` | `"vector"` or `"hybrid"` |
| `enable_reranking` | bool | `false` | Apply cross-encoder reranking |
| `user` | string | *(all)* | Filter by user |
| `bucket_alias` | string | *(all)* | Filter by bucket |
| `modality` | string | `"text"` | Filter by modality, e.g. `text_file` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_URL` | profile-derived, default `http://127.0.0.1:8081` | RAG service endpoint |
| `RAG_USER` | `user1` | User scope for retrieval |
| `RAG_BUCKET_ALIAS` | `user1-textfiles` | Bucket alias used for retrieval |
| `RAG_MODALITY` | `text_file` | Modality used for retrieval |
| `RAG_XP_NAME` | *(unset)* | Optional experience filter |

## Expected Output

```
╔══════════════════════════════════════════════╗
║       Search Mode Comparison Summary         ║
╠══════════════════════════════════════════════╣
║ Vector-Only          │ 3 results, avg=0.8234 ║
║ Hybrid (RRF)         │ 3 results, avg=0.8456 ║
║ Reranked             │ 3 results, avg=0.8901 ║
╚══════════════════════════════════════════════╝
```

## See Also

- [Demo 05: RAG Search](../05_rag_search/README.md) — basic RAG search and Q&A
- [Demo 02: Text Embedding](../02_text_embedding/README.md) — embedding pipeline (prerequisite)
- [RAG Implementation](../../doc/rag_implementation.md) — architecture details
