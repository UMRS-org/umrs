# Demo 27: Pipeline CRUD & RAG Profiles

> **Retired 2026-10-18** (ai4s-platform hardening H-10.1 follow-up): the
> `/umrs/pipelines` CRUD is deleted (definitions lived in process memory and
> no ingest or search read them) and `/umrs/rag-profiles` answers 410 since
> 2026-08-22. Only the component registry remains:
> `GET /umrs/components/{chunkers,embedders,rerankers}`. Chunking and
> embedding are site settings (`UMRS_RAG_CHUNK_MAX_TOKENS`,
> `UMRS_RAG_EMBEDDING_MODEL`). The steps below are kept for history only.

## Purpose

Demonstrates the UMRS pipeline management API — creating, listing,
updating, and deleting pipelines, plus RAG profile composition
(chunker + embedder + reranker) and the component registry.

## API Endpoints Covered

| Method | Path                                      | Description              |
|--------|-------------------------------------------|--------------------------|
| GET    | `/umrs/pipelines`                         | List pipelines           |
| POST   | `/umrs/pipelines`                         | Create pipeline          |
| GET    | `/umrs/pipelines/{id}`                    | Get pipeline             |
| PUT    | `/umrs/pipelines/{id}`                    | Update pipeline          |
| DELETE | `/umrs/pipelines/{id}`                    | Delete pipeline          |
| GET    | `/umrs/pipelines/rag-profiles`            | List RAG profiles        |
| POST   | `/umrs/pipelines/rag-profiles`            | Create RAG profile       |
| GET    | `/umrs/pipelines/components/chunkers`     | List chunkers            |
| GET    | `/umrs/pipelines/components/embedders`    | List embedders           |
| GET    | `/umrs/pipelines/components/rerankers`    | List rerankers           |

## Prerequisites

- UMRS HTTP sidecar running (`UMRS_HTTP_URL` or default `http://localhost:8001`)
- `curl` and `jq` installed
- A valid project ID (default: `default-project`)

## Step-by-Step Commands

### Step 1: List available pipeline components

```bash
UMRS_URL="${UMRS_HTTP_URL:-http://localhost:8001}"

# List available chunkers
curl -sf "${UMRS_URL}/umrs/components/chunkers" \
  -H "X-Project-ID: default-project" | jq .

# List available embedders
curl -sf "${UMRS_URL}/umrs/components/embedders" \
  -H "X-Project-ID: default-project" | jq .

# List available rerankers
curl -sf "${UMRS_URL}/umrs/components/rerankers" \
  -H "X-Project-ID: default-project" | jq .
```

### Step 2: List existing pipelines

```bash
curl -sf "${UMRS_URL}/umrs/pipelines" \
  -H "X-Project-ID: default-project" | jq .
```

**Expected output:**
```json
[]
```

### Step 3: Create a new pipeline

```bash
PIPELINE_ID=$(curl -sf -X POST "${UMRS_URL}/umrs/pipelines" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "name": "demo-text-pipeline",
    "description": "Demo text embedding pipeline",
    "steps": [
      {
        "name": "chunk",
        "type": "chunker",
        "config": {"chunk_size": 512, "overlap": 50}
      },
      {
        "name": "embed",
        "type": "embedder",
        "config": {"model": "text-embedding-3-small"}
      }
    ]
  }' | jq -r '.pipeline_id // .id // empty')

echo "[demo27] Created pipeline ID: ${PIPELINE_ID}"
```

### Step 4: Get pipeline details

```bash
curl -sf "${UMRS_URL}/umrs/pipelines/${PIPELINE_ID}" \
  -H "X-Project-ID: default-project" | jq .
```

### Step 5: Update the pipeline

```bash
curl -sf -X PATCH "${UMRS_URL}/umrs/pipelines/${PIPELINE_ID}" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "description": "Updated demo text embedding pipeline"
  }' | jq .
```

### Step 6: List RAG profiles

```bash
curl -sf "${UMRS_URL}/umrs/rag-profiles" \
  -H "X-Project-ID: default-project" | jq .
```

## Cleanup

```bash
curl -sf -X DELETE "${UMRS_URL}/umrs/pipelines/${PIPELINE_ID}" \
  -H "X-Project-ID: default-project"
echo "[demo27] Deleted pipeline: ${PIPELINE_ID}"
```
