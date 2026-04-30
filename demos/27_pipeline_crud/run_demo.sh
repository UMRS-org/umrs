#!/usr/bin/env bash
# =============================================================================
# Demo 27: Pipeline CRUD & RAG Profiles
# =============================================================================
# Shows: /umrs/pipelines CRUD, RAG profiles, component registry
#
# Prerequisites:
#   - UMRS server reachable via HTTP (default: localhost:8001)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="pipe_crud" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-http://localhost:8001}"

banner "Demo 27: Pipeline CRUD & RAG Profiles"
echo ""

log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: List pipelines
log "Step 1: List pipelines (GET /umrs/pipelines)"
echo ""
PIPELINES=$(curl -sf "${UMRS_URL}/umrs/pipelines" 2>/dev/null || echo '{"pipelines":[]}')
echo "$PIPELINES" | python3 -m json.tool 2>/dev/null || echo "$PIPELINES"
echo ""

# Step 2: Create a pipeline
log "Step 2: Create a pipeline (POST /umrs/pipelines)"
echo ""
echo "  Request body:"
echo '  {
    "name": "text-analysis",
    "description": "Analyze text documents with chunking and embedding",
    "steps": [
      {"name": "chunk", "type": "chunker", "config": {"method": "sentence", "max_tokens": 512}},
      {"name": "embed", "type": "embedder", "config": {"model": "all-MiniLM-L6-v2"}}
    ]
  }'
echo ""

# Step 3: RAG Profiles
log "Step 3: RAG profile management"
echo ""
echo "  RAG profiles compose: chunker + embedder + reranker"
echo ""
echo "  # List profiles"
echo "  GET /umrs/pipelines/rag-profiles"
echo ""
echo "  # Create profile"
echo "  POST /umrs/pipelines/rag-profiles"
echo '  {
    "name": "scientific-papers",
    "chunker": {"method": "sentence", "max_tokens": 512, "overlap": 50},
    "embedder": {"model": "all-MiniLM-L6-v2"},
    "reranker": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_k": 5}
  }'
echo ""

PROFILES=$(curl -sf "${UMRS_URL}/umrs/pipelines/rag-profiles" 2>/dev/null || echo '{"profiles":[]}')
echo "  Current profiles:"
echo "$PROFILES" | python3 -m json.tool 2>/dev/null || echo "$PROFILES"
echo ""

# Step 4: Component registry
log "Step 4: Available components"
echo ""
echo "  # List available chunkers"
echo "  GET /umrs/pipelines/components/chunkers"
echo ""
echo "  # List available embedders"
echo "  GET /umrs/pipelines/components/embedders"
echo ""
echo "  # List available rerankers"
echo "  GET /umrs/pipelines/components/rerankers"
echo ""

for component in chunkers embedders rerankers; do
    echo "  ${component}:"
    COMPS=$(curl -sf "${UMRS_URL}/umrs/pipelines/components/${component}" 2>/dev/null || echo '[]')
    echo "$COMPS" | python3 -m json.tool 2>/dev/null || echo "    (not available)"
    echo ""
done

# Step 5: r4 CLI equivalents
log "Step 5: r4 CLI equivalents"
echo ""
echo "  # RAG ingestion pipeline management"
echo "  r4 rag ingest list"
echo "  r4 rag ingest create --name 'text-analysis' --config pipeline.yaml"
echo "  r4 rag ingest show pipeline-id"
echo "  r4 rag ingest update pipeline-id --config updated.yaml"
echo "  r4 rag ingest delete pipeline-id"
echo ""
echo "  # RAG retrieval profile management"
echo "  r4 rag retriever list"
echo "  r4 rag retriever create --name 'scientific-papers' --config profile.yaml"
echo "  r4 rag retriever components chunkers"
echo "  r4 rag retriever components embedders"
echo "  r4 rag retriever components rerankers"
echo ""

banner "Demo 27 Complete"
