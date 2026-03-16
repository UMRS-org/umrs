#!/usr/bin/env bash
# =============================================================================
# RAG Semantic Search Demo
# =============================================================================
# Demonstrates the UMRS RAG search pipeline:
#   Step 1 – Check UMRS server + RAG service readiness
#   Step 2 – Ensure embeddings exist (run 01_text_embedding first if not)
#   Step 3 – Vector search (cosine similarity via pgvector)
#   Step 4 – Hybrid search (BM25 + vector with Reciprocal Rank Fusion)
#   Step 5 – Vector search + cross-encoder reranking
#   Step 6 – Custom queries (interactive or batch)
#
# Prerequisites:
#   - Services running (demos/00_start_services/)
#   - Embeddings computed (demos/01_text_embedding/)
#   - RAG service enabled: UMRS_RAG_ENABLE=1 in .env
#
# The RAG service runs inside ai-login on port 8081.
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

AI_LOGIN_CONTAINER="ai-login"
UMRS_CLIENT_CONTAINER="umrs-client"
PG_CONTAINER="demo-postgres"
PG_SUPERUSER="${POSTGRES_USER:-umrs}"
RAG_URL="http://localhost:8081"

LOG_PREFIX="[rag_search]"
log() { echo "$LOG_PREFIX $*"; }
warn() { echo "$LOG_PREFIX WARNING: $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }
run_cmd() {
    echo "$LOG_PREFIX \$ $*"
    "$@"
}

# ---------- Pretty-printer for RAG results ----------
PRETTY_PRINT_PY='
import json, sys, textwrap
try:
    data = json.load(sys.stdin)
    results = data.get("results", data.get("contexts", []))
    answer = data.get("answer")
    rewritten = data.get("rewritten_queries", [])

    if answer:
        print()
        print("    Answer:")
        for line in textwrap.wrap(answer, width=76):
            print("      %s" % line)
        if rewritten:
            print()
            print("    Rewritten queries:")
            for q in rewritten:
                print("      - %s" % q)
        print()
        print("    Retrieved contexts:")

    if not results:
        print("    (no results)")
        sys.exit(0)

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        content = r.get("content", "")[:200]
        meta = r.get("metadata", {})
        parts = []
        for key in ("hybrid_score", "bm25_score", "vector_score", "rerank_score"):
            v = meta.get(key)
            if v is not None:
                parts.append("%s=%.4f" % (key.replace("_score",""), v))
        detail = "  (%s)" % ", ".join(parts) if parts else ""
        print("    #%d  score=%.4f%s" % (i, score, detail))
        for line in textwrap.wrap(content, width=76):
            print("        %s" % line)
        if i < len(results):
            print()
except Exception as e:
    print("    (parse error: %s)" % e, file=sys.stderr)
    sys.exit(1)
'

rag_search() {
    local query="$1"
    shift
    # Remaining args are extra JSON fields (e.g., '"search_method": "hybrid"')
    local extra=""
    if [[ $# -gt 0 ]]; then
        extra=", $*"
    fi

    local json_payload
    json_payload=$(python3 -c "
import json, sys
payload = json.loads('{\"query\": \"\", \"top_k\": 3 $extra}')
payload['query'] = sys.argv[1]
print(json.dumps(payload))
" "$query")

    log "  Query: \"$query\""
    log "  \$ curl -s -X POST $RAG_URL/rag/search -H 'Content-Type: application/json' -d '...'"
    local response
    response="$(podman exec "$AI_LOGIN_CONTAINER" curl -sf -X POST "$RAG_URL/rag/search" \
        -H 'Content-Type: application/json' \
        -d "$json_payload" 2>&1)" || {
        warn "  Search request failed"
        return 1
    }
    echo "$response" | python3 -c "$PRETTY_PRINT_PY" 2>/dev/null || echo "    $response"
}

rag_qa() {
    local query="$1"
    shift
    local extra=""
    if [[ $# -gt 0 ]]; then
        extra=", $*"
    fi

    local json_payload
    json_payload=$(python3 -c "
import json, sys
payload = json.loads('{\"query\": \"\", \"top_k\": 5 $extra}')
payload['query'] = sys.argv[1]
print(json.dumps(payload))
" "$query")

    log "  Query: \"$query\""
    log "  \$ curl -s -X POST $RAG_URL/rag/qa -H 'Content-Type: application/json' -d '...'"
    local response
    response="$(podman exec "$AI_LOGIN_CONTAINER" curl -sf -X POST "$RAG_URL/rag/qa" \
        -H 'Content-Type: application/json' \
        -d "$json_payload" 2>&1)" || {
        warn "  QA request failed (is LLM/vLLM running for /rag/qa?)"
        return 1
    }
    echo "$response" | python3 -c "$PRETTY_PRINT_PY" 2>/dev/null || echo "    $response"
}

# ==========================================================================
# Step 1: Check services
# ==========================================================================
log "Step 1: Checking UMRS server and RAG service..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client ping || \
    die "UMRS server not reachable. Start services first: bash ../00_start_services/start_all.sh"
log "UMRS server is ready."

log "Checking RAG service at $RAG_URL ..."
attempts=10
for i in $(seq 1 "$attempts"); do
    if podman exec "$AI_LOGIN_CONTAINER" curl -sf "$RAG_URL/docs" >/dev/null 2>&1; then
        log "RAG service is ready."
        break
    fi
    if [[ "$i" -eq "$attempts" ]]; then
        die "RAG service not responding at $RAG_URL. Ensure UMRS_RAG_ENABLE=1 in .env and restart services."
    fi
    sleep 3
done

# ==========================================================================
# Step 2: Verify embeddings exist
# ==========================================================================
log ""
log "Step 2: Verifying embeddings exist in pgvector..."

emb_count="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -tAc \
    "SELECT sum(n) FROM (
        SELECT (SELECT count(*) FROM information_schema.columns
                WHERE column_name='embedding' AND table_schema='public'
                AND table_catalog=datname) as n
        FROM pg_database
        WHERE datname NOT IN ('postgres','template0','template1','umrs')
    ) sub;" 2>/dev/null || echo 0)"
emb_count="$(echo "$emb_count" | tr -d ' ')"

if [[ -z "$emb_count" || "$emb_count" == "0" || "$emb_count" == "" ]]; then
    # Try a more direct check
    user_dbs="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -tAc \
        "SELECT datname FROM pg_database WHERE datname NOT IN ('postgres','template0','template1','umrs');" 2>/dev/null || true)"
    total=0
    for db in $user_dbs; do
        tables="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
            "SELECT table_name FROM information_schema.columns WHERE column_name='embedding' AND table_schema='public';" 2>/dev/null || true)"
        for tbl in $tables; do
            n="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
                "SELECT count(*) FROM \"$tbl\" WHERE embedding IS NOT NULL;" 2>/dev/null || echo 0)"
            total=$((total + $(echo "$n" | tr -d ' ')))
        done
    done
    emb_count="$total"
fi

if [[ "$emb_count" -gt 0 ]]; then
    log "Found $emb_count embeddings. Ready for RAG search."
else
    die "No embeddings found. Run demos/01_text_embedding/ first."
fi

# ==========================================================================
# Step 3: Vector search (cosine similarity)
# ==========================================================================
log ""
log "Step 3: Vector search (cosine similarity via pgvector HNSW)"
log ""

rag_search "What machine learning methods are used for fault detection in seismic data?"

echo
rag_search "How do seismic P-waves and S-waves differ?"

# ==========================================================================
# Step 4: Hybrid search (BM25 + vector RRF)
# ==========================================================================
log ""
log "Step 4: Hybrid search (BM25 + vector with Reciprocal Rank Fusion)"
log ""

rag_search "What is the finite difference method for wave propagation?" \
    '"search_method": "hybrid"'

echo
rag_search "How does GPU acceleration improve HPC simulations?" \
    '"search_method": "hybrid"'

# ==========================================================================
# Step 5: Vector search + cross-encoder reranking
# ==========================================================================
log ""
log "Step 5: Vector search + cross-encoder reranking"
log ""

rag_search "What is RiVault and how does it provide secure inference?" \
    '"enable_reranking": true'

echo
rag_search "Compare numerical methods for seismic modeling" \
    '"enable_reranking": true'

# ==========================================================================
# Step 6: RAG Q&A (requires LLM — optional)
# ==========================================================================
log ""
log "Step 6: RAG Q&A with LLM (optional — requires vLLM running)"
log ""
log "Attempting /rag/qa endpoint..."

rag_qa "Explain how machine learning is applied to geophysical data analysis, citing specific methods." \
    '"enable_reranking": true' || {
    log ""
    log "RAG Q&A requires a running LLM backend (vLLM via Mini-Rivault)."
    log "The /rag/search endpoints above work without an LLM."
    log ""
    log "To enable Q&A, ensure Mini-Rivault is running with a generative model."
}

# ==========================================================================
# Summary
# ==========================================================================
log ""
log "RAG search demo complete!"
log ""
log "API endpoints available at $RAG_URL:"
log "  POST /rag/search  — semantic search (vector, hybrid, reranking)"
log "  POST /rag/qa      — question answering with LLM (needs vLLM)"
log "  GET  /docs        — interactive API documentation"
log ""
log "Example curl:"
log "  podman exec ai-login curl -s -X POST $RAG_URL/rag/search \\"
log "    -H 'Content-Type: application/json' \\"
log "    -d '{\"query\": \"your question here\", \"top_k\": 5}'"
