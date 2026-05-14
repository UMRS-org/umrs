#!/usr/bin/env bash
# =============================================================================
# Demo 05: RAG Semantic Search
# =============================================================================
# Demonstrates the UMRS RAG search pipeline:
#   Step 1 – Check UMRS server + RAG service readiness
#   Step 2 – Verify embeddings exist (run 02_text_embedding first if needed)
#   Step 3 – Vector search (cosine similarity via pgvector)
#   Step 4 – Hybrid search (BM25 + vector with Reciprocal Rank Fusion)
#   Step 5 – Vector search + cross-encoder reranking
#   Step 6 – RAG Q&A (optional, requires LLM)
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000, 8081)
#   - UMRS server running
#   - RAG service enabled (port 8081)
#   - Embeddings computed (run demo 02 first)
#
# The RAG service listens on port 8081.
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="rag_search" source "${SCRIPT_DIR}/../lib/common.sh"

RAG_URL="${RAG_URL:-http://127.0.0.1:8081}"
RAG_BUCKET_ALIAS="${RAG_BUCKET_ALIAS:-${TEXT_BUCKET:-demo-textfiles}}"
RAG_MODALITY="${RAG_MODALITY:-text_file}"

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

build_payload() {
    local query="$1" top_k="$2" extra_json="${3:-}"
    QUERY="$query" \
    TOP_K="$top_k" \
    EXTRA_JSON="$extra_json" \
    RAG_BUCKET_ALIAS="$RAG_BUCKET_ALIAS" \
    RAG_MODALITY="$RAG_MODALITY" \
    python3 - <<'PY'
import json
import os

payload = {
    "query": os.environ["QUERY"],
    "top_k": int(os.environ["TOP_K"]),
}

bucket_alias = os.environ.get("RAG_BUCKET_ALIAS", "").strip()
if bucket_alias:
    payload["bucket_alias"] = bucket_alias

modality = os.environ.get("RAG_MODALITY", "").strip()
if modality:
    payload["modality"] = modality

extra_json = os.environ.get("EXTRA_JSON", "").strip()
if extra_json:
    payload.update(json.loads("{" + extra_json + "}"))

print(json.dumps(payload))
PY
}

rag_search() {
    local query="$1"
    shift
    local extra_json="${1:-}"

    local json_payload
    json_payload=$(build_payload "$query" 3 "$extra_json")

    log "  Query: \"$query\""
    local response
    response="$(curl -sf -X POST "${RAG_URL%/}/rag/search" \
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
    local extra_json="${1:-}"

    local json_payload
    json_payload=$(build_payload "$query" 5 "$extra_json")

    log "  Query: \"$query\""
    local response
    response="$(curl -sf -X POST "${RAG_URL%/}/rag/qa" \
        -H 'Content-Type: application/json' \
        -d "$json_payload" 2>&1)" || {
        warn "  QA request failed (is LLM/vLLM running for /rag/qa?)"
        return 1
    }
    echo "$response" | python3 -c "$PRETTY_PRINT_PY" 2>/dev/null || echo "    $response"
}

# ==========================================================================
banner "Demo 05: RAG Semantic Search"
echo "RAG service  : ${RAG_URL}"
echo "RAG bucket   : ${RAG_BUCKET_ALIAS}"
echo "RAG modality : ${RAG_MODALITY}"
echo ""
print_mini_rivault_mount_summary
echo ""

# ---------- Step 1: Check services ----------
log "Step 1: Checking UMRS server and RAG service..."
preflight

log "Checking RAG service at $RAG_URL ..."
attempts=5
for i in $(seq 1 "$attempts"); do
    if curl -sf "${RAG_URL%/}/docs" >/dev/null 2>&1; then
        ok "RAG service is ready"
        break
    fi
    if [[ "$i" -eq "$attempts" ]]; then
        die "RAG service not responding at $RAG_URL.
  Ensure the SSH tunnel for port 8081 is active and RAG is enabled."
    fi
    sleep 3
done

# ---------- Step 2: Verify embeddings ----------
log ""
log "Step 2: Verifying embeddings exist..."
log "  (Run demo 02 first if no embeddings are available)"
# A quick status check — if status output mentions embedding counts, great
$CLIENT status all 2>/dev/null | head -5 || true

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

rag_qa "Explain how machine learning is applied to geophysical data analysis, citing specific methods." \
    '"enable_reranking": true' || {
    log ""
    log "RAG Q&A requires a running LLM backend (vLLM via Mini-Rivault)."
    log "The /rag/search endpoints above work without an LLM."
}

# ==========================================================================
log ""
banner "Demo 05 Complete"
log "API endpoints: POST ${RAG_URL%/}/rag/search | POST ${RAG_URL%/}/rag/qa | GET ${RAG_URL%/}/docs"
