#!/usr/bin/env bash
# =============================================================================
# RAG Semantic Search Demo — hpbd-atom deployment
# =============================================================================
# Runs vector search, hybrid search, and Q&A against embeddings
# already stored in pgvector.
#
# Prerequisites:
#   - SSH tunnels running (ssh_tunnel.sh)
#   - Embeddings computed (01_text_embedding.sh)
#   - RAG service enabled on the server (UMRS_RAG_ENABLE=1)
# =============================================================================
set -euo pipefail

USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
RAG_URL="${UMRS_RAG_URL:-http://localhost:8081}"

log()  { echo "[rag] $*"; }
warn() { echo "[rag] WARNING: $*"; }
die()  { echo "[rag] ERROR: $*" >&2; exit 1; }

# ---------- Pretty-printer ----------
PRETTY='
import json, sys, textwrap
try:
    data = json.load(sys.stdin)
    answer = data.get("answer")
    results = data.get("results", data.get("contexts", []))
    if answer:
        print("\n    Answer:")
        for l in textwrap.wrap(answer, 76): print("      " + l)
        print("\n    Contexts:")
    if not results:
        print("    (no results)")
        sys.exit(0)
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        content = r.get("content", "")[:200]
        meta = r.get("metadata", {})
        parts = []
        for k in ("hybrid_score","bm25_score","vector_score","rerank_score"):
            v = meta.get(k)
            if v is not None: parts.append("%s=%.4f" % (k.replace("_score",""), v))
        detail = "  (%s)" % ", ".join(parts) if parts else ""
        print("    #%d  score=%.4f%s" % (i, score, detail))
        for l in textwrap.wrap(content, 76): print("        " + l)
        if i < len(results): print()
except Exception as e:
    print("    (parse error: %s)" % e, file=sys.stderr)
'

rag_search() {
    local query="$1"; shift
    local extra=""
    [[ $# -gt 0 ]] && extra=", $*"

    local payload
    payload=$(python3 -c "
import json, sys
p = json.loads('{\"query\": \"\", \"top_k\": 3 $extra}')
p['query'] = sys.argv[1]
print(json.dumps(p))
" "$query")

    log "  Query: \"$query\""
    local resp
    resp="$(curl -sf -X POST "$RAG_URL/rag/search" \
        -H 'Content-Type: application/json' \
        -d "$payload" 2>&1)" || { warn "  request failed"; return 1; }
    echo "$resp" | python3 -c "$PRETTY" 2>/dev/null || echo "    $resp"
}

rag_qa() {
    local query="$1"; shift
    local extra=""
    [[ $# -gt 0 ]] && extra=", $*"

    local payload
    payload=$(python3 -c "
import json, sys
p = json.loads('{\"query\": \"\", \"top_k\": 5 $extra}')
p['query'] = sys.argv[1]
print(json.dumps(p))
" "$query")

    log "  Query: \"$query\""
    local resp
    resp="$(curl -sf -X POST "$RAG_URL/rag/qa" \
        -H 'Content-Type: application/json' \
        -d "$payload" 2>&1)" || { warn "  QA failed (is LLM running for /rag/qa?)"; return 1; }
    echo "$resp" | python3 -c "$PRETTY" 2>/dev/null || echo "    $resp"
}

# ========== Step 1: Preflight ==========
log "Step 1: Checking services..."
umrs-client -c "$USER_CONF" ping >/dev/null || die "UMRS server not reachable. Is ssh_tunnel.sh running?"
log "  UMRS server: OK"

curl -sf "$RAG_URL/docs" >/dev/null 2>&1 || die "RAG service not reachable at $RAG_URL"
log "  RAG service:  OK"

# ========== Step 2: Vector search ==========
echo
log "Step 2: Vector search (cosine similarity)"
echo
rag_search "What numerical methods are used for seismic wave simulation?"
echo
rag_search "How do P-waves and S-waves differ?"

# ========== Step 3: Hybrid search ==========
echo
log "Step 3: Hybrid search (BM25 + vector, RRF)"
echo
rag_search "GPU acceleration for HPC simulations" '"search_method": "hybrid"'
echo
rag_search "What is full waveform inversion?" '"search_method": "hybrid"'

# ========== Step 4: Q&A (requires LLM) ==========
echo
log "Step 4: Q&A with retrieval-augmented generation"
log "(requires a running LLM backend — will skip gracefully if unavailable)"
echo
rag_qa "Explain how containerization helps reproducibility on HPC clusters." || true

echo
log "Done!"
