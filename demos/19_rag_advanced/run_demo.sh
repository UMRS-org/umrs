#!/usr/bin/env bash
# =============================================================================
# Demo 19: RAG Advanced — Search Mode Comparison
# =============================================================================
# Demonstrates all RAG retrieval modes and compares results:
#   1. Vector-only search
#   2. BM25 keyword search
#   3. Hybrid search (RRF fusion)
#   4. Reranked hybrid search
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000, 8081)
#   - UMRS server running with RAG service on port 8081
#   - Text data already embedded (run Demo 02 first)
#   - umrs-client installed and in PATH
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="rag_adv" source "${SCRIPT_DIR}/../lib/common.sh"

RAG_URL="${RAG_URL:-http://localhost:8081}"
RESULTS_DIR="${SCRIPT_DIR}/results"
QUERY="${1:-What is the main topic of the documents?}"
RAG_USER="${RAG_USER:-user1}"
RAG_BUCKET_ALIAS="${RAG_BUCKET_ALIAS:-user1-textfiles}"
RAG_MODALITY="${RAG_MODALITY:-text_file}"
RAG_XP_NAME="${RAG_XP_NAME:-}"

banner "Demo 19: RAG Advanced — Search Mode Comparison"
echo "RAG endpoint : ${RAG_URL}"
echo "Query        : ${QUERY}"
echo "User         : ${RAG_USER}"
echo "Bucket alias : ${RAG_BUCKET_ALIAS}"
echo "Modality     : ${RAG_MODALITY}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# Check RAG service
if curl -sf "${RAG_URL}/docs" >/dev/null 2>&1; then
    ok "RAG service reachable at ${RAG_URL}"
else
    die "RAG service not reachable at ${RAG_URL} — is the tunnel up?"
fi

mkdir -p "$RESULTS_DIR"

# ---------- Helper function ----------
rag_search() {
    local mode="$1"
    local label="$2"
    local extra_params="$3"

    log ""
    log "--- ${label} ---"

    local payload
    payload=$(QUERY="$QUERY" \
        RAG_USER="$RAG_USER" \
        RAG_BUCKET_ALIAS="$RAG_BUCKET_ALIAS" \
        RAG_MODALITY="$RAG_MODALITY" \
        RAG_XP_NAME="$RAG_XP_NAME" \
        EXTRA_JSON="$extra_params" \
        python3 - <<'PYEOF'
import json
import os

payload = {
    "query": os.environ["QUERY"],
    "top_k": 3,
    "user": os.environ["RAG_USER"],
    "bucket_alias": os.environ["RAG_BUCKET_ALIAS"],
    "modality": os.environ["RAG_MODALITY"],
}

xp_name = os.environ.get("RAG_XP_NAME", "").strip()
if xp_name:
    payload["xp_name"] = xp_name

extra_json = os.environ.get("EXTRA_JSON", "").strip()
if extra_json:
    payload.update(json.loads("{" + extra_json.lstrip(', ').strip() + "}"))

print(json.dumps(payload))
PYEOF
    )

    local response
    response=$(curl -sf -X POST "${RAG_URL}/rag/search" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>&1) || {
        warn "  ${label} request failed"
        echo "{}" > "${RESULTS_DIR}/${mode}.json"
        return 1
    }

    echo "$response" | python3 -m json.tool > "${RESULTS_DIR}/${mode}.json" 2>/dev/null || true
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    results = data.get('results', data.get('documents', []))
    if not results:
        print('  (no results)')
    for i, r in enumerate(results[:3]):
        score = r.get('score', r.get('similarity', 'N/A'))
        text = r.get('text', r.get('content', r.get('document', '')))[:80]
        print(f'  {i+1}. [score={score:.4f}] {text}...' if isinstance(score, float) else f'  {i+1}. [score={score}] {text}...')
except Exception as e:
    print(f'  (parse error: {e})')
" 2>/dev/null || log "  (could not parse response)"
    ok "${label} complete"
}

# ---------- Step 2: Vector-only search ----------
rag_search "vector" "Vector-Only Search" ""

# ---------- Step 3: Hybrid search (RRF) ----------
rag_search "hybrid" "Hybrid Search (RRF Fusion)" ', "search_method": "hybrid"'

# ---------- Step 4: Reranked hybrid search ----------
rag_search "reranked" "Reranked Hybrid Search" ', "search_method": "hybrid", "enable_reranking": true'

# ---------- Step 5: Compare results ----------
log ""
log "Step 5: Comparing retrieval modes..."

python3 - "$RESULTS_DIR" << 'PYEOF'
import json, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
modes = ["vector", "hybrid", "reranked"]
labels = ["Vector-Only", "Hybrid (RRF)", "Reranked"]

print("\n╔══════════════════════════════════════════════╗")
print("║       Search Mode Comparison Summary         ║")
print("╠══════════════════════════════════════════════╣")

for mode, label in zip(modes, labels):
    path = results_dir / f"{mode}.json"
    if not path.exists():
        print(f"║ {label:20s} │ No results           ║")
        continue
    try:
        data = json.loads(path.read_text())
        results = data.get("results", data.get("documents", []))
        count = len(results)
        if results:
            scores = [r.get("score", r.get("similarity", 0)) for r in results]
            avg = sum(s for s in scores if isinstance(s, (int, float))) / max(len(scores), 1)
            print(f"║ {label:20s} │ {count} results, avg={avg:.4f} ║")
        else:
            print(f"║ {label:20s} │ {count} results             ║")
    except Exception:
        print(f"║ {label:20s} │ Parse error            ║")

print("╚══════════════════════════════════════════════╝")
PYEOF

ok "Comparison complete"

log ""
log "Detailed results saved to: ${RESULTS_DIR}/"

banner "Demo 19 Complete"
