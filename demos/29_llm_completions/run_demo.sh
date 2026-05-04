#!/usr/bin/env bash
# =============================================================================
# Demo 29: LLM Completions (OpenAI-Compatible)
# =============================================================================
# Shows: /umrs/llm/v1/chat/completions, /v1/completions,
#        /v1/embeddings, /v1/models
#
# Prerequisites:
#   - UMRS server with LLM backend configured (OpenAI-compat or Mini-RiVault)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="llm_api" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-${UMRS_HTTP_URL:-http://localhost:8001}}"
LLM_MODEL="${LLM_MODEL:-rivault-gpt-oss-static}"
PROJECT_ID="${UMRS_PROJECT_ID:-demo29}"
USER_ID="${UMRS_USER_ID:-demo-user}"

banner "Demo 29: LLM Completions (OpenAI-Compatible API)"
echo ""

log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: List models
log "Step 1: List available models (GET /umrs/llm/v1/models)"
echo ""
MODELS=$(curl -sf \
  -H "X-Project-ID: ${PROJECT_ID}" \
  -H "X-User-ID: ${USER_ID}" \
  "${UMRS_URL}/umrs/llm/v1/models" 2>/dev/null || echo '{"data":[]}')
echo "$MODELS" | python3 -m json.tool 2>/dev/null || echo "$MODELS"
echo ""

# Step 2: Chat completion
log "Step 2: Chat completion (POST /umrs/llm/v1/chat/completions)"
echo ""
echo "  Request:"
echo '  {
    "model": "rivault-gpt-oss-static",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0
  }'
echo ""

CHAT_PAYLOAD=$(LLM_MODEL="$LLM_MODEL" python3 - <<'PY'
import json
import os

print(json.dumps({
  "model": os.environ["LLM_MODEL"],
  "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
  "max_tokens": 10,
  "temperature": 0,
}))
PY
)

RESPONSE=$(curl -sf -X POST "${UMRS_URL}/umrs/llm/v1/chat/completions" \
    -H "Content-Type: application/json" \
  -H "X-Project-ID: ${PROJECT_ID}" \
  -H "X-User-ID: ${USER_ID}" \
  -d "$CHAT_PAYLOAD" 2>/dev/null || echo '{"error": "LLM backend not available"}')

echo "  Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Step 3: Text completion
log "Step 3: Text completion (POST /umrs/llm/v1/completions)"
echo ""
echo "  Request:"
echo '  {
    "model": "rivault-gpt-oss-static",
    "prompt": "The capital of France is",
    "max_tokens": 10
  }'
echo ""

# Step 4: Embeddings
log "Step 4: Embeddings (POST /umrs/llm/v1/embeddings)"
echo ""
echo "  Request:"
echo '  {
    "model": "text-embedding-model",
    "input": "Hello world"
  }'
echo ""
echo "  Response:"
echo '  {
    "data": [{"embedding": [0.1, 0.2, ...], "index": 0}],
    "model": "text-embedding-model",
    "usage": {"prompt_tokens": 2, "total_tokens": 2}
  }'
echo ""

# Step 5: Comparison with Gateway
log "Step 5: Gateway vs UMRS direct API"
echo ""
echo "  ┌──────────────────────────────────────────────────────────────────┐"
echo "  │  Path                                  │ Use Case               │"
echo "  ├──────────────────────────────────────────────────────────────────┤"
echo "  │  Gateway: /gw/llm/v1/chat/completions  │ User requests (auth)   │"
echo "  │  UMRS:    /umrs/llm/v1/chat/completions│ Internal/admin calls   │"
echo "  │  LiteLLM: :11434/v1/chat/completions   │ Direct (inside mini-RV)│"
echo "  └──────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Production rule: Users ALWAYS go through Gateway."
echo "  UMRS LLM endpoints are for internal orchestration."
echo ""

banner "Demo 29 Complete"
