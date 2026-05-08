#!/usr/bin/env bash
# =============================================================================
# Demo 25: Endpoint Provisioning API
# =============================================================================
# Shows: POST /umrs/endpoints/provision, GET /umrs/endpoints/active,
#        warm pool reuse, multi-model endpoints, LoRA support,
#        MCP/RAG mounts and EgressGuard allowlist profile intent
#
# Prerequisites:
#   - UMRS server reachable via HTTP (default: localhost:8001)
#   - At least one model registered or pre-staged
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="ep_prov" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-${UMRS_HTTP_URL:-http://localhost:8001}}"

banner "Demo 25: Endpoint Provisioning API"
echo ""

# Step 0: Pre-flight
log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: List current endpoints
log "Step 1: List active endpoints (GET /umrs/endpoints/active)"
echo ""
echo "  curl -s -H 'X-Project-ID: demo25' ${UMRS_URL}/umrs/endpoints/active | jq '.'"
echo ""
ENDPOINTS=$(curl -sf -H "X-Project-ID: demo25" "${UMRS_URL}/umrs/endpoints/active" 2>/dev/null || echo '{"endpoints":[]}')
echo "$ENDPOINTS" | python3 -m json.tool 2>/dev/null || echo "$ENDPOINTS"
echo ""

# Step 2: Provision an endpoint
log "Step 2: Provision a new endpoint (POST /umrs/endpoints/provision)"
echo ""
prepare_mini_rivault_mount_demo
echo "  Request body:"
python3 - <<PY
import json

payload = {
  "model": "${DEMO25_MODEL:-text-embedding}",
  "tier": "project",
  "project_id": "demo25",
  "max_model_len": 4096,
  "wait": False,
  "mounts": {
    "mcp_servers": [
      {"source": "${DEMO_MCP_SOURCE}", "target": "/mcp-servers/demo-tools"}
    ],
    "project_data": {"source": "${DEMO_PROJECT_DATA_SOURCE}", "target": "/project"},
    "rag_index": {"source": "${DEMO_RAG_INDEX_SOURCE}", "target": "/rag-index"},
  },
  "egress_domains": "${DEMO_EGRESS_DOMAINS}".split(","),
}
print(json.dumps(payload, indent=2))
PY
echo ""
echo "  In production, UMRS creates a durable job (endpoint_provision)"
echo "  that submits a Slurm/K8s/Compose job to launch the container."
echo ""
echo "  The endpoint registry tracks:"
echo "    - instance_id, litellm_url, model name"
echo "    - owner, tier, status (provisioning/ready/stopping)"
echo "    - job_id, backend_ref (Slurm JOBID or K8s name)"
echo ""

print_mini_rivault_mount_summary
echo ""
echo "  The mount fields are translated to launch env:"
echo "    - RIVAULT_EXTRA_PROFILES=custom-mcp egress rag"
echo "    - MCP_SERVERS_PATH or RIVAULT_MCP_MOUNTS_JSON"
echo "    - PROJECT_DATA_PATH, RAG_INDEX_PATH"
echo "    - EGRESS_DOMAIN_ALLOWLIST"
echo ""

if [[ "${DEMO25_PROVISION:-0}" = "1" ]]; then
  log "Step 2b: Submitting the demo provisioning request"
  request_payload=$(python3 - <<PY
import json

payload = {
  "model": "${DEMO25_MODEL:-text-embedding}",
  "tier": "project",
  "project_id": "demo25",
  "max_model_len": 4096,
  "wait": False,
  "mounts": {
    "mcp_servers": [
      {"source": "${DEMO_MCP_SOURCE}", "target": "/mcp-servers/demo-tools"}
    ],
    "project_data": {"source": "${DEMO_PROJECT_DATA_SOURCE}", "target": "/project"},
    "rag_index": {"source": "${DEMO_RAG_INDEX_SOURCE}", "target": "/rag-index"},
  },
  "egress_domains": "${DEMO_EGRESS_DOMAINS}".split(","),
}
print(json.dumps(payload))
PY
)
  curl -sf -X POST "${UMRS_URL%/}/umrs/endpoints/provision" \
    -H 'Content-Type: application/json' \
    -d "$request_payload" | python3 -m json.tool
else
  log "Set DEMO25_PROVISION=1 to submit the request. The default keeps the demo non-disruptive."
fi
echo ""

# Step 3: Warm pool reuse
log "Step 3: Warm endpoint reuse"
echo ""
echo "  When an existing endpoint matches the requested model, tier,"
echo "  mount/profile intent, and is still warm (within TTL), UMRS reuses it instead of"
echo "  provisioning a new one. This avoids cold-start latency."
echo ""
echo "  Verification: GET /umrs/endpoints checks backend_alive before"
echo "  returning warm endpoints to the Gateway."
echo ""

# Step 4: Multi-model & LoRA
log "Step 4: Multi-model and LoRA endpoints"
echo ""
echo '  # Multi-model endpoint (co-located models)'
echo '  {"models": ["model-a", "model-b"], "serving_mode": "co-located"}'
echo ""
echo '  # Multi-LoRA endpoint'
echo '  {"model": "base-model", "serving_mode": "multi-lora",'
echo '   "lora_adapters": [{"name": "adapter1", "path": "/adapters/a1"}]}'
echo ""

# Step 5: Endpoint lifecycle via CLI
log "Step 5: CLI equivalents"
echo ""
echo "  # Via umrs-client (durable job submission)"
echo '  umrs-client job submit endpoint_provision \'
echo '      --spec '"'"'{"model": "sam3"}'"'"' --project demo25'
echo ""
echo "  # Monitor"
echo "  umrs-client monitor --project demo25 --tab jobs"
echo ""

banner "Demo 25 Complete"
