#!/usr/bin/env bash
# =============================================================================
# Demo 25: Endpoint Provisioning API
# =============================================================================
# Shows: POST /umrs/endpoints/provision, GET /umrs/endpoints,
#        warm pool reuse, multi-model endpoints, LoRA support
#
# Prerequisites:
#   - UMRS server reachable via HTTP (default: localhost:8001)
#   - At least one model registered or pre-staged
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="ep_prov" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-http://localhost:8001}"

banner "Demo 25: Endpoint Provisioning API"
echo ""

# Step 0: Pre-flight
log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: List current endpoints
log "Step 1: List active endpoints (GET /umrs/endpoints)"
echo ""
echo "  curl -s ${UMRS_URL}/umrs/endpoints | jq '.'"
echo ""
ENDPOINTS=$(curl -sf "${UMRS_URL}/umrs/endpoints" 2>/dev/null || echo '{"endpoints":[]}')
echo "$ENDPOINTS" | python3 -m json.tool 2>/dev/null || echo "$ENDPOINTS"
echo ""

# Step 2: Provision an endpoint
log "Step 2: Provision a new endpoint (POST /umrs/endpoints/provision)"
echo ""
echo "  Request body:"
echo '  {
    "model": "text-embedding",
    "tier": "user",
    "project_id": "demo25",
    "max_model_len": 4096
  }'
echo ""
echo "  In production, UMRS creates a durable job (endpoint_provision)"
echo "  that submits a Slurm/K8s/Compose job to launch the container."
echo ""
echo "  The endpoint registry tracks:"
echo "    - instance_id, litellm_url, model name"
echo "    - owner, tier, status (provisioning/ready/stopping)"
echo "    - job_id, backend_ref (Slurm JOBID or K8s name)"
echo ""

# Step 3: Warm pool reuse
log "Step 3: Warm endpoint reuse"
echo ""
echo "  When an existing endpoint matches the requested model, tier,"
echo "  and is still warm (within TTL), UMRS reuses it instead of"
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
