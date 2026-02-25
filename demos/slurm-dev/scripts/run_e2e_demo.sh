#!/bin/bash
# =============================================================================
# UMRS E2E Demo: Complete orchestration script
# =============================================================================
#
# Runs the full demo lifecycle:
#   Phase 1: Start core Slurm + UMRS infrastructure
#   Phase 2: Upload test data to MinIO
#   Phase 3: Run text RAG demo (Slurm embedding job)
#   Phase 4: Wait for embedding computation to complete
#   Phase 5: Stop Slurm embedding job (free GPU)
#   Phase 6: Start always-on LLM (vLLM + LiteLLM + OpenWebUI)
#   Phase 7: Run RAG Q&A with LLM
#   Phase 8: Summary
#
# Usage:
#   cd demos/slurm-dev
#   bash scripts/run_e2e_demo.sh
#
# Prerequisites:
#   - podman-compose installed
#   - GPU available (/dev/nvidia0)
#   - Model weights at MODEL_PATH (default: ./rivault/models)
# =============================================================================

set -e

COMPOSE_FILE="docker-compose.yaml"
LLM_ENV="rivault/demo-llm.env"
LLM_COMPOSE="rivault/demo-llm.yaml"
LLM_OVERRIDE="rivault/x_nvidia-override.yaml"

# Detect compose command
if command -v podman-compose >/dev/null 2>&1; then
    COMPOSE="podman-compose"
else
    echo "ERROR: podman-compose not found."
    echo "Install with: pip install podman-compose"
    exit 1
fi

# User config file (used by client_umrs.py -c)
USER_CONF="/root/.config/umrs/users/user_rivault.json"

echo ""
echo "============================================================"
echo "  UMRS E2E Demo"
echo "  Compose command: ${COMPOSE}"
echo "============================================================"

# =============================================================================
# Phase 1: Start core infrastructure
# =============================================================================
echo ""
echo "Phase 1: Starting all services..."
echo "------------------------------------------------------------"
echo "  (depends_on healthchecks handle startup ordering automatically)"

$COMPOSE -f "$COMPOSE_FILE" up -d
echo "  All services started. Waiting for cluster to become ready..."

# Wait for compute node to be idle (implies munge, postgres, minio, ai-login are all healthy)
for i in $(seq 1 60); do
    NODE_STATE=$(podman exec ai-login sinfo -N -h -o "%T" 2>/dev/null || echo "unknown")
    if echo "$NODE_STATE" | grep -q "idle"; then
        echo "  Cluster ready (compute node idle)."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  WARNING: Cluster not fully ready after 60 attempts."
        echo "  Check logs: podman logs ai-login"
    fi
    sleep 5
done

# =============================================================================
# Phase 2: Upload test data (via umrs-client)
# =============================================================================
echo ""
echo "Phase 2: Uploading test data via umrs-client..."
echo "------------------------------------------------------------"

podman exec umrs-client bash /opt/scripts/setup_demo_data.sh

# =============================================================================
# Phase 3: Text RAG demo (embedding via Slurm -> Rivault -> vLLM)
# =============================================================================
echo ""
echo "Phase 3: Running text RAG demo..."
echo "------------------------------------------------------------"
echo "  (This triggers: upload -> UMRS -> sbatch -> Rivault -> vLLM -> embeddings -> pgvector)"

podman exec umrs-client python3 /opt/scripts/demo_text_rag.py \
    --conf "$USER_CONF" \
    --server-ip ai-login \
    --rag-url http://ai-login:8081 \
    --skip-wait || {
    echo "  WARNING: Text RAG demo encountered errors (embedding job may still be running)."
    echo "  Check UMRS server logs: podman logs ai-login"
}

# =============================================================================
# Phase 4: Wait for embedding computation to complete
# =============================================================================
echo ""
echo "Phase 4: Waiting for embedding jobs to complete..."
echo "------------------------------------------------------------"
echo "  (Polling task status via umrs-client every 10s, timeout 10 minutes)"

EMBED_TIMEOUT=60  # iterations * 10s = 600s = 10 minutes
for i in $(seq 1 $EMBED_TIMEOUT); do
    # Extract only the JSON line (client prints non-JSON status lines to stdout)
    STATUS_RAW=$(podman exec umrs-client python3 /opt/umrs/client_umrs.py \
        -c "$USER_CONF" --output json status all 2>/dev/null || echo "")
    STATUS_JSON=$(echo "$STATUS_RAW" | grep '^{' | tail -1)
    # Check if any threads are still Running
    RUNNING=$(echo "$STATUS_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    summaries = data.get('summaries', {})
    for key, entry in summaries.items():
        for thread_info in entry.get('data', []):
            if thread_info.get('status') == 'Running':
                print(thread_info.get('thread_id', 'unknown'))
except Exception:
    pass
" 2>/dev/null || echo "")
    if [ -z "$RUNNING" ]; then
        echo "  All embedding tasks completed."
        break
    fi
    RUNNING_COUNT=$(echo "$RUNNING" | wc -w)
    if [ "$i" -eq 1 ] || [ $((i % 6)) -eq 0 ]; then
        echo "  $RUNNING_COUNT task(s) still running (attempt $i/$EMBED_TIMEOUT)..."
    fi
    if [ "$i" -eq "$EMBED_TIMEOUT" ]; then
        echo "  WARNING: Timed out after 10 minutes. $RUNNING_COUNT task(s) still active."
        echo "  Proceeding to cancel remaining jobs."
    fi
    sleep 10
done

# =============================================================================
# Phase 5: Stop Slurm embedding job (free GPU for always-on LLM)
# =============================================================================
echo ""
echo "Phase 5: Stopping Slurm embedding job to free GPU..."
echo "------------------------------------------------------------"

# Cancel any running Slurm jobs
JOB_IDS=$(podman exec ai-login squeue -h -o "%A" 2>/dev/null || echo "")
if [ -n "$JOB_IDS" ]; then
    for jid in $JOB_IDS; do
        echo "  Cancelling Slurm job: $jid"
        podman exec ai-login scancel "$jid" || true
    done
    echo "  Waiting for GPU to be released..."
    sleep 10
else
    echo "  No active Slurm jobs."
fi

# =============================================================================
# Phase 6: Start always-on LLM (separate compose project)
# =============================================================================
echo ""
echo "Phase 6: Starting always-on LLM (vLLM + LiteLLM + OpenWebUI)..."
echo "------------------------------------------------------------"

# Create the isolated network for the LLM stack
podman network create --internal --disable-dns \
    --subnet 192.168.5.0/24 demo-llm-net 2>/dev/null || true

# Launch the always-on LLM stack
$COMPOSE --env-file "$LLM_ENV" \
    -f "$LLM_COMPOSE" -f "$LLM_OVERRIDE" \
    up -d

echo "  Waiting for vLLM to load model (this may take a few minutes)..."
for i in $(seq 1 60); do
    # Check if the vLLM container is up and responding
    if podman exec rivault_vllm_1 curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo "  vLLM model loaded and ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  WARNING: vLLM not ready after 5 minutes."
        echo "  Check logs: podman logs rivault_vllm_1"
    fi
    sleep 5
done

# =============================================================================
# Phase 7: RAG QA with always-on LLM
# =============================================================================
echo ""
echo "Phase 7: Testing RAG Q&A with always-on LLM..."
echo "------------------------------------------------------------"

# The RAG service on ai-login connects to the LLM via host port 14000
podman exec umrs-client python3 /opt/scripts/demo_text_rag.py \
    --conf "$USER_CONF" \
    --server-ip ai-login \
    --rag-url http://ai-login:8081 \
    --skip-upload --skip-wait || {
    echo "  WARNING: RAG Q&A encountered errors."
    echo "  Check RAG logs: podman exec ai-login cat /var/log/umrs-rag.log"
}

# =============================================================================
# Phase 8: Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  E2E Demo Complete!"
echo "============================================================"
echo ""
echo "  Services available:"
echo "    OpenWebUI (chat):     http://localhost:13000"
echo "    LiteLLM API:          http://localhost:14000"
echo "    UMRS RAG:             http://localhost:8081"
echo "    MinIO Console:        http://localhost:19001"
echo "    PostgreSQL:           localhost:15432"
echo "    UMRS ZMQ:             localhost:5555"
echo ""
echo "  Try these in OpenWebUI:"
echo "    - 'How do seismic waves propagate?'"
echo "    - 'What is RiVault?'"
echo "    - 'Explain GPU acceleration for scientific computing'"
echo ""
echo "  To stop all services:"
echo "    $COMPOSE -f $COMPOSE_FILE down"
echo "    $COMPOSE --env-file $LLM_ENV -f $LLM_COMPOSE -f $LLM_OVERRIDE down"
echo ""
