#!/bin/bash
# =============================================================================
# UMRS E2E Demo: Complete orchestration script
# =============================================================================
#
# Runs the full demo lifecycle:
#   Phase 1: Start core Slurm + UMRS infrastructure
#   Phase 2: Upload test data to MinIO
#   Phase 3: Run text RAG demo (Slurm embedding job)
#   Phase 4: Stop Slurm embedding job (free GPU)
#   Phase 5: Start always-on LLM (vLLM + LiteLLM + OpenWebUI)
#   Phase 6: Run RAG Q&A with LLM
#   Phase 7: Summary
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

echo ""
echo "============================================================"
echo "  UMRS E2E Demo"
echo "  Compose command: ${COMPOSE}"
echo "============================================================"

# =============================================================================
# Phase 1: Start core infrastructure
# =============================================================================
echo ""
echo "Phase 1: Starting core services..."
echo "------------------------------------------------------------"

$COMPOSE -f "$COMPOSE_FILE" up -d munge postgres minio
echo "  Waiting for infrastructure services..."
sleep 5

$COMPOSE -f "$COMPOSE_FILE" up -d ai-login
echo "  Waiting for Slurm controller + UMRS server..."

# Wait for ai-login healthcheck (slurmctld + UMRS ZeroMQ)
for i in $(seq 1 60); do
    if podman exec ai-login sinfo -N >/dev/null 2>&1 && \
       echo 'ping' | podman exec -i ai-login nc -w 5 localhost 5555 >/dev/null 2>&1; then
        echo "  Slurm controller + UMRS server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  WARNING: Slurm controller or UMRS not ready after 60 attempts."
        echo "  Check logs: podman logs ai-login"
    fi
    sleep 5
done

$COMPOSE -f "$COMPOSE_FILE" up -d compute-01
echo "  Waiting for compute node..."

for i in $(seq 1 30); do
    NODE_STATE=$(podman exec ai-login sinfo -N -h -o "%T" 2>/dev/null || echo "unknown")
    if echo "$NODE_STATE" | grep -q "idle"; then
        echo "  Compute node ready (idle)."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  WARNING: Compute node not idle after 30 attempts."
    fi
    sleep 5
done

# Start client container
$COMPOSE -f "$COMPOSE_FILE" up -d umrs-client
echo "  UMRS client container started."

# =============================================================================
# Phase 2: Upload test data
# =============================================================================
echo ""
echo "Phase 2: Uploading test data to MinIO..."
echo "------------------------------------------------------------"

podman exec ai-login bash /opt/scripts/setup_demo_data.sh

# =============================================================================
# Phase 3: Text RAG demo (embedding via Slurm -> Rivault -> vLLM)
# =============================================================================
echo ""
echo "Phase 3: Running text RAG demo..."
echo "------------------------------------------------------------"
echo "  (This triggers: upload -> UMRS -> sbatch -> Rivault -> vLLM -> embeddings -> pgvector)"

podman exec umrs-client python3 /opt/scripts/demo_text_rag.py \
    --conf /root/conf_umrs.json \
    --server-ip ai-login \
    --rag-url http://ai-login:8081 \
    --skip-wait || {
    echo "  WARNING: Text RAG demo encountered errors (embedding job may still be running)."
    echo "  Check UMRS server logs: podman logs ai-login"
}

# =============================================================================
# Phase 4: Stop Slurm embedding job (free GPU for always-on LLM)
# =============================================================================
echo ""
echo "Phase 4: Stopping Slurm embedding job to free GPU..."
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
# Phase 5: Start always-on LLM (separate compose project)
# =============================================================================
echo ""
echo "Phase 5: Starting always-on LLM (vLLM + LiteLLM + OpenWebUI)..."
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
# Phase 6: RAG QA with always-on LLM
# =============================================================================
echo ""
echo "Phase 6: Testing RAG Q&A with always-on LLM..."
echo "------------------------------------------------------------"

# The RAG service on ai-login connects to the LLM via host port 14000
podman exec umrs-client python3 /opt/scripts/demo_text_rag.py \
    --conf /root/conf_umrs.json \
    --server-ip ai-login \
    --rag-url http://ai-login:8081 \
    --skip-upload --skip-wait || {
    echo "  WARNING: RAG Q&A encountered errors."
    echo "  Check RAG logs: podman exec ai-login cat /var/log/umrs-rag.log"
}

# =============================================================================
# Phase 7: Summary
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
