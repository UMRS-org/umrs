#!/bin/bash
# =============================================================================
# Mini-RiVault Job Script for Slurm Development Environment
# =============================================================================
#
# This script runs when a mini-rivault job is submitted to Slurm.
# It writes job info for UMRS discovery and either:
#   - Starts mini-rivault in the container (if available)
#   - Or signals that an external mini-rivault should connect
#
# Usage (submitted by UMRS orchestrator):
#   sbatch --export=ALL,VLLM_MODEL_PATH=/models/foo mini-rivault-job.sh
#
# Environment Variables:
#   VLLM_MODEL_PATH      - Path to the model (required)
#   LITELLM_PORT         - LiteLLM proxy port (default: 11434)
#   LITELLM_MASTER_KEY   - API key (default: sk-demo)
#   MINI_RIVAULT_MODE    - "internal" or "external" (default: external)
#
# =============================================================================

set -e

# Configuration with defaults
: ${VLLM_MODEL_PATH:=/models}
: ${VLLM_PORT:=8000}
: ${LITELLM_PORT:=11434}
: ${LITELLM_MASTER_KEY:=sk-demo}
: ${JOB_INFO_DIR:=/var/run/rivault}
: ${MINI_RIVAULT_MODE:=external}

# Get job ID from Slurm or use a test ID
JOB_ID=${SLURM_JOB_ID:-test-$$}
NODE=$(hostname)

echo "============================================="
echo "Mini-RiVault Job"
echo "============================================="
echo "Job ID:         $JOB_ID"
echo "Node:           $NODE"
echo "Model Path:     $VLLM_MODEL_PATH"
echo "LiteLLM Port:   $LITELLM_PORT"
echo "Mode:           $MINI_RIVAULT_MODE"
echo "============================================="

# Create job info directory
mkdir -p "$JOB_INFO_DIR"

# Write job info file (shell format for sourcing)
JOB_INFO_FILE="$JOB_INFO_DIR/rivault-job-${JOB_ID}.info"
cat > "$JOB_INFO_FILE" << EOF
RIVAULT_JOB_ID=${JOB_ID}
RIVAULT_NODE=${NODE}
RIVAULT_LITELLM_PORT=${LITELLM_PORT}
RIVAULT_VLLM_PORT=${VLLM_PORT}
RIVAULT_MODEL_PATH=${VLLM_MODEL_PATH}
RIVAULT_START_TIME=$(date -Iseconds)
RIVAULT_STATUS=starting
RIVAULT_MODE=${MINI_RIVAULT_MODE}
EOF

# Write JSON job info for UMRS REST API
JOB_INFO_JSON="$JOB_INFO_DIR/rivault-job-${JOB_ID}.json"
cat > "$JOB_INFO_JSON" << EOF
{
  "job_id": "${JOB_ID}",
  "backend": "slurm",
  "host": "${NODE}",
  "litellm_port": ${LITELLM_PORT},
  "vllm_port": ${VLLM_PORT},
  "model_path": "${VLLM_MODEL_PATH}",
  "start_time": "$(date -Iseconds)",
  "status": "starting",
  "mode": "${MINI_RIVAULT_MODE}"
}
EOF

echo "Job info written to: $JOB_INFO_FILE"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up job $JOB_ID..."
    
    # Update status
    if [[ -f "$JOB_INFO_FILE" ]]; then
        sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=stopped/' "$JOB_INFO_FILE"
        echo "RIVAULT_STOP_TIME=$(date -Iseconds)" >> "$JOB_INFO_FILE"
    fi
    
    if [[ -f "$JOB_INFO_JSON" ]]; then
        # Update JSON status (simple sed replace)
        sed -i 's/"status": "[^"]*"/"status": "stopped"/' "$JOB_INFO_JSON"
    fi
    
    # Stop any running processes
    if [[ -n "$MINI_RIVAULT_PID" ]]; then
        echo "Stopping mini-rivault (PID: $MINI_RIVAULT_PID)..."
        kill "$MINI_RIVAULT_PID" 2>/dev/null || true
        wait "$MINI_RIVAULT_PID" 2>/dev/null || true
    fi
    
    # Give time for cleanup, then remove files
    sleep 2
    rm -f "$JOB_INFO_FILE" "$JOB_INFO_JSON"
    
    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# =============================================================================
# Mode-specific behavior
# =============================================================================

if [[ "$MINI_RIVAULT_MODE" == "internal" ]]; then
    # ---------------------------------------------------------------------
    # Internal mode: Start mini-rivault inside the container
    # This requires vLLM and LiteLLM to be installed
    # ---------------------------------------------------------------------
    echo ""
    echo "Starting mini-rivault (internal mode)..."
    
    # Check for dependencies
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python3 not found"
        sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=failed/' "$JOB_INFO_FILE"
        exit 1
    fi
    
    # Check if vLLM is available
    if python3 -c "import vllm" 2>/dev/null; then
        echo "vLLM found, starting inference server..."
        
        # Start vLLM in background
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$VLLM_MODEL_PATH" \
            --port "$VLLM_PORT" \
            --max-model-len 4096 \
            &
        VLLM_PID=$!
        
        # Wait for vLLM to start
        echo "Waiting for vLLM to start..."
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
                echo "vLLM is ready"
                break
            fi
            sleep 5
        done
        
        MINI_RIVAULT_PID=$VLLM_PID
    else
        echo "WARNING: vLLM not installed, running in mock mode"
        # Start a simple mock server
        python3 -c "
import http.server
import json
import socketserver

class MockHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        elif self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'data': [{'id': 'rivault-gpt-oss-static'}]}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {
            'id': 'mock',
            'choices': [{'message': {'content': 'Mock response from mini-rivault'}}]
        }
        self.wfile.write(json.dumps(response).encode())

with socketserver.TCPServer(('', ${LITELLM_PORT}), MockHandler) as httpd:
    httpd.serve_forever()
" &
        MINI_RIVAULT_PID=$!
    fi
    
    # Update status to ready
    sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=ready/' "$JOB_INFO_FILE"
    sed -i 's/"status": "[^"]*"/"status": "ready"/' "$JOB_INFO_JSON"
    echo "Mini-RiVault is ready on port $LITELLM_PORT"
    
    # Wait for job to be cancelled or timeout
    echo "Job running. Press Ctrl+C or cancel job to stop."
    wait $MINI_RIVAULT_PID

else
    # ---------------------------------------------------------------------
    # External mode: Wait for external mini-rivault to connect
    # This is used when mini-rivault runs on the host machine
    # ---------------------------------------------------------------------
    echo ""
    echo "External mode: Waiting for mini-rivault to be started externally..."
    echo ""
    echo "To connect external mini-rivault, run on the HOST:"
    echo "  export LITELLM_PORT=$LITELLM_PORT"
    echo "  export VLLM_MODEL_PATH=$VLLM_MODEL_PATH"
    echo "  ./mini-rivault-start.sh"
    echo ""
    
    # Poll for external service
    echo "Polling for LiteLLM on port $LITELLM_PORT..."
    MAX_WAIT=600
    WAITED=0
    
    while true; do
        if curl -sf "http://localhost:${LITELLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "External mini-rivault detected!"
            sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=ready/' "$JOB_INFO_FILE"
            sed -i 's/"status": "[^"]*"/"status": "ready"/' "$JOB_INFO_JSON"
            break
        fi
        
        sleep 5
        WAITED=$((WAITED + 5))
        
        if [[ $WAITED -ge $MAX_WAIT ]]; then
            echo "TIMEOUT: External mini-rivault not detected after ${MAX_WAIT}s"
            sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=failed/' "$JOB_INFO_FILE"
            sed -i 's/"status": "[^"]*"/"status": "failed"/' "$JOB_INFO_JSON"
            exit 1
        fi
        
        echo "  Still waiting... ($WAITED/${MAX_WAIT}s)"
    done
    
    # Keep job running until cancelled
    echo "Job running. Monitoring external service..."
    while true; do
        if ! curl -sf "http://localhost:${LITELLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "External service disconnected"
            break
        fi
        sleep 30
    done
fi
