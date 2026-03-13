#!/bin/bash
# =============================================================================
# Mini-RiVault Job Script (wrapper for serve-model.slurm)
# =============================================================================
# This Slurm batch script runs on the HOST compute node and launches the
# full Rivault container stack (vLLM + LiteLLM + NGINX + supporting
# services) via the host's native podman-compose — no nested containers.
#
# It outputs the FQDN header that UMRS's extract_node_address() expects,
# then delegates to the actual serve-model.slurm from src/umrs/rivault/.
#
# Environment variables (set by start-host-slurmd.sh or caller):
#   RIVAULT_DIR      – path to rivault/ directory (default: /opt/rivault)
#   JOB_INFO_DIR     – path to shared job-info dir (default: /var/run/rivault)
#   VLLM_PATH_TO_HUGGINGFACE – model path for vLLM
# =============================================================================

#SBATCH --job-name=mini-rivault
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# --- Output FQDN header (required by UMRS rivault.py extract_node_address) ---
echo "=============================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NODELIST"
echo "Running on node: $(hostname)"
echo "Fully qualified name: $(hostname -f)"
echo "=============================="

# RIVAULT_DIR: set by start-host-slurmd.sh (exported into slurmd env),
# or fall back to /opt/rivault for containerised execution.
RIVAULT_DIR="${RIVAULT_DIR:-/opt/rivault}"

if [[ ! -d "$RIVAULT_DIR" ]]; then
    echo "ERROR: Rivault directory not found at $RIVAULT_DIR"
    echo "  Set RIVAULT_DIR or create a symlink:"
    echo "  sudo ln -s $(pwd)/rivault /opt/rivault"
    exit 1
fi

# --- Write job info JSON ---
# JOB_INFO_DIR should be the ./job-info directory that is also volume-mounted
# into ai-login so the UMRS server can read it.
JOB_INFO_DIR="${JOB_INFO_DIR:-/var/run/rivault}"
mkdir -p "$JOB_INFO_DIR"
JOB_ID="${SLURM_JOB_ID:-test-$$}"
cat > "$JOB_INFO_DIR/rivault-job-${JOB_ID}.json" << EOF
{
  "job_id": "${JOB_ID}",
  "host": "$(hostname)",
  "fqdn": "$(hostname -f)",
  "rivault_dir": "${RIVAULT_DIR}",
  "start_time": "$(date -Iseconds)",
  "status": "starting"
}
EOF

# --- Launch Rivault via serve-model.slurm ---
cd "$RIVAULT_DIR"
exec bash ./serve-model.slurm
