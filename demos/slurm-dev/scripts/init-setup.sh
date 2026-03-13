#!/usr/bin/env bash
# =============================================================================
# init-setup.sh — One-time setup for the Slurm Development Demo
# =============================================================================
# Auto-detects hardware (CPUs, RAM, GPUs, hostname, architecture) and
# generates all configuration files needed to run the E2E demo.
#
# Usage:
#   bash scripts/init-setup.sh          # interactive (prompts for overrides)
#   bash scripts/init-setup.sh --auto   # non-interactive (all defaults)
#   bash scripts/init-setup.sh --help
#
# Generated / updated files:
#   .env                 — environment variables
#   slurm.conf           — Slurm cluster config (CPU/RAM/GPU)
#   gres.conf            — Slurm GPU resource mapping
#   s3cfg_rivault_user   — S3 credentials (random password)
#
# Prerequisites checked: podman, podman-compose, slurmd, munge, nvidia-smi
# =============================================================================
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEMO_DIR"

# ---- Colours ----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[init-setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[init-setup]  ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}[init-setup]  ⚠${NC} $*"; }
fail()  { echo -e "${RED}[init-setup]  ✗${NC} $*"; }

AUTO=false
[[ "${1:-}" == "--auto" ]] && AUTO=true
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,/^# =====/{ /^# /s/^# //p }' "$0"
    exit 0
fi

# ---- Helper: prompt with default -------------------------------------------
ask() {
    local var="$1" prompt="$2" default="$3"
    if $AUTO; then
        eval "$var=\"$default\""
        return
    fi
    read -rp "$(echo -e "${CYAN}[init-setup]${NC} $prompt [${default}]: ")" val
    eval "$var=\"\${val:-$default}\""
}

# =============================================================================
# Phase 1: Preflight checks
# =============================================================================
info "Phase 1: Preflight checks"

ERRORS=0
check_cmd() {
    if command -v "$1" &>/dev/null; then
        ok "$1 found ($(command -v "$1"))"
    else
        fail "$1 NOT found — $2"
        ERRORS=$((ERRORS + 1))
    fi
}

check_cmd podman          "Install: https://podman.io/docs/installation"
check_cmd podman-compose  "Install: pip install podman-compose"
check_cmd slurmd          "Install: sudo apt-get install -y slurmd"
check_cmd munged          "Install: sudo apt-get install -y munge"

# GPU (optional but recommended)
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    ok "NVIDIA GPU detected"
    HAS_GPU=true
else
    warn "nvidia-smi not found or no GPU — GPU features will be unavailable"
fi

# Check nvidia-ctk for CDI (needed by Rivault containers)
if $HAS_GPU; then
    if command -v nvidia-ctk &>/dev/null; then
        ok "nvidia-ctk found (CDI support)"
    else
        warn "nvidia-ctk not found — Rivault GPU containers may fail"
        warn "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
fi

# Ports
for port in 6817 5555 8081 15432 19000 19001; do
    if ss -tlnp 2>/dev/null | grep -q ":${port} " ; then
        warn "Port $port already in use (may conflict)"
    fi
done

if [[ $ERRORS -gt 0 ]]; then
    fail "$ERRORS required tool(s) missing. Install them and re-run."
    exit 1
fi
echo ""

# =============================================================================
# Phase 2: Detect hardware
# =============================================================================
info "Phase 2: Detecting hardware"

# Hostname / FQDN
DETECTED_FQDN="$(hostname -f 2>/dev/null || hostname)"
ok "FQDN: $DETECTED_FQDN"

# CPUs
DETECTED_CPUS="$(nproc)"
ok "CPUs: $DETECTED_CPUS"

# Memory (MB, usable by Slurm — leave ~2GB for OS)
TOTAL_MEM_MB="$(free -m | awk '/^Mem:/{print $2}')"
SLURM_MEM_MB=$((TOTAL_MEM_MB - 2048))
[[ $SLURM_MEM_MB -lt 1024 ]] && SLURM_MEM_MB=$TOTAL_MEM_MB
ok "Memory: ${TOTAL_MEM_MB}MB total, ${SLURM_MEM_MB}MB for Slurm"

# GPUs
GPU_COUNT=0
GPU_DEVICES=""
if $HAS_GPU; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"
    # Build device file list: /dev/nvidia0,/dev/nvidia1,...
    GPU_DEVICES=$(seq 0 $((GPU_COUNT - 1)) | sed 's|^|/dev/nvidia|' | paste -sd,)
    ok "GPUs: $GPU_COUNT ($GPU_DEVICES)"
fi

# Architecture (for vLLM docker tag)
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  VLLM_TAG_DEFAULT="latest" ;;
    aarch64) VLLM_TAG_DEFAULT="cu130-nightly-aarch64" ;;
    *)       VLLM_TAG_DEFAULT="latest"; warn "Unknown arch $ARCH, defaulting vLLM tag to 'latest'" ;;
esac
ok "Architecture: $ARCH → vLLM tag: $VLLM_TAG_DEFAULT"
echo ""

# =============================================================================
# Phase 3: Interactive overrides (skipped with --auto)
# =============================================================================
info "Phase 3: Configuration"

ask COMPUTE_HOST_FQDN "Host FQDN"          "$DETECTED_FQDN"
ask NODE_CPUS         "Slurm CPUs"          "$DETECTED_CPUS"
ask NODE_MEM          "Slurm Memory (MB)"   "$SLURM_MEM_MB"
ask NODE_GPUS         "Number of GPUs"      "$GPU_COUNT"
ask VLLM_DOCKER_TAG   "vLLM Docker tag"     "$VLLM_TAG_DEFAULT"
ask VLLM_MODEL        "vLLM model"          "intfloat/e5-base-v2"
ask VLLM_GPU_MEM_UTIL "vLLM GPU mem util"   "0.80"
echo ""

# =============================================================================
# Phase 4: Generate .env
# =============================================================================
info "Phase 4: Generating .env"

# Generate a random S3 password for rivault_user (16 hex chars)
S3_RIVAULT_PASSWORD="$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' | head -c 32)"

cat > .env << ENVEOF
# =============================================================================
# Generated by init-setup.sh on $(date -Iseconds)
# =============================================================================

# -----------------------------------------------------------------------------
# Host Compute Node
# -----------------------------------------------------------------------------
COMPUTE_HOST_FQDN=${COMPUTE_HOST_FQDN}

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
MODEL_PATH=./rivault/models
DEMO_MODEL_PATH=/models/qwen-0.5b

# -----------------------------------------------------------------------------
# Slurm Configuration
# -----------------------------------------------------------------------------
SLURM_PARTITION=dev
SLURM_TIME=01:00:00
SLURM_GPUS=${NODE_GPUS}
SLURM_CPUS=4
SLURM_MEM=8G

# -----------------------------------------------------------------------------
# Rivault / LiteLLM Configuration
# -----------------------------------------------------------------------------
NGINX_LITELLM_DOCKER_PORT=11434
LITELLM_MASTER_KEY=sk-demo-dev
RIVAULT_REMOTE_PORT=11434

VLLM_MODEL=${VLLM_MODEL}
VLLM_DOCKER_TAG=${VLLM_DOCKER_TAG}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL}
VLLM_PATH_TO_HUGGINGFACE=./models

# -----------------------------------------------------------------------------
# UMRS Database Configuration
# -----------------------------------------------------------------------------
DB_HOST=postgres
DB_USER_PASSWORD=umrs_dev
UMRS_SERVER_HOST=ai-login

POSTGRES_USER=umrs
POSTGRES_PASSWORD=umrs_dev
POSTGRES_DB=umrs

SENTENCE_TRANSFORMERS_HOME=/shared/models/sentence_transformers
HF_HOME=/shared/models/hf

# -----------------------------------------------------------------------------
# MinIO (S3-compatible storage)
# -----------------------------------------------------------------------------
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

UMRS_SERVER_ZMQ_HOST=ai-login
POSTGRES_HOST=postgres
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# S3 credentials for rivault_user (auto-generated)
S3_RIVAULT_USER_PASSWORD=${S3_RIVAULT_PASSWORD}

# -----------------------------------------------------------------------------
# Offline Mode
# -----------------------------------------------------------------------------
TRANSFORMERS_OFFLINE=0
HF_HUB_OFFLINE=0

# Force Slurm backend
UMRS_BACKEND=slurm
JOB_INFO_DIR=/var/run/rivault

# -----------------------------------------------------------------------------
# RAG Service
# -----------------------------------------------------------------------------
UMRS_RAG_ENABLE=1
UMRS_RAG_PORT=8081
VLLM_BASE_URL=http://localhost:14000/v1
ENVEOF

ok ".env generated"

# =============================================================================
# Phase 5: Generate slurm.conf (patch node definition)
# =============================================================================
info "Phase 5: Updating slurm.conf"

GRES_STR=""
[[ "$NODE_GPUS" -gt 0 ]] && GRES_STR=" Gres=gpu:${NODE_GPUS}"

# Replace the NodeName line
sed -i "s|^NodeName=compute-01 .*|NodeName=compute-01 CPUs=${NODE_CPUS} RealMemory=${NODE_MEM}${GRES_STR} State=UNKNOWN|" slurm.conf
ok "slurm.conf: compute-01 → CPUs=${NODE_CPUS} RealMemory=${NODE_MEM}${GRES_STR}"

# =============================================================================
# Phase 6: Generate gres.conf
# =============================================================================
info "Phase 6: Updating gres.conf"

if [[ "$NODE_GPUS" -gt 0 ]] && $HAS_GPU; then
    cat > gres.conf << GRESEOF
# GRES config for Slurm — generated by init-setup.sh on $(date -Iseconds)
# GPU devices auto-detected from nvidia-smi.

NodeName=compute-01 Name=gpu File=${GPU_DEVICES} Count=${NODE_GPUS}
GRESEOF
    ok "gres.conf: ${NODE_GPUS} GPU(s) at ${GPU_DEVICES}"
else
    cat > gres.conf << GRESEOF
# GRES config for Slurm — generated by init-setup.sh on $(date -Iseconds)
# No GPUs detected. Add GPU entries manually if needed.
GRESEOF
    ok "gres.conf: no GPUs configured"
fi

# =============================================================================
# Phase 7: Generate s3cfg_rivault_user
# =============================================================================
info "Phase 7: Updating s3cfg_rivault_user"

cat > s3cfg_rivault_user << S3EOF
[default]
host_base = minio:9000
host_bucket = minio:9000/%(bucket)s
use_https = False
check_ssl_certificate = False
access_key = rivault_user
secret_key = ${S3_RIVAULT_PASSWORD}
bucket_location = us-east-1
S3EOF

ok "s3cfg_rivault_user: password auto-generated"

# =============================================================================
# Phase 8: Ensure required directories exist
# =============================================================================
info "Phase 8: Ensuring directories"

mkdir -p munge-key job-info umrs-config rivault/models
ok "munge-key/ job-info/ umrs-config/ rivault/models/ created"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Host FQDN:       $COMPUTE_HOST_FQDN"
echo "  Slurm node:      compute-01 (${NODE_CPUS} CPUs, ${NODE_MEM}MB RAM, ${NODE_GPUS} GPU(s))"
echo "  vLLM tag:         $VLLM_DOCKER_TAG"
echo "  vLLM model:       $VLLM_MODEL"
echo "  S3 rivault_user:  (password auto-generated in .env and s3cfg_rivault_user)"
echo ""
echo "  Next steps:"
echo "    1. Start control plane:  podman-compose up -d"
echo "    2. Start host slurmd:    sudo bash scripts/start-host-slurmd.sh"
echo "    3. Run E2E demo:         bash scripts/run_e2e_demo.sh"
echo ""
