#!/usr/bin/env bash
# =============================================================================
# One-time server setup for UMRS on hpbd-atom (login node).
# =============================================================================
# Creates a Python virtual environment, installs server dependencies, and
# links the Slurm wrapper script to the expected config path.
#
# Run this ONCE after cloning the repo or after major dependency changes.
#
# Usage (on hpbd-atom):
#   cd deploy/hpbd-atom
#   ./setup_server.sh
#
# After setup, use ./start.sh to launch the server.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"
SRC_DIR="${REPO_ROOT}/src"
VENV_DIR="${SCRIPT_DIR}/.venv"
RIVAULT_DATA_DIR="${HOME}/.config/umrs/rivault_data"

log()  { echo "[setup] $*"; }
die()  { echo "[setup] ERROR: $*" >&2; exit 1; }

# ─── Guard: must run on a Linux host ─────────────────────────────────────────
if [[ "$(uname -s)" != "Linux" ]]; then
    die "setup_server.sh must be run on hpbd-atom (Linux), not on $(uname -s)."
fi

[[ -f "${SRC_DIR}/pyproject.toml" ]] || die "Cannot find ${SRC_DIR}/pyproject.toml — is the repo intact?"

# ─── Find suitable Python (3.10+) ───────────────────────────────────────────
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    # Fall back to python3 if it's 3.10+
    PY_VER=$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
    if [[ "$PY_VER" -ge 10 ]]; then
        PYTHON=python3
    else
        die "Python 3.10+ required but not found. Available: $(python3 --version 2>&1)"
    fi
fi
log "Using $($PYTHON --version 2>&1)"

# ─── Create virtual environment ──────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    log "Virtual environment already exists at ${VENV_DIR}"
    log "  To rebuild: rm -rf ${VENV_DIR} && ./setup_server.sh"
else
    log "Creating virtual environment at ${VENV_DIR} ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ─── Upgrade pip ─────────────────────────────────────────────────────────────
log "Upgrading pip..."
pip install --quiet --upgrade pip

# ─── Install PyTorch (CPU-only — optional fallback) ──────────────────────────
# All production inference (embeddings, vision, LLMs, custom models) runs on
# GPU compute nodes via Slurm.  CPU-only torch is installed as an optional
# fallback for debugging or when GPU nodes are unavailable.  It enables local
# sentence-transformers mode if model_type is set to TransformerEncoder or
# SentenceTransformer instead of the default rivault://openai:// scheme.
log "Installing PyTorch (CPU-only, optional fallback) ..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ─── Install server requirements ─────────────────────────────────────────────
log "Installing server dependencies from ${SRC_DIR}/requirements.txt ..."
# Some packages (torch, torchvision) are already installed above;
# pip will skip them if the version satisfies the requirement.
pip install --quiet -r "${SRC_DIR}/requirements.txt"

# ─── Install r4-contracts when available (shared DTOs required by umrs) ─────
R4_CONTRACTS_DIR="${R4_CONTRACTS_DIR:-${REPO_ROOT}/../r4-contracts}"
if [[ -f "${R4_CONTRACTS_DIR}/pyproject.toml" ]]; then
    log "Installing r4-contracts (editable) from ${R4_CONTRACTS_DIR} ..."
    pip install --quiet -e "${R4_CONTRACTS_DIR}"
else
    log "r4-contracts not found at ${R4_CONTRACTS_DIR} — pip will resolve from the package index."
fi

# ─── Install umrs package in editable mode ───────────────────────────────────
log "Installing umrs package (editable mode) ..."
pip install --quiet -e "${SRC_DIR}"

# ─── Verify key imports ─────────────────────────────────────────────────────
log "Verifying installation..."
python3 -c "
import r4_contracts; print('  r4-contracts OK')
import umrs; print(f'  umrs package OK ({umrs.__file__})')
import torch; print(f'  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
from sentence_transformers import SentenceTransformer; print('  sentence-transformers OK')
" || die "Import verification failed — check the installation."

# ─── Link the Slurm wrapper script ──────────────────────────────────────────
# rivault_launcher.py (via config.py) looks for the wrapper at:
#   ~/.config/umrs/rivault_data/serve-model.slurm
# Symlink our hpbd-atom-specific wrapper there.
mkdir -p "$RIVAULT_DATA_DIR"

SLURM_TARGET="${SCRIPT_DIR}/serve-model.slurm"
SLURM_LINK="${RIVAULT_DATA_DIR}/serve-model.slurm"

if [[ -L "$SLURM_LINK" ]]; then
    log "Updating symlink: ${SLURM_LINK} → ${SLURM_TARGET}"
    ln -sf "$SLURM_TARGET" "$SLURM_LINK"
elif [[ -e "$SLURM_LINK" ]]; then
    log "WARNING: ${SLURM_LINK} exists but is not a symlink — backing up."
    mv "$SLURM_LINK" "${SLURM_LINK}.bak.$(date +%s)"
    ln -s "$SLURM_TARGET" "$SLURM_LINK"
else
    log "Creating symlink: ${SLURM_LINK} → ${SLURM_TARGET}"
    ln -s "$SLURM_TARGET" "$SLURM_LINK"
fi

# ─── Install MinIO Client (mc) ───────────────────────────────────────────────
# The UMRS server calls `mc` for user/policy management in minio.py.
MC_BIN="${VENV_DIR}/bin/mc"
if [[ -x "$MC_BIN" ]]; then
    log "MinIO Client (mc) already installed at ${MC_BIN}"
else
    log "Installing MinIO Client (mc) ..."
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)  MC_ARCH="amd64" ;;
        aarch64) MC_ARCH="arm64" ;;
        *)       die "Unsupported architecture for mc: ${ARCH}" ;;
    esac
    # dl.min.io died with the MinIO archive; mc still releases from GitHub.
    MC_TAG="RELEASE.2025-08-13T08-35-41Z"
    MC_URL="https://github.com/minio/mc/releases/download/${MC_TAG}/mc.linux-${MC_ARCH}.${MC_TAG}"
    if command -v curl &>/dev/null; then
        curl -fsSL "$MC_URL" -o "$MC_BIN"
    elif command -v wget &>/dev/null; then
        wget -q "$MC_URL" -O "$MC_BIN"
    else
        die "Neither curl nor wget found — cannot download mc."
    fi
    chmod +x "$MC_BIN"
    log "MinIO Client installed at ${MC_BIN}"
fi

# ─── Copy admin s3cfg (server needs it to create per-user credentials) ───────
mkdir -p "${HOME}/.config/umrs"
if [[ ! -f "${HOME}/.config/umrs/s3cfg" ]]; then
    log "Copying admin s3cfg to ~/.config/umrs/s3cfg"
    cp "${SCRIPT_DIR}/s3cfg" "${HOME}/.config/umrs/s3cfg"
elif ! diff -q "${SCRIPT_DIR}/s3cfg" "${HOME}/.config/umrs/s3cfg" >/dev/null 2>&1; then
    log "NOTE: ~/.config/umrs/s3cfg differs from ${SCRIPT_DIR}/s3cfg — keeping existing."
fi

# ─── .env setup ──────────────────────────────────────────────────────────────
if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
    log "Copying env.example to .env — edit it before starting."
    cp "${SCRIPT_DIR}/env.example" "${SCRIPT_DIR}/.env"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
log "=== Server setup complete ==="
echo ""
echo "  Venv:       ${VENV_DIR}"
echo "  Slurm link: ${SLURM_LINK} → ${SLURM_TARGET}"
echo "  Config:     ${SCRIPT_DIR}/conf_umrs.json"
echo ""
echo "Next steps:"
echo "  1. Edit .env if needed (MODEL_DIR, passwords, etc.)"
echo "  2. Start services:  ./start.sh"
echo "  3. From your Mac:   ./ssh_tunnel.sh"
echo "  4. Run demos:       cd demos && ./00_health_check.sh"
