#!/usr/bin/env bash
# =============================================================================
# Model Registry Demo — hpbd-atom deployment
# =============================================================================
# Demonstrates the model registry workflow for air-gapped HPC systems:
#
#   1. List current models
#   2. Register an existing model from the shared filesystem
#   3. Upload a real HMAE model from local Mac (optional)
#   3b. Upload a small test model
#   4. Verify the registry
#
# On HPC systems the shared filesystem (home) is the transport layer.
# No internet access is needed on compute nodes — models are pre-staged
# on the login node and registered via the UMRS control plane.
#
# Usage:
#   ./09_model_registry.sh
#   HMAE_LOCAL_MODEL_DIR=~/Downloads/hmae_checkpoint ./09_model_registry.sh
# =============================================================================
set -euo pipefail
# ── client PATH fallback (hpbd-atom: umrs-client lives in ~/umrs-client-venv) ─
command -v umrs-client >/dev/null 2>&1 || export PATH="$HOME/umrs-client-venv/bin:$PATH"

# Staging root: the server-side path where models are stored.
STAGING_ROOT="${UMRS_MODEL_STAGING_ROOT:-${MODEL_DIR:-/home/mnagaso/workspace/umrs_models}}"
REMOTE_HMAE_MODEL_PATH="${UMRS_REMOTE_HMAE_MODEL_PATH:-/home/mnagaso/workspace/umrs_models/hmae}"

CLIENT="umrs-client"

log()  { echo "[model_reg] $*"; }
ok()   { echo "[model_reg]   ✓ $1"; }
fail() { echo "[model_reg]   ✗ $1" >&2; }
die()  { echo "[model_reg] ERROR: $*" >&2; exit 1; }

# ---------- Step 0: Preflight ----------
log "Staging root (server-side): $STAGING_ROOT"
$CLIENT ping >/dev/null 2>&1 || die "Server not reachable. Is the ssh tunnel running?"
log "Server is ready."

# ---------- Step 1: List current models ----------
echo
log "Step 1: List currently registered models"
$CLIENT model list
echo

# ---------- Step 2: Register a model already on shared filesystem ----------
# Pick any directory under the staging root as an example.
EXAMPLE_DIR="$STAGING_ROOT/sentence_transformers"
if [[ -d "$EXAMPLE_DIR" ]]; then
    log "Step 2: Register an existing model from shared filesystem"
    log "  Path: $EXAMPLE_DIR"

    # The server sees this as /shared/models/sentence_transformers
    $CLIENT model register "sentence-transformers-cache" \
        "/shared/models/sentence_transformers" \
        --type embedding --scope system \
        --description "Pre-cached sentence-transformers models on shared home" \
        || log "  (already registered — continuing)"
    ok "register complete"
else
    log "Step 2: SKIP — $EXAMPLE_DIR not found (no pre-cached models yet)"
fi
echo

# ---------- Step 3: Upload a real HMAE model (if available) ----------
# The HMAE model checkpoint is typically trained offline and stored locally.
# Set HMAE_LOCAL_MODEL_DIR to point at the directory containing
# latest_checkpoint.pth.  Uses 'model upload --register-path' to copy
# locally and register the server-side path with the UMRS server.
HMAE_LOCAL_MODEL_DIR="${HMAE_LOCAL_MODEL_DIR:-}"

if [[ -n "$HMAE_LOCAL_MODEL_DIR" ]] && [[ -d "$HMAE_LOCAL_MODEL_DIR" ]] \
   && [[ -f "$HMAE_LOCAL_MODEL_DIR/latest_checkpoint.pth" ]]; then
    log "Step 3a: Upload real HMAE model via 'model upload'"
    log "  Source (local):      $HMAE_LOCAL_MODEL_DIR"
    log "  Register path (HPC): $REMOTE_HMAE_MODEL_PATH"

    $CLIENT model upload "$HMAE_LOCAL_MODEL_DIR" \
        --name "hmae-pretrained" \
        --type embedding --scope system \
        --description "HMAE ViT pretrained checkpoint (s8d-1k)" \
        --register-path "$REMOTE_HMAE_MODEL_PATH" \
        --force

    ok "HMAE model upload complete"
else
    if [[ -n "$HMAE_LOCAL_MODEL_DIR" ]]; then
        log "Step 3a: SKIP — HMAE_LOCAL_MODEL_DIR set but not found or missing checkpoint:"
        log "   $HMAE_LOCAL_MODEL_DIR"
    else
        log "Step 3a: SKIP — HMAE_LOCAL_MODEL_DIR not set (set it to upload the HMAE checkpoint)"
    fi
fi
echo

# ---------- Step 3b: Upload a small test model ----------
log "Step 3b: Upload a test model via 'model upload'"

# Create a tiny test model directory to demonstrate the upload flow
TEST_MODEL_DIR="$(mktemp -d)"
echo '{"model_type": "test", "framework": "demo"}' > "$TEST_MODEL_DIR/config.json"
echo "dummy weights" > "$TEST_MODEL_DIR/model.bin"
trap 'rm -rf "$TEST_MODEL_DIR"' EXIT

REMOTE_TEST_PATH="$STAGING_ROOT/test-upload-model"
log "  Created test model at: $TEST_MODEL_DIR"
log "  Register path (HPC):   $REMOTE_TEST_PATH"

$CLIENT model upload "$TEST_MODEL_DIR" \
    --name "test-upload-model" \
    --type embedding --scope user \
    --description "Demo model uploaded via model upload" \
    --register-path "$REMOTE_TEST_PATH" \
    --force

ok "test model upload complete"
echo

# ---------- Step 4: Verify ----------
log "Step 4: Verify — list all registered models"
$CLIENT model list

echo
log "Done.  The air-gapped HPC model workflow:"
log "  1. Download/prepare models on your local machine"
log "  2. 'umrs-client model upload <path> --register-path <server-path>' copies locally + registers server path"
log "  3. Compute nodes access models via the shared filesystem — no internet needed"
log ""
log "  Tip: set HMAE_LOCAL_MODEL_DIR to upload the HMAE checkpoint:"
log "    HMAE_LOCAL_MODEL_DIR=/path/to/hmae_checkpoint ./09_model_registry.sh"

# ---------- Cleanup: remove the test model from staging & registry ----------
log "Cleaning up test model..."
rm -rf "${TMPDIR:-/tmp}/umrs_model_staging/test-upload-model" 2>/dev/null || true
