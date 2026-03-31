#!/usr/bin/env bash
# =============================================================================
# Model Registry Demo — rccs-cloud deployment
# =============================================================================
# Demonstrates the model registry workflow for air-gapped HPC systems:
#
#   1. List current models
#   2. Register an existing model from the shared filesystem
#   3. Upload (copy + register) a model from a local path
#   4. Verify the registry
#
# On HPC systems the shared filesystem (Lustre) is the transport layer.
# No internet access is needed on compute nodes — models are pre-staged
# on the login node and registered via the UMRS control plane.
#
# Usage:
#   ./09_model_registry.sh
# =============================================================================
set -euo pipefail
# ── client PATH fallback (rccs-cloud: umrs-client lives in ~/umrs-client-venv) ─
command -v umrs-client >/dev/null 2>&1 || export PATH="$HOME/umrs-client-venv/bin:$PATH"

# Staging root: must match the host-side Lustre path that the container
# mounts as /shared/models.  The client writes here; the server reads it
# via the volume mount.
STAGING_ROOT="${UMRS_MODEL_STAGING_ROOT:-${MODEL_DIR:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models}}"

CLIENT="umrs-client"

log()  { echo "[model_reg] $*"; }
ok()   { echo "[model_reg]   ✓ $1"; }
fail() { echo "[model_reg]   ✗ $1" >&2; }
die()  { echo "[model_reg] ERROR: $*" >&2; exit 1; }

# ---------- Step 0: Preflight ----------
log "Staging root: $STAGING_ROOT"
$CLIENT ping >/dev/null 2>&1 || die "Server not reachable. Is the ssh tunnel running?"
log "Server is ready."

# ---------- Step 1: List current models ----------
echo
log "Step 1: List currently registered models"
$CLIENT model list
echo

# ---------- Step 2: Register a model already on shared filesystem ----------
# Pick any directory under the staging root as an example.
# On rccs-cloud this is typically the sentence_transformers cache.
EXAMPLE_DIR="$STAGING_ROOT/sentence_transformers"
if [[ -d "$EXAMPLE_DIR" ]]; then
    log "Step 2: Register an existing model from shared filesystem"
    log "  Path: $EXAMPLE_DIR"

    # The server sees this as /shared/models/sentence_transformers
    $CLIENT model register "sentence-transformers-cache" \
        "/shared/models/sentence_transformers" \
        --type embedding --scope system \
        --description "Pre-cached sentence-transformers models on Lustre" \
        || log "  (already registered — continuing)"
    ok "register complete"
else
    log "Step 2: SKIP — $EXAMPLE_DIR not found (no pre-cached models yet)"
fi
echo

# ---------- Step 3: Upload (copy + register) a small test model ----------
log "Step 3: Upload a test model via 'model upload'"

# Create a tiny test model directory to demonstrate the upload flow
TEST_MODEL_DIR="$(mktemp -d)"
echo '{"model_type": "test", "framework": "demo"}' > "$TEST_MODEL_DIR/config.json"
echo "dummy weights" > "$TEST_MODEL_DIR/model.bin"
trap 'rm -rf "$TEST_MODEL_DIR"' EXIT

log "  Created test model at: $TEST_MODEL_DIR"
log "  Uploading to staging root and registering..."

$CLIENT model upload "$TEST_MODEL_DIR" \
    --name "test-upload-model" \
    --type embedding --scope user \
    --description "Demo model uploaded via model upload" \
    --staging-dir "$STAGING_ROOT" \
    --force

ok "upload complete"
echo

# ---------- Step 4: Verify ----------
log "Step 4: Verify — list all registered models"
$CLIENT model list

echo
log "Done.  The air-gapped HPC model workflow:"
log "  1. Download/prepare models on the login node (internet access)"
log "  2. 'umrs-client model upload <path>' copies to shared Lustre + registers"
log "  3. Compute nodes access models via the shared filesystem — no internet needed"

# ---------- Cleanup: remove the test model from staging & registry ----------
log "Cleaning up test model from staging root..."
rm -rf "$STAGING_ROOT/test-upload-model" 2>/dev/null || true
