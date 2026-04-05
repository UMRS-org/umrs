#!/usr/bin/env bash
# =============================================================================
# Demo 01: Health Check
# =============================================================================
# Verify that the UMRS server is reachable and the user's configuration is
# valid.  This is the simplest possible demo — run it first to confirm
# connectivity before trying any data pipeline.
#
# Prerequisites:
#   - SSH tunnels active (ZMQ 5555/5556, MinIO 9000)
#   - umrs-client installed and in PATH
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="health_check" source "${SCRIPT_DIR}/../lib/common.sh"

banner "Demo 01: Health Check"
echo "User config : ${CONF}"
echo ""

# Step 1: Ping the server
log "Step 1: Ping server..."
preflight
echo ""

# Step 2: List configured buckets
log "Step 2: List configured buckets..."
$CLIENT list_buckets || log "(no buckets configured yet — that's OK for a fresh install)"
echo ""

banner "Demo 01 Complete"
log "The server is reachable.  Proceed with Demo 02 (Text Embedding)."
