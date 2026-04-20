#!/usr/bin/env bash
# =============================================================================
# Demo 21: Remote Plugin Execution (Local Simulation)
# =============================================================================
# Demonstrates the plugin system with different source types and
# execution backends. Runs locally — no server required.
#
# Shows:
#   1. PluginSpec definition (git repo, S3/wheel, local path)
#   2. Plugin validation
#   3. Plugin execution simulation
#   4. Audit log of plugin operations
#
# Prerequisites:
#   - Python 3.10+ with umrs package installed (pip install -e src/)
#   - No server required (local simulation)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="plugins" source "${SCRIPT_DIR}/../lib/common.sh"

RESULTS_DIR="${SCRIPT_DIR}/results"

banner "Demo 21: Remote Plugin Execution"
echo "This demo runs locally — no server required."
echo ""

# ---------- Step 1: Prepare plugin definitions ----------
log "Step 1: Preparing plugin definitions..."
mkdir -p "$RESULTS_DIR"

python3 "${SCRIPT_DIR}/simulate_plugins.py" "$RESULTS_DIR"

ok "Plugin simulation complete"

log ""
log "Results saved to: ${RESULTS_DIR}/"

banner "Demo 21 Complete"
