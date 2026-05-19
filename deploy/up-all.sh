#!/usr/bin/env bash
# =============================================================================
# up-all.sh — Launch endpoints on both RCCS Cloud and AI4S simultaneously
# =============================================================================
# Provisions:
#   - RCCS Cloud: Bonsai (via rccs-front gateway on port 8000)
#   - AI4S:       DeepSeek-V4-Pro, GLM-5.1, Kimi k2.5 (via ai4s gateway on port 8002)
#
# Prerequisites:
#   1. rccs-front stack running: cd ../rccs-front && ./start.sh
#   2. ai4s stack running:       cd ../ai4s && ./start.sh
#   3. r4 CLI installed:         pip install -e ../../r4
#
# Usage:
#   ./up-all.sh           # Launch all endpoints (parallel per cluster)
#   ./up-all.sh --dry-run # Preview without submitting
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCS_FRONT_DIR="${SCRIPT_DIR}/../rccs-front"
AI4S_DIR="${SCRIPT_DIR}/../ai4s"

# Gateway URLs (default ports for co-located stacks on umrs-front)
RCCS_GATEWAY="${RCCS_GATEWAY_URL:-http://localhost:8000}"
AI4S_GATEWAY="${AI4S_GATEWAY_URL:-http://localhost:8002}"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Multi-Cluster Endpoint Provisioning                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  RCCS Cloud: ${RCCS_GATEWAY}  (Bonsai)              ║"
echo "║  AI4S:       ${AI4S_GATEWAY}  (DeepSeek, GLM, Kimi) ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# --- RCCS Cloud ---
echo "▶ [rccs-cloud] Provisioning Bonsai..."
R4_GATEWAY_URL="${RCCS_GATEWAY}" r4 pool up "${RCCS_FRONT_DIR}/pool-rccs-cloud.yaml" ${DRY_RUN} &
PID_RCCS=$!

# --- AI4S ---
echo "▶ [ai4s] Provisioning DeepSeek-V4-Pro, GLM-5.1, Kimi k2.5..."
R4_GATEWAY_URL="${AI4S_GATEWAY}" r4 pool up "${AI4S_DIR}/pool-ai4s.yaml" ${DRY_RUN} &
PID_AI4S=$!

# Wait for both
echo ""
echo "⏳ Waiting for both clusters to complete provisioning..."
echo ""

FAIL=0
wait $PID_RCCS || { echo "✗ RCCS Cloud provisioning failed (exit=$?)"; FAIL=1; }
wait $PID_AI4S || { echo "✗ AI4S provisioning failed (exit=$?)"; FAIL=1; }

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "✓ All endpoints provisioned successfully"
    echo ""
    echo "  RCCS Cloud endpoints:  R4_GATEWAY_URL=${RCCS_GATEWAY} r4 endpoint list"
    echo "  AI4S endpoints:        R4_GATEWAY_URL=${AI4S_GATEWAY} r4 endpoint list"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "═══════════════════════════════════════════════════════════════"
    echo "⚠ Some provisioning failed. Check logs above."
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi
