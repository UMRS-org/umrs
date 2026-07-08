#!/usr/bin/env bash
# =============================================================================
# stop.sh — tear down the login-node UMRS deployment (TEMPLATE)
# =============================================================================
#   ./stop.sh              # stop native server + infra containers
#   ./stop.sh --server     # stop only the native server (keep containers)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
PID_FILE="${SCRIPT_DIR}/server.pid"
STOP_MODE="${1:-}"

# ─── Native UMRS server ──────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping UMRS server (PID ${pid})..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# ─── Infra containers ────────────────────────────────────────────────────────
if [[ "$STOP_MODE" != "--server" ]]; then
    echo "Stopping infra containers (data preserved in volumes)..."
    podman compose -f "${SCRIPT_DIR}/docker-compose.yaml" down
fi

echo "Stopped."
