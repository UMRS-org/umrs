#!/usr/bin/env bash
# =============================================================================
# Stop the UMRS stack on hpbd-atom — hybrid architecture.
# =============================================================================
# Stops the native umrs-server process, then the infrastructure containers.
#
# Usage:
#   ./stop.sh          # graceful stop (data preserved in volumes)
#   ./stop.sh --clean  # stop + remove volumes (data lost!)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PID_FILE="./server.pid"
LOG_FILE="./logs/server.log"

# ─── Stop the native UMRS server ────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    SERVER_PID=$(cat "$PID_FILE")
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping umrs-server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        # Wait up to 10s for graceful shutdown
        for i in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.5
        done
        # Force kill if still running
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  Force-killing server..."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
        echo "  Server stopped."
    else
        echo "umrs-server (PID $SERVER_PID) is not running."
    fi
    rm -f "$PID_FILE"
else
    echo "No server.pid — umrs-server not managed by start.sh (or already stopped)."
fi

# ─── Best-effort: cancel any Slurm jobs started by server ────────────────────
if command -v squeue &>/dev/null; then
    SLURM_JOBS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep -i 'rivault\|serve-model' | awk '{print $1}' || true)
    if [[ -n "$SLURM_JOBS" ]]; then
        echo "Cancelling Slurm jobs: $SLURM_JOBS"
        # shellcheck disable=SC2086
        scancel $SLURM_JOBS 2>/dev/null || true
    fi
fi

# ─── Stop infrastructure containers ─────────────────────────────────────────
if [[ "${1:-}" = "--clean" ]]; then
    echo "Stopping containers and removing volumes..."
    podman compose down -v
else
    echo "Stopping containers (data preserved in volumes)..."
    podman compose down
fi

echo ""
echo "Done. To restart: ./start.sh"
