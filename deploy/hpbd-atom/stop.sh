#!/usr/bin/env bash
# =============================================================================
# Stop the UMRS stack on hpbd-atom — hybrid architecture.
# =============================================================================
# Stops the native umrs-server process, then the infrastructure containers.
#
# Usage:
#   ./stop.sh            # graceful stop (data preserved in volumes)
#   ./stop.sh --server   # stop only the server (containers keep running)
#   ./stop.sh --clean    # stop + remove volumes (data lost!)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PID_FILE="./server.pid"
RAG_PID_FILE="./rag-service.pid"
LOG_FILE="./logs/server.log"

MODE="full"

case "${1:-}" in
    "")
        ;;
    --server)
        MODE="server"
        ;;
    --clean)
        MODE="clean"
        ;;
    *)
        echo "ERROR: Unknown option: ${1}"
        echo "Usage: ./stop.sh [--server|--clean]"
        exit 1
        ;;
esac

stop_managed_process() {
    local pid_file="$1"
    local label="$2"
    local missing_message="${3:-}"

    if [[ ! -f "$pid_file" ]]; then
        if [[ -n "$missing_message" ]]; then
            echo "$missing_message"
        fi
        return 0
    fi

    local process_pid
    process_pid=$(cat "$pid_file")
    if kill -0 "$process_pid" 2>/dev/null; then
        echo "Stopping ${label} (PID ${process_pid})..."
        kill "$process_pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$process_pid" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$process_pid" 2>/dev/null; then
            echo "  Force-killing ${label}..."
            kill -9 "$process_pid" 2>/dev/null || true
        fi
        echo "  ${label} stopped."
    else
        echo "${label} (PID ${process_pid}) is not running."
    fi
    rm -f "$pid_file"
}

# ─── Stop the native UMRS server ────────────────────────────────────────────
stop_managed_process "$PID_FILE" "umrs-server" \
    "No server.pid — umrs-server not managed by start.sh (or already stopped)."
stop_managed_process "$RAG_PID_FILE" "rag-service"

# ─── Server-only mode: skip containers and Slurm cleanup ────────────────────
if [[ "$MODE" = "server" ]]; then
    echo ""
    echo "Done (server only). Containers still running."
    exit 0
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
if [[ "$MODE" = "clean" ]]; then
    echo "Stopping containers and removing volumes..."
    podman compose down -v
else
    echo "Stopping containers (data preserved in volumes)..."
    podman compose down
fi

echo ""
echo "Done. To restart: ./start.sh"
