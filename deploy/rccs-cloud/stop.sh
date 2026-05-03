#!/usr/bin/env bash
# =============================================================================
# Stop the UMRS stack on rccs-cloud — hybrid architecture.
# =============================================================================
# Stops the native umrs-server process and, unless `--server` is requested,
# also tears down the infrastructure containers.
#
# Usage:
#   ./stop.sh                  # full stop (data preserved in volumes)
#   ./stop.sh --server         # stop only the native server (containers stay)
#   ./stop.sh --postgres-clean # full stop + remove only postgres volume
#   ./stop.sh --clean          # full stop + remove volumes (data lost!)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PID_FILE="./server.pid"
RAG_PID_FILE="./rag-service.pid"
LOG_FILE="./logs/server.log"
MODE="full"

usage() {
        cat <<'EOF'
Usage: ./stop.sh [--server|--postgres-clean|--clean]

Stop the RCCS UMRS stack.

Options:
    --server          Stop only the native UMRS server and RAG service; keep containers and Slurm jobs running.
    --postgres-clean  Full stop, then remove only the postgres volume.
    --clean           Full stop, remove all compose volumes. This deletes local deployment state.
    -h, --help        Show this help and exit without changing running services.
EOF
}

compose_project_name() {
    if [[ -n "${COMPOSE_PROJECT_NAME:-}" ]]; then
        printf '%s\n' "$COMPOSE_PROJECT_NAME"
    else
        basename "$PWD"
    fi
}

postgres_volume_name() {
    printf '%s_pg_data\n' "$(compose_project_name)"
}

remove_volume_if_present() {
    local volume_name="$1"

    if podman volume inspect "$volume_name" >/dev/null 2>&1; then
        echo "Removing volume: ${volume_name}"
        podman volume rm -f "$volume_name" >/dev/null
    else
        echo "Volume ${volume_name} not present — nothing to remove."
    fi
}

case "${1:-}" in
    "" )
        ;;
    --server )
        MODE="server"
        ;;
    --postgres-clean )
        MODE="postgres-clean"
        ;;
    --clean )
        MODE="clean"
        ;;
    -h|--help )
        usage
        exit 0
        ;;
    * )
        echo "ERROR: Unknown option: ${1}"
        usage
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

if [[ "$MODE" = "server" ]]; then
    echo "Server-only stop requested — leaving Slurm jobs and infrastructure containers running."
    echo ""
    echo "Done. To restart only the server: ./start.sh --server"
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
elif [[ "$MODE" = "postgres-clean" ]]; then
    echo "Stopping containers and removing only the postgres volume..."
    podman compose down
    remove_volume_if_present "$(postgres_volume_name)"
else
    echo "Stopping containers (data preserved in volumes)..."
    podman compose down
fi

echo ""
if [[ "$MODE" = "postgres-clean" ]]; then
    echo "Done. Postgres state was reset; MinIO data was preserved. To restart: ./start.sh"
else
    echo "Done. To restart: ./start.sh"
fi
