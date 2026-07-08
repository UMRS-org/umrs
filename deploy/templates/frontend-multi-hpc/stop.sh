#!/usr/bin/env bash
# =============================================================================
# stop.sh — tear down the frontend → multi-HPC stack (TEMPLATE)
# =============================================================================
#   ./stop.sh              # stop UMRS servers + infra (volumes preserved)
#   ./stop.sh --server     # stop only the UMRS servers (keep infra)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
COMPOSE="podman compose -f $(pwd)/docker-compose.yaml"
STOP_MODE="${1:-}"

if [[ "$STOP_MODE" == "--server" ]]; then
    echo "Stopping UMRS servers (infra kept)..."
    $COMPOSE stop umrs-server-cluster-a umrs-server-cluster-b
else
    echo "Stopping full stack (data preserved in volumes)..."
    $COMPOSE down
fi

echo "Stopped."
