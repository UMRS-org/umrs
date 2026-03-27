#!/usr/bin/env bash
# Stop the UMRS server stack on rccs-cloud.
#
# Usage:
#   ./stop.sh          # stop containers (data preserved)
#   ./stop.sh --clean  # stop + remove volumes (data lost!)

set -euo pipefail
cd "$(dirname "$0")"

select_compose_cmd() {
    if command -v podman-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(podman-compose)
        return 0
    fi

    if podman compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(podman compose)
        return 0
    fi

    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
        return 0
    fi

    cat <<'EOF'
Error: no working compose backend found.
Install one of:
1) pip install --user podman-compose
2) docker compose plugin
EOF
    return 1
}

select_compose_cmd

if [ "${1:-}" = "--clean" ]; then
    echo "Stopping and removing volumes..."
    "${COMPOSE_CMD[@]}" down -v
else
    echo "Stopping services (data preserved in volumes)..."
    "${COMPOSE_CMD[@]}" down
fi

echo "Done."
