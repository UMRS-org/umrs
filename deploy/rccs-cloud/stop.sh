#!/usr/bin/env bash
# Stop the UMRS server stack on rccs-cloud.
#
# Usage:
#   ./stop.sh          # stop containers (data preserved)
#   ./stop.sh --clean  # stop + remove volumes (data lost!)

set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--clean" ]; then
    echo "Stopping and removing volumes..."
    podman compose down -v
else
    echo "Stopping services (data preserved in volumes)..."
    podman compose down
fi

echo "Done."
