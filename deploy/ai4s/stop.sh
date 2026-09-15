#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Stop UMRS AI4S deployment
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping UMRS AI4S services..."
podman-compose down

echo "Services stopped."
