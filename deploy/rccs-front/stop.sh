#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Stop UMRS front-end server deployment
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping UMRS front-end services..."
podman-compose down

echo "Services stopped."
