#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

PROVISION="${DEMO25_PROVISION:-0}"
MOUNT_ROOT="${MINI_RIVAULT_MOUNT_DEMO_ROOT:-/tmp/mini-rivault-mounts}"

mkdir -p "${MOUNT_ROOT}/mcp-servers/demo-tools"
mkdir -p "${MOUNT_ROOT}/project"
mkdir -p "${MOUNT_ROOT}/rag-index"

cat <<JSON
{
  "mounts": [
    {"source": "${MOUNT_ROOT}/mcp-servers/demo-tools", "target": "/mcp-servers/demo-tools", "mode": "ro"},
    {"source": "${MOUNT_ROOT}/project", "target": "/project", "mode": "ro"},
    {"source": "${MOUNT_ROOT}/rag-index", "target": "/rag-index", "mode": "ro"}
  ],
  "egress_domains": ["api.example.com"]
}
JSON

echo "RIVAULT_EXTRA_PROFILES=custom-mcp egress rag"
echo "Set DEMO25_PROVISION=1 to actually provision an endpoint."
