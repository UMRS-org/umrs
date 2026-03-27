#!/usr/bin/env bash
# Client setup script for UMRS on local Mac.
# Copies client configs to ~/.config/umrs/ and installs the umrs package.
#
# Usage:
#   ./setup_client.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLIENT_DIR="${SCRIPT_DIR}/client"
CONFIG_DIR="${HOME}/.config/umrs"

echo "=== UMRS Client Setup ==="

# Create config directories
mkdir -p "${CONFIG_DIR}/users"

# Copy config files
cp "${CLIENT_DIR}/conf_umrs.json" "${CONFIG_DIR}/conf_umrs.json"
cp "${CLIENT_DIR}/s3cfg"          "${CONFIG_DIR}/s3cfg"
cp "${CLIENT_DIR}/user1.json"     "${CONFIG_DIR}/users/user1.json"

echo "Configs installed to ${CONFIG_DIR}/"
echo "  conf_umrs.json"
echo "  s3cfg"
echo "  users/user1.json"

# Install umrs package in editable mode
SRC_DIR="${SCRIPT_DIR}/../../src"
if [ -f "${SRC_DIR}/pyproject.toml" ]; then
    echo ""
    echo "Installing umrs package from ${SRC_DIR} ..."
    uv pip install -e "${SRC_DIR}"
    echo "umrs package installed."
else
    echo ""
    echo "WARNING: Could not find ${SRC_DIR}/pyproject.toml"
    echo "Install manually: uv pip install -e /path/to/umrs/src"
fi

echo ""
echo "=== Client setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Start SSH tunnels:  ./ssh_tunnel.sh"
echo "  2. Run client:         python -c 'from umrs.client_umrs import UMRSClient; ...'"
