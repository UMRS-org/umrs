#!/usr/bin/env bash
# Client setup script for UMRS on local Mac.
# Copies client configs to ~/.config/umrs/ and installs the umrs package.
#
# Usage:
#   ./setup_client.sh

set -euo pipefail

# shellcheck source=./cluster-context.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster-context.sh"

SCRIPT_DIR="$UMRS_CLUSTER_DIR"
CLIENT_DIR="${SCRIPT_DIR}/client"
CONFIG_DIR="${HOME}/.config/umrs"
PROFILE_NAME="$UMRS_CLIENT_PROFILE"

echo "=== UMRS Client Setup ==="

# Create config directories
mkdir -p "${CONFIG_DIR}/users" "${CONFIG_DIR}/connections"

# Copy config files
cp "${CLIENT_DIR}/conf_umrs.json" "${CONFIG_DIR}/conf_umrs.json"
# NOTE: Do NOT copy the admin s3cfg template to the client.  It contains
# placeholder values (SET_S3_ROOT_USER) and the auto-discovery logic in
# the client would pick it up and override the per-user s3cfg that the
# server provisioning creates (e.g. ~/.s3cfg_user1).  The user's s3cfg is
# written during the first `umrs reload` and does not need seeding.
cp "${CLIENT_DIR}/user1.json"     "${CONFIG_DIR}/users/user1.json"
cp "${CLIENT_DIR}/connections/${PROFILE_NAME}.json" "${CONFIG_DIR}/connections/${PROFILE_NAME}.json"
printf '%s\n' "${PROFILE_NAME}" > "${CONFIG_DIR}/current_connection_profile"

echo "Configs installed to ${CONFIG_DIR}/"
echo "  conf_umrs.json"
echo "  users/user1.json"
echo "  connections/${PROFILE_NAME}.json"
echo "  current_connection_profile (${PROFILE_NAME})"

# Install umrs package in editable mode
SRC_DIR="${SCRIPT_DIR}/../../src"
# umrs-protocol (this checkout's protocol/) and r4-contracts (the superproject
# sibling) are unpublished: install both from local paths first, so resolving
# umrs never fetches either name from a package index (dependency confusion).
PROTOCOL_DIR="${SCRIPT_DIR}/../../protocol"
R4_CONTRACTS_DIR="${R4_CONTRACTS_DIR:-${SCRIPT_DIR}/../../../r4-contracts}"
if [ -f "${SRC_DIR}/pyproject.toml" ]; then
    for local_pkg in "${PROTOCOL_DIR}" "${R4_CONTRACTS_DIR}"; do
        if [ ! -f "${local_pkg}/pyproject.toml" ]; then
            echo "ERROR: ${local_pkg}/pyproject.toml not found (set R4_CONTRACTS_DIR for r4-contracts)." >&2
            exit 1
        fi
    done
    echo ""
    echo "Installing umrs-protocol and r4-contracts from local paths ..."
    uv pip install -e "${PROTOCOL_DIR}" -e "${R4_CONTRACTS_DIR}"
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
echo "  1. Activate your Python environment if needed"
echo "  2. Inspect the selected connection:  umrs --output text profile show"
echo "  3. Start SSH tunnels:                ./ssh_tunnel.sh"
echo "  4. Verify connectivity:              umrs ping"
echo "  5. Run Demo 01:                      cd ../../demos/01_health_check && bash run_demo.sh"
