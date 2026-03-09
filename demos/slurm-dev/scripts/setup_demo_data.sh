#!/bin/bash
# =============================================================================
# Setup Demo Data: Upload test data to UMRS via client_umrs.py
# =============================================================================
# Run inside umrs-client container:
#   podman exec umrs-client bash /opt/scripts/setup_demo_data.sh
#
# Requires UMRS_SERVER_HOST env var to point to the UMRS server (e.g. ai-login).
# =============================================================================

set -e

S3CFG="/root/.s3cfg"
TEXT_DATA_DIR="/opt/test_data/text"

TEXT_BUCKET="rivault-demo-bucket"

# Path to client script (mounted from repo)
CLIENT="/opt/umrs/client_umrs.py"
CONF="/root/conf_umrs.json"
USER_CONF="/root/.config/umrs/users/user_rivault.json"


echo "============================================"
echo "  UMRS Demo Data Setup"
echo "============================================"

# --- Wait for UMRS server to be reachable via the client control channel ---
echo ""
echo "Waiting for UMRS server..."
for i in $(seq 1 40); do
    if python3 "$CLIENT" -c "$USER_CONF" ping  >/dev/null 2>&1; then
        echo "  UMRS server is reachable."
        break
    fi
    if [ "$i" -eq 40 ]; then
        echo "  ERROR: UMRS server not reachable after 40 attempts."
        exit 1
    fi
    sleep 2
done

# --- Upload text files using client_umrs upload_dir (sends files through data queue) ---
echo ""
echo "Uploading text files via client_umrs to bucket: ${TEXT_BUCKET}"
if [ -d "$TEXT_DATA_DIR" ]; then
    python3 "$CLIENT" -c "$USER_CONF" upload_dir "$TEXT_DATA_DIR" "$TEXT_BUCKET" rivault_user || {
        echo "  WARNING: client_umrs upload_dir for text failed."
    }
else
    echo "  WARNING: ${TEXT_DATA_DIR} not found, skipping text upload."
fi

# --- Verify ---
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"

echo ""
echo "Fetching status summary..."
python3 "$CLIENT" -c "$USER_CONF" --output text status all || echo "  (status retrieval failed)"

echo ""
echo "Demo data setup complete."
