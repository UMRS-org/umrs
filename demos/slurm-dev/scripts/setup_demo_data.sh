#!/bin/bash
# =============================================================================
# Setup Demo Data: Create MinIO buckets and upload test data
# =============================================================================
# Run inside ai-login container:
#   podman exec ai-login bash /opt/scripts/setup_demo_data.sh
# =============================================================================

set -e

S3CFG="/root/.s3cfg"
TEXT_DATA_DIR="/opt/test_data/text"
IMAGE_DATA_DIR="/opt/test_data/images"

TEXT_BUCKET="rivault-demo-bucket"
IMAGE_BUCKET="rivault-image-bucket"

# Path to client script (mounted from repo)
CLIENT="/opt/umrs/client_umrs.py"
CONF="/root/conf_umrs.json"

echo "============================================"
echo "  UMRS Demo Data Setup"
echo "============================================"

# --- Wait for UMRS server to be reachable via the client control channel ---
echo ""
echo "Waiting for UMRS server..."
for i in $(seq 1 40); do
    if python3 "$CLIENT" ping -c "$CONF" >/dev/null 2>&1; then
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
    python3 "$CLIENT" upload_dir "$TEXT_DATA_DIR" "$TEXT_BUCKET" rivault_user -c "$CONF" || {
        echo "  WARNING: client_umrs upload_dir for text failed."
    }
else
    echo "  WARNING: ${TEXT_DATA_DIR} not found, skipping text upload."
fi

# --- Upload image files using client_umrs upload_dir ---
echo ""
echo "Uploading image files via client_umrs to bucket: ${IMAGE_BUCKET}"
if [ -d "$IMAGE_DATA_DIR" ]; then
    python3 "$CLIENT" upload_dir "$IMAGE_DATA_DIR" "$IMAGE_BUCKET" rivault_user -c "$CONF" || {
        echo "  WARNING: client_umrs upload_dir for images failed."
    }
else
    echo "  WARNING: ${IMAGE_DATA_DIR} not found, skipping image upload."
fi

# --- Verify ---
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"

echo ""
echo ""
echo "Triggering embedding computation (server-side update)..."
python3 "$CLIENT" update -c "$CONF" || echo "  WARNING: embedding update command failed"

echo ""
echo "Fetching status summary..."
python3 "$CLIENT" status all -c "$CONF" --output text || echo "  (status retrieval failed)"

echo ""
echo "Demo data setup complete."
