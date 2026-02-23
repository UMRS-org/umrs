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

echo "============================================"
echo "  UMRS Demo Data Setup"
echo "============================================"

# --- Wait for MinIO to be reachable ---
echo ""
echo "Waiting for MinIO..."
for i in $(seq 1 30); do
    if s3cmd -c "$S3CFG" ls s3:// >/dev/null 2>&1; then
        echo "  MinIO is reachable."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: MinIO not reachable after 30 attempts."
        exit 1
    fi
    sleep 2
done

# --- Create text bucket ---
echo ""
echo "Creating bucket: s3://${TEXT_BUCKET}"
s3cmd -c "$S3CFG" mb "s3://${TEXT_BUCKET}" 2>/dev/null || echo "  (bucket may already exist)"

# --- Upload text files ---
if [ -d "$TEXT_DATA_DIR" ]; then
    echo "Uploading text files from ${TEXT_DATA_DIR}..."
    for f in "${TEXT_DATA_DIR}"/*.txt; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        echo "  -> ${fname}"
        s3cmd -c "$S3CFG" put "$f" "s3://${TEXT_BUCKET}/${fname}"
    done
else
    echo "  WARNING: ${TEXT_DATA_DIR} not found, skipping text upload."
fi

# --- Create image bucket ---
echo ""
echo "Creating bucket: s3://${IMAGE_BUCKET}"
s3cmd -c "$S3CFG" mb "s3://${IMAGE_BUCKET}" 2>/dev/null || echo "  (bucket may already exist)"

# --- Upload image files ---
if [ -d "$IMAGE_DATA_DIR" ]; then
    echo "Uploading image files from ${IMAGE_DATA_DIR}..."
    for f in "${IMAGE_DATA_DIR}"/*.jpg "${IMAGE_DATA_DIR}"/*.jpeg "${IMAGE_DATA_DIR}"/*.png; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        echo "  -> ${fname}"
        s3cmd -c "$S3CFG" put "$f" "s3://${IMAGE_BUCKET}/${fname}"
    done
else
    echo "  WARNING: ${IMAGE_DATA_DIR} not found, skipping image upload."
fi

# --- Verify ---
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"

echo ""
echo "Text bucket contents:"
s3cmd -c "$S3CFG" ls "s3://${TEXT_BUCKET}/" || echo "  (empty or error)"

echo ""
echo "Image bucket contents:"
s3cmd -c "$S3CFG" ls "s3://${IMAGE_BUCKET}/" || echo "  (empty or error)"

echo ""
echo "Demo data setup complete."
