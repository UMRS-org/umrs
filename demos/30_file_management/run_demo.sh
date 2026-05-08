#!/usr/bin/env bash
# =============================================================================
# Demo 30: File Management API
# =============================================================================
# Shows: POST /umrs/files/presign, POST /umrs/files/presign/batch,
#        GET /umrs/files/{key}/metadata, DELETE /umrs/files/{key}
#
# Prerequisites:
#   - UMRS server with MinIO/S3 configured
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="file_mgmt" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-${UMRS_HTTP_URL:-http://localhost:8001}}"
PROJECT_ID="${UMRS_PROJECT_ID:-demo30}"

banner "Demo 30: File Management API"
echo ""

log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: Generate upload URL
log "Step 1: Generate an upload URL (POST /umrs/files/presign)"
echo ""
echo "  curl -X POST ${UMRS_URL}/umrs/files/presign \\"
echo "      -H 'X-Project-ID: ${PROJECT_ID}' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"operation\":\"upload\",\"key\":\"document.pdf\",\"content_type\":\"application/pdf\"}'"
echo ""
echo "  Response:"
echo '  {"url": "http://...", "method": "PUT", "key": "demo30/document.pdf", "ttl_sec": 3600}'
echo ""

UPLOAD_RESULT=$(curl -sf -X POST "${UMRS_URL}/umrs/files/presign" \
    -H "X-Project-ID: ${PROJECT_ID}" \
    -H "Content-Type: application/json" \
    -d '{"operation":"upload","key":"demo30_test.txt","ttl_sec":3600,"content_type":"text/plain"}' \
    2>/dev/null || echo '{"status": "presign API not available in this deployment"}')
echo "  Result: $UPLOAD_RESULT"
echo ""

# Step 2: Batch presign
log "Step 2: Batch presign (POST /umrs/files/presign/batch)"
echo ""
echo "  curl -X POST ${UMRS_URL}/umrs/files/presign/batch \\"
echo "      -H 'X-Project-ID: ${PROJECT_ID}' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"requests\":[{\"operation\":\"upload\",\"key\":\"file1.txt\"}]}'"
echo ""
echo "  Response:"
echo '  {"results": [{"url": "http://...", "method": "PUT"}], "errors": []}'
echo ""

# Step 3: Download URL and metadata
log "Step 3: Download URL and metadata"
echo ""
echo "  curl -X POST ${UMRS_URL}/umrs/files/presign \\"
echo "      -H 'X-Project-ID: ${PROJECT_ID}' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"operation\":\"download\",\"key\":\"document.pdf\"}'"
echo ""
echo "  curl -s -H 'X-Project-ID: ${PROJECT_ID}' ${UMRS_URL}/umrs/files/document.pdf/metadata"
echo ""
echo "  Presigned URL properties:"
echo "    • Time-limited (default: 3600s)"
echo "    • No MinIO credentials needed"
echo "    • Namespaced by project ID"
echo ""

# Step 4: Delete
log "Step 4: Delete a file (DELETE /umrs/files/{key})"
echo ""
echo "  curl -X DELETE -H 'X-Project-ID: ${PROJECT_ID}' ${UMRS_URL}/umrs/files/document.pdf"
echo ""

# Step 5: Comparison with CLI/SDK
log "Step 5: CLI and SDK equivalents"
echo ""
echo "  # umrs-client (ZMQ transport)"
echo "  umrs-client upload_file document.pdf my-project alice"
echo "  umrs-client upload_dir ./data/ my-project alice"
echo "  umrs-client download_results my-project ./output/ alice"
echo ""
echo "  # r4 SDK (auto-selects ZMQ or HTTP)"
echo "  client.data.upload('document.pdf', project_id='my-project')"
echo "  client.data.upload_directory('./data/', project_id='my-project')"
echo "  client.data.download('file-id', output_path='./output/')"
echo "  client.data.list_files(project_id='my-project')"
echo "  client.data.get_presigned_url('file-id', ttl_seconds=3600)"
echo ""

banner "Demo 30 Complete"
