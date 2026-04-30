#!/usr/bin/env bash
# =============================================================================
# Demo 30: File Management API
# =============================================================================
# Shows: POST /umrs/files/upload, POST /umrs/files/bulk-upload,
#        GET /umrs/files/{key}, DELETE /umrs/files/{key}
#
# Prerequisites:
#   - UMRS server with MinIO/S3 configured
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="file_mgmt" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-http://localhost:8001}"

banner "Demo 30: File Management API"
echo ""

log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: Upload a file
log "Step 1: Upload a file (POST /umrs/files/upload)"
echo ""
echo "  curl -X POST ${UMRS_URL}/umrs/files/upload \\"
echo "      -F 'file=@document.pdf' \\"
echo "      -F 'bucket=my-project'"
echo ""
echo "  Response:"
echo '  {"file_key": "my-project/document.pdf", "size_bytes": 1024, "status": "uploaded"}'
echo ""

# Create a temp file for actual upload test
TMPFILE=$(mktemp)
echo "Demo 30 test content" > "$TMPFILE"
trap "rm -f $TMPFILE" EXIT

UPLOAD_RESULT=$(curl -sf -X POST "${UMRS_URL}/umrs/files/upload" \
    -F "file=@${TMPFILE};filename=demo30_test.txt" \
    -F "bucket=demo30" 2>/dev/null || echo '{"status": "upload API not available in this deployment"}')
echo "  Result: $UPLOAD_RESULT"
echo ""

# Step 2: Bulk upload
log "Step 2: Bulk upload (POST /umrs/files/bulk-upload)"
echo ""
echo "  curl -X POST ${UMRS_URL}/umrs/files/bulk-upload \\"
echo "      -F 'files=@file1.txt' \\"
echo "      -F 'files=@file2.txt' \\"
echo "      -F 'files=@file3.txt' \\"
echo "      -F 'bucket=my-project'"
echo ""
echo "  Response:"
echo '  {"uploaded": 3, "results": [...]}'
echo ""

# Step 3: Download (presigned URL)
log "Step 3: Download via presigned URL (GET /umrs/files/{key})"
echo ""
echo "  curl -s ${UMRS_URL}/umrs/files/my-project/document.pdf -o output.pdf"
echo ""
echo "  Or get a presigned URL:"
echo "  curl -s ${UMRS_URL}/umrs/files/my-project/document.pdf?presign=true"
echo ""
echo "  Presigned URL properties:"
echo "    • Time-limited (default: 3600s)"
echo "    • No MinIO credentials needed"
echo "    • Audited access"
echo ""

# Step 4: Delete
log "Step 4: Delete a file (DELETE /umrs/files/{key})"
echo ""
echo "  curl -X DELETE ${UMRS_URL}/umrs/files/my-project/document.pdf"
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
