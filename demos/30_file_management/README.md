# Demo 30: File Management API

## Purpose

Demonstrates the UMRS file management REST API — upload, bulk upload,
download (with presigned URLs), and deletion of files stored in MinIO/S3.

## API Endpoints Covered

| Method | Path                          | Description                 |
|--------|-------------------------------|-----------------------------|
| POST   | `/umrs/files/upload`          | Upload single file          |
| POST   | `/umrs/files/bulk-upload`     | Upload multiple files       |
| GET    | `/umrs/files/{file_key}`      | Download / get presigned URL|
| DELETE | `/umrs/files/{file_key}`      | Delete a file               |

## Prerequisites

- UMRS HTTP sidecar running (`UMRS_HTTP_URL` or default `http://localhost:8001`)
- `curl` and `jq` installed
- A valid project ID (default: `default-project`)
- S3/MinIO backend configured and accessible

## Step-by-Step Commands

### Step 1: Upload a single file

```bash
UMRS_URL="${UMRS_HTTP_URL:-http://localhost:8001}"

# Create a test file
echo "Hello, UMRS file management!" > /tmp/demo30-test.txt

# Upload via Gateway-mediated upload endpoint
UPLOAD_RESPONSE=$(curl -sf -X POST "${UMRS_URL}/umrs/files/upload" \
  -H "X-Project-ID: default-project" \
  -F "file=@/tmp/demo30-test.txt" \
  -F "purpose=demo30")

echo "${UPLOAD_RESPONSE}" | jq .
FILE_KEY=$(echo "${UPLOAD_RESPONSE}" | jq -r '.key // empty')
echo "[demo30] Uploaded file key: ${FILE_KEY}"
```

### Step 2: Get file metadata

```bash
curl -sf "${UMRS_URL}/umrs/files/${FILE_KEY}/metadata" \
  -H "X-Project-ID: default-project" | jq .
```

**Expected output:**
```json
{
  "key": "demo30/abc123_demo30-test.txt",
  "bucket": "umrs-files",
  "size": 35,
  "content_type": "text/plain",
  "last_modified": "2026-07-02T12:00:00Z",
  "etag": "\"abc123\""
}
```

### Step 3: Generate a presigned download URL

```bash
curl -sf -X POST "${UMRS_URL}/umrs/files/presign" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d "{\n  \"operation\": \"download\",\n  \"key\": \"${FILE_KEY}\",\n  \"ttl_sec\": 3600\n}" | jq .
```

### Step 4: Generate a presigned upload URL

```bash
curl -sf -X POST "${UMRS_URL}/umrs/files/presign" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "operation": "upload",
    "key": "demo30/presigned-upload.txt",
    "ttl_sec": 3600,
    "content_type": "text/plain"
  }' | jq .
```

### Step 5: Check if files exist

```bash
curl -sf -X POST "${UMRS_URL}/umrs/files/exists" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d "{\"keys\": [\"${FILE_KEY}\"]}" | jq .
```

### Step 6: Delete the file

```bash
curl -sf -X DELETE "${UMRS_URL}/umrs/files/${FILE_KEY}" \
  -H "X-Project-ID: default-project"
echo "[demo30] Deleted file: ${FILE_KEY}"
```

## Notes

- All file operations are namespaced by `project_id`.
- Presigned URLs have a configurable TTL (default: 3600 seconds).
- File keys are validated to prevent path traversal attacks.
- Bulk upload is supported via the `/umrs/files/bulk-upload` endpoint.

## Cleanup

```bash
rm -f /tmp/demo30-test.txt
echo "[demo30] File management demo complete."
```
