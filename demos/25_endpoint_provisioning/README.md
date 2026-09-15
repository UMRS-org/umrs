# Demo 25: Endpoint Provisioning API

## Purpose

Demonstrates the UMRS endpoint provisioning REST API — managing
Mini-RiVault endpoint lifecycle including warm pool reuse,
multi-model co-location, LoRA adapter support, and the project-tier
MCP/RAG/egress mount contract.

## Prerequisites

- UMRS server running (HTTP API on port 8001)
- At least one model registered or pre-staged

## API Endpoints Covered

| Method | Path                       | Description                |
|--------|----------------------------|----------------------------|
| POST   | `/umrs/endpoints/provision`| Provision a new endpoint   |
| POST   | `/umrs/endpoints/up`       | Alias for provision        |
| GET    | `/umrs/endpoints`          | List active endpoints      |
| DELETE | `/umrs/endpoints/{id}`     | Stop an endpoint           |

## Mount Contract Covered

The demo creates sample mount assets and shows a provisioning payload with:

- `mounts.mcp_servers[]` targeting `/mcp-servers/demo-tools`
- `mounts.project_data` targeting `/project`
- `mounts.rag_index` targeting `/rag-index`
- `egress_domains` for the EgressGuard allowlist

Set `DEMO25_PROVISION=1` to submit the request. The default is read-only so
the full demo suite does not allocate a new endpoint unless explicitly asked.

## Prerequisites

- UMRS HTTP sidecar running (`UMRS_HTTP_URL` or default `http://localhost:8001`)
- At least one model registered via `umrs-client model register`
- `curl` and `jq` installed
- A valid project ID (default: `default-project`)

## Step-by-Step Commands

### Step 1: Check endpoint health

```bash
UMRS_URL="${UMRS_HTTP_URL:-http://localhost:8001}"
curl -sf "${UMRS_URL}/umrs/endpoints/health" | jq .
```

**Expected output:**
```json
{
  "status": "healthy",
  "active_endpoints": 0
}
```

### Step 2: List current endpoints

```bash
curl -sf "${UMRS_URL}/umrs/endpoints" \
  -H "X-Project-ID: default-project" | jq .
```

**Expected output:**
```json
{
  "endpoints": [],
  "total": 0
}
```

### Step 3: View endpoint statistics

```bash
curl -sf "${UMRS_URL}/umrs/endpoints/stats" \
  -H "X-Project-ID: default-project" | jq .
```

### Step 4: Provision a new endpoint (read-only by default)

```bash
# Set DEMO25_PROVISION=1 to actually create an endpoint
if [[ "${DEMO25_PROVISION:-0}" = "1" ]]; then
  curl -sf -X POST "${UMRS_URL}/umrs/endpoints/provision" \
    -H "Content-Type: application/json" \
    -H "X-Project-ID: default-project" \
    -d '{
      "model": "rivault-gpt-oss-static",
      "tier": "user",
      "ttl_seconds": 3600,
      "mounts": {
        "mcp_servers": [],
        "project_data": null,
        "rag_index": null
      },
      "egress_domains": []
    }' | jq .
else
  echo "[demo25] Read-only mode. Set DEMO25_PROVISION=1 to provision an endpoint."
fi
```

### Step 5: Query active endpoints

```bash
curl -sf "${UMRS_URL}/umrs/endpoints/active" \
  -H "X-Project-ID: default-project" | jq .
```

### Step 6: Get endpoint details (if any exist)

```bash
ENDPOINT_ID=$(curl -sf "${UMRS_URL}/umrs/endpoints" \
  -H "X-Project-ID: default-project" | jq -r '.endpoints[0].job_id // empty')

if [[ -n "${ENDPOINT_ID}" ]]; then
  curl -sf "${UMRS_URL}/umrs/endpoints/${ENDPOINT_ID}" \
    -H "X-Project-ID: default-project" | jq .

  # View metrics for this endpoint
  curl -sf "${UMRS_URL}/umrs/endpoints/${ENDPOINT_ID}/metrics" \
    -H "X-Project-ID: default-project" | jq .
else
  echo "[demo25] No active endpoints to inspect."
fi
```

## Cleanup

```bash
# Delete all endpoints for this project
ENDPOINTS=$(curl -sf "${UMRS_URL}/umrs/endpoints" \
  -H "X-Project-ID: default-project" | jq -r '.endpoints[].job_id')

for eid in ${ENDPOINTS}; do
  curl -sf -X DELETE "${UMRS_URL}/umrs/endpoints/${eid}" \
    -H "X-Project-ID: default-project"
  echo "[demo25] Deleted endpoint: ${eid}"
done
```
