# Demo 28: Egress Audit

## Purpose

Demonstrates the UMRS egress audit API — querying and logging
outbound network events from Mini-RiVault instances. All egress
from MCP servers and tools is recorded with full request/response
metadata for compliance and security review.

The demo also ties audit events back to endpoint provisioning: `egress_domains`
become `EGRESS_DOMAIN_ALLOWLIST` for the EgressGuard profile, and mounted MCP
tools inherit proxy settings so outbound traffic remains auditable.

## API Endpoints Covered

| Method | Path                          | Description                |
|--------|-------------------------------|----------------------------|
| GET    | `/umrs/egress/audit`          | Query audit log entries    |
| POST   | `/umrs/egress/audit`          | Log an egress event        |

## Prerequisites

- UMRS HTTP sidecar running (`UMRS_HTTP_URL` or default `http://localhost:8001`)
- `curl` and `jq` installed
- A valid project ID (default: `default-project`)
- Egress audit logging enabled on the server

## Step-by-Step Commands

### Step 1: Query egress audit log

```bash
UMRS_URL="${UMRS_HTTP_URL:-http://localhost:8001}"

curl -sf "${UMRS_URL}/umrs/egress/audit" \
  -H "X-Project-ID: default-project" | jq .
```

**Expected output:**
```json
{
  "entries": [],
  "total": 0
}
```

### Step 2: Log a test egress event

```bash
curl -sf -X POST "${UMRS_URL}/umrs/egress" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "method": "GET",
    "url": "https://api.example.com/test",
    "status_code": 200,
    "duration_ms": 150,
    "metadata": {"demo": "demo28"}
  }' | jq .
```

### Step 3: Query audit log after event

```bash
curl -sf "${UMRS_URL}/umrs/egress/audit" \
  -H "X-Project-ID: default-project" | jq '.entries[-1]'
```

### Step 4: Filter audit log by method

```bash
curl -sf "${UMRS_URL}/umrs/egress/audit?method=GET" \
  -H "X-Project-ID: default-project" | jq .
```

## Notes

- Egress audit entries are stored in the `egress_audit` table.
- All outbound network traffic from MCP servers and tools is logged.
- Audit log retention is configurable via `EGRESS_AUDIT_RETENTION_DAYS`.
- This demo requires the egress audit table to exist (migration `0005_egress_audit.sql`).

## Cleanup

```bash
# No cleanup needed — audit entries are retained per policy.
echo "[demo28] Egress audit entries retained per retention policy."
```
