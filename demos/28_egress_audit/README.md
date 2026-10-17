# Demo 28: Egress Audit

## Purpose

Demonstrates the UMRS egress audit API: recording and querying outbound
network decisions (allowed or denied) made on behalf of agents and tools.
Each row names the user, the project, the destination and the decision, for
compliance and security review.

The demo also ties audit events back to endpoint provisioning: `egress_domains`
become `EGRESS_DOMAIN_ALLOWLIST` for the EgressGuard profile, and mounted MCP
tools inherit proxy settings so outbound traffic remains auditable.

## API Endpoints Covered

| Method | Path                 | Who may call it (strict mode)                         |
|--------|----------------------|-------------------------------------------------------|
| POST   | `/umrs/egress/audit` | a gateway-signed **admin** only (the operator sink)   |
| GET    | `/umrs/egress/audit` | any signed caller: their own trail in their project; admins may filter freely |

Through the gateway the same routes are `/gw/egress/audit` (`r4 agent audit`
reads `GET /gw/egress/audit`).

## Identity (read this first)

> Updated 2026-10-18 (hardening plan H-1). Before that, both routes took no
> identity: anyone could forge audit rows, and a GET without filters returned
> every project's egress decisions.

UMRS runs in strict mode by default (`UMRS_AUTH_MODE=strict`). There, every
call must carry the gateway-signed identity tuple:

- `X-User-ID`, `X-Verified-Project-ID`, `X-Verified-Is-Admin` and
- `X-Verified-HMAC`, the gateway's HMAC over those three values
  (`UMRS_HANDSHAKE_HMAC_SECRET`).

The gateway adds these headers after it validates your token, so the normal
path is `/gw/egress/audit` with a bearer token. A direct call to the sidecar
with only `X-Project-ID`, or with a missing or wrong HMAC, answers **401**.

- **POST** needs the signed admin flag (`X-Verified-Is-Admin: true` inside
  the HMAC). A signed non-admin gets **403**.
- **GET** returns the caller's own trail in the caller's verified project.
  A non-admin's `user_id` filter naming someone else, or a `project_id`
  filter outside their verified project, gets **403**. Admins may filter by
  any user and project.

Dev mode (`UMRS_AUTH_MODE=mvp`) keeps the old open behaviour for callers that
send no identity headers; the commands below show both forms.

## Prerequisites

- UMRS HTTP sidecar running (`UMRS_HTTP_URL` or default `http://localhost:8001`)
- `curl` and `jq` installed
- Strict mode: a gateway URL and a bearer token (an admin's token for Step 2),
  or, for a local sidecar, dev mode (`UMRS_AUTH_MODE=mvp`)
- Postgres persistence is optional: with `UMRS_DB_URL` set (and `psycopg2`
  installed) rows go to the `egress_requests` table of migration
  `0005_egress_audit.sql`; otherwise they live in an in-memory ring buffer

## Step-by-Step Commands

### Step 1: Query your egress audit trail

Strict mode, through the gateway (any signed user):

```bash
GATEWAY_URL="${GATEWAY_URL:?set the gateway URL, e.g. https://gateway.example.org}"
TOKEN="${TOKEN:?set a bearer token}"

curl -sf "${GATEWAY_URL}/gw/egress/audit" \
  -H "Authorization: Bearer ${TOKEN}" | jq .
```

Dev mode, directly on the sidecar (no identity headers):

```bash
UMRS_URL="${UMRS_HTTP_URL:-http://localhost:8001}"

curl -sf "${UMRS_URL}/umrs/egress/audit" | jq .
```

**Expected output** (no rows yet):
```json
{
  "items": [],
  "count": 0
}
```

### Step 2: Record a test egress decision (admin only)

`user_id`, `project_id`, `destination_url`, `destination_domain` and `action`
(`allowed` or `denied`) are required. In strict mode `TOKEN` must belong to an
admin; a non-admin gets 403. In dev mode, drop the `Authorization` header and
use `${UMRS_URL}/umrs/egress/audit`.

```bash
curl -sf -X POST "${GATEWAY_URL}/gw/egress/audit" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user",
    "project_id": "default-project",
    "tool_name": "demo28",
    "destination_url": "https://api.example.com/test",
    "destination_domain": "api.example.com",
    "method": "GET",
    "action": "allowed",
    "response_status": 200
  }' | jq .
```

**Expected output** (`persisted` is `false` without a database):
```json
{
  "request_id": "…",
  "persisted": false
}
```

### Step 3: Query the trail after the event

The newest row comes first. A non-admin sees it only when they are
`demo-user` in `default-project`; an admin sees it with the filters below.

```bash
curl -sf "${GATEWAY_URL}/gw/egress/audit?user_id=demo-user&project_id=default-project" \
  -H "Authorization: Bearer ${TOKEN}" | jq '.items[0]'
```

### Step 4: Filter by decision and time

```bash
curl -sf "${GATEWAY_URL}/gw/egress/audit?action=allowed&since=2026-01-01T00:00:00Z&limit=20" \
  -H "Authorization: Bearer ${TOKEN}" | jq .
```

## Notes

- Filters: `user_id`, `project_id`, `action` (`allowed` | `denied`), `since`
  (ISO-8601) and `limit` (1-1000, default 100). There is no `method` filter.
- No in-repo service posts rows today: the Python gateway that did was
  removed (ADR-014), and EgressGuard writes its JSONL log locally. POST is
  the operator sink.
- Writes are best-effort: a database failure keeps the in-memory row and
  answers `persisted: false`.

## Cleanup

```bash
# No cleanup needed — audit entries are retained per policy.
echo "[demo28] Egress audit entries retained per retention policy."
```
