# Demo 26: MCP Server Management

## Purpose

Demonstrates UMRS MCP (Model Context Protocol) server management and the
preferred mounted MCP path for mini-RiVault endpoints: user/project MCP code
is mounted under `/mcp-servers`, validated by `r4_mcp.runners.custom_mcp`, and
run by the `custom-mcp` profile inside the endpoint boundary.

## API Endpoints Covered

| Method | Path                          | Description                |
|--------|-------------------------------|----------------------------|
| GET    | `/umrs/mcp/servers`           | List registered servers    |
| POST   | `/umrs/mcp/servers`           | Register a server          |
| GET    | `/umrs/mcp/servers/{id}`      | Server details             |
| DELETE | `/umrs/mcp/servers/{id}`      | Unregister a server        |
| GET    | `/umrs/mcp/tools`             | List all tools             |
| POST   | `/umrs/mcp/invoke`            | Invoke a tool              |
| GET    | `/umrs/mcp/health`            | Health check               |

External MCP registration remains supported for login-node or existing service
integrations. GPU-adjacent compute-node MCP work should use endpoint mounts,
not a separate compute-node MCP launcher.

## Prerequisites

- UMRS HTTP sidecar running (`UMRS_HTTP_URL` or default `http://localhost:8001`)
- `curl` and `jq` installed
- A valid project ID (default: `default-project`)

## Step-by-Step Commands

### Step 1: Check MCP health

```bash
UMRS_URL="${UMRS_HTTP_URL:-http://localhost:8001}"
curl -sf "${UMRS_URL}/umrs/mcp/health" | jq .
```

**Expected output:**
```json
{
  "status": "healthy",
  "registered_servers": 0
}
```

### Step 2: List registered MCP servers

```bash
curl -sf "${UMRS_URL}/umrs/mcp/servers" \
  -H "X-Project-ID: default-project" | jq .
```

**Expected output:**
```json
[]
```

### Step 3: Register a new MCP server

```bash
SERVER_ID=$(curl -sf -X POST "${UMRS_URL}/umrs/mcp/servers" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "name": "demo-tools",
    "version": "0.1.0",
    "description": "Demo MCP server for testing",
    "tools": [
      {
        "name": "summarize_mount",
        "description": "Summarize mounted project data",
        "input_schema": {
          "type": "object",
          "properties": {
            "topic": {"type": "string"}
          }
        }
      }
    ]
  }' | jq -r '.server_id // .id // empty')

echo "[demo26] Registered server ID: ${SERVER_ID}"
```

### Step 4: Get server details

```bash
if [[ -n "${SERVER_ID}" ]]; then
  curl -sf "${UMRS_URL}/umrs/mcp/servers/${SERVER_ID}" \
    -H "X-Project-ID: default-project" | jq .
else
  echo "[demo26] No server ID returned. Check the registration response."
fi
```

### Step 5: List all available MCP tools

```bash
curl -sf "${UMRS_URL}/umrs/mcp/tools" \
  -H "X-Project-ID: default-project" | jq .
```

### Step 6: Invoke a tool (if server is running)

```bash
curl -sf -X POST "${UMRS_URL}/umrs/mcp/invoke" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: default-project" \
  -d '{
    "server_name": "demo-tools",
    "tool_name": "summarize_mount",
    "arguments": {"topic": "test"}
  }' | jq .
```

## Cleanup

```bash
if [[ -n "${SERVER_ID}" ]]; then
  curl -sf -X DELETE "${UMRS_URL}/umrs/mcp/servers/${SERVER_ID}" \
    -H "X-Project-ID: default-project"
  echo "[demo26] Unregistered server: ${SERVER_ID}"
fi
```
