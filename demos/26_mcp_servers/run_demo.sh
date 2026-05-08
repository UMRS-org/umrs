#!/usr/bin/env bash
# =============================================================================
# Demo 26: MCP Server Management
# =============================================================================
# Shows: POST/GET/DELETE /umrs/mcp/servers, GET /umrs/mcp/tools,
#        POST /umrs/mcp/invoke, GET /umrs/mcp/health
#
# Prerequisites:
#   - UMRS server reachable via HTTP (default: localhost:8001)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="mcp_srv" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-${UMRS_HTTP_URL:-http://localhost:8001}}"

banner "Demo 26: MCP Server Management"
echo ""

# Step 0: Pre-flight
log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: List MCP servers
log "Step 1: List registered MCP servers (GET /umrs/mcp/servers)"
echo ""
echo "  curl -s ${UMRS_URL}/umrs/mcp/servers | jq '.'"
echo ""
SERVERS=$(curl -sf "${UMRS_URL}/umrs/mcp/servers" 2>/dev/null || echo '{"servers":[]}')
echo "$SERVERS" | python3 -m json.tool 2>/dev/null || echo "$SERVERS"
echo ""

# Step 2: Register an MCP server
log "Step 2: Register an MCP server (POST /umrs/mcp/servers)"
echo ""
print_mini_rivault_mount_summary
echo ""
echo "  Preferred endpoint path: mount MCP server code into mini-RiVault"
echo "  and enable the custom-mcp profile through endpoint provisioning."
echo "  The mounted directory must contain config.json and an entry point."
echo ""
if python3 -m r4_mcp.runners.custom_mcp --config-dir "${DEMO_MCP_SOURCE}" --validate-only >/dev/null 2>&1; then
  ok "Mounted MCP manifest validated"
else
  warn "r4_mcp validator unavailable in this client environment; skipping local manifest validation"
fi
echo ""
echo "  Request body:"
echo '  {
    "name": "demo-calculator",
    "url": "http://localhost:9090",
    "description": "A simple calculator MCP server",
    "auth_type": "none"
  }'
echo ""
echo "  Response:"
echo '  {"server_id": "mcp-abc123", "name": "demo-calculator", "status": "registered"}'
echo ""
echo "  Note: registration remains useful for external/login-node MCP servers."
echo "  GPU-adjacent compute-node MCP work should use the mounted custom-mcp"
echo "  profile shown above, not a second launcher path."
echo ""

# Step 3: List tools
log "Step 3: List all MCP tools (GET /umrs/mcp/tools)"
echo ""
echo "  curl -s ${UMRS_URL}/umrs/mcp/tools | jq '.'"
echo ""
TOOLS=$(curl -sf "${UMRS_URL}/umrs/mcp/tools" 2>/dev/null || echo '{"tools":[]}')
echo "$TOOLS" | python3 -m json.tool 2>/dev/null || echo "$TOOLS"
echo ""

# Step 4: Invoke a tool
log "Step 4: Invoke a tool (POST /umrs/mcp/invoke or /umrs/mcp/call)"
echo ""
echo "  Request body:"
echo '  {
    "tool": "calculator",
    "arguments": {"expression": "2 + 2"}
  }'
echo ""
echo "  Response:"
echo '  {"content": [{"type": "text", "text": "4"}], "isError": false, "server": "demo-calculator"}'
echo ""
echo "  Security: All tool outputs are scanned for injection attacks (T-214)."
echo ""

# Step 5: Health check
log "Step 5: MCP health check (GET /umrs/mcp/health)"
echo ""
echo "  curl -s ${UMRS_URL}/umrs/mcp/health | jq '.'"
echo ""

# Step 6: Unregister
log "Step 6: Unregister an MCP server (DELETE /umrs/mcp/servers/{id})"
echo ""
echo "  curl -s -X DELETE ${UMRS_URL}/umrs/mcp/servers/mcp-abc123"
echo ""

banner "Demo 26 Complete"
