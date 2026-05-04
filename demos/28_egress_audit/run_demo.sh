#!/usr/bin/env bash
# =============================================================================
# Demo 28: Egress Audit
# =============================================================================
# Shows: GET/POST /umrs/egress/audit
#
# Prerequisites:
#   - UMRS server reachable via HTTP (default: localhost:8001)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="egress" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-${UMRS_HTTP_URL:-http://localhost:8001}}"

banner "Demo 28: Egress Audit"
echo ""

log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: Query audit log
log "Step 1: Query egress audit log (GET /umrs/egress/audit)"
echo ""
echo "  curl -s '${UMRS_URL}/umrs/egress/audit?limit=10' | jq '.'"
echo ""
AUDIT=$(curl -sf "${UMRS_URL}/umrs/egress/audit?limit=10" 2>/dev/null || echo '{"items":[],"count":0}')
echo "$AUDIT" | python3 -m json.tool 2>/dev/null || echo "$AUDIT"
echo ""

# Step 2: Audit log format
log "Step 2: Audit log entry format"
echo ""
echo '  {
    "request_id": "...",
    "user_id": "alice",
    "project_id": "research-lab",
    "tool_name": "fetch_api",
    "destination_url": "https://api.example.com/data",
    "destination_domain": "api.example.com",
    "method": "GET",
    "action": "allowed"
  }'
echo ""

# Step 3: Log an egress event
log "Step 3: Log an egress event (POST /umrs/egress/audit)"
echo ""
echo "  Request body:"
echo '  {
    "user_id": "demo-user",
    "project_id": "demo-project",
    "destination_url": "https://api.example.com/data",
    "destination_domain": "api.example.com",
    "method": "GET",
    "action": "allowed"
  }'
echo ""

# Step 4: Filtering audit logs
log "Step 4: Filtering audit log entries"
echo ""
echo "  # By action"
echo "  GET /umrs/egress/audit?action=denied"
echo "  GET /umrs/egress/audit?action=allowed"
echo ""
echo "  # By user or project"
echo "  GET /umrs/egress/audit?user_id=alice"
echo "  GET /umrs/egress/audit?project_id=research-lab"
echo ""
echo "  # By time range"
echo "  GET /umrs/egress/audit?since=2026-04-29T00:00:00Z"
echo ""
echo "  # With limit"
echo "  GET /umrs/egress/audit?limit=50"
echo ""

# Step 5: r4 CLI equivalent
log "Step 5: r4 CLI equivalent"
echo ""
echo "  r4 ask agent audit               # Recent events"
echo "  r4 ask agent audit --last 24h     # Last 24 hours"
echo "  r4 ask agent audit --action denied"
echo "  r4 ask agent audit --user alice"
echo "  r4 ask agent audit --project research-lab --json"
echo ""

banner "Demo 28 Complete"
