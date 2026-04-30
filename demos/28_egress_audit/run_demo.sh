#!/usr/bin/env bash
# =============================================================================
# Demo 28: Egress Audit
# =============================================================================
# Shows: GET /umrs/egress_audit/audit, POST /umrs/egress_audit/log
#
# Prerequisites:
#   - UMRS server reachable via HTTP (default: localhost:8001)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="egress" source "${SCRIPT_DIR}/../lib/common.sh"

UMRS_URL="${UMRS_URL:-http://localhost:8001}"

banner "Demo 28: Egress Audit"
echo ""

log "Step 0: Server connectivity..."
preflight
echo ""

# Step 1: Query audit log
log "Step 1: Query egress audit log (GET /umrs/egress_audit/audit)"
echo ""
echo "  curl -s '${UMRS_URL}/umrs/egress_audit/audit?limit=10' | jq '.'"
echo ""
AUDIT=$(curl -sf "${UMRS_URL}/umrs/egress_audit/audit?limit=10" 2>/dev/null || echo '{"entries":[]}')
echo "$AUDIT" | python3 -m json.tool 2>/dev/null || echo "$AUDIT"
echo ""

# Step 2: Audit log format
log "Step 2: Audit log entry format"
echo ""
echo '  {
    "timestamp": "2026-04-30T10:30:00Z",
    "instance_id": "user-alice-abc123",
    "tier": "user",
    "owner_id": "alice",
    "request": {
      "method": "GET",
      "url": "https://api.example.com/data",
      "headers": {"User-Agent": "..."},
      "body_hash": "sha256:..."
    },
    "response": {
      "status": 200,
      "size_bytes": 1024,
      "body_hash": "sha256:..."
    },
    "policy_matched": "user-alice-default",
    "action": "allowed"
  }'
echo ""

# Step 3: Log an egress event
log "Step 3: Log an egress event (POST /umrs/egress_audit/log)"
echo ""
echo "  Request body:"
echo '  {
    "instance_id": "user-demo-xyz",
    "tier": "user",
    "owner_id": "demo-user",
    "request": {
      "method": "GET",
      "url": "https://api.example.com/data"
    },
    "response": {
      "status": 200,
      "size_bytes": 1024
    },
    "policy_matched": "user-demo-default",
    "action": "allowed"
  }'
echo ""

# Step 4: Filtering audit logs
log "Step 4: Filtering audit log entries"
echo ""
echo "  # By action"
echo "  GET /umrs/egress_audit/audit?action=denied"
echo "  GET /umrs/egress_audit/audit?action=allowed"
echo ""
echo "  # By owner"
echo "  GET /umrs/egress_audit/audit?owner_id=alice"
echo ""
echo "  # By time range"
echo "  GET /umrs/egress_audit/audit?since=2026-04-29T00:00:00Z"
echo ""
echo "  # With limit"
echo "  GET /umrs/egress_audit/audit?limit=50"
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
