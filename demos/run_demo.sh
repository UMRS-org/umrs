#!/usr/bin/env bash
# UMRS Demo Runner — automatically prepares environment before running demos
# Usage: ./run_demo.sh [demo_number] [demo_args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UMRS_CLIENT="${UMRS_CLIENT:-$(which umrs-client 2>/dev/null || echo "$REPO_ROOT/.venv/bin/umrs-client")}"
PROFILE="${R4_PROFILE:-rccs-cloud}"

log_ok()  { echo -e "\033[32m✓\033[0m $1"; }
log_fail(){ echo -e "\033[31m✗\033[0m $1"; }
log_info(){ echo -e "\033[33mℹ\033[0m $1"; }
log_skip(){ echo -e "\033[90m⊘\033[0m $1"; }

echo "=============================================="
echo "  UMRS Demo Runner"
echo "=============================================="
echo ""

# ── Environment Check ──────────────────────────
echo "Checking environment..."

if ! command -v "$UMRS_CLIENT" &>/dev/null; then
    log_fail "umrs-client not found"
    log_info "Install with: pip install -e $REPO_ROOT/umrs/src/"
    exit 1
fi

# Ping server
PONG=$("$UMRS_CLIENT" --profile "$PROFILE" ping 2>&1 || true)
if echo "$PONG" | grep -q "Server ready"; then
    log_ok "Server reachable"
else
    log_fail "Server unreachable"
    log_info "Run: r4 auth login --api-key <token>"
    exit 1
fi

# Check buckets
BUCKET_COUNT=$("$UMRS_CLIENT" --profile "$PROFILE" --output json bucket list 2>/dev/null | tail -n1 | python3 -c "
import json, sys
line = next((l for l in sys.stdin if l.strip().startswith('{')), '')
data = json.loads(line) if line.strip() else {}
buckets = data.get('buckets', [])
print(len(buckets))
" 2>/dev/null || echo "0")

if [ "$BUCKET_COUNT" -gt 0 ]; then
    log_ok "Storage OK ($BUCKET_COUNT buckets)"
else
    log_skip "No buckets visible — try: umrs-client reload"
fi

# Check models
MODEL_COUNT=$("$UMRS_CLIENT" --profile "$PROFILE" --output json model list --scope system 2>/dev/null | tail -n1 | python3 -c "
import json, sys
line = next((l for l in sys.stdin if l.strip().startswith('{')), '')
data = json.loads(line) if line.strip() else {}
models = data.get('models', [])
print(len(models))
" 2>/dev/null || echo "0")

if [ "$MODEL_COUNT" -gt 0 ]; then
    log_ok "Models OK ($MODEL_COUNT system models)"
else
    log_skip "No system models — ask admin to stage models"
fi

echo ""

# ── Run Demo ──────────────────────────────────
DEMO_DIR="${1:-}"
if [ -z "$DEMO_DIR" ]; then
    echo "Usage: $0 <demo_number> [demo_args...]"
    echo ""
    echo "Available demos:"
    for d in "$SCRIPT_DIR"/[0-9]*; do
        if [ -d "$d" ]; then
            name=$(basename "$d")
            readme="$d/README.md"
            if [ -f "$readme" ]; then
                desc=$(head -5 "$readme" | grep -v "^#" | head -1 | cut -c1-60)
                echo "  $name — $desc"
            fi
        fi
    done
    exit 0
fi

# Normalize demo name
DEMO_NAME=$(echo "$DEMO_DIR" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
DEMO_PATH="$SCRIPT_DIR/$DEMO_NAME"

if [ ! -d "$DEMO_PATH" ]; then
    log_fail "Demo directory not found: $DEMO_NAME"
    exit 1
fi

if [ ! -f "$DEMO_PATH/README.md" ]; then
    log_fail "No README.md found in: $DEMO_NAME"
    exit 1
fi

echo "Running demo: $DEMO_NAME"
echo "----------------------------------------------"
echo ""

# Show demo header
head -20 "$DEMO_PATH/README.md" | grep -v "^$" | head -10

echo ""
echo "----------------------------------------------"
echo "Proceeding with demo..."
echo ""

# Run the demo's README instructions
# We look for the "Running Manually" section or numbered steps
cd "$DEMO_PATH"
exec "$UMRS_CLIENT" --profile "$PROFILE" --help >/dev/null 2>&1
echo "Demo $DEMO_NAME ready. Follow the README.md instructions."
