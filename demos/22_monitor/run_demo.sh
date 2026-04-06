#!/usr/bin/env bash
# =============================================================================
# Demo 22: Jobs Monitor (TUI Dashboard)
# =============================================================================
# Demonstrates the live-refreshing terminal dashboard:
#   1. Submit a couple of jobs (so there's something to see)
#   2. Launch the monitor in --once mode (single snapshot)
#   3. Show how to launch the interactive live dashboard
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556)
#   - UMRS server running
#   - umrs-client installed, `rich` library installed
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="monitor" source "${SCRIPT_DIR}/../lib/common.sh"

banner "Demo 22: Jobs Monitor (TUI Dashboard)"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

python3 -c "import rich" 2>/dev/null || die "'rich' library not installed. Run: pip install 'rich>=13.0'"
ok "rich library available"

# ---------- Step 2: Seed some jobs ----------
log ""
log "Step 2: Submitting sample jobs so the dashboard has data..."

$CLIENT job submit index_build \
    --spec '{"database": "demo_db", "table": "embeddings"}' \
    --project demo >/dev/null 2>&1 && ok "Submitted index_build job" || warn "Submit failed (non-fatal)"

$CLIENT job submit cleanup \
    --spec '{"target": "stale_objects"}' \
    --project demo >/dev/null 2>&1 && ok "Submitted cleanup job" || warn "Submit failed (non-fatal)"

# ---------- Step 3: One-shot dashboard ----------
log ""
log "Step 3: Printing dashboard snapshot (--once mode)..."
echo ""

$CLIENT monitor --once
echo ""
ok "One-shot dashboard rendered"

# ---------- Step 4: Filtered view ----------
log ""
log "Step 4: Filtered view — only queued jobs..."
echo ""

$CLIENT monitor --once --status queued
echo ""
ok "Filtered dashboard rendered"

# ---------- Step 5: Interactive usage guide ----------
log ""
log "Step 5: Interactive dashboard usage"
echo ""
echo "  To launch the live-refreshing dashboard, run:"
echo ""
echo "    umrs-client monitor"
echo ""
echo "  Options:"
echo "    --interval 5           Refresh every 5 seconds (default: 3)"
echo "    --status running       Filter by job status"
echo "    --project myproject    Filter by project"
echo "    --once                 Print once and exit (no live loop)"
echo ""
echo "  Keyboard shortcuts (live mode):"
echo "    q       Quit"
echo "    r       Force refresh"
echo "    j / ↓   Move selection down"
echo "    k / ↑   Move selection up"
echo "    Enter   Toggle job detail view (shows attempts)"
echo "    f       Cycle status filter"
echo "    /       Toggle compact / wide table"
echo ""

banner "Demo 22 complete"
