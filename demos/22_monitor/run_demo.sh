#!/usr/bin/env bash
# =============================================================================
# Demo 22: UMRS Monitor (Multi-Tab TUI Dashboard)
# =============================================================================
# Demonstrates the live-refreshing multi-tab terminal dashboard:
#   1. Submit a couple of jobs (so there's something to see)
#   2. Launch the monitor in --once mode (single snapshot, Jobs tab)
#   3. Show the Status tab view
#   4. Show the Models tab view
#   5. Show the Workflows tab view
#   6. Show the Connection tab view
#   7. Show how to launch the interactive live dashboard
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556)
#   - UMRS server running
#   - umrs-client installed, `rich` library installed
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="monitor" source "${SCRIPT_DIR}/../lib/common.sh"

banner "Demo 22: UMRS Monitor (Multi-Tab TUI Dashboard)"
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

# ---------- Step 3: One-shot Jobs tab ----------
log ""
log "Step 3: Jobs tab snapshot (--once --tab jobs)..."
echo ""

$CLIENT monitor --once --tab jobs
echo ""
ok "Jobs tab rendered"

# ---------- Step 4: Filtered Jobs view ----------
log ""
log "Step 4: Filtered Jobs view — only queued jobs..."
echo ""

$CLIENT monitor --once --tab jobs --status queued
echo ""
ok "Filtered Jobs tab rendered"

# ---------- Step 5: Status tab ----------
log ""
log "Step 5: Status tab snapshot..."
echo ""

$CLIENT monitor --once --tab status
echo ""
ok "Status tab rendered"

# ---------- Step 6: Models tab ----------
log ""
log "Step 6: Models tab snapshot..."
echo ""

$CLIENT monitor --once --tab models
echo ""
ok "Models tab rendered"

# ---------- Step 7: Workflows tab ----------
log ""
log "Step 7: Workflows tab snapshot..."
echo ""

$CLIENT monitor --once --tab workflows
echo ""
ok "Workflows tab rendered"

# ---------- Step 8: Connection tab ----------
log ""
log "Step 8: Connection tab snapshot (includes server health check)..."
echo ""

$CLIENT monitor --once --tab connection
echo ""
ok "Connection tab rendered"

# ---------- Step 9: Interactive usage guide ----------
log ""
log "Step 9: Interactive dashboard usage"
echo ""
echo "  To launch the live-refreshing dashboard, run:"
echo ""
echo "    umrs-client monitor"
echo ""
echo "  Tab selection:"
echo "    --tab jobs          Jobs queue (default)"
echo "    --tab status        Server task summaries"
echo "    --tab models        Model registry"
echo "    --tab workflows     Workflow registry"
echo "    --tab connection    Connection profile & health"
echo ""
echo "  Other options:"
echo "    --interval 5           Refresh every 5 seconds (default: 3)"
echo "    --status running       Filter jobs by status"
echo "    --project myproject    Filter jobs by project"
echo "    --once                 Print once and exit (no live loop)"
echo "    -c /path/to/conf.json  Mount config (enables in-TUI reload)"
echo ""
echo "  Keyboard shortcuts (live mode):"
echo "    1-5     Switch tab"
echo "    q       Quit"
echo "    r       Force refresh"
echo "    ?       Toggle help overlay"
echo "    j / ↓   Move selection down    (Jobs, Models, Workflows)"
echo "    k / ↑   Move selection up      (Jobs, Models, Workflows)"
echo "    Enter   Toggle detail view     (Jobs, Models, Workflows)"
echo "    f       Cycle status filter    (Jobs)"
echo "    /       Toggle compact / wide  (Jobs)"
echo "    c       Cancel selected job    (Jobs)"
echo "    R       Trigger server reload  (Status, requires --conf)"
echo ""

banner "Demo 22 complete"
