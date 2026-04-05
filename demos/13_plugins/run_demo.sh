#!/usr/bin/env bash
# =============================================================================
# Demo 13 — Plugin Execution System
# =============================================================================
# Demonstrates the UMRS plugin system: spec definition, validation,
# execution, result collection, and audit logging.
#
# Runs locally — no Slurm or K8s required.
# Shows sbatch preview for what HPC execution looks like.
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LOG_PREFIX="[plugins]"
log() { echo "$LOG_PREFIX $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }

# ---------- Step 1: Verify plugin system is importable ----------
log "Step 1: Checking plugin system installation..."
python3 -c "from umrs.plugins import PluginSpec, PluginStatus; print('Plugin system OK')" || \
    die "Plugin system not importable. Install with: pip install -e $PROJECT_ROOT/src"
log "Plugin system is available."

# ---------- Step 2: Run the demo ----------
log ""
log "Step 2: Running plugin workflow demo..."
log "  Phases: Define → Validate → Preview → Execute → Audit"
log ""

cd "$SCRIPT_DIR"
python3 run_plugin_demo.py --input-dir "$PROJECT_ROOT/data/file_of_texts"
EXIT_CODE=$?

log ""
if [ $EXIT_CODE -eq 0 ]; then
    log "Plugin demo completed successfully."
else
    log "Plugin demo completed with errors."
fi

# ---------- Step 3: Show HPC usage ----------
log ""
log "Step 3: For HPC (Slurm) deployment:"
log "  from umrs.plugins import SlurmPluginRunner"
log "  runner = SlurmPluginRunner(partition='gpu')"
log "  result = await runner.execute(spec)"
log ""
log "  For K8s deployment:"
log "  from umrs.plugins import K8sPluginRunner"
log "  runner = K8sPluginRunner(namespace='rivault-plugins')"
log "  result = await runner.execute(spec)"

exit $EXIT_CODE
