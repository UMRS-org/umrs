#!/usr/bin/env bash
# =============================================================================
# Pipeline SDK Demo — rccs-cloud deployment
# =============================================================================
# Demonstrates the 6-phase Pipeline SDK lifecycle using the LOCAL backend
# (runs in-process, no extra Slurm job needed) against sample text data.
#
# Phases: configure → ingest → compute → collect → verify → present
#
# Prerequisites:
#   - umrs package installed (setup_client.sh)
#
# Usage:
#   ./06_pipeline_sdk.sh                          # uses bundled sample_data/
#   ./06_pipeline_sdk.sh /path/to/texts           # your own text directory
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/sample_data}"

log()  { echo "[pipeline_sdk] $*"; }
die()  { echo "[pipeline_sdk] ERROR: $*" >&2; exit 1; }

# ─── Preflight ───────────────────────────────────────────────────────────────
python3 -c "from umrs.pipeline_sdk import Pipeline, PipelineConfig; print('Pipeline SDK OK')" 2>/dev/null || \
    die "umrs package not importable. Run setup_client.sh first."
[[ -d "$DATA_DIR" ]] || die "Data directory not found: $DATA_DIR"

log "Data dir: $DATA_DIR"
echo

# ─── Run the TextAnalysisPipeline ────────────────────────────────────────────
log "Running TextAnalysisPipeline (LOCAL backend) ..."
log "  Phases: configure → ingest → compute → collect → verify → present"
echo

cd "$REPO_ROOT/demos/11_pipeline_sdk"
python3 text_analysis_pipeline.py --input-dir "$DATA_DIR"
EXIT_CODE=$?

echo
if [[ "$EXIT_CODE" -eq 0 ]]; then
    log "Pipeline completed successfully."
else
    log "Pipeline finished with $EXIT_CODE failure(s)."
fi

# ─── Hint: swap to UMRS backend ──────────────────────────────────────────────
echo
log "To run with real UMRS embedding backend (ZMQ via SSH tunnel):"
log "  1. Edit text_analysis_pipeline.py:"
log "     ComputeBackend.LOCAL  →  ComputeBackend.UMRS_EMBEDDING"
log "     user_config_path = '$HOME/.config/umrs/users/user1.json'"
log "  2. Re-run: python3 $REPO_ROOT/demos/11_pipeline_sdk/text_analysis_pipeline.py --input-dir $DATA_DIR"

exit $EXIT_CODE
