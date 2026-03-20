#!/usr/bin/env bash
# =============================================================================
# Demo 11 — Pipeline SDK Workflow
# =============================================================================
# Demonstrates the Pipeline SDK 6-phase lifecycle using a LOCAL compute
# backend.  Runs entirely in-process — no UMRS server or containers needed.
#
# To test with real services, modify the pipeline to use UMRS_EMBEDDING
# or BENTOML_TASK backend (see text_analysis_pipeline.py comments).
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LOG_PREFIX="[pipeline_sdk]"
log() { echo "$LOG_PREFIX $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }

# ---------- Step 1: Verify Pipeline SDK is importable ----------
log "Step 1: Checking Pipeline SDK installation..."
python3 -c "from umrs.pipeline_sdk import Pipeline, PipelineConfig; print('Pipeline SDK OK')" || \
    die "Pipeline SDK not importable. Install with: pip install -e $PROJECT_ROOT/src"
log "Pipeline SDK is available."

# ---------- Step 2: Show sample data ----------
log ""
log "Step 2: Sample text files in data/file_of_texts/:"
ls -la "$PROJECT_ROOT/data/file_of_texts/" 2>/dev/null || \
    die "Sample data not found at $PROJECT_ROOT/data/file_of_texts/"

# ---------- Step 3: Run the pipeline ----------
log ""
log "Step 3: Running TextAnalysisPipeline (LOCAL backend)..."
log "  Phases: configure → ingest → compute → collect → verify → present"
log ""

cd "$SCRIPT_DIR"
python3 text_analysis_pipeline.py --input-dir "$PROJECT_ROOT/data/file_of_texts"
EXIT_CODE=$?

log ""
if [ $EXIT_CODE -eq 0 ]; then
    log "Pipeline completed successfully."
else
    log "Pipeline completed with $EXIT_CODE failure(s)."
fi

# ---------- Step 4: Run with custom data (optional) ----------
log ""
log "Step 4: You can also run with custom data:"
log "  python3 text_analysis_pipeline.py --input-dir /path/to/your/texts --top-n 5"

# ---------- Step 5: Show full-service example ----------
log ""
log "Step 5: For production usage with UMRS server + BentoML:"
log "  - Change ComputeBackend.LOCAL → ComputeBackend.UMRS_EMBEDDING"
log "  - Set user_config_path, umrs_server_ip in PipelineConfig"
log "  - Use UMRSClientConnector in compute() / collect()"
log "  - See src/umrs/pipeline_sdk/connectors/ for available connectors"

exit $EXIT_CODE
