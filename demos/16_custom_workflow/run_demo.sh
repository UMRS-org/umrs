#!/usr/bin/env bash
# =============================================================================
# Demo 16: Custom Server-Side Workflow
# =============================================================================
# Demonstrates uploading a custom workflow (workflow.json + step scripts)
# to the UMRS server and running it through a workflow-backed bucket alias.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running
#   - umrs-client installed and in PATH
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="custom_wf" source "${SCRIPT_DIR}/../lib/common.sh"

WORKFLOW_DIR="${SCRIPT_DIR}/workflow"
RESULTS_DIR="${SCRIPT_DIR}/results"
BUCKET_ALIAS="${TEXT_BUCKET:-user1-custom-text}"

banner "Demo 16: Custom Server-Side Workflow"
echo "Config       : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo ""

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Prepare workflow files ----------
log ""
log "Step 2: Preparing custom workflow files..."
mkdir -p "$WORKFLOW_DIR" "$RESULTS_DIR"

cat > "${WORKFLOW_DIR}/workflow.json" << 'JSON_EOF'
{
  "name": "custom-text-counter",
  "version": "1.0",
  "description": "Count words and characters in uploaded text files",
  "source": {
    "type": "uploaded",
    "path": "."
  },
  "execution_mode": "in_process",
  "steps": [
    {
      "name": "count_words",
      "entrypoint": "counter:WordCounter",
      "config": {"min_word_length": 3}
    }
  ],
  "output": {
    "return_type": "file"
  }
}
JSON_EOF

cat > "${WORKFLOW_DIR}/counter.py" << 'PYEOF'
"""Custom step: count words and characters in text files."""

class WordCounter:
    name = "count_words"

    def setup(self, config):
        self.min_word_length = config.get("min_word_length", 1)

    def teardown(self):
        pass

    def run(self, ctx):
        results = {}
        for key in ctx.input_keys:
            if key.endswith(".txt"):
                data = ctx.download_bytes(key)
                text = data.decode("utf-8", errors="replace")
                words = [w for w in text.split() if len(w) >= self.min_word_length]
                results[key] = {
                    "total_words": len(text.split()),
                    "filtered_words": len(words),
                    "characters": len(text),
                    "lines": text.count("\n") + 1,
                }

        import json
        summary = json.dumps(results, indent=2)
        ctx.upload_bytes("word_count_results.json", summary.encode())

        return type("R", (), {
            "output_keys": ["word_count_results.json"],
            "embeddings": None,
            "checksums": {},
            "metadata": {"summary": results},
        })()
PYEOF
ok "Workflow files created"

# ---------- Step 3: Upload workflow ----------
log ""
log "Step 3: Uploading custom workflow to server..."
WORKFLOW_UPLOAD_DIR="$(prepare_workflow_upload_dir "$WORKFLOW_DIR")"
trap 'rm -rf "${WORKFLOW_UPLOAD_DIR:-}"' EXIT
$CLIENT upload-workflow "$WORKFLOW_UPLOAD_DIR" --name "custom-text-counter" 2>&1 \
    | while IFS= read -r line; do log "  $line"; done
ok "Workflow uploaded"

# ---------- Step 4: Upload sample data ----------
log ""
log "Step 4: Uploading sample text data..."
SAMPLE_DIR="${SCRIPT_DIR}/../sample_data/texts"

if [[ -d "$SAMPLE_DIR" ]]; then
    FILE_COUNT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*.txt" | wc -l | tr -d ' ')
    log "  Found ${FILE_COUNT} text file(s)"
    $CLIENT upload_dir "$SAMPLE_DIR" "$BUCKET_ALIAS"
    ok "Data uploaded"
else
    warn "No sample text data found at $SAMPLE_DIR — using existing bucket data"
fi

# ---------- Step 5: Wait and collect results ----------
log ""
log "Step 5: Waiting for workflow outputs..."
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
$CLIENT download_results "$BUCKET_ALIAS" "${RESULTS_DIR}/" --force-overwrite 2>/dev/null \
    && ok "Results downloaded to ${RESULTS_DIR}/" \
    || warn "Could not download results — check server logs"

if [[ -f "${RESULTS_DIR}/word_count_results.json" ]]; then
    log "Results:"
    cat "${RESULTS_DIR}/word_count_results.json"
fi

banner "Demo 16 Complete"
