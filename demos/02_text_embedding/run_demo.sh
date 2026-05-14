#!/usr/bin/env bash
# =============================================================================
# Demo 02: Text Embedding
# =============================================================================
# Uploads text files → waits for server-side embedding → exports vectors.
#
# Prerequisites:
#   - UMRS server reachable (SSH tunnels if remote: 5555, 5556, 9000)
#   - UMRS server running with Text Embedding experience
#   - umrs-client installed and in PATH
#
# Bucket alias used: demo-textfiles  (from experience.json fragment)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="text_embed" source "${SCRIPT_DIR}/../lib/common.sh"

# Ensure Text Embedding experience is in user config (idempotent).
ensure_experience "${SCRIPT_DIR}/experience.json"

BUCKET_ALIAS="${TEXT_BUCKET:-demo-textfiles}"
SAMPLE_DIR="${SCRIPT_DIR}/../sample_data/texts"
RESULTS_DIR="${SCRIPT_DIR}/results"
EMBEDDING_WAIT_POLLS="${EMBEDDING_WAIT_POLLS:-60}"
EMBEDDING_WAIT_SLEEP="${EMBEDDING_WAIT_SLEEP:-10}"
EMBEDDING_CMD_TIMEOUT="${EMBEDDING_CMD_TIMEOUT:-20}"

banner "Demo 02: Text Embedding"
echo "Config       : ${CONF}"
echo "Bucket alias : ${BUCKET_ALIAS}"
echo "Sample data  : ${SAMPLE_DIR}"
echo ""

model_visible() {
    local scope="$1" name="$2"
    run_client_bounded "$EMBEDDING_CMD_TIMEOUT" --output json model list --scope "$scope" 2>/dev/null \
        | grep -q "\"name\": \"${name}\""
}

run_client_bounded() {
    local timeout="$1"
    shift
    CLIENT="$CLIENT" CLIENT_TIMEOUT="$timeout" python3 - "$@" <<'PY'
import os
import shlex
import subprocess
import sys

timeout = float(os.environ["CLIENT_TIMEOUT"])
cmd = shlex.split(os.environ["CLIENT"]) + sys.argv[1:]

try:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
except subprocess.TimeoutExpired as exc:
    output = exc.stdout or ""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    sys.stdout.write(output)
    print(f"command timed out after {timeout:g}s: {' '.join(cmd)}")
    raise SystemExit(124)

sys.stdout.write(proc.stdout or "")
raise SystemExit(proc.returncode)
PY
}

wait_for_embeddings_csv() {
    local max="${1:-60}" sleep_sec="${2:-10}"
    local out_file="${RESULTS_DIR}/embeddings.csv"
    local tmp_file="${RESULTS_DIR}/.embeddings.partial.csv"

    mkdir -p "$RESULTS_DIR"
    rm -f "$tmp_file"

    for i in $(seq 1 "$max"); do
        local status_raw
        status_raw="$(run_client_bounded "$EMBEDDING_CMD_TIMEOUT" --output text status process_objects 2>/dev/null || true)"
        if echo "$status_raw" | grep -qi "error\|failed\|aborted"; then
            warn "PROCESS_OBJECTS reported an error:"
            while IFS= read -r line; do
                [[ -n "$line" ]] && warn "  $line"
            done <<< "$status_raw"
            rm -f "$tmp_file"
            return 1
        fi

        if run_client_bounded "$EMBEDDING_CMD_TIMEOUT" export_embeddings "$BUCKET_ALIAS" "$tmp_file" --format csv >/dev/null 2>&1; then
            if [[ -f "$tmp_file" ]]; then
                local lines
                lines=$(wc -l < "$tmp_file" | tr -d ' ')
                if [[ "$lines" -gt 1 ]]; then
                    mv "$tmp_file" "$out_file"
                    log "  embeddings ready after ${i} poll(s)"
                    return 0
                fi
            fi
        fi

        [[ $((i % 6)) -eq 0 ]] && log "  waiting... ($i/$max)"
        sleep "$sleep_sec"
    done

    rm -f "$tmp_file"
    return 1
}

# ---------- Step 1: Pre-flight ----------
log "Step 1: Pre-flight checks..."
preflight

# ---------- Step 2: Prepare embedding model ----------
log ""
log "Step 2: Ensuring embedding model is available..."
MODEL_REPO="${EMBEDDING_MODEL:-intfloat/e5-base-v2}"
MODEL_NAME="${EMBEDDING_MODEL_NAME:-${MODEL_REPO##*/}}"
MODEL_STAGING_DIR="${MODEL_STAGING_DIR:-}"
if model_visible system "$MODEL_NAME" || model_visible user "$MODEL_NAME"; then
    ok "Embedding model already visible in the registry: ${MODEL_NAME}"
elif [[ -n "$MODEL_STAGING_DIR" ]]; then
    $CLIENT model download "$MODEL_REPO" --scope user --server-side --staging-dir "$MODEL_STAGING_DIR" 2>&1 \
        | while IFS= read -r line; do log "  $line"; done
    ok "Embedding model download requested via MODEL_STAGING_DIR=${MODEL_STAGING_DIR}"
else
    warn "Embedding model '${MODEL_NAME}' is not visible in the registry."
    warn "Continuing because some deployments pre-stage models outside the user-visible catalog."
    warn "Set MODEL_STAGING_DIR to a writable, allowed server path if staging is required."
fi

# ---------- Step 3: Upload text data ----------
log ""
log "Step 3: Uploading text files to bucket '${BUCKET_ALIAS}'..."
[[ -d "$SAMPLE_DIR" ]] || die "Sample data not found: $SAMPLE_DIR"
FILE_COUNT=$(find "$SAMPLE_DIR" -maxdepth 1 -type f -name "*.txt" | wc -l)
log "  Found ${FILE_COUNT} text file(s)"
$CLIENT upload_dir "$SAMPLE_DIR" "$BUCKET_ALIAS"
ok "Text files uploaded"

# ---------- Step 4: Wait for embeddings ----------
log ""
log "Step 4: Waiting for embedding rows (${EMBEDDING_WAIT_POLLS} polls × ${EMBEDDING_WAIT_SLEEP}s)..."
wait_for_embeddings_csv "$EMBEDDING_WAIT_POLLS" "$EMBEDDING_WAIT_SLEEP" \
    || die "Timed out waiting for non-empty exported embeddings."

# ---------- Step 5: Verify & Export ----------
log ""
log "Step 5: Verifying exported embeddings CSV..."
ok "Embeddings exported to ${RESULTS_DIR}/embeddings.csv"
if [[ -f "${RESULTS_DIR}/embeddings.csv" ]]; then
    LINES=$(wc -l < "${RESULTS_DIR}/embeddings.csv")
    log "  ${LINES} row(s) in embeddings.csv"
fi
# ---------- Done ----------
log ""
banner "Demo 02 Complete"
log "Results in: ${RESULTS_DIR}/"
