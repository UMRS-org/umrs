#!/usr/bin/env bash
# =============================================================================
# Text Embedding Demo — hpbd-atom deployment
# =============================================================================
# Uploads text files → waits for embeddings → checks status.
# Run from your local Mac with SSH tunnels active.
#
# Usage:
#   ./01_text_embedding.sh                      # uses bundled sample data
#   ./01_text_embedding.sh /path/to/my_texts    # uses your own directory
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
TEXT_BUCKET="${UMRS_TEXT_BUCKET:-user1-texts}"

# Allow user to supply a custom data directory as first argument
DATA_DIR="${1:-$SCRIPT_DIR/sample_data}"

log()  { echo "[text_emb] $*"; }
warn() { echo "[text_emb] WARNING: $*"; }
die()  { echo "[text_emb] ERROR: $*" >&2; exit 1; }

CLIENT="umrs-client -c $USER_CONF"

# ---------- Step 0: Preflight ----------
log "Config:  $USER_CONF"
log "Bucket:  $TEXT_BUCKET"
log "Data:    $DATA_DIR"
[[ -f "$USER_CONF" ]] || die "User config not found: $USER_CONF  (run setup_client.sh first)"
[[ -d "$DATA_DIR" ]]  || die "Data directory not found: $DATA_DIR"

echo
log "Step 1: Ping server"
$CLIENT ping || die "Server not reachable. Is ssh_tunnel.sh running?"
log "Server is ready."

# ---------- Step 2: Upload ----------
echo
log "Step 2: Upload text files from $DATA_DIR"
$CLIENT upload_dir "$DATA_DIR" "$TEXT_BUCKET"
log "Upload complete."

# ---------- Step 3: Wait for processing ----------
echo
log "Step 3: Waiting for embeddings (timeout ~5 min)..."

attempts=30
for i in $(seq 1 "$attempts"); do
    status_raw="$($CLIENT --output json status all 2>/dev/null || true)"
    status_json="$(echo "$status_raw" | grep '^{' | tail -n 1 || true)"

    if [[ -n "$status_json" ]]; then
        state="$(echo "$status_json" | python3 -c '
import json, sys
def tasks(p):
    for _, e in (p.get("summaries") or {}).items():
        yield from e.get("data", [])
    s = p.get("summary")
    if isinstance(s, dict):
        yield from s.get("data", [])
p = json.load(sys.stdin)
ts = [t for t in tasks(p) if isinstance(t, dict)]
r = sum(1 for t in ts if t.get("status","").lower() in {"running","submitted","launched","initializing"})
e = sum(1 for t in ts if t.get("status","").lower() in {"error","failed","aborted"})
c = sum(1 for t in ts if t.get("status","").lower() in {"completed","done"})
if r:    print(f"running {r}")
elif e:  print(f"error {e}")
elif c:  print("done")
elif not ts: print("no_tasks")
else:    print("unknown")
' 2>/dev/null || echo 'unknown')"

        case "$state" in
            done)       log "All embedding tasks completed!"; break ;;
            error*)     warn "Some tasks errored. Check server logs."; break ;;
            running*)   [[ $((i % 6)) -eq 0 ]] && log "  still running... ($i/$attempts)" ;;
            no_tasks)   [[ $((i % 6)) -eq 0 ]] && log "  no tasks yet... ($i/$attempts)" ;;
            *)          [[ $((i % 6)) -eq 0 ]] && log "  pending... ($i/$attempts)" ;;
        esac
    fi
    [[ "$i" -eq "$attempts" ]] && die "Timeout waiting for embeddings."
    sleep 10
done

# ---------- Step 4: List buckets ----------
echo
log "Step 4: List available buckets"
$CLIENT list_buckets

# ---------- Step 5: Export (optional) ----------
echo
log "Step 5: Export embeddings"
EXPORT_FILE="${SCRIPT_DIR}/embeddings.csv"
$CLIENT export_embeddings "$TEXT_BUCKET" "$EXPORT_FILE" 2>/dev/null && \
    log "Exported to $EXPORT_FILE" || \
    warn "Export not available (may need direct DB access). Skipping."

echo
log "Done! Embeddings are stored in pgvector on hpbd-atom."
