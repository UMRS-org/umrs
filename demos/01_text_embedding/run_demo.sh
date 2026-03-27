#!/usr/bin/env bash
# =============================================================================
# Text Embedding Demo
# =============================================================================
# Uploads text files → waits for embeddings → verifies → exports
# Assumes services are already running (see demos/00_start_services/).
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
SLURM_DEV_DIR="$DEMO_DIR/slurm-dev"

AI_LOGIN_CONTAINER="ai-login"
UMRS_CLIENT_CONTAINER="umrs-client"
USER_CONF="/root/.config/umrs/users/user_rivault.json"
PG_CONTAINER="demo-postgres"
PG_SUPERUSER="${POSTGRES_USER:-umrs}"
TEXT_BUCKET="rivault-demo-bucket"

LOG_PREFIX="[text_embedding]"
log() { echo "$LOG_PREFIX $*"; }
warn() { echo "$LOG_PREFIX WARNING: $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }
run_cmd() {
    echo "$LOG_PREFIX \$ $*"
    "$@"
}

# ---------- Step 1: Ping ----------
log "Step 1: Checking UMRS server..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client ping || \
    die "UMRS server not reachable. Start services first: bash ../00_start_services/start_all.sh"
log "Server is ready."

# ---------- Step 2: Prepare embedding model ----------
log ""
log "Step 2: Downloading embedding model via umrs-client..."
# Uses server-side download: the client sends a download request to the UMRS
# server, which fetches the model from HuggingFace directly on the HPC node.
# This is the normal workflow — the client machine and HPC don't share storage.
MODEL_REPO="${EMBEDDING_MODEL:-intfloat/e5-base-v2}"
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" \
    umrs-client model download "$MODEL_REPO" \
    --scope system --server-side 2>&1 | while IFS= read -r line; do log "  $line"; done
log "Embedding model download requested."

# ---------- Step 3: Upload text data ----------
log ""
log "Step 3: Uploading text files to bucket '${TEXT_BUCKET}'..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" bash /opt/scripts/setup_demo_data.sh

# ---------- Step 4: Wait for embeddings ----------
log ""
log "Step 4: Waiting for embedding tasks to finish (timeout: 10 min)..."

attempts=60
for i in $(seq 1 "$attempts"); do
    status_raw="$(podman exec "$UMRS_CLIENT_CONTAINER" umrs-client \
        --output json status all 2>/dev/null || true)"
    status_json="$(echo "$status_raw" | grep '^{' | tail -n 1 || true)"

    if [[ -n "$status_json" ]]; then
        task_state="$(echo "$status_json" | python3 -c '
import json, sys
def threads(p):
    for _, e in (p.get("summaries") or {}).items():
        yield from e.get("data", [])
    s = p.get("summary")
    if isinstance(s, dict):
        yield from s.get("data", [])
p = json.load(sys.stdin)
tasks = [t for t in threads(p) if isinstance(t, dict)]
in_progress_states = {"running", "submitted", "launched", "initializing"}
done_states = {"completed", "done"}
r = sum(1 for t in tasks if t.get("status","").lower() in in_progress_states)
e = sum(1 for t in tasks if t.get("status","").lower() in {"error", "failed", "aborted"})
c = sum(1 for t in tasks if t.get("status","").lower() in done_states)
if r:
    print(f"running {r}")
elif e:
    print(f"error {e}")
elif c:
    print("done")
elif not tasks:
    print("no_tasks")
else:
    print("unknown")
' 2>/dev/null || echo 'unknown')"

        case "$task_state" in
            done)
                log "All embedding tasks completed!"
                break
                ;;
            error*)
                warn "Some tasks ended with errors. Check: podman logs $AI_LOGIN_CONTAINER"
                break
                ;;
            running*)
                [ $((i % 6)) -eq 0 ] && log "Still running... (attempt $i/$attempts)"
                ;;
            no_tasks)
                [ $((i % 6)) -eq 0 ] && log "No tasks found yet... (attempt $i/$attempts)"
                ;;
            *)
                [ $((i % 6)) -eq 0 ] && log "Tasks in unknown/pending state... (attempt $i/$attempts). Check: podman exec ai-login squeue"
                ;;
        esac
    fi
    [ "$i" -eq "$attempts" ] && die "Timeout waiting for embeddings"
    sleep 10
done

# ---------- Step 5: Verify ----------
log ""
log "Step 5: Verifying embeddings in pgvector..."

user_dbs="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -tAc \
    "SELECT datname FROM pg_database WHERE datname NOT IN ('postgres','template0','template1','umrs');" 2>/dev/null || true)"

for db in $user_dbs; do
    tables="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
        "SELECT table_name FROM information_schema.columns WHERE column_name='embedding' AND table_schema='public';" 2>/dev/null || true)"
    for tbl in $tables; do
        log "\$ podman exec $PG_CONTAINER psql -U $PG_SUPERUSER -d $db -c \"SELECT count(*) FROM $tbl WHERE embedding IS NOT NULL;\""
        count="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
            "SELECT count(*) FROM \"$tbl\" WHERE embedding IS NOT NULL;" 2>/dev/null || echo 0)"
        log "  $db.$tbl: $(echo "$count" | tr -d ' ') embeddings"
    done
done

# ---------- Step 6: Export ----------
log ""
log "Step 6: Exporting embeddings to ./embeddings.csv..."
run_cmd podman exec "$UMRS_CLIENT_CONTAINER" umrs-client \
    export_embeddings "$TEXT_BUCKET" /tmp/embeddings.csv \
    --db-host postgres --db-port 5432 --db-user "$PG_SUPERUSER" --db-password "${POSTGRES_PASSWORD:-umrs_dev}" 2>/dev/null || true
run_cmd podman cp "$UMRS_CLIENT_CONTAINER:/tmp/embeddings.csv" "$SCRIPT_DIR/embeddings.csv" 2>/dev/null || true

if [[ -f "$SCRIPT_DIR/embeddings.csv" ]]; then
    lines=$(wc -l < "$SCRIPT_DIR/embeddings.csv")
    log "Exported $((lines - 1)) rows to embeddings.csv"
else
    warn "Export failed — you can export manually: umrs-client export_embeddings ..."
fi

log ""
log "Text embedding demo complete!"
