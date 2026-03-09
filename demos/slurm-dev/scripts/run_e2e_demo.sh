#!/usr/bin/env bash
# =============================================================================
# UMRS E2E Demo: Complete orchestration script
# =============================================================================
#
# Runs the embedding-only demo flow:
#   Phase 0  – Cleanup (Slurm jobs, Rivault containers, task/status DBs,
#              embedding tables, stale Slurm output files)
#   Phase 1  – Start control-plane services
#   Phase 1b – Prepare embedding model on shared volume
#   Phase 2  – Upload demo data
#   Phase 3  – Wait for embeddings to finish
#   Phase 4  – Summary
#
# The always-on LLM stack (demo-llm.yaml) is NOT started by this script.
# Launch it separately if you need RAG Q&A:
#   podman-compose --env-file rivault/demo-llm.env \
#       -f rivault/demo-llm.yaml -f rivault/x_nvidia-override.yaml up -d
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

COMPOSE_FILE="$DEMO_DIR/docker-compose.yaml"

AI_LOGIN_CONTAINER="ai-login"
UMRS_CLIENT_CONTAINER="umrs-client"
USER_CONF="/root/.config/umrs/users/user_rivault.json"

# Postgres credentials (sourced from .env or defaults)
PG_CONTAINER="demo-postgres"
PG_SUPERUSER="${POSTGRES_USER:-umrs}"
PG_SUPERUSER_PASSWORD="${POSTGRES_PASSWORD:-umrs_dev}"

LOG_PREFIX="[run_e2e_demo]"

log() { echo "$LOG_PREFIX $*"; }
warn() { echo "$LOG_PREFIX WARNING: $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

wait_for_condition() {
    local max_attempts="$1"
    local sleep_seconds="$2"
    local description="$3"
    local probe_cmd="$4"

    local i
    for i in $(seq 1 "$max_attempts"); do
        if eval "$probe_cmd" >/dev/null 2>&1; then
            log "$description: ready"
            return 0
        fi
        if [[ "$i" -eq "$max_attempts" ]]; then
            return 1
        fi
        sleep "$sleep_seconds"
    done
    return 1
}

detect_compose() {
    if command -v podman-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(podman-compose)
        return
    fi

    if command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(podman compose)
        return
    fi

    die "Neither podman-compose nor podman compose is available."
}

compose_main() {
    (cd "$DEMO_DIR" && "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" "$@")
}

preflight_checks() {
    log "Running preflight checks"

    require_cmd podman
    require_cmd python3
    detect_compose

    [[ -f "$COMPOSE_FILE" ]] || die "Missing compose file: $COMPOSE_FILE"

    log "Compose command: ${COMPOSE_CMD[*]}"
}

phase0_cleanup() {
    echo
    log "Phase 0: Cleaning up previous state"

    # --- 0a. Cancel any active Slurm jobs ---
    log "Cancelling active Slurm jobs..."
    if podman inspect "$AI_LOGIN_CONTAINER" >/dev/null 2>&1; then
        local job_ids jid
        job_ids="$(podman exec "$AI_LOGIN_CONTAINER" squeue -h -o '%A' 2>/dev/null || true)"
        if [[ -n "$job_ids" ]]; then
            for jid in $job_ids; do
                log "  scancel $jid"
                podman exec "$AI_LOGIN_CONTAINER" scancel "$jid" 2>/dev/null || true
            done
            sleep 5
        else
            log "  No active Slurm jobs"
        fi
    else
        log "  ai-login not running, skipping scancel"
    fi

    # --- 0b. Remove old Rivault containers and network (host, root level) ---
    log "Removing old Rivault containers..."
    sudo podman ps -qa --filter 'name=rivault_' --format '{{.ID}}' 2>/dev/null | \
        xargs -r sudo podman rm -f 2>/dev/null || true
    sudo podman network rm -f genai-private-net 2>/dev/null || true
    # Also check user-level
    podman ps -qa --filter 'name=rivault_' --format '{{.ID}}' 2>/dev/null | \
        xargs -r podman rm -f 2>/dev/null || true
    podman network rm -f genai-private-net 2>/dev/null || true
    log "  Rivault containers and network cleaned"

    # --- 0c. Bring down control plane (so it restarts fresh) ---
    log "Bringing down control plane..."
    compose_main down 2>/dev/null || true

    # --- 0d. Clear UMRS task history (SQLite) and status store (diskcache) ---
    # These are on the umrs-config volume, which is bind-mounted at ./umrs-config
    local umrs_cfg="$DEMO_DIR/umrs-config"
    if [[ -d "$umrs_cfg" ]]; then
        # Remove task history DB
        if [[ -f "$umrs_cfg/tasks_database/tasks.db" ]]; then
            rm -f "$umrs_cfg/tasks_database/tasks.db"
            log "  Cleared tasks database"
        fi
        # Remove diskcache status store
        if [[ -d "$umrs_cfg/tmp/status_store" ]]; then
            rm -rf "$umrs_cfg/tmp/status_store"
            log "  Cleared status store cache"
        fi
        # Remove old Slurm output/error files
        find "$umrs_cfg/tmp/" -maxdepth 1 -name 'slurm-*.out' -o -name 'slurm-*.err' 2>/dev/null | \
            xargs -r rm -f
        log "  Cleared old Slurm output files"
    fi

    # --- 0e. Reset embedding tables in pgvector ---
    # The postgres volume persists across restarts; bring up postgres alone
    # to truncate embedding data from prior failed runs.
    log "Resetting embedding tables in pgvector..."
    compose_main up -d "$PG_CONTAINER" 2>/dev/null || true
    wait_for_condition 30 2 "postgres" "podman exec $PG_CONTAINER pg_isready -U $PG_SUPERUSER" || {
        warn "postgres did not become ready; skipping DB reset"
        return 0
    }

    # List user databases (exclude system DBs)
    local user_dbs
    user_dbs="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -tAc \
        "SELECT datname FROM pg_database WHERE datname NOT IN ('postgres','template0','template1','umrs');" 2>/dev/null || true)"
    if [[ -n "$user_dbs" ]]; then
        for db in $user_dbs; do
            log "  Dropping database: $db"
            # Terminate active connections first
            podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -c \
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$db' AND pid <> pg_backend_pid();" >/dev/null 2>&1 || true
            podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -c \
                "DROP DATABASE IF EXISTS \"$db\";" 2>/dev/null || true
        done
    fi

    # Drop non-superuser roles (the server recreates them on startup)
    local user_roles
    user_roles="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -tAc \
        "SELECT rolname FROM pg_roles WHERE rolname NOT IN ('$PG_SUPERUSER','pg_read_all_data','pg_write_all_data','pg_read_all_settings','pg_read_all_stats','pg_stat_scan_tables','pg_monitor','pg_read_server_files','pg_write_server_files','pg_execute_server_program','pg_signal_backend','pg_checkpoint','pg_use_reserved_connections','pg_create_subscription','pg_maintain') AND rolname NOT LIKE 'pg_%';" 2>/dev/null || true)"
    if [[ -n "$user_roles" ]]; then
        for role in $user_roles; do
            log "  Dropping role: $role"
            podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -c \
                "DROP ROLE IF EXISTS \"$role\";" 2>/dev/null || true
        done
    fi
    log "  Database reset complete"

    # Stop postgres again (phase1 will bring everything up together)
    compose_main down 2>/dev/null || true

    log "Phase 0 cleanup complete"
}

phase1_start_control_plane() {
    echo
    log "Phase 1: Starting control-plane services"

    if ! pgrep -f slurmd >/dev/null 2>&1; then
        warn "Host slurmd is not running yet; waiting up to 120s"
        wait_for_condition 24 5 "host slurmd" "pgrep -f slurmd" || {
            die "slurmd was not detected. Start it first: sudo bash $SCRIPT_DIR/start-host-slurmd.sh"
        }
    fi
    log "Host slurmd detected (PID: $(pgrep -f slurmd | head -n 1))"

    compose_main up -d

    wait_for_condition 60 5 "ai-login healthy" "podman inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' $AI_LOGIN_CONTAINER | grep -qiE 'healthy|running'" || {
        die "ai-login did not become healthy/running"
    }
    wait_for_condition 60 5 "umrs-client running" "podman inspect -f '{{.State.Status}}' $UMRS_CLIENT_CONTAINER | grep -qi running" || {
        die "umrs-client did not reach running state"
    }

    wait_for_condition 60 5 "Slurm node registration" "podman exec $AI_LOGIN_CONTAINER sinfo -N -h -o '%N %T' | grep -Eq '^compute-01 (idle|mix|alloc)'" || {
        warn "compute-01 not reported as idle/mix/alloc in time"
        warn "Check: podman exec $AI_LOGIN_CONTAINER sinfo -N"
    }
}

phase1b_prepare_embedding_model() {
    echo
    log "Phase 1b: Preparing embedding model on ai-login (shared volume)"

    # Download the embedding model into the shared volume so that:
    #   - vLLM (launched by Slurm) can serve it from /shared/models/hub/
    #   - SentenceTransformers (local backend) can load it from SENTENCE_TRANSFORMERS_HOME
    #
    # The model name is read from the user config; falls back to intfloat/e5-base-v2.
    log ">>> podman exec $AI_LOGIN_CONTAINER python3 (download embedding model)"
    podman exec "$AI_LOGIN_CONTAINER" python3 -u - <<'PYEOF'
import json, os, sys

# --- 1. Determine which embedding model(s) to prepare ---
user_cfg_path = "/root/.config/umrs/users/user_rivault.json"
models_to_download = set()
try:
    with open(user_cfg_path) as f:
        cfg = json.load(f)
    for xp in cfg.get("experiences", []):
        for mod in xp.get("modalities", []):
            m = mod.get("model", {})
            mt = m.get("model_type", "")
            if mt.startswith("rivault://") or mt.startswith("openai://"):
                name = m.get("model_name")
                if name:
                    models_to_download.add(name)
except Exception as e:
    print(f"  Warning: could not parse user config: {e}", file=sys.stderr)

if not models_to_download:
    models_to_download = {"intfloat/e5-base-v2"}

# --- 2. Download into HF hub cache (for vLLM) ---
# vLLM mounts VLLM_PATH_TO_HUGGINGFACE as /root/.cache/huggingface,
# and looks for models in <mount>/hub/. On the host this is rivault/models/hub/.
# Inside ai-login the shared volume is /shared/models, so the hub cache
# that vLLM will see is /shared/models/hub/.
HUB_CACHE = "/shared/models/hub"
os.makedirs(HUB_CACHE, exist_ok=True)

from huggingface_hub import snapshot_download, scan_cache_dir

for model_name in sorted(models_to_download):
    # Check if already present in hub cache
    try:
        info = scan_cache_dir(HUB_CACHE)
        cached = {r.repo_id for r in info.repos}
        if model_name in cached:
            print(f"  {model_name}: already in hub cache, skipping download.")
            continue
    except Exception:
        pass

    print(f"  Downloading {model_name} into {HUB_CACHE} ...")
    snapshot_download(repo_id=model_name, cache_dir=HUB_CACHE)
    print(f"  {model_name}: done.")

print("  Embedding model preparation complete.")
PYEOF

    log "Embedding model ready on shared volume"
}

phase2_upload_data() {
    echo
    log "Phase 2: Uploading demo data"
    log ">>> podman exec $UMRS_CLIENT_CONTAINER bash /opt/scripts/setup_demo_data.sh"
    podman exec "$UMRS_CLIENT_CONTAINER" bash /opt/scripts/setup_demo_data.sh
}

phase3_wait_for_embeddings() {
    echo
    log "Phase 3: Waiting for embedding tasks to finish (timeout: 10 minutes)"

    log ">>> podman exec $UMRS_CLIENT_CONTAINER python3 /opt/umrs/client_umrs.py -c $USER_CONF --output json status all"

    local attempts=60
    local i
    for i in $(seq 1 "$attempts"); do
        local status_raw status_json task_state
        status_raw="$(podman exec "$UMRS_CLIENT_CONTAINER" python3 /opt/umrs/client_umrs.py -c "$USER_CONF" --output json status all 2>/dev/null || true)"
        status_json="$(echo "$status_raw" | grep '^{' | tail -n 1 || true)"

        if [[ -n "$status_json" ]]; then
            # Parse task states: returns "running N", "error N", or "done"
            task_state="$(echo "$status_json" | python3 -c '
import json
import sys

def get_threads(payload):
    summaries = payload.get("summaries")
    if isinstance(summaries, dict):
        for _, entry in summaries.items():
            for thread in entry.get("data", []):
                yield thread

    summary = payload.get("summary")
    if isinstance(summary, dict):
        data = summary.get("data")
        if isinstance(data, list):
            for thread in data:
                yield thread

payload = json.load(sys.stdin)
running = 0
errors = 0
for t in get_threads(payload):
    s = str(t.get("status", "")).lower()
    if s == "running":
        running += 1
    elif s == "error":
        errors += 1

if running > 0:
    print(f"running {running}")
elif errors > 0:
    print(f"error {errors}")
else:
    print("done")
' 2>/dev/null || echo 'unknown')"

            case "$task_state" in
                done)
                    log "All embedding tasks completed successfully"
                    return 0
                    ;;
                error*)
                    local err_count="${task_state#error }"
                    warn "$err_count task(s) ended with Error status!"
                    warn "Check server logs: podman logs $AI_LOGIN_CONTAINER"
                    return 1
                    ;;
                running*)
                    local run_count="${task_state#running }"
                    if [[ "$i" -eq 1 || $((i % 6)) -eq 0 ]]; then
                        log "Embedding tasks still running: $run_count (attempt $i/$attempts)"
                    fi
                    ;;
                *)
                    if [[ "$i" -eq 1 || $((i % 6)) -eq 0 ]]; then
                        warn "Unknown task state: $task_state (attempt $i/$attempts)"
                    fi
                    ;;
            esac
        else
            if [[ "$i" -eq 1 || $((i % 6)) -eq 0 ]]; then
                warn "Could not parse status JSON yet (attempt $i/$attempts)"
            fi
        fi

        sleep 10
    done

    die "Embedding wait timed out after $((attempts * 10))s. Check logs: podman logs $AI_LOGIN_CONTAINER"
}

phase3b_verify_embeddings_in_db() {
    echo
    log "Phase 3b: Verifying embeddings in pgvector database"

    # Query all user databases (excluding system DBs) for embedding tables
    local user_dbs
    user_dbs="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d postgres -tAc \
        "SELECT datname FROM pg_database WHERE datname NOT IN ('postgres','template0','template1','umrs');" 2>/dev/null || true)"

    if [[ -z "$user_dbs" ]]; then
        warn "No user databases found in pgvector"
        return 1
    fi

    local total_embeddings=0
    local total_nulls=0
    local tables_checked=0
    local has_problems=0

    for db in $user_dbs; do
        log "  Database: $db"

        # Find all tables that have an 'embedding' column
        local tables_with_emb
        tables_with_emb="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
            "SELECT table_name FROM information_schema.columns WHERE column_name='embedding' AND table_schema='public';" 2>/dev/null || true)"

        if [[ -z "$tables_with_emb" ]]; then
            warn "    No tables with 'embedding' column found"
            has_problems=1
            continue
        fi

        for tbl in $tables_with_emb; do
            tables_checked=$((tables_checked + 1))

            # Count total rows
            local row_count
            row_count="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
                "SELECT count(*) FROM \"$tbl\";" 2>/dev/null || echo 0)"
            row_count="$(echo "$row_count" | tr -d ' ')"

            # Count rows with non-null embeddings
            local emb_count
            emb_count="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
                "SELECT count(*) FROM \"$tbl\" WHERE embedding IS NOT NULL;" 2>/dev/null || echo 0)"
            emb_count="$(echo "$emb_count" | tr -d ' ')"

            # Count rows with null embeddings
            local null_count
            null_count="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
                "SELECT count(*) FROM \"$tbl\" WHERE embedding IS NULL;" 2>/dev/null || echo 0)"
            null_count="$(echo "$null_count" | tr -d ' ')"

            # Get embedding vector dimension
            local emb_dim
            emb_dim="$(podman exec "$PG_CONTAINER" psql -U "$PG_SUPERUSER" -d "$db" -tAc \
                "SELECT atttypmod FROM pg_attribute WHERE attrelid='public.\"$tbl\"'::regclass AND attname='embedding';" 2>/dev/null || echo '?')"
            emb_dim="$(echo "$emb_dim" | tr -d ' ')"

            total_embeddings=$((total_embeddings + emb_count))
            total_nulls=$((total_nulls + null_count))

            if [[ "$row_count" -eq 0 ]]; then
                warn "    $tbl: 0 rows (EMPTY — no embeddings computed!)"
                has_problems=1
            elif [[ "$null_count" -gt 0 ]]; then
                warn "    $tbl: $row_count rows, $emb_count with embeddings (dim=$emb_dim), $null_count NULL embeddings"
                has_problems=1
            else
                log "    $tbl: $row_count rows, all with embeddings (dim=$emb_dim) ✓"
            fi
        done
    done

    echo
    log "  Summary: $tables_checked table(s) checked, $total_embeddings embedding(s) computed, $total_nulls NULL embedding(s)"

    if [[ "$total_embeddings" -eq 0 ]]; then
        warn "NO EMBEDDINGS FOUND in the database!"
        warn "The embedding pipeline did not produce any results."
        warn "Check: podman logs $AI_LOGIN_CONTAINER | tail -50"
        return 1
    fi

    if [[ "$has_problems" -ne 0 ]]; then
        warn "Some issues detected (see above)"
        return 1
    fi

    log "All embeddings verified successfully"
    return 0
}

phase4_summary() {
    echo
    log "Phase 4: E2E embedding demo complete"
    echo
    echo "Services running:"
    echo "  UMRS ZMQ server:   localhost:5555"
    echo "  MinIO Console:     http://localhost:19001"
    echo "  PostgreSQL:        localhost:15432"
    echo "  UMRS RAG:          http://localhost:8081"
    echo
    echo "Useful commands:"
    echo "  Status:      podman exec $UMRS_CLIENT_CONTAINER python3 /opt/umrs/client_umrs.py -c $USER_CONF --output json status all"
    echo "  Slurm queue: podman exec $AI_LOGIN_CONTAINER squeue"
    echo "  Logs:        podman logs $AI_LOGIN_CONTAINER"
    echo
    echo "To stop control plane:"
    echo "  ${COMPOSE_CMD[*]} -f $COMPOSE_FILE down"
    echo
    echo "To also start the LLM stack for RAG Q&A (separate GPU):"
    echo "  cd $DEMO_DIR && podman-compose --env-file rivault/demo-llm.env \\"
    echo "      -f rivault/demo-llm.yaml -f rivault/x_nvidia-override.yaml up -d"
}

main() {
    preflight_checks
    phase0_cleanup
    phase1_start_control_plane
    phase1b_prepare_embedding_model
    phase2_upload_data
    phase3_wait_for_embeddings || {
        warn "Phase 3 detected errors — continuing to DB check for diagnostics"
    }
    phase3b_verify_embeddings_in_db || {
        warn "Phase 3b: Embedding verification FAILED"
    }
    phase4_summary
}

main "$@"
