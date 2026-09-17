# attempt_budget.sh — per-job respawn-budget claim + stale-reaper machinery
# (ADR-048, WP-20 leg: source me; never execute me directly).
#
# Why: the durable retry chain multiplies respawn capacity across attempts —
# attempt N+1 arrives with a fresh per-attempt budget while attempt N's stack
# may still hold the GPU. The ADR-048 invariant makes the budget lifecycle the
# JOB's: a node-local, flock-protected claim file keyed to the durable job id.
#
# Files (under ${JOB_INFO_DIR:-/tmp/job-info}):
#   <job_id>.budget        claim ledger (JSON, flockgal-protected)
#   <job_id>.budget.lock   flock handle for the ledger
#
# Contract (bash 4+ / flock(1) / python3 required):
#
#   source attempt_budget.sh
#   budget_claim_job "<durable_job_id>" "<slurm_id>" \
#       || budget_refuse "<durable_job_id>" && exit 125
#     - atomically increments budget_used when free; records this attempt
#     - ${BUDGET_EXIT_EXHAUSTED=125} honored on exhaustion
#
#   budget_reap_lineage "<durable_job_id>"
#     - reaps this job's own left-over containers/networks whose recorded
#       owner lineage is the same job id (never another job's).
#
#   budget_mark_exhausted "<job_id>" "<slurm_id>"
#     - writes RIVAULT_TERMINAL_REASON=budget_exhausted into the shared
#       rivault-job-<job_id>.info drop so the checker chain (remote slurm,
#       janitor) surfaces the marker, then exits ${BUDGET_EXIT_EXHAUSTED}.

set -u

BUDGET_EXIT_EXHAUSTED=${BUDGET_EXIT_EXHAUSTED:-125}

budget_dir() { printf '%s' "${JOB_INFO_DIR:-/tmp/job-info}"; }

budget_file() { printf '%s/%s.budget' "$(budget_dir)" "$1"; }

budget_lock_path() { printf '%s/%s.budget.lock' "$(budget_dir)" "$1"; }

# Initialize a job budget file (idempotent): budget_total comes from
# RIVAULT_ENGINE_RESPAWN_MAX_PER_JOB (default 3 — same as the legacy
# per-attempt value; today's legacy callers keep their semantic by default).
budget_init() {
    local job_id="$1" total
    total="${RIVAULT_ENGINE_RESPAWN_MAX_PER_JOB:-${RIVAULT_ENGINE_RESPAWN_MAX:-3}}"
    mkdir -p "$(budget_dir)"
    local f; f="$(budget_file "$job_id")"
    [[ -f "$f" ]] && return 0
    printf '{\n  "job_id": "%s",\n  "budget_total": %s,\n  "budget_used": 0,\n  "claims": []\n}\n' \
        "$job_id" "$total" > "$f"
}

# Atomically decide whether one more respawn is allowed (budget_used inc).
# Prints the post-claim ledger; rc 0 = allowed, rc 1 = budget exhausted.
budget_claim_job() {
    local job_id="$1" attempt="${2:-}"
    budget_init "$job_id"
    local lockf tmpf rc
    lockf="$(budget_lock_path "$job_id")"
    tmpf="$(mktemp -t budget-XXXXXX.py)"
    cat >"$tmpf" <<'PYEOF'
import json, sys
path, att = sys.argv[1], sys.argv[2]
with open(path) as fh:
    ledger = json.load(fh)
if ledger["budget_used"] >= ledger["budget_total"]:
    sys.exit(1)
ledger["budget_used"] += 1
if att:
    from datetime import datetime, timezone
    ledger["claims"].append(
        {"attempt": att, "at": datetime.now(timezone.utc).isoformat()}
    )
with open(path, "w") as fh:
    json.dump(ledger, fh)
sys.exit(0)
PYEOF
    : >"$lockf"  # flock(1) requires the file to exist
    flock "$lockf" python3 "$tmpf" "$(budget_file "$job_id")" "$attempt"
    rc=$?
    rm -f "$tmpf"
    return "$rc"
}

budget_used_read() {
    local f; f="$(budget_file "$1")"
    python3 -c "import json; print(json.load(open('$f'))['budget_used'])" 2>/dev/null || echo 0
}

budget_total_read() {
    local f; f="$(budget_file "$1")"
    python3 -c "import json; print(json.load(open('$f'))['budget_total'])" 2>/dev/null || echo "${RIVAULT_ENGINE_RESPAWN_MAX_PER_JOB:-3}"
}

# Reap only same-lineage left-overs. The 6b-3 invariant — never touch another
# job's conmon/crun — is preserved by requiring BOTH the name prefix and the
# durable-job label match. No-op when job knows nothing about this lineage.
budget_reap_lineage() {
    local job_id="$1" label="umrs_durable_job_id"
    local ctr
    podman ps -a --filter "label=${label}=${job_id}" --format '{{.Names}}' 2>/dev/null | while IFS= read -r ctr; do
        [[ -n "$ctr" ]] || continue
        echo "[budget] reaping lineage leftover: $ctr (job=$job_id)"
        podman rm -f --time=2 "$ctr" >/dev/null 2>&1 || true
    done
    # Per-job venue networks created by venue.sh: <net>-<slurm-id> — only the
    # lineage-shaped ones. venue.sh already names its own gc class per job.
    local net_prefix="genai-loop-net-"
    podman network ls --format '{{.Name}}' 2>/dev/null | grep "^${net_prefix}" | \
    while IFS= read -r nw; do
        # Skip just-created networks whose lineage id string is present in the
        # current job info dir's recorded ids; the venue sweeps them instead.
        [[ -z "$nw" ]] && continue
        if ! grep -rq "$nw" "${JOB_INFO_DIR:-/tmp/job-info}" 2>/dev/null; then
            echo "[budget] skipping foreign-looking network $nw (state recorded elsewhere)"
            continue
        fi
        podman network rm "$nw" >/dev/null 2>&1 || true
    done
    return 0
}

# Mark the job info file with the terminal marker + exit 125 (durable).
budget_mark_exhausted() {
    local job_id="$1" slurm_job_id="${2:-}" infof
    infof="$(budget_dir)/rivault-job-${slurm_job_id:-$job_id}.info"
    mkdir -p "$(dirname "$infof")"
    {
        grep -Ev "^RIVAULT_TERMINAL_REASON=" "$infof" 2>/dev/null || true
        printf 'RIVAULT_TERMINAL_REASON=budget_exhausted\n'
        printf 'RIVAULT_BUDGET_USED=%s\n' "$(budget_used_read "$job_id")"
        printf 'RIVAULT_BUDGET_TOTAL=%s\n' "$(budget_total_read "$job_id")"
    } > "${infof}.tmp.$$"
    mv -f "${infof}.tmp.$$" "$infof"
    echo "[budget] engine respawn budget exhausted for job=$job_id (${ADV:-}&exit ${BUDGET_EXIT_EXHAUSTED})" >&2
    return "${BUDGET_EXIT_EXHAUSTED}"
}
