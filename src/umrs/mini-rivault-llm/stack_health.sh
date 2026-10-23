# shellcheck shell=bash disable=SC2034  # STACK_HEALTH_* are read by the caller
# stack_health.sh — essential-sidecar health for the serving-stack monitor
# (source me; never execute me directly).
#
# Why (2026-10-22, bdrt jobs 8218/8219): the LiteLLM container exited at start
# (empty DATABASE_URL) while vLLM stayed healthy. The monitor loop in
# serve-model.slurm watched only the engine, so the job ran 30 h while nginx
# answered 502 to every request, UMRS kept the endpoint RUNNING and the GPU
# stayed taken. LiteLLM and the nginx front are the endpoint: the stack is
# not serving when either one is gone.
#
# Contract (bash 4+):
#
#   source stack_health.sh
#   stack_health_init
#   while ...; do
#       if stack_health_tick "<compose project>" "<published port>"; then
#           # STACK_HEALTH_RESPAWN: compose services to recreate, in order
#           # STACK_HEALTH_REASON:  one line for the job log
#           # STACK_HEALTH_PLANNED: 1 when every exited sidecar ended by plan
#           stack_restart_is_free || ...claim one respawn from the budget...
#           for svc in $(stack_removal_order "<project>" "$STACK_HEALTH_RESPAWN"); do
#               ...podman rm -f, dependents first...
#           done
#           ...recreate the same services in the reverse order...
#       fi
#   done
#
# The caller claims each unplanned respawn from the same per-job budget as an
# engine respawn (attempt_budget.sh, ADR-048), so a sidecar that keeps dying
# uses up the budget and the job exits 125 instead of serving 502 forever.
#
# Two signals, checked once per monitor tick:
#   1. a sidecar container is exited/stopped/dead -> respawn that sidecar now;
#   2. the published front answers 502/503/504 or not at all on
#      ${RIVAULT_FRONT_PROBE_PATH}, probed every ${RIVAULT_FRONT_PROBE_EVERY_TICKS}
#      ticks, for ${RIVAULT_FRONT_FAIL_STREAK} probes in a row -> respawn every
#      essential sidecar. Failed probes do not count inside a grace window:
#      ${RIVAULT_FRONT_STARTUP_GRACE_SEC} from the start until the front first
#      answers, ${RIVAULT_FRONT_RESPAWN_GRACE_SEC} after each respawn (2026-10-23
#      review F6: a respawn used to restart the 900 s start grace, so a front
#      that stayed down took ~36 min to use up the budget). Any other HTTP
#      status (200, 401, 404 ...) means nginx reached LiteLLM.
#
# Planned exits (2026-10-23 review F1): uvicorn ends a single-worker LiteLLM
# cleanly after MAX_REQUESTS_BEFORE_RESTART requests ("Maximum request limit",
# exit 0). Such an exit is restarted without charging the job's respawn budget
# (ADR-048 budgets never refill, so a long-lived endpoint would otherwise be
# torn down by its own request count), at most ${RIVAULT_SIDECAR_PLANNED_RESTARTS}
# times per ${RIVAULT_SIDECAR_PLANNED_WINDOW_SEC}; beyond that it is charged,
# so a sidecar that exits 0 in a loop still ends the job.
#
# Dependency order (2026-10-23 review F2): podman-compose creates nginx with
# --requires on litellm (depends_on), and podman refuses to remove a container
# another one requires. stack_removal_order names the dependents first, so the
# caller removes nginx (and whatever requires nginx) before litellm, and
# recreates them in the reverse order.
#
# Tunables:
#   RIVAULT_ESSENTIAL_SIDECARS          services that must stay up ("litellm nginx")
#   RIVAULT_FRONT_PROBE                 0 disables signal 2 (signal 1 stays)
#   RIVAULT_FRONT_PROBE_PATH            probe path through nginx (/health/liveliness)
#   RIVAULT_FRONT_PROBE_EVERY_TICKS     probe the front every N monitor ticks (12)
#   RIVAULT_FRONT_FAIL_STREAK           failed probes that make a failure persistent (3)
#   RIVAULT_FRONT_STARTUP_GRACE_SEC     grace before the first good answer (900)
#   RIVAULT_FRONT_RESPAWN_GRACE_SEC     grace after a respawn (120)
#   RIVAULT_SIDECAR_PLANNED_RESTARTS    free planned restarts per window (6)
#   RIVAULT_SIDECAR_PLANNED_WINDOW_SEC  that window (3600)

_SH_FRONT_STREAK=0
_SH_FRONT_SEEN_OK=0
_SH_GRACE_UNTIL=0
_SH_TICK=0
_SH_PLANNED_AT=""
STACK_HEALTH_RESPAWN=""
STACK_HEALTH_REASON=""
STACK_HEALTH_PLANNED=0

_sh_now() { date +%s; }

# Job start: nothing seen yet, the start grace runs.
stack_health_init() {
    _SH_FRONT_STREAK=0
    _SH_FRONT_SEEN_OK=0
    _SH_TICK=0
    _SH_PLANNED_AT=""
    _SH_GRACE_UNTIL=$(( $(_sh_now) + ${RIVAULT_FRONT_STARTUP_GRACE_SEC:-900} ))
}

# After a respawn: the streak restarts and a short grace runs; whether the
# front ever answered is kept.
stack_health_respawned() {
    _SH_FRONT_STREAK=0
    _SH_GRACE_UNTIL=$(( $(_sh_now) + ${RIVAULT_FRONT_RESPAWN_GRACE_SEC:-120} ))
}

stack_sidecar_container() { printf '%s_%s_1' "$1" "$2"; }

# Print the essential sidecars whose container exists and is not running.
# A container that does not exist (its profile is off) is not checked.
stack_exited_sidecars() {
    local project="$1" svc state dead=""
    for svc in ${RIVAULT_ESSENTIAL_SIDECARS:-litellm nginx}; do
        state="$(podman container inspect "$(stack_sidecar_container "$project" "$svc")" \
            --format '{{.State.Status}}' 2>/dev/null)" || continue
        case "$state" in
            exited|stopped|dead) dead="${dead:+$dead }$svc" ;;
        esac
    done
    printf '%s' "$dead"
}

# rc 0 when the container ended by plan: exit code 0, or uvicorn's
# max-requests exit in the log tail.
stack_planned_exit() {
    local ctr="$1" code
    code="$(podman container inspect "$ctr" --format '{{.State.ExitCode}}' 2>/dev/null)"
    [ "$code" = "0" ] && return 0
    podman logs --tail 50 "$ctr" 2>&1 | grep -q "Maximum request limit"
}

# rc 0 when every named sidecar ended by plan.
stack_all_planned() {
    local project="$1" svc
    for svc in $2; do
        stack_planned_exit "$(stack_sidecar_container "$project" "$svc")" || return 1
    done
    return 0
}

# rc 0 (and the restart is taken) while the planned-restart window has room.
stack_planned_restart_take() {
    local now t kept="" n=0
    now="$(_sh_now)"
    for t in $_SH_PLANNED_AT; do
        if [ $(( now - t )) -lt "${RIVAULT_SIDECAR_PLANNED_WINDOW_SEC:-3600}" ]; then
            kept="${kept:+$kept }$t"
            n=$(( n + 1 ))
        fi
    done
    _SH_PLANNED_AT="$kept"
    [ "$n" -lt "${RIVAULT_SIDECAR_PLANNED_RESTARTS:-6}" ] || return 1
    _SH_PLANNED_AT="${kept:+$kept }$now"
}

# rc 0 when this respawn costs nothing: a planned exit with window room left.
stack_restart_is_free() {
    [ "${STACK_HEALTH_PLANNED:-0}" = "1" ] && stack_planned_restart_take
}

# Print the containers of this stack that require $2, deepest first.
stack_dependents() {
    local project="$1" target="$2" id c
    id="$(podman container inspect "$target" --format '{{.Id}}' 2>/dev/null)" || return 0
    [ -n "$id" ] || return 0
    for c in $(podman ps -a --filter "name=^${project}_" --format '{{.Names}}' 2>/dev/null); do
        [ "$c" = "$target" ] && continue
        case "$(podman container inspect "$c" --format '{{.Dependencies}}' 2>/dev/null)" in
            *"$id"*)
                stack_dependents "$project" "$c"
                printf '%s\n' "$c"
                ;;
        esac
    done
}

# The services to remove for a respawn of $2, dependents first; recreate them
# in the reverse order.
stack_removal_order() {
    local project="$1" order="" c s i
    local -a svcs
    read -r -a svcs <<< "$2"
    for (( i = ${#svcs[@]} - 1; i >= 0; i-- )); do
        c="$(stack_sidecar_container "$project" "${svcs[$i]}")"
        for c in $(stack_dependents "$project" "$c") "$c"; do
            s="${c#"${project}_"}"
            s="${s%_1}"
            case " $order " in
                *" $s "*) ;;
                *) order="${order:+$order }$s" ;;
            esac
        done
    done
    printf '%s' "$order"
}

# rc 0 when the front reached LiteLLM (any status but 502/503/504/no answer).
stack_front_answers() {
    local port="$1" code
    code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 \
        "http://127.0.0.1:${port}${RIVAULT_FRONT_PROBE_PATH:-/health/liveliness}" 2>/dev/null)"
    case "${code:-000}" in
        000|502|503|504) return 1 ;;
    esac
    return 0
}

# The probe runs only when it can mean something: enabled, a port to probe,
# and the LiteLLM container part of this stack.
stack_front_probe_enabled() {
    local project="$1" port="$2"
    [ "${RIVAULT_FRONT_PROBE:-1}" != "0" ] || return 1
    [ -n "$port" ] || return 1
    podman container exists "$(stack_sidecar_container "$project" litellm)" 2>/dev/null
}

# rc 0 on every RIVAULT_FRONT_PROBE_EVERY_TICKS-th call (2026-10-23 review F1:
# a probe every 5 s was 17,280 requests a day against a proxy that exits
# after MAX_REQUESTS_BEFORE_RESTART of them).
stack_front_probe_due() {
    _SH_TICK=$(( _SH_TICK + 1 ))
    [ "$_SH_TICK" -ge "${RIVAULT_FRONT_PROBE_EVERY_TICKS:-12}" ] || return 1
    _SH_TICK=0
}

# rc 0 when the front has failed persistently; updates the streak state.
stack_front_down() {
    local port="$1"
    if stack_front_answers "$port"; then
        _SH_FRONT_STREAK=0
        _SH_FRONT_SEEN_OK=1
        _SH_GRACE_UNTIL=0
        return 1
    fi
    _SH_FRONT_STREAK=$(( _SH_FRONT_STREAK + 1 ))
    [ "$(_sh_now)" -ge "$_SH_GRACE_UNTIL" ] || return 1
    [ "$_SH_FRONT_STREAK" -ge "${RIVAULT_FRONT_FAIL_STREAK:-3}" ]
}

# One monitor tick. rc 0 = respawn STACK_HEALTH_RESPAWN; rc 1 = healthy or
# still starting. Never call it in $(...): the streak state lives in this shell.
stack_health_tick() {
    local project="$1" port="${2:-}" dead
    STACK_HEALTH_RESPAWN=""
    STACK_HEALTH_REASON=""
    STACK_HEALTH_PLANNED=0
    dead="$(stack_exited_sidecars "$project")"
    if [ -n "$dead" ]; then
        STACK_HEALTH_RESPAWN="$dead"
        STACK_HEALTH_REASON="sidecar exited: ${dead}"
        if stack_all_planned "$project" "$dead"; then
            STACK_HEALTH_PLANNED=1
            STACK_HEALTH_REASON="${STACK_HEALTH_REASON} (planned exit)"
        fi
        stack_health_respawned
        return 0
    fi
    stack_front_probe_enabled "$project" "$port" || return 1
    stack_front_probe_due || return 1
    stack_front_down "$port" || return 1
    STACK_HEALTH_RESPAWN="${RIVAULT_ESSENTIAL_SIDECARS:-litellm nginx}"
    STACK_HEALTH_REASON="front 127.0.0.1:${port} failed ${_SH_FRONT_STREAK} probes in a row"
    stack_health_respawned
    return 0
}
