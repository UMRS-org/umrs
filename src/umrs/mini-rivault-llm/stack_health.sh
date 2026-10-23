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
#           ...claim one respawn from the per-job budget, recreate them...
#       fi
#   done
#
# The caller claims each respawn from the same per-job budget as an engine
# respawn (attempt_budget.sh, ADR-048), so a sidecar that keeps dying uses up
# the budget and the job exits 125 instead of serving 502 forever.
#
# Two signals, checked once per monitor tick:
#   1. a sidecar container is exited/stopped/dead -> respawn that sidecar now;
#   2. the published front answers 502/503/504 or not at all on
#      ${RIVAULT_FRONT_PROBE_PATH} for ${RIVAULT_FRONT_FAIL_STREAK} ticks in a
#      row -> respawn every essential sidecar. Before the front has answered
#      once, failures count only after ${RIVAULT_FRONT_STARTUP_GRACE_SEC}.
#      Any other HTTP status (200, 401, 404 ...) means nginx reached LiteLLM.
#
# Tunables:
#   RIVAULT_ESSENTIAL_SIDECARS       services that must stay up ("litellm nginx")
#   RIVAULT_FRONT_PROBE              0 disables signal 2 (signal 1 stays)
#   RIVAULT_FRONT_PROBE_PATH         probe path through nginx (/health/liveliness)
#   RIVAULT_FRONT_FAIL_STREAK        failed ticks that make a failure persistent (24)
#   RIVAULT_FRONT_STARTUP_GRACE_SEC  grace before the first good answer (900)

_SH_FRONT_STREAK=0
_SH_FRONT_SEEN_OK=0
_SH_STARTED_AT=0
STACK_HEALTH_RESPAWN=""
STACK_HEALTH_REASON=""

stack_health_init() {
    _SH_FRONT_STREAK=0
    _SH_FRONT_SEEN_OK=0
    _SH_STARTED_AT="$(date +%s)"
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

# rc 0 when the front has failed persistently; updates the streak state.
stack_front_down() {
    local port="$1" now
    if stack_front_answers "$port"; then
        _SH_FRONT_STREAK=0
        _SH_FRONT_SEEN_OK=1
        return 1
    fi
    _SH_FRONT_STREAK=$(( _SH_FRONT_STREAK + 1 ))
    now="$(date +%s)"
    if [ "$_SH_FRONT_SEEN_OK" -eq 0 ] \
        && [ $(( now - _SH_STARTED_AT )) -lt "${RIVAULT_FRONT_STARTUP_GRACE_SEC:-900}" ]; then
        return 1
    fi
    [ "$_SH_FRONT_STREAK" -ge "${RIVAULT_FRONT_FAIL_STREAK:-24}" ]
}

# One monitor tick. rc 0 = respawn STACK_HEALTH_RESPAWN; rc 1 = healthy or
# still starting. Never call it in $(...): the streak state lives in this shell.
stack_health_tick() {
    local project="$1" port="${2:-}" dead
    STACK_HEALTH_RESPAWN=""
    STACK_HEALTH_REASON=""
    dead="$(stack_exited_sidecars "$project")"
    if [ -n "$dead" ]; then
        STACK_HEALTH_RESPAWN="$dead"
        STACK_HEALTH_REASON="sidecar exited: ${dead}"
        stack_health_init
        return 0
    fi
    stack_front_probe_enabled "$project" "$port" || return 1
    stack_front_down "$port" || return 1
    STACK_HEALTH_RESPAWN="${RIVAULT_ESSENTIAL_SIDECARS:-litellm nginx}"
    STACK_HEALTH_REASON="front 127.0.0.1:${port} failed ${_SH_FRONT_STREAK} probes in a row"
    stack_health_init
    return 0
}
