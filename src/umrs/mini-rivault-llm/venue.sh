#!/usr/bin/env bash
# =============================================================================
# venue.sh — mini-RiVault venue library (ADR-045: HPC venue portability)
# =============================================================================
# Sourced (never executed) by:
#   - serve-model.slurm                 the compute-node wrapper
#   - deploy/_common/harness_probe.sh   per-partition venue preflight
#   - deploy/_common/stage_images.sh    login-node image staging
#
# Every public function is prefixed `rivault_`, internal helpers `_rivault_`.
# Portable bash (3.2+): no mapfile, no associative arrays. Needs podman;
# squeue is optional (network GC degrades to "keep" without it).
#
# Each section fixes one defect class observed on a real HPC venue
# (RCCS Cloud, 2026-09-07, WP-01 round 3b):
#
#   D1  Job-private container configuration. $HOME is shared across nodes on
#       most HPC systems: writing THIS job's runroot into ~/.config/containers/
#       repointed every other running job of the same user, whose teardown then
#       failed and left its network and containers behind. Rootless podman also
#       (a) only honours a runroot under $XDG_RUNTIME_DIR and (b) keeps the
#       runroot recorded in its state database, so the database kept the first
#       job's dead $XDG_RUNTIME_DIR forever; the legacy shims "fixed" that by
#       deleting other jobs' runroot dirs, corrupting whatever was running.
#   D2  Private-network allocation from a pool + garbage collection of networks
#       whose Slurm job is gone. Three fixed candidate subnets were exhausted by
#       three crashed jobs, after which every job on the venue died in ~6 s with
#       "failed to create genai-private-net-<id> with any candidate subnet".
#   D3  Image staging. Compute nodes keep node-local image stores and often
#       have no registry route; a locally built image reaches them only through
#       an archive on the shared filesystem (podman save -> podman load).
#   D4  GPU override generated from what the host actually has (device nodes,
#       driver userspace libraries under the host's lib dir) instead of a file
#       hardcoded for one architecture.
# =============================================================================
# Output variables set for callers (RIVAULT_NET_*, RIVAULT_IMAGE_SOURCE, ...):
# shellcheck disable=SC2034

rivault_log() {
    printf '[rivault] %s\n' "$*"
}

rivault_venue_user() {
    id -un 2>/dev/null || whoami
}

# -----------------------------------------------------------------------------
# D1 — per-job container configuration
# -----------------------------------------------------------------------------

# rivault_venue_defaults JOB_TAG
#   Fills (never overwrites) the venue paths:
#     RIVAULT_VENUE_ROOT          node-local root       (/var/tmp/podman_<user>)
#     RIVAULT_STORAGE_ROOT        podman graphroot      (<root>/storage)
#     RIVAULT_NETWORK_CONFIG_DIR  network configs       (<root>/networks)
#     RIVAULT_RUNROOT             podman runroot        ($XDG_RUNTIME_DIR/containers — podman's
#                                                        rootless default; the only place it honours)
#     RIVAULT_JOB_CONF_DIR        this job's conf files ($XDG_RUNTIME_DIR/rivault-<tag>/containers)
#   Site overrides arrive as environment (site.yaml runtime.storage_root ->
#   RIVAULT_STORAGE_ROOT, runtime.network_config_dir -> RIVAULT_NETWORK_CONFIG_DIR).
rivault_venue_defaults() {
    local job_tag="${1:-$$}" user run_parent
    user="$(rivault_venue_user)"
    : "${RIVAULT_VENUE_ROOT:=/var/tmp/podman_${user}}"
    : "${RIVAULT_STORAGE_ROOT:=${RIVAULT_VENUE_ROOT}/storage}"
    : "${RIVAULT_NETWORK_CONFIG_DIR:=${RIVAULT_VENUE_ROOT}/networks}"
    # Slurm gives every job its own XDG_RUNTIME_DIR on most sites; without one,
    # give podman a node-local per-user dir instead of letting it guess.
    : "${XDG_RUNTIME_DIR:=/tmp/rivault-${user}/run}"
    : "${RIVAULT_RUNROOT:=${XDG_RUNTIME_DIR}/containers}"
    : "${RIVAULT_JOB_CONF_DIR:=${XDG_RUNTIME_DIR}/rivault-${job_tag}/containers}"
    export XDG_RUNTIME_DIR RIVAULT_VENUE_ROOT RIVAULT_STORAGE_ROOT RIVAULT_NETWORK_CONFIG_DIR
    export RIVAULT_RUNROOT RIVAULT_JOB_CONF_DIR
}

# rivault_venue_network_backend
#   "netavark" when the partition profile asks for the switch
#   (RIVAULT_PARTITION_NEEDS_NETAVARK=1) or RIVAULT_NETWORK_BACKEND is set;
#   empty otherwise (podman decides: its persisted choice, else autodetect).
rivault_venue_network_backend() {
    if [[ -n "${RIVAULT_NETWORK_BACKEND:-}" ]]; then
        printf '%s\n' "${RIVAULT_NETWORK_BACKEND}"
    elif [[ "${RIVAULT_PARTITION_NEEDS_NETAVARK:-0}" == "1" ]]; then
        printf 'netavark\n'
    fi
}

# rivault_venue_write_container_configs [CGROUP_MANAGER] [NETWORK_BACKEND]
#   Writes storage.conf and containers.conf into RIVAULT_JOB_CONF_DIR and
#   points CONTAINERS_STORAGE_CONF / CONTAINERS_CONF at them, so every podman
#   and podman-compose call of this job reads job-private configuration. The
#   per-user files under ~/.config/containers are neither read nor written.
rivault_venue_write_container_configs() {
    local cgroup_manager="${1:-cgroupfs}" network_backend="${2:-}" mount_program=""
    [[ -n "${RIVAULT_JOB_CONF_DIR:-}" ]] || rivault_venue_defaults
    mkdir -p "${RIVAULT_JOB_CONF_DIR}" "${RIVAULT_STORAGE_ROOT}" \
             "${RIVAULT_NETWORK_CONFIG_DIR}" "${RIVAULT_RUNROOT}" || return 1
    if [[ -x /usr/bin/fuse-overlayfs ]]; then
        mount_program=/usr/bin/fuse-overlayfs
    elif command -v fuse-overlayfs >/dev/null 2>&1; then
        mount_program="$(command -v fuse-overlayfs)"
    fi
    {
        echo '[storage]'
        echo 'driver = "overlay"'
        echo "runroot = \"${RIVAULT_RUNROOT}\""
        echo "graphroot = \"${RIVAULT_STORAGE_ROOT}\""
        echo "rootless_storage_path = \"${RIVAULT_STORAGE_ROOT}\""
        if [[ -n "${mount_program}" ]]; then
            echo '[storage.options]'
            echo "mount_program = \"${mount_program}\""
            echo '[storage.options.overlay]'
            echo 'mountopt = "xattr_permissions=2,threaded=1"'
        fi
    } > "${RIVAULT_JOB_CONF_DIR}/storage.conf"
    {
        echo '[engine]'
        echo "cgroup_manager = \"${cgroup_manager}\""
        echo '[containers]'
        echo 'label = false'
        echo '[network]'
        echo "network_config_dir = \"${RIVAULT_NETWORK_CONFIG_DIR}\""
        if [[ -n "${network_backend}" ]]; then
            echo "network_backend = \"${network_backend}\""
        fi
    } > "${RIVAULT_JOB_CONF_DIR}/containers.conf"
    export CONTAINERS_STORAGE_CONF="${RIVAULT_JOB_CONF_DIR}/storage.conf"
    export CONTAINERS_CONF="${RIVAULT_JOB_CONF_DIR}/containers.conf"
    if [[ -n "${network_backend}" ]]; then
        # Keep podman's persisted choice consistent with the configured one.
        printf '%s' "${network_backend}" > "${RIVAULT_STORAGE_ROOT}/defaultNetworkBackend" 2>/dev/null || true
    fi
}

# rivault_venue_adopt_runroot
#   Rootless podman silently keeps the runroot recorded in its state database
#   when storage.conf names another one (only an explicit --runroot fails with
#   "database run root ... does not match"). A database created by an earlier
#   job therefore pins the runroot to that job's dead $XDG_RUNTIME_DIR (podman
#   recreates the dead dir — which the legacy shims then deleted from under
#   running containers). When no container is running on this node, drop the
#   container-only libpod state so podman re-creates it against this job's
#   runroot; images and layers are untouched. With containers running, keep the
#   pinned runroot — it works, it is just an oddly named directory.
rivault_venue_adopt_runroot() {
    local current running entry
    current="$(podman info --format '{{.Store.RunRoot}}' 2>/dev/null || true)"
    if [[ -z "${current}" || "${current}" == "${RIVAULT_RUNROOT}" ]]; then
        return 0
    fi
    running="$(podman ps -q 2>/dev/null | head -1)"
    if [[ -n "${running}" ]]; then
        rivault_log "venue: keeping database-pinned runroot ${current} (containers are running on this node)"
        return 0
    fi
    rivault_log "venue: re-pinning runroot ${current} -> ${RIVAULT_RUNROOT} (no containers running; container state reset, images kept)"
    rm -rf "${RIVAULT_STORAGE_ROOT}/libpod" 2>/dev/null || true
    rm -f "${RIVAULT_STORAGE_ROOT}/db.sql" 2>/dev/null || true
    if [[ -d "${RIVAULT_STORAGE_ROOT}/overlay-containers" ]]; then
        printf '[]' > "${RIVAULT_STORAGE_ROOT}/overlay-containers/containers.json" 2>/dev/null || true
        for entry in "${RIVAULT_STORAGE_ROOT}"/overlay-containers/*/; do
            [[ -d "${entry}" ]] && rm -rf "${entry}" 2>/dev/null
        done
    fi
    current="$(podman info --format '{{.Store.RunRoot}}' 2>/dev/null || true)"
    if [[ "${current}" != "${RIVAULT_RUNROOT}" ]]; then
        rivault_log "venue: WARN runroot is still ${current:-?} after the reset"
    fi
}

# rivault_venue_prepare JOB_TAG [CGROUP_MANAGER]
#   Defaults + config files + runroot adoption + one summary line. The
#   wrapper's single entry point.
rivault_venue_prepare() {
    local job_tag="${1:-$$}" cgroup_manager="${2:-cgroupfs}" backend
    rivault_venue_defaults "${job_tag}"
    backend="$(rivault_venue_network_backend)"
    rivault_venue_write_container_configs "${cgroup_manager}" "${backend}" || return 1
    rivault_venue_adopt_runroot
    rivault_log "venue: graphroot=${RIVAULT_STORAGE_ROOT} runroot=${RIVAULT_RUNROOT}"
    rivault_log "venue: network configs=${RIVAULT_NETWORK_CONFIG_DIR} backend=${backend:-podman-default} conf=${RIVAULT_JOB_CONF_DIR}"
}

# -----------------------------------------------------------------------------
# D2 — private network: garbage collection + pool allocation
# -----------------------------------------------------------------------------

# _rivault_job_alive JOBID -> 0 running/pending, 1 gone, 2 cannot tell
_rivault_job_alive() {
    local jobid="$1" out
    command -v squeue >/dev/null 2>&1 || return 2
    if out="$(squeue -h -j "${jobid}" -o '%i' 2>/dev/null)" && [[ -n "${out}" ]]; then
        return 0
    fi
    # squeue fails for an unknown job AND for an unreachable controller —
    # a plain listing tells the two apart.
    squeue -h -o '%i' >/dev/null 2>&1 || return 2
    return 1
}

# _rivault_net_gc_verdict NAME SUFFIX -> prints "remove" or the reason to keep
_rivault_net_gc_verdict() {
    local name="$1" suffix="$2" attached
    if [[ "${suffix}" =~ ^[0-9]+$ ]]; then
        _rivault_job_alive "${suffix}"
        case $? in
            0) printf 'job %s still running\n' "${suffix}" ;;
            1) printf 'remove\n' ;;
            *) printf 'slurm unavailable, not touching job %s\n' "${suffix}" ;;
        esac
        return 0
    fi
    if [[ "${RIVAULT_NET_GC_WITHOUT_SLURM:-0}" != "1" ]]; then
        printf 'no job id in name (set RIVAULT_NET_GC_WITHOUT_SLURM=1 to reap)\n'
        return 0
    fi
    attached="$(podman ps -a -q --filter "network=${name}" 2>/dev/null | head -1)"
    if [[ -n "${attached}" ]]; then
        printf 'containers attached\n'
    else
        printf 'remove\n'
    fi
}

# rivault_net_gc PREFIX [KEEP_NAME]
#   Removes private networks named PREFIX<jobid> whose Slurm job no longer
#   exists (leftover containers attached to them go too). Networks of running
#   jobs, KEEP_NAME and names without a job id are kept. With
#   RIVAULT_NET_GC_DRY_RUN=1 it only reports. Sets RIVAULT_NET_GC_REMOVED and
#   RIVAULT_NET_GC_KEPT (counts).
rivault_net_gc() {
    local prefix="$1" keep="${2:-}" name suffix verdict
    RIVAULT_NET_GC_REMOVED=0
    RIVAULT_NET_GC_KEPT=0
    while IFS= read -r name; do
        [[ -n "${name}" ]] || continue
        [[ "${name}" == "${prefix}"* ]] || continue
        [[ "${name}" != "${keep}" ]] || continue
        suffix="${name#"${prefix}"}"
        verdict="$(_rivault_net_gc_verdict "${name}" "${suffix}")"
        if [[ "${verdict}" != "remove" ]]; then
            RIVAULT_NET_GC_KEPT=$((RIVAULT_NET_GC_KEPT + 1))
            rivault_log "net-gc: keeping ${name} (${verdict})"
            continue
        fi
        if [[ "${RIVAULT_NET_GC_DRY_RUN:-0}" == "1" ]]; then
            rivault_log "net-gc: would remove ${name} (job ${suffix} gone)"
            RIVAULT_NET_GC_REMOVED=$((RIVAULT_NET_GC_REMOVED + 1))
        elif _rivault_net_remove "${name}"; then
            rivault_log "net-gc: removed ${name} (job ${suffix} gone)"
            RIVAULT_NET_GC_REMOVED=$((RIVAULT_NET_GC_REMOVED + 1))
        else
            RIVAULT_NET_GC_KEPT=$((RIVAULT_NET_GC_KEPT + 1))
            rivault_log "net-gc: WARN could not remove ${name}"
        fi
    done < <(podman network ls --format '{{.Name}}' 2>/dev/null)
}

# _rivault_net_remove NAME
#   `podman network rm -f` first; when leftover containers of a dead job block
#   it (their run state is gone, so the implicit container removal fails),
#   force-remove those containers one by one and retry.
_rivault_net_remove() {
    local name="$1" cid pass
    if podman network rm -f "${name}" >/dev/null 2>&1; then
        return 0
    fi
    # Two passes: a dependent container (nginx -> litellm) blocks the first
    # removal of the one it depends on.
    for pass in 1 2; do
        for cid in $(podman ps -a -q --filter "network=${name}" 2>/dev/null); do
            podman rm -f "${cid}" >/dev/null 2>&1 || true
        done
        [[ "${pass}" -eq 1 ]] || break
    done
    podman network rm -f "${name}" >/dev/null 2>&1
}

_rivault_seed_number() {
    local seed="$1"
    if [[ "${seed}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${seed}"
    else
        printf '%s' "${seed}" | cksum | awk '{print $1}'
    fi
}

# rivault_net_candidates POOL SEED [COUNT] [PREFERRED]
#   Prints candidate "a.b.c" prefixes (one per line): PREFERRED first when
#   given, then COUNT third octets walked from a SEED-derived start inside
#   POOL. POOL is a.b.0.0/16 (walk 1..254; different jobs start at different
#   offsets) or a.b.c.0/24 (one fixed candidate).
rivault_net_candidates() {
    local pool="$1" seed="$2" count="${3:-16}" preferred="${4:-}"
    local a b c mask start i third cand seen=" "
    if [[ "${pool}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)\.[0-9]+/([0-9]+)$ ]]; then
        a="${BASH_REMATCH[1]}"; b="${BASH_REMATCH[2]}"; c="${BASH_REMATCH[3]}"; mask="${BASH_REMATCH[4]}"
    else
        rivault_log "ERROR: private subnet pool '${pool}' must look like a.b.0.0/16 or a.b.c.0/24"
        return 1
    fi
    if [[ -n "${preferred}" ]]; then
        printf '%s\n' "${preferred}"
        seen=" ${preferred} "
    fi
    case "${mask}" in
        24)
            cand="${a}.${b}.${c}"
            [[ "${seen}" == *" ${cand} "* ]] || printf '%s\n' "${cand}"
            ;;
        16)
            start=$(( $(_rivault_seed_number "${seed}") % 254 + 1 ))
            for ((i = 0; i < count; i++)); do
                third=$(( (start - 1 + i) % 254 + 1 ))
                cand="${a}.${b}.${third}"
                [[ "${seen}" == *" ${cand} "* ]] && continue
                printf '%s\n' "${cand}"
                seen="${seen}${cand} "
            done
            ;;
        *)
            rivault_log "ERROR: private subnet pool '${pool}': only /16 pools or a single /24 are supported"
            return 1
            ;;
    esac
}

rivault_net_diagnostics() {
    rivault_log "diagnostics:"
    rivault_log "  podman: $(podman --version 2>/dev/null || echo 'not found')"
    rivault_log "  $(podman info --format 'backend={{.Host.NetworkBackend}} graphroot={{.Store.GraphRoot}} runroot={{.Store.RunRoot}}' 2>&1 | head -1)"
    rivault_log "  containers.conf=${CONTAINERS_CONF:-<default>} storage.conf=${CONTAINERS_STORAGE_CONF:-<default>}"
    rivault_log "  network config dir: ${RIVAULT_NETWORK_CONFIG_DIR:-<podman default>}"
    rivault_log "  networks:"
    podman network ls 2>&1 | sed 's/^/[rivault]    /'
    if command -v ip >/dev/null 2>&1; then
        rivault_log "  host IPv4 addresses:"
        ip -4 -o addr 2>/dev/null | awk '{print "[rivault]    " $2 " " $4}'
    fi
}

# rivault_net_create NAME [POOL] [SEED] [PREFERRED] [MAX_ATTEMPTS]
#   Creates the job's private /24 with static addressing (--disable-dns; the
#   compose file assigns fixed addresses below PRIV_IP1TO3). Tries PREFERRED,
#   then MAX_ATTEMPTS pool candidates, then lets podman pick a free /24 and
#   reads it back. Every attempt's error is kept and printed on failure.
#   On success sets RIVAULT_NET_IP1TO3, RIVAULT_NET_SUBNET, RIVAULT_NET_MODE
#   (preferred|pool|auto).
rivault_net_create() {
    local name="$1" pool="${2:-172.16.0.0/16}" seed="${3:-$$}" preferred="${4:-}" max="${5:-16}"
    local cand out subnet attempts=0 e
    local -a errors=()
    RIVAULT_NET_IP1TO3=""; RIVAULT_NET_SUBNET=""; RIVAULT_NET_MODE=""
    while IFS= read -r cand; do
        [[ -n "${cand}" ]] || continue
        attempts=$((attempts + 1))
        if out="$(podman network create --disable-dns --subnet "${cand}.0/24" "${name}" 2>&1)"; then
            RIVAULT_NET_IP1TO3="${cand}"
            RIVAULT_NET_SUBNET="${cand}.0/24"
            if [[ "${cand}" == "${preferred}" ]]; then
                RIVAULT_NET_MODE="preferred"
            else
                RIVAULT_NET_MODE="pool"
            fi
            rivault_log "${name} created: subnet ${RIVAULT_NET_SUBNET} (${RIVAULT_NET_MODE}, attempt ${attempts})"
            return 0
        fi
        errors[${#errors[@]}]="${cand}.0/24: $(printf '%s' "${out}" | tr '\n' ' ')"
    done < <(rivault_net_candidates "${pool}" "${seed}" "${max}" "${preferred}")

    # Last resort: podman picks a free /24 (it checks its own configs and the
    # host's routes); read the allocation back for the static addresses.
    attempts=$((attempts + 1))
    if out="$(podman network create --disable-dns "${name}" 2>&1)"; then
        subnet="$(podman network inspect "${name}" --format '{{range .Subnets}}{{.Subnet}} {{end}}' 2>/dev/null | awk '{print $1}')"
        if [[ "${subnet}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\.0/24$ ]]; then
            RIVAULT_NET_IP1TO3="${BASH_REMATCH[1]}"
            RIVAULT_NET_SUBNET="${subnet}"
            RIVAULT_NET_MODE="auto"
            rivault_log "${name} created: subnet ${subnet} (auto-allocated by podman, attempt ${attempts})"
            return 0
        fi
        errors[${#errors[@]}]="auto: podman allocated '${subnet:-?}', not a /24"
        podman network rm -f "${name}" >/dev/null 2>&1 || true
    else
        errors[${#errors[@]}]="auto: $(printf '%s' "${out}" | tr '\n' ' ')"
    fi

    rivault_log "ERROR: failed to create ${name} after ${attempts} attempts (pool ${pool}, seed ${seed}):"
    if [[ ${#errors[@]} -gt 0 ]]; then
        for e in "${errors[@]}"; do
            rivault_log "  - ${e}"
        done
    fi
    rivault_net_diagnostics
    return 1
}

# -----------------------------------------------------------------------------
# D3 — image staging through the shared filesystem
# -----------------------------------------------------------------------------

# rivault_arch_suffix [MACHINE] -> arm64 | amd64 | <machine>
rivault_arch_suffix() {
    local machine="${1:-$(uname -m)}"
    case "${machine}" in
        aarch64|arm64) printf 'arm64\n' ;;
        x86_64|amd64)  printf 'amd64\n' ;;
        *)             printf '%s\n' "${machine}" ;;
    esac
}

# rivault_image_archive_path DIR REF [ARCH]
#   "<DIR>/<name>-<tag>-<arch>.tar" from the last path component of REF:
#   localhost/r4-mcp:latest -> <DIR>/r4-mcp-latest-arm64.tar
rivault_image_archive_path() {
    local dir="$1" ref="$2" arch="${3:-$(rivault_arch_suffix)}" leaf name tag
    leaf="${ref##*/}"
    leaf="${leaf%%@*}"
    if [[ "${leaf}" == *:* ]]; then
        name="${leaf%%:*}"; tag="${leaf#*:}"
    else
        name="${leaf}"; tag="latest"
    fi
    printf '%s/%s-%s-%s.tar\n' "${dir%/}" "${name}" "${tag}" "${arch}"
}

# rivault_ensure_image REF [ARCHIVE_DIR] [ARCHIVE_PATH]
#   Returns 0 when REF is usable on this node: already present, loaded from
#   the archive (ARCHIVE_PATH, or the conventional name under ARCHIVE_DIR), or
#   pulled. Returns 1 with one line naming what was looked for. Sets
#   RIVAULT_IMAGE_SOURCE (present|archive|pull).
rivault_ensure_image() {
    local ref="$1" dir="${2:-}" archive="${3:-}" loaded loaded_ref
    RIVAULT_IMAGE_SOURCE=""
    if podman image exists "${ref}" 2>/dev/null; then
        RIVAULT_IMAGE_SOURCE="present"
        return 0
    fi
    if [[ -z "${archive}" && -n "${dir}" ]]; then
        archive="$(rivault_image_archive_path "${dir}" "${ref}")"
    fi
    if [[ -n "${archive}" && -f "${archive}" ]]; then
        rivault_log "image ${ref}: loading archive ${archive}"
        if loaded="$(podman load -i "${archive}" 2>&1)"; then
            if ! podman image exists "${ref}" 2>/dev/null; then
                # The archive may carry the image under another name — retag.
                loaded_ref="$(printf '%s\n' "${loaded}" | sed -n -e 's/^Loaded image: *//p' -e 's/^Loaded image(s): *//p' | head -1)"
                if [[ -n "${loaded_ref}" ]]; then
                    podman tag "${loaded_ref}" "${ref}" >/dev/null 2>&1 || true
                fi
            fi
            if podman image exists "${ref}" 2>/dev/null; then
                RIVAULT_IMAGE_SOURCE="archive"
                rivault_log "image ${ref}: loaded from archive"
                return 0
            fi
            rivault_log "WARN: archive ${archive} loaded but ${ref} is still absent ($(printf '%s' "${loaded}" | tr '\n' ' '))"
        else
            rivault_log "WARN: podman load ${archive} failed: $(printf '%s' "${loaded}" | tr '\n' ' ')"
        fi
    fi
    if podman pull "${ref}" >/dev/null 2>&1; then
        RIVAULT_IMAGE_SOURCE="pull"
        rivault_log "image ${ref}: pulled"
        return 0
    fi
    if [[ -n "${archive}" ]]; then
        rivault_log "image ${ref}: not present on this node, no archive at ${archive}, and pull failed"
    else
        rivault_log "image ${ref}: not present on this node, no archive dir declared (site.yaml runtime.image_archive_dir), and pull failed"
    fi
    return 1
}

# -----------------------------------------------------------------------------
# D4 — GPU override generated from the host
# -----------------------------------------------------------------------------

# rivault_write_gpu_override PATH [DEV_DIR] [LIB_DIRS]
#   Writes a compose override that maps every NVIDIA device node found under
#   DEV_DIR (default /dev) and binds the driver userspace libraries found in
#   LIB_DIRS (colon-separated; default: the host's multiarch dir and
#   /usr/lib64) into the vllm and bentoml-vision services. Returns 1 when no
#   /dev/nvidia<N> exists so the caller can keep its embedded default.
rivault_write_gpu_override() {
    local path="$1" dev_dir="${2:-/dev}" lib_dirs="${3:-}" dev rel lib lib_dir found_gpu=0
    local devices="" volumes=""
    if [[ -z "${lib_dirs}" ]]; then
        lib_dirs="/usr/lib/$(uname -m)-linux-gnu:/usr/lib64"
    fi
    for dev in "${dev_dir}"/nvidia[0-9]* "${dev_dir}"/nvidiactl "${dev_dir}"/nvidia-uvm \
               "${dev_dir}"/nvidia-uvm-tools "${dev_dir}"/nvidia-caps/nvidia-cap1 \
               "${dev_dir}"/nvidia-caps/nvidia-cap2; do
        [[ -e "${dev}" ]] || continue
        case "${dev}" in "${dev_dir}"/nvidia[0-9]*) found_gpu=1 ;; esac
        rel="/dev${dev#"${dev_dir}"}"   # DEV_DIR is only a probe root; containers see /dev/...
        devices="${devices}      - ${rel}:${rel}:rwm"$'\n'
    done
    [[ "${found_gpu}" -eq 1 ]] || return 1
    # libcuda / libnvidia-ml are required by every CUDA container; the PTX JIT
    # compiler is required by images that ship no SASS for the local GPU.
    for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
        local IFS=':'
        for lib_dir in ${lib_dirs}; do
            [[ -e "${lib_dir}/${lib}" ]] || continue
            volumes="${volumes}      - ${lib_dir}/${lib}:${lib_dir}/${lib}:ro"$'\n'
            if [[ "${lib}" == "libnvidia-ptxjitcompiler.so.1" ]]; then
                volumes="${volumes}      - ${lib_dir}/${lib}:${lib_dir}/${lib%.1}:ro"$'\n'
            fi
            break
        done
    done
    {
        echo 'services:'
        echo '  vllm:'
        # shellcheck disable=SC2016  # compose interpolates this at `up`, not here
        echo '    shm_size: ${VLLM_NCCL_MEM_LIMIT:?error}'
        echo '    devices:'
        printf '%s' "${devices}"
        if [[ -n "${volumes}" ]]; then
            echo '    volumes:'
            printf '%s' "${volumes}"
        fi
        echo '    environment:'
        echo '      - NVIDIA_VISIBLE_DEVICES=all'
        echo '      - NVIDIA_DRIVER_CAPABILITIES=compute,utility'
        echo ''
        echo '  bentoml-vision:'
        echo '    devices:'
        printf '%s' "${devices}"
        if [[ -n "${volumes}" ]]; then
            echo '    volumes:'
            printf '%s' "${volumes}"
        fi
        echo '    environment:'
        echo '      - NVIDIA_VISIBLE_DEVICES=all'
        echo '      - NVIDIA_DRIVER_CAPABILITIES=compute,utility'
    } > "${path}"
}
