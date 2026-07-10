#!/usr/bin/env bash
# =============================================================================
# setup_deepseek_dspark.sh — stage the DSpark DeepSeek-V4-Flash prerequisites
# =============================================================================
# Prepares the three out-of-band prerequisites that `r4 pool up` (the DSpark
# entry in pool-rccs-3models.yaml) CANNOT do by itself:
#
#   1. IMAGE   — build the DSpark Stage-C NVFP4 runtime image and make it
#                available on BOTH GB10 nodes of the target partition. The
#                proposer concurrency patch is baked in by the repo's own
#                overlay build (Stage-C), so no runtime bind-mount is needed.
#   2. WEIGHTS — download fraserprice/DeepSeek-V4-Flash-DSpark into the shared
#                Lustre model root that the serve job bind-mounts.
#   3. CATALOG — register the staged weights in the rccs UMRS catalog under the
#                name the roster's `model:` field expects.
#
# Run this ON rccs-cloud (login1.cloud.r-ccs.riken.jp), from this directory,
# BEFORE `r4 pool up umrs/deploy/rccs/pool-rccs-3models.yaml`.
#
# The image build is a heavy, operator-supervised step: it drives the DSpark
# recipe repo's authoritative build script through a docker->podman shim (the
# recipe is docker-based; rccs-cloud runs podman) and then distributes the
# resulting image to each compute node over the shared filesystem. Nothing here
# starts a service — endpoint bring-up stays with `r4 pool up`.
#
# Usage:
#   ./setup_deepseek_dspark.sh --all              # image + weights + register
#   ./setup_deepseek_dspark.sh --weights          # just download the weights
#   ./setup_deepseek_dspark.sh --image            # just build+distribute image
#   ./setup_deepseek_dspark.sh --register         # just register in catalog
#   ./setup_deepseek_dspark.sh --all --dry-run    # preview every command
#   ./setup_deepseek_dspark.sh --verify           # check all 3 prereqs are met
#
# Key overrides (env vars):
#   MODEL_ROOT     shared Lustre model root (default matches the roster)
#   DSPARK_NODES   space-separated GB10 nodes to hold the image
#   PARTITION      Slurm partition for the (srun) build/load steps
#   MODEL_SCOPE    r4 model scope for registration (default: user)
#   HF_TOKEN       HuggingFace token (only if the repo is gated)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ─── Tunables (env-overridable; defaults mirror pool-rccs-3models.yaml) ───────
MODEL_ROOT="${MODEL_ROOT:-/lvs0/rccs-dmp/${USER}/umrs_models}"
CATALOG_NAME="${CATALOG_NAME:-deepseek-v4-flash-dspark}"
HF_MODEL="${HF_MODEL:-fraserprice/DeepSeek-V4-Flash-DSpark}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/DeepSeek-V4-Flash-DSpark}"
DSPARK_IMAGE="${DSPARK_IMAGE:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
DSPARK_REPO_URL="${DSPARK_REPO_URL:-https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark}"
DSPARK_REPO_DIR="${DSPARK_REPO_DIR:-${MODEL_ROOT}/_src/dspark-recipe}"
# The two GB10 nodes of the roster's ng-dgx-m3 partition (see site.yaml:
# ng-dgx-m3-[14-15]). Both must hold the image for the 2-node TP=2 launch.
DSPARK_NODES="${DSPARK_NODES:-ng-dgx-m3-14 ng-dgx-m3-15}"
PARTITION="${PARTITION:-ng-dgx-m3}"
BUILD_TIME="${BUILD_TIME:-4:00:00}"
IMAGE_TAR="${IMAGE_TAR:-${MODEL_ROOT}/_images/dspark-stagec.tar}"
# Registration goes through the r4 gateway client (the same tool `r4 pool up`
# uses); the active r4 source must point at the rccs gateway. scope=user matches
# the other rccs models (e.g. Qwen3.6-35B-A3B-NVFP4).
MODEL_SCOPE="${MODEL_SCOPE:-user}"

DRY_RUN=false
DO_IMAGE=false
DO_WEIGHTS=false
DO_REGISTER=false
DO_VERIFY=false

usage() {
    sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[dspark-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dspark-setup] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[dspark-setup] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

# run — echo (and, unless --dry-run, execute) a command.
run() {
    printf '  \033[2m$ %s\033[0m\n' "$*"
    if [[ "${DRY_RUN}" == true ]]; then
        return 0
    fi
    "$@"
}

# run_sh — same as run() but for a shell pipeline passed as a single string.
run_sh() {
    printf '  \033[2m$ %s\033[0m\n' "$1"
    if [[ "${DRY_RUN}" == true ]]; then
        return 0
    fi
    bash -c "$1"
}

need() {
    command -v "$1" >/dev/null 2>&1 && return 0
    if [[ "${DRY_RUN}" == true ]]; then
        warn "required tool not found (ignored in --dry-run): $1"
        return 0
    fi
    die "required tool not found: $1"
}

# On a compute node the container runtime is podman; the DSpark recipe scripts
# call `docker`. Provide a throwaway PATH shim so their build script transparently
# drives podman without editing the vendored scripts.
make_podman_shim() {
    local shim_dir="$1"
    mkdir -p "${shim_dir}"
    cat > "${shim_dir}/docker" <<'SHIM'
#!/usr/bin/env bash
# docker -> podman shim (DSpark recipe compatibility on rccs-cloud)
exec podman "$@"
SHIM
    chmod +x "${shim_dir}/docker"
}

# ─── Step 1: build + distribute the Stage-C image ────────────────────────────
build_image() {
    log "STEP image: build DSpark Stage-C image '${DSPARK_IMAGE}' and place it on: ${DSPARK_NODES}"
    need git
    need srun

    # 1a. Fetch the authoritative recipe repo.
    if [[ -d "${DSPARK_REPO_DIR}/.git" ]]; then
        run git -C "${DSPARK_REPO_DIR}" fetch --depth 1 origin
        run git -C "${DSPARK_REPO_DIR}" reset --hard origin/HEAD
    else
        run mkdir -p "$(dirname "${DSPARK_REPO_DIR}")"
        run git clone --depth 1 "${DSPARK_REPO_URL}" "${DSPARK_REPO_DIR}"
    fi

    # 1b. Build on the FIRST node (has GPU + CUDA toolkit + the podman store),
    #     driving the repo's own build script through the docker->podman shim.
    local first_node
    first_node="$(awk '{print $1}' <<<"${DSPARK_NODES}")"
    local shim_dir="${DSPARK_REPO_DIR}/.podman-shim"
    if [[ "${DRY_RUN}" != true ]]; then
        make_podman_shim "${shim_dir}"
    else
        printf '  \033[2m$ make_podman_shim %s\033[0m\n' "${shim_dir}"
    fi

    log "Building on ${first_node} (this compiles the NVFP4/DSpark overlay — expect a long run)."
    # WORKER_BUILD=0: the recipe's own build script would otherwise ssh+rsync to
    # a second WORKER_HOST and build there too (needs .env.dspark + inter-node
    # ssh we do not configure). We build once here and distribute the image to
    # peers via podman save/load (below), so disable the recipe's worker build.
    run srun --partition="${PARTITION}" --nodelist="${first_node}" --nodes=1 \
        --ntasks=1 --time="${BUILD_TIME}" \
        env PATH="${shim_dir}:${PATH}" DOCKER=podman CONTAINER_ENGINE=podman WORKER_BUILD=0 \
        bash -c "cd '${DSPARK_REPO_DIR}' && ./build-dspark-vllm-runtime.sh"

    # 1c. Verify the tag exists on the first node, then distribute to the rest
    #     via `podman save` to shared Lustre + `podman load` on each peer.
    run srun --partition="${PARTITION}" --nodelist="${first_node}" --nodes=1 --ntasks=1 \
        podman image exists "${DSPARK_IMAGE}" \
        || die "image ${DSPARK_IMAGE} not present on ${first_node} after build"

    local peers=""
    local m
    for m in ${DSPARK_NODES}; do
        [[ "${m}" == "${first_node}" ]] && continue
        peers="${peers} ${m}"
    done
    peers="$(xargs <<<"${peers}" || true)"
    if [[ -z "${peers}" ]]; then
        log "Single build node; nothing to distribute."
        return 0
    fi

    run mkdir -p "$(dirname "${IMAGE_TAR}")"
    log "Saving image to shared filesystem: ${IMAGE_TAR}"
    run srun --partition="${PARTITION}" --nodelist="${first_node}" --nodes=1 --ntasks=1 \
        podman save -o "${IMAGE_TAR}" "${DSPARK_IMAGE}"

    local n
    for n in ${peers}; do
        log "Loading image on ${n}"
        run srun --partition="${PARTITION}" --nodelist="${n}" --nodes=1 --ntasks=1 \
            podman load -i "${IMAGE_TAR}"
    done
    log "Image present on all target nodes."
}

# ─── Step 2: download the weights ────────────────────────────────────────────
stage_weights() {
    log "STEP weights: download ${HF_MODEL} -> ${MODEL_DIR}"
    # Prefer the modern `hf` CLI; fall back to huggingface-cli.
    local hf_bin=""
    if command -v hf >/dev/null 2>&1; then
        hf_bin="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        hf_bin="huggingface-cli"
    elif [[ "${DRY_RUN}" == true ]]; then
        hf_bin="hf"   # preview only; not actually invoked
        warn "no hf CLI found (ignored in --dry-run); previewing with 'hf'"
    else
        die "neither 'hf' nor 'huggingface-cli' found (pip install -U huggingface_hub)"
    fi

    run mkdir -p "${MODEL_DIR}"

    # Auth: DO NOT pass --token on the command line (it would be echoed into
    # logs). The hf/huggingface CLIs read HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)
    # straight from the environment, so a gated repo just needs HF_TOKEN set in
    # the shell that runs this script.
    if [[ -n "${HF_TOKEN:-}" ]]; then
        log "using HF_TOKEN from environment (not printed)"
        export HF_TOKEN
    fi

    # The login node has egress; compute nodes do not — staging happens here.
    if [[ "${hf_bin}" == "hf" ]]; then
        run "${hf_bin}" download "${HF_MODEL}" --local-dir "${MODEL_DIR}"
    else
        run "${hf_bin}" download "${HF_MODEL}" --local-dir "${MODEL_DIR}" \
            --local-dir-use-symlinks False
    fi

    if [[ "${DRY_RUN}" != true ]]; then
        [[ -f "${MODEL_DIR}/config.json" ]] \
            || warn "no config.json under ${MODEL_DIR} — verify the download completed"
    fi
    log "Weights staged at ${MODEL_DIR}"
}

# ─── Step 3: register in the UMRS catalog ────────────────────────────────────
register_model() {
    log "STEP register: '${CATALOG_NAME}' -> ${MODEL_DIR} (scope ${MODEL_SCOPE})"
    need r4
    if [[ "${DRY_RUN}" != true && ! -d "${MODEL_DIR}" ]]; then
        die "model dir ${MODEL_DIR} missing — run --weights first"
    fi
    # Registration is a gateway operation via r4 (the active r4 source must
    # point at the rccs gateway; the same source `r4 pool up` uses). The
    # --path is resolved on the UMRS server, so it must be the server-visible
    # Lustre path.
    run r4 model register "${CATALOG_NAME}" --path "${MODEL_DIR}" \
        --type llm --scope "${MODEL_SCOPE}"
    log "Registered. The roster 'model: ${CATALOG_NAME}' will now resolve."
}

# ─── Verify ──────────────────────────────────────────────────────────────────
verify() {
    log "VERIFY prerequisites for ${CATALOG_NAME}"
    local ok=true

    # Weights
    if [[ -f "${MODEL_DIR}/config.json" ]]; then
        log "  [ok] weights: ${MODEL_DIR}/config.json present"
    else
        warn "  [MISS] weights: ${MODEL_DIR}/config.json not found"; ok=false
    fi

    # Catalog
    if command -v r4 >/dev/null 2>&1; then
        if r4 model info "${CATALOG_NAME}" >/dev/null 2>&1; then
            log "  [ok] catalog: ${CATALOG_NAME} registered"
        else
            warn "  [MISS] catalog: ${CATALOG_NAME} not found via 'r4 model info'"; ok=false
        fi
    else
        warn "  [skip] catalog: r4 not on PATH"
    fi

    # Image on each node
    local n
    for n in ${DSPARK_NODES}; do
        if srun --partition="${PARTITION}" --nodelist="${n}" --nodes=1 --ntasks=1 \
             --time=00:02:00 podman image exists "${DSPARK_IMAGE}" 2>/dev/null; then
            log "  [ok] image: ${DSPARK_IMAGE} on ${n}"
        else
            warn "  [MISS] image: ${DSPARK_IMAGE} absent on ${n}"; ok=false
        fi
    done

    if [[ "${ok}" == true ]]; then
        log "All prerequisites satisfied. Bring up with:"
        log "  r4 pool up ${REPO_ROOT}/deploy/rccs/pool-rccs-3models.yaml"
    else
        die "one or more prerequisites are missing (see WARN lines above)"
    fi
}

# ─── Arg parsing ─────────────────────────────────────────────────────────────
[[ $# -eq 0 ]] && usage 1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)      DO_IMAGE=true; DO_WEIGHTS=true; DO_REGISTER=true ;;
        --image)    DO_IMAGE=true ;;
        --weights)  DO_WEIGHTS=true ;;
        --register) DO_REGISTER=true ;;
        --verify)   DO_VERIFY=true ;;
        --dry-run)  DRY_RUN=true ;;
        -h|--help)  usage 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

log "config: MODEL_DIR=${MODEL_DIR}"
log "config: IMAGE=${DSPARK_IMAGE} NODES='${DSPARK_NODES}' PARTITION=${PARTITION}"
[[ "${DRY_RUN}" == true ]] && log "DRY-RUN: commands are printed, not executed."

# Weights first (register depends on them); image is independent.
[[ "${DO_WEIGHTS}"  == true ]] && stage_weights
[[ "${DO_IMAGE}"    == true ]] && build_image
[[ "${DO_REGISTER}" == true ]] && register_model
[[ "${DO_VERIFY}"   == true ]] && verify

log "Done."
