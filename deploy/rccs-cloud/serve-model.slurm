#!/bin/bash
# =============================================================================
# serve-model.slurm — rccs-cloud (login1.cloud.r-ccs.riken.jp) wrapper
# =============================================================================
# Submitted by the umrs-server container via sbatch.
# Outputs the FQDN header that rivault_launcher.extract_node_address() needs,
# sets RIVAULT_DIR to the shared Lustre path, then delegates to the main
# mini-rivault-llm/serve-model.slurm.
#
# Partition / resource tuning:
#   Edit #SBATCH directives below to match the desired compute partition.
#   Current fallback targets the first active ng-dgx-m* partition while ng-dgx-s is unavailable.
# =============================================================================

#SBATCH --job-name=umrs-rivault
#SBATCH --partition=ng-dgx-m0
#SBATCH --nodes=1
#SBATCH --time=4:00:00

# ─── Output FQDN header (required by rivault_launcher.extract_node_address) ───
echo "=============================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NODELIST"
echo "Running on node: $(hostname)"
echo "Fully qualified name: $(hostname -f)"
echo "=============================="

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    _resolved_script="$(readlink -f "$SCRIPT_PATH" 2>/dev/null || true)"
    if [[ -n "$_resolved_script" ]]; then
        SCRIPT_PATH="$_resolved_script"
    fi
fi
WRAPPER_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "${WRAPPER_DIR}/../.." && pwd)"
SITE_CONFIG_PATH="${UMRS_SITE_CONFIG:-${WRAPPER_DIR}/site.yaml}"

load_site_env_defaults() {
    local python_bin
    local rendered_exports

    python_bin="$(command -v python3 || true)"
    [[ -n "$python_bin" && -f "$SITE_CONFIG_PATH" ]] || return 0

    rendered_exports="$({
        PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            UMRS_SITE_CONFIG="$SITE_CONFIG_PATH" \
            "$python_bin" - <<'PY'
import os
import shlex

from umrs.orchestrator.site_config import load_site_config

cfg = load_site_config(os.environ["UMRS_SITE_CONFIG"])
for key, value in sorted(cfg.to_env().items()):
    if os.environ.get(key):
        continue
    print(f"export {key}={shlex.quote(value)}")
PY
    } 2>/dev/null)" || return 0

    if [[ -n "$rendered_exports" ]]; then
        eval "$rendered_exports"
    fi
}

export UMRS_SITE_CONFIG="$SITE_CONFIG_PATH"
load_site_env_defaults

mask_secret_state() {
    if [[ -n "${1:-}" ]]; then
        printf '%s' '<set>'
    else
        printf '%s' '<unset>'
    fi
}

# ─── Rivault source directory on shared Lustre ───────────────────────────────
# Points to mini-rivault-llm/ inside the umrs repo on Lustre so that all
# serve-model.yaml, nginx_fs/, coredns_fs/, litellm_fs/ etc. are accessible
# from compute nodes. Override via RIVAULT_DIR env var.
# Prefer SLURM_SUBMIT_DIR (the cwd of the sbatch invocation) so the
# inner script is always picked from the same worktree as the wrapper.
_DEFAULT_REPO="${SLURM_SUBMIT_DIR:-/lvs0/dne1/rccs-dmp/mnagaso/umrs}"
RIVAULT_DIR="${RIVAULT_DIR:-${_DEFAULT_REPO}/src/umrs/mini-rivault-llm}"

if [[ ! -d "$RIVAULT_DIR" ]]; then
    echo "ERROR: RIVAULT_DIR not found: $RIVAULT_DIR"
    echo "  Check that the repo is cloned at /lvs0/dne1/rccs-dmp/mnagaso/umrs"
    exit 1
fi

# ─── MinIO SSH tunnel target (login node) ─────────────────────────────────────
# SLURM_SUBMIT_HOST = login1.cloud.r-ccs.riken.jp (the node that ran sbatch)
export UMRS_S3_TUNNEL_HOST="${UMRS_S3_TUNNEL_HOST:-${SLURM_SUBMIT_HOST}}"
export UMRS_S3_TUNNEL_PORT="${UMRS_S3_TUNNEL_PORT:-9000}"

# ─── Vision stack profile ─────────────────────────────────────────────────────
# These ${VAR:-default} values are fallback-only. The colocated site.yaml is
# loaded first when available, so wrapper defaults are only a safety net when
# site config or explicit env overrides are unavailable.
# Use "harness vllm vision-bentoml" to start all services:
#   - harness: nginx, litellm, ll-postgres, coredns, minio proxy
#   - vllm: model server for text embeddings (e5-base-v2, clip, etc.)
#   - vision-bentoml: BentoML vision service (SAM3, HMAE)
export RIVAULT_PROFILES="${RIVAULT_PROFILES:-harness vllm vision-bentoml}"

# ─── Fixed NGINX port for LiteLLM/BentoML proxy ──────────────────────────────
# The inner serve-model.slurm randomises NGINX_LITELLM_DOCKER_PORT when unset.
# We pin it so the UMRS server can create an SSH tunnel to a known port
# (RIVAULT_REMOTE_PORT in .env must match this value).
export NGINX_LITELLM_DOCKER_PORT="${NGINX_LITELLM_DOCKER_PORT:-8001}"

# ─── LiteLLM API key (shared between Slurm job and UMRS server) ─────────────
# Must match LITELLM_MASTER_KEY in .env so the server can authenticate.
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-umrs-rccs-litellm-key}"

# ─── Model cache on Lustre (inherit from server or set default) ──────────────
export SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/sam3}"
export HMAE_MODEL_PATH="${HMAE_MODEL_PATH:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae}"
export HMAE_CONFIG_PATH="${HMAE_CONFIG_PATH:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae/infer_hmae_vit_b.yaml}"
export HMAE_CHECKPOINT="${HMAE_CHECKPOINT:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae/latest_checkpoint.pth}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-/tmp}"

# ─── vLLM configuration (embedding model on GPU) ─────────────────────────────
# VLLM_PATH_TO_HUGGINGFACE: HuggingFace hub cache on Lustre (shared with login)
# VLLM_MODEL: model to serve via vLLM (must be in HF cache)
# VLLM_DOCKER_TAG: vLLM image tag — auto-selected per arch if not overridden
#   aarch64 → cu130-nightly-aarch64   (Blackwell, GB10)
#   x86_64  → cu130-nightly-x86_64    (H200, L40S, etc.)
# VLLM_GPU_MEM_UTIL: reserve headroom for BentoML vision while keeping enough KV cache for 4k-token LLM endpoints
export VLLM_PATH_TO_HUGGINGFACE="${VLLM_PATH_TO_HUGGINGFACE:-/lvs0/rccs-dmp/mnagaso/umrs_models/hf}"
export VLLM_MODEL="${VLLM_MODEL:-intfloat/e5-base-v2}"
if [[ -z "${VLLM_DOCKER_TAG:-}" ]]; then
    case "$(uname -m)" in
        aarch64|arm64) VLLM_DOCKER_TAG="cu130-nightly-aarch64" ;;
        *)             VLLM_DOCKER_TAG="cu130-nightly-x86_64"  ;;
    esac
fi
export VLLM_DOCKER_TAG
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.78}"

# ─── Fix HOME (container sets HOME=/root, but job runs as actual user) ────────
# Inside the umrs-server container HOME=/root. When sbatch is submitted,
# that HOME is exported to the job environment. On the compute node the
# job runs as the real user (masaru.nagaso), who cannot write to /root/.
# Fix HOME to the actual user home so serve-model.slurm can set up
# rootless podman storage.conf and pip packages in the right place.
_real_home="$(getent passwd "$(whoami)" | cut -d: -f6)"
if [[ -n "$_real_home" && "$_real_home" != "$HOME" ]]; then
    echo "[wrapper] Fixing HOME: ${HOME} → ${_real_home}"
    export HOME="$_real_home"
fi

# ─── Ensure podman-compose >= 1.2.0 for --profile support ─────────────────────
# Always install into a local /tmp venv on the compute node. This is fast
# (pip uses cached wheels) and avoids all NFS/PATH/version-check edge cases.
_PC_VENV="/tmp/rivault-pc-$$"
echo "[wrapper] Installing podman-compose >= 1.2.0 into ${_PC_VENV} ..."
python3 -m venv --system-site-packages "$_PC_VENV"
"${_PC_VENV}/bin/pip" install --quiet 'podman-compose>=1.2.0'
export PATH="${_PC_VENV}/bin:${PATH}"
echo "[wrapper] podman-compose ready: $(which podman-compose) — $(podman-compose -v 2>/dev/null)"

# ─── Podman storage config (must match inner serve-model.slurm) ──────────────
# The inner serve-model.slurm creates this same file; we set it up here first
# so that `podman build` stores images in the same graphroot that
# `podman-compose up` will later use.
# NOTE: $USER may be empty in Slurm jobs inherited from the container env;
# use $(whoami) to reliably get the current user name.
_WHOAMI="$(whoami)"
export USER="${_WHOAMI}"           # inner serve-model.slurm uses ${USER} in storage.conf
# Use a STABLE per-user runroot so Podman's graphroot lock never goes stale
# across Slurm jobs ($XDG_RUNTIME_DIR changes per job, which desyncs the lock).
_PODMAN_RUNROOT="/tmp/podman-runroot-${_WHOAMI}"
mkdir -p "${_PODMAN_RUNROOT}" "${HOME}/.config/containers/"
cat <<EOF > "${HOME}/.config/containers/storage.conf"
[storage]
driver="overlay"
runroot = "${_PODMAN_RUNROOT}"
rootless_storage_path="/var/tmp/podman_${_WHOAMI}/storage"
graphroot = "/var/tmp/podman_${_WHOAMI}/storage"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="nodev,metacopy=on"
EOF
# Migrate Podman's internal DB to match the new storage.conf (lightweight,
# updates BoltDB without destroying images or layers).
podman system migrate 2>/dev/null || true

# ─── Vision container image (build on compute node, cache on Lustre) ─────────
# Use a local image name so podman-compose never tries to pull from GHCR.
# Build natively on the compute node to match the target architecture.
#
# Cache hierarchy:
#   1. podman local store (/var/tmp/podman_$USER/storage) — survives across jobs on same node
#   2. Lustre tarball (per-arch) — shared across all compute nodes of same arch
#   3. Build from source — first time only, then saved to Lustre
#
# Per-arch tarballs:
#   VISION_TARBALL is treated as a base path. The wrapper appends -<arch>.tar:
#     /lvs0/.../umrs-vision-arm64.tar   (for aarch64 nodes: ng-dgx-m*)
#     /lvs0/.../umrs-vision-amd64.tar   (for x86_64 nodes: ai-h200-brc, ai-l40s, genoa, etc.)
#   This avoids cross-arch overwrites when jobs run on different node types.
_VISION_IMAGE="${BENTOML_VISION_IMAGE:-umrs-vision:latest}"
_VISION_TARBALL_BASE="${VISION_TARBALL:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/umrs-vision.tar}"
_UMRS_REPO="${UMRS_REPO:-${_DEFAULT_REPO}}"

# Derive per-arch tarball path from base
# Strip .tar suffix, append -<arch>.tar
case "$(uname -m)" in
    aarch64|arm64) _VISION_HOST_ARCH="arm64" ;;
    x86_64|amd64)  _VISION_HOST_ARCH="amd64" ;;
    *)             _VISION_HOST_ARCH="$(uname -m)" ;;
esac
_VISION_TARBALL="${_VISION_TARBALL_BASE%.tar}-${_VISION_HOST_ARCH}.tar"

export BENTOML_VISION_IMAGE="${_VISION_IMAGE}"

echo "[wrapper] Launch context:"
echo "[wrapper]   site config: ${UMRS_SITE_CONFIG:-<none>}"
echo "[wrapper]   profiles: ${RIVAULT_PROFILES}"
echo "[wrapper]   rivault dir: ${RIVAULT_DIR}"
echo "[wrapper]   repo root: ${_UMRS_REPO}"
echo "[wrapper]   compose files: ${RIVAULT_DIR}/serve-model.yaml ; ${RIVAULT_DIR}/userdata-access.yaml"
echo "[wrapper]   litellm port: ${NGINX_LITELLM_DOCKER_PORT}"
echo "[wrapper]   litellm key: $(mask_secret_state "${LITELLM_MASTER_KEY:-}")"
echo "[wrapper]   hf cache: ${VLLM_PATH_TO_HUGGINGFACE}"
echo "[wrapper]   sam3 path: ${SAM3_MODEL_PATH}"
echo "[wrapper]   hmae path: ${HMAE_MODEL_PATH}"
echo "[wrapper]   vision image: ${BENTOML_VISION_IMAGE}"
echo "[wrapper]   vision tarball: ${_VISION_TARBALL} (arch=${_VISION_HOST_ARCH})"

# Fix cgroup manager for Slurm compute nodes (no systemd user session)
export CONTAINERS_CGROUP_MANAGER=cgroupfs
loginctl enable-linger "$(whoami)" 2>/dev/null || true

# ─── GPU passthrough: use CDI or device-based x_nvidia-override.yaml ──────────
# Prefer CDI (nvidia-ctk) when available — it handles both device nodes AND
# host driver library mounts automatically. Fall back to raw device mapping
# (which may require manual library volume mounts for containers like vLLM).
# Generate the override in a runtime path rather than rewriting the tracked
# source file under ${RIVAULT_DIR}; that keeps shared repo checkouts clean for
# deploy/update cycles.
if ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
    _OVERRIDE_DIR="${RIVAULT_GPU_OVERRIDE_DIR:-${TMPDIR:-/tmp}}"
    mkdir -p "${_OVERRIDE_DIR}"
    _OVERRIDE="${RIVAULT_GPU_OVERRIDE_FILE:-${_OVERRIDE_DIR%/}/umrs-x_nvidia-override-${SLURM_JOB_ID:-$$}.yaml}"
    export RIVAULT_GPU_OVERRIDE_FILE="${_OVERRIDE}"
    export RIVAULT_GPU_OVERRIDE_GENERATED=1
    if nvidia-ctk cdi list 2>/dev/null | grep -q 'nvidia.com/gpu=all'; then
        echo "[wrapper] CDI available — generating CDI-based GPU override: ${_OVERRIDE}"
        cat > "${_OVERRIDE}" <<GPUEOF
services:
  vllm:
    shm_size: \${VLLM_NCCL_MEM_LIMIT:?error}
    devices:
      - nvidia.com/gpu=all
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

  bentoml-vision:
    devices:
      - nvidia.com/gpu=all
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
GPUEOF
    else
        echo "[wrapper] CDI not available — generating device-based GPU override: ${_OVERRIDE}"
        _GPU_DEVS=""
        for _dev in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-caps/nvidia-cap1 /dev/nvidia-caps/nvidia-cap2; do
            [[ -e "$_dev" ]] && _GPU_DEVS="${_GPU_DEVS}      - ${_dev}:${_dev}:rwm\n"
        done
        cat > "${_OVERRIDE}" <<GPUEOF
services:
  vllm:
    shm_size: \${VLLM_NCCL_MEM_LIMIT:?error}
    devices:
$(echo -e "$_GPU_DEVS")
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

  bentoml-vision:
    devices:
$(echo -e "$_GPU_DEVS")
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
GPUEOF
    fi
    echo "[wrapper] GPU override written."
else
    unset RIVAULT_GPU_OVERRIDE_FILE || true
    unset RIVAULT_GPU_OVERRIDE_GENERATED || true
    echo "[wrapper] No NVIDIA GPUs detected at /dev/nvidia*"
fi

# ─── Clean up stale podman state from previous runs ──────────────────────────
# Each Slurm job starts fresh: remove all containers, pods, and networks so
# that lock-corrupted state from previous failed jobs never blocks startup.
# NOTE: We intentionally KEEP the image cache (/var/tmp/podman_$USER/storage)
# so that large images (vLLM ~10GB, vision ~5GB) don't need to be re-pulled
# or re-loaded on every job.

# ─── OCI runtime detection (userns_mode compatibility) ───────────────────────
# crun supports "keep-id:uid=0,gid=0" which rootless Lustre bind-mounts need.
# runc (<= 1.3.x, found on aarch64 ng-dgx nodes) does NOT support keep-id
# and fails with: "runc create failed: can't get final child's PID from pipe".
# For runc nodes, fall back to "host" userns_mode which skips user namespace
# remapping and works with both runc and Lustre mounts.
if [[ -z "${RIVAULT_USERNS_MODE:-}" ]]; then
    _oci_runtime="$(podman info --format '{{.Host.OCIRuntime.Name}}' 2>/dev/null || echo unknown)"
    echo "[wrapper] Detected OCI runtime: ${_oci_runtime}"
    case "${_oci_runtime}" in
        crun*)
            export RIVAULT_USERNS_MODE="keep-id:uid=0,gid=0"
            ;;
        runc*|*)
            # runc and unknown runtimes: use host userns (safe default)
            export RIVAULT_USERNS_MODE="host"
            ;;
    esac
    echo "[wrapper] userns_mode set to: ${RIVAULT_USERNS_MODE}"
fi

podman pod rm -af 2>/dev/null || true
# Clean orphaned containers from previous Slurm jobs.
# When a prior job's $XDG_RUNTIME_DIR was cleaned by Slurm, the crun state
# files vanish but container metadata persists in graphroot. Normal
# podman rm can't stop containers whose runtime state is gone.
# "container cleanup --rm --all" cleans metadata without deleting images.
pkill -9 conmon 2>/dev/null || true
sleep 1
podman container cleanup --rm --all 2>/dev/null || true
podman rm -af 2>/dev/null || true
podman network prune -f 2>/dev/null || true

# Also clean stale storage left by earlier runs that used an empty $USER
# (graphroot "/var/tmp/podman_/storage/" instead of the correct one).
if [[ -d "/var/tmp/podman_/storage" ]]; then
    echo "[wrapper] Removing stale /var/tmp/podman_/storage"
    rm -rf "/var/tmp/podman_/storage"
fi

_build_vision_image() {
    echo "[wrapper] Building vision image on compute node (may take several minutes)..."
    if podman build --cgroup-manager=cgroupfs -t "${_VISION_IMAGE}" \
         -f "${_UMRS_REPO}/src/umrs/vision_service/Dockerfile" "${_UMRS_REPO}"; then
        echo "[wrapper] Saving tarball to Lustre: ${_VISION_TARBALL}"
        mkdir -p "$(dirname "${_VISION_TARBALL}")"
        podman save "${_VISION_IMAGE}" -o "${_VISION_TARBALL}.tmp" && mv "${_VISION_TARBALL}.tmp" "${_VISION_TARBALL}"
        echo "[wrapper] Vision image built and cached (${_VISION_HOST_ARCH})."
        return 0
    fi

    echo "[wrapper] ERROR: podman build failed (exit $?). Vision service will not be available."
    return 1
}

# ─── Load or build the vision image ─────────────────────────────────────────
# The tarball path already includes the host arch suffix, so no cross-arch
# mismatch check is needed — each arch loads its own tarball.
if [[ -f "${_VISION_TARBALL}" ]]; then
    echo "[wrapper] Loading vision image from Lustre tarball: ${_VISION_TARBALL}"
    podman load -i "${_VISION_TARBALL}"
    echo "[wrapper] Vision image loaded (${_VISION_HOST_ARCH})."
else
    echo "[wrapper] No tarball at ${_VISION_TARBALL}; building natively..."
    _build_vision_image || true
fi

# ─── Delegate to the main serve-model.slurm in RIVAULT_DIR ───────────────────
# Use --norc --noprofile so .bashrc doesn't reset PATH (which would hide the
# venv podman-compose we just installed above).
cd "$RIVAULT_DIR"
exec bash --norc --noprofile ./serve-model.slurm
