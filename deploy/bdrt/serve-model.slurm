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
##SBATCH --partition=ng-dgx-m0  # disabled — partition set by orchestrator via --partition CLI arg
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120

# ─── Output FQDN header (required by rivault_launcher.extract_node_address) ───
echo "=============================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NODELIST"
echo "Running on node: $(hostname)"
echo "Fully qualified name: $(hostname -f)"
echo "=============================="

if [[ -n "${RIVAULT_WRAPPER_DIR:-}" ]]; then
    WRAPPER_DIR="$RIVAULT_WRAPPER_DIR"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
    if command -v readlink >/dev/null 2>&1; then
        _resolved_script="$(readlink -f "$SCRIPT_PATH" 2>/dev/null || true)"
        if [[ -n "$_resolved_script" ]]; then
            SCRIPT_PATH="$_resolved_script"
        fi
    fi
    WRAPPER_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
fi
REPO_ROOT="$(cd "${WRAPPER_DIR}/../.." && pwd)"
SITE_CONFIG_PATH="${UMRS_SITE_CONFIG:-${WRAPPER_DIR}/site.yaml}"

if [[ -z "${LITELLM_MASTER_KEY:-}" && -f "${WRAPPER_DIR}/.env" ]]; then
    _litellm_master_key="$(awk -F= '$1 == "LITELLM_MASTER_KEY" {sub(/^[^=]*=/, ""); print; exit}' "${WRAPPER_DIR}/.env")"
    if [[ -n "$_litellm_master_key" ]]; then
        export LITELLM_MASTER_KEY="$_litellm_master_key"
    fi
    unset _litellm_master_key
fi

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

# ─── Partition profile lookup ────────────────────────────────────────────────
# Resolve the Slurm partition for this job and export per-partition profile
# vars (userns_mode, OCI runtime, network backend, vLLM docker tag, etc.)
# from site.yaml, eliminating fragile runtime detection.
_PARTITION_PROFILE_LOADED=0
load_partition_profile() {
    local python_bin
    local rendered_exports
    local partition="${1:-}"

    [[ -n "$partition" ]] || return 0
    python_bin="$(command -v python3 || true)"
    [[ -n "$python_bin" && -f "$SITE_CONFIG_PATH" ]] || return 0

    rendered_exports="$({
        "$python_bin" - "$SITE_CONFIG_PATH" "$partition" <<'PY'
import sys, os, shlex, yaml

config_path = sys.argv[1]
partition   = sys.argv[2]

with open(config_path, "r") as fh:
    cfg = yaml.safe_load(fh)

pp = (cfg or {}).get("partition_profiles", {}).get(partition)
if pp is None:
    sys.exit(0)

env_map = {
    "RIVAULT_PARTITION_ARCH":             pp.get("arch"),
    "RIVAULT_PARTITION_GPU_VENDOR":       pp.get("gpu_vendor"),
    "RIVAULT_PARTITION_OCI_RUNTIME":        pp.get("oci_runtime"),
    "RIVAULT_USERNS_MODE":                pp.get("userns_mode"),
    "RIVAULT_PARTITION_NETWORK_BACKEND":  pp.get("network_backend"),
    "RIVAULT_PARTITION_GPUS":             pp.get("gpus"),
}
if pp.get("needs_netavark_switch", False):
    env_map["RIVAULT_PARTITION_NEEDS_NETAVARK"] = "1"
else:
    env_map["RIVAULT_PARTITION_NEEDS_NETAVARK"] = "0"
if pp.get("vllm_docker_tag"):
    env_map["VLLM_DOCKER_TAG"] = pp["vllm_docker_tag"]
if pp.get("vllm_docker_image"):
    env_map["VLLM_DOCKER_IMAGE"] = pp["vllm_docker_image"]

for key, value in sorted(env_map.items()):
    if value is not None:
        print(f"export {key}={shlex.quote(str(value))}")
PY
    } 2>/dev/null)" || return 0

    if [[ -n "$rendered_exports" ]]; then
        eval "$rendered_exports"
        _PARTITION_PROFILE_LOADED=1
        echo "[wrapper] Partition profile loaded for: ${partition}"
    fi
}

# Resolve which partition this job landed on.  SLURM_JOB_PARTITION is set by
# Slurm on the compute node.  If it contains a comma-separated list (multi-
# partition submission), take the first one.
_ACTIVE_PARTITION="${SLURM_JOB_PARTITION%%,*}"
if [[ -z "$_ACTIVE_PARTITION" ]]; then
    _ACTIVE_PARTITION="${SLURM_PARTITION:-}"
fi
load_partition_profile "$_ACTIVE_PARTITION"

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
# Prefer SLURM_SUBMIT_DIR only when sbatch was launched from a full UMRS
# checkout; front-end remote submissions often run from the login home dir.
_DEFAULT_REPO="$REPO_ROOT"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/umrs/mini-rivault-llm" ]]; then
    _DEFAULT_REPO="$SLURM_SUBMIT_DIR"
fi
RIVAULT_DIR="${RIVAULT_DIR:-${_DEFAULT_REPO}/src/umrs/mini-rivault-llm}"

if [[ ! -d "$RIVAULT_DIR" ]]; then
    echo "ERROR: RIVAULT_DIR not found: $RIVAULT_DIR"
    echo "  Check that the repo is cloned at $REPO_ROOT or set RIVAULT_DIR."
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
#
# The fallback default is the LLM/text profile set (no vision sidecar). The
# UMRS server passes RIVAULT_PROFILES explicitly per endpoint:
#   - LLM / embedding  -> "harness vllm litellm-db"
#   - vision (BentoML) -> "harness vision-bentoml litellm-db" (+ vllm if co-located)
# Keeping vision-bentoml OUT of the fallback prevents a text-model job from
# accidentally launching the BentoML vision stack and starving vLLM of GPU.
# Available profiles:
#   - harness: nginx, litellm, coredns, minio proxy
#   - vllm: model server for LLM / text embeddings (e5-base-v2, clip, etc.)
#   - vision-bentoml: BentoML vision service (SAM3, HMAE)
#   - litellm-db: PostgreSQL backend for LiteLLM (spend logs, schema)
export RIVAULT_PROFILES="${RIVAULT_PROFILES:-harness vllm litellm-db}"

# ─── NGINX port for LiteLLM/BentoML proxy ────────────────────────────────────
# Endpoint provisioning passes LITELLM_PORT per job. Mirror it into the compose
# publish variable so shell exports cannot override the generated env file.
# Manual wrapper runs still default to 8001 for operator probes.
export NGINX_LITELLM_DOCKER_PORT="${LITELLM_PORT:-${NGINX_LITELLM_DOCKER_PORT:-8001}}"

# ─── LiteLLM API key (shared between Slurm job and UMRS server) ─────────────
# Must match LITELLM_MASTER_KEY in .env so the server can authenticate.
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:?Set LITELLM_MASTER_KEY in .env}"

# ─── Model cache on shared home (inherit from server or set default) ──────────
export SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-/home/${USER}/umrs_models/sam3.1}"
export HMAE_MODEL_PATH="${HMAE_MODEL_PATH:-/home/${USER}/umrs_models/hmae}"
export HMAE_CONFIG_PATH="${HMAE_CONFIG_PATH:-/home/${USER}/umrs_models/hmae/infer_hmae_vit_b.yaml}"
export HMAE_CHECKPOINT="${HMAE_CHECKPOINT:-/home/${USER}/umrs_models/hmae/latest_checkpoint.pth}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-/tmp}"

# ─── vLLM configuration (embedding model on GPU) ─────────────────────────────
# VLLM_PATH_TO_HUGGINGFACE: HuggingFace hub cache on shared home
# VLLM_MODEL: model to serve via vLLM (must be in HF cache)
# VLLM_DOCKER_TAG: vLLM image tag — auto-selected per arch if not overridden
#   aarch64 → cu130-nightly-aarch64   (Blackwell, GB10)
#   x86_64  → cu130-nightly-x86_64    (H200, L40S, etc.)
# VLLM_GPU_MEM_UTIL: low memory fraction — keep room for BentoML vision
export VLLM_PATH_TO_HUGGINGFACE="${VLLM_PATH_TO_HUGGINGFACE:-/home/${USER}/umrs_models}"
export VLLM_MODEL="${VLLM_MODEL:-intfloat/e5-base-v2}"
# VLLM_DOCKER_TAG: prefer partition profile value; fall back to arch auto-detect.
if [[ -z "${VLLM_DOCKER_TAG:-}" ]]; then
    case "$(uname -m)" in
        aarch64|arm64) VLLM_DOCKER_TAG="cu130-nightly-aarch64" ;;
        *)             VLLM_DOCKER_TAG="cu130-nightly-x86_64"  ;;
    esac
fi
export VLLM_DOCKER_TAG
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.10}"

# ─── BDRT path normalization guard ───────────────────────────────────────────
# Some launch paths may still inject legacy /lvs0 model cache locations.
# On BDRT compute nodes those paths are unavailable; normalize to /home.
_LEGACY_MODEL_ROOT="/lvs0/dne1/rccs-dmp/${USER}/umrs_models"
_BDRT_MODEL_ROOT="/home/${USER}/umrs_models"
normalize_model_path_var() {
    local key="${1}"
    local value="${!key:-}"
    [[ -n "${value}" ]] || return 0
    if [[ "${value}" == "${_LEGACY_MODEL_ROOT}"* ]]; then
        local normalized="${value/#${_LEGACY_MODEL_ROOT}/${_BDRT_MODEL_ROOT}}"
        export "${key}=${normalized}"
        echo "[wrapper] Normalized ${key}: ${value} -> ${normalized}"
    fi
}
normalize_model_path_var "VLLM_PATH_TO_HUGGINGFACE"
normalize_model_path_var "SAM3_MODEL_PATH"
normalize_model_path_var "HMAE_MODEL_PATH"
normalize_model_path_var "HMAE_CONFIG_PATH"
normalize_model_path_var "HMAE_CHECKPOINT"
normalize_model_path_var "VISION_TARBALL"

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
mkdir -p "${HOME}/.config/containers/"
cat <<EOF > "${HOME}/.config/containers/storage.conf"
[storage]
driver="overlay"
runroot = "${XDG_RUNTIME_DIR}/containers/storage"
rootless_storage_path="/var/tmp/podman_${_WHOAMI}/storage"
graphroot = "/var/tmp/podman_${_WHOAMI}/storage"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="xattr_permissions=2,threaded=1"
EOF

# ─── Clean up stale podman state from previous runs ──────────────────────────
# Each Slurm job starts fresh: remove all containers, pods, and networks so
# that lock-corrupted state from previous failed jobs never blocks startup.
# NOTE: We intentionally KEEP the image cache (/var/tmp/podman_$USER/storage)
# so that large images (vLLM ~10GB, vision ~5GB) don't need to be re-pulled
# or re-loaded on every job.
# IMPORTANT: This block MUST run BEFORE the netavark switch below. podman
# system migrate resets the network backend to the system default ("cni");
# the netavark switch then overrides it to "netavark" and runs its own
# targeted migrate.

# Step 1: Kill ALL container runtime processes so nothing is running
pkill -9 -u "$_WHOAMI" conmon 2>/dev/null || true
pkill -9 -u "$_WHOAMI" crun   2>/dev/null || true
sleep 2
# Step 2: Stop any stale systemd transient scopes that podman uses to track
# containers. These survive Slurm job boundaries and cause podman to
# re-discover dead containers even after their metadata is removed.
systemctl --user reset-failed 2>/dev/null || true
systemctl --user list-units --type=scope --state=running --no-legend 2>/dev/null \
  | awk '/libpod/{print $1}' \
  | xargs -r systemctl --user stop 2>/dev/null || true
# Step 3: Clear stale runroot container storage from earlier Slurm jobs.
# Must happen BEFORE graphroot cleanup because podman reads runroot state.
_GRAPHROOT_CLEANUP="/var/tmp/podman_${_WHOAMI}/storage"
_MY_JOB="job-${SLURM_JOB_ID}"
find /tmp/ -maxdepth 1 -name "job-*" -not -name "${_MY_JOB}" -user "$_WHOAMI" 2>/dev/null \
  | while read -r _jd; do rm -rf "${_jd}/xdg/containers" 2>/dev/null; done || true
# Step 4: Clear stale container metadata from the graphroot.
# Containers stuck in "stopping" state survive across Slurm jobs because
# overlay-containers/containers.json persists in the graphroot while the
# per-job runroot is lost.  Clearing these entries lets podman start fresh.
# Image layers (overlay-images, overlay-layers) are preserved so large
# images don't need re-loading.
if [[ -f "${_GRAPHROOT_CLEANUP}/overlay-containers/containers.json" ]]; then
    echo "[wrapper] Clearing stale container entries from graphroot"
    printf '[]' > "${_GRAPHROOT_CLEANUP}/overlay-containers/containers.json"
    # Remove per-container metadata dirs (layer mounts, userdata, etc.)
    find "${_GRAPHROOT_CLEANUP}/overlay-containers/" -mindepth 1 -maxdepth 1 \
         -type d -exec rm -rf {} + 2>/dev/null || true
fi
# Podman 5.x uses a SQLite DB (db.sql) at the graphroot root, NOT the
# old BoltDB (libpod/bolt_state.db).  Delete BOTH to cover all versions.
rm -f "${_GRAPHROOT_CLEANUP}/db.sql" 2>/dev/null || true
# Also wipe the entire libpod state directory (legacy bolt DB, tmp)
rm -rf "${_GRAPHROOT_CLEANUP}/libpod" 2>/dev/null || true
mkdir -p "${_GRAPHROOT_CLEANUP}/libpod"
# Clear runtime libpod state (pause pid, transient scopes)
rm -rf "/run/user/$(id -u)/libpod" 2>/dev/null || true
# Clear /tmp/run-user-* runtime state (podman uses this on some nodes for
# container storage and network state outside the XDG_RUNTIME_DIR tree)
rm -rf "/tmp/run-user-$(id -u)/containers" 2>/dev/null || true
rm -rf "/tmp/run-user-$(id -u)/libpod/tmp" 2>/dev/null || true
# Step 5: Run migrate to reinitialize (creates fresh state DB)
podman system migrate 2>/dev/null || true
echo "[wrapper] Post-cleanup: $(podman ps -aq 2>/dev/null | wc -l | tr -d ' ') containers remaining"
# Step 6: Clean up any remaining resources
podman network prune -f 2>/dev/null || true
# Also remove stale CNI conflist files that cause validation warnings
rm -rf "${HOME}/.config/cni/net.d/"*.conflist 2>/dev/null || true

# ─── Force netavark network backend ─────────────────────────────────────────
# Prefer the partition profile's needs_netavark_switch flag.  Fall back to
# runtime detection when no profile is loaded (backwards compatible).
_GRAPHROOT="/var/tmp/podman_${_WHOAMI}/storage"
_NET_BACKEND_FILE="${_GRAPHROOT}/defaultNetworkBackend"
_need_netavark_switch=0
if [[ "${RIVAULT_PARTITION_NEEDS_NETAVARK:-}" == "1" ]]; then
    _need_netavark_switch=1
elif [[ "${RIVAULT_PARTITION_NEEDS_NETAVARK:-}" == "0" ]]; then
    _need_netavark_switch=0
elif [[ -x /usr/libexec/podman/netavark ]]; then
    # No profile loaded — fall back to runtime detection
    if [[ ! -x /usr/libexec/cni/bridge ]] && [[ ! -x /usr/lib/cni/bridge ]] && [[ ! -x /opt/cni/bin/bridge ]]; then
        _need_netavark_switch=1
    fi
fi
if [[ "${_need_netavark_switch}" -eq 1 ]]; then
    echo "[wrapper] Forcing netavark network backend (CNI plugins missing, netavark available)"
    mkdir -p "${_GRAPHROOT}"
    printf '%s' "netavark" > "${_NET_BACKEND_FILE}"
    # Remove stale CNI network configs that won't work with netavark
    rm -rf "${HOME}/.config/cni/net.d/"*.conflist 2>/dev/null || true
    # Clean network state so migrate works cleanly
    rm -rf "${_GRAPHROOT}/networks" 2>/dev/null || true
    rm -rf "${_GRAPHROOT}/libpod" 2>/dev/null || true
    # Migrate podman state for the new backend
    podman system migrate 2>/dev/null || true
    echo "[wrapper] Network backend: $(cat "${_NET_BACKEND_FILE}" 2>/dev/null)"
fi

# ─── Vision container image (build on compute node, cache on Lustre) ─────────
# Use a local image name so podman-compose never tries to pull from GHCR.
# Build natively on the compute node to match the target architecture.
#
# Cache hierarchy:
#   1. podman local store (/var/tmp/podman_$USER/storage) — survives across jobs on same node
#   2. Shared-home tarball (per-arch) — shared across all compute nodes of same arch
#   3. Build from source — first time only, then saved to shared home
#
# Per-arch tarballs:
#   VISION_TARBALL is treated as a base path. The wrapper appends -<arch>.tar:
#     /home/.../umrs-vision-arm64.tar   (for aarch64 nodes)
#     /home/.../umrs-vision-amd64.tar   (for x86_64 nodes)
#   This avoids cross-arch overwrites when jobs run on different node types.
_VISION_IMAGE="${BENTOML_VISION_IMAGE:-umrs-vision:latest}"
_VISION_TARBALL_BASE="${VISION_TARBALL:-/home/${USER}/umrs_models/umrs-vision.tar}"
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
export DEFAULT_ADAPTERS_DIR="${DEFAULT_ADAPTERS_DIR:-${_UMRS_REPO}/src/umrs/vision_service/default_adapters}"
export USER_ADAPTERS_DIR="${USER_ADAPTERS_DIR:-${RIVAULT_DIR}/user_adapters}"

# Some podman-compose builds on rccs-cloud resolve the bentoml adapter bind
# from ${RIVAULT_DIR}/default_adapters even when serve-model.yaml points at
# ${DEFAULT_ADAPTERS_DIR}. Ensure that path exists and mirrors the real
# adapter directory before the inner compose launcher starts.
if [[ "$(readlink "${RIVAULT_DIR}/default_adapters" 2>/dev/null || true)" != "${DEFAULT_ADAPTERS_DIR}" ]]; then
    rm -rf "${RIVAULT_DIR}/default_adapters"
    ln -s "${DEFAULT_ADAPTERS_DIR}" "${RIVAULT_DIR}/default_adapters"
fi

echo "[wrapper] Launch context:"
echo "[wrapper]   site config: ${UMRS_SITE_CONFIG:-<none>}"
echo "[wrapper]   partition: ${_ACTIVE_PARTITION:-<unknown>} (profile=${_PARTITION_PROFILE_LOADED})"
echo "[wrapper]   profiles: ${RIVAULT_PROFILES}"
echo "[wrapper]   rivault dir: ${RIVAULT_DIR}"
echo "[wrapper]   repo root: ${_UMRS_REPO}"
echo "[wrapper]   default adapters dir: ${DEFAULT_ADAPTERS_DIR}"
echo "[wrapper]   user adapters dir: ${USER_ADAPTERS_DIR}"
echo "[wrapper]   compose files: ${RIVAULT_DIR}/serve-model.yaml ; ${RIVAULT_DIR}/userdata-access.yaml"
echo "[wrapper]   litellm port: ${NGINX_LITELLM_DOCKER_PORT}"
echo "[wrapper]   litellm key: $(mask_secret_state "${LITELLM_MASTER_KEY:-}")"
echo "[wrapper]   hf cache: ${VLLM_PATH_TO_HUGGINGFACE}"
echo "[wrapper]   sam3 path: ${SAM3_MODEL_PATH}"
echo "[wrapper]   hmae path: ${HMAE_MODEL_PATH}"
echo "[wrapper]   vision image: ${BENTOML_VISION_IMAGE}"
echo "[wrapper]   vision tarball: ${_VISION_TARBALL} (arch=${_VISION_HOST_ARCH})"

JOB_INFO_DIR="${JOB_INFO_DIR:-${WRAPPER_DIR}/jobs}"
JOB_INFO_FILE="${JOB_INFO_DIR}/rivault-job-${SLURM_JOB_ID}.info"
mkdir -p "$JOB_INFO_DIR"
cat > "$JOB_INFO_FILE" <<EOF
RIVAULT_STATUS=running
RIVAULT_NODE=$(hostname -f 2>/dev/null || hostname)
RIVAULT_LITELLM_PORT=${NGINX_LITELLM_DOCKER_PORT}
RIVAULT_VLLM_PORT=${VLLM_PORT:-8000}
RIVAULT_MODEL_PATH=${VLLM_MODEL_PATH:-${USER_MODEL_PATH:-}}
EOF
trap 'rm -f "$JOB_INFO_FILE"' EXIT
echo "[wrapper]   job info: ${JOB_INFO_FILE}"

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
    # Always use direct device mapping — Podman 4.9.3 on RCCS cannot resolve
    # CDI devices via podman-compose (nvidia.com/gpu=all is unresolvable).
    echo "[wrapper] Generating device-based GPU override: ${_OVERRIDE}"
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
    volumes:
      - /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro
      - /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

  bentoml-vision:
    devices:
$(echo -e "$_GPU_DEVS")
    volumes:
      - /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro
      - /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
GPUEOF
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
# Prefer the partition profile's userns_mode.  Fall back to runtime detection
# via podman info when no profile is loaded (backwards compatible).
if [[ -z "${RIVAULT_USERNS_MODE:-}" ]]; then
    if [[ "${_PARTITION_PROFILE_LOADED}" -eq 1 ]]; then
        echo "[wrapper] userns_mode already set from partition profile"
    else
        _oci_runtime="$(podman info --format '{{.Host.OCIRuntime.Name}}' 2>/dev/null || echo unknown)"
        echo "[wrapper] Detected OCI runtime: ${_oci_runtime} (no partition profile)"
        case "${_oci_runtime}" in
            crun*)
                export RIVAULT_USERNS_MODE="keep-id:uid=0,gid=0"
                ;;
            runc*|*)
                export RIVAULT_USERNS_MODE="host"
                ;;
        esac
    fi
fi
echo "[wrapper] userns_mode: ${RIVAULT_USERNS_MODE:-<unset>}"

podman pod rm -af 2>/dev/null || true
# Clean orphaned containers from previous Slurm jobs.
# (Primary cleanup runs earlier in the script before the netavark switch;
# this second pass catches anything created by OCI runtime detection or
# GPU override generation.)
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
    local build_status

    podman build --cgroup-manager=cgroupfs -t "${_VISION_IMAGE}" \
         -f "${_UMRS_REPO}/src/umrs/vision_service/Dockerfile" "${_UMRS_REPO}"
    build_status=$?
    if [[ "${build_status}" -eq 0 ]]; then
        echo "[wrapper] Saving tarball to shared home: ${_VISION_TARBALL}"
        mkdir -p "$(dirname "${_VISION_TARBALL}")"
        podman save "${_VISION_IMAGE}" -o "${_VISION_TARBALL}.tmp" && mv "${_VISION_TARBALL}.tmp" "${_VISION_TARBALL}"
        echo "[wrapper] Vision image built and cached (${_VISION_HOST_ARCH})."
        return 0
    fi

    echo "[wrapper] ERROR: podman build failed (exit ${build_status}). Vision service will not be available."
    return "${build_status}"
}

# ─── Load or build the vision image ─────────────────────────────────────────
# The tarball path already includes the host arch suffix, so no cross-arch
# mismatch check is needed — each arch loads its own tarball.
if [[ -f "${_VISION_TARBALL}" ]]; then
    echo "[wrapper] Loading vision image from shared-home tarball: ${_VISION_TARBALL}"
    podman load -i "${_VISION_TARBALL}"
    echo "[wrapper] Vision image loaded (${_VISION_HOST_ARCH})."
else
    echo "[wrapper] No tarball at ${_VISION_TARBALL}; building natively..."
    _build_vision_image || exit $?
fi

# ─── Delegate to the main serve-model.slurm in RIVAULT_DIR ───────────────────
# Use --norc --noprofile so .bashrc doesn't reset PATH (which would hide the
# venv podman-compose we just installed above).
# Mark that the wrapper already handled cleanup so the inner script skips its
# duplicate recovery block.
export RIVAULT_WRAPPER_CLEANUP=1

# ─── GOMAXPROCS: prevent Go-based services from NPROC exhaustion ─────────────
# On rccs-cloud the per-user NPROC ulimit is low (~4096). Go runtime's default
# goroutine scheduling creates one OS thread per available CPU core, which
# combined with podman's conmon processes quickly exceeds the limit.
# Setting GOMAXPROCS=1 restricts Go services (BentoML, MinIO proxy, etc.)
# to a single OS thread per goroutine group, keeping total thread count safe.
export GOMAXPROCS="${GOMAXPROCS:-1}"
echo "[wrapper] GOMAXPROCS=${GOMAXPROCS}"

cd "$RIVAULT_DIR"
exec bash --norc --noprofile ./serve-model.slurm
