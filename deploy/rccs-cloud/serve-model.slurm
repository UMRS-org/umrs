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
#   For a quick test/vision inference job use dgx-s.
# =============================================================================

#SBATCH --job-name=umrs-rivault
#SBATCH --partition=ng-dgx-s
#SBATCH --nodes=1
#SBATCH --time=4:00:00

# ─── Output FQDN header (required by rivault_launcher.extract_node_address) ───
echo "=============================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NODELIST"
echo "Running on node: $(hostname)"
echo "Fully qualified name: $(hostname -f)"
echo "=============================="

# ─── Rivault source directory on shared Lustre ───────────────────────────────
# Points to mini-rivault-llm/ inside the umrs repo on Lustre so that all
# serve-model.yaml, nginx_fs/, coredns_fs/, litellm_fs/ etc. are accessible
# from compute nodes. Override via RIVAULT_DIR env var.
RIVAULT_DIR="${RIVAULT_DIR:-/lvs0/dne1/rccs-dmp/mnagaso/umrs/src/umrs/mini-rivault-llm}"

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
# Use "harness vision-bentoml" so the BentoML vision service is started.
# This covers both SAM3 segmentation and HMAE inference.
export RIVAULT_PROFILES="${RIVAULT_PROFILES:-harness vision-bentoml}"

# ─── Model cache on Lustre (inherit from server or set default) ──────────────
export SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/sam3}"
export HMAE_MODEL_PATH="${HMAE_MODEL_PATH:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae}"
export HMAE_CONFIG_PATH="${HMAE_CONFIG_PATH:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae/infer_hmae_vit_b.yaml}"
export HMAE_CHECKPOINT="${HMAE_CHECKPOINT:-/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae/latest_checkpoint.pth}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-/tmp}"

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
mountopt="nodev,metacopy=on"
EOF

# ─── Vision container image (build on compute node, cache on Lustre) ─────────
# Use a local image name so podman-compose never tries to pull from GHCR.
# Build natively on the compute node (aarch64) to match the GPU hardware.
#
# Cache hierarchy:
#   1. podman local store (/var/tmp/podman_$USER/storage) — survives across jobs on same node
#   2. Lustre tarball — shared across all compute nodes
#   3. Build from source — first time only, then saved to Lustre
_VISION_IMAGE="umrs-vision:latest"
_VISION_TARBALL="/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/umrs-vision.tar"
_UMRS_REPO="/lvs0/dne1/rccs-dmp/mnagaso/umrs"

export BENTOML_VISION_IMAGE="${_VISION_IMAGE}"

# Fix cgroup manager for Slurm compute nodes (no systemd user session)
export CONTAINERS_CGROUP_MANAGER=cgroupfs
loginctl enable-linger "$(whoami)" 2>/dev/null || true

# ─── GPU passthrough: generate device-based x_nvidia-override.yaml ────────────
# podman 4.x on this cluster cannot resolve CDI `nvidia.com/gpu=all`.
# Instead, detect GPU device nodes and overwrite x_nvidia-override.yaml
# with direct /dev/nvidia* device mappings. The inner serve-model.slurm
# picks up x_nvidia-override.yaml automatically when nvidia-smi detects GPUs.
if ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
    _GPU_DEVS=""
    for _dev in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-caps/nvidia-cap1 /dev/nvidia-caps/nvidia-cap2; do
        [[ -e "$_dev" ]] && _GPU_DEVS="${_GPU_DEVS}      - ${_dev}:${_dev}:rwm\n"
    done
    if [[ -n "$_GPU_DEVS" ]]; then
        _OVERRIDE="${RIVAULT_DIR}/x_nvidia-override.yaml"
        echo "[wrapper] Generating device-based GPU override: ${_OVERRIDE}"
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
        echo "[wrapper] Device-based GPU override written."
    fi
else
    echo "[wrapper] No NVIDIA GPUs detected at /dev/nvidia*"
fi

# ─── Clean up stale podman state from previous runs ──────────────────────────
# Each Slurm job starts fresh: reset ALL containers/pods/networks so that
# lock-corrupted state from previous failed jobs never blocks startup.
# `podman system reset --force` removes containers, pods, networks, but
# also clears the image store. We reload the vision image from the Lustre
# tarball afterward (fast — just a local copy).
podman system reset --force 2>/dev/null || true

# Also clean stale storage left by earlier runs that used an empty $USER
# (graphroot "/var/tmp/podman_/storage/" instead of the correct one).
if [[ -d "/var/tmp/podman_/storage" ]]; then
    echo "[wrapper] Removing stale /var/tmp/podman_/storage"
    rm -rf "/var/tmp/podman_/storage"
fi

# ─── Load or build the vision image ─────────────────────────────────────────
if [[ -f "${_VISION_TARBALL}" ]]; then
    echo "[wrapper] Loading vision image from Lustre tarball: ${_VISION_TARBALL}"
    podman load -i "${_VISION_TARBALL}"
    echo "[wrapper] Vision image loaded."
else
    echo "[wrapper] Building vision image on compute node (first time — may take several minutes)..."
    if podman build --cgroup-manager=cgroupfs -t "${_VISION_IMAGE}" \
         -f "${_UMRS_REPO}/src/umrs/vision_service/Dockerfile" "${_UMRS_REPO}"; then
        echo "[wrapper] Saving tarball to Lustre: ${_VISION_TARBALL}"
        mkdir -p "$(dirname "${_VISION_TARBALL}")"
        podman save "${_VISION_IMAGE}" -o "${_VISION_TARBALL}.tmp" && mv "${_VISION_TARBALL}.tmp" "${_VISION_TARBALL}"
        echo "[wrapper] Vision image built and cached."
    else
        echo "[wrapper] ERROR: podman build failed (exit $?). Vision service will not be available."
    fi
fi

# ─── Delegate to the main serve-model.slurm in RIVAULT_DIR ───────────────────
# Use --norc --noprofile so .bashrc doesn't reset PATH (which would hide the
# venv podman-compose we just installed above).
cd "$RIVAULT_DIR"
exec bash --norc --noprofile ./serve-model.slurm
