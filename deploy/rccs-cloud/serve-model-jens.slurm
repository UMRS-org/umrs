#!/bin/bash
# =============================================================================
# serve-model-jens.slurm — Jens VisionFrontend stack on Slurm
# =============================================================================
# Launches Jens's standalone VisionFrontend BentoML service (SAM3 + HMAE)
# with podman-compose on a Slurm compute node.
#
# This script is selected by VISION_STACK=jens in site.yaml or env vars.
# It reuses the same site_config.py for paths, partitions, and storage but
# runs the original Jens compose configuration rather than the UMRS generic
# mini-rivault-llm stack.
#
# Usage:
#   sbatch serve-model-jens.slurm          # from deploy/rccs-cloud/
#   VISION_STACK=jens bash serve-model-jens.slurm   # interactive
# =============================================================================
set -x

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    _resolved_script="$(readlink -f "$SCRIPT_PATH" 2>/dev/null || true)"
    if [[ -n "$_resolved_script" ]]; then
        SCRIPT_PATH="$_resolved_script"
    fi
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Load site.yaml defaults ──────────────────────────────────────────────
load_site_env_defaults() {
    local python_bin rendered_exports
    python_bin="$(command -v python3 || true)"
    [[ -n "$python_bin" ]] || return 0

    rendered_exports="$({
        PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            UMRS_SITE_CONFIG="${UMRS_SITE_CONFIG:-}" \
            "$python_bin" - <<'PY'
import os, shlex
from umrs.orchestrator.site_config import load_site_config
try:
    cfg = load_site_config(os.environ.get("UMRS_SITE_CONFIG") or None)
except FileNotFoundError:
    raise SystemExit(0)
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
load_site_env_defaults

# ── Jens stack root ──────────────────────────────────────────────────────
JENS_DIR="${JENS_DIR:-${REPO_ROOT}/tmp/internal_genAI}"
if [[ ! -d "$JENS_DIR" ]]; then
    echo "ERROR: Jens stack directory not found: $JENS_DIR" >&2
    exit 1
fi
cd "$JENS_DIR"

# ── Podman storage ──────────────────────────────────────────────────────
mkdir -p "${HOME}/.config/containers/"
cat <<EOF > "${HOME}/.config/containers/storage.conf"
[storage]
driver="overlay"
runroot = "${XDG_RUNTIME_DIR}/containers/storage"
rootless_storage_path="${XDG_RUNTIME_DIR}/containers/storage"
graphroot = "${XDG_RUNTIME_DIR}/containers/storage"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="xattr_permissions=2,threaded=1"
force_mask="700"
EOF

# ── Generate env file ───────────────────────────────────────────────────
uuid=$(cat /proc/sys/kernel/random/uuid)
S3_HOST="${UMRS_S3_TUNNEL_HOST:-sshforwarded}"
S3_PORT="${UMRS_S3_TUNNEL_PORT:-9010}"

cat <<EOF > "./env.${uuid}"
NGINX_LITELLM_DOCKER_PORT=$(shuf -i 1025-65535 -n 1)
LITELLM_SALT_KEY=$(openssl rand -hex 8)
LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:-$(openssl rand -hex 8)}
UMRS_S3_BUCKET_HOST=${S3_HOST}
UMRS_S3_BUCKET_PORT=${S3_PORT}
UMRS_S3_ACCESS_KEY=$(openssl rand -hex 8)
UMRS_S3_SECRET_KEY=$(openssl rand -hex 8)
PRIV_IP1TO3=192.168.4
CONTAINER_RUNTIME_DNS=/etc/resolv.conf
UUID=${uuid}
VLLM_NCCL_MEM_LIMIT=8G
VLLM_RCCL_MEM_LIMIT=8G
EOF

_cleanup_started=0
_ssh_tunnel_pids=()

start_managed_ssh_tunnel() {
    local bind_spec="$1"
    local remote_host="$2"
    local ssh_pid

    ssh -nNT -L "$bind_spec" "$remote_host" \
        -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
        </dev/null >/dev/null 2>&1 &
    ssh_pid=$!

    sleep 1
    if kill -0 "$ssh_pid" 2>/dev/null; then
        _ssh_tunnel_pids+=("$ssh_pid")
        return 0
    fi

    wait "$ssh_pid"
    return $?
}

cleanup_ssh_tunnels() {
    local ssh_pid

    if [[ ${#_ssh_tunnel_pids[@]} -eq 0 ]]; then
        return 0
    fi

    for ssh_pid in "${_ssh_tunnel_pids[@]}"; do
        kill "$ssh_pid" 2>/dev/null || true
    done
    sleep 1
    for ssh_pid in "${_ssh_tunnel_pids[@]}"; do
        if kill -0 "$ssh_pid" 2>/dev/null; then
            kill -9 "$ssh_pid" 2>/dev/null || true
        fi
    done
    for ssh_pid in "${_ssh_tunnel_pids[@]}"; do
        wait "$ssh_pid" 2>/dev/null || true
    done
    _ssh_tunnel_pids=()
}

cleanup_stack() {
    local reason="${1:-exit}"
    local rc="${2:-0}"

    if [[ "${_cleanup_started}" -ne 0 ]]; then
        return
    fi
    _cleanup_started=1
    trap - EXIT TERM INT

    echo "Cleaning up Jens stack (${reason})..."
    cleanup_ssh_tunnels
    podman-compose --project-name "exit" \
        --env-file "./env.${uuid}" --file "./userdata-access.yaml" \
        down >/dev/null 2>&1 || true
    podman-compose --project-name "rivault" \
        --env-file "./env.${uuid}" --file "./serve-model.yaml" "${GPUoverride[@]}" \
        down >/dev/null 2>&1 || true
    podman network rm -f genai-private-net >/dev/null 2>&1 || true
    rm -f "./env.${uuid}" \
        "./nginx_fs/out-nginx.conf.${uuid}" \
        "./coredns_fs/coredns.conf.${uuid}" >/dev/null 2>&1 || true
    return "$rc"
}

trap 'cleanup_stack "signal:TERM" 143; exit 143' TERM
trap 'cleanup_stack "signal:INT" 130; exit 130' INT
trap 'cleanup_stack "exit" "$?"' EXIT

# ── GPU detection ───────────────────────────────────────────────────────
GPUoverride=()
HAVE_GPU="none"
if /bin/rocm-smi -i 2>/dev/null | /bin/grep -qi '^GPU\[0\]'; then
    HAVE_GPU="rocm"
    [[ -f "./x_rocm-override.yaml" ]] && GPUoverride+=("--file" "./x_rocm-override.yaml")
elif /bin/nvidia-smi --list-gpus 2>/dev/null | /bin/grep -qi '^GPU 0:'; then
    HAVE_GPU="cuda"
    [[ -f "./x_nvidia-override.yaml" ]] && GPUoverride+=("--file" "./x_nvidia-override.yaml")
fi
echo "GPU: ${HAVE_GPU}"

# ── Podman / cgroup setup ──────────────────────────────────────────────
loginctl enable-linger "$(whoami)" 2>/dev/null || true
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
systemd --user &
sleep 5
systemctl --user daemon-reload 2>/dev/null || true
systemctl --user enable --now podman.socket 2>/dev/null || true
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"

# ── venv + podman-compose ──────────────────────────────────────────────
[ -n "${SLURM_JOB_PARTITION}" ] \
    && TDIR=".tmp/$(uname -m).${HAVE_GPU}/${SLURM_JOB_PARTITION}" \
    || TDIR=".tmp/$(uname -m).${HAVE_GPU}/$(hostname -s)"
mkdir -p "${TDIR}"
if [[ ! -d "./${TDIR}/venv" ]]; then
    python3 -m venv "./${TDIR}/venv"
fi
source "./${TDIR}/venv/bin/activate"
which podman-compose >/dev/null 2>&1 || pip3 install pyyaml==6.0.3 podman-compose==1.5.0

podman --version
podman-compose --version

# ── Network ────────────────────────────────────────────────────────────
for IP1TO3 in '192.168.4' '172.16.4' '10.0.4'; do
    podman network create --internal --disable-dns --subnet "${IP1TO3}.0/24" genai-private-net && break
done 2>/dev/null || true
sed -i  -e "s@^\(PRIV_IP1TO3=\).*@\1${IP1TO3}@g" \
        -e "s@^\(UUID=\).*@\1${uuid}@g" \
        "./env.${uuid}"

# ── DNS / nginx config templating ──────────────────────────────────────
_S3_HOST_FILE="$(/bin/grep '^UMRS_S3_BUCKET_HOST=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')"
_S3_PORT_FILE="$(/bin/grep '^UMRS_S3_BUCKET_PORT=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')"
RUNTIME_DNS="$(/bin/grep '^CONTAINER_RUNTIME_DNS=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')"

if [[ -f "./nginx_fs/out-nginx.conf" ]]; then
    sed -e "s@UMRS_S3_BUCKET_PORT@${_S3_PORT_FILE}@g" \
        "./nginx_fs/out-nginx.conf" > "./nginx_fs/out-nginx.conf.${uuid}"
fi
if [[ -f "./coredns_fs/coredns.conf" ]]; then
    sed -e "s@UMRS_S3_BUCKET_HOST@${_S3_HOST_FILE}@g" \
        -e "s@CONTAINER_RUNTIME_DNS@${RUNTIME_DNS}@g" \
        "./coredns_fs/coredns.conf" > "./coredns_fs/coredns.conf.${uuid}"
fi

# ── SSH tunnel for S3 (if sshforwarded) ────────────────────────────────
LOCAL_IP=
if [[ "x${_S3_HOST_FILE}" == "xsshforwarded" ]]; then
    LOCAL_IP=$(ip -br addr show nm-bond 2>/dev/null | /bin/awk '{print $3}' | /bin/cut -d/ -f1)
    if [[ -n "$LOCAL_IP" && -f "./coredns_fs/coredns.conf.${uuid}" ]]; then
        sed -i -e "s@LOCAL_IP_FOR_SSH_FORWARDING@${LOCAL_IP}@g" "./coredns_fs/coredns.conf.${uuid}"
    fi

    start_managed_ssh_tunnel "${LOCAL_IP}:${_S3_PORT_FILE}:localhost:${_S3_PORT_FILE}" rccscloud

    # Verify tunnel
    _TUNNEL_OK=0
    for _attempt in 1 2 3; do
        if (echo >/dev/tcp/"${LOCAL_IP}"/"${_S3_PORT_FILE}") 2>/dev/null; then
            _TUNNEL_OK=1; break
        fi
        sleep 2
    done
    if [[ $_TUNNEL_OK -eq 0 ]]; then
        echo "ERROR: SSH tunnel not reachable at ${LOCAL_IP}:${_S3_PORT_FILE}" >&2
        exit 1
    fi
    echo "SSH tunnel verified: ${LOCAL_IP}:${_S3_PORT_FILE}"
fi

# ── SELinux contexts ───────────────────────────────────────────────────
for ctx_path in \
    ./nginx_fs/umrs.conf \
    ./bento/hmaeinfer/ \
    ./bento/sam3infer/ \
    ./litellm_fs/config.yaml \
    ./nginx_fs/out-conf.d/ \
    "./nginx_fs/out-nginx.conf.${uuid}" \
    "./coredns_fs/coredns.conf.${uuid}"; do
    [[ -e "$ctx_path" ]] && chcon -Rt svirt_sandbox_file_t "$ctx_path" 2>/dev/null || true
done

# ── Image cache ────────────────────────────────────────────────────────
pdm() {
    local filter="${1}"
    podman images --sort repository --format "{{.ID}} {{.Repository}}" \
        | { [[ -n "$filter" ]] && /bin/grep "$filter" || /bin/cat; } \
        | /bin/awk '{print $1}'
}

if [[ -n "${REBUILD:-}" ]] || [[ ! -f "./${TDIR}/minirivault.tar" ]]; then
    rm -f "./${TDIR}/minirivault.tar"
    podman-compose --project-name "rivault" \
        --env-file "./env.${uuid}" --file "./serve-model.yaml" "${GPUoverride[@]}" \
        pull
    podman-compose --project-name "rivault" \
        --env-file "./env.${uuid}" --file "./serve-model.yaml" "${GPUoverride[@]}" \
        build
    podman save -m -o "./${TDIR}/minirivault.tar" $(pdm nginx) $(pdm litellm)
fi
if [[ -n "${REBUILD:-}" ]] || [[ ! -f "./${TDIR}/miniexit.tar" ]]; then
    rm -f "./${TDIR}/miniexit.tar"
    podman-compose --project-name "exit" \
        --env-file "./env.${uuid}" --file "./userdata-access.yaml" \
        pull
    podman-compose --project-name "exit" \
        --env-file "./env.${uuid}" --file "./userdata-access.yaml" \
        build
    podman save -m -o "./${TDIR}/miniexit.tar" $(pdm nginx) $(pdm coredns)
fi

# ── Start services ─────────────────────────────────────────────────────
podman load -i "./${TDIR}/minirivault.tar"
podman-compose --project-name "rivault" \
    --env-file "./env.${uuid}" --file "./serve-model.yaml" "${GPUoverride[@]}" \
    up --remove-orphans --detach

podman load -i "./${TDIR}/miniexit.tar"
podman-compose --project-name "exit" \
    --env-file "./env.${uuid}" --file "./userdata-access.yaml" \
    up --remove-orphans --detach

podman-compose --project-name "rivault" \
    --env-file "./env.${uuid}" --file "./serve-model.yaml" "${GPUoverride[@]}" ps
podman-compose --project-name "exit" \
    --env-file "./env.${uuid}" --file "./userdata-access.yaml" ps
podman network ls

# ── Readiness gate for VisionFrontend ──────────────────────────────────
_BENTOML_IP="${IP1TO3}.10"  # match Jens's compose network assignment
RIVAULT_VISION_READY_TIMEOUT="${RIVAULT_VISION_READY_TIMEOUT:-180}"

_bentoml_ready=0
_start_wait=$SECONDS
while (( SECONDS - _start_wait < RIVAULT_VISION_READY_TIMEOUT )); do
    if curl -sf "http://${_BENTOML_IP}:3000/readyz" >/dev/null 2>&1; then
        _bentoml_ready=1
        break
    fi
    sleep 10
done
if [[ $_bentoml_ready -eq 1 ]]; then
    echo "VisionFrontend ready in $(( SECONDS - _start_wait ))s"
else
    echo "WARNING: VisionFrontend not ready after ${RIVAULT_VISION_READY_TIMEOUT}s"
fi

# ── Report ─────────────────────────────────────────────────────────────
ip a
set +x
echo ""
echo "=== Jens VisionFrontend Stack ==="
echo "Host     : $(hostname -f)"
echo "GPU      : ${HAVE_GPU}"
echo "BentoML  : http://${_BENTOML_IP}:3000"
/bin/grep 'NGINX_LITELLM_DOCKER_PORT\|LITELLM_MASTER_KEY' "./env.${uuid}"
echo ""

sleep infinity

# ── Cleanup (manual) ──────────────────────────────────────────────────
# podman-compose --project-name "exit" --env-file "./env.${uuid}" --file "./userdata-access.yaml" down
# podman-compose --project-name "rivault" --env-file "./env.${uuid}" --file "./serve-model.yaml" ${GPUoverride[@]} down
# podman network rm -f genai-private-net
# rm -f "./env.${uuid}" "./nginx_fs/out-nginx.conf.${uuid}" "./coredns_fs/coredns.conf.${uuid}"
# ssh -O exit rccscloud
