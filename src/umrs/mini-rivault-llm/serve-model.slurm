#!/bin/bash

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    _resolved_script="$(readlink -f "$SCRIPT_PATH" 2>/dev/null || true)"
    if [[ -n "$_resolved_script" ]]; then
        SCRIPT_PATH="$_resolved_script"
    fi
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ─── GOMAXPROCS: prevent Go-based services from NPROC exhaustion ─────────────
# Default to 1 if not already set by the outer wrapper or environment.
export GOMAXPROCS="${GOMAXPROCS:-1}"

load_site_env_defaults() {
    local python_bin
    local rendered_exports

    python_bin="$(command -v python3 || true)"
    [[ -n "$python_bin" ]] || return 0

    rendered_exports="$({
        PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            UMRS_SITE_CONFIG="${UMRS_SITE_CONFIG:-}" \
            "$python_bin" - <<'PY'
import os
import shlex

from umrs.orchestrator.site_config import load_site_config

try:
    path = os.environ.get("UMRS_SITE_CONFIG") or None
    cfg = load_site_config(path)
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

mask_secret_state() {
    if [[ -n "${1:-}" ]]; then
        printf '%s' '<set>'
    else
        printf '%s' '<unset>'
    fi
}

mkdir -p "${HOME}/.config/containers/"

cat <<EOF > "${HOME}/.config/containers/storage.conf"
[storage]
driver="overlay"
runroot = "${XDG_RUNTIME_DIR}/containers/storage"
rootless_storage_path="/var/tmp/podman_${USER}/storage"
graphroot = "/var/tmp/podman_${USER}/storage"           # Note: local non-NFS path so overlay works and images persist across jobs
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="xattr_permissions=2,threaded=1"
EOF

uuid=$(cat /proc/sys/kernel/random/uuid)

# ─── Derive vLLM model-path variables from orchestrator env ──────────────────
# VLLM_MODEL_PATH (host filesystem path) is set by the orchestrator when it
# provisions an endpoint for a registered model.  When absent, fall back to the
# legacy HuggingFace-cache layout.
_vllm_host_model_path="${VLLM_HOST_MODEL_PATH:-${VLLM_MODEL_PATH:-}}"
if [[ -z "$_vllm_host_model_path" ]]; then
    _hf_cache="${VLLM_PATH_TO_HUGGINGFACE:-./models/vllm/huggingface}"
    _model_id="${VLLM_MODEL:-Qwen/Qwen3-0.6B}"
    _vllm_host_model_path="${_hf_cache}/hub/models--$(echo "$_model_id" | tr '/' '--')/snapshots"
fi
_vllm_container_model_path="${VLLM_CONTAINER_MODEL_PATH:-/models/$(basename "$_vllm_host_model_path")}"
_vllm_model_ref="${VLLM_MODEL_REF:-${_vllm_container_model_path}}"
_vllm_served_model_name="${VLLM_SERVED_MODEL_NAME:-${VLLM_MODEL:-Qwen/Qwen3-0.6B}}"

cat <<EOF > "./env.${uuid}"
###MANDATORY
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen3-0.6B}
VLLM_PATH_TO_HUGGINGFACE=${VLLM_PATH_TO_HUGGINGFACE:-./models/vllm/huggingface}
SAM3_MODEL=${SAM3_MODEL:-facebook/sam2-hiera-small}
VISION_MODEL_PATH=${VISION_MODEL_PATH:-./models/sam3}
SAM3_MODEL_PATH=${SAM3_MODEL_PATH:-./models/sam3}
HMAE_MODEL_PATH=${HMAE_MODEL_PATH:-./models/hmae}
HMAE_CONFIG_PATH=${HMAE_CONFIG_PATH:-./models/hmae/infer_hmae_vit_b.yaml}
HMAE_CHECKPOINT=${HMAE_CHECKPOINT:-./models/hmae/latest_checkpoint.pth}
VLLM_HOST_MODEL_PATH=${_vllm_host_model_path}
VLLM_CONTAINER_MODEL_PATH=${_vllm_container_model_path}
VLLM_MODEL_REF=${_vllm_model_ref}
VLLM_SERVED_MODEL_NAME=${_vllm_served_model_name}
VLLM_TP=${VLLM_TP:-1}						# tensor parallelism
VLLM_PP=${VLLM_PP:-1}						# pipeline parallelism
# R-1 (Phase 2k): honor the orchestrator-allocated port (LITELLM_PORT) over
# any pre-existing NGINX_LITELLM_DOCKER_PORT, and only fall back to a random
# pick when neither is set. PortManager in UMRS coordinates LITELLM_PORT, so
# preferring it eliminates the silent collision risk between concurrent jobs.
NGINX_LITELLM_DOCKER_PORT=${LITELLM_PORT:-${NGINX_LITELLM_DOCKER_PORT:-$(shuf -i 1025-65535 -n 1)}}
# R-3 (Phase 2k): 256-bit entropy for keys instead of 64. Note that
# LITELLM_SALT_KEY is generated per-job here because LiteLLM's Postgres state
# is currently tmpfs-only; if/when persistence is added, the salt MUST come
# from a stable secret store or all encrypted columns become unreadable.
LITELLM_SALT_KEY=${LITELLM_SALT_KEY:-$(openssl rand -hex 32)}
LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:-$(openssl rand -hex 32)}
VLLM_API_KEY=${VLLM_API_KEY:-$(openssl rand -hex 32)}
MCPO_API_KEY=${MCPO_API_KEY:-$(openssl rand -hex 32)}
UMRS_S3_BUCKET_HOST=${UMRS_S3_BUCKET_HOST:-${SLURM_SUBMIT_HOST}}		# httpforever.com
UMRS_S3_BUCKET_PORT=${UMRS_S3_BUCKET_PORT:-9000}
PRIV_IP1TO3=${PRIV_IP1TO3:-192.168.4}
CONTAINER_RUNTIME_DNS=${CONTAINER_RUNTIME_DNS:-/etc/resolv.conf}			# Note: docker uses 127.0.0.11 while podman forces me to use /etc/resolv.conf
UUID=00000000-0000-0000-0000-000000000000
RIVAULT_USERNS_MODE=${RIVAULT_USERNS_MODE:-keep-id:uid=0,gid=0}
###OPTIONAL
VLLM_DOCKER_TAG=${VLLM_DOCKER_TAG:-v0.20.0}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.95}
VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER:-hermes}
#VLLM_MODEL_TIKTOKEN=
#VLLM_CPU_KVCACHE_SPACE=
#VLLM_REPLICAS=
#VLLM_CPU_LIMIT=
#VLLM_MEM_LIMIT=
#MCPO_DOCKER_TAG=
#MCPO_CPU_LIMIT=
#MCPO_MEM_LIMIT=
#NGINX_DOCKER_TAG=
#NGINX_OUT_IP4=
#NGINX_REPLICAS=
#NGINX_CPU_LIMIT=
#NGINX_MEM_LIMIT=
#COREDNS_DOCKER_TAG=
#COREDNS_IP4=
#COREDNS_REPLICAS=
#COREDNS_CPU_LIMIT=
#COREDNS_MEM_LIMIT=
#LITELLM_DOCKER_TAG=
#LITELLM_POSTGRES_USER=
#LITELLM_POSTGRES_PASSWORD=
#LITELLM_POSTGRES_DB=
#LITELLM_MAX_REQUESTS_BEFORE_RESTART=
#LITELLM_REPLICAS=
#LITELLM_CPU_LIMIT=
#LITELLM_MEM_LIMIT=
#POSTGRES_VERSION=
#POSTGRES_REPLICAS=
#POSTGRES_CPU_LIMIT=
#POSTGRES_MEM_LIMIT=
#GENAI_TZ=
VLLM_NCCL_MEM_LIMIT=8G
VLLM_RCCL_MEM_LIMIT=8G
EOF

GPUoverride=()
_GPU_OVERRIDE_FILE=""
if [[ -n "${RIVAULT_GPU_OVERRIDE_FILE:-}" && -f "${RIVAULT_GPU_OVERRIDE_FILE}" ]]; then
    _GPU_OVERRIDE_FILE="${RIVAULT_GPU_OVERRIDE_FILE}"
    # Outer wrapper already validated GPU presence and generated the override;
    # trust it directly — nvidia-smi may be invisible inside the cgroup/userns
    # that the wrapper sets up.
    GPUoverride+=("--file" "${_GPU_OVERRIDE_FILE}")
    echo "[rivault] GPU override provided by wrapper: ${_GPU_OVERRIDE_FILE}"
elif /bin/nvidia-smi --list-gpus 2>/dev/null | /bin/grep -qi '^GPU 0:' ; then
    GPUoverride+=("--file" "./x_nvidia-override.yaml")
    echo "[rivault] NVIDIA GPU detected via nvidia-smi"
elif /bin/rocm-smi -i 2>/dev/null | /bin/grep -qi '^GPU\[0\]' ; then
    GPUoverride+=("--file" "./x_rocm-override.yaml")
    echo "[rivault] ROCm GPU detected via rocm-smi"
fi

# Ensure podman-compose >= 1.2.0 (needed for --profile support)
_pc_ok=false
if command -v podman-compose >/dev/null 2>&1; then
    _pc_ver="$(podman-compose -v 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    if [[ -n "$_pc_ver" ]] && printf '%s\n%s\n' "1.2.0" "$_pc_ver" | sort -V | head -1 | grep -q '^1\.2\.0$'; then
        _pc_ok=true
    else
        echo "[rivault] podman-compose ${_pc_ver:-unknown} too old (need >= 1.2.0)"
    fi
fi
if ! $_pc_ok; then
    echo "[rivault] Installing podman-compose >= 1.2.0 ..."
    pip3 install --quiet 'podman-compose>=1.2.0' 2>/dev/null || \
        pip3 install --user --quiet 'podman-compose>=1.2.0' 2>/dev/null || \
        { python3 -m venv --system-site-packages /tmp/pc-venv && /tmp/pc-venv/bin/pip install --quiet 'podman-compose>=1.2.0' && export PATH="/tmp/pc-venv/bin:$PATH"; }
    echo "[rivault] podman-compose: $(podman-compose -v 2>/dev/null)"
fi

## download model example (LLM for vLLM)
#mkdir -p ./models/vllm/huggingface/modules
#pip3 install huggingface_hub
#python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"Qwen/Qwen3-0.6B\", ignore_patterns=[\"metal/*\", \"original/*\"], cache_dir=\"./models/vllm/huggingface/hub\")"
#ls -la ./models/vllm/huggingface/modules/../hub/

# pre-download SAM3 image embedding model
########_sam3_model=$(/bin/grep '^SAM3_MODEL=' "./env.${uuid}" | /bin/cut -d'=' -f2)
########_vision_path=$(/bin/grep '^VISION_MODEL_PATH=' "./env.${uuid}" | /bin/cut -d'=' -f2)
########if [[ -n "$_sam3_model" && ! -d "${_vision_path}" ]]; then
########	echo "[rivault] Downloading SAM3 model '${_sam3_model}' to '${_vision_path}'..."
########	mkdir -p "${_vision_path}"
########	pip3 install -q huggingface_hub
########	python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"${_sam3_model}\", local_dir=\"${_vision_path}\")"
########	echo "[rivault] SAM3 model download complete."
########else
########	echo "[rivault] SAM3 model already present at '${_vision_path}', skipping download."
########fi

# fix some cgroup/podman stuff on some nodes — full systemd bootstrap
# from upstream reference. Needed on nodes where user session is not pre-started.
loginctl enable-linger $(whoami) 2>/dev/null || true
if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
fi
# Only attempt systemd user session bootstrap when BOTH systemctl and systemd
# binaries exist. On ai-l40s / x86_64 nodes systemctl is present but systemd
# is not, causing "systemd: command not found" and leaving the podman socket
# non-existent. Setting DOCKER_HOST to a missing socket then causes
# podman-compose to fail with "External network does not exists".
if command -v systemctl >/dev/null 2>&1 && command -v systemd >/dev/null 2>&1; then
    systemd --user &
    sleep 5
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user enable --now podman.socket 2>/dev/null || true
    # Only export DOCKER_HOST when the socket was actually created
    _podman_sock="unix:///run/user/$(id -u)/podman/podman.sock"
    if [[ -S "/run/user/$(id -u)/podman/podman.sock" ]]; then
        export DOCKER_HOST="${_podman_sock}"
    fi
fi

echo "[rivault] Launch context:"
echo "[rivault]   site config: ${UMRS_SITE_CONFIG:-<none>}"
echo "[rivault]   profiles: ${RIVAULT_PROFILES:-harness vllm}"
echo "[rivault]   script dir: ${SCRIPT_DIR}"
echo "[rivault]   compose files: ./serve-model.yaml ; ./userdata-access.yaml"
echo "[rivault]   litellm port: ${NGINX_LITELLM_DOCKER_PORT:-auto}"
echo "[rivault]   litellm key: $(mask_secret_state "${LITELLM_MASTER_KEY:-}")"
echo "[rivault]   hf cache: ${VLLM_PATH_TO_HUGGINGFACE:-./models/vllm/huggingface}"
echo "[rivault]   sam3 path: ${SAM3_MODEL_PATH:-./models/sam3}"
echo "[rivault]   hmae path: ${HMAE_MODEL_PATH:-./models/hmae}"
echo "[rivault]   gpu override: ${_GPU_OVERRIDE_FILE:-embedded default}"

# ─── Recover corrupt rootless podman state ──────────────────────────────────
# When a prior Slurm job ends, its $XDG_RUNTIME_DIR is cleaned by Slurm but
# podman's GraphRoot in /var/tmp keeps stale pause-process references, causing
# "invalid internal status, try resetting the pause process" errors.
# Fix: stop stale systemd scopes, delete libpod state DB (not the overlays —
# those have UID-mapped files that rm can't remove), then migrate.
#
# SKIP when running from the deploy wrapper — the wrapper already handles
# cleanup before delegating here (RIVAULT_WRAPPER_CLEANUP=1 is set by it).
if [[ "${RIVAULT_WRAPPER_CLEANUP:-0}" != "1" ]] && ! podman info >/dev/null 2>&1; then
    echo "[rivault] WARN: podman state corrupt — recovering..."
    _graph_parent="/var/tmp/podman_$(whoami)"
    # 1. Stop stale systemd scopes left by prior Slurm jobs
    for _scope in $(systemctl --user list-units --type=scope --no-legend 2>/dev/null \
                    | grep podman | awk '{print $1}'); do
        systemctl --user stop "$_scope" 2>/dev/null || true
    done
    # 2. Kill stale container runtime processes
    for _proc in conmon crun catatonit; do
        pids=$(pgrep -u "$(whoami)" -x "$_proc" 2>/dev/null || true)
        [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    done
    # 3. Delete libpod state DB and container metadata (NOT overlays — they
    #    have UID-mapped files owned by sub-UIDs that plain rm can't remove)
    rm -rf "${_graph_parent}/storage/libpod" 2>/dev/null || true
    rm -rf "${_graph_parent}/storage/overlay-containers" 2>/dev/null || true
    rm -rf "${_graph_parent}/storage/networks" 2>/dev/null || true
    rm -rf "${_graph_parent}/storage/volumes" 2>/dev/null || true
    rm -f  "${_graph_parent}/storage/db.sql" 2>/dev/null || true
    rm -f  "${_graph_parent}/storage/defaultNetworkBackend" 2>/dev/null || true
    rm -f "/run/user/$(id -u)/systemd/transient/podman-pause-"*.scope 2>/dev/null || true
    systemctl --user daemon-reload 2>/dev/null || true
    # 3b. Clear /tmp/run-user-* runtime state (podman uses this on some nodes
    #     for container network and IPAM state outside XDG_RUNTIME_DIR)
    rm -rf "/tmp/run-user-$(id -u)/containers" 2>/dev/null || true
    rm -rf "/tmp/run-user-$(id -u)/libpod/tmp" 2>/dev/null || true
    # 4. Migrate to rebuild internal state
    podman system migrate 2>/dev/null || true
    sleep 1
    if podman info >/dev/null 2>&1; then
        echo "[rivault] Podman state recovered."
    else
        echo "[rivault] ERROR: podman recovery failed — trying full reset..."
        podman system reset --force 2>/dev/null || true
        sleep 2
        if ! podman info >/dev/null 2>&1; then
            echo "[rivault] FATAL: podman is unusable after full reset. Cannot proceed."
            exit 1
        fi
    fi
fi

# start
# Tear down any leftovers from previous jobs: remove all containers that are
# still connected to genai-private-net (handles unnamed/bracketed-name podman
# quirks), then remove the network itself so IPAM starts from a clean slate.
podman ps -qa --filter network=genai-private-net --format '{{.ID}}' 2>/dev/null | \
	xargs -r podman rm -f 2>/dev/null || true
podman ps -qa --filter 'name=rivault_' --format '{{.ID}}' 2>/dev/null | \
	xargs -r podman rm -f 2>/dev/null || true
podman network rm -f genai-private-net 2>/dev/null || true

# Verify the network is actually gone (belt-and-suspenders)
# When containers from a prior Slurm job are orphaned (crun state files gone
# after $XDG_RUNTIME_DIR cleanup), podman can't stop them normally.
# Fix: cleanup --rm clears metadata; pkill conmon kills any survivors.
if podman network exists genai-private-net 2>/dev/null; then
	echo "[rivault] WARN: genai-private-net still exists; cleaning orphaned containers..."
	pkill -9 conmon 2>/dev/null || true
	sleep 1
	podman container cleanup --rm --all 2>/dev/null || true
	podman rm -af 2>/dev/null || true
	podman network rm -f genai-private-net 2>/dev/null || true
	if podman network exists genai-private-net 2>/dev/null; then
		echo "[rivault] ERROR: cannot remove genai-private-net"; exit 1
	fi
fi

# Create a fresh genai-private-net, trying candidate subnets until one works.
# Prefer a configurable subnet first and avoid reusing 192.168.4 by default,
# because stale rootless IPAM allocations on crashed jobs have repeatedly
# poisoned that range on RCCS Cloud.
# NOTE: do NOT use --internal here; rootful podman's conmon port-forwarder
# cannot relay data across an internal bridge (TCP connects but hangs).
# The published port (NGINX_LITELLM_DOCKER_PORT) must be reachable from the
# host so the SSH tunnel from the control-plane container can reach it.
_net_created=0
_preferred_ip1to3="${PRIV_IP1TO3:-172.16.4}"
_candidate_subnets=()
for _candidate in "${_preferred_ip1to3}" '172.16.4' '10.0.4' '192.168.4'; do
    if [[ " ${_candidate_subnets[*]} " == *" ${_candidate} "* ]]; then
        continue
    fi
    _candidate_subnets+=("${_candidate}")
done
for IP1TO3 in "${_candidate_subnets[@]}"; do
	# Note: --disable-dns needed because infra lacks dns, but we need fixed ips
	if podman network create --disable-dns --subnet "${IP1TO3}.0/24" genai-private-net 2>/dev/null; then
		_net_created=1
		break
	fi
done
if [ "$_net_created" -ne 1 ]; then
	echo "[rivault] ERROR: failed to create genai-private-net with any candidate subnet"
	exit 1
fi
echo "[rivault] genai-private-net created with subnet ${IP1TO3}.0/24"

# Verify network is visible to podman (catches netavark storage issues)
if ! podman network exists genai-private-net 2>/dev/null; then
	echo "[rivault] WARN: network exists check failed after create; running system migrate..."
	podman system migrate 2>/dev/null || true
	if ! podman network exists genai-private-net 2>/dev/null; then
		echo "[rivault] ERROR: genai-private-net not visible after migrate"
		echo "[rivault] DEBUG: network ls output:"
		podman network ls 2>&1
		echo "[rivault] DEBUG: network backend:"
		cat "/var/tmp/podman_$(whoami)/storage/defaultNetworkBackend" 2>/dev/null
		echo ""
		echo "[rivault] DEBUG: network files:"
		find "/var/tmp/podman_$(whoami)/storage" -name "*genai*" 2>/dev/null
		find "${XDG_DATA_HOME:-$HOME/.local/share}/containers/storage" -name "*genai*" 2>/dev/null
		find "$HOME/.config/containers/networks" -name "*genai*" 2>/dev/null
		exit 1
	fi
fi

# ─── SSH tunnel: forward MinIO from login node to compute node ────────────────
# The BentoML container accesses S3 via minio-host:19000 (-> host-gateway:19000).
# Bind to 0.0.0.0 so containers (via host-gateway) can reach MinIO,
# and use port 19000 (BENTOML_MINIO_HOST_PORT) as the local port.
UMRS_S3_TUNNEL_PORT=${UMRS_S3_TUNNEL_PORT:-9000}
UMRS_S3_TUNNEL_HOST=${UMRS_S3_TUNNEL_HOST:-${SLURM_SUBMIT_HOST:-localhost}}
_S3_LOCAL_PORT="${BENTOML_MINIO_HOST_PORT:-19000}"
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

echo "[rivault] Setting up SSH tunnels for S3 (MinIO) access..."
# Tunnel 1: 0.0.0.0:19000 → login:9000 (for containers via host-gateway)
start_managed_ssh_tunnel "0.0.0.0:${_S3_LOCAL_PORT}:127.0.0.1:${UMRS_S3_TUNNEL_PORT}" "${UMRS_S3_TUNNEL_HOST}" \
    && echo "[rivault] SSH tunnel: 0.0.0.0:${_S3_LOCAL_PORT} → ${UMRS_S3_TUNNEL_HOST}:${UMRS_S3_TUNNEL_PORT} (for containers)" \
    || echo "[rivault] WARNING: SSH tunnel (container) to ${UMRS_S3_TUNNEL_HOST}:${UMRS_S3_TUNNEL_PORT} failed"
# Tunnel 2: localhost:9000 → login:9000 (for host-level presigned URL access)
start_managed_ssh_tunnel "127.0.0.1:${UMRS_S3_TUNNEL_PORT}:127.0.0.1:${UMRS_S3_TUNNEL_PORT}" "${UMRS_S3_TUNNEL_HOST}" \
    && echo "[rivault] SSH tunnel: localhost:${UMRS_S3_TUNNEL_PORT} → ${UMRS_S3_TUNNEL_HOST}:${UMRS_S3_TUNNEL_PORT} (for host)" \
    || echo "[rivault] WARNING: SSH tunnel (host) already bound or failed (OK if port ${UMRS_S3_TUNNEL_PORT} == ${_S3_LOCAL_PORT})"

# ─── Verify SSH tunnel liveness ─────────────────────────────────────────────
# Wait briefly then probe the forwarded port to catch silent tunnel failures
# early rather than discovering them on the first inference request.
_tunnel_ok=0
for _attempt in 1 2 3; do
    sleep 2
    if bash -c "echo > /dev/tcp/127.0.0.1/${_S3_LOCAL_PORT}" 2>/dev/null; then
        _tunnel_ok=1
        break
    fi
    echo "[rivault] SSH tunnel probe attempt ${_attempt}/3 failed, retrying..."
done
if [ "$_tunnel_ok" -eq 1 ]; then
    echo "[rivault] SSH tunnel verified: port ${_S3_LOCAL_PORT} is reachable"
else
    echo "[rivault] WARNING: SSH tunnel port ${_S3_LOCAL_PORT} not reachable after 3 probes."
    echo "[rivault]          S3/MinIO access may fail. Check tunnel host and login node."
fi

# SELinux context for bind-mounted dirs (safe no-op when SELinux is disabled)
for _sel_path in \
    "./nginx_fs/umrs.conf" \
    "./litellm_fs/config.yaml" \
    "./nginx_fs/out-conf.d/" \
    "./coredns_fs/" \
    "./bento/"; do
    [[ -e "$_sel_path" ]] && chcon -Rt svirt_sandbox_file_t "$_sel_path" 2>/dev/null || true
done

sed -i  -e "s@^\(PRIV_IP1TO3=\).*@\1${IP1TO3}@g" \
	-e "s@^\(UUID=\).*@\1${uuid}@g" \
	"./env.${uuid}" \
	|| { echo "[rivault] FATAL: failed to rewrite PRIV_IP1TO3/UUID in env.${uuid}"; exit 1; }

# R-4 (Phase 2k): every placeholder substitution below is REQUIRED for the
# downstream container to start. A silent grep miss (e.g. an env line missing
# a value) historically surfaces only when vLLM/LiteLLM fail to parse their
# config 2 minutes later. Validate substitution outputs before committing.
_substitute_required() {
    local label="$1" out="$2" placeholder="$3"
    if [[ ! -s "$out" ]]; then
        echo "[rivault] FATAL: ${label} substitution produced empty file (${out})"
        exit 1
    fi
    if grep -q "${placeholder}" "$out" 2>/dev/null; then
        echo "[rivault] FATAL: ${label} still contains unresolved placeholder '${placeholder}'"
        exit 1
    fi
}

if ! sed -e "s@UMRS_S3_BUCKET_PORT@$(/bin/grep '^UMRS_S3_BUCKET_PORT=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	"./nginx_fs/out-nginx.conf" > "./nginx_fs/out-nginx.conf.${uuid}"; then
	echo "[rivault] FATAL: nginx out-nginx.conf substitution failed"; exit 1
fi
_substitute_required "out-nginx.conf" "./nginx_fs/out-nginx.conf.${uuid}" "UMRS_S3_BUCKET_PORT"

if ! sed -e "s@UMRS_S3_BUCKET_HOST@$(/bin/grep '^UMRS_S3_BUCKET_HOST=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	-e "s@CONTAINER_RUNTIME_DNS@$(/bin/grep '^CONTAINER_RUNTIME_DNS=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	"./coredns_fs/coredns.conf" > "./coredns_fs/coredns.conf.${uuid}"; then
	echo "[rivault] FATAL: coredns substitution failed"; exit 1
fi
_substitute_required "coredns.conf" "./coredns_fs/coredns.conf.${uuid}" "UMRS_S3_BUCKET_HOST"

if ! sed -e "s@VLLM_SERVED_MODEL_NAME_PLACEHOLDER@$(/bin/grep '^VLLM_SERVED_MODEL_NAME=' "./env.${uuid}" | awk -F'=' '{print $2}')@g" \
	-e "s@VLLM_MODEL_PLACEHOLDER@$(/bin/grep '^VLLM_MODEL=' "./env.${uuid}" | awk -F'=' '{print $2}')@g" \
	"./litellm_fs/config.yaml" > "./litellm_fs/config.yaml.${uuid}"; then
	echo "[rivault] FATAL: litellm config substitution failed"; exit 1
fi
_substitute_required "litellm/config.yaml" "./litellm_fs/config.yaml.${uuid}" "VLLM_SERVED_MODEL_NAME_PLACEHOLDER"
_substitute_required "litellm/config.yaml" "./litellm_fs/config.yaml.${uuid}" "VLLM_MODEL_PLACEHOLDER"

# Profiles to activate (override via RIVAULT_PROFILES env var)
# If site defaults were loaded above, runtime.profiles wins. Otherwise the
# generic source fallback remains "harness vllm". For vision models add
# "vision-bentoml":
#   export RIVAULT_PROFILES="harness vllm vision-bentoml"
RIVAULT_PROFILES=${RIVAULT_PROFILES:-"harness vllm"}
_profile_flags=""
for _p in $RIVAULT_PROFILES; do
	_profile_flags="${_profile_flags} --profile ${_p}"
done

_cleanup_started=0
RIVAULT_CONTROL_DIR="${RIVAULT_CONTROL_DIR:-}"
RIVAULT_STOP_REQUEST_FILE=""
if [[ -n "${RIVAULT_CONTROL_DIR}" && -n "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "${RIVAULT_CONTROL_DIR}" 2>/dev/null || true
    RIVAULT_STOP_REQUEST_FILE="${RIVAULT_CONTROL_DIR%/}/stop-${SLURM_JOB_ID}.request"
fi

cleanup_stack() {
    local reason="${1:-exit}"
    local rc="${2:-0}"

    if [[ "${_cleanup_started}" -ne 0 ]]; then
        return
    fi
    _cleanup_started=1
    trap - EXIT TERM INT

    echo "[rivault] Cleaning up stack (${reason})..."
    cleanup_ssh_tunnels
    podman-compose --project-name "exit" --in-pod false \
               --env-file "./env.${uuid}" \
               --file "./userdata-access.yaml" \
               down >/dev/null 2>&1 || true
    podman-compose --project-name "rivault" --in-pod false \
               --env-file "./env.${uuid}" \
               --file "./serve-model.yaml" ${GPUoverride[@]} \
               down >/dev/null 2>&1 || true
    podman network rm -f genai-private-net >/dev/null 2>&1 || true
    # Clear /tmp/run-user-* runtime state left by podman
    rm -rf "/tmp/run-user-$(id -u)/containers" 2>/dev/null || true
    rm -rf "/tmp/run-user-$(id -u)/libpod/tmp" 2>/dev/null || true
    if [[ -n "${RIVAULT_STOP_REQUEST_FILE}" ]]; then
        rm -f "${RIVAULT_STOP_REQUEST_FILE}" >/dev/null 2>&1 || true
    fi
    if [[ "${RIVAULT_GPU_OVERRIDE_GENERATED:-0}" == "1" && -n "${RIVAULT_GPU_OVERRIDE_FILE:-}" ]]; then
        rm -f "${RIVAULT_GPU_OVERRIDE_FILE}" >/dev/null 2>&1 || true
    fi
    rm -f "./env.${uuid}" \
          "./nginx_fs/out-nginx.conf.${uuid}" \
          "./coredns_fs/coredns.conf.${uuid}" \
          "./litellm_fs/config.yaml.${uuid}" >/dev/null 2>&1 || true
    return "$rc"
}

trap 'cleanup_stack "signal:TERM" 143; exit 143' TERM
trap 'cleanup_stack "signal:INT" 130; exit 130' INT
trap 'cleanup_stack "exit" "$?"' EXIT

if ! podman-compose --project-name "rivault" --in-pod false \
         --env-file "./env.${uuid}" \
         --file "./serve-model.yaml" ${GPUoverride[@]} \
         ${_profile_flags} \
         up \
	       --remove-orphans --detach; then
	echo "[rivault] ERROR: failed to start rivault compose stack"
	exit 1
fi

if ! podman-compose --project-name "exit" --in-pod false \
         --env-file "./env.${uuid}" \
         --file "./userdata-access.yaml" \
         --profile harness \
         up \
	       --remove-orphans --detach; then
	echo "[rivault] ERROR: failed to start exit compose stack"
	exit 1
fi

ip a

# ─── Vision service readiness gate ──────────────────────────────────────────
# If vision-bentoml profile is active, wait for BentoML to actually be ready
# (port 3000 bound AND /readyz returning 200) before declaring the stack up.
# This catches cases where the container starts but the service fails to load.
if echo "${RIVAULT_PROFILES}" | grep -q "vision-bentoml"; then
    _BENTOML_IP="${PRIV_IP1TO3:-10.0.0}.${BENTOML_VISION_IP4:-35}"
    _VISION_READY=0
    _VISION_READY_URL=""
    _VISION_TIMEOUT=${RIVAULT_VISION_READY_TIMEOUT:-180}  # 3 min default
    _VISION_ELAPSED=0
    _VISION_READY_URLS=(
        "http://127.0.0.1:${BENTOML_VISION_DOCKER_PORT:-3001}/readyz"
        "http://127.0.0.1:${NGINX_LITELLM_DOCKER_PORT:-8001}/vision/readyz"
        "http://${_BENTOML_IP}:3000/readyz"
    )
    echo "[rivault] Waiting for BentoML vision service readiness (timeout=${_VISION_TIMEOUT}s)..."
    echo "[rivault] Readiness candidates: ${_VISION_READY_URLS[*]}"
    while [ "$_VISION_ELAPSED" -lt "$_VISION_TIMEOUT" ]; do
        # Check if container is still running
        if ! podman ps --filter "name=bentoml-vision" --format '{{.Names}}' 2>/dev/null | grep -q bentoml-vision; then
            echo "[rivault] WARNING: bentoml-vision container is not running. Checking restart..."
            sleep 5
            _VISION_ELAPSED=$(( _VISION_ELAPSED + 5 ))
            continue
        fi
        # Probe readiness endpoints through the published direct proxy first.
        # Some rootless podman setups do not route the host cleanly to the
        # private container IP, while the host-published 3001/8001 paths are
        # already what the launcher and clients use.
        for _url in "${_VISION_READY_URLS[@]}"; do
            if curl -fsS --connect-timeout 2 --max-time 5 -o /dev/null "${_url}" 2>/dev/null; then
                _VISION_READY=1
                _VISION_READY_URL="${_url}"
                break
            fi
        done
        if [ "$_VISION_READY" -eq 1 ]; then
            break
        fi
        sleep 5
        _VISION_ELAPSED=$(( _VISION_ELAPSED + 5 ))
    done
    if [ "$_VISION_READY" -eq 1 ]; then
        echo "[rivault] BentoML vision service is READY via ${_VISION_READY_URL} (took ~${_VISION_ELAPSED}s)"
    else
        echo "[rivault] ERROR: BentoML vision service did NOT become ready within ${_VISION_TIMEOUT}s"
        echo "[rivault]        Checked endpoints: ${_VISION_READY_URLS[*]}"
        echo "[rivault]        Container logs (last 50 lines):"
        podman logs --tail 50 rivault_bentoml-vision_1 2>&1 | sed 's/^/[rivault]          /' || true
        # R-5 (Phase 2k): fail the Slurm job rather than letting the registry
        # advertise a half-up stack. The cleanup trap will tear down the
        # compose stack; UMRS durable retry will create a new attempt.
        if [[ "${RIVAULT_VISION_STRICT:-1}" == "1" ]]; then
            echo "[rivault]        RIVAULT_VISION_STRICT=1 → exiting non-zero so UMRS retries cleanly."
            exit 2
        else
            echo "[rivault]        RIVAULT_VISION_STRICT=0 → continuing; vision routes will return 503."
        fi
    fi
fi

# Keep job alive while containers are running; exit on idle timeout (safety net).
# Slurm cancellation or an explicit signal should now run the cleanup trap
# above so the compose stack is torn down promptly.
RIVAULT_JOB_TIMEOUT=${RIVAULT_JOB_TIMEOUT:-7200}   # default 2 h; override via sbatch --export
INTERVAL=${RIVAULT_JOB_POLL_INTERVAL:-5}
ELAPSED=0
echo "[rivault] Stack is up. Monitoring containers (timeout=${RIVAULT_JOB_TIMEOUT}s)..."
while [ "$ELAPSED" -lt "$RIVAULT_JOB_TIMEOUT" ]; do
    if [[ -n "${RIVAULT_STOP_REQUEST_FILE}" && -f "${RIVAULT_STOP_REQUEST_FILE}" ]]; then
        echo "[rivault] Graceful stop requested via ${RIVAULT_STOP_REQUEST_FILE}. Exiting monitor loop."
        exit 0
    fi
    RUNNING=$(podman ps -q --filter "network=genai-private-net" 2>/dev/null | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "[rivault] All containers stopped. Exiting."
        break
    fi
    sleep "$INTERVAL"
    ELAPSED=$(( ELAPSED + INTERVAL ))
done
if [ "$ELAPSED" -ge "$RIVAULT_JOB_TIMEOUT" ]; then
	echo "[rivault] Timeout ${RIVAULT_JOB_TIMEOUT}s reached. Exiting."
	exit 124
fi

# test
#curl -H "Authorization: Bearer $(/bin/grep '^LITELLM_MASTER_KEY=' "./env.${uuid}" |/bin/cut -d'=' -f2)" \
#        "http://localhost:$(/bin/grep '^NGINX_LITELLM_DOCKER_PORT=' "./env.${uuid}" |/bin/cut -d'=' -f2)/v1/models"
#podman exec -it $(podman ps -a|/bin/grep vllm|cut -d' ' -f1) curl -v httpforever.com:9000	#XXX: need 'http://$host:80' in out-nginx.conf

# stop
#podman-compose --env-file "./env.${uuid}" \
#               --file "./userdata-access.yaml" down
#podman-compose --env-file "./env.${uuid}" \
#               --file "./serve-model.yaml" ${GPUoverride[@]} down
#
#podman network rm -f genai-private-net
#
#rm -f "./env.${uuid}" "./nginx_fs/out-nginx.conf.${uuid}" "./coredns_fs/coredns.conf.${uuid}" "./litellm_fs/config.yaml.${uuid}"
