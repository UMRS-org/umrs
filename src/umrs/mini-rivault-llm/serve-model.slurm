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
mountopt="nodev,metacopy=on"
EOF

uuid=$(cat /proc/sys/kernel/random/uuid)
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
VLLM_TP=${VLLM_TP:-1}						# tensor parallelism
VLLM_PP=${VLLM_PP:-1}						# pipeline parallelism
NGINX_LITELLM_DOCKER_PORT=${NGINX_LITELLM_DOCKER_PORT:-$(shuf -i 1025-65535 -n 1)}
LITELLM_SALT_KEY=${LITELLM_SALT_KEY:-$(openssl rand -hex 8)}
LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:-$(openssl rand -hex 8)}
VLLM_API_KEY=${VLLM_API_KEY:-$(openssl rand -hex 8)}
MCPO_API_KEY=${MCPO_API_KEY:-$(openssl rand -hex 8)}
UMRS_S3_BUCKET_HOST=${UMRS_S3_BUCKET_HOST:-${SLURM_SUBMIT_HOST}}		# httpforever.com
UMRS_S3_BUCKET_PORT=${UMRS_S3_BUCKET_PORT:-9000}
PRIV_IP1TO3=${PRIV_IP1TO3:-192.168.4}
CONTAINER_RUNTIME_DNS=${CONTAINER_RUNTIME_DNS:-/etc/resolv.conf}			# Note: docker uses 127.0.0.11 while podman forces me to use /etc/resolv.conf
UUID=00000000-0000-0000-0000-000000000000
###OPTIONAL
VLLM_DOCKER_TAG=${VLLM_DOCKER_TAG:-nightly}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.95}
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
if /bin/nvidia-smi --list-gpus | /bin/grep -qi '^GPU 0:' ; then
	GPUoverride+=("--file" "./x_nvidia-override.yaml")
elif /bin/rocm-smi -i | /bin/grep -qi '^GPU\[0\]' ; then
	GPUoverride+=("--file" "./x_rocm-override.yaml")
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

# fix some cgroup stuff
loginctl enable-linger $(whoami)

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
# When containers have dependency chains (e.g. --volumes-from), the filtered
# removal above may miss dependents.  Nuke ALL containers as a last resort.
if podman network exists genai-private-net 2>/dev/null; then
	echo "[rivault] WARN: genai-private-net still exists after rm; force-removing lingering containers..."
	podman ps -aq --filter network=genai-private-net 2>/dev/null | xargs -r podman rm -f 2>/dev/null || true
	# Also remove any remaining containers that may block network deletion
	# (e.g. dependent containers not caught by the network filter)
	podman ps -aq 2>/dev/null | xargs -r podman rm -f 2>/dev/null || true
	podman network rm -f genai-private-net || { echo "[rivault] ERROR: cannot remove genai-private-net"; exit 1; }
fi

# Create a fresh genai-private-net, trying candidate subnets until one works
# NOTE: do NOT use --internal here; rootful podman's conmon port-forwarder
# cannot relay data across an internal bridge (TCP connects but hangs).
# The published port (NGINX_LITELLM_DOCKER_PORT) must be reachable from the
# host so the SSH tunnel from the control-plane container can reach it.
_net_created=0
for IP1TO3 in '192.168.4' '172.16.4' '10.0.4'; do
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

# ─── SSH tunnel: forward MinIO from login node to compute node ────────────────
# The BentoML container accesses S3 via minio-host:19000 (-> host-gateway:19000).
# Bind to 0.0.0.0 so containers (via host-gateway) can reach MinIO,
# and use port 19000 (BENTOML_MINIO_HOST_PORT) as the local port.
UMRS_S3_TUNNEL_PORT=${UMRS_S3_TUNNEL_PORT:-9000}
UMRS_S3_TUNNEL_HOST=${UMRS_S3_TUNNEL_HOST:-${SLURM_SUBMIT_HOST:-localhost}}
_S3_LOCAL_PORT="${BENTOML_MINIO_HOST_PORT:-19000}"
echo "[rivault] Setting up SSH tunnels for S3 (MinIO) access..."
# Tunnel 1: 0.0.0.0:19000 → login:9000 (for containers via host-gateway)
ssh -fN -L "0.0.0.0:${_S3_LOCAL_PORT}:127.0.0.1:${UMRS_S3_TUNNEL_PORT}" "${UMRS_S3_TUNNEL_HOST}" \
    -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes 2>&1 \
    && echo "[rivault] SSH tunnel: 0.0.0.0:${_S3_LOCAL_PORT} → ${UMRS_S3_TUNNEL_HOST}:${UMRS_S3_TUNNEL_PORT} (for containers)" \
    || echo "[rivault] WARNING: SSH tunnel (container) to ${UMRS_S3_TUNNEL_HOST}:${UMRS_S3_TUNNEL_PORT} failed"
# Tunnel 2: localhost:9000 → login:9000 (for host-level presigned URL access)
ssh -fN -L "127.0.0.1:${UMRS_S3_TUNNEL_PORT}:127.0.0.1:${UMRS_S3_TUNNEL_PORT}" "${UMRS_S3_TUNNEL_HOST}" \
    -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes 2>&1 \
    && echo "[rivault] SSH tunnel: localhost:${UMRS_S3_TUNNEL_PORT} → ${UMRS_S3_TUNNEL_HOST}:${UMRS_S3_TUNNEL_PORT} (for host)" \
    || echo "[rivault] WARNING: SSH tunnel (host) already bound or failed (OK if port ${UMRS_S3_TUNNEL_PORT} == ${_S3_LOCAL_PORT})"

sed -i  -e "s@^\(PRIV_IP1TO3=\).*@\1${IP1TO3}@g" \
	-e "s@^\(UUID=\).*@\1${uuid}@g" \
	"./env.${uuid}"

sed	-e "s@UMRS_S3_BUCKET_PORT@$(/bin/grep '^UMRS_S3_BUCKET_PORT=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	"./nginx_fs/out-nginx.conf" > "./nginx_fs/out-nginx.conf.${uuid}"
sed     -e "s@UMRS_S3_BUCKET_HOST@$(/bin/grep '^UMRS_S3_BUCKET_HOST=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	-e "s@CONTAINER_RUNTIME_DNS@$(/bin/grep '^CONTAINER_RUNTIME_DNS=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	"./coredns_fs/coredns.conf" > "./coredns_fs/coredns.conf.${uuid}"
sed	-e "s@VLLM_MODEL_PLACEHOLDER@$(/bin/grep '^VLLM_MODEL=' "./env.${uuid}" | awk -F'=' '{print $2}')@g" \
	"./litellm_fs/config.yaml" > "./litellm_fs/config.yaml.${uuid}"

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

podman-compose --env-file "./env.${uuid}" \
               --file "./serve-model.yaml" ${GPUoverride[@]} \
               ${_profile_flags} \
               up \
	       --remove-orphans --detach

podman-compose --env-file "./env.${uuid}" \
               --file "./userdata-access.yaml" \
               --profile harness \
               up \
	       --remove-orphans --detach

ip a

# Keep job alive while containers are running; exit on idle timeout (safety net).
# UMRS server will also call `scancel` once all embedding threads are done, which
# stops the containers and causes this loop to exit cleanly.
RIVAULT_JOB_TIMEOUT=${RIVAULT_JOB_TIMEOUT:-7200}   # default 2 h; override via sbatch --export
INTERVAL=30
ELAPSED=0
echo "[rivault] Stack is up. Monitoring containers (timeout=${RIVAULT_JOB_TIMEOUT}s)..."
while [ "$ELAPSED" -lt "$RIVAULT_JOB_TIMEOUT" ]; do
    RUNNING=$(podman ps -q --filter "network=genai-private-net" 2>/dev/null | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "[rivault] All containers stopped. Exiting."
        break
    fi
    sleep "$INTERVAL"
    ELAPSED=$(( ELAPSED + INTERVAL ))
done
[ "$ELAPSED" -ge "$RIVAULT_JOB_TIMEOUT" ] && \
    echo "[rivault] Timeout ${RIVAULT_JOB_TIMEOUT}s reached. Exiting."

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
