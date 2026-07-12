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

# ─── Task 4: assert-offline — fail fast if offline invariants violated ──────
# Establish the offline inference posture in the batch shell so it is inherited
# by every child process (podman, compose, vLLM). The container also receives
# these via `-e`, but exporting here makes the invariant authoritative and lets
# the assert below validate the real environment rather than an unset shell.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
_assert_offline_invariants() {
    local failed=0
    for _var in TRANSFORMERS_OFFLINE HF_HUB_OFFLINE; do
        if [ "${!_var:-0}" != "1" ]; then
            echo "FAIL: ${_var}=${!_var:-not set} (expected 1)"
            failed=1
        fi
    done
    for _var in http_proxy HTTP_PROXY https_proxy HTTPS_PROXY; do
        if [ -n "${!_var:-}" ]; then
            echo "FAIL: ${_var} is set — proxy forbidden in offline mode"
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "FAIL: offline invariants violated. Refusing to start inference."
        exit 1
    fi
    echo "OK: offline invariants satisfied."
}
_assert_offline_invariants

mkdir -p "${HOME}/.config/containers/"

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

prepare_mcp_mount_overlay() {
    [[ -n "${RIVAULT_MCP_MOUNTS_JSON:-}" ]] || return 0

    local overlay="./runtime-mounts/${uuid}/mcp-servers"
    mkdir -p "$overlay"

    MCP_MOUNT_OVERLAY="$overlay" python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path(os.environ["MCP_MOUNT_OVERLAY"]).resolve()
mounts = json.loads(os.environ.get("RIVAULT_MCP_MOUNTS_JSON", "[]"))
for mount in mounts:
    source = Path(str(mount.get("source", ""))).expanduser().resolve()
    target = str(mount.get("target", "/mcp-servers")).strip() or "/mcp-servers"
    if not source.exists():
        raise SystemExit(f"MCP mount source does not exist: {source}")
    normalized_target = target.rstrip("/") or "/"
    if normalized_target != "/mcp-servers" and not normalized_target.startswith("/mcp-servers/"):
        raise SystemExit(f"MCP mount target escapes /mcp-servers: {target}")
    relative = normalized_target.removeprefix("/mcp-servers").strip("/") or source.name
    destination = (root / relative).resolve()
    if root not in (destination, *destination.parents):
        raise SystemExit(f"MCP mount target escapes /mcp-servers: {target}")
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)
PY
    export MCP_SERVERS_PATH="$overlay"
}

prepare_mcp_mount_overlay

# ─── Derive vLLM model-path variables from orchestrator env ──────────────────
# VLLM_MODEL_PATH (host filesystem path) is set by the orchestrator when it
# provisions an endpoint for a registered model.  When absent, fall back to the
# legacy HuggingFace-cache layout.
_vllm_host_model_path="${VLLM_HOST_MODEL_PATH:-${VLLM_MODEL_PATH:-}}"
if [[ -z "$_vllm_host_model_path" ]]; then
    _hf_cache="${VLLM_PATH_TO_HUGGINGFACE:-./models/vllm/huggingface}"
    _model_id="${VLLM_MODEL:-Qwen/Qwen3-0.6B}"
    _vllm_host_model_path="${_hf_cache}/hub/models--$(echo "$_model_id" | sed 's|/|--|g')/snapshots"
fi

# ─── Resolve through HuggingFace cache snapshots layout ──────────────────────
# HF cache stores model weights under snapshots/<commit-hash>/.  If the
# resolved path ends in a "snapshots" directory, or doesn't contain
# config.json directly, descend into the first subdirectory that does.
# This fixes "Invalid repository ID or local directory" errors from vLLM.
_resolve_model_dir() {
    local dir="$1"
    # Already contains config.json → use as-is
    if [[ -f "${dir}/config.json" || -f "${dir}/params.json" ]]; then
        echo "$dir"
        return
    fi
    # Look for config.json one level down (snapshots/<hash>/ or named subdir)
    local candidate
    for candidate in "${dir}"/*/; do
        [[ -d "$candidate" ]] || continue
        if [[ -f "${candidate}config.json" || -f "${candidate}params.json" ]]; then
            # Remove trailing slash
            echo "${candidate%/}"
            return
        fi
    done
    # Fallback: return original (let vLLM produce a clear error)
    echo "$dir"
}

if [[ -d "$_vllm_host_model_path" ]]; then
    _vllm_host_model_path="$(_resolve_model_dir "$_vllm_host_model_path")"
fi

# ─── Handle HF cache symlinks ────────────────────────────────────────────────
# HF cache snapshot directories contain symlinks like config.json → ../../blobs/xxx.
# If we bind-mount only the snapshot dir, symlinks break inside the container.
# Detect this case and mount the full HF model directory instead, adjusting
# the container model path to the correct subdirectory.
_vllm_host_mount_path="${_vllm_host_model_path}"
_vllm_container_model_subdir=""
if [[ -L "${_vllm_host_model_path}/config.json" || -L "${_vllm_host_model_path}/params.json" ]]; then
    # Symlinks detected — need to mount the parent model dir (includes blobs/)
    # Path structure: .../hub/models--org--name/snapshots/<hash>
    _model_root="${_vllm_host_model_path}"
    # Walk up until we find the blobs/ sibling directory
    while [[ "$_model_root" != "/" && ! -d "${_model_root}/blobs" ]]; do
        _model_root="$(dirname "$_model_root")"
    done
    if [[ -d "${_model_root}/blobs" ]]; then
        # Compute relative path from model root to snapshot
        _rel_path="${_vllm_host_model_path#${_model_root}/}"
        _vllm_host_mount_path="${_model_root}"
        _vllm_container_model_subdir="/${_rel_path}"
    fi
fi
_vllm_mount_name="$(basename "$_vllm_host_mount_path")"
# VLLM_HOST_MODEL_PATH is the bind-mount source (may be model root or snapshot)
_vllm_host_model_path="${_vllm_host_mount_path}"
# VLLM_CONTAINER_MODEL_PATH is the bind-mount target in the container
_vllm_container_model_path="${VLLM_CONTAINER_MODEL_PATH:-/models/${_vllm_mount_name}}"
# VLLM_MODEL_REF is what vLLM --model receives (may include subdirectory)
_vllm_model_ref="${VLLM_MODEL_REF:-${_vllm_container_model_path}${_vllm_container_model_subdir}}"
_vllm_served_model_name="${VLLM_SERVED_MODEL_NAME:-${VLLM_MODEL:-Qwen/Qwen3-0.6B}}"

# ─── Resolve tokenizer path for GGUF / separate-tokenizer models ─────────────
# vLLM needs a real tokenizer directory when the model path is a GGUF file.
# If VLLM_TOKENIZER_HOST_PATH is provided, bind-mount it into the container.
# If VLLM_TOKENIZER_ARGS is not set but VLLM_TOKENIZER is, build the CLI arg.
_vllm_tokenizer_host_path="${VLLM_TOKENIZER_HOST_PATH:-}"
_vllm_tokenizer_container_path="${VLLM_TOKENIZER_CONTAINER_PATH:-/tokenizer}"
if [[ -z "$_vllm_tokenizer_host_path" && -n "${VLLM_TOKENIZER:-}" ]]; then
    if [[ "${VLLM_TOKENIZER}" == /* ]]; then
        _vllm_tokenizer_host_path="${VLLM_TOKENIZER}"
    else
        # Treat as a directory under the HF cache root
        _vllm_tokenizer_host_path="${VLLM_PATH_TO_HUGGINGFACE:-./models/vllm/huggingface}/hub/models--$(echo "$VLLM_TOKENIZER" | sed 's|/|--|g')/snapshots"
        if [[ -d "$_vllm_tokenizer_host_path" ]]; then
            _vllm_tokenizer_host_path="$(_resolve_model_dir "$_vllm_tokenizer_host_path")"
        fi
    fi
fi
_vllm_tokenizer_args="${VLLM_TOKENIZER_ARGS:-}"
if [[ -z "$_vllm_tokenizer_args" && -n "${VLLM_TOKENIZER:-}" ]]; then
    _vllm_tokenizer_args=" --tokenizer=${_vllm_tokenizer_container_path}"
fi
if [[ -n "${VLLM_TRUST_REMOTE_CODE:-}" ]]; then
    _vllm_tokenizer_args="${_vllm_tokenizer_args} --trust-remote-code"
fi
# NVFP4/quantized-MoE checkpoints (e.g. Qwen3.6-35B-A3B-NVFP4 on GB10) must pin a
# quant-aware fused-MoE backend. Otherwise vLLM auto-selects the "Unquantized MoE
# backend" for the experts, which registers no per-expert input_scale param and
# aborts weight-load with KeyError: '...experts.w2_input_scale'. `marlin` is the
# DGX Spark (GB10) NVFP4 MoE backend from the vLLM recipe.
if [[ -n "${VLLM_MOE_BACKEND:-}" ]]; then
    _vllm_tokenizer_args="${_vllm_tokenizer_args} --moe-backend ${VLLM_MOE_BACKEND}"
fi

# Merge the roster/pool-supplied VLLM_EXTRA_ARGS (arbitrary vLLM flags declared
# in an endpoint's env_vars, e.g. DSpark's --kv-cache-dtype / --speculative-config)
# with the tokenizer/trust/moe flags derived above into ONE value that reaches
# BOTH launch paths:
#   • single-node: podman-compose interpolates ${VLLM_EXTRA_ARGS}; the process
#     env takes precedence over --env-file, so we export the merged value to
#     keep the compose interpolation and the env file in agreement.
#   • multi-node: `podman run --env-file env.<uuid>` does NOT inherit the job
#     shell env, so the merged value MUST be written into env.<uuid> (see the
#     VLLM_EXTRA_ARGS line in the heredoc below). Without this merge every
#     roster flag is silently dropped on the multi-node head/worker containers.
_roster_extra_args="${VLLM_EXTRA_ARGS:-}"
_vllm_extra_args="${_vllm_tokenizer_args}${_roster_extra_args:+ ${_roster_extra_args}}"
_vllm_extra_args="${_vllm_extra_args# }"   # strip a single leading space
export VLLM_EXTRA_ARGS="${_vllm_extra_args}"

NGINX_LITELLM_DOCKER_PORT="${LITELLM_PORT:-${NGINX_LITELLM_DOCKER_PORT:-$(shuf -i 1025-65535 -n 1)}}"
export NGINX_LITELLM_DOCKER_PORT

# Profiles must be resolved before writing env.${uuid}; several generated
# values, especially LiteLLM database settings, depend on active profiles.
RIVAULT_PROFILES=${RIVAULT_PROFILES:-"harness vllm"}
if [[ -n "${RIVAULT_EXTRA_PROFILES:-}" ]]; then
    for _extra_profile in $RIVAULT_EXTRA_PROFILES; do
        if [[ " ${RIVAULT_PROFILES} " != *" ${_extra_profile} "* ]]; then
            RIVAULT_PROFILES="${RIVAULT_PROFILES} ${_extra_profile}"
        fi
    done
fi

_litellm_db_profile_active=0
if [[ " ${RIVAULT_PROFILES} " == *" litellm-db "* ]]; then
    _litellm_db_profile_active=1
fi
_litellm_postgres_user="${LITELLM_POSTGRES_USER:-postgres}"
_litellm_postgres_password="${LITELLM_POSTGRES_PASSWORD:-litellm}"
_litellm_postgres_db="${LITELLM_POSTGRES_DB:-litellm}"
_litellm_database_url_default=""
if [[ "${_litellm_db_profile_active}" -eq 1 ]]; then
    _litellm_database_url_default="postgresql://${_litellm_postgres_user}:${_litellm_postgres_password}@ll-postgres:5432/${_litellm_postgres_db}"
fi
_litellm_db_sql_migrations_default="false"
if [[ "${_litellm_db_profile_active}" -eq 1 ]]; then
    _litellm_db_sql_migrations_default="true"
fi
_litellm_db_sql_migrations="${LITELLM_DB_SQL_MIGRATIONS:-${_litellm_db_sql_migrations_default}}"
_litellm_disable_schema_update_default="true"
if [[ "${_litellm_db_sql_migrations}" != "true" && ( "${_litellm_db_profile_active}" -eq 1 || -n "${LITELLM_DATABASE_URL:-}" ) ]]; then
    _litellm_disable_schema_update_default="false"
fi

# Proxy resolution: point to egress-guard only when the 'egress' profile is
# active (i.e. egress-guard container will actually be running). When
# HTTPS_PROXY is set to a non-running service, aiohttp creates a default SSL
# context at session creation time and applies it to ALL connections including
# internal HTTP ones (LiteLLM → vLLM), causing spurious ssl:<SSLContext>
# ConnectionRefused errors on port 8000.
if [[ " ${RIVAULT_PROFILES} " == *" egress "* ]]; then
    _rivault_http_proxy="${RIVAULT_HTTP_PROXY:-http://egress-guard:8888}"
else
    _rivault_http_proxy="${RIVAULT_HTTP_PROXY:-}"   # empty → no proxy
fi
_rivault_https_proxy="${RIVAULT_HTTPS_PROXY:-${_rivault_http_proxy}}"

cat <<EOF > "./env.${uuid}"
# ═══════════════════════════════════════════════════════════════════════════════
# Mini-RiVault environment file — generated per-job (UUID=${uuid})
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 1. vLLM: LLM Model Identity & Paths ─────────────────────────────────────
# Which HuggingFace model ID to serve (used by vLLM --model flag)
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen3-0.6B}
# Local HuggingFace hub cache directory (legacy layout fallback)
VLLM_PATH_TO_HUGGINGFACE=${VLLM_PATH_TO_HUGGINGFACE:-./models/vllm/huggingface}
# Resolved host path to the model weights directory
VLLM_HOST_MODEL_PATH=${_vllm_host_model_path}
# Mount target path inside the vLLM container
VLLM_CONTAINER_MODEL_PATH=${_vllm_container_model_path}
# Model reference passed to vLLM --model (container path or HF ID)
VLLM_MODEL_REF=${_vllm_model_ref}
# Name exposed via /v1/models endpoint (what clients see)
VLLM_SERVED_MODEL_NAME=${_vllm_served_model_name}
# Extra vLLM CLI arguments (e.g. " --tokenizer=/tokenizer" for GGUF). Emitted
# only when non-empty and appended to VLLM_MODEL_REF so no empty token is left
# in the podman-compose command string. Includes the merged roster
# VLLM_EXTRA_ARGS (see the merge block above) so multi-node podman-run
# containers — which read ONLY this file — receive the roster flags.
${_vllm_extra_args:+VLLM_EXTRA_ARGS=${_vllm_extra_args}}
# Host path bind-mounted as the tokenizer directory
VLLM_TOKENIZER_HOST_PATH=${_vllm_tokenizer_host_path:-/dev/null}
# Container path where the tokenizer directory is mounted
VLLM_TOKENIZER_CONTAINER_PATH=${_vllm_tokenizer_container_path:-/tokenizer}

# ─── 2. vLLM: Parallelism & Performance ──────────────────────────────────────
# Tensor parallelism degree (split layers across GPUs horizontally)
VLLM_TP=${VLLM_TP:-1}
# Pipeline parallelism degree (split layers across GPUs vertically; >1 for
# multi-node large-model inference via Slurm --nodes)
VLLM_PP=${VLLM_PP:-1}
# Number of Slurm nodes requested (>1 enables multi-node PP)
SLURM_NODES=${SLURM_NODES:-1}
# Distributed executor backend (set to 'ray' for multi-node PP)
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-}
# Ray cluster address (head_ip:port; set by multi-node bootstrap)
RAY_ADDRESS=${RAY_ADDRESS:-}
# Container image tag for the vLLM service
VLLM_DOCKER_TAG=${VLLM_DOCKER_TAG:-v0.20.0}
# Fraction of GPU memory vLLM is allowed to use for KV-cache
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.95}
# Tool-call parsing strategy (hermes = Hermes-style function call format)
VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER:-hermes}
# Reasoning parser for models that emit thinking/reasoning content
VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER:-}
# Trust remote code from the model repository
VLLM_TRUST_REMOTE_CODE=${VLLM_TRUST_REMOTE_CODE:-}
# Opt-in eager mode (empty = run WITH CUDA graphs, higher decode throughput).
# Consumed by the single-node compose command AND the multi-node head exec line.
# Models needing eager (DeepSeek-V4 NaN guard, GLM-5.2) set VLLM_ENFORCE_EAGER=1
# in their pool env_vars. Written here so the multi-node podman-run container —
# which does NOT inherit the job shell env — can see it.
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-}
# NCCL/RCCL shared-memory limits for multi-GPU collective comms
VLLM_NCCL_MEM_LIMIT=8G
VLLM_RCCL_MEM_LIMIT=8G
# LiteLLM worker processes (system tier with concurrent users: set 4-8;
# user/project tier single-user: keep at 1 for fast startup)
LITELLM_CPU_LIMIT=${LITELLM_CPU_LIMIT:-1}

# ─── 3. Vision Models (BentoML service) ──────────────────────────────────────
# SAM3 (Segment Anything Model 2) — HuggingFace repo ID and local path
SAM3_MODEL=${SAM3_MODEL:-facebook/sam2-hiera-small}
VISION_MODEL_PATH=${VISION_MODEL_PATH:-./models/sam3}
SAM3_MODEL_PATH=${SAM3_MODEL_PATH:-./models/sam3}
# HMAE (Hierarchical Masked Autoencoder) — weights, config, checkpoint
HMAE_MODEL_PATH=${HMAE_MODEL_PATH:-./models/hmae}
HMAE_CONFIG_PATH=${HMAE_CONFIG_PATH:-./models/hmae/infer_hmae_vit_b.yaml}
HMAE_CHECKPOINT=${HMAE_CHECKPOINT:-./models/hmae/latest_checkpoint.pth}

# ─── 4. Networking & Ports ────────────────────────────────────────────────────
# Published LiteLLM port on the host (clients connect here).
# R-1 (Phase 2k): prefer orchestrator-allocated LITELLM_PORT to avoid
# collision between concurrent jobs; fall back to random if unset.
NGINX_LITELLM_DOCKER_PORT=${NGINX_LITELLM_DOCKER_PORT}
# First three octets of the private container subnet (genai-private-net)
PRIV_IP1TO3=${PRIV_IP1TO3:-192.168.4}
# DNS resolver for containers (podman uses host resolv.conf, not 127.0.0.11)
CONTAINER_RUNTIME_DNS=${CONTAINER_RUNTIME_DNS:-/etc/resolv.conf}
# S3/MinIO endpoint forwarded from login node via SSH tunnel
UMRS_S3_BUCKET_HOST=${UMRS_S3_BUCKET_HOST:-${SLURM_SUBMIT_HOST}}
UMRS_S3_BUCKET_PORT=${UMRS_S3_BUCKET_PORT:-9000}

# ─── 5. Authentication & Secrets ─────────────────────────────────────────────
# R-3 (Phase 2k): 256-bit entropy. LITELLM_SALT_KEY is per-job because
# LiteLLM Postgres is tmpfs-only; move to stable secret store if persisted.
LITELLM_SALT_KEY=${LITELLM_SALT_KEY:-$(openssl rand -hex 32)}
# LiteLLM proxy master key (Bearer token for /v1/* requests)
LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:-$(openssl rand -hex 32)}
# Internal vLLM API key (LiteLLM → vLLM backend auth)
VLLM_API_KEY=${VLLM_API_KEY:-$(openssl rand -hex 32)}
# MCP orchestrator API key (reserved for MCPO when re-enabled)
MCPO_API_KEY=${MCPO_API_KEY:-$(openssl rand -hex 32)}
# LiteLLM Postgres DSN — ll-postgres container provides the DB when
# the 'litellm-db' profile is active. Uses container hostname (ll-postgres)
# resolved via the genai-private-net overlay network.
LITELLM_POSTGRES_USER=${_litellm_postgres_user}
LITELLM_POSTGRES_PASSWORD=${_litellm_postgres_password}
LITELLM_POSTGRES_DB=${_litellm_postgres_db}
LITELLM_DATABASE_URL=${LITELLM_DATABASE_URL:-${_litellm_database_url_default}}
LITELLM_DISABLE_SCHEMA_UPDATE=${LITELLM_DISABLE_SCHEMA_UPDATE:-${_litellm_disable_schema_update_default}}
LITELLM_DB_SQL_MIGRATIONS=${_litellm_db_sql_migrations}

# ─── 6. EgressGuard & Proxy ──────────────────────────────────────────────────
# Proxy is set only when the 'egress' profile is active; empty otherwise.
# RIVAULT_* variants are canonical; HTTP_PROXY/http_proxy are conventional
# aliases consumed by curl, pip, etc. inside containers.
RIVAULT_HTTP_PROXY=${_rivault_http_proxy}
RIVAULT_HTTPS_PROXY=${_rivault_https_proxy}
RIVAULT_NO_PROXY=${RIVAULT_NO_PROXY:-localhost,127.0.0.1,::1,litellm,vllm,ll-postgres,coredns,nginx,egress-guard}
HTTP_PROXY=${HTTP_PROXY:-${_rivault_http_proxy}}
HTTPS_PROXY=${HTTPS_PROXY:-${_rivault_https_proxy}}
NO_PROXY=${NO_PROXY:-${RIVAULT_NO_PROXY:-localhost,127.0.0.1,::1,litellm,vllm,ll-postgres,coredns,nginx,egress-guard}}
http_proxy=${http_proxy:-${_rivault_http_proxy}}
https_proxy=${https_proxy:-${_rivault_https_proxy}}
no_proxy=${no_proxy:-${RIVAULT_NO_PROXY:-localhost,127.0.0.1,::1,litellm,vllm,ll-postgres,coredns,nginx,egress-guard}}
# Comma-separated domains allowed through egress-guard (empty = fully offline)
EGRESS_DOMAIN_ALLOWLIST=${EGRESS_DOMAIN_ALLOWLIST:-}

# ─── 7. Multi-Tenancy & Instance Identity ────────────────────────────────────
# Deployment tier: system | project | user (controls isolation policy)
RIVAULT_TIER=${RIVAULT_TIER:-user}
# Unique instance ID for this mini-RiVault job
RIVAULT_INSTANCE_ID=${RIVAULT_INSTANCE_ID:-${uuid}}
# Owning project or user (for audit and resource scoping)
RIVAULT_OWNER_ID=${RIVAULT_OWNER_ID:-${PROJECT_ID:-unknown}}
# Active compose profiles for this mini-RiVault job
RIVAULT_PROFILES=${RIVAULT_PROFILES}
# Additional compose profiles to activate (space-separated)
RIVAULT_EXTRA_PROFILES=${RIVAULT_EXTRA_PROFILES:-}

# ─── 8. Container Runtime ────────────────────────────────────────────────────
# Placeholder replaced by sed after network creation (actual subnet octet)
UUID=00000000-0000-0000-0000-000000000000
# User namespace mode for rootless podman (map container root to host user)
RIVAULT_USERNS_MODE=${RIVAULT_USERNS_MODE:-keep-id:uid=0,gid=0}

# ─── 9. User-Mounted Resources (MCP / RAG / Project Data) ────────────────────
# Host path to user-provided MCP server code (bind-mounted read-only)
MCP_SERVERS_PATH=${MCP_SERVERS_PATH:-/tmp/mcp-servers}
# Host path to project datasets (bind-mounted read-only)
PROJECT_DATA_PATH=${PROJECT_DATA_PATH:-/tmp/project-data}
# Host path to pre-built RAG index (bind-mounted read-only)
RAG_INDEX_PATH=${RAG_INDEX_PATH:-/tmp/rag-index}
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
    # R-FastStart: 2 s is enough for the user session socket to appear;
    # the original 5 s was overly conservative.
    sleep 2
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
# R-FastStart: 2 probes × 1 s instead of 3 × 2 s — saves up to 4 s while still
# providing a meaningful liveness signal before the stack launches.
_tunnel_ok=0
for _attempt in 1 2; do
    sleep 1
    if bash -c "echo > /dev/tcp/127.0.0.1/${_S3_LOCAL_PORT}" 2>/dev/null; then
        _tunnel_ok=1
        break
    fi
    echo "[rivault] SSH tunnel probe attempt ${_attempt}/2 failed, retrying..."
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
    -e "s@UMRS_S3_BUCKET_HOST@$(/bin/grep '^UMRS_S3_BUCKET_HOST=' "./env.${uuid}" | awk -F'[=#\t ]+' '{print $2}')@g" \
	"./nginx_fs/out-nginx.conf" > "./nginx_fs/out-nginx.conf.${uuid}"; then
	echo "[rivault] FATAL: nginx out-nginx.conf substitution failed"; exit 1
fi
_substitute_required "out-nginx.conf" "./nginx_fs/out-nginx.conf.${uuid}" "UMRS_S3_BUCKET_PORT"
_substitute_required "out-nginx.conf" "./nginx_fs/out-nginx.conf.${uuid}" "UMRS_S3_BUCKET_HOST"

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

# Profiles to activate were resolved before env.${uuid} generation so profile
# dependent values stay consistent with the compose stack.
_profile_flags=""
for _p in $RIVAULT_PROFILES; do
	_profile_flags="${_profile_flags} --profile ${_p}"
done

_cleanup_started=0
RIVAULT_CONTROL_DIR="${RIVAULT_CONTROL_DIR:-}"
RIVAULT_STOP_REQUEST_FILE=""
RAY_HEAD_PID=""
WORKER_PIDS=()
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
    # Stop Ray workers and head (multi-node only)
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then
        kill "${WORKER_PIDS[@]}" 2>/dev/null || true
    fi
    if [[ -n "${RAY_HEAD_PID}" ]]; then
        kill "${RAY_HEAD_PID}" 2>/dev/null || true
    fi
    # Stop standalone multi-node vLLM (host-network container, not on the bridge)
    podman rm -f rivault-vllm-mn >/dev/null 2>&1 || true
    podman rm -f rivault-ray-head >/dev/null 2>&1 || true
    # Stop Ray worker containers on the remaining nodes (multi-node only)
    if [[ ${#RAY_NODES[@]} -gt 1 ]]; then
        for ((i=1; i<${#RAY_NODES[@]}; i++)); do
            srun --nodes=1 --ntasks=1 --nodelist="${RAY_NODES[$i]}" --overlap \
                podman rm -f rivault-ray-worker >/dev/null 2>&1 || true
        done
    fi
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
    rm -rf "./runtime-litellm-migrations.${uuid}" >/dev/null 2>&1 || true
    return "$rc"
}

trap 'cleanup_stack "signal:TERM" 143; exit 143' TERM
trap 'cleanup_stack "signal:INT" 130; exit 130' INT
trap 'cleanup_stack "exit" "$?"' EXIT

run_litellm_db_sql_migrations() {
    if [[ "${_litellm_db_profile_active}" -ne 1 || "${_litellm_db_sql_migrations}" != "true" ]]; then
        return 0
    fi

    local migrations_dir="./runtime-litellm-migrations.${uuid}"
    local litellm_image="ghcr.io/berriai/litellm-non_root:${LITELLM_DOCKER_TAG:-main-stable}"
    local postgres_image="docker.io/postgres:${POSTGRES_VERSION:-17}"
    local postgres_container_id=""
    local db_ready=0
    local applied_count=""

    echo "[rivault] Preparing LiteLLM DB schema using bundled SQL migrations..."
    rm -rf "${migrations_dir}" 2>/dev/null || true
    mkdir -p "${migrations_dir}"

    if ! podman-compose --project-name "rivault" --in-pod false \
             --env-file "./env.${uuid}" \
             --file "./serve-model.yaml" \
             ${GPUoverride[@]} \
             ${_profile_flags} \
             up --remove-orphans --detach ll-postgres; then
        echo "[rivault] ERROR: failed to start ll-postgres for LiteLLM DB migrations"
        return 1
    fi

    for _attempt in $(seq 1 60); do
        postgres_container_id="$(podman ps --filter 'name=rivault_ll-postgres_1' --format '{{.ID}}' | head -1)"
        if [[ -n "${postgres_container_id}" ]] && \
           podman run --rm --network "container:${postgres_container_id}" \
               -e "PGPASSWORD=${_litellm_postgres_password}" \
               "${postgres_image}" \
               psql -h 127.0.0.1 -U "${_litellm_postgres_user}" -d "${_litellm_postgres_db}" \
                    -Atc 'select 1' >/dev/null 2>&1; then
            db_ready=1
            break
        fi
        sleep 2
    done
    if [[ "${db_ready}" -ne 1 ]]; then
        echo "[rivault] ERROR: ll-postgres did not become ready for LiteLLM DB migrations"
        return 1
    fi

    if ! podman run --rm --user 0:0 \
             -v "$(pwd)/${migrations_dir}:/out:rw" \
             --entrypoint sh \
             "${litellm_image}" \
             -lc 'set -e
for src in \
    /app/litellm-proxy-extras/litellm_proxy_extras/migrations \
    /usr/lib/python*/site-packages/litellm_proxy_extras/migrations; do
    if [ -d "$src" ]; then
        cp -R "$src" /out/migrations
        exit 0
    fi
done
echo "LiteLLM migrations directory not found in image" >&2
exit 42'; then
        echo "[rivault] ERROR: failed to extract LiteLLM SQL migrations from ${litellm_image}"
        return 1
    fi

    applied_count="$({
        podman run --rm --network "container:${postgres_container_id}" \
            -v "$(pwd)/${migrations_dir}/migrations:/migrations:ro" \
            -e "PGPASSWORD=${_litellm_postgres_password}" \
            -e "LITELLM_POSTGRES_USER=${_litellm_postgres_user}" \
            -e "LITELLM_POSTGRES_DB=${_litellm_postgres_db}" \
            "${postgres_image}" \
            sh -lc 'set -e
count=0
for file in /migrations/*/migration.sql; do
    [ -f "$file" ] || continue
    count=$((count + 1))
    psql -v ON_ERROR_STOP=1 \
         -h 127.0.0.1 \
         -U "$LITELLM_POSTGRES_USER" \
         -d "$LITELLM_POSTGRES_DB" \
         -f "$file" >/dev/null
done
printf "%s" "$count"'
    } 2>/dev/null)" || {
        echo "[rivault] ERROR: failed to apply LiteLLM SQL migrations"
        return 1
    }

    if [[ -z "${applied_count}" || "${applied_count}" -eq 0 ]]; then
        echo "[rivault] ERROR: no LiteLLM SQL migrations were applied"
        return 1
    fi
    echo "[rivault] LiteLLM DB schema initialized (${applied_count} SQL migrations)."
}

# ─── Multi-node Ray cluster bootstrap ────────────────────────────────────────
# When Slurm allocates more than one node, bootstrap a Ray cluster across the
# allocation before starting the compose stack.  The vLLM container (in compose)
# then connects to the Ray cluster via --distributed-executor-backend ray and
# RAY_ADDRESS, allowing pipeline-parallel serving of models that exceed one
# node's GPU memory (e.g. GLM-5.2 223 GB across 2 × 128 GB GB10 nodes).
#
# Ray head runs in a --network host container on the head node; Ray workers run
# in --network host containers on the remaining nodes via srun.  The vLLM
# container (on the genai-private-net bridge) reaches the Ray head through the
# bridge gateway, which routes to the host's external IP.
_MULTI_NODE=0
if [[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" -gt 1 ]]; then
    _MULTI_NODE=1
    echo "[rivault] Multi-node allocation: ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES}} nodes"

    mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}")
    HEAD_NODE="${RAY_NODES[0]:-$(hostname)}"
    HEAD_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -1)"
    if [[ -z "${HEAD_IP}" ]]; then
        HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')"
    fi
    if [[ -z "${HEAD_IP}" ]]; then
        echo "[rivault] FATAL: cannot determine head node IP for Ray bootstrap" >&2
        exit 1
    fi

    RAY_PORT=6379
    RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    VLLM_IMAGE="${VLLM_DOCKER_IMAGE:-docker.io/vllm/vllm-openai:${VLLM_DOCKER_TAG:-v0.20.0}}"

    # Default the tensor-parallel degree in the SCRIPT's shell scope. The env
    # file heredoc's `VLLM_TP=${VLLM_TP:-1}` only substitutes into the file
    # text — it does NOT assign this shell variable. The multi-node head/worker
    # podman commands below interpolate `${VLLM_TP}` (vllm serve -tp) and
    # `${_gpus_per_node}` (ray start --num-gpus) directly via outer-shell
    # expansion, so an unset VLLM_TP produced empty flags and `ray start`
    # failed, so the GCS never bound and the whole stack timed out. Roster may
    # pin VLLM_TP (e.g. DSpark TP=2); default is TP=1 (across-node PP).
    VLLM_TP="${VLLM_TP:-1}"

    # Physical GPUs per node. GB10 DGX-Spark nodes have exactly 1; other sites
    # override via VLLM_GPUS_PER_NODE. This is the actual device count each Ray
    # node advertises to the cluster (`ray start --num-gpus`) and is DISTINCT
    # from VLLM_TP. The previous code conflated the two (num-gpus=VLLM_TP,
    # want_gpus=VLLM_TP*nodes), which only holds when tensor-parallel stays
    # within a single node. A cross-node TP group (e.g. DSpark DeepSeek-V4:
    # TP=2 spanning 2 × 1-GPU nodes) needs num-gpus = real per-node count = 1,
    # and a total GPU target of gpus_per_node × nodes, NOT VLLM_TP × nodes.
    _gpus_per_node="${VLLM_GPUS_PER_NODE:-1}"
    _nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"

    # Total real GPUs in the Ray cluster. vLLM places exactly TP × PP workers
    # onto them, so TP × PP MUST equal this count.
    _want_gpus=$(( _gpus_per_node * _nodes ))

    # Derive PP from the cluster size and the requested TP when the roster did
    # not pin it (PP = total_gpus / TP). One formula covers both topologies:
    #   • GLM-5.2 (TP unset → 1): PP = _want_gpus / 1 = nodes → across-node PP.
    #   • DSpark  (TP = 2):       PP = _want_gpus / 2 = 1     → cross-node TP group.
    # An explicit roster VLLM_PP > 1 is respected verbatim.
    if [[ "${VLLM_PP:-1}" -le 1 ]]; then
        if (( VLLM_TP <= 0 || _want_gpus % VLLM_TP != 0 )); then
            echo "[rivault] FATAL: cluster GPUs (${_want_gpus}) not divisible by VLLM_TP (${VLLM_TP})" >&2
            exit 1
        fi
        VLLM_PP=$(( _want_gpus / VLLM_TP ))
        echo "[rivault] Derived VLLM_PP=${VLLM_PP} (=${_want_gpus} GPUs / TP ${VLLM_TP})"
    fi

    # Invariant: TP × PP must exactly consume the cluster GPUs, else vLLM's
    # worker placement will hang waiting for GPUs that do not exist (or leave
    # GPUs idle).
    if (( VLLM_TP * VLLM_PP != _want_gpus )); then
        echo "[rivault] FATAL: TP×PP (${VLLM_TP}×${VLLM_PP}) != cluster GPUs (${_gpus_per_node}×${_nodes}=${_want_gpus})" >&2
        exit 1
    fi

    # Export Ray vars and persist them to the env file BEFORE launching the
    # head container (the co-located vLLM driver reads VLLM_PP / RAY_ADDRESS
    # from the env file).
    export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray
    export RAY_ADDRESS="${RAY_ADDRESS}"
    export VLLM_PP
    cat >> "./env.${uuid}" <<RAYEOF

# ─── Multi-node Ray (auto-generated) ─────────────────────────────────────────
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}
RAY_ADDRESS=${RAY_ADDRESS}
VLLM_PP=${VLLM_PP}
RAYEOF

    # Resolve vLLM host mounts / paths from the env file for the head container.
    # The vLLM driver MUST run in the SAME container as the Ray head: a driver
    # in a separate container cannot resolve its local Ray node (the Ray session
    # dir /tmp/ray is per-container), which fails with
    #   "No node info found matching attributes: ''".
    # Co-locating the driver with `ray start --head` shares the session, node
    # identity and Ray install, so ray.init() resolves the local node cleanly.
    _envval() { /bin/grep "^$1=" "./env.${uuid}" | head -1 | awk -F'=' '{print $2}'; }
    _mn_hf="$(_envval VLLM_PATH_TO_HUGGINGFACE)"
    _mn_model_host="$(_envval VLLM_HOST_MODEL_PATH)"
    _mn_model_ctr="$(_envval VLLM_CONTAINER_MODEL_PATH)"
    _mn_tok_host="$(_envval VLLM_TOKENIZER_HOST_PATH)"; _mn_tok_host="${_mn_tok_host:-/dev/null}"
    _mn_tok_ctr="$(_envval VLLM_TOKENIZER_CONTAINER_PATH)"; _mn_tok_ctr="${_mn_tok_ctr:-/tokenizer}"
    _mn_s3host="$(_envval UMRS_S3_BUCKET_HOST)"
    _mn_priv="$(_envval PRIV_IP1TO3)"; _mn_priv="${_mn_priv:-10.0.0}"
    _mn_nginxout="$(_envval NGINX_OUT_IP4)"; _mn_nginxout="${_mn_nginxout:-252}"
    # User-namespace mode MUST mirror the single-node compose (serve-model.yaml:
    # `userns_mode: ${RIVAULT_USERNS_MODE:-keep-id:uid=0,gid=0}`). The value is
    # RUNTIME-DEPENDENT: crun sites use keep-id (needed for Lustre), runc sites
    # (rccs-cloud GB10 DGX-Spark) use `host`. Injected from site.yaml as
    # RIVAULT_USERNS_MODE. Using the wrong mode is why the Ray worker actors
    # could NVML-detect the GPU yet fail to create a CUDA context.
    _mn_userns="$(_envval RIVAULT_USERNS_MODE)"; _mn_userns="${_mn_userns:-keep-id:uid=0,gid=0}"

    _mn_vllm_args=(
        # --- GPU device passthrough (identical to the single-node compose) ---
        --device /dev/nvidia0 --device /dev/nvidiactl
        --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
        --device /dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-caps/nvidia-cap2
        # --- Container namespace/cap config (MUST match serve-model.yaml) ---
        # The single-node compose sets `userns_mode: ${RIVAULT_USERNS_MODE:-...}`
        # + `cap_add: sys_nice`. The userns value is RUNTIME-DEPENDENT (crun →
        # keep-id, runc/rccs-cloud → host). With the WRONG mode the rootless
        # container can QUERY the GPU via NVML (nvmlDeviceGetCount()>0) but
        # CANNOT create a CUDA *context*: the vLLM RayWorkerWrapper actors then
        # crash on their first device op (`torch.zeros(1, device=cuda)` in
        # cuda_communicator.py:75) with `CUDA error: unknown error`
        # (cudaErrorUnknown) on BOTH nodes — before any NCCL traffic. Mirroring
        # the single-node namespace/cap config is what lets CUDA-context
        # creation succeed in the Ray worker subprocess.
        --userns "${_mn_userns}"
        --cap-add SYS_NICE
        # NOTE: do NOT add `--ipc host` here — podman rejects combining a host
        # IPC namespace with `--shm-size` (see the `--shm-size` on the head
        # and worker `podman run` below), which the Ray object store relies on.
        -v "${_mn_hf}/:/root/.cache/huggingface:ro"
        -v "${_mn_hf}/modules/:/root/.cache/huggingface/modules/"
        -v "${_mn_model_host}:${_mn_model_ctr}:ro"
        -v /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro
        -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro
        -v /usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro
        -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1
        # Match the single-node GPU override env exactly.
        -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        # Match single-node: point Triton straight at the bind-mounted driver
        # dir so it never needs the (unrefreshed) ldconfig cache to find libcuda.
        -e TRITON_LIBCUDA_PATH=/usr/lib/aarch64-linux-gnu
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
        -e CUDA_VISIBLE_DEVICES=0
        # Multi-node NCCL: disable the cuMem/VMM allocator (unstable on GB10
        # Grace-Blackwell) and surface init/transport diagnostics on next run.
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_DEBUG=INFO
        # GB10 has UNIFIED memory (GPU allocations + kernel page cache share the
        # same pool as host RAM). Two levers keep the ~94 GiB/node weight load
        # within the 121.63 GiB pool: (1) --object-store-memory=1g on `ray start`
        # (below) caps the otherwise ~10 GB Ray plasma reservation, which is
        # unused in the PP path (weights load from disk, pipeline tensors go over
        # NCCL/GPU) and was the bulk of the head's non-torch overhead; (2) a small
        # /dev/shm (below). With those, steady + load-time usage stays well under
        # Ray's memory-monitor kill threshold, so we leave the monitor ENABLED
        # (fully disabling it let a node thrash into unresponsiveness under load).
        # 0.95 protects the node from a true runaway while tolerating the
        # reclaimable page-cache spike during the GGUF read.
        # NOTE: per-model overrides are possible via RAY_memory_usage_threshold
        # and RAY_memory_monitor_refresh_ms in the pool roster env_vars. A roster
        # value of RAY_memory_monitor_refresh_ms=0 disables the monitor entirely
        # (needed for large GGUF models where Lustre page cache pre-fills the pool).
        -e RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.95}"
        -e RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-}"
        -e VLLM_LOGGING_LEVEL=DEBUG
    )
    if [[ "${_mn_tok_host}" != "/dev/null" ]]; then
        _mn_vllm_args+=( -v "${_mn_tok_host}:${_mn_tok_ctr}:ro" )
    fi
    if [[ -n "${_mn_s3host}" ]]; then
        _mn_vllm_args+=( --add-host "${_mn_s3host}:${_mn_priv}.${_mn_nginxout}" )
    fi

    # Start the Ray head AND the co-located vLLM driver in one container.
    # Sequence inside the container: install Ray → start Ray head (daemonized,
    # no --block) → wait for every GPU (head + workers) to register → exec
    # `vllm serve` as a Ray driver on the local node.
    echo "[rivault] Starting Ray head + co-located vLLM on ${HEAD_NODE} (${HEAD_IP})..."
    podman rm -f rivault-vllm-mn >/dev/null 2>&1 || true
    podman run -d --name rivault-vllm-mn --network host --shm-size=4g \
        --entrypoint /bin/sh \
        --env-file "./env.${uuid}" \
        "${_mn_vllm_args[@]}" \
        "${VLLM_IMAGE}" \
        -c 'pip install -q --no-index --find-links /root/.cache/huggingface/ray_aarch64_env/wheels ray 2>&1 | grep -v "^$\|already\|Requirement" || pip install -q ray; ray start --head --node-ip-address="'"${HEAD_IP}"'" --port="'"${RAY_PORT}"'" --num-gpus="'"${_gpus_per_node}"'" --object-store-memory=1073741824 --include-dashboard=false; echo "[vllm-head] Ray head started; waiting for '"${_want_gpus}"' GPUs..."; i=0; while [ $i -lt 120 ]; do g=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | head -1 | sed "s@.*/@@; s@ GPU@@; s@[.].*@@"); [ "${g:-0}" = "'"${_want_gpus}"'" ] && break; sleep 5; i=$((i+1)); done; echo "[vllm-head] GPUs=${g:-0}; launching vLLM"; exec vllm serve "$VLLM_MODEL_REF" $VLLM_EXTRA_ARGS --served-model-name "$VLLM_SERVED_MODEL_NAME" --api-key "$VLLM_API_KEY" -tp "$VLLM_TP" -pp "$VLLM_PP" --distributed-executor-backend ray --host 0.0.0.0 --port 8000 --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.95}" ${VLLM_ENFORCE_EAGER:+--enforce-eager} --enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER:-hermes}"' \
        || { echo "[rivault] FATAL: failed to launch Ray head + vLLM container"; exit 1; }
    RAY_HEAD_PID=""

    # Wait for the Ray head GCS to accept connections. Use a host-side TCP probe
    # (Ray is not baked into the image, so spinning an ephemeral container per
    # poll would reinstall Ray each time). The head container runs
    # `pip install ray` before `ray start`; installing ray + its deps (grpcio,
    # etc.) inside the egress-restricted container can take a couple of minutes
    # on a cold cache before the GCS port binds. Allow up to 600s (300 x 2s) so
    # a slow cold install cannot race the probe. NOTE: if `ray start --head`
    # itself fails (e.g. a bad `--num-gpus` value), GCS never binds and this
    # probe will exhaust the full window before FATAL — check the head
    # container's own logs for the `ray start` error in that case.
    _ray_ready=0
    for _ in $(seq 1 300); do
        if (exec 3<>"/dev/tcp/${HEAD_IP}/${RAY_PORT}") 2>/dev/null; then
            exec 3>&- 3<&- 2>/dev/null || true
            _ray_ready=1
            break
        fi
        sleep 2
    done
    if [[ "${_ray_ready}" -ne 1 ]]; then
        echo "[rivault] FATAL: Ray head did not become ready at ${RAY_ADDRESS}" >&2
        exit 1
    fi

    # Start Ray workers on the remaining nodes (one task per node).
    for ((i=1; i<${#RAY_NODES[@]}; i++)); do
        wnode="${RAY_NODES[$i]}"
        echo "[rivault] Starting Ray worker on ${wnode}..."
        # Launch each worker inside a backgrounded retry subshell. `ray start
        # --block` runs for the life of the job; if the worker container exits
        # (e.g. a transient rootless-podman / GPU-access glitch on the node),
        # the loop relaunches it. This makes the cluster resilient to an
        # intermittent failure where a freshly-started worker container came up
        # WITHOUT working GPU access, so its RayWorkerWrapper actor saw
        # nvmlDeviceGetCount()==0 and raised
        #   "current platform  does not support ray".
        (
            for _wattempt in 1 2 3 4 5; do
                # Reset any stale rootless-podman pause process on the worker
                # node. A leftover pause process from a previous job makes every
                # podman invocation fail with: "invalid internal status, try
                # resetting the pause process with podman system migrate".
                srun --nodes=1 --ntasks=1 --nodelist="${wnode}" --overlap \
                    podman system migrate >/dev/null 2>&1 || true
                # Give the worker container the SAME GPU/CUDA/model access as the
                # head (`_mn_vllm_args`): vLLM's RayWorkerWrapper actors spawn
                # inside this container and need the nvidia-caps devices + CUDA
                # libs + the model mounted at the SAME container path as the head
                # for pipeline-stage weight loading. The NVML self-check before
                # `ray start` asserts the container can actually see the GPU:
                # `ray start --num-gpus` otherwise advertises a phantom GPU to
                # the head even when the container has no device access, which
                # passes the head GPU gate but crashes the vLLM worker later.
                srun --nodes=1 --ntasks=1 --nodelist="${wnode}" \
                    podman run --name rivault-ray-worker --replace --network host --shm-size=4g \
                        --entrypoint /bin/sh \
                        "${_mn_vllm_args[@]}" \
                        "${VLLM_IMAGE}" \
                        -c "pip install -q --no-index --find-links /root/.cache/huggingface/ray_aarch64_env/wheels ray 2>/dev/null || pip install -q ray; python3 -c 'import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount() > 0' && exec ray start --address='${RAY_ADDRESS}' --num-gpus='${_gpus_per_node}' --object-store-memory=1073741824 --block"
                echo "[rivault] Ray worker on ${wnode} exited (attempt ${_wattempt}/5); relaunching in 5s..." >&2
                sleep 5
            done
            echo "[rivault] FATAL: Ray worker on ${wnode} failed to stay up after 5 attempts" >&2
        ) &
        WORKER_PIDS+=($!)
    done

    # Wait until Ray reports every GPU across the cluster (gpus_per_node × nodes).
    # Use `ray status` (a lightweight GCS query) rather than ray.init(), which
    # spins a full Ray driver + plasma store and can hang when /dev/shm is
    # small. The whole probe container is wrapped in a hard `timeout` so the
    # gate can never block the job indefinitely; if the count can't be
    # confirmed we warn and proceed (the co-located vLLM driver waits for the
    # GPUs itself and re-validates on ray.init).
    _have_gpus="$(timeout 200 podman run --rm --network host --shm-size=4g --entrypoint /bin/sh \
        "${VLLM_IMAGE}" -c "pip install -q --no-index --find-links /root/.cache/huggingface/ray_aarch64_env/wheels ray 2>/dev/null || pip install -q ray; i=0; while [ \$i -lt 30 ]; do g=\$(ray status --address='${RAY_ADDRESS}' 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 | sed 's@.*/@@; s@ GPU@@; s@[.].*@@'); if [ \"\${g:-0}\" = \"${_want_gpus}\" ]; then echo \$g; break; fi; sleep 5; i=\$((i+1)); done; echo \${g:-0}" \
        2>/dev/null | tail -1)"
    _have_gpus="${_have_gpus:-0}"
    echo "[rivault] Ray cluster ready: ${_have_gpus}/${_want_gpus} GPUs"
    if [[ "${_have_gpus}" != "${_want_gpus}" ]]; then
        echo "[rivault] WARNING: Ray reported ${_have_gpus}/${_want_gpus} GPUs; gathering Ray cluster diagnostics before proceeding..."
        echo "[rivault] --- ray status (full) ---"
        timeout 60 podman exec rivault-vllm-mn ray status 2>&1 | sed 's@^@[rivault][ray-status] @' || true
        echo "[rivault] --- worker Ray container logs ---"
        for ((i=1; i<${#RAY_NODES[@]}; i++)); do
            wnode="${RAY_NODES[$i]}"
            echo "[rivault] worker ${wnode} rivault-ray-worker logs:"
            timeout 60 srun --nodes=1 --ntasks=1 --nodelist="${wnode}" --overlap \
                podman logs --tail 40 rivault-ray-worker 2>&1 | sed "s@^@[rivault][worker:${wnode}] @" || true
        done
        echo "[rivault] --- end diagnostics; proceeding — the co-located vLLM driver waits for GPUs itself. ---"
    fi

    # Confirm the head + vLLM container is still alive (it should be either
    # waiting for GPUs or already loading the model).
    sleep 3
    if ! podman ps --filter name=rivault-vllm-mn --format '{{.Names}}' 2>/dev/null | grep -q rivault-vllm-mn; then
        echo "[rivault] FATAL: Ray head + vLLM container exited immediately"
        podman logs --tail 60 rivault-vllm-mn 2>&1 | sed 's/^/[rivault]   /' || true
        exit 1
    fi
    echo "[rivault] Ray head + co-located vLLM container is running (model loading in background)."

    # Drop the bridged 'vllm' compose profile and repoint LiteLLM at the head IP.
    # Also drop 'vision-bentoml': a multi-node pipeline-parallel LLM consumes
    # every GPU across all allocated nodes (TP×PP == total GPUs), leaving none
    # for the BentoML vision service, so it cannot run in this topology.
    RIVAULT_PROFILES="$(echo "${RIVAULT_PROFILES}" | tr ' ' '\n' | grep -vxE 'vllm|vision-bentoml' | tr '\n' ' ')"
    echo "[rivault] Multi-node: dropped 'vllm' and 'vision-bentoml' profiles (GPUs fully used by PP LLM)."
    _profile_flags=""
    for _p in ${RIVAULT_PROFILES}; do
        _profile_flags="${_profile_flags} --profile ${_p}"
    done
    sed -i "s@http://vllm:8000/v1@http://${HEAD_IP}:8000/v1@g" "./litellm_fs/config.yaml.${uuid}" \
        || echo "[rivault] WARNING: failed to repoint LiteLLM api_base to head IP"
    echo "[rivault] LiteLLM repointed to co-located vLLM at http://${HEAD_IP}:8000/v1"
fi

if ! run_litellm_db_sql_migrations; then
    exit 1
fi

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
