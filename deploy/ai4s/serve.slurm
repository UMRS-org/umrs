#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# AI4S real vLLM serve wrapper (Singularity / GB200 aarch64)
# ─────────────────────────────────────────────────────────────────────────────
# Replaces serve-openai-smoke.slurm with a real mini-RiVault runtime:
# runs the aarch64 vLLM+LiteLLM+NGINX SIF on the allocated GB200 node and
# exposes an OpenAI-compatible endpoint on RIVAULT_LITELLM_PORT (the port the
# umrs-front Gateway probes and proxies to).
#
# Orchestrator contract (remote_slurm_backend._submit_wrapper_job):
#   Injected env: VLLM_MODEL_PATH, VLLM_SERVED_MODEL_NAME, RIVAULT_DIR,
#                 VLLM_TENSOR_PARALLEL_SIZE, VLLM_MAX_MODEL_LEN,
#                 VLLM_GPU_MEMORY_UTILIZATION, LITELLM_MASTER_KEY,
#                 LITELLM_PORT, JOB_INFO_DIR, RIVAULT_CONTROL_DIR.
#   Job-info file rivault-job-<id>.info reports RIVAULT_LITELLM_PORT/NODE/STATUS.
#   A stop-<id>.request file in RIVAULT_CONTROL_DIR triggers graceful shutdown.
#
# NOTE: This wrapper supports BOTH single-node tensor parallelism (TP ≤ GPUs on
# the node, i.e. ≤ 4 on GB200 NVL4) and multi-node tensor+pipeline parallelism.
# When Slurm allocates more than one node (SLURM_NNODES > 1, e.g. the 4n4gpu-p
# partition) it bootstraps a Ray cluster across the nodes and launches vLLM with
# -tp <gpus_per_node> -pp <node_count> using the Ray distributed executor, so
# models that exceed one node's aggregate GPU memory (DeepSeek-V4-Pro / GLM-5.1 / GLM-5.2)
# can be served at full context. The SIF must include Ray (see
# deploy/singularity/mini-rivault-aarch64-ray.def).
# ─────────────────────────────────────────────────────────────────────────────

# JOB_INFO_DIR: derive from the script's location when not explicit.
# Falls back to SLURM_SUBMIT_DIR/jobs so the orchestrator can discover
# job-info files from any deployment root.
_DEFAULT_JOB_INFO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/jobs"
JOB_INFO_DIR="${JOB_INFO_DIR:-${_DEFAULT_JOB_INFO_DIR}}"
RIVAULT_CONTROL_DIR="${RIVAULT_CONTROL_DIR:-${JOB_INFO_DIR}/control}"

# External port the Gateway connects to (NGINX inside the container listens here).
EXTERNAL_PORT="${LITELLM_PORT:-${NGINX_LITELLM_DOCKER_PORT:-8001}}"

# Internal container ports (not exposed outside the node).
INNER_VLLM_PORT="${VLLM_PORT:-8000}"
INNER_LITELLM_PORT="4000"

# SHORT_MODEL_NAME: the registered UMRS short name (e.g. "llm-jp-4-32b-a3b-thinking").
# Set via orchestrator env var MODEL_NAME; distinct from VLLM_SERVED_MODEL_NAME
# which is the HuggingFace ID.  vLLM serves all names so both the readiness
# probe (short name) and API clients (HF ID) can reach the model.
#
# Capture the short name from the injected MODEL_NAME env BEFORE overwriting
# MODEL_NAME with the HuggingFace ID below. Doing it the other way round (the
# previous bug) made SHORT_MODEL_NAME read the already-overwritten HF ID, so the
# short name was lost: vLLM only served the HF ID and the readiness probe — which
# queries the short name — 404'd, leaving the endpoint stuck in provisioning.
SHORT_MODEL_NAME="${MODEL_NAME:-}"
MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL_NAME:-rivault-model}}"
# If the registered short name equals the HF ID (no distinct alias), clear the
# duplicate so vLLM does not warn on duplicate --served-model-name keys.
if [[ "${SHORT_MODEL_NAME}" == "${MODEL_NAME}" ]]; then
    SHORT_MODEL_NAME=""
fi
HOST_MODEL_PATH="${VLLM_MODEL_PATH:?VLLM_MODEL_PATH is required}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY is required}"

# vLLM tuning (orchestrator-provided with safe fallbacks).
# Tensor parallel = GPUs per node; pipeline parallel = node count. On a
# multi-node allocation Slurm sets SLURM_NNODES, so derive PP from it unless an
# explicit override is provided. World size = TP × PP must equal the total GPU
# count across the allocation (e.g. 4 GPUs/node × 4 nodes = 16 on 4n4gpu-p).
VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_PP="${VLLM_PIPELINE_PARALLEL_SIZE:-${SLURM_NNODES:-1}}"
NODE_COUNT="${SLURM_NNODES:-1}"

# Parallelism strategy across a multi-node allocation. Two layouts are supported:
#
#   tp_pp  (default)  TP = GPUs/node, PP = node count. Pipeline parallel splits
#                     the model by layer across nodes. Simple and memory-robust,
#                     but introduces pipeline bubbles that hurt single-stream
#                     decode latency at low concurrency.
#   tep               Tensor + Expert Parallel: TP = full world size (GPUs/node ×
#                     nodes), PP = 1, plus --enable-expert-parallel. This is the
#                     vLLM GB200 recipe layout for the large MoE checkpoints
#                     (DeepSeek-V4-Pro 2-node, GLM-5.1 2-node): it removes the PP
#                     bubble at the cost of more cross-node tensor-parallel
#                     traffic (NVLink within a tray + InfiniBand between trays).
#                     Memory layout differs from tp_pp, so any tep deployment
#                     must be canary-validated for KV fit before production use.
#
# Opt-in only: VLLM_PARALLEL_STRATEGY defaults to tp_pp so existing multi-node
# behaviour is unchanged. VLLM_EXPERT_PARALLEL=1 can independently force
# --enable-expert-parallel without changing the TP/PP split.
VLLM_PARALLEL_STRATEGY="$(printf '%s' "${VLLM_PARALLEL_STRATEGY:-tp_pp}" | tr '[:upper:]' '[:lower:]')"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_EXPERT_PARALLEL:-0}"
# GPUs physically present per allocated node. The Ray bootstrap below needs this
# per-node count for `ray start --num-gpus` and the cluster GPU-ready check,
# independent of the vLLM -tp value (which TEP widens to the full world size).
VLLM_GPUS_PER_NODE="${VLLM_TP}"
if [[ "${VLLM_PARALLEL_STRATEGY}" == "tep" && "${NODE_COUNT}" -gt 1 ]]; then
    VLLM_TP=$(( VLLM_TP * NODE_COUNT ))   # full world size as tensor parallel
    VLLM_PP=1                             # no pipeline parallel in TEP
    VLLM_ENABLE_EXPERT_PARALLEL=1         # MoE expert parallelism on
fi

# Empty / 0 / "full" / "native" means: do not pin --max-model-len; let vLLM use
# the model's native maximum context ("full token size").
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

# Resolve the SIF image. RIVAULT_DIR may be the .sif itself or its directory.
# Default to the Ray-enabled v4 SIF: the wrapper backend does not inject
# RIVAULT_DIR, so this basename is the effective source of truth for the image
# (multi-node serving requires Ray, which only the v4 SIF bundles).
RIVAULT_DIR="${RIVAULT_DIR:-/work/hps0/home/ea0022/images}"
if [[ "${RIVAULT_DIR}" == *.sif ]]; then
    SIF_IMAGE="${RIVAULT_DIR}"
else
    SIF_IMAGE="${RIVAULT_SIF:-${RIVAULT_DIR%/}/mini-rivault-aarch64-v4.sif}"
fi

# GLM-5.2 (IndexShare / glm_moe_dsa) requires vLLM >= 0.23.0 for the glm_moe_dsa
# architecture. The default v4 SIF ships 0.22.1rc1 which predates this support
# and falls back to DEEPSEEK_V32_INDEXER — producing garbled output. The v6 SIF
# (0.23.1rc1 + Ray 2.48.0) has full glm_moe_dsa support. Override the SIF default
# here so the orchestrator does not need to forward RIVAULT_SIF via env_vars
# (which r4 up --pool-file drops; only the server roster forwards env_vars).
# Explicit RIVAULT_SIF env always wins — this only changes the fallback default.
case "$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
    *glm*5.2*)
        if [[ -z "${RIVAULT_SIF:-}" && "${SIF_IMAGE}" != *v6* ]]; then
            _V6_SIF="${RIVAULT_DIR%/}/mini-rivault-aarch64-v6.sif"
            if [[ -f "${_V6_SIF}" ]]; then
                SIF_IMAGE="${_V6_SIF}"
                echo "[serve] GLM-5.2: defaulting to v6 SIF (vLLM 0.23+) for glm_moe_dsa support"
            else
                echo "[serve] WARNING: GLM-5.2 needs v6 SIF (vLLM 0.23+) but ${_V6_SIF} not found — garbled output likely" >&2
            fi
        fi
        ;;
esac

# ── Runtime fallback deprecation guard ──────────────────────────────────────
# When the orchestrator sets explicit VLLM_* env vars via DeploymentSpec,
# these derivation functions are dormant (VAR already set).
# When they activate (VAR unset), emit a deprecation warning to stderr.
#
# To eliminate this path: populate runtime_config in the registry for
# every model and redeploy via DeploymentSpec.
_emit_runtime_fallback_warning() {
    local var_name="$1"
    local derived_value="$2"
    echo "[DEPRECATED] serve.slurm derived ${var_name}=${derived_value}" >&2
    echo "  Model=${MODEL_NAME:-unknown}" >&2
    echo "  Populate runtime_config in registry to eliminate this fallback." >&2
}

# Derive the vLLM tool-call parser from the model name when not explicitly set.
derive_tool_parser() {
    local name_lc
    name_lc="$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')"
    case "${name_lc}" in
        *deepseek*v4*|*deepseek-v4*) echo "deepseek_v4" ;;
        *deepseek*)                  echo "deepseek_v3" ;;
        *glm-4.5*|*glm4.5*|*glm45*)  echo "glm45" ;;
        *glm*)                       echo "glm47" ;;
        *kimi*)                      echo "kimi_k2" ;;
        *llm-jp*|*llm_jp*)              echo "hermes" ;;
        *gpt-oss*swallow*)           echo "" ;;
        *gemma*4*)                   echo "gemma4" ;;
        *)                           echo "hermes" ;;
    esac
}
if [[ -z "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
    VLLM_TOOL_CALL_PARSER="$(derive_tool_parser)"
    _emit_runtime_fallback_warning "VLLM_TOOL_CALL_PARSER" "$VLLM_TOOL_CALL_PARSER"
fi

# Derive the vLLM reasoning parser from the model name when not explicitly set.
# GLM-4.5/4.7/5.x emit <think>…</think> reasoning before any answer or tool call.
# Served with --enable-auto-tool-choice but NO --reasoning-parser, vLLM does not
# split the reasoning out, so the <think> block leaks into `content` and the
# tool-call parser has to scan past it. The official GLM serving recipe pairs the
# glm47 tool parser with the glm45 reasoning parser; setting it routes the
# </think>-gated reasoning into the response `reasoning` field as intended.
# (Note: this is correctness/recipe-alignment, NOT the tools "!!!!" garbage fix —
# that was the FlashInfer-TRTLLM MoE kernel; see VLLM_USE_FLASHINFER_MOE_FP16=0
# below.) "glm45" is the registered reasoning parser for the whole GLM-4.x/5.x
# family in this SIF's vLLM (vllm/reasoning/__init__.py). Only set it for models
# that actually need it; others keep vLLM's default (no reasoning parser).
derive_reasoning_parser() {
    local name_lc
    name_lc="$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')"
    case "${name_lc}" in
        *glm*) echo "glm45" ;;
        *gemma*4*) echo "gemma4" ;;
        *kimi*) echo "kimi_k2" ;;
        *gpt-oss*swallow*) echo "openai_gptoss" ;;
        *llm-jp*|*llm_jp*) echo "" ;;       # llmjp4 plugin not in SIF; disable for now
        *)     echo "" ;;
    esac
}
if [[ -z "${VLLM_REASONING_PARSER:-}" ]]; then
    VLLM_REASONING_PARSER="$(derive_reasoning_parser)"
    if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
        _emit_runtime_fallback_warning "VLLM_REASONING_PARSER" "$VLLM_REASONING_PARSER"
    fi
fi
# Explicit disable: the orchestrator sets VLLM_REASONING_PARSER=none (sentinel)
# when a caller provisions with reasoning="off". Because the line above uses
# ${VAR:-default}, an empty string would re-trigger derivation, so we use a
# sentinel and normalise it (and a few friendly aliases) back to empty here.
case "$(printf '%s' "${VLLM_REASONING_PARSER}" | tr '[:upper:]' '[:lower:]')" in
    none|off|disabled|false|0) VLLM_REASONING_PARSER="" ;;
esac

# Some model families require a specific KV-cache dtype. DeepSeek-V4's FlashMLA
# fp8 attention layout only supports an fp8 kv-cache (auto fails an assertion),
# so default it to fp8 unless the orchestrator overrides VLLM_KV_CACHE_DTYPE.
#
# NOTE on GLM-5/5.1 (glm_moe_dsa): do NOT force fp8 here. GLM is served on the
# default (auto -> bf16) kv-cache. The historical "wall of !" on tools-bearing
# prompts was NOT a kv-cache issue — it was the FlashInfer-TRTLLM MoE kernel
# producing NaN logits for GLM's expert routing (fixed via
# VLLM_USE_FLASHINFER_MOE_FP16=0, see the GLM MoE block further below). Forcing
# fp8 here was tried and did NOT help, and fp8 brings its own DSA decode-path
# risks for GLM, so leave GLM on bf16.
derive_kv_cache_dtype() {
    local parser_lc
    parser_lc="$(printf '%s' "${VLLM_TOOL_CALL_PARSER}" | tr '[:upper:]' '[:lower:]')"
    case "${parser_lc}" in
        deepseek_v4) echo "fp8" ;;
        *)           echo "" ;;
    esac
}
if [[ -z "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
    VLLM_KV_CACHE_DTYPE="$(derive_kv_cache_dtype)"
    if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
        _emit_runtime_fallback_warning "VLLM_KV_CACHE_DTYPE" "$VLLM_KV_CACHE_DTYPE"
    fi
fi

mkdir -p "${JOB_INFO_DIR}" "${RIVAULT_CONTROL_DIR}"

RIVAULT_NODE="$(hostname -f 2>/dev/null || hostname)"
JOB_INFO_FILE="${JOB_INFO_DIR}/rivault-job-${SLURM_JOB_ID}.info"
STOP_REQUEST_FILE="${RIVAULT_CONTROL_DIR}/stop-${SLURM_JOB_ID}.request"
CONTAINER_PID=""
RAY_HEAD_PID=""
WORKER_PIDS=()

write_job_info() {
    local status="$1"
    cat > "${JOB_INFO_FILE}" <<EOF
RIVAULT_JOB_ID=${SLURM_JOB_ID}
RIVAULT_STATUS=${status}
RIVAULT_NODE=${RIVAULT_NODE}
RIVAULT_LITELLM_PORT=${EXTERNAL_PORT}
RIVAULT_VLLM_PORT=${INNER_VLLM_PORT}
RIVAULT_MODEL_PATH=${HOST_MODEL_PATH}
RIVAULT_MODEL_NAME=${MODEL_NAME}
RIVAULT_SIF=${SIF_IMAGE}
RIVAULT_TOOL_PARSER=${VLLM_TOOL_CALL_PARSER}
RIVAULT_START_TIME=$(date -Iseconds)
EOF
}

cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    if [[ -n "${CONTAINER_PID}" ]]; then
        kill "${CONTAINER_PID}" >/dev/null 2>&1 || true
        wait "${CONTAINER_PID}" >/dev/null 2>&1 || true
    fi
    # Tear down the Ray cluster (multi-node only). Best-effort: stop the head
    # and any worker srun steps so the allocation releases cleanly.
    if [[ -n "${RAY_HEAD_PID}" ]]; then
        kill "${RAY_HEAD_PID}" >/dev/null 2>&1 || true
    fi
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then
        kill "${WORKER_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    rm -f "${STOP_REQUEST_FILE}" >/dev/null 2>&1 || true
    if [[ -n "${RUNTIME_CACHE:-}" ]]; then
        rm -rf "${RUNTIME_CACHE}" >/dev/null 2>&1 || true
    fi
    if [[ -f "${JOB_INFO_FILE}" ]]; then
        sed -i 's/RIVAULT_STATUS=.*/RIVAULT_STATUS=stopped/' "${JOB_INFO_FILE}" || true
        echo "RIVAULT_STOP_TIME=$(date -Iseconds)" >> "${JOB_INFO_FILE}" || true
    fi
    exit "${rc}"
}
trap cleanup EXIT TERM INT

# ── Preflight validation ─────────────────────────────────────────────────────
if [[ ! -f "${SIF_IMAGE}" ]]; then
    echo "[serve] FATAL: SIF image not found: ${SIF_IMAGE}" >&2
    write_job_info failed
    exit 1
fi
if [[ ! -d "${HOST_MODEL_PATH}" ]]; then
    echo "[serve] FATAL: model path not found: ${HOST_MODEL_PATH}" >&2
    write_job_info failed
    exit 1
fi

CONTAINER_CMD="$(command -v singularity || command -v apptainer || true)"
if [[ -z "${CONTAINER_CMD}" ]]; then
    echo "[serve] FATAL: neither singularity nor apptainer found on PATH" >&2
    write_job_info failed
    exit 1
fi

echo "[serve] SLURM_JOB_ID=${SLURM_JOB_ID} node=${RIVAULT_NODE}"
echo "[serve] model=${MODEL_NAME} path=${HOST_MODEL_PATH}"
echo "[serve] sif=${SIF_IMAGE}"
echo "[serve] external_port=${EXTERNAL_PORT} tp=${VLLM_TP} pp=${VLLM_PP} nodes=${NODE_COUNT} maxlen=${VLLM_MAX_MODEL_LEN:-native}"
echo "[serve] tool_parser=${VLLM_TOOL_CALL_PARSER} reasoning_parser=${VLLM_REASONING_PARSER:-none} runtime=${CONTAINER_CMD}"

write_job_info starting

# ── Launch vLLM (OpenAI-compatible server) ───────────────────────────────────
# The mini-RiVault SIF bundles vLLM+LiteLLM+NGINX, but its internal start.sh
# renders config into read-only image paths (/opt/mini-rivault, /etc/nginx),
# which cannot be made writable on AI4S: the site's underlay mount mode
# disables --writable-tmpfs under --fakeroot, and a plain overlay leaves those
# root-owned dirs unwritable for the unprivileged job user.
#
# vLLM already exposes an OpenAI-compatible API with bearer-token auth
# (--api-key), so we run it directly on the external port the Gateway probes.
# All writable runtime state (HF/torch/triton/vLLM caches, HOME) is redirected
# to a per-job /tmp directory; the model weights stay read-only at /models.
#
# Single-node allocations (NODE_COUNT==1) run vLLM directly. Multi-node
# allocations (e.g. 4n4gpu-p) bootstrap a Ray cluster across the nodes first,
# then launch vLLM on the head with -tp <gpus/node> -pp <nodes> and the Ray
# distributed executor backend.
RUNTIME_CACHE="${TMPDIR:-/tmp}/rivault-${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_CACHE}/home" "${RUNTIME_CACHE}/cache" "${RUNTIME_CACHE}/ray"

# FlashInfer (used by DeepSeek-V4 / GLM attention) resolves its cubin directory
# to the installed flashinfer_cubin package path and writes checksums / on-demand
# cubins there at engine init. That path is read-only inside the SIF, so vLLM
# crashes with "Read-only file system". Bind a writable host copy of the bundled
# cubins over the package path. Seed it once with:
#   singularity exec --cleanenv <sif> cp -a \
#     /opt/venv/lib/python3.12/site-packages/flashinfer_cubin/cubins/. <host_dir>/
FLASHINFER_CUBIN_HOST="${FLASHINFER_CUBIN_DIR_HOST:-/work/hps0/home/ea0022/images/flashinfer_cubins}"
FLASHINFER_CUBIN_TARGET="/opt/venv/lib/python3.12/site-packages/flashinfer_cubin/cubins"

# Shared singularity exec flags (env + binds) used by every launch path.
#
# NOTE: We intentionally do NOT use `--net --network none` here. The serving
# container must expose vLLM/LiteLLM on ${EXTERNAL_PORT} over the host network
# (the umrs-front Gateway reaches it via an SSH tunnel to the compute node),
# and multi-node Ray (DeepSeek-V4-Pro, GLM-5.1) requires inter-node TCP for the
# distributed executor. A private/empty network namespace breaks both. On AI4S,
# unprivileged users may only use `--network none` (bridge/CNI is forbidden), so
# there is no portmap workaround. Egress containment for inference is enforced
# instead by TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1 (below) plus node-level
# firewall policy — see docs/admin/security/sandbox-network-isolation.md.
SING_COMMON=(
    exec --nv --cleanenv
    --bind "${HOST_MODEL_PATH}:/models:ro"
    --bind "${RUNTIME_CACHE}:${RUNTIME_CACHE}"
    --env HOME="${RUNTIME_CACHE}/home"
    --env XDG_CACHE_HOME="${RUNTIME_CACHE}/cache"
    --env HF_HOME="${RUNTIME_CACHE}/cache/hf"
    --env VLLM_CACHE_ROOT="${RUNTIME_CACHE}/cache/vllm"
    --env TRITON_CACHE_DIR="${RUNTIME_CACHE}/cache/triton"
    --env TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE}/cache/inductor"
    --env OUTLINES_CACHE_DIR="${RUNTIME_CACHE}/cache/outlines"
    --env XGRAMMAR_CACHE_DIR="${RUNTIME_CACHE}/cache/xgrammar"
    --env TRANSFORMERS_OFFLINE=1
    --env HF_HUB_OFFLINE=1
    --env VLLM_NO_USAGE_STATS=1
    --env VLLM_DO_NOT_TRACK=1
    --env RAY_DISABLE_IMPORT_WARNING=1
    --env LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/venv/lib/python3.12/site-packages/torch/lib"
)

# Mount the writable FlashInfer cubin dir over the read-only package path when a
# seeded host copy is available (required for DeepSeek-V4 / GLM attention).
if [[ -d "${FLASHINFER_CUBIN_HOST}" ]]; then
    SING_COMMON+=( --bind "${FLASHINFER_CUBIN_HOST}:${FLASHINFER_CUBIN_TARGET}" )
fi

# GPT-OSS / Harmony tiktoken vocab (offline). The gpt-oss serving path imports
# openai_harmony, whose compiled tiktoken-rs backend downloads the HarmonyGptOss
# vocab from HuggingFace at first load. That fails on AI4S compute nodes (no
# egress + the HF token lacks openai/gpt-oss LFS access), so the model 500s.
# Fix: pre-seed the tiktoken-rs cache on a host with internet (the file is keyed
# by a URL hash, e.g. fb374d419588a4632f3f557e76b4b70aebbca790) and bind it in
# with TIKTOKEN_RS_CACHE_DIR pointing at it. openai_harmony then resolves the
# vocab from cache with no network. Harmless for non-harmony models (they never
# import openai_harmony, so the var is unused).
TIKTOKEN_RS_CACHE_HOST="${TIKTOKEN_RS_CACHE_DIR_HOST:-/work/hps0/home/ea0022/images/tiktoken_cache_harmony}"
if [[ -d "${TIKTOKEN_RS_CACHE_HOST}" ]]; then
    SING_COMMON+=(
        --bind "${TIKTOKEN_RS_CACHE_HOST}:${TIKTOKEN_RS_CACHE_HOST}:ro"
        --env TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_RS_CACHE_HOST}"
    )
fi

# GLM-5.1 tool-calling garbage fix (wall of "!" / token id 0, finish_reason=
# length) on the GB200 (sm100) build of this vLLM nightly.
#
# Symptom: any tools-bearing GLM request whose rendered prompt exceeds a small,
# content-dependent threshold (as few as ~260 tokens with 3 tools, or a long
# single-tool prompt) returns a wall of "!" instead of text or a tool_call.
# Plain chat is unaffected at any length (verified to 6k+ tokens), and
# DeepSeek-V4-Pro / Kimi-K2.6 serve the identical tools payload correctly.
#
# Root cause (bisected live): the FlashInfer TRTLLM *unquantized MoE* kernel
# produces NaN logits for the expert routing induced by GLM-5.1's tools-rendered
# prompt tokens. GLM-5.1 (GlmMoeDsa) reuses DeepSeek's DSA code path but has
# distinct head dims (qk_nope=192, v=256, interleaved RoPE), and the TRTLLM MoE
# kernel mis-handles its routing. The failure tracks tool-block *content*, not
# prompt length: identical-length diverse non-tool prose works; the same tool
# JSON pasted as plain system text works; only the `tools` array (which the GLM
# chat template renders into a <tools> block) triggers it.
#
# Fix: VLLM_USE_FLASHINFER_MOE_FP16=0 makes the unquantized-MoE oracle drop
# FLASHINFER_TRTLLM from its backend list (fused_moe/oracle/unquantized.py),
# falling back to the numerically robust Triton MoE kernel. GLM-only; DeepSeek-V4
# and Kimi keep FlashInfer TRTLLM. Applied to the engine head and every Ray
# worker via SING_COMMON. Verified: full tool-count sweep (3..25 tools) and real
# tool_calls now succeed with no "!" output. The attention backend is left at its
# default (FLASHINFER_MLA_SPARSE, naturally selected at 16 heads/rank under tp4)
# and is NOT implicated in this bug.
# GLM MoE backend selection: force Triton MoE kernel.
#
# vLLM 0.23.1rc1 auto-selects FlashInfer TRTLLM Unquantized MoE for GLM, but
# that backend produces NaN logits for GLM expert routing on tool-bearing
# prompts (observed as the "!!!" wall). The numerically robust Triton MoE
# kernel avoids this.
#
# NOTE: The old env var VLLM_USE_FLASHINFER_MOE_FP16=0 is silently IGNORED by
# vLLM 0.23.1rc1 (it logs "Unknown vLLM environment variable detected"). The
# correct API is the --moe-backend CLI flag, available since vLLM 0.23.x.
#
# IMPORTANT: VLLM_ARGS is initialized AFTER this block, so we must NOT append
# to VLLM_ARGS here (it would be overwritten). Instead, we set a variable
# (VLLM_MOE_BACKEND) and apply it after VLLM_ARGS=(...) is initialized.
VLLM_MOE_BACKEND=""
case "$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
    *nvfp4*)
        # NVFP4 quantization (modelopt_fp4) requires cutlass/flashinfer MoE.
        # vLLM's NVFP4 oracle rejects triton with:
        #   ValueError: moe_backend='triton' is not supported for NvFP4 MoE.
        # Expected: cutlass, flashinfer_trtllm, flashinfer_cutlass, etc.
        echo "[serve] NVFP4: skipping triton MoE — NVFP4 requires cutlass/flashinfer"
        ;;
    *glm*)
        if [[ "${VLLM_FORCE_FLASHINFER_MOE:-0}" == "1" ]]; then
            # Canary override: FP8/FP4 FlashInfer MoE on Blackwell (upstream GB200 recipe).
            # Only enable on FP8-weight canaries; BF16 FlashInfer-TRTLLM MoE NaNs on tool prompts.
            echo "[serve] GLM: FlashInfer MoE enabled (VLLM_FORCE_FLASHINFER_MOE=1, MXFP4/MXFP8)"
            SING_COMMON+=( --env VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 )
        else
            echo "[serve] GLM: forcing Triton MoE (--moe-backend triton) to avoid FlashInfer-TRTLLM MoE NaN on tool prompts"
            VLLM_MOE_BACKEND="triton"
        fi
        ;;
    *llm-jp*)
        # llm-jp-4-32b-a3b: the MoE expert intermediate_size is 64, which is not
        # a multiple of 128. The FlashInfer-TRTLLM BF16 MoE kernel on Blackwell
        # (sm100) hard-fails engine init with
        #   "Check failed: args->intermediate_size % 128 == 0 (64 vs. 0)"
        # (flashinfer::Bf16MoeLauncher::check_moe via trtllm_bf16_moe). Dropping
        # FLASHINFER_TRTLLM via VLLM_USE_FLASHINFER_MOE_FP16=0 falls back to the
        # Triton MoE kernel, which has no such alignment constraint.
        echo "[serve] llm-jp: forcing Triton MoE (VLLM_USE_FLASHINFER_MOE_FP16=0); expert intermediate_size=64 is not %128, FlashInfer-TRTLLM BF16 MoE rejects it on sm100"
        SING_COMMON+=( --env VLLM_USE_FLASHINFER_MOE_FP16=0 )
        ;;
    *gpt-oss*|*gptoss*)
        # GPT-OSS-Swallow (BF16, 120B): single GB200 GPU has 184 GiB but BF16
        # weights alone consume ~240 GiB, so the model requires tensor-parallel
        # (TP=2) across 2 GPUs. Force TP=2 regardless of the model registry or
        # environment defaults, which currently default to 1.
        echo "[serve] GPT-OSS: BF16 requires TP>=2 (120B > 184 GiB single-GPU) — forcing VLLM_TP=2"
        VLLM_TP=2
        ;;
esac

# vLLM server arguments shared across single/multi-node paths.
VLLM_ARGS=(
    --model /models
    --served-model-name "${MODEL_NAME}"
    ${SHORT_MODEL_NAME:+"--served-model-name" "${SHORT_MODEL_NAME}"}
    --host 0.0.0.0
    --port "${EXTERNAL_PORT}"
    -tp "${VLLM_TP}"
    -pp "${VLLM_PP}"
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
    --trust-remote-code
    --api-key "${LITELLM_MASTER_KEY}"
)
# Tool-call parser (see derive_tool_parser): omitted when empty (model has no
# tool-call training, e.g. GPT-OSS-Swallow). When not empty, add the parser flag.
if [[ -n "${VLLM_TOOL_CALL_PARSER}" ]]; then
    VLLM_ARGS+=( --tool-call-parser "${VLLM_TOOL_CALL_PARSER}" )
fi
# Apply MoE backend selection (set in the GLM case block above).
# This must come AFTER VLLM_ARGS=(...) initialization to avoid being overwritten.
if [[ -n "${VLLM_MOE_BACKEND}" ]]; then
    VLLM_ARGS+=( --moe-backend "${VLLM_MOE_BACKEND}" )
fi
# Auto-tool-choice: only enable when a tool-call parser is configured.
# vLLM validates --enable-auto-tool-choice requires --tool-call-parser.
# Models without tool-call training (e.g. GPT-OSS-Swallow) must skip this flag.
_ENABLE_AUTO_TC="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-on}"
if [[ "${_ENABLE_AUTO_TC}" != "0" && "${_ENABLE_AUTO_TC}" != "false" && "${_ENABLE_AUTO_TC}" != "off" ]]; then
    if [[ -n "${VLLM_TOOL_CALL_PARSER}" ]]; then
        VLLM_ARGS+=( --enable-auto-tool-choice )
    fi
fi
# Reasoning parser (see derive_reasoning_parser): GLM-5/5.1 is a thinking model;
# the glm45 reasoning parser splits the </think>-gated reasoning into the
# response `reasoning` field and is part of the official GLM serving recipe.
# Omitted when empty.
if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
    VLLM_ARGS+=( --reasoning-parser "${VLLM_REASONING_PARSER}" )
fi
# Chat-template content format. The official vLLM GLM-5/5.1 recipe
# (docs.vllm.ai/projects/recipes GLM5, recipes.vllm.ai/zai-org/GLM-5.1) passes
# --chat-template-content-format=string on EVERY launch. Without it vLLM defaults
# to "auto", which can render message content as OpenAI structured "parts" and
# subtly change the rendered prompt shape. Forcing "string" matches the reference
# deployment and renders content deterministically. GLM-only to avoid changing
# the prompt shape for other families that already serve correctly. (Note: this
# is NOT what fixed the tools "!!!!" garbage — that was the FlashInfer-TRTLLM MoE
# kernel; see the VLLM_USE_FLASHINFER_MOE_FP16=0 block above. This flag is kept
# only to match the upstream reference launch.)
case "$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
    *glm*) VLLM_ARGS+=( --chat-template-content-format string ) ;;
esac
# Gemma4: multimodal (text+image) with custom tool protocol. The vLLM recipe
# (recipes.vllm.ai/Google/gemma-4-31B-it) passes --limit-mm-per-prompt for
# image support and --chat-template for tool calling. When the gemma4 tool
# parser is active, add the chat template and MM limit. The chat template
# file lives inside the SIF at the vLLM examples path; the exact path is at
# vllm's site-packages (installed wheel, NOT editable). We let vLLM discover
# it via the tool parser's built-in default.
case "$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
    *gemma*4*)
        # Multimodal: up to 4 images per prompt (recipe default).
        VLLM_ARGS+=( --limit-mm-per-prompt '{"image": 4}' )
        ;;
esac
# GLM-5.2 (IndexShare / glm_moe_dsa): GLM-5.2 introduces IndexShare, which
# reuses one DeepSeek-style sparse-attention indexer across every four sparse
# layers (arxiv 2603.12201). The "skip" layers reuse a neighbour layer's top-k
# indices and carry NO indexer weights in the BF16 checkpoint (k_norm exists
# only on the group-owner layers: the first_k_dense_replace dense layers plus
# every index_topk_freq-th sparse layer). vLLM still builds an indexer module on
# every layer, so under BF16 the post-load strict weight check raises
#   ValueError: Following weights were not initialized from checkpoint:
#   {'model.layers.N.self_attn.indexer.k_norm.{weight,bias}'}
# and engine-core init fails. The official vLLM recipe
# (recipes.vllm.ai/zai-org/GLM-5.2, Troubleshooting) fixes this by disabling the
# strict weight-track check via --model-loader-extra-config; it is safe because
# skip layers never run their indexer (those weights are never read). FP8
# checkpoints disable the check automatically, so this is BF16-only. GLM-5.1 has
# no IndexShare skip layers and is unaffected — scope strictly to 5.2. Requires
# vLLM >= 0.23.0 for the glm_moe_dsa architecture (the v6 SIF; the v4 SIF's
# 0.22.x predates glm_moe_dsa support entirely).
case "$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
    *glm*5.2*) VLLM_ARGS+=( --model-loader-extra-config '{"enable_weights_track": false}' ) ;;
esac
# Pin context length only when explicitly requested; otherwise omit the flag so
# vLLM serves the model's native maximum context ("full token size").
case "$(printf '%s' "${VLLM_MAX_MODEL_LEN}" | tr '[:upper:]' '[:lower:]')" in
    ""|0|full|native|max|auto) : ;;
    *) VLLM_ARGS+=( --max-model-len "${VLLM_MAX_MODEL_LEN}" ) ;;
esac

# Multimodal encoder TP mode (e.g. Kimi-K2.7-Code MoonViT).
# The vLLM recipe for Kimi-K2.7-Code (recipes.vllm.ai/moonshotai/Kimi-K2.7-Code)
# requires --mm-encoder-tp-mode data to run the small MoonViT vision encoder
# data-parallel instead of tensor-parallel. Without this flag, the encoder
# defaults to tensor-parallel mode, which can cause model initialization failures
# or 400 errors from the API. The pool config sets VLLM_MM_ENCODER_TP_MODE=data
# for Kimi-K2.7-Code; this block forwards it to vLLM as --mm-encoder-tp-mode.
VLLM_MM_ENCODER_TP_MODE="${VLLM_MM_ENCODER_TP_MODE:-}"
if [[ -n "${VLLM_MM_ENCODER_TP_MODE}" ]]; then
    VLLM_ARGS+=( --mm-encoder-tp-mode "${VLLM_MM_ENCODER_TP_MODE}" )
fi

# Concurrency cap. More concurrent sequences raise aggregate throughput under
# load at the cost of KV-cache pressure. The orchestrator forwards
# VLLM_MAX_NUM_SEQS (site.yaml vllm.max_num_seqs); when unset, leave vLLM's own
# default in place rather than guessing.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
    VLLM_ARGS+=( --max-num-seqs "${VLLM_MAX_NUM_SEQS}" )
fi

# Automatic prefix caching. Reuses the KV cache for shared prompt prefixes
# (system prompts, few-shot examples, multi-turn history), so repeated requests
# skip prefill over the cached prefix -> much lower time-to-first-token. This is
# also vLLM V1's default; keep it explicit so the behaviour is pinned across SIF
# upgrades. Set VLLM_ENABLE_PREFIX_CACHING=0 to disable.
#
# GLM-5.x (glm_moe_dsa): prefix caching is now SAFE because the root cause of
# the "!!!" wall (FlashInfer TRTLLM Unquantized MoE producing NaN logits for GLM
# expert routing) is fixed by forcing --moe-backend triton above. Previously,
# prefix caching was disabled for GLM to prevent cross-session NaN KV cache
# contamination, but now that the Triton MoE kernel is used instead of
# FlashInfer TRTLLM, the NaN logits no longer occur and prefix caching can be
# enabled for GLM as well.
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    VLLM_ARGS+=( --enable-prefix-caching )
fi

# Cap the prefill batch for very large context windows. vLLM's startup memory
# profiler runs a dummy forward of max-num-batched-tokens tokens, which defaults
# to the full context length. At native 1M context (DeepSeek-V4) that dummy
# activation exhausts host RAM and the batch step is OOM-killed
# ("Detected 1 oom_kill event ... determine_available_memory"). Chunked prefill
# (always on in vLLM v1) still serves the full max-model-len window; only the
# per-step prefill is chunked, so the served "full token size" is unchanged.
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
if [[ -z "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    case "${VLLM_MAX_MODEL_LEN}" in
        ''|*[!0-9]*) : ;;  # non-numeric (native/full/empty): leave to vLLM default
        *) if [[ "${VLLM_MAX_MODEL_LEN}" -gt 131072 ]]; then
               VLLM_MAX_NUM_BATCHED_TOKENS=16384
           fi ;;
    esac
fi
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    VLLM_ARGS+=( --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" )
fi

# Weight quantization (e.g. fp8). Opt-in via VLLM_QUANTIZATION from pool/registry.
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
if [[ -n "${VLLM_QUANTIZATION}" ]]; then
    VLLM_ARGS+=( --quantization "${VLLM_QUANTIZATION}" )
fi

# CUDA graph capture: ON by default for fast decode, but auto-disabled above
# 262144 context. Two reasons converge at the very-large-context (DeepSeek-V4 1M)
# tier:
#   1. Correctness: with graphs ON at native 1M context this nightly vLLM +
#      DeepSeek-V4 hybrid attention (CSA+HCA) produces garbled/incoherent output
#      (observed on job 10074: 200 OK but non-language tokens), while GLM-5.1
#      (202752) and Kimi-K2.6 (245760) — both below the threshold and graphs-ON —
#      serve correctly. Eager mode restores coherent DeepSeek output.
#   2. Host RAM: graph capture pins host RAM proportional to the captured graph
#      batch metadata and historically OOM-killed startup at 1M context.
# The >262144 length threshold below is a secondary guard (host-RAM at 1M). The
# DeepSeek-V4 correctness problem is NOT actually length-dependent: a 131072
# launch (job 10986) below this threshold still produced multilingual gibberish
# with graphs ON. DeepSeek-V4 is therefore pinned to eager by MODEL NAME in the
# block further below (alongside GLM), so any max_model_len is safe. GLM and Kimi
# keep graphs for decode throughput (GLM via its own name-based eager guard). Set
# VLLM_ENFORCE_EAGER=1 to force eager for any model, or VLLM_ENFORCE_EAGER=0 to
# opt a model back into graph capture. The full max-model-len window is unchanged
# either way.
# CUDA graph mode: opt-in decode-only graphs for throughput canaries.
# VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY skips --enforce-eager unless
# VLLM_ENFORCE_EAGER=1 is explicitly set (wins over cudagraph).
VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"
if [[ -n "${VLLM_CUDAGRAPH_MODE}" && "${VLLM_ENFORCE_EAGER}" != "1" ]]; then
    echo "[serve] CUDA graphs: cudagraph_mode=${VLLM_CUDAGRAPH_MODE}"
    VLLM_ARGS+=( --compilation-config "{\"cudagraph_mode\":\"${VLLM_CUDAGRAPH_MODE}\"}" )
elif [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    VLLM_ARGS+=( --enforce-eager )
elif [[ -z "${VLLM_ENFORCE_EAGER}" ]]; then
    case "${VLLM_MAX_MODEL_LEN}" in
        ''|*[!0-9]*) : ;;  # non-numeric (native/full/empty): leave graphs on
        *) if [[ "${VLLM_MAX_MODEL_LEN}" -gt 262144 ]]; then
               VLLM_ENFORCE_EAGER=1
           fi ;;
    esac
    # GLM-5.1 (and the GLM MoE family): the primary tools "!!!!" garbage fix is
    # forcing the Triton MoE backend (VLLM_USE_FLASHINFER_MOE_FP16=0, see the GLM
    # MoE block earlier). Eager mode is retained as an additional numeric guard
    # for the CUDA-graph capture path on this nightly vLLM + MoE build, since the
    # tools/agent prompt shape exercises code paths plain chat never reaches. All
    # tool-calling verification was performed with eager mode on. Set
    # VLLM_ENFORCE_EAGER=0 or VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY to re-enable
    # CUDA graphs once a SIF build is confirmed NaN-free on the tools path.
    #
    # DeepSeek-V4: CUDA-graph capture produces NaN/garbage logits for DeepSeek-V4
    # on this nightly vLLM build regardless of context length (observed as
    # multilingual gibberish on even trivial prompts when launched below the
    # 262144 length threshold above). The length heuristic only protected the
    # native 1M launch; pin eager by model name so any max_model_len is safe.
    if [[ -z "${VLLM_ENFORCE_EAGER}" ]]; then
        case "$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
            *glm*) VLLM_ENFORCE_EAGER=1 ;;
            *deepseek*v4*|*deepseek-v4*) VLLM_ENFORCE_EAGER=1 ;;
        esac
    fi
    if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
        VLLM_ARGS+=( --enforce-eager )
    fi
fi

# Apply a model-specific KV-cache dtype when required (e.g. DeepSeek-V4 fp8).
if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]]; then
    VLLM_ARGS+=( --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}" )
fi

# Expert parallelism for MoE models. Shards experts across the parallel workers
# instead of replicating them, which is the vLLM GB200 recipe for the large MoE
# checkpoints (DeepSeek-V4-Pro / GLM-5.1 / Kimi-K2.6). Enabled automatically by
# the tep parallel strategy; can also be forced via VLLM_EXPERT_PARALLEL=1.
if [[ "${VLLM_ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
    VLLM_ARGS+=( --enable-expert-parallel )
fi

if [[ "${NODE_COUNT}" -gt 1 ]]; then
    # ── Multi-node: bootstrap a Ray cluster across the allocation ─────────────
    VLLM_ARGS+=( --distributed-executor-backend ray )

    mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    HEAD_NODE="${RAY_NODES[0]}"
    HEAD_IP="$(hostname -i 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' | head -1)"
    if [[ -z "${HEAD_IP}" ]]; then
        HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')"
    fi
    RAY_PORT=6379
    RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    WANT_GPUS=$(( VLLM_GPUS_PER_NODE * NODE_COUNT ))
    echo "[serve] multi-node: nodes=${NODE_COUNT} head=${HEAD_NODE} (${HEAD_IP}) want_gpus=${WANT_GPUS} strategy=${VLLM_PARALLEL_STRATEGY} tp=${VLLM_TP} pp=${VLLM_PP}"

    # RUNTIME_CACHE lives on each node's local /tmp; the batch script only ran
    # mkdir on the head node. Create it on every allocated node so the worker
    # containers can bind-mount it (otherwise singularity fails with
    # "mount source ... doesn't exist").
    srun --nodes="${NODE_COUNT}" --ntasks-per-node=1 \
        mkdir -p "${RUNTIME_CACHE}/home" "${RUNTIME_CACHE}/cache" "${RUNTIME_CACHE}/ray" || true

    # Start the Ray head on this (head) node. --block keeps the head process
    # alive in the background while vLLM connects to the cluster separately.
    # Unset the SIF-baked VLLM_PORT=8000 before launching Ray so the vLLM
    # RayWorkerProc actors Ray later spawns do not inherit it and double-bind
    # ZMQ on :8000 ("Address already in use").
    # Also export VLLM_USE_FLASHINFER_MOE_FP16 explicitly: SING_COMMON passes it
    # via --env, but Ray runtime env isolation can strip container-level vars.
    # Exporting in the parent shell ensures RayWorkerProc actors see it.
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        "${SIF_IMAGE}" /bin/sh -c '
            export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-}"
            unset VLLM_PORT
            exec /opt/venv/bin/ray start --head \
                --node-ip-address="$1" --port="$2" \
                --num-gpus="$3" --temp-dir="$4" \
                --include-dashboard=false --block' _ \
            "${HEAD_IP}" "${RAY_PORT}" "${VLLM_GPUS_PER_NODE}" "${RUNTIME_CACHE}/ray" &
    RAY_HEAD_PID=$!

    # Wait for the head GCS to accept connections.
    for _ in $(seq 1 30); do
        if "${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" \
                /opt/venv/bin/ray status --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    # Start Ray workers on the remaining nodes (one task per node).
    for ((i=1; i<${#RAY_NODES[@]}; i++)); do
        wnode="${RAY_NODES[$i]}"
        echo "[serve] starting ray worker on ${wnode}"
        # Same VLLM_PORT unset + VLLM_USE_FLASHINFER_MOE_FP16 export as the
        # head: worker-node Ray actors must not inherit the baked :8000, and
        # must see the MoE backend override to avoid NaN logits on GLM.
        srun --nodes=1 --ntasks=1 --nodelist="${wnode}" \
            "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
                "${SIF_IMAGE}" /bin/sh -c '
                    export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-}"
                    unset VLLM_PORT
                    exec /opt/venv/bin/ray start \
                        --address="$1" --num-gpus="$2" \
                        --temp-dir="$3" --block' _ \
                    "${RAY_ADDRESS}" "${VLLM_GPUS_PER_NODE}" "${RUNTIME_CACHE}/ray" &
        WORKER_PIDS+=($!)
    done

    # Wait until Ray reports every GPU across the cluster (TP × nodes).
    have=0
    for _ in $(seq 1 60); do
        have="$("${CONTAINER_CMD}" "${SING_COMMON[@]}" "${SIF_IMAGE}" \
            /opt/venv/bin/python -c "import ray; ray.init(address='${RAY_ADDRESS}'); print(int(ray.cluster_resources().get('GPU',0)))" \
            2>/dev/null | tail -1)"
        if [[ "${have}" == "${WANT_GPUS}" ]]; then
            break
        fi
        sleep 5
    done
    echo "[serve] ray cluster GPUs ready: ${have}/${WANT_GPUS}"

    # The mini-RiVault SIF bakes VLLM_PORT=8000 into its %environment (survives
    # --cleanenv). vLLM's get_open_port() honours VLLM_PORT, so the fixed value
    # is reused by multiple subsystems (engine-coordinator + the cross-node
    # torch.distributed TCPStore), which double-bind the same port and fail with
    # EADDRINUSE. Unset VLLM_PORT inside the container so vLLM allocates dynamic
    # free ports per subsystem. Pin VLLM_HOST_IP to the head for rendezvous.
    #
    # Also explicitly export VLLM_USE_FLASHINFER_MOE_FP16 in the vLLM launch
    # shell.  SING_COMMON passes it via --env, but RayWorkerProc actors spawned
    # by the Ray cluster may not inherit container-level env vars reliably
    # (Ray runtime env isolation).  Exporting in the parent shell ensures every
    # vLLM worker sees it — belt-and-suspenders with the --env flag.
    write_job_info running
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        --env RAY_ADDRESS="${RAY_ADDRESS}" \
        --env VLLM_HOST_IP="${HEAD_IP}" \
        "${SIF_IMAGE}" /bin/sh -c '
            export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-}"
            unset VLLM_PORT
            exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "$@"' _ \
            "${VLLM_ARGS[@]}" &
    CONTAINER_PID=$!
else
    # ── Single-node: run vLLM directly (proven path) ─────────────────────────
    write_job_info running
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        "${SIF_IMAGE}" /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
            "${VLLM_ARGS[@]}" &
    CONTAINER_PID=$!
fi

# ── Supervise: exit on container death or graceful stop request ──────────────
while kill -0 "${CONTAINER_PID}" >/dev/null 2>&1; do
    if [[ -f "${STOP_REQUEST_FILE}" ]]; then
        echo "[serve] graceful stop requested: ${STOP_REQUEST_FILE}"
        exit 0
    fi
    sleep 2
done

wait "${CONTAINER_PID}"

# NOTE: The Assistant Service (ADR-012) is no longer co-launched on the
# compute node. It runs as a persistent CPU-only service on the frontend
# server (umrs-front), where the RAG service is also co-located. See
# deploy/ansible/roles/front_end/templates/frontend-assistant-service.yaml.j2.
