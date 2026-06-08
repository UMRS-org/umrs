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
# models that exceed one node's aggregate GPU memory (DeepSeek-V4-Pro / GLM-5.1)
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

MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL_NAME:-rivault-model}}"
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
        *gemma*4*)                   echo "gemma4" ;;
        *)                           echo "hermes" ;;
    esac
}
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-$(derive_tool_parser)}"

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
        *)     echo "" ;;
    esac
}
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-$(derive_reasoning_parser)}"

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
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-$(derive_kv_cache_dtype)}"

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
case "$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')" in
    *glm*)
        echo "[serve] GLM: forcing Triton MoE (VLLM_USE_FLASHINFER_MOE_FP16=0) to avoid FlashInfer-TRTLLM MoE NaN on tool prompts"
        SING_COMMON+=( --env VLLM_USE_FLASHINFER_MOE_FP16=0 )
        ;;
esac

# vLLM server arguments shared across single/multi-node paths.
VLLM_ARGS=(
    --model /models
    --served-model-name "${MODEL_NAME}"
    --host 0.0.0.0
    --port "${EXTERNAL_PORT}"
    -tp "${VLLM_TP}"
    -pp "${VLLM_PP}"
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
    --api-key "${LITELLM_MASTER_KEY}"
)
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
# Pin context length only when explicitly requested; otherwise omit the flag so
# vLLM serves the model's native maximum context ("full token size").
case "$(printf '%s' "${VLLM_MAX_MODEL_LEN}" | tr '[:upper:]' '[:lower:]')" in
    ""|0|full|native|max|auto) : ;;
    *) VLLM_ARGS+=( --max-model-len "${VLLM_MAX_MODEL_LEN}" ) ;;
esac

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
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"
if [[ -z "${VLLM_ENFORCE_EAGER}" ]]; then
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
    # VLLM_ENFORCE_EAGER=0 to re-enable CUDA graphs once a SIF build is confirmed
    # NaN-free on the tools path (would improve decode throughput).
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
fi
if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    VLLM_ARGS+=( --enforce-eager )
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
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        "${SIF_IMAGE}" /bin/sh -c 'unset VLLM_PORT; exec /opt/venv/bin/ray start --head \
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
        # Same VLLM_PORT unset as the head: worker-node Ray actors must not
        # inherit the baked :8000 or they fail with ZMQError "Address already
        # in use" during vLLM engine init.
        srun --nodes=1 --ntasks=1 --nodelist="${wnode}" \
            "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
                "${SIF_IMAGE}" /bin/sh -c 'unset VLLM_PORT; exec /opt/venv/bin/ray start \
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
    write_job_info running
    "${CONTAINER_CMD}" "${SING_COMMON[@]}" \
        --env RAY_ADDRESS="${RAY_ADDRESS}" \
        --env VLLM_HOST_IP="${HEAD_IP}" \
        "${SIF_IMAGE}" /bin/sh -c 'unset VLLM_PORT; exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "$@"' _ \
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
