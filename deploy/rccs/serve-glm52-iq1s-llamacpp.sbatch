#!/bin/bash
#SBATCH -p ai-l40s
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=512G
#SBATCH --time=23:00:00
#SBATCH -J glm52-iq1s
#SBATCH -o /lvs0/rccs-dmp/%u/ai4s-platform/umrs/deploy/rccs/jobs/glm52-iq1s-%j.out
#SBATCH -e /lvs0/rccs-dmp/%u/ai4s-platform/umrs/deploy/rccs/jobs/glm52-iq1s-%j.err
#
# Serve unsloth GLM-5.2 dynamic 1-bit (UD-IQ1_S, 216 GB merged GGUF) via
# llama.cpp llama-server on the ai-l40s-00 node (8x L40S = 352 GB VRAM,
# x86_64, 1.5 TB RAM).
#
# Why llama.cpp, not vLLM: the IQ1_S GGUF cannot fit vLLM PP=2 on the 2-node
# GB10 pods (216 GB weights + CUDA ctx > 2x121 GB; see repo memory
# glm-iq1s-empty-vllm-layout-2026-07-09). Unsloth's own runbook
# (https://unsloth.ai/docs/models/glm-5.2) recommends llama.cpp for the
# 1/2-bit dynamic quants. llama.cpp tensor-splits across the 8 L40S GPUs.
#
# Offline invariants: model is pre-staged on Lustre; llama.cpp binary is
# built with -DLLAMA_CURL=OFF (no download capability at all).
#
# Usage:
#   sbatch serve-glm52-iq1s-llamacpp.sbatch
#   # then from login1:
#   curl http://ai-l40s-00:12800/v1/models
#   # unsloth recommended sampling: temperature=1.0 top_p=0.95 min_p=0.01
#   # thinking modes: --chat-template-kwargs '{"enable_thinking":false}'
#   #                 or '{"reasoning_effort":"high"|"max"}' per request.

set -euo pipefail

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

LLAMA_BIN="${LLAMA_BIN:-/lvs0/rccs-dmp/${USER}/tools/llama.cpp/build-l40s/bin/llama-server}"
MODEL_GGUF="${MODEL_GGUF:-/lvs0/dne1/rccs-dmp/${USER}/umrs_models/glm-5.2-iq1s/UD-IQ1_S/GLM-5.2-UD-IQ1_merged.gguf}"
PORT="${GLM52_PORT:-12800}"
CTX="${GLM52_CTX:-65536}"          # model ceiling is 1,048,576; KV-quantized q4_1 keeps this cheap
NGL="${GLM52_NGL:-999}"            # offload all layers; llama.cpp splits across 8 GPUs
API_KEY="${GLM52_API_KEY:-}"       # optional bearer token

[ -x "$LLAMA_BIN" ] || { echo "FATAL: llama-server not found/executable: $LLAMA_BIN"; exit 1; }
[ -f "$MODEL_GGUF" ] || { echo "FATAL: model GGUF missing: $MODEL_GGUF"; exit 1; }

echo "[glm52-iq1s] node: $(hostname)"
echo "[glm52-iq1s] binary: $LLAMA_BIN"
echo "[glm52-iq1s] model: $MODEL_GGUF"
echo "[glm52-iq1s] port: $PORT  ctx: $CTX  ngl: $NGL"
nvidia-smi -L

ARGS=(
    --model "$MODEL_GGUF"
    --host 0.0.0.0
    --port "$PORT"
    --ctx-size "$CTX"
    --n-gpu-layers "$NGL"
    --split-mode layer
    --flash-attn on
    --cache-type-k q4_1
    --cache-type-v q4_1
    --jinja
    --temp 1.0
    --top-p 0.95
    --min-p 0.01
    --parallel 4
    --alias glm-5.2-iq1s
    --no-warmup
)
[ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")

exec "$LLAMA_BIN" "${ARGS[@]}"
