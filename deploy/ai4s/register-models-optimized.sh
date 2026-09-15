#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Register all models needed for pool-ai4s-optimized.yaml
# ─────────────────────────────────────────────────────────────
# Run this on umrs-front after staging the model weights.
#
# Prerequisites:
#   - Model weights staged at /work/hps0/home/ea0022/umrs_models/
#   - R4_GATEWAY_URL set (e.g., http://localhost:8002)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_DIR="/work/hps0/home/ea0022/umrs_models"

# Check if a model is already registered
is_registered() {
    local name="$1"
    r4 model list 2>/dev/null | grep -q "^${name}\b" && return 0 || return 1
}

register_model() {
    local name="$1"
    local path="$2"
    local hf_id="$3"
    local max_len="$4"
    local tp="$5"

    if is_registered "$name"; then
        echo "  ✓ ${name} already registered"
        backfill_capabilities "$name"
        return 0
    fi

    if [[ ! -d "$path" ]]; then
        echo "  ✗ ${name}: model path not found: ${path}"
        echo "    Stage the model first: ./stage-ornith-1.0-397b.sh"
        return 1
    fi

    echo "  Registering ${name}..."
    r4 model register "$name" \
        --path "$path" \
        --scope system --type llm --backend vllm \
        --service-kind openai \
        --hf-id "$hf_id" \
        --max-model-len "$max_len" --gpu-mem-util 0.90 --tp "$tp"
    echo "  ✓ ${name} registered"
    backfill_capabilities "$name"
}

# Backfill vision/tool capabilities from config.json (also fixes already-registered models).
backfill_capabilities() {
    local name="$1"
    echo "  Detecting capabilities for ${name}..."
    r4 model detect "$name" || true
}

echo "=== Registering models for pool-ai4s-optimized.yaml ==="
echo ""

# DeepSeek-V4-Pro (1.6T MoE, 49B active)
register_model "DeepSeek-V4-Pro" \
    "${MODEL_DIR}/DeepSeek-V4-Pro" \
    "deepseek-ai/DeepSeek-V4-Pro" \
    "1048576" \
    "8"

# GLM-5.2 (753B MoE)
register_model "GLM-5.2" \
    "${MODEL_DIR}/GLM-5.2" \
    "zai-org/GLM-5.2" \
    "1048576" \
    "8"

# GLM-5.2-FP8 (FP8 quantized variant)
register_model "GLM-5.2-FP8" \
    "${MODEL_DIR}/GLM-5.2-FP8" \
    "zai-org/GLM-5.2-FP8" \
    "1048576" \
    "8"

# GLM-5.2-NVFP4 (NVIDIA NVFP4 quantization, 4-bit MoE experts)
register_model "GLM-5.2-NVFP4" \
    "${MODEL_DIR}/GLM-5.2-NVFP4" \
    "nvidia/GLM-5.2-NVFP4" \
    "1048576" \
    "8"

# Kimi-K2.6 (1T MoE, INT4)
register_model "Kimi-K2.6" \
    "${MODEL_DIR}/Kimi-K2.6" \
    "moonshotai/Kimi-K2.6" \
    "262144" \
    "4"

# Kimi-K2.7-Code (coding-focused agentic MoE)
register_model "Kimi-K2.7-Code" \
    "${MODEL_DIR}/Kimi-K2.7-Code" \
    "moonshotai/Kimi-K2.7-Code" \
    "262144" \
    "4"

# Ornith-1.0-397B (DeepReinforce, 397B MoE agentic coding)
register_model "Ornith-1.0-397B" \
    "${MODEL_DIR}/Ornith-1.0-397B" \
    "deepreinforce-ai/Ornith-1.0-397B" \
    "262144" \
    "8"

echo ""
echo "=== Registration complete ==="
echo ""
echo "Verify registrations:"
echo "  r4 model list"
echo ""
echo "Bring up the optimized pool:"
echo "  r4 pool up ../../umrs/deploy/ai4s/pool-ai4s-optimized.yaml"
echo ""
