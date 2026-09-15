#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Download nvidia/GLM-5.2-NVFP4 on AI4S Lustre
# ─────────────────────────────────────────────────────────────
# Run this on the AI4S login node (ea0022@134.160.189.24) where
# Lustre /work is accessible and outbound HTTPS is allowed.
#
# Model: nvidia/GLM-5.2-NVFP4
#   - 753B total / 40B activated (MoE, glm_moe_dsa)
#   - NVIDIA NVFP4 quantization (~376 GiB on disk)
#   - 1M native context
#   - vLLM 0.23.0+ required (--quantization modelopt_fp4)
#
# Prerequisites:
#   pip install --user 'huggingface-hub[cli,hf_transfer]'
#
# After download, register the model in UMRS:
#   r4 model register GLM-5.2-NVFP4 \
#     --path /work/hps0/home/ea0022/umrs_models/GLM-5.2-NVFP4 \
#     --scope system --type llm --backend vllm \
#     --service-kind openai \
#     --hf-id nvidia/GLM-5.2-NVFP4 \
#     --max-model-len 1048576 --gpu-mem-util 0.90 --tp 8 \
#     --no-detect
#
# Then bring up the endpoint:
#   r4 pool up ../../umrs/deploy/ai4s/pool-ai4s-optimized.yaml
# ─────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_DIR="/work/hps0/home/ea0022/umrs_models"
LUSTRE_STRIPE_COUNT="${LUSTRE_STRIPE_COUNT:-8}"
LUSTRE_STRIPE_SIZE="${LUSTRE_STRIPE_SIZE:-4M}"

# Enable fast parallel downloads if available
if python3 -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "hf_transfer: enabled"
fi

export PATH="$HOME/.local/bin:$PATH"

# Authenticate if token available (public models work without, but faster with)
if [[ -f "$HOME/.hf_token" ]]; then
    export HF_TOKEN="$(< "$HOME/.hf_token")"
    echo "HF_TOKEN: loaded from ~/.hf_token"
fi

download_model() {
    local hf_id="$1"
    local dest="$2"

    echo ""
    echo "=== Downloading ${hf_id} ==="
    echo "  Destination: ${dest}"

    mkdir -p "$dest"

    # Lustre striping for large files
    if command -v lfs &>/dev/null; then
        lfs setstripe -c "$LUSTRE_STRIPE_COUNT" -S "$LUSTRE_STRIPE_SIZE" "$dest" 2>/dev/null || true
    fi

    if command -v hf &>/dev/null; then
        hf download "$hf_id" --local-dir "$dest"
    elif command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$hf_id" --local-dir "$dest" --resume-download
    else
        echo "ERROR: HuggingFace CLI not found."
        echo "Install: pip install --user 'huggingface-hub[cli,hf_transfer]'"
        exit 1
    fi

    echo "  Done: $(du -sh "$dest" | cut -f1)"
}

echo "=== AI4S GLM-5.2-NVFP4 Download ==="
echo ""
echo "  Target directory: ${MODEL_DIR}"
echo "  Model: nvidia/GLM-5.2-NVFP4"
echo "  Size: ~376 GB (NVFP4 quantized, 753B MoE)"
echo ""
echo "  Consider running in tmux — download will take hours."
echo ""

download_model "nvidia/GLM-5.2-NVFP4" "${MODEL_DIR}/GLM-5.2-NVFP4"

echo ""
echo "=== Download complete ==="
echo ""
echo "Register the model in UMRS:"
echo ""
echo "  r4 model register GLM-5.2-NVFP4 \\"
echo "    --path /work/hps0/home/ea0022/umrs_models/GLM-5.2-NVFP4 \\"
echo "    --scope system --type llm --backend vllm \\"
echo "    --service-kind openai \\"
echo "    --hf-id nvidia/GLM-5.2-NVFP4 \\"
echo "    --max-model-len 1048576 --gpu-mem-util 0.90 --tp 8 \\"
echo "    --no-detect"
echo ""
echo "Then bring up the endpoint:"
echo ""
echo "  r4 pool up ../../umrs/deploy/ai4s/pool-ai4s-optimized.yaml"
echo ""
