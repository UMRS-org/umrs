#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Download Ornith-1.0-397B on AI4S Lustre
# ─────────────────────────────────────────────────────────────
# Run this on the AI4S login node (ea0022@134.160.189.24) where
# Lustre /work is accessible and outbound HTTPS is allowed.
#
# Prerequisites:
#   pip install --user 'huggingface-hub[cli,hf_transfer]'
#
# After download, register the model in UMRS:
#   r4 model register Ornith-1.0-397B \
#     --path /work/hps0/home/ea0022/umrs_models/Ornith-1.0-397B \
#     --scope system --type llm --backend vllm \
#     --service-kind openai \
#     --hf-id deepreinforce-ai/Ornith-1.0-397B \
#     --max-model-len 262144 --gpu-mem-util 0.90 --tp 8 \
#     --no-detect
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

echo "=== AI4S Ornith-1.0-397B Download ==="
echo ""
echo "  Target directory: ${MODEL_DIR}"
echo "  Model: deepreinforce-ai/Ornith-1.0-397B"
echo "  Size: ~794 GB (397B MoE, BF16)"
echo ""
echo "  Consider running in tmux — download will take hours."
echo ""

download_model "deepreinforce-ai/Ornith-1.0-397B" "${MODEL_DIR}/Ornith-1.0-397B"

echo ""
echo "=== Download complete ==="
echo ""
echo "Register the model in UMRS:"
echo ""
echo "  r4 model register Ornith-1.0-397B \\"
echo "    --path /work/hps0/home/ea0022/umrs_models/Ornith-1.0-397B \\"
echo "    --scope system --type llm --backend vllm \\"
echo "    --service-kind openai \\"
echo "    --hf-id deepreinforce-ai/Ornith-1.0-397B \\"
echo "    --max-model-len 262144 --gpu-mem-util 0.90 --tp 8 \\"
echo "    --no-detect"
echo ""
echo "Then bring up the endpoint:"
echo ""
echo "  r4 pool up pool-ai4s-optimized.yaml"
echo ""
