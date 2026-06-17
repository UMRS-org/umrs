#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Download DeepSeek-V4-Pro, GLM-5.1, Kimi-K2.6 on AI4S Lustre
# ─────────────────────────────────────────────────────────────
# Run this on the AI4S login node (ea0022@134.160.189.24) where
# Lustre /work is accessible and outbound HTTPS is allowed.
#
# Prerequisites:
#   pip install --user 'huggingface-hub[cli,hf_transfer]'
#
# Models are already registered in UMRS (r4 model list shows them).
# This script downloads the weights to the expected paths.
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

echo "=== AI4S Pool Model Download ==="
echo ""
echo "  Target directory: ${MODEL_DIR}"
echo "  Models:"
echo "    1. deepseek-ai/DeepSeek-V4-Pro  (~1.7 TB, 8× B200)"
echo "    2. zai-org/GLM-5.1              (~1.5 TB, 8× B200)"
echo "    3. zai-org/GLM-5.2              (~1.5 TB, 16× B200, 1M ctx)"
echo "    4. moonshotai/Kimi-K2.6         (~500 GB, 4× B200)"
echo ""
echo "  Consider running in tmux/screen — downloads will take hours."
echo ""

download_model "deepseek-ai/DeepSeek-V4-Pro" "${MODEL_DIR}/DeepSeek-V4-Pro"
download_model "zai-org/GLM-5.1"             "${MODEL_DIR}/GLM-5.1"
download_model "zai-org/GLM-5.2"             "${MODEL_DIR}/GLM-5.2"
download_model "moonshotai/Kimi-K2.6"        "${MODEL_DIR}/Kimi-K2.6"

echo ""
echo "=== All downloads complete ==="
echo ""
echo "Models are already registered in UMRS. Bring up endpoints with:"
echo ""
echo "  r4 pool up pool-ai4s.yaml"
echo ""
