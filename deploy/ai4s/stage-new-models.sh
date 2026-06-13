#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Download 4 new AI4S models on AI4S Lustre
# ─────────────────────────────────────────────────────────────
# Run this on the AI4S login node (ea0022@134.160.189.24) where
# Lustre /work is accessible and outbound HTTPS is allowed.
#
# Models:
#   1. moonshotai/Kimi-K2.7-Code            (~500 GB, 4× B200, INT4)
#   2. tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1-MXFP4  (~60 GB, 1× B200, MXFP4)
#   3. llm-jp/llm-jp-4-32b-a3b-thinking     (~64 GB, 1× B200, BF16)
#   4. google/diffusiongemma-26B-A4B-it     (~50 GB, 1× B200, BF16)
#
# Prerequisites:
#   pip install --user 'huggingface-hub[cli,hf_transfer]'
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

echo "=== AI4S New Model Download ==="
echo ""
echo "  Target directory: ${MODEL_DIR}"
echo "  Models:"
echo "    1. moonshotai/Kimi-K2.7-Code             (~500 GB, INT4, 4× B200)"
echo "    2. tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1-MXFP4  (~60 GB, MXFP4, 1× B200)"
echo "    3. llm-jp/llm-jp-4-32b-a3b-thinking      (~64 GB, BF16, 1× B200)"
echo "    4. google/diffusiongemma-26B-A4B-it      (~50 GB, BF16, 1× B200)"
echo ""
echo "  Total: ~674 GB"
echo "  Consider running in tmux/screen — Kimi-K2.7-Code will take hours."
echo ""

# Smaller models first (quick feedback), then the big one.
download_model "tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1-MXFP4"  "${MODEL_DIR}/GPT-OSS-Swallow-120B-RL-v0.1-MXFP4"
download_model "llm-jp/llm-jp-4-32b-a3b-thinking"                 "${MODEL_DIR}/llm-jp-4-32b-a3b-thinking"
download_model "google/diffusiongemma-26B-A4B-it"                 "${MODEL_DIR}/diffusiongemma-26B-A4B-it"

# Kimi last — largest download.
download_model "moonshotai/Kimi-K2.7-Code"                        "${MODEL_DIR}/Kimi-K2.7-Code"

echo ""
echo "=== All downloads complete ==="
echo ""
echo "Register the model(s) if not already registered:"
echo ""
echo "  r4 model register Kimi-K2.7-Code \\"
echo "    --path /work/hps0/home/ea0022/umrs_models/Kimi-K2.7-Code \\"
echo "    --scope system --type llm --backend vllm \\"
echo "    --service-kind openai \\"
echo "    --hf-id moonshotai/Kimi-K2.7-Code \\"
echo "    --max-model-len 245760 --gpu-mem-util 0.9 --tp 4 \\"
echo "    --no-detect"
echo ""
echo "Then bring up endpoints with:"
echo ""
echo "  R4_GATEWAY_URL=http://localhost:8002 r4 pool up pool-ai4s.yaml"
echo ""
