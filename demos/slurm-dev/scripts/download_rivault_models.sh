#!/usr/bin/env bash
set -euo pipefail

# Downloads a set of HF models into the rivault/models directory.
# Usage:
#   ./download_rivault_models.sh                # download defaults
#   ./download_rivault_models.sh repo/id other/repo
#
# It will try to use the `hf` CLI if present, otherwise falls back to
# `huggingface_hub.snapshot_download` via Python. Set HF_TOKEN in the
# environment for authenticated, higher-rate downloads.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_BASE="$SCRIPT_DIR/../../src/umrs/mini-rivault-llm/models"
mkdir -p "$TARGET_BASE"


# Helper: map short model names to HF repo IDs
map_model() {
  local m="$1"
  # if contains a slash, assume it's already a repo id
  if [[ "$m" == *"/"* ]]; then
    echo "$m" && return
  fi
  case "$m" in
    clip-vit-base-patch32|clip-ViT-B-32|clip-vit-base)
      echo "openai/clip-vit-base-patch32" ;;
    e5-base-v2|e5-base)
      echo "intfloat/e5-base-v2" ;;
    all-mpnet-base-v2)
      echo "sentence-transformers/all-mpnet-base-v2" ;;
    all-MiniLM-L6-v2|all-minilm-l6-v2)
      echo "sentence-transformers/all-MiniLM-L6-v2" ;;
    sam3|sam2-hiera-small|sam2-hiera)
      echo "facebook/sam2-hiera-small" ;;
    *)
      # fallback: try prefixing with sentence-transformers
      echo "sentence-transformers/$m" ;;
  esac
}

# Helper: map HF repo IDs to local output directory names.
# SAM3 must land at models/sam3 so the mcp-vision container finds it via
# VISION_MODEL_PATH (defaults to ./models/sam3 in serve-model.slurm).
map_outdir() {
  local repo="$1"
  case "$repo" in
    facebook/sam2*)
      echo "sam3" ;;
    *)
      echo "${repo##*/}" ;;
  esac
}

# If user provided models on CLI, use them. Otherwise extract from user config JSON.
if [ "$#" -gt 0 ]; then
  RAW_MODELS=("$@")
else
  # read model_name entries from demos/slurm-dev/user_files/user_rivault.json
  USER_CFG="$SCRIPT_DIR/../user_files/user_rivault.json"
  if [ -f "$USER_CFG" ] && command -v jq >/dev/null 2>&1; then
    mapfile -t RAW_MODELS < <(jq -r '.experiences[].modalities[].model.model_name' "$USER_CFG" | sed -e '/^$/d')
  else
    # fallback defaults
    RAW_MODELS=("openai/clip-vit-base-patch32" "intfloat/e5-base-v2")
  fi
fi

# Map and deduplicate
MODELS=()
declare -A seen
for m in "${RAW_MODELS[@]}"; do
  repo=$(map_model "$m")
  if [ -z "${seen[$repo]:-}" ]; then
    MODELS+=("$repo")
    seen[$repo]=1
  fi
done

echo "Target base directory: $TARGET_BASE"
echo "HF token present: ${HF_TOKEN:+yes}" || true

if command -v hf >/dev/null 2>&1; then
  echo "Using 'hf' CLI to download models"
  for m in "${MODELS[@]}"; do
    name="$(map_outdir "$m")"
    outdir="$TARGET_BASE/$name"
    mkdir -p "$outdir"
    echo "Downloading $m -> $outdir"
    # Try hf download; some hf versions support --local-dir
    if hf download "$m" --local-dir "$outdir" 2>/dev/null; then
      echo "  -> ok"
    else
      echo "  -> hf download failed, falling back to python snapshot_download"
      python3 - <<PY
from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id="$m", local_dir="$outdir")
print('  -> ok')
PY
    fi
  done
else
  echo "'hf' CLI not found; using python huggingface_hub.snapshot_download"
  for m in "${MODELS[@]}"; do
    name="$(map_outdir "$m")"
    outdir="$TARGET_BASE/$name"
    mkdir -p "$outdir"
    echo "Downloading $m -> $outdir"
    python3 - <<PY
from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id="$m", local_dir="$outdir")
print('  -> ok')
PY
  done
fi

echo "All done. Models saved under: $TARGET_BASE"

# Install spaCy models referenced in the user config (if any)
SPACY_MODELS=()
USER_CFG="$SCRIPT_DIR/../user_files/user_rivault.json"
if [ -f "$USER_CFG" ] && command -v jq >/dev/null 2>&1; then
  mapfile -t SPACY_MODELS < <(jq -r '..|.spacy_model? // empty' "$USER_CFG" | sed -e '/^$/d' | sort -u)
fi

if [ "${#SPACY_MODELS[@]}" -gt 0 ]; then
  echo "SpaCy models to install: ${SPACY_MODELS[*]}"
  # Ensure we have a Python executable that can run spacy and install models.
  PY_EXEC=""
  if python3 -c 'import spacy' >/dev/null 2>&1; then
    PY_EXEC=python3
  else
    echo "spaCy not found; attempting installation using pip"
    # Try system/user pip install first
    if python3 -m pip install -U pip setuptools wheel >/dev/null 2>&1 && \
       python3 -m pip install -U spacy >/dev/null 2>&1; then
      PY_EXEC=python3
    else
      echo "System pip install failed; attempting user install (--user)"
      if python3 -m pip install --user -U spacy >/dev/null 2>&1; then
        PY_EXEC=python3
        export PATH="$HOME/.local/bin:$PATH"
      else
        echo "User pip install failed; attempting to create a virtualenv at $TARGET_BASE/.venv"
        if python3 -m venv "$TARGET_BASE/.venv" >/dev/null 2>&1; then
          PY_EXEC="$TARGET_BASE/.venv/bin/python"
          "$PY_EXEC" -m pip install -U pip setuptools wheel spacy >/dev/null 2>&1 || true
        else
          echo "Failed to create virtualenv; spaCy model installation will be skipped." >&2
          PY_EXEC=""
        fi
      fi
    fi
  fi

  for sm in "${SPACY_MODELS[@]}"; do
    echo "Installing spaCy model: $sm"
    if [ -n "$PY_EXEC" ]; then
      if "$PY_EXEC" -m spacy download "$sm" >/dev/null 2>&1; then
        echo "  -> $sm installed"
      else
        echo "  -> failed to install $sm via 'spacy download' (tried with $PY_EXEC)"
      fi
    else
      echo "  -> skipping $sm: no available Python environment for installation"
    fi
  done
fi
