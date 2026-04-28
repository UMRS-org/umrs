"""
Auto-detect model capabilities from local model files.

Reads config.json, tokenizer_config.json, and generation_config.json
to infer capabilities like tool calling, vision, context length, etc.

This is the server-side (UMRS) detection module, used by both the
model download job and the detect API endpoint.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .models import ModelCapabilities

logger = logging.getLogger(__name__)

# Architecture classes that indicate vision support
_VISION_ARCHITECTURES = {
    "qwen2vlforconditionalgeneration",
    "qwen2_5_vlforconditionalgeneration",
    "qwenvlforconditionalgeneration",
    "llavaforconditionalgeneration",
    "llavanextforconditionalgeneration",
    "llavanextvideo",
    "internvlchatmodel",
    "phi3vforconditionalgeneration",
    "minicpmv",
    "cogvlmforcausallm",
    "deepseekvisuallanguagemodel",
    "mantaforconditionalgeneration",
    "pixtralforconditionalgeneration",
    "molmoforconditionalgeneration",
    "chameleonforconditionalgeneration",
    "fuyu",
    "paligemmaforconditionalgeneration",
    "glm4vforcausallm",
    "idefics2forconditionalgeneration",
    "idefics3forconditionalgeneration",
}

# Patterns in chat_template that indicate tool calling support
_TOOL_CALLING_RE = re.compile(
    r"<tool_call>|<\|tool_call\|>|\"tool_calls\"|tool_call|"
    r"\"function\"|<function|<\|plugin\|>|tools|"
    r"<\|tool_start\|>|<\|tool_end\|>",
    re.IGNORECASE,
)

# Model type → normalized family
_FAMILY_MAP = {
    "qwen2": "qwen", "qwen2_moe": "qwen", "qwen": "qwen", "qwen2_vl": "qwen",
    "qwen3": "qwen", "qwen3_moe": "qwen",
    "llama": "llama",
    "mistral": "mistral", "mixtral": "mistral",
    "gemma": "gemma", "gemma2": "gemma",
    "phi3": "phi", "phi": "phi",
    "deepseek": "deepseek", "deepseek_v2": "deepseek", "deepseek_v3": "deepseek",
    "internlm": "internlm", "internlm2": "internlm",
    "chatglm": "glm", "glm": "glm", "glm4": "glm",
    "starcoder2": "starcoder", "starcoder": "starcoder",
    "codellama": "codellama",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _normalize_family(model_type: str) -> str:
    """Normalize model_type string to a canonical family name."""
    lower = model_type.lower()
    # Direct lookup
    if lower in _FAMILY_MAP:
        return _FAMILY_MAP[lower]
    # Prefix-based matching
    for prefix in ("qwen", "llama", "mistral", "mixtral", "phi", "gemma",
                    "deepseek", "internlm", "chatglm", "glm", "starcoder",
                    "codellama"):
        if lower.startswith(prefix):
            return _FAMILY_MAP.get(prefix, prefix)
    return lower


def detect_capabilities(model_path: str) -> Optional[ModelCapabilities]:
    """
    Detect model capabilities from files in model_path.

    Returns ModelCapabilities if anything was detected, None otherwise.
    """
    base = Path(model_path)
    if not base.is_dir():
        logger.debug("Model path %s is not a directory", model_path)
        return None

    config = _read_json(base / "config.json")
    if config is None:
        logger.debug("No config.json in %s", model_path)
        return None

    # --- Model family ---
    model_type = config.get("model_type")
    model_family = _normalize_family(model_type) if model_type else None

    # --- Vision ---
    architectures = config.get("architectures", [])
    supports_vision = False
    if architectures:
        arch_lower = {a.lower() for a in architectures}
        supports_vision = bool(arch_lower & _VISION_ARCHITECTURES)
    if not supports_vision:
        vision_keys = {"vision_config", "visual", "image_size", "vision_tower"}
        if vision_keys & set(config.keys()):
            supports_vision = True

    # --- Context length ---
    max_pos = config.get("max_position_embeddings")
    max_position_embeddings = max_pos if isinstance(max_pos, int) and max_pos > 0 else None

    # --- Tool calling ---
    supports_tool_calling = False
    tok_config = _read_json(base / "tokenizer_config.json")
    if tok_config:
        chat_template = tok_config.get("chat_template", "")
        if isinstance(chat_template, list):
            chat_template = " ".join(
                t.get("template", "") if isinstance(t, dict) else str(t)
                for t in chat_template
            )
        if isinstance(chat_template, str) and chat_template:
            supports_tool_calling = bool(_TOOL_CALLING_RE.search(chat_template))

    # --- Structured output ---
    supports_structured_output = supports_tool_calling

    # --- Max output tokens ---
    max_output_tokens = None
    gen_config = _read_json(base / "generation_config.json")
    if gen_config:
        max_new = gen_config.get("max_new_tokens")
        if isinstance(max_new, int) and max_new > 0:
            max_output_tokens = max_new

    logger.info(
        "Detected capabilities for %s: family=%s tools=%s vision=%s ctx=%s",
        model_path, model_family, supports_tool_calling, supports_vision,
        max_position_embeddings,
    )

    return ModelCapabilities(
        supports_tool_calling=supports_tool_calling,
        supports_vision=supports_vision,
        supports_structured_output=supports_structured_output,
        max_output_tokens=max_output_tokens,
        model_family=model_family,
    )
