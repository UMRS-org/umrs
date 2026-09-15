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
from typing import Any

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
    # Kimi K2.x multimodal LLMs (MoonViT; K2.6 / K2.7-Code)
    "kimik25forconditionalgeneration",
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
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen": "qwen",
    "qwen2_vl": "qwen",
    "qwen3": "qwen",
    "qwen3_moe": "qwen",
    "llama": "llama",
    "mistral": "mistral",
    "mixtral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma",
    "phi3": "phi",
    "phi": "phi",
    "deepseek": "deepseek",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
    "internlm": "internlm",
    "internlm2": "internlm",
    "chatglm": "glm",
    "glm": "glm",
    "glm4": "glm",
    "starcoder2": "starcoder",
    "starcoder": "starcoder",
    "codellama": "codellama",
}


def _read_json(path: Path) -> dict[str, Any] | None:
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
    for prefix in (
        "qwen",
        "llama",
        "mistral",
        "mixtral",
        "phi",
        "gemma",
        "deepseek",
        "internlm",
        "chatglm",
        "glm",
        "starcoder",
        "codellama",
    ):
        if lower.startswith(prefix):
            return _FAMILY_MAP.get(prefix, prefix)
    return lower


def detect_capabilities(
    model_path: str,
    model_name: str | None = None,
) -> ModelCapabilities | None:
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

    tok_config = _read_json(base / "tokenizer_config.json")
    gen_config = _read_json(base / "generation_config.json")

    return detect_capabilities_from_files(
        config, tok_config, gen_config, model_name=model_name
    )


def detect_capabilities_from_files(
    config: dict[str, Any] | None,
    tok_config: dict[str, Any] | None,
    gen_config: dict[str, Any] | None,
    model_name: str | None = None,
) -> ModelCapabilities | None:
    """
    Detect capabilities from pre-loaded config dicts.

    Each argument is the parsed JSON contents of the corresponding model
    file (``config.json``, ``tokenizer_config.json``,
    ``generation_config.json``). ``None`` means the file was missing or
    unreadable.

    Returns ModelCapabilities if anything was detected, None otherwise.
    """
    if config is None:
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
        vision_keys = {
            "vision_config",
            "visual",
            "image_size",
            "vision_tower",
            "media_placeholder_token_id",
            "mm_hidden_size",
            "mm_projector_type",
        }
        if vision_keys & set(config.keys()):
            supports_vision = True

    # --- Context length ---
    max_pos = config.get("max_position_embeddings")
    # Fallback: check nested text_config (e.g. Gemma multimodal models)
    if not isinstance(max_pos, int) or max_pos <= 0:
        text_cfg = config.get("text_config")
        if isinstance(text_cfg, dict):
            max_pos = text_cfg.get("max_position_embeddings")
    max_position_embeddings = (
        max_pos if isinstance(max_pos, int) and max_pos > 0 else None
    )

    # --- Tool calling ---
    supports_tool_calling = False
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
    if gen_config:
        max_new = gen_config.get("max_new_tokens")
        if isinstance(max_new, int) and max_new > 0:
            max_output_tokens = max_new

    logger.info(
        "Detected capabilities: family=%s tools=%s vision=%s ctx=%s",
        model_family,
        supports_tool_calling,
        supports_vision,
        max_position_embeddings,
    )

    caps = ModelCapabilities(
        supports_tool_calling=supports_tool_calling,
        supports_vision=supports_vision,
        supports_structured_output=supports_structured_output,
        max_output_tokens=max_output_tokens,
        max_position_embeddings=max_position_embeddings,
        model_family=model_family,
    )
    if model_name:
        from umrs.model_registry.capabilities import resolve_capabilities

        return resolve_capabilities(model_name, caps)
    return caps


def merge_detected_capabilities(
    caps: ModelCapabilities,
    model_path: str,
    model_name: str | None = None,
) -> ModelCapabilities:
    """Merge filesystem-detected capabilities into ``caps`` without overwriting explicit flags."""
    detected = detect_capabilities(model_path, model_name=model_name)
    if detected is None:
        return caps

    if detected.supports_vision:
        caps.supports_vision = True
    if detected.supports_tool_calling:
        caps.supports_tool_calling = True
    if detected.supports_structured_output:
        caps.supports_structured_output = True
    if detected.max_output_tokens is not None and caps.max_output_tokens is None:
        caps.max_output_tokens = detected.max_output_tokens
    if (
        detected.max_position_embeddings is not None
        and caps.max_position_embeddings is None
    ):
        caps.max_position_embeddings = detected.max_position_embeddings
    if detected.model_family and not caps.model_family:
        caps.model_family = detected.model_family
    return caps
