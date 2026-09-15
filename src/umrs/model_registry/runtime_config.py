"""
RuntimeConfig — authoritative runtime configuration for a model.

Stored as JSONB in ``model_catalog_entries.runtime_config``, this replaces
the implicit configuration currently derived in ``serve.slurm`` bash from
model-name pattern matching.

Each field maps to a vLLM CLI flag or environment variable that was previously
set by case-matching on the model name.  By storing these in the registry:

- Provisioning (adapters.py) reads runtime_config and injects env vars
- serve.slurm receives explicit VLLM_* vars; bash derivations become fallback-only
- Preflight validation can check parser availability against LauncherStack
- Certification stores the tested configuration alongside the stack identity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeConfig:
    """Model runtime configuration — stored as JSONB in the registry.

    All fields that were previously derived in serve.slurm bash pattern matching.
    Each field is optional — empty/None means "use serve.slurm fallback" for
    backward compatibility during migration.
    """

    #: Model family identifier (e.g. "deepseek_v4", "glm", "kimi_k2").
    #: Used to select family-specific behavior when capability flags are insufficient.
    model_family: str | None = None

    #: vLLM tool-call parser name (e.g. "deepseek_v4", "glm47", "kimi_k2").
    #: Set to empty string "" to explicitly disable tool-call parser.
    #: None = not configured (falls back to serve.slurm derivation).
    tool_call_parser: str | None = None

    #: vLLM reasoning parser name (e.g. "glm45", "kimi_k2", "gemma4").
    #: Set to empty string "" to explicitly disable reasoning parser.
    #: None = not configured (falls back to serve.slurm derivation).
    reasoning_parser: str | None = None

    #: KV-cache dtype override (e.g. "fp8" for DeepSeek-V4 FlashMLA).
    #: None = use vLLM default ("auto").
    kv_cache_dtype: str | None = None

    #: Force CUDA eager mode (disable graph capture).
    #: True for DeepSeek-V4 (garbled output) and GLM-5.1 (NaN on tools).
    #: None = not configured (falls back to serve.slurm heuristic).
    enforce_eager: bool | None = None

    #: Whether FlashInfer-TRTLLM MoE kernel works for this model.
    #: False for GLM (NaN on routing) and llm-jp (expert dim not %128).
    #: None = not configured (falls back to serve.slurm derivation).
    flashinfer_moe_fp16: bool | None = None

    #: Chat template content format for vLLM.
    #: "string" for GLM family per upstream recipe.
    #: None = use vLLM default ("auto").
    chat_template_format: str | None = None

    #: Multi-modal encoder tensor-parallel mode.
    #: "data" for Kimi K2.7 Code (MoonViT data-parallel).
    #: None = not configured.
    mm_encoder_tp_mode: str | None = None

    #: Whether the model requires --trust-remote-code.
    #: Almost always True for custom-architecture models.
    #: None = not configured (defaults to True for most families).
    trust_remote_code: bool | None = None

    #: Whether to enable --enable-auto-tool-choice.
    #: True for models that support tool calling natively.
    #: None = not configured (defaults to True for most families).
    enable_auto_tool_choice: bool | None = None

    #: Whether to enable --enable-prefix-caching (APC).
    #: True for most models; False if KV-cache pressure is critical.
    #: None = not configured (defaults to True).
    enable_prefix_caching: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (omit None values)."""
        d: dict[str, Any] = {}
        if self.model_family is not None:
            d["model_family"] = self.model_family
        if self.tool_call_parser is not None:
            d["tool_call_parser"] = self.tool_call_parser
        if self.reasoning_parser is not None:
            d["reasoning_parser"] = self.reasoning_parser
        if self.kv_cache_dtype is not None:
            d["kv_cache_dtype"] = self.kv_cache_dtype
        if self.enforce_eager is not None:
            d["enforce_eager"] = self.enforce_eager
        if self.flashinfer_moe_fp16 is not None:
            d["flashinfer_moe_fp16"] = self.flashinfer_moe_fp16
        if self.chat_template_format is not None:
            d["chat_template_format"] = self.chat_template_format
        if self.mm_encoder_tp_mode is not None:
            d["mm_encoder_tp_mode"] = self.mm_encoder_tp_mode
        if self.trust_remote_code is not None:
            d["trust_remote_code"] = self.trust_remote_code
        if self.enable_auto_tool_choice is not None:
            d["enable_auto_tool_choice"] = self.enable_auto_tool_choice
        if self.enable_prefix_caching is not None:
            d["enable_prefix_caching"] = self.enable_prefix_caching
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeConfig":
        """Deserialize from a JSON-decoded dict."""
        if not data:
            return cls()
        return cls(
            model_family=data.get("model_family"),
            tool_call_parser=data.get("tool_call_parser"),
            reasoning_parser=data.get("reasoning_parser"),
            kv_cache_dtype=data.get("kv_cache_dtype"),
            enforce_eager=data.get("enforce_eager"),
            flashinfer_moe_fp16=data.get("flashinfer_moe_fp16"),
            chat_template_format=data.get("chat_template_format"),
            mm_encoder_tp_mode=data.get("mm_encoder_tp_mode"),
            trust_remote_code=data.get("trust_remote_code"),
            enable_auto_tool_choice=data.get("enable_auto_tool_choice"),
            enable_prefix_caching=data.get("enable_prefix_caching"),
        )

    def effective_tool_call_parser(self) -> str | None:
        """Return the effective parser, treating "" as explicit disable."""
        if self.tool_call_parser == "":
            return None
        return self.tool_call_parser

    def effective_reasoning_parser(self) -> str | None:
        """Return the effective reasoning parser, treating "" as disable."""
        if self.reasoning_parser == "":
            return None
        return self.reasoning_parser
