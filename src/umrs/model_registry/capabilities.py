"""
Model Capabilities — declarative flags replacing serve.slurm bash pattern matching.

Each model declares what it can do and what it needs from the runtime.
The capability flags are stored in ``model_catalog_entries.capabilities`` (JSONB)
and drive DeploymentSpec generation and preflight validation.

Capabilities replace these serve.slurm bash case statements:

    *deepseek*v4*) echo "deepseek_v4" ;;         → family="deepseek_v4"
    *glm*) echo "glm47" ;;                       → family="glm"
    *kimi*) echo "kimi_k2" ;;                    → family="kimi_k2"

    *glm*) VLLM_USE_FLASHINFER_MOE_FP16=0 ;;     → requires_triton_moe_fallback=true
    *deepseek*v4*) --enforce-eager ;;            → requires_eager_mode=true
    *glm*) --chat-template-content-format string → requires_chat_template_string=true

New models declare capabilities.  No bash edits needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── Family → parser mapping (serves as default when registry empty) ──────────

FAMILY_TOOL_PARSER: dict[str, str] = {
    "deepseek_v4": "deepseek_v4",
    "deepseek_v3": "deepseek_v3",
    "deepseek": "deepseek_v3",
    "glm": "glm47",
    "kimi_k2": "kimi_k2",
    "kimi": "kimi_k2",
    "llm_jp": "hermes",
    "gpt_oss": "",  # No tool-call training
    "gemma4": "gemma4",
    "diffusion_gemma": "",  # No tool-call training
    "qwen": "hermes",
    "ornith": "qwen3_xml",
    "llama": "hermes",
    "mistral": "hermes",
}

FAMILY_REASONING_PARSER: dict[str, str] = {
    "glm": "glm45",
    "kimi_k2": "kimi_k2",
    "kimi": "kimi_k2",
    "gpt_oss": "gpt_oss",
    "gemma4": "gemma4",
    "llm_jp": "",  # llmjp4 plugin not in SIF
    "deepseek_v4": "",  # No response-gated reasoning
    "deepseek_v3": "",
    "deepseek": "",
    "diffusion_gemma": "gemma4",
    "qwen": "",
    "ornith": "qwen3",
    "llama": "",
    "mistral": "",
}

FAMILY_KV_CACHE_DTYPE: dict[str, str | None] = {
    "deepseek_v4": "fp8",  # FlashMLA fp8 attention
    "deepseek_v3": None,
    "deepseek": None,
    "glm": None,  # bf16 default (NOT fp8 — GLM has its own DSA code path)
    "kimi_k2": "fp8",  # CANARY: INT4 weights + fp8 KV free headroom on GB200
    "kimi": "fp8",  # CANARY: keep in sync with kimi_k2
    "llm_jp": None,
    "gpt_oss": None,
    "gemma4": None,
    "diffusion_gemma": None,
    "ornith": None,
}

# Families that require eager mode (no CUDA graph capture)
FAMILY_REQUIRES_EAGER: set[str] = {
    "deepseek_v4",  # CUDA graph garbage on current build
    "glm",  # NaN on tools path with CUDA graphs
}

# Families that must use Triton MoE fallback (FlashInfer-TRTLLM NaN)
FAMILY_REQUIRES_TRITON_MOE: set[str] = {
    "glm",  # NaN on GLM expert routing
    "llm_jp",  # expert intermediate_size=64 not %128, FlashInfer rejects
}

# Families that require --chat-template-content-format string
FAMILY_REQUIRES_CHAT_TEMPLATE_STRING: set[str] = {
    "glm",  # Upstream vLLM GLM recipe
}

# Families that require --mm-encoder-tp-mode data
FAMILY_REQUIRES_MM_ENCODER_TP_DATA: set[str] = {
    "kimi_k2",  # MoonViT data-parallel for multi-node (currently single-node K2.7)
}

# Per-model reasoning effort (catalog name patterns, most specific first).
# Used at registration to populate reasoning_effort_* on each catalog entry.
#
# Kimi K2.6: binary thinking (Kimi API thinking.type enabled|disabled; vLLM
# reasoning_effort none|high). K2.7 Code: thinking always on — no entry here.
MODEL_REASONING_EFFORT: list[tuple[str, list[str], str]] = [
    ("kimi", ["none", "high"], "high"),
    ("glm-5.2", ["none", "high", "max"], "max"),
    ("glm-5.1", ["none", "low", "medium", "high"], "medium"),
    ("glm", ["none", "low", "medium", "high"], "medium"),
    ("gpt-oss-swallow", ["medium"], "medium"),
    ("llm-jp-4", ["low", "medium", "high"], "medium"),
    ("deepseek-v4", ["none", "high", "max"], "high"),
    ("gemma-4", ["none", "low", "medium", "high"], "medium"),
    ("gemma_4", ["none", "low", "medium", "high"], "medium"),
    ("gemma 4", ["none", "low", "medium", "high"], "medium"),
    ("gemma4", ["none", "low", "medium", "high"], "medium"),
]


def derive_reasoning_effort(model_name: str) -> tuple[list[str], str] | None:
    """Match catalog model name against MODEL_REASONING_EFFORT (first win)."""
    name_lc = (model_name or "").lower()
    if not name_lc:
        return None
    # K2.7 Code: thinking always on; no user-facing effort control.
    if "kimi-k2.7" in name_lc or "kimi_k2.7" in name_lc:
        return None
    for pattern, levels, default in MODEL_REASONING_EFFORT:
        if pattern in name_lc:
            return (list(levels), default)
    return None


def resolve_capabilities(
    model_name: str,
    caps: ModelCapabilities | None = None,
) -> ModelCapabilities:
    """Merge explicit capabilities with derived reasoning-effort defaults."""
    caps = caps or ModelCapabilities()
    if caps.reasoning_effort_levels is None or caps.reasoning_effort_default is None:
        derived = derive_reasoning_effort(model_name)
        if derived is not None:
            levels, default = derived
            if caps.reasoning_effort_levels is None:
                caps.reasoning_effort_levels = levels
            if caps.reasoning_effort_default is None:
                caps.reasoning_effort_default = default
            if not caps.supports_reasoning and levels:
                caps.supports_reasoning = True
    return caps


@dataclass
class ModelCapabilities:
    """
    Declarative model capabilities.

    Replace serve.slurm bash pattern matching with explicit flags.
    Stored in model_catalog_entries.capabilities (JSONB).
    """

    # ── What the model CAN do ────────────────────────────────────────────────

    supports_tool_calling: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False  # Has  think blocks
    supports_structured_output: bool = False

    # ── What the model NEEDS from the runtime ────────────────────────────────

    requires_eager_mode: bool = False  # --enforce-eager
    requires_triton_moe_fallback: bool = False  # VLLM_USE_FLASHINFER_MOE_FP16=0
    requires_fp8_kv_cache: bool = False  # --kv-cache-dtype fp8
    requires_chat_template_string: bool = False  # --chat-template-content-format string
    requires_mm_encoder_tp_data: bool = False  # --mm-encoder-tp-mode data

    # ── Identity ─────────────────────────────────────────────────────────────

    model_family: str | None = None  # "glm", "deepseek_v4", "kimi_k2", etc.
    max_output_tokens: int | None = None
    max_position_embeddings: int | None = None

    # Per-model thinking effort (picker + API wire tokens)
    reasoning_effort_levels: list[str] | None = None
    reasoning_effort_default: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "supports_tool_calling": self.supports_tool_calling,
            "supports_vision": self.supports_vision,
            "supports_reasoning": self.supports_reasoning,
            "supports_structured_output": self.supports_structured_output,
            "requires_eager_mode": self.requires_eager_mode,
            "requires_triton_moe_fallback": self.requires_triton_moe_fallback,
            "requires_fp8_kv_cache": self.requires_fp8_kv_cache,
            "requires_chat_template_string": self.requires_chat_template_string,
            "requires_mm_encoder_tp_data": self.requires_mm_encoder_tp_data,
        }
        if self.model_family is not None:
            d["model_family"] = self.model_family
        if self.max_output_tokens is not None:
            d["max_output_tokens"] = self.max_output_tokens
        if self.max_position_embeddings is not None:
            d["max_position_embeddings"] = self.max_position_embeddings
        if self.reasoning_effort_levels is not None:
            d["reasoning_effort_levels"] = self.reasoning_effort_levels
        if self.reasoning_effort_default is not None:
            d["reasoning_effort_default"] = self.reasoning_effort_default
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ModelCapabilities:
        if not data:
            return cls()
        return cls(
            supports_tool_calling=data.get("supports_tool_calling", False),
            supports_vision=data.get("supports_vision", False),
            supports_reasoning=data.get("supports_reasoning", False),
            supports_structured_output=data.get("supports_structured_output", False),
            requires_eager_mode=data.get("requires_eager_mode", False),
            requires_triton_moe_fallback=data.get(
                "requires_triton_moe_fallback", False
            ),
            requires_fp8_kv_cache=data.get("requires_fp8_kv_cache", False),
            requires_chat_template_string=data.get(
                "requires_chat_template_string", False
            ),
            requires_mm_encoder_tp_data=data.get("requires_mm_encoder_tp_data", False),
            model_family=data.get("model_family"),
            max_output_tokens=data.get("max_output_tokens"),
            max_position_embeddings=data.get("max_position_embeddings"),
            reasoning_effort_levels=data.get("reasoning_effort_levels"),
            reasoning_effort_default=data.get("reasoning_effort_default"),
        )

    @classmethod
    def from_family(cls, family: str) -> ModelCapabilities:
        """Derive capabilities from model family tag.

        This is the transition path: use family-based defaults until
        every model has explicit capabilities in the registry.
        """
        return cls(
            supports_tool_calling=family in FAMILY_TOOL_PARSER
            and FAMILY_TOOL_PARSER[family] != "",
            supports_reasoning=family in FAMILY_REASONING_PARSER
            and FAMILY_REASONING_PARSER[family] != "",
            requires_eager_mode=family in FAMILY_REQUIRES_EAGER,
            requires_triton_moe_fallback=family in FAMILY_REQUIRES_TRITON_MOE,
            requires_fp8_kv_cache=FAMILY_KV_CACHE_DTYPE.get(family) == "fp8",
            requires_chat_template_string=family
            in FAMILY_REQUIRES_CHAT_TEMPLATE_STRING,
            requires_mm_encoder_tp_data=family in FAMILY_REQUIRES_MM_ENCODER_TP_DATA,
            model_family=family,
        )
