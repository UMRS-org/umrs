"""Tests for per-model reasoning effort derivation."""

from umrs.model_registry.capabilities import (
    ModelCapabilities,
    derive_reasoning_effort,
    resolve_capabilities,
)


def test_derive_reasoning_effort_kimi_k27_thinking_only():
    levels, default = derive_reasoning_effort("Kimi-K2.7-Code")
    assert levels == ["high", "max"]
    assert default == "high"


def test_derive_reasoning_effort_kimi_k26():
    levels, default = derive_reasoning_effort("Kimi-K2.6")
    assert levels == ["none", "high", "max"]
    assert default == "high"


def test_derive_reasoning_effort_glm_versions_differ():
    levels_51, default_51 = derive_reasoning_effort("GLM-5.1")
    levels_52, default_52 = derive_reasoning_effort("GLM-5.2")
    assert "low" in levels_51
    assert "low" not in levels_52
    assert default_52 == "max"


def test_derive_reasoning_effort_gpt_oss_single_level():
    levels, default = derive_reasoning_effort("GPT-OSS-Swallow-120B-RL-v0.1-MXFP4")
    assert levels == ["medium"]
    assert default == "medium"


def test_derive_reasoning_effort_unknown_returns_none():
    assert derive_reasoning_effort("Some-Random-Model") is None


def test_resolve_capabilities_preserves_explicit_levels():
    caps = ModelCapabilities(
        reasoning_effort_levels=["low"],
        reasoning_effort_default="low",
    )
    resolved = resolve_capabilities("Kimi-K2.6", caps)
    assert resolved.reasoning_effort_levels == ["low"]
    assert resolved.reasoning_effort_default == "low"


def test_resolve_capabilities_fills_from_pattern():
    resolved = resolve_capabilities("llm-jp-4-32b-a3b-thinking", ModelCapabilities())
    assert resolved.reasoning_effort_levels == ["low", "medium", "high"]
    assert resolved.reasoning_effort_default == "medium"
    assert resolved.supports_reasoning is True
