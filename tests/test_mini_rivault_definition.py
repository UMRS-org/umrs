"""Tests for the Mini-RiVault Singularity definition."""

from __future__ import annotations

from pathlib import Path


def test_mini_rivault_def_wires_prefix_caching_flag():
    repo_root = Path(__file__).resolve().parents[2]
    definition = repo_root / "deploy" / "singularity" / "mini-rivault.def"

    text = definition.read_text()

    assert "VLLM_ENABLE_PREFIX_CACHING" in text
    assert "prefix_caching_args+=(--enable-prefix-caching)" in text
    assert '"${prefix_caching_args[@]}"' in text
    assert "command=/opt/mini-rivault/start-vllm.sh" in text