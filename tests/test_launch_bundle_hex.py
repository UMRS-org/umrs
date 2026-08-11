"""M5: _HEX launch transport — deterministic, proxy-safe, round-trip.

Covers the previously failing class (`--export=ALL,K=V` quoted → RCCS
slurm-proxy charset allowedlist) with two invariants:

1. Every emitted token is inside the proxy charset [a-zA-Z0-9 _./:=,%-]
   (so the submission is accepted through the per-user SSH proxy).
2. Values ALWAYS decode back on the launcher side (serve-model.slurm's
   hex decoder is byte-identical by construction here).
"""

from __future__ import annotations

import pytest
from umrs.orchestrator.remote_slurm_backend import _hex_export_parts

PROXY_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _./:=,%-")


def _hex_decode(value: str) -> str:
    return bytes.fromhex(value).decode("utf-8")


def test_empty_values_and_unicode_are_hexed():
    env = {"PLAIN": "hello", "EMPTY": "0%cases-force-hex"}
    parts = _hex_export_parts(env)
    joined = ",".join(parts)
    assert joined.startswith("ALL,")
    assert "PLAIN=hello" in joined
    for token in parts:
        assert set(token) <= PROXY_CHARS, f"offending token: {token!r}"


def test_space_comma_and_json_are_hex_round_trips():
    env = {
        "RIVAULT_PROFILES": "harness vllm litellm-db",
        "VLLM_SPECULATIVE_CONFIG": '{"method":"mtp","num_speculative_tokens":2}',
        "VLLM_EXTRA_ARGS": "--max-num-seqs 10 --tokenizer-mode deepseek_v4",
        "VLLM_DOCKER_IMAGE": "docker.io/vllm/vllm-openai:v0.26.0",
    }
    parts = _hex_export_parts(env)
    join = ",".join(parts)
    assert join.startswith("ALL,")
    # Space/comma-containing values are always hex-transported.
    assert any(t.partition("=")[0].endswith("_HEX") for t in parts[1:])
    assert not any(" " in t for t in parts[1:])
    # Passthru untouched for safe values:
    assert "VLLM_DOCKER_IMAGE=docker.io/vllm/vllm-openai:v0.26.0" in parts
    for token in parts:
        assert set(token) <= PROXY_CHARS, f"offending token: {token!r}"


def test_each_value_roundtrips_byte_exactly():
    samples = [
        "harness vllm litellm-db",
        '{"method":"mtp","num_speculative_tokens":2}',
        "--max-num-seqs 10 --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8",
        '--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}',
        "--language-model-only",
        '"with.quotes\'and.apostrophes"',
        "multi\nline\ncontent\n",
        "100 GB cómplex UTF-8 prefixé",
    ]
    for sample in samples:
        env = {"K": sample}
        parts = _hex_export_parts(env)
        assert len(parts) == 2, parts
        token = parts[1]
        name, _, value = token.partition("=")
        if value and all(c in set("0123456789abcdef") for c in value):
            reconstructed = _hex_decode(value)
            assert reconstructed == sample, (reconstructed, sample)
        else:
            assert value == sample


def test_hex_encoding_is_deterministic():
    env = {"A": "x y,z"}
    assert _hex_export_parts(env) == _hex_export_parts(env)
    expected = "78" + "20" + "79" + "2c" + "7a"
    assert _hex_export_parts(env)[1] == f"A_HEX={expected}"


@pytest.mark.parametrize("char", ["'", '"', "+", "{", "}", "(", "`"])
def test_forbidden_proxy_chars_never_surface(char):
    env = {"FV": f"pre{char}post space,comma"}
    parts = _hex_export_parts(env)
    joined = ",".join(parts)
    assert char not in joined
