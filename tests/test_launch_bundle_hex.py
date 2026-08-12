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
from umrs.orchestrator.remote_slurm_backend import (
    RemoteSlurmBackend,
    _hex_export_parts,
)


def test_remote_backend_overrides_status_methods():
    """Regression: helper insertion must not close the class body early.

    M5 inserted ``_hex_export_parts`` mid-class; every method defined after it
    silently became unreachable code inside that function, and the class fell
    back to the *local* SlurmBackend implementations (subprocess squeue) —
    invisible until the first deployment without slurm CLIs in the image.
    """
    remote_mod = "umrs.orchestrator.remote_slurm_backend"
    for name in (
        "get_job_status",
        "_query_slurm_job_status",
        "_query_squeue_status",
        "_query_sacct_status",
        "_read_remote_job_info",
        "cleanup",
    ):
        method = getattr(RemoteSlurmBackend, name, None)
        assert method is not None, f"{name} missing on RemoteSlurmBackend"
        assert method.__module__ == remote_mod, (
            f"{name} resolved from {method.__module__}, expected override in {remote_mod}"
        )


@pytest.mark.asyncio
async def test_squeue_usage_text_never_flows_as_job_ids():
    """Regression: squeue usage words can never become job ids again.

    Before the numeric-only guard, a remote squeue that printed its full usage
    help (unknown '%i' format string on mixed cluster flavors) had every usage
    word forwarded as a job id, producing 'stop-Usage: squeue [-A account]...'
    pollution in the remote control dir (live RCCS 2026-08-12).
    """
    import types
    from unittest.mock import AsyncMock

    usage_lines = (
        "270373\n"
        "Usage: squeue [-A account] [--clusters names] [-i seconds] [--job jobid]\n"
        "      [--expand-patterns] [--json=data_parser] [--yaml=data_parser]\n"
        "270437"
    )

    class _SSH:
        def __init__(self, stdout):
            self._stdout = stdout

        async def run_remote(self, _command: str, **_kwargs):
            return types.SimpleNamespace(stdout=self._stdout, stderr="", ok=True, returncode=0)

    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
    backend._get_ssh = AsyncMock(return_value=_SSH(usage_lines))
    called: list[str] = []

    async def fake_status(job_id):
        called.append(job_id)
        return types.SimpleNamespace(job_id=job_id, status="running")

    backend.get_job_status = AsyncMock(side_effect=fake_status)

    await backend._remote_append_squeue_jobs([], None)

    assert called == ["270373", "270437"], called


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
