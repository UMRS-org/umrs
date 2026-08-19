"""M6: patches channel — env serialization, round-trip, and empty lane."""

from __future__ import annotations

from pathlib import Path

from umrs.orchestrator.serving_spec import PatchSpec, patches_to_env

QW = PatchSpec(
    source="/lvs0/rccs-dmp/masaru.nagaso/umrs_models/patches/qwen3_5.py",
    target="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py",
    sha256="9" * 64,
)

# Test-file anchored template dir (cwd-independent: CI runs with cwd=umrs).
_MINI_RIVAULT_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm"
)


def test_patches_to_env_produces_parseable_triplet():
    env = patches_to_env([QW])
    assert env["VLLM_PATCHES_COUNT"] == "1"
    entries = env["VLLM_PATCHES"].split()
    assert len(entries) == 1
    src, tgt, sha = entries[0].split(":")
    assert src == QW.source
    assert tgt == QW.target
    assert sha == QW.sha256


def test_empty_patches_produce_no_env():
    assert patches_to_env([]) == {}


def test_multiple_patches_order_is_preserved():
    second = PatchSpec(source="/x/b.py", target="/usr/local/lib/b.py", sha256="a" * 64)
    env = patches_to_env([QW, second])
    entries = env["VLLM_PATCHES"].split()
    assert entries[0].split(":")[0] == QW.source
    assert entries[1].split(":")[0] == second.source
    assert env["VLLM_PATCHES_COUNT"] == "2"


def test_triplet_contains_no_unsafe_wire_chars():
    # Wire contract complements M5: triplet template chars are always passthru
    # safe so proxy forwarding + regex parse stay deterministic.
    env = patches_to_env([QW])
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:/%-")
    for entry in env["VLLM_PATCHES"].split():
        for field in entry.split(":"):
            assert set(field) <= safe, field


# ---------------------------------------------------------------------------
# Launch-lane guards (2026-08-12 drift found live on RCCS)
# ---------------------------------------------------------------------------


def test_shared_patch_resolver_is_called_before_env_write():
    """`VLLM_PATCHES` must resolve before env.<uuid> is generated, for ALL lanes.

    Live drift class discovered on RCCS: the MN-only loop left the compose
    flavor with inert /dev/null patch mounts, silently skipping recipe patches.
    """
    import pathlib
    import re

    template = _MINI_RIVAULT_DIR / "serve-model.slurm"
    slurm = template.read_text(encoding="utf-8")
    fn_pos = slurm.index("_apply_vllm_patches()")
    call_pos = slurm.index("\n_apply_vllm_patches\n")
    heredoc_pos = slurm.index('cat <<EOF > "./env.${uuid}"')
    assert fn_pos < call_pos < heredoc_pos, "resolver must run before env write"
    assert re.search(r"_mn_vllm_args\+=\(\s*\"\$\{_VLLM_PATCH_MOUNTS\[@\]\}\"\s*\)", slurm), (
        "MN branch must consume the shared mount array"
    )


def test_engine_flag_lanes_wired_in_both_launch_paths():
    """Typed engine flags must reach every `vllm serve` invocation."""
    slurm = (_MINI_RIVAULT_DIR / "serve-model.slurm").read_text(encoding="utf-8")
    yaml = (_MINI_RIVAULT_DIR / "serve-model.yaml").read_text(encoding="utf-8")
    for lane in (
        "VLLM_LANGUAGE_MODEL_ONLY",
        "VLLM_KV_CACHE_MEMORY_BYTES",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
    ):
        assert lane in slurm, f"{lane} missing from MN/exec lane"
        assert lane in yaml, f"{lane} missing from compose env declarations"
    assert "--language-model-only" in yaml
    assert "--kv-cache-memory-bytes" in yaml
    assert "--max-num-seqs" in yaml
    assert "--max-num-batched-tokens" in yaml
