"""M6: patches channel — env serialization, round-trip, and empty lane."""

from __future__ import annotations

from umrs.orchestrator.serving_spec import PatchSpec, patches_to_env

QW = PatchSpec(
    source="/lvs0/rccs-dmp/masaru.nagaso/umrs_models/patches/qwen3_5.py",
    target="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py",
    sha256="9" * 64,
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
