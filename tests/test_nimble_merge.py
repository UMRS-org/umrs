"""deploy/rccs/nimble_merge.py reproduces PEFT's BF16 LoRA merge (ADR-076).

Driven through its CLI on a synthetic two-shard checkpoint (the script is a
deploy tool, not a package module).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "rccs" / "nimble_merge.py"
W1 = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
W2 = "model.language_model.layers.1.mlp.down_proj.weight"
UNTOUCHED = "model.visual.blocks.0.attn.qkv.weight"


def _checkpoint(tmp_path: Path):
    gen = torch.Generator().manual_seed(0)
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    w1 = torch.randn(8, 6, generator=gen).to(torch.bfloat16)
    w2 = torch.randn(6, 4, generator=gen).to(torch.bfloat16)
    vis = torch.randn(3, 3, generator=gen).to(torch.bfloat16)
    st.save_file({W1: w1, UNTOUCHED: vis}, str(base / "model-00001-of-00002.safetensors"))
    st.save_file({W2: w2}, str(base / "model-00002-of-00002.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    W1: "model-00001-of-00002.safetensors",
                    UNTOUCHED: "model-00001-of-00002.safetensors",
                    W2: "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (base / "config.json").write_text("{}")
    lora = {}
    for name, (out_f, in_f) in {W1: (8, 6), W2: (6, 4)}.items():
        stem = "base_model.model." + name[: -len(".weight")]
        lora[stem + ".lora_A.weight"] = torch.randn(2, in_f, generator=gen)
        lora[stem + ".lora_B.weight"] = torch.randn(out_f, 2, generator=gen)
    st.save_file(lora, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"lora_alpha": 4, "r": 2}))
    (adapter / "tokenizer.json").write_text("{}")
    return base, adapter, {W1: w1, W2: w2, UNTOUCHED: vis}, lora


def _run(base: Path, adapter: Path, out: Path, sha: str | None = None):
    args = [sys.executable, str(SCRIPT), "--base", str(base), "--adapter", str(adapter)]
    args += ["--output", str(out)]
    if sha is not None:
        args += ["--adapter-sha256", sha]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def merged_run(tmp_path):
    base, adapter, weights, lora = _checkpoint(tmp_path)
    out = tmp_path / "merged"
    sha = _sha(adapter / "adapter_model.safetensors")
    result = _run(base, adapter, out, sha)
    assert result.returncode == 0, result.stderr
    return base, adapter, out, sha, weights, lora


def _expected(name: str, weights: dict, lora: dict):
    stem = "base_model.model." + name[: -len(".weight")]
    a, b = lora[stem + ".lora_A.weight"], lora[stem + ".lora_B.weight"]
    return weights[name] + (b.float() @ a.float() * 2.0).to(torch.bfloat16)


def test_merge_matches_peft_bf16_arithmetic(merged_run):
    _, _, out, _, weights, lora = merged_run
    merged = {}
    for shard in out.glob("*.safetensors"):
        merged.update(st.load_file(str(shard)))
    assert {merged[n].dtype for n in (W1, W2)} == {torch.bfloat16}
    assert torch.equal(merged[W1], _expected(W1, weights, lora))
    assert torch.equal(merged[W2], _expected(W2, weights, lora))
    assert torch.equal(merged[UNTOUCHED], weights[UNTOUCHED])


def test_merge_output_is_complete_and_atomic(merged_run, tmp_path):
    _, _, out, _, _, _ = merged_run
    ready = json.loads((out / "READY.json").read_text())
    assert (ready["merged_weights"], ready["scale"]) == (2, 2.0)
    assert (out / "tokenizer.json").exists()
    assert (out / "config.json").exists()
    assert not (tmp_path / "merged.partial").exists()


def test_rerun_verifies_instead_of_rewriting(merged_run):
    base, adapter, out, sha, _, _ = merged_run
    again = _run(base, adapter, out, sha)
    assert again.returncode == 0
    assert "verified existing merge" in again.stdout


def test_pinned_adapter_hash_is_enforced(tmp_path):
    base, adapter, _, _ = _checkpoint(tmp_path)
    result = _run(base, adapter, tmp_path / "m")  # default = the release pin
    assert result.returncode != 0
    assert "pinned" in result.stderr
    assert not (tmp_path / "m").exists()
