#!/usr/bin/env python3
"""Merge the Bespoke-Nimble-9B LoRA into its Qwen3.5-9B base (ADR-076).

Offline and dependency-light on purpose: torch + safetensors only (both ship
in the vllm-openai image), no PEFT, no network. Reproduces PEFT's CPU
``merge_and_unload(safe_merge=True)`` arithmetic in BF16:

    delta = (B.float() @ A.float() * lora_alpha / r).to(bfloat16)
    W     = W + delta                      (BF16 add, as PEFT does)

Inputs are two local snapshots (downloaded on the login node):

    --base    Qwen/Qwen3.5-9B @ c2022362...   (config.json + *.safetensors + index)
    --adapter bespokelabs/Bespoke-Nimble-9B @ bd792f44...  (adapter_* + tokenizer + contract)

The output is a complete servable checkpoint: base files copied, the shards
holding adapted weights rewritten, the adapter's tokenizer/chat template and
release contract copied, and READY.json recording every hash. Re-running on a
finished output verifies READY.json and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ADAPTER_SHA256 = "29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER_REVISION = "bd792f44ec8e265be861bfcdf4e05967ffe0e858"
PREFIX = "base_model.model."
# Files taken from the adapter release (the tokenizer it was trained with).
ADAPTER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "schema_config.json",
    "serving_config.json",
    "temperature_config.json",
    "provenance.json",
    "LICENSE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def lora_pairs(adapter_tensors: dict) -> dict[str, tuple]:
    """base weight name -> (A, B) from PEFT adapter tensor names."""
    pairs: dict[str, tuple] = {}
    for name in adapter_tensors:
        if not name.endswith(".lora_A.weight"):
            continue
        stem = name[: -len(".lora_A.weight")]
        b_name = stem + ".lora_B.weight"
        if b_name not in adapter_tensors:
            raise SystemExit(f"adapter has {name} without {b_name}")
        base = (stem[len(PREFIX) :] if stem.startswith(PREFIX) else stem) + ".weight"
        pairs[base] = (adapter_tensors[name], adapter_tensors[b_name])
    return pairs


def merge(
    base_dir: Path, adapter_dir: Path, out_dir: Path, *, adapter_sha256: str = ADAPTER_SHA256
) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    ready = out_dir / "READY.json"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_sha = sha256(adapter_file)
    if adapter_sha != adapter_sha256:
        raise SystemExit(f"adapter sha256 {adapter_sha} != pinned {adapter_sha256}")
    if ready.exists():
        saved = json.loads(ready.read_text())
        if saved.get("adapter_sha256") != adapter_sha:
            raise SystemExit(f"{out_dir} holds a different merge; use a fresh directory")
        print(f"verified existing merge: {out_dir}")
        return saved

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / float(config["r"])
    pairs = lora_pairs(load_file(str(adapter_file)))
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(set(pairs) - set(index))
    if missing:
        raise SystemExit(f"{len(missing)} adapted weights absent from the base: {missing[:3]}")

    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for item in base_dir.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors"):
            shutil.copy2(item, partial / item.name)

    shards = sorted(set(index.values()))
    merged = 0
    for shard in shards:
        tensors = load_file(str(base_dir / shard))
        touched = [name for name in tensors if name in pairs]
        for name in touched:
            a, b = pairs[name]
            weight = tensors[name]
            if tuple(weight.shape) != (b.shape[0], a.shape[1]):
                raise SystemExit(
                    f"{name}: base {tuple(weight.shape)} vs lora {b.shape[0]}x{a.shape[1]}"
                )
            delta = (b.float() @ a.float() * scale).to(weight.dtype)
            tensors[name] = (weight + delta).contiguous()
            merged += 1
        save_file(tensors, str(partial / shard), metadata={"format": "pt"})
        print(f"{shard}: {len(touched)} adapted", flush=True)
    if merged != len(pairs):
        raise SystemExit(f"merged {merged} of {len(pairs)} adapted weights")

    for name in ADAPTER_FILES:
        if (adapter_dir / name).exists():
            shutil.copy2(adapter_dir / name, partial / name)

    record = {
        "model": "bespokelabs/Bespoke-Nimble-9B",
        "revision": ADAPTER_REVISION,
        "adapter_sha256": adapter_sha,
        "base": "Qwen/Qwen3.5-9B",
        "base_revision": BASE_REVISION,
        "merged_weights": merged,
        "scale": scale,
        "precision": "bfloat16",
        "merge": "W + (B@A*alpha/r in fp32 -> bf16), PEFT safe_merge CPU arithmetic",
        "torch": torch.__version__,
        "shards": {shard: sha256(partial / shard) for shard in shards},
    }
    (partial / "READY.json").write_text(json.dumps(record, indent=2) + "\n")
    partial.rename(out_dir)
    print(f"merged {merged} weights -> {out_dir}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-sha256",
        default=ADAPTER_SHA256,
        help="expected adapter_model.safetensors sha256 (default: the pinned release)",
    )
    args = parser.parse_args(argv)
    merge(args.base, args.adapter, args.output, adapter_sha256=args.adapter_sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
