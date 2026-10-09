"""The site's text embedder: which one, and staging it for offline use.

Maintainer decision 2026-10-09: text embedding works out of the box. The
umrs-server image bundles ``intfloat/e5-base-v2`` (768-d) at
:data:`BUNDLED_TEXT_EMBEDDER_DIR`, staged at BUILD time — UMRS never downloads
a model at runtime (offline rule). ``UMRS_KNOWLEDGE_EMBED_MODEL`` still points
a site at another pre-staged model directory.

The configured value is also the embedding-space name written to
``knowledge_node_embeddings.model_name`` (ADR-083), and the RAG service's query
encoder defaults to it, so queries and stored vectors share one space.

Staging CLI (image build, ``standalone.sh embedder pull``, operators)::

    python -m umrs.knowledge.text_embedder stage --dest /opt/umrs/models/e5-base-v2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TEXT_EMBEDDER_ENV = "UMRS_KNOWLEDGE_EMBED_MODEL"
BUNDLED_TEXT_EMBEDDER_REPO = "intfloat/e5-base-v2"
# The commit the stand-alone stack ran on when the bundle was adopted.
BUNDLED_TEXT_EMBEDDER_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
BUNDLED_TEXT_EMBEDDER_DIR = "/opt/umrs/models/e5-base-v2"

# Weights, config and tokenizer of a sentence-transformers model — not the
# ONNX/OpenVINO variants or the duplicate pytorch_model.bin (half the size).
STAGED_FILES = [
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
]


def configured_text_embedder() -> str:
    """The text embedder's model directory: the env setting, else the bundle."""
    return os.environ.get(TEXT_EMBEDDER_ENV, "").strip() or BUNDLED_TEXT_EMBEDDER_DIR


def embedding_space(model: str) -> str:
    """The space a model name denotes: its basename. Sites stage the same model
    under their own paths (``/models/e5-base-v2``, the bundle's
    ``/opt/umrs/models/e5-base-v2``, the hub id ``intfloat/e5-base-v2``) —
    the convention reindex and bundle import already match by."""
    return model.rstrip("/").rsplit("/", 1)[-1]


def is_staged(model_dir: str | os.PathLike[str]) -> bool:
    """A directory a sentence-transformers model can load from, offline."""
    path = Path(model_dir)
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def stage(repo: str, dest: str, revision: str | None = None) -> Path:
    """Download ``repo`` into ``dest`` (needs network — build/staging time only)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=dest, revision=revision, allow_patterns=STAGED_FILES)
    if not is_staged(dest):
        raise RuntimeError(f"{repo} staged into {dest} without config.json + model.safetensors")
    return Path(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m umrs.knowledge.text_embedder")
    sub = parser.add_subparsers(dest="command", required=True)
    p_stage = sub.add_parser("stage", help="download a text embedder for offline use")
    p_stage.add_argument("--repo", default=BUNDLED_TEXT_EMBEDDER_REPO)
    p_stage.add_argument("--revision", default=None)
    p_stage.add_argument("--dest", default=BUNDLED_TEXT_EMBEDDER_DIR)
    sub.add_parser("show", help="print the configured embedder and whether it is staged")
    args = parser.parse_args(argv)

    if args.command == "stage":
        revision = args.revision
        if revision is None and args.repo == BUNDLED_TEXT_EMBEDDER_REPO:
            revision = BUNDLED_TEXT_EMBEDDER_REVISION
        print(f"staged {args.repo}@{revision or 'main'} -> {stage(args.repo, args.dest, revision)}")
        return 0
    model = configured_text_embedder()
    print(f"{model} ({'staged' if is_staged(model) else 'NOT staged'})")
    return 0 if is_staged(model) else 1


if __name__ == "__main__":
    sys.exit(main())
