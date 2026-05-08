"""Helpers to inspect and resolve local cache directories for HF-based models.

These utilities keep the logic shared between the server and worker so they can
reason about where models are cached on disk before pushing/pulling them from
RiVault. Hugging Face describes the default layout for the hub cache under
``~/.cache/huggingface/hub/models--<repo>/snapshots/<revision>`` (see
https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache), and
SentenceTransformers defaults to ``~/.cache/torch/sentence_transformers`` unless
``SENTENCE_TRANSFORMERS_HOME`` is set (documented at
https://www.sbert.net/docs/package_reference/SentenceTransformer.html).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

CACHE_KIND_SENTENCE_TRANSFORMERS = "sentence_transformers"
CACHE_KIND_HF_HUB = "huggingface_hub"


@dataclass(frozen=True)
class CacheEntry:
    """Describe a resolved local cache location for a specific model."""

    kind: str
    base_dir: str
    relative_path: str
    path: str


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _deduplicate(paths: Iterable[str]) -> Iterator[str]:
    seen = set()
    for raw in paths:
        if not raw:
            continue
        resolved = _expand(raw)
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def _home() -> str:
    return _expand("~")


def sentence_transformers_home() -> str:
    """Return the preferred SentenceTransformers cache root for this host."""

    env = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    if env:
        return _expand(env)
    return os.path.join(_home(), ".cache", "torch", "sentence_transformers")


def huggingface_hub_root() -> str:
    """Return the preferred Hugging Face hub cache root for this host."""

    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        return _expand(hf_hub_cache)

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return _expand(os.path.join(hf_home, "hub"))

    transformers_cache = os.getenv("TRANSFORMERS_CACHE")
    if transformers_cache:
        return _expand(transformers_cache)

    return os.path.join(_home(), ".cache", "huggingface", "hub")


def cache_base_for_kind(kind: str) -> str:
    if kind == CACHE_KIND_SENTENCE_TRANSFORMERS:
        return sentence_transformers_home()
    if kind == CACHE_KIND_HF_HUB:
        return huggingface_hub_root()
    raise ValueError(f"Unknown cache kind '{kind}'")


def _iter_sentence_transformer_bases() -> Iterator[str]:
    fallback = [
        os.getenv("SENTENCE_TRANSFORMERS_HOME"),
        os.path.join(_home(), ".cache", "torch", "sentence_transformers"),
        os.path.join(_home(), ".cache", "sentence_transformers"),
    ]
    yield from _deduplicate(path for path in fallback if path)


def _iter_hf_hub_bases() -> Iterator[str]:
    fallback = [
        os.getenv("HF_HUB_CACHE"),
        os.path.join(os.getenv("HF_HOME", ""), "hub") if os.getenv("HF_HOME") else None,
        os.getenv("TRANSFORMERS_CACHE"),
        os.path.join(_home(), ".cache", "huggingface", "hub"),
    ]
    yield from _deduplicate(path for path in fallback if path)


def _resolve_sentence_transformers(model_name: str) -> CacheEntry | None:
    """Resolve a SentenceTransformers cache directory for ``model_name``.

    The library commonly stores models under directory names such as
    ``sentence-transformers_all-MiniLM-L6-v2`` or
    ``sentence-transformers_clip-ViT-B-32``, so we look for any directory
    whose name contains the model name (or a simple underscore variant).
    """

    # Simple patterns we accept when scanning cache directories
    needle_plain = model_name
    needle_underscored = model_name.replace("/", "-")

    for base in _iter_sentence_transformer_bases():
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            candidate = os.path.join(base, entry)
            if not os.path.isdir(candidate):
                continue
            if (
                entry == needle_plain
                or entry.endswith("_" + needle_plain)
                or needle_plain in entry
                or needle_underscored in entry
            ):
                relative = os.path.relpath(candidate, base)
                return CacheEntry(
                    kind=CACHE_KIND_SENTENCE_TRANSFORMERS,
                    base_dir=base,
                    relative_path=relative,
                    path=candidate,
                )
    return None


def _resolve_hf_snapshots(model_name: str) -> CacheEntry | None:
    """Resolve HF hub snapshots for a given model.

    For SentenceTransformers models it is common for the underlying HF
    repository name to be prefixed with ``sentence-transformers/``, e.g.
    ``sentence-transformers/clip-ViT-B-32``. When callers pass the short
    name ("clip-ViT-B-32"), we try both forms so that caches such as
    ``models--sentence-transformers--clip-ViT-B-32`` are discovered.
    """

    candidate_names = [model_name]
    if not model_name.startswith("sentence-transformers/"):
        candidate_names.append(f"sentence-transformers/{model_name}")

    for base in _iter_hf_hub_bases():
        for candidate in candidate_names:
            safe_name = candidate.replace("/", "--")
            repo_root = os.path.join(base, f"models--{safe_name}")
            snapshots_root = os.path.join(repo_root, "snapshots")
            if not os.path.isdir(snapshots_root):
                continue
            snapshot_dirs = [
                os.path.join(snapshots_root, entry)
                for entry in os.listdir(snapshots_root)
                if os.path.isdir(os.path.join(snapshots_root, entry))
            ]
            if not snapshot_dirs:
                continue
            snapshot_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            latest = snapshot_dirs[0]
            relative = os.path.relpath(latest, base)
            return CacheEntry(
                kind=CACHE_KIND_HF_HUB,
                base_dir=base,
                relative_path=relative,
                path=latest,
            )
    return None


def find_model_cache(model_name: str) -> CacheEntry | None:
    """Return information about a locally cached model, if any."""

    for resolver in (_resolve_sentence_transformers, _resolve_hf_snapshots):
        entry = resolver(model_name)
        if entry is not None:
            return entry
    return None


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
