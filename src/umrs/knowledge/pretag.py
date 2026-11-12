"""Optional GLiNER pre-tagging of extraction windows (ADR-116 D-3).

When ``UMRS_KNOWLEDGE_GLINER_MODEL`` names a staged directory, each window is
tagged with zero-shot labels taken from the graph's extraction vocabulary
(``vocab.gliner_labels()``), and the candidates go into the extraction prompt
as hints. Tags are never written as nodes on their own.

The model loads once per process with ``GLiNER.from_pretrained(dir,
local_files_only=True)`` under ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``.
With ``UMRS_EMBEDDING_SANDBOX=podman`` the tagging runs in the sandbox worker
(op ``tag``, one launch per artifact for all its windows); otherwise in the
calling thread. Unset means no pre-tagging and no note; set but unusable
means no pre-tagging and the note ``pretag_unavailable:<reason>``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from typing import Any

GLINER_ENV = "UMRS_KNOWLEDGE_GLINER_MODEL"
# The path the sandbox host sees for the model, when the server sees it
# under another path; unset = the same path.
GLINER_HOST_ENV = "UMRS_KNOWLEDGE_GLINER_MODEL_HOST_DIR"
SANDBOX_MODEL_DIR = "/models-ingest/gliner"
THRESHOLD = 0.5
MAX_CANDIDATES = 40
# GLiNER reads a few hundred tokens per call; a window is tagged in pieces.
PIECE_CHARS = 1500

Tagger = Callable[[Sequence[str]], "list[list[dict[str, Any]]]"]

_MODELS: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def configured_model_dir() -> str:
    return os.environ.get(GLINER_ENV, "").strip()


def load_model(model_dir: str) -> Any:
    """The GLiNER model in ``model_dir``, loaded once (offline, local files)."""
    from .doc_parse import set_offline_env

    with _MODEL_LOCK:
        model = _MODELS.get(model_dir)
        if model is None:
            set_offline_env()
            from gliner import GLiNER

            model = GLiNER.from_pretrained(model_dir, local_files_only=True)
            _MODELS[model_dir] = model
        return model


def _pieces(text: str, size: int = PIECE_CHARS) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _candidates(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicated (case-insensitive text), best score first, at most 40."""
    best: dict[str, dict[str, Any]] = {}
    for item in found:
        text = " ".join(str(item.get("text") or "").split())
        key = text.lower()
        score = float(item.get("score") or 0.0)
        if key and (key not in best or score > best[key]["score"]):
            best[key] = {
                "text": text,
                "label": str(item.get("label") or ""),
                "score": score,
            }
    ranked = sorted(best.values(), key=lambda c: -c["score"])
    return ranked[:MAX_CANDIDATES]


def tag_texts(
    model: Any,
    texts: Sequence[str],
    labels: Sequence[str],
    threshold: float = THRESHOLD,
) -> list[list[dict[str, Any]]]:
    """Per text, its candidate entities ``{text, label, score}``."""
    out: list[list[dict[str, Any]]] = []
    for text in texts:
        found: list[dict[str, Any]] = []
        for piece in _pieces(text):
            if piece.strip():
                found += model.predict_entities(
                    piece, list(labels), threshold=threshold
                )
        out.append(_candidates(found))
    return out


def _tag_in_sandbox(
    config: Any, model_dir: str, texts: Sequence[str], labels: Sequence[str]
) -> list[list[dict[str, Any]]]:
    from .doc_parse import WORKER_MODULE, sandbox_host_path
    from .embed_sandbox import run_worker

    host_dir = sandbox_host_path(model_dir, GLINER_HOST_ENV)
    reply = run_worker(
        config,
        {
            "op": "tag",
            "model_dir": SANDBOX_MODEL_DIR,
            "texts": list(texts),
            "labels": list(labels),
            "threshold": THRESHOLD,
        },
        module=WORKER_MODULE,
        mounts=[(host_dir, SANDBOX_MODEL_DIR)],
    )
    candidates = reply.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(texts):
        raise RuntimeError("tag reply has the wrong shape")
    return candidates


def _reason(exc: BaseException) -> str:
    if isinstance(exc, ImportError):
        return "gliner_not_installed"
    from .embed_sandbox import EmbeddingSandboxError

    if isinstance(exc, EmbeddingSandboxError):
        return "sandbox_failed"
    return type(exc).__name__


def _run_tagging(
    model_dir: str, texts: Sequence[str], labels: Sequence[str]
) -> list[list[dict[str, Any]]]:
    from .embed_sandbox import worker_sandbox_config

    config = worker_sandbox_config()
    if config is not None:
        return _tag_in_sandbox(config, model_dir, texts, labels)
    return tag_texts(load_model(model_dir), texts, labels)


def pretag_windows(
    texts: Sequence[str], labels: Sequence[str], notes: list[str]
) -> list[list[dict[str, Any]]] | None:
    """Candidates per window, or None (off, or unavailable with a note)."""
    model_dir = configured_model_dir()
    if not model_dir or not texts:
        return None
    if not os.path.isdir(model_dir):
        notes.append("pretag_unavailable:model_dir_missing")
        return None
    if not labels:
        notes.append("pretag_unavailable:no_labels")
        return None
    try:
        return _run_tagging(model_dir, texts, labels)
    except Exception as exc:  # noqa: BLE001 — pre-tagging is optional, named
        notes.append(f"pretag_unavailable:{_reason(exc)}")
        return None


def window_tagger(vocab: Any, notes: list[str]) -> Tagger | None:
    """A tagger over the vocabulary's labels, or None when GLiNER is unset."""
    if not configured_model_dir():
        return None
    try:
        labels = list(vocab.gliner_labels())
    except Exception:  # noqa: BLE001 — a vocabulary without labels tags nothing
        labels = []

    def tag(texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        return pretag_windows(texts, labels, notes) or [[] for _ in texts]

    return tag


__all__ = [
    "GLINER_ENV",
    "load_model",
    "pretag_windows",
    "tag_texts",
    "window_tagger",
]
