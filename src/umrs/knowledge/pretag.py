"""GLiNER pre-tagging of extraction windows (ADR-116 D-3).

Each window is tagged with zero-shot labels taken from the graph's
extraction vocabulary (``vocab.gliner_labels()``), and the candidates go into
the extraction prompt as hints. Tags are never written as nodes on their own.

On by default since 2026-11-13: ``UMRS_KNOWLEDGE_GLINER_MODEL`` empty means
the directory bundled in the server image, ``off`` switches pre-tagging off
(``ingest_models.configured_gliner_dir``). For one document the tagging
tries, in order:

1. the shared ``umrs-docmodels`` service when ``UMRS_KNOWLEDGE_DOCMODELS_URL``
   is set. A refusal about the texts themselves (``ingest_models.
   SERVICE_FINAL_REASONS``) ends the chain with ``pretag_unavailable:<reason>``;
   any other failure adds ``docmodels_unavailable:<reason>`` and goes to 2
   only as far as ``UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK`` allows (default
   ``sandbox``; else ``pretag_unavailable:no_local_fallback``);
2. the local model when its directory is staged: the sandbox worker with
   ``UMRS_EMBEDDING_SANDBOX=podman`` (op ``tag``, one launch per document),
   otherwise a bounded thread in this process, the model loaded once per
   process with ``GLiNER.from_pretrained(dir, local_files_only=True)`` under
   ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``;
3. no hints, with ``pretag_unavailable:<reason>`` (once per notes list).

The chain shares one budget per document (``docmodels_budget`` of the job
deadline; see ``ingest_models``), so a slow tagging can no longer overrun
the job.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from . import ingest_models

GLINER_ENV = ingest_models.GLINER_MODEL_ENV
# The path the sandbox host sees for the model, when the server sees it
# under another path; unset = the same path.
GLINER_HOST_ENV = "UMRS_KNOWLEDGE_GLINER_MODEL_HOST_DIR"
SANDBOX_MODEL_DIR = "/models-ingest/gliner"
THRESHOLD = 0.5
MAX_CANDIDATES = 40
# GLiNER reads a few hundred tokens per call; a window is tagged in pieces.
PIECE_CHARS = 1500

Tagger = Callable[[Sequence[str]], "list[list[dict[str, Any]]]"]
Candidates = list[list[dict[str, Any]]]

_MODELS: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def configured_model_dir() -> str:
    """``ingest_models.configured_gliner_dir`` ("" = pre-tagging off)."""
    return ingest_models.configured_gliner_dir()


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
) -> Candidates:
    """Per text, its candidate entities ``{text, label, score}``."""
    out: Candidates = []
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
) -> Candidates:
    from .doc_parse import WORKER_MODULE, sandbox_mounts
    from .embed_sandbox import run_worker

    mounts, worker_dir = sandbox_mounts(model_dir, GLINER_HOST_ENV, SANDBOX_MODEL_DIR)
    reply = run_worker(
        config,
        {
            "op": "tag",
            "model_dir": worker_dir,
            "texts": list(texts),
            "labels": list(labels),
            "threshold": THRESHOLD,
        },
        module=WORKER_MODULE,
        mounts=mounts,
    )
    candidates = reply.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(texts):
        raise RuntimeError("tag reply has the wrong shape")
    return candidates


def _reason(exc: BaseException) -> str:
    from .doc_parse import BoundedTimeout

    if isinstance(exc, BoundedTimeout):
        return "timeout"
    if isinstance(exc, ImportError):
        return "gliner_not_installed"
    from .embed_sandbox import EmbeddingSandboxError

    if isinstance(exc, EmbeddingSandboxError):
        return "sandbox_failed"
    return type(exc).__name__


def _note(notes: list[str], note: str) -> None:
    if note not in notes:
        notes.append(note)


def _run_tagging(
    model_dir: str, texts: Sequence[str], labels: Sequence[str], timeout_s: float
) -> Candidates:
    """The local tagging, bounded by ``timeout_s`` in both lanes."""
    from .doc_parse import bounded_call
    from .embed_sandbox import worker_sandbox_config

    config = worker_sandbox_config()
    if config is not None:
        config = replace(config, timeout=min(config.timeout, timeout_s))
        return _tag_in_sandbox(config, model_dir, texts, labels)
    return bounded_call(
        lambda: tag_texts(load_model(model_dir), texts, labels),
        timeout_s,
        "kg-gliner",
    )


class _ServiceRefused(Exception):
    """The service refused the texts themselves: no local retry."""


def _tag_via_service(
    texts: Sequence[str], labels: Sequence[str], timeout_s: float, notes: list[str]
) -> Candidates | None:
    """The service's candidates, or None (unavailable: noted). A refusal
    about the texts raises :class:`_ServiceRefused` (noted)."""
    from . import docmodels_client

    try:
        return docmodels_client.tag(
            list(texts),
            list(labels),
            timeout_s,
            threshold=THRESHOLD,
            model_dir=configured_model_dir(),
            base_url=ingest_models.configured_docmodels_url(),
        )
    except docmodels_client.DocmodelsUnavailable as exc:
        if exc.reason in ingest_models.SERVICE_FINAL_REASONS:
            _note(notes, f"pretag_unavailable:{exc.reason}")
            raise _ServiceRefused(exc.reason) from exc
        _note(notes, f"docmodels_unavailable:{exc.reason}")
        return None


def _served(
    texts: Sequence[str], labels: Sequence[str], budget: float, notes: list[str]
) -> tuple[Candidates | None, bool]:
    """``(candidates, local allowed)`` from the service step of the chain."""
    if not ingest_models.configured_docmodels_url() or budget <= 0:
        return None, True
    from .doc_parse import local_fallback_refusal

    try:
        served = _tag_via_service(texts, labels, budget, notes)
    except _ServiceRefused:
        return None, False
    if served is not None:
        return served, False
    refusal = local_fallback_refusal()
    if refusal:
        _note(notes, f"pretag_unavailable:{refusal}")
    return None, not refusal


def _tag_locally(
    model_dir: str,
    texts: Sequence[str],
    labels: Sequence[str],
    timeout_s: float,
    notes: list[str],
) -> Candidates | None:
    if timeout_s <= 0:
        _note(notes, "pretag_unavailable:budget_exhausted")
        return None
    if not ingest_models.is_gliner_staged(model_dir):
        _note(notes, "pretag_unavailable:model_dir_missing")
        return None
    try:
        return _run_tagging(model_dir, texts, labels, timeout_s)
    except Exception as exc:  # noqa: BLE001 — pre-tagging is optional, named
        _note(notes, f"pretag_unavailable:{_reason(exc)}")
        return None


def pretag_windows(
    texts: Sequence[str],
    labels: Sequence[str],
    notes: list[str],
    timeout_s: float | None = None,
) -> Candidates | None:
    """Candidates per window, or None (off, or unavailable with a note).

    ``timeout_s`` bounds the whole chain (service, then local); None means
    ``UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S``.
    """
    model_dir = configured_model_dir()
    if not model_dir or not texts:
        return None
    if not labels:
        _note(notes, "pretag_unavailable:no_labels")
        return None
    budget = (
        ingest_models.configured_docmodels_timeout() if timeout_s is None else timeout_s
    )
    stop = time.monotonic() + budget
    served, local = _served(texts, labels, budget, notes)
    if served is not None or not local:
        return served
    return _tag_locally(model_dir, texts, labels, stop - time.monotonic(), notes)


def window_tagger(
    vocab: Any, notes: list[str], deadline: float | None = None
) -> Tagger | None:
    """A tagger over the vocabulary's labels, or None when GLiNER is off.

    ``deadline`` is the understand job's ``time.monotonic()`` deadline: each
    call is bounded by ``ingest_models.docmodels_budget(deadline)``.
    """
    if not configured_model_dir():
        return None
    try:
        labels = list(vocab.gliner_labels())
    except Exception:  # noqa: BLE001 — a vocabulary without labels tags nothing
        labels = []

    def tag(texts: Sequence[str]) -> Candidates:
        budget = ingest_models.docmodels_budget(deadline)
        return pretag_windows(texts, labels, notes, budget) or [[] for _ in texts]

    return tag


__all__ = [
    "GLINER_ENV",
    "load_model",
    "pretag_windows",
    "tag_texts",
    "window_tagger",
]
