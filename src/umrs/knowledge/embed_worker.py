"""The ``embedding-job`` sandbox's compute core (ADR-061 profile, D-12).

Runs INSIDE the embedding sandbox container as ``python -m
umrs.knowledge.embed_worker``: reads one JSON request on stdin, chunks each
text with the model's own tokenizer, embeds the chunks, and writes one JSON
reply on stdout. It has no database, object-store or network access — the
server keeps every read and write (destination stores only) and hands the
worker nothing but text.

Request::

    {"model_dir": "/model", "prefix_model_name": "<server model name>",
     "max_tokens": 256, "texts": ["...", ...],
     "scheme": "fixed-v0" | "sent-v1", "titles": ["...", ...],
     "headings": [true | false, ...]}

``scheme`` and ``titles`` arrived with ADR-093 D-11 (C9); a request without
them is chunked ``fixed-v0`` exactly as before. ``titles[i]`` is text i's
header title (sent-v1 only). ``headings[i]`` (optional, sent-v1 only) says
whether text i's '#' lines are markdown headings: false for code, config
and plain text, so a '# comment' never names a section (review
chunk-data-8); a text without an entry reads its '#' lines as headings.

Reply::

    {"results": [{"chunks": [{"text", "start", "end"}, ...],
                  "vectors": [[...], ...]}, ...]}
    or {"error": "<message>"} with exit status 1

A ``sent-v1`` result also carries ``"scheme": "sent-v1"`` and each chunk
``header``, ``section`` and ``num_tokens``; the vector embeds ``header``,
a newline and ``text`` (chunking.embed_input). A result without ``scheme``
is ``fixed-v0`` — which is also what an older worker image returns, so the
server never records a scheme the worker did not produce.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any


def _reply_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    out = {"text": chunk["text"], "start": chunk.get("start"), "end": chunk.get("end")}
    if "header" in chunk:  # sent-v1
        out.update(
            header=chunk["header"],
            section=chunk.get("section", ""),
            num_tokens=chunk.get("num_tokens"),
        )
    return out


def _result(
    chunks: list[dict[str, Any]], vectors: list[Any], scheme: str
) -> dict[str, Any]:
    from .chunking import FIXED_V0

    result: dict[str, Any] = {
        "chunks": [_reply_chunk(c) for c in chunks],
        "vectors": [list(map(float, v)) for v in vectors],
    }
    if scheme != FIXED_V0:
        result["scheme"] = scheme
    return result


def _nth(values: Sequence[Any], i: int, default: Any) -> Any:
    """``values[i]``, or ``default`` past the end (a shorter list)."""
    return values[i] if i < len(values) else default


def chunk_and_embed(
    embedder: Any,
    texts: list[str],
    *,
    prefix_model_name: str,
    max_tokens: int,
    scheme: str = "fixed-v0",
    titles: Sequence[str] | None = None,
    headings: Sequence[bool] | None = None,
) -> list[dict[str, Any]]:
    """Chunk + embed each text (shared by the in-process and sandbox lanes)."""
    from umrs.shared.utils import add_document_prefix

    from .chunking import chunk_text, embed_input, resolve_scheme

    scheme = resolve_scheme(scheme)
    names = list(titles or [])
    flags = list(headings or [])
    tokenizer = getattr(getattr(embedder, "client", None), "tokenizer", None)
    results: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        chunks = [
            c
            for c in chunk_text(
                text,
                scheme=scheme,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                title=str(_nth(names, i, "")),
                headings=bool(_nth(flags, i, True)),
            )
            if c["text"].strip()  # never embed an empty chunk
        ]
        inputs = [embed_input(c) for c in chunks]
        vectors = (
            embedder.embed_documents(add_document_prefix(inputs, prefix_model_name))
            if inputs
            else []
        )
        results.append(_result(chunks, vectors, scheme))
    return results


def _load_embedder(model_dir: str) -> Any:
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name=model_dir)


def _options(request: dict[str, Any]) -> dict[str, Any]:
    """``chunk_and_embed``'s options from one request (none given: fixed-v0)."""
    return {
        "prefix_model_name": str(request.get("prefix_model_name") or ""),
        "max_tokens": int(request.get("max_tokens") or 256),
        "scheme": str(request.get("scheme") or "fixed-v0"),
        "titles": [str(t or "") for t in request.get("titles") or []],
        "headings": [bool(h) for h in request.get("headings") or []],
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        embedder = _load_embedder(str(request["model_dir"]))
        results = chunk_and_embed(
            embedder,
            [str(t) for t in request.get("texts") or []],
            **_options(request),
        )
    except Exception as exc:  # noqa: BLE001 — the reply names every failure
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1
    json.dump({"results": results}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
