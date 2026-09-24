"""The ``embedding-job`` sandbox's compute core (ADR-061 profile, D-12).

Runs INSIDE the embedding sandbox container as ``python -m
umrs.knowledge.embed_worker``: reads one JSON request on stdin, chunks each
text with the model's own tokenizer, embeds the chunks, and writes one JSON
reply on stdout. It has no database, object-store or network access — the
server keeps every read and write (destination stores only) and hands the
worker nothing but text.

Request::

    {"model_dir": "/model", "prefix_model_name": "<server model name>",
     "max_tokens": 256, "texts": ["...", ...]}

Reply::

    {"results": [{"chunks": [{"text", "start", "end"}, ...],
                  "vectors": [[...], ...]}, ...]}
    or {"error": "<message>"} with exit status 1
"""

from __future__ import annotations

import json
import sys
from typing import Any


def chunk_and_embed(
    embedder: Any, texts: list[str], *, prefix_model_name: str, max_tokens: int
) -> list[dict[str, Any]]:
    """Chunk + embed each text (shared by the in-process and sandbox lanes)."""
    from umrs.shared.utils import _chunk_text_without_nlp, add_document_prefix

    tokenizer = getattr(getattr(embedder, "client", None), "tokenizer", None)
    results: list[dict[str, Any]] = []
    for text in texts:
        chunks = [
            c
            for c in _chunk_text_without_nlp(
                text, tokenizer=tokenizer, max_tokens=max_tokens
            )
            if c["text"].strip()  # never embed an empty chunk
        ]
        chunk_texts = [c["text"] for c in chunks]
        vectors = (
            embedder.embed_documents(
                add_document_prefix(chunk_texts, prefix_model_name)
            )
            if chunk_texts
            else []
        )
        results.append(
            {
                "chunks": [
                    {"text": c["text"], "start": c.get("start"), "end": c.get("end")}
                    for c in chunks
                ],
                "vectors": [list(map(float, v)) for v in vectors],
            }
        )
    return results


def _load_embedder(model_dir: str) -> Any:
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name=model_dir)


def main() -> int:
    try:
        request = json.load(sys.stdin)
        embedder = _load_embedder(str(request["model_dir"]))
        results = chunk_and_embed(
            embedder,
            [str(t) for t in request.get("texts") or []],
            prefix_model_name=str(request.get("prefix_model_name") or ""),
            max_tokens=int(request.get("max_tokens") or 256),
        )
    except Exception as exc:  # noqa: BLE001 — the reply names every failure
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1
    json.dump({"results": results}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
