"""The ingest sandbox worker (ADR-116 D-1, D-3).

Runs inside the same sandbox as the embedder (``--network=none``, read-only
model mounts, Hugging Face offline) as ``python -m umrs.knowledge.ingest_worker``:
one JSON request on stdin, one JSON reply on stdout. It has no database,
object-store or network access; the server keeps every read and write.

Requests::

    {"op": "parse", "filename": "paper.pdf", "data_b64": "<base64>",
     "artifacts_path": "/models-ingest/docling", "max_pages": 80}
    -> {"text": "<markdown>", "version": "<docling version>"}

``artifacts_path`` / ``model_dir`` is a read-only mount, or for a model
bundled in the image its path there (``/opt/umrs/models/...``, no mount).
``max_pages`` (optional) is Docling's page limit.

    {"op": "tag", "model_dir": "/models-ingest/gliner", "texts": ["..."],
     "labels": ["..."], "threshold": 0.5}
    -> {"candidates": [[{"text", "label", "score"}, ...], ...]}

Any failure: ``{"error": "<message>"}`` with exit status 1.
"""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Callable
from typing import Any


def _parse(request: dict[str, Any]) -> dict[str, Any]:
    from .doc_parse import docling_convert

    data = base64.b64decode(str(request.get("data_b64") or ""))
    max_pages = request.get("max_pages")
    text, version = docling_convert(
        data,
        str(request.get("filename") or "document.pdf"),
        str(request["artifacts_path"]),
        int(max_pages) if isinstance(max_pages, int) and max_pages > 0 else None,
    )
    return {"text": text, "version": version}


def _tag(request: dict[str, Any]) -> dict[str, Any]:
    from .pretag import THRESHOLD, load_model, tag_texts

    model = load_model(str(request["model_dir"]))
    candidates = tag_texts(
        model,
        [str(t) for t in request.get("texts") or []],
        [str(label) for label in request.get("labels") or []],
        float(request.get("threshold") or THRESHOLD),
    )
    return {"candidates": candidates}


_OPS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "parse": _parse,
    "tag": _tag,
}


def handle(request: Any) -> dict[str, Any]:
    """The reply to one request (raises on a bad request or a failed op)."""
    if not isinstance(request, dict):
        raise TypeError("request is not a JSON object")
    op = _OPS.get(str(request.get("op") or ""))
    if op is None:
        raise ValueError(f"unknown op {request.get('op')!r}")
    return op(request)


def main() -> int:
    try:
        reply = handle(json.load(sys.stdin))
    except Exception as exc:  # noqa: BLE001 — the reply names every failure
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1
    json.dump(reply, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
