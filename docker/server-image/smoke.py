"""Server image smoke test (ADR-100): run INSIDE the built umrs-server image.

    podman run --rm localhost/umrs-server:latest python /opt/umrs/server-image-smoke.py

Dockerfile.server runs it as the last build step, so an image that cannot
import its key libraries or load the spaCy model is never tagged. It would
have failed the 2026-10-25 build: preshed 3.0.14 made
``spacy.load("en_core_web_sm")`` raise OverflowError, and the rccs Text
Embedding experience lost its text modalities at boot.

It loads the model the way umrs.inference.loading.EmbeddingModel does.
Exit 0 and one line per check on success; the first failure raises.
"""

from __future__ import annotations

import importlib
import sys
import time

# Import names of the libraries the server process and its modality loaders
# import. A missing or ABI-broken wheel fails here, not at the first request.
LIBRARIES = (
    "zmq", "boto3", "PIL", "psycopg2", "asyncpg", "pandas", "tabulate",
    "diskcache", "jsonschema", "httpx", "requests", "yaml", "fastapi",
    "uvicorn", "multipart", "jose", "torch", "torchvision", "transformers",
    "sentence_transformers", "sklearn", "spacy", "langchain",
    "langchain_community", "langchain_openai", "bentoml", "cv2", "sysv_ipc",
    "pypdf", "pptx", "rich",
)

# umrs modules on the server's boot path and the embedding modality loader.
UMRS_MODULES = ("umrs.inference.loading", "umrs.server.main", "umrs.knowledge.text_embedder")

SPACY_MODEL = "en_core_web_sm"


def _timed(label: str, func) -> None:
    start = time.monotonic()
    func()
    print(f"ok  {label} ({time.monotonic() - start:.1f}s)", flush=True)


def check_imports(names: tuple[str, ...]) -> None:
    for name in names:
        _timed(f"import {name}", lambda name=name: importlib.import_module(name))


def check_spacy_model() -> None:
    import spacy

    nlp = spacy.load(SPACY_MODEL)
    doc = nlp("UMRS embeds text. It splits sentences first.")
    assert len(doc) > 0, "the model produced no tokens"
    # The EmbeddingModel path: a reduced pipeline plus the sentencizer.
    lean = spacy.load(SPACY_MODEL, disable=["ner", "tagger", "parser"])
    lean.add_pipe("sentencizer")
    sentences = list(lean("UMRS embeds text. It splits sentences first.").sents)
    assert len(sentences) == 2, f"expected 2 sentences, got {len(sentences)}"


def main() -> int:
    check_imports(LIBRARIES)
    check_imports(UMRS_MODULES)
    _timed(f"spacy.load({SPACY_MODEL!r}) + sentencizer", check_spacy_model)
    print("server image smoke: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
