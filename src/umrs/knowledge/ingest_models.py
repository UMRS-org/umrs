"""Knowledge ingest model settings (ADR-116 D-9) and the staging CLI.

UMRS never downloads a model at runtime (offline rule). The server image
build stages Docling and GLiNER into ``/opt/umrs/models`` (the bundled
defaults); a maintainer may stage other copies once, with network, into the
front's ``/models`` and point the settings below at them::

    python -m umrs.knowledge.ingest_models stage docling --dest /models/docling
    python -m umrs.knowledge.ingest_models stage gliner --dest /models/gliner_medium-v2.1
    python -m umrs.knowledge.ingest_models show

The staging code, the pins and the staged-directory checks live in
``ingest_models_stage`` (standard library only, the one file the image's
model stage copies) and are re-exported here.

Settings (read by the understand stage and the docmodels service). This
module is their one source of defaults; ``doc_parse``, ``pretag`` and the
service read them through the ``configured_*`` functions below. Since
2026-11-13 both models are on by default, with their weights baked into the
server image:

- ``UMRS_KNOWLEDGE_DOC_PARSER``: ``docling`` (default) or ``pypdf``;
  ``off``, ``none``, ``false`` and ``0`` also mean ``pypdf``.
- ``UMRS_KNOWLEDGE_DOCLING_ARTIFACTS``: Docling's staged directory; empty
  means the bundled ``/opt/umrs/models/docling``.
- ``UMRS_KNOWLEDGE_GLINER_MODEL``: GLiNER's staged directory; empty means
  the bundled ``/opt/umrs/models/gliner_medium-v2.1``; ``off``, ``none``,
  ``false`` and ``0`` switch pre-tagging off.
- ``UMRS_KNOWLEDGE_DOCMODELS_URL``: the shared ``umrs-docmodels`` service,
  asked before the local path; empty or ``off`` means local only.
- ``UMRS_KNOWLEDGE_DOCMODELS_TOKEN``: the service's shared token (umrs-front
  generates one per host); empty means none is sent.
- ``UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK`` (default ``sandbox``): what
  may run locally when a service URL is set and the service could not
  answer (busy, loading, down): ``sandbox`` only the one-shot ingest sandbox
  (``UMRS_EMBEDDING_SANDBOX=podman``, which frees its memory), ``inprocess``
  also this server process (the models then stay loaded in it), ``off``
  nothing. A refusal about the document itself (``SERVICE_FINAL_REASONS``)
  never runs locally. Without a service URL the local path always runs.
- ``UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S`` (default 120): seconds one parse,
  and separately one tagging, may take per document, outside the LLM
  extraction budget (``docmodels_budget``).
- ``UMRS_KNOWLEDGE_DOCLING_MAX_PAGES`` (default 80) and
  ``UMRS_KNOWLEDGE_DOCLING_MAX_BYTES`` (default 64 MiB): a longer PDF, or a
  larger document, is not given to Docling (``doc_parser_fallback:
  too_many_pages`` / ``too_large``). The service reads the same two
  settings, so one value caps both sides.

This module imports only the standard library at load time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import ingest_models_stage as _stage_module
from .ingest_models_stage import (
    BUNDLED_DOCLING_DIR,
    BUNDLED_GLINER_DIR,
    BUNDLED_MODELS_ROOT,
    DEFAULT_GLINER_REPO,
    DOCLING_REVISIONS,
    DOCLING_SUBDIRS,
    GLINER_BACKBONE_REVISION,
    GLINER_REPO,
    GLINER_REVISION,
    GLINER_WEIGHTS,
    MARKER,
    LicenceError,
    StagingError,
    gliner_revision,
    is_docling_staged,
    is_gliner_staged,
    stage_docling,
    stage_gliner,
)

DOC_PARSER_ENV = "UMRS_KNOWLEDGE_DOC_PARSER"
DOCLING_ARTIFACTS_ENV = "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS"
GLINER_MODEL_ENV = "UMRS_KNOWLEDGE_GLINER_MODEL"
DOCMODELS_URL_ENV = "UMRS_KNOWLEDGE_DOCMODELS_URL"
DOCMODELS_TOKEN_ENV = "UMRS_KNOWLEDGE_DOCMODELS_TOKEN"
DOCMODELS_LOCAL_FALLBACK_ENV = "UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK"
DOCMODELS_TIMEOUT_ENV = "UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S"
DOCLING_MAX_PAGES_ENV = "UMRS_KNOWLEDGE_DOCLING_MAX_PAGES"
DOCLING_MAX_BYTES_ENV = "UMRS_KNOWLEDGE_DOCLING_MAX_BYTES"
DOC_PARSERS = ("pypdf", "docling")
DEFAULT_DOC_PARSER = "docling"
# Values that switch a default-on model off.
OFF_VALUES = frozenset({"off", "none", "false", "0"})
DEFAULT_DOCMODELS_TIMEOUT_S = 120.0
DEFAULT_DOCLING_MAX_PAGES = 80
DEFAULT_DOCLING_MAX_BYTES = 64 * 1024 * 1024
LOCAL_FALLBACKS = ("off", "sandbox", "inprocess")
DEFAULT_LOCAL_FALLBACK = "sandbox"
# Service answers about the document itself (or a transfer that died after
# the document was sent): the same bytes are never parsed or tagged again
# locally, whatever the local-fallback setting.
SERVICE_FINAL_REASONS = frozenset(
    {
        "too_large",
        "too_many_pages",
        "parse_failed",
        "tag_failed",
        "bad_request",
        "transport_error",
        "timeout",
    }
)


# ── settings ────────────────────────────────────────────────────────────────


def _setting(name: str) -> str:
    return os.environ.get(name, "").strip()


def configured_doc_parser() -> str:
    """``docling`` (default; any other value too) or ``pypdf`` (``pypdf`` or an off value)."""
    value = _setting(DOC_PARSER_ENV).lower()
    return "pypdf" if value == "pypdf" or value in OFF_VALUES else DEFAULT_DOC_PARSER


def configured_docling_artifacts() -> str:
    """Docling's staged directory; empty means the bundled one."""
    return _setting(DOCLING_ARTIFACTS_ENV) or BUNDLED_DOCLING_DIR


def configured_gliner_dir() -> str:
    """GLiNER's staged directory (empty = the bundled one), or ``""`` when off."""
    value = _setting(GLINER_MODEL_ENV)
    if value.lower() in OFF_VALUES:
        return ""
    return value or BUNDLED_GLINER_DIR


def configured_docmodels_url() -> str:
    """The docmodels service base URL, or ``""`` (unset or ``off``: local only)."""
    value = _setting(DOCMODELS_URL_ENV)
    return "" if value.lower() in OFF_VALUES else value.rstrip("/")


def configured_docmodels_token() -> str:
    """The token the client proves the service with and sends ("" = none)."""
    return _setting(DOCMODELS_TOKEN_ENV)


def configured_local_fallback() -> str:
    """``sandbox`` (default), ``inprocess`` or ``off`` (also an off value)."""
    value = _setting(DOCMODELS_LOCAL_FALLBACK_ENV).lower()
    if value in OFF_VALUES:
        return "off"
    return value if value in LOCAL_FALLBACKS else DEFAULT_LOCAL_FALLBACK


def _positive(name: str, default: float) -> float:
    try:
        value = float(_setting(name) or default)
    except ValueError:
        return default
    return value if value > 0 else default


def configured_docmodels_timeout() -> float:
    """Seconds one parse (or one tagging) may take per document (default 120)."""
    return _positive(DOCMODELS_TIMEOUT_ENV, DEFAULT_DOCMODELS_TIMEOUT_S)


def configured_docling_max_pages() -> int:
    """The longest PDF, in pages, given to Docling (default 80)."""
    return int(_positive(DOCLING_MAX_PAGES_ENV, DEFAULT_DOCLING_MAX_PAGES))


def configured_docling_max_bytes() -> int:
    """The largest document, in bytes, given to Docling (default 64 MiB)."""
    return int(_positive(DOCLING_MAX_BYTES_ENV, DEFAULT_DOCLING_MAX_BYTES))


def docmodels_budget(deadline: float | None, now: float | None = None) -> float:
    """Seconds one docmodels call (one parse, or one tagging) may take now.

    The rule: with ``T = configured_docmodels_timeout()``, a call gets ``T``
    without a job deadline and ``min(T, deadline + T - now)`` with one. So
    each call is capped at ``T`` and is not taken from the LLM window budget,
    and no call runs past ``deadline + T``: the document models push a job
    past its deadline by ``T`` at most, whatever the number of documents.
    ``<= 0`` means the call is skipped (``budget_exhausted``).
    """
    cap = configured_docmodels_timeout()
    if deadline is None:
        return cap
    now = time.monotonic() if now is None else now
    return min(cap, deadline + cap - now)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _show() -> int:
    docling_dir = configured_docling_artifacts()
    gliner_dir = configured_gliner_dir()
    parser = configured_doc_parser()
    docling_ok = is_docling_staged(docling_dir)
    gliner_ok = not gliner_dir or is_gliner_staged(gliner_dir)
    print(f"{DOC_PARSER_ENV}={parser}")
    print(
        f"{DOCLING_ARTIFACTS_ENV}={docling_dir} ({'staged' if docling_ok else 'NOT staged'})"
    )
    gliner_state = "staged" if gliner_ok else "NOT staged"
    print(
        f"{GLINER_MODEL_ENV}={gliner_dir} ({gliner_state})"
        if gliner_dir
        else f"{GLINER_MODEL_ENV}=off"
    )
    print(f"{DOCMODELS_URL_ENV}={configured_docmodels_url() or '(local only)'}")
    print(f"{DOCMODELS_LOCAL_FALLBACK_ENV}={configured_local_fallback()}")
    return 0 if gliner_ok and (docling_ok or parser == "pypdf") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m umrs.knowledge.ingest_models")
    sub = parser.add_subparsers(dest="command", required=True)
    _stage_module.add_stage_commands(sub)
    sub.add_parser(
        "show",
        help="print the configured parser and models and whether they are staged",
    )
    args = parser.parse_args(argv)
    if args.command == "show":
        return _show()
    return _stage_module.run_command(args)


__all__ = [
    "BUNDLED_DOCLING_DIR",
    "BUNDLED_GLINER_DIR",
    "BUNDLED_MODELS_ROOT",
    "DEFAULT_GLINER_REPO",
    "DOCLING_REVISIONS",
    "DOCLING_SUBDIRS",
    "GLINER_BACKBONE_REVISION",
    "GLINER_REPO",
    "GLINER_REVISION",
    "GLINER_WEIGHTS",
    "MARKER",
    "SERVICE_FINAL_REASONS",
    "LicenceError",
    "StagingError",
    "configured_doc_parser",
    "configured_docling_artifacts",
    "configured_docling_max_bytes",
    "configured_docling_max_pages",
    "configured_docmodels_timeout",
    "configured_docmodels_token",
    "configured_docmodels_url",
    "configured_gliner_dir",
    "configured_local_fallback",
    "docmodels_budget",
    "gliner_revision",
    "is_docling_staged",
    "is_gliner_staged",
    "main",
    "stage_docling",
    "stage_gliner",
]


if __name__ == "__main__":
    sys.exit(main())
