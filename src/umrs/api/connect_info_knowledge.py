"""The ``knowledge_models`` block of connect-info (ADR-116 amendment 2026-11-13).

Docling (PDF layout parsing) and GLiNER (entity pre-tagging) are on by
default, served warm by the ``umrs-docmodels`` service when the front has
one, else run by the understand job itself. This block tells a client what
the front is set to and whether the service answers, so ``r4 doctor`` can
say it.

connect-info is readable without signing in (it is the discovery call), so
the block is address-free: it says whether a service URL is set, never the
URL, and whether a model directory is the bundled one or a custom one, never
the path.

``service.ready`` is true only when the service answered that it has
finished loading, is not stuck, and has loaded (not merely staged) every
model this front has switched on, from the directory this front is set to
(a front with its own model directory is not served by the shared service).
``service.local_fallback`` says what this server runs itself when the
service cannot answer: ``inprocess``, ``sandbox`` (the ingest sandbox lane
is set up), or ``off`` (pypdf and no hints; also ``sandbox`` set without
the lane).

The service's ``GET /info`` is asked at most once a minute (``CACHE_TTL_S``)
with ``INFO_TIMEOUT_S`` per connect and per read (the token proof, when a
token is set, is one more read), so a slow or absent service costs
connect-info a few seconds once a minute. A failure is cached the same way,
and concurrent requests wait for one refill instead of each asking.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from umrs.knowledge import docmodels_client
from umrs.knowledge import ingest_models as im

CACHE_TTL_S = 60.0
INFO_TIMEOUT_S = 2.0

Info = dict[str, Any] | None
FetchInfo = Callable[[float], Info]

_cache: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
_refill = threading.Lock()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _settings() -> tuple[str, str, str, str]:
    """Parser, Docling dir, GLiNER dir ("" = off), service URL ("" = none)."""
    return (
        im.configured_doc_parser(),
        im.configured_docling_artifacts(),
        im.configured_gliner_dir(),
        im.configured_docmodels_url(),
    )


def fetch_service_info(timeout: float = INFO_TIMEOUT_S) -> Info:
    """The docmodels service's ``/info`` answer, or None when it does not answer."""
    if not im.configured_docmodels_url():
        return None
    try:
        return docmodels_client.info(timeout=timeout)
    except docmodels_client.DocmodelsUnavailable:
        return None


def _source(path: str, bundled: str) -> str:
    if not path:
        return "off"
    return "bundled" if path.rstrip("/") == bundled.rstrip("/") else "custom"


def _docling_version() -> str:
    try:
        return version("docling")
    except PackageNotFoundError:
        return ""


def _docling_block(parser: str, artifacts: str, info: Info) -> dict[str, Any]:
    reported = _mapping(_mapping(info).get("docling"))
    return {
        "enabled": parser == "docling",
        "source": _source(artifacts, im.BUNDLED_DOCLING_DIR),
        "staged": bool(im.is_docling_staged(artifacts)),
        "version": str(reported.get("version") or _docling_version()),
    }


def _gliner_block(gliner_dir: str, info: Info) -> dict[str, Any]:
    reported = _mapping(_mapping(info).get("gliner"))
    source = _source(gliner_dir, im.BUNDLED_GLINER_DIR)
    bundled = source == "bundled"
    return {
        "enabled": bool(gliner_dir),
        "source": source,
        "staged": bool(gliner_dir) and bool(im.is_gliner_staged(gliner_dir)),
        "repo": str(reported.get("repo") or (im.GLINER_REPO if bundled else "")),
        "revision": str(
            reported.get("revision") or (im.GLINER_REVISION if bundled else "")
        ),
    }


def _loaded(info: Info, name: str, key: str, path: str) -> bool:
    """The service reports ``name`` loaded from ``path`` (its ``key`` field)."""
    block = _mapping(_mapping(info).get(name))
    served = str(block.get(key) or "").rstrip("/")
    return block.get("loaded") is True and served == path.rstrip("/")


def _service_ready(info: Info, parser: str, artifacts: str, gliner_dir: str) -> bool:
    """The service finished loading, is not stuck, and has loaded every
    model this front has switched on, from this front's directory."""
    if info is None or info.get("ready") is not True or info.get("stuck"):
        return False
    docling_ok = parser != "docling" or _loaded(info, "docling", "artifacts", artifacts)
    gliner_ok = not gliner_dir or _loaded(info, "gliner", "dir", gliner_dir)
    return docling_ok and gliner_ok


def local_fallback() -> str:
    """The local fallback in effect (``doc_parse.local_fallback_refusal``)."""
    from umrs.knowledge.doc_parse import local_fallback_refusal

    return "off" if local_fallback_refusal() else im.configured_local_fallback()


def _service_block(info: Info, settings: tuple[str, str, str, str]) -> dict[str, Any]:
    parser, artifacts, gliner_dir, url = settings
    if not url:
        return {"url_set": False, "ready": None}
    return {
        "url_set": True,
        "ready": _service_ready(info, parser, artifacts, gliner_dir),
        "local_fallback": local_fallback(),
    }


def build_knowledge_models(fetch: FetchInfo | None = None) -> dict[str, Any]:
    """The block, uncached: settings, local staging, and the service's answer."""
    parser, artifacts, gliner_dir, url = _settings()
    info = (fetch or fetch_service_info)(INFO_TIMEOUT_S) if url else None
    return {
        "doc_parser": parser,
        "docling": _docling_block(parser, artifacts, info),
        "gliner": _gliner_block(gliner_dir, info),
        "service": _service_block(info, (parser, artifacts, gliner_dir, url)),
    }


def connect_info_knowledge_models(
    *,
    fetch: FetchInfo | None = None,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """The block, reused for ``CACHE_TTL_S`` while the settings are unchanged."""
    key = _settings()
    with _refill:
        hit = _cache.get(key)
        if hit is not None and now() - hit[0] < CACHE_TTL_S:
            return hit[1]
        block = build_knowledge_models(fetch)
        _cache[key] = (now(), block)
        return block


def clear_cache() -> None:
    """Forget the cached block (tests)."""
    _cache.clear()
