"""Client of the ``umrs-docmodels`` service (ADR-116 amendment, 2026-11-13).

The understand job asks the shared, warm docmodels service first
(``UMRS_KNOWLEDGE_DOCMODELS_URL``, for example ``http://umrs-docmodels:8091``)
and falls back when it cannot answer (``doc_parse`` / ``pretag`` decide
how far). Every failure is a :class:`DocmodelsUnavailable` with a short
reason the caller writes into the job notes:

- ``not_configured``: no URL;
- ``connect_failed``: the service did not accept a connection (one retry);
- ``timeout``: no reply within the timeout;
- ``transport_error``: another HTTP transport failure (the connection died
  after the request was sent, for example the service was killed);
- ``untrusted``: a token is configured and the listener could not prove it
  holds the same token (see below), so no document was sent;
- the service's own error code on a non-200 reply (``busy``, ``loading``,
  ``too_many_pages``, ``too_large``, ``model_mismatch``, ``unauthorized``,
  ``docling_unavailable``, ...), else ``http_<status>``;
- ``bad_reply``: a 200 reply of the wrong shape.

Shared token (``UMRS_KNOWLEDGE_DOCMODELS_TOKEN``; umrs-front generates one
per host, the stand-alone stack uses none on its internal network). Before
it sends a document, text or the token, the client asks ``GET /proof`` with
a fresh nonce and checks ``HMAC-SHA256(token, nonce)``: a process that took
over the service's loopback port while the service was down or restarting
gets neither the token nor any document. The request then carries
``Authorization: Bearer <token>``.

Each request also carries its time budget (``X-Docmodels-Budget``), so the
service waits for a free slot and runs the work no longer than the caller
waits for it, and the model the caller is set to (``artifacts`` /
``model_dir``), so a front with its own model is answered ``model_mismatch``
instead of being served another one.

Synchronous (httpx): the understand job calls it from a worker thread.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Any

import httpx

from . import ingest_models

# A down service must not eat the document's budget: connecting is quick or
# it is not there.
CONNECT_TIMEOUT_S = 3.0
BUDGET_HEADER = "X-Docmodels-Budget"
NONCE_HEADER = "X-Docmodels-Nonce"


class DocmodelsUnavailable(RuntimeError):
    """The service could not answer; ``reason`` is the short note text."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ParseReply:
    text: str
    parser: str  # "docling@<version>"
    pages: int

    @property
    def version(self) -> str:
        return self.parser.split("@", 1)[1] if "@" in self.parser else "unknown"


def proof(token: str, nonce: str) -> str:
    """The service's answer to ``nonce``: hex HMAC-SHA256 keyed by the token."""
    return hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest()


def _base_url(base_url: str | None) -> str:
    url = base_url if base_url is not None else ingest_models.configured_docmodels_url()
    url = url.strip().rstrip("/")
    if not url:
        raise DocmodelsUnavailable("not_configured")
    return url


def _timeout(timeout: float) -> httpx.Timeout:
    seconds = max(0.001, float(timeout))
    return httpx.Timeout(seconds, connect=min(seconds, CONNECT_TIMEOUT_S))


def _new_client(timeout: float) -> httpx.Client:
    """One short-lived client per call (tests replace this seam)."""
    return httpx.Client(timeout=_timeout(timeout))


def _retrying(call: Any) -> httpx.Response:
    """``call()`` with one retry on a refused connection, nothing else retried."""
    try:
        return call()
    except httpx.ConnectError:
        return call()


def _auth_headers(http: httpx.Client, base: str) -> dict[str, str]:
    """``Authorization`` once the listener proved the token; {} without one."""
    token = ingest_models.configured_docmodels_token()
    if not token:
        return {}
    nonce = secrets.token_hex(16)
    reply = _retrying(lambda: http.get(base + "/proof", headers={NONCE_HEADER: nonce}))
    try:
        answer = reply.json().get("proof") if reply.status_code == 200 else None
    except (ValueError, AttributeError):
        answer = None
    if not isinstance(answer, str) or not hmac.compare_digest(
        answer, proof(token, nonce)
    ):
        raise DocmodelsUnavailable("untrusted")
    return {"Authorization": f"Bearer {token}"}


def _exchange(
    http: httpx.Client,
    base: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> httpx.Response:
    """The proof (when a token is set), then the request itself."""
    headers = _auth_headers(http, base)
    if payload is None:
        return _retrying(lambda: http.get(base + path, headers=headers))
    headers[BUDGET_HEADER] = f"{max(0.0, float(timeout)):.3f}"
    return _retrying(lambda: http.post(base + path, json=payload, headers=headers))


def _error_code(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    code = body.get("error") if isinstance(body, dict) else None
    return code if isinstance(code, str) and code else f"http_{response.status_code}"


def _call(
    path: str,
    payload: dict[str, Any] | None,
    timeout: float,
    base_url: str | None,
    client: httpx.Client | None,
) -> dict[str, Any]:
    base = _base_url(base_url)
    own = client is None
    http = client or _new_client(timeout)
    try:
        response = _exchange(http, base, path, payload, timeout)
    except httpx.ConnectError as exc:
        raise DocmodelsUnavailable("connect_failed") from exc
    except httpx.TimeoutException as exc:
        raise DocmodelsUnavailable("timeout") from exc
    except httpx.HTTPError as exc:
        raise DocmodelsUnavailable("transport_error") from exc
    finally:
        if own:
            http.close()
    return _reply_object(response)


def _reply_object(response: httpx.Response) -> dict[str, Any]:
    if response.status_code != 200:
        raise DocmodelsUnavailable(_error_code(response))
    try:
        body = response.json()
    except ValueError as exc:
        raise DocmodelsUnavailable("bad_reply") from exc
    if not isinstance(body, dict):
        raise DocmodelsUnavailable("bad_reply")
    return body


def parse(
    data: bytes,
    filename: str,
    timeout: float,
    *,
    max_pages: int | None = None,
    artifacts: str | None = None,
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> ParseReply:
    """Docling's markdown of one document, parsed by the service.

    ``artifacts`` is the Docling directory this caller is set to; the
    service answers ``model_mismatch`` when it holds another one.
    """
    payload: dict[str, Any] = {
        "filename": filename,
        "data_b64": base64.b64encode(data).decode("ascii"),
    }
    if max_pages is not None:
        payload["max_pages"] = int(max_pages)
    if artifacts:
        payload["artifacts"] = artifacts
    body = _call("/parse", payload, timeout, base_url, client)
    text, parser = body.get("text"), body.get("parser")
    if not isinstance(text, str) or not isinstance(parser, str):
        raise DocmodelsUnavailable("bad_reply")
    pages = body.get("pages")
    return ParseReply(
        text=text, parser=parser, pages=pages if isinstance(pages, int) else 0
    )


def tag(
    texts: list[str],
    labels: list[str],
    timeout: float,
    *,
    threshold: float | None = None,
    model_dir: str | None = None,
    base_url: str | None = None,
    client: httpx.Client | None = None,
) -> list[list[dict[str, Any]]]:
    """GLiNER candidates per text (the ingest worker's ``tag`` reply shape).

    ``model_dir`` is the GLiNER directory this caller is set to; the service
    answers ``model_mismatch`` when it holds another one.
    """
    payload: dict[str, Any] = {"texts": list(texts), "labels": list(labels)}
    if threshold is not None:
        payload["threshold"] = float(threshold)
    if model_dir:
        payload["model_dir"] = model_dir
    body = _call("/tag", payload, timeout, base_url, client)
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(texts):
        raise DocmodelsUnavailable("bad_reply")
    return candidates


def info(
    timeout: float, *, base_url: str | None = None, client: httpx.Client | None = None
) -> dict[str, Any]:
    """The service's ``GET /info`` (what connect-info shows). ``timeout``
    bounds each phase (connect, then each read), not their sum."""
    return _call("/info", None, timeout, base_url, client)


__all__ = ["DocmodelsUnavailable", "ParseReply", "info", "parse", "proof", "tag"]
