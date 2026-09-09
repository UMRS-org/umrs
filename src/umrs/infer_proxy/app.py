"""Inference proxy app (R1-2, design §3.3, D-1/D-3/D-6).

Starlette ASGI service. Per request:

  1. session token → live lease read (no cache); binding checks
     (session/sid/ip/source).
  2. Path policy per provider kind; 403 path_denied with audit on miss.
  3. Unseal the provider key (private half lives only here).
  4. Re-resolve allow_host, reject non-global IPs; strip all inbound
     credential shapes; inject the provider credential; stream.

Upstream 3xx never followed; off-host Location → 502 redirect_offhost.
Audit per request via the broker audit table (no bodies, no headers, no
tokens). No per-user labels in metrics.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from umrs.security.provider_url import _is_global_unicast as _ip_is_public
from umrs.security.seal import SealedBlob, SealError, unseal_credential
from umrs.security.sealed_sources import SealedSourceStore

from .leases import check_lease, extract_session_token, lease_by_token
from .policies import allows, inject_credential, policy_for, strip_credential_headers
from .urljoin import join_provider_url

logger = logging.getLogger("umrs.infer_proxy")

try:  # h2 is optional: image flavours today ship http/1.1 only.
    import h2 as _h2  # noqa: F401

    _HTTP2_AVAILABLE = True
except ImportError:
    _HTTP2_AVAILABLE = False
    logger.info("h2 unavailable — inference proxy will offer HTTP/1.1 upstream")

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class SecretRedactingFilter(logging.Filter):
    """Never log tokens/keys even inside exception traces (design §3.3.4)."""

    _PATTERNS = (
        re.compile(r"(?i)bearer\s+\S+"),
        re.compile(r"(?i)(authorization|x-api-key|x-goog-api-key|api-key)[:=]\s*\S+"),
        re.compile(r"sk-[A-Za-z0-9_-]{4,}"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in self._PATTERNS:
            message = pattern.sub("<redacted>", message)
        record.msg = message
        record.args = ()
        return True


def _json_error(status: int, reason: str) -> JSONResponse:
    return JSONResponse({"error": reason}, status_code=status)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def _broker_from(request: Request) -> Any:
    return getattr(request.app.state, "broker", None)


def _store_from(request: Request) -> Any:
    return getattr(request.app.state, "sealed_store", None)


def _http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def _audit_broker(
    request: Request, lease_id: str, operation: str, detail: str, actor: str = "proxy"
) -> None:
    broker = _broker_from(request)
    if broker is not None:
        broker._audit_log(lease_id, operation, actor, detail)


def resolve_checked_ips(host: str) -> list[str]:
    """Resolve once, validate every address is global unicast, return sorted.

    The pin contract (R2-2 / TOCTOU): the SAME list is the only set the
    connector may dial — no second resolution at connect time.
    """
    import ipaddress
    import socket

    try:
        ipaddress.ip_address(host)
        return [host] if _ip_is_public(host) else []
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, 0, socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    ips = sorted({info[4][0] for info in infos})
    if not ips:
        return []
    return ips if all(_ip_is_public(ip) for ip in ips) else []


def _pin_url(url: str, pin_ip: str) -> str:
    """Swap the URL host for the pinned IP (Host header + SNI stay the original)."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    host_str = f"[{pin_ip}]" if ":" in pin_ip else pin_ip
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host_str}{port}{parts.path or '/'}{('?' + parts.query) if parts.query else ''}"


async def healthz(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@dataclass
class _Prepared:
    """Authorization + source resolution output (or the error to return)."""

    error: Any = None
    lease: Any = None
    full: Any = None
    policy: Any = None
    host: str = ""
    path: str = ""
    pin_ip: str = ""
    started: float = 0.0


def _denied(request: Request, lease: Any, status: int, reason: str) -> JSONResponse:
    if lease is not None:
        _audit_broker(request, lease.lease_id, "deny", reason)
    return _json_error(status, reason)


def _verbatim_error_status(reason: str) -> int:
    return 401 if reason.startswith(("token", "lease", "ip", "sid")) else 403


def _prepare(request: Request) -> _Prepared:
    """Token → lease verdict → sealed source → policy + host check."""
    sid = request.path_params["sid"]
    token = extract_session_token(request.headers)
    if not token:
        return _Prepared(error=_json_error(401, "token_invalid"))
    pool = getattr(request.app.state, "umrs_db_pool", None)
    lease = lease_by_token(pool, token)
    client_ip = _client_ip(request)
    verdict = check_lease(
        lease,
        sid=sid,
        client_ip=client_ip,
        source_override=request.headers.get("x-r4-source"),
    )
    logger.info(
        "infer_eval sid=%s client=%s verdict=%s",
        sid,
        client_ip,
        verdict.reason or "ok",
    )
    if verdict.deny():
        return _Prepared(
            error=_denied(
                request,
                verdict.lease,
                _verbatim_error_status(verdict.reason),
                verdict.reason,
            )
        )
    lease = verdict.lease
    store = _store_from(request)
    full = store.fetch_for_unseal(verdict.source_id) if store else None
    if full is None:
        return _Prepared(error=_denied(request, lease, 410, "source_tombstoned"))
    path = "/" + request.path_params.get("path", "")
    if request.query_params:
        path += "?" + str(request.query_params)
    policy = policy_for(full.provider_kind)
    if not allows(policy, request.method, path, full.allow_paths):
        return _Prepared(error=_denied(request, lease, 403, "path_denied"))
    host = full.allow_host.split("://", 1)[1].split(":")[0].strip("[]")
    pin_ip = _first_ip_or_none(resolve_checked_ips(host))
    if pin_ip is None:
        return _Prepared(error=_denied(request, lease, 470, "host_denied"))
    return _Prepared(
        lease=lease,
        full=full,
        policy=policy,
        host=host,
        path=path,
        pin_ip=pin_ip,
        started=time.monotonic(),
    )


def _first_ip_or_none(ips: list[str]) -> str | None:
    """Deterministic pin — sorted first; empty = host denied."""
    return ips[0] if ips else None


def _unseal_api_key(request: Request, full: Any) -> str:
    blob = SealedBlob(
        ciphertext=full.ciphertext,
        nonce=full.nonce,
        dek_wrapped=full.dek_wrapped,
        seal_key_id="",  # not needed to unseal; row carries its own key id for rotation
    )
    return unseal_credential(
        blob,
        privkey_b64=request.app.state.unseal_privkey,
        owner=full.owner,
        name=full.name,
        provider_kind=full.provider_kind,
        allow_host=full.allow_host,
    )


def _upstream_headers(
    request: Request, policy: Any, api_key: str, allow_host: str
) -> dict:
    headers = {}
    for name, value in request.headers.raw:
        decoded = name.decode("latin-1").lower()
        if decoded in _HOP_BY_HOP:
            continue
        headers[decoded] = value.decode("latin-1")
    headers = strip_credential_headers(headers)
    headers = inject_credential(policy, headers, api_key)
    tail = allow_host.split("://", 1)[1]
    headers["host"] = tail if ":" in tail else tail.split(":", 1)[0]
    return headers


def _check_redirect(
    request: Request, upstream: Any, host: str, req_headers: dict
) -> bool:
    """302/3xx off-host → must 502 redirect_offhost. Returns True when off-host."""
    if not (300 <= upstream.status_code < 400):
        return False
    location = upstream.headers.get("location", "")
    if not location:
        return False
    from urllib.parse import urlsplit

    location_host = urlsplit(location).hostname or ""
    return bool(location_host and location_host != host)


class SessionRateLimiter:
    """Sliding-window per-session rate limiter (R2-3, opt-in; off at 0).

    In-memory by design: the proxy is single-process per venue profile;
    a horizontally scaled deployment pins one proxy per egress-net (ops
    runbook notes the per-process bound as the intended ceiling).
    """

    def __init__(self, rpm: int = 0) -> None:
        self._rpm = rpm
        self._windows: dict[str, list[float]] = {}

    def allow(self, sid: str) -> bool:
        if self._rpm <= 0:
            return True
        now = time.monotonic()
        window = self._windows.setdefault(sid, [])
        cutoff = now - 60.0
        window[:] = [t for t in window if t >= cutoff]
        if len(window) >= self._rpm:
            return False
        window.append(now)
        return True


async def proxy_request(request: Request) -> Any:
    """ANY /s/{sid}/{path:path} — the credential-injecting forward."""
    prep = _prepare(request)
    if prep.error is not None:
        return prep.error
    lease = prep.lease

    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None and not limiter.allow(request.path_params["sid"]):
        return _denied(request, lease, 429, "rate_denied")

    try:
        api_key = _unseal_api_key(request, prep.full)
    except SealError:
        return _denied(request, lease, 500, "unseal_failed")

    headers = _upstream_headers(request, prep.policy, api_key, prep.full.allow_host)
    url = join_provider_url(prep.full.base_url, prep.path)
    if prep.pin_ip:
        url = _pin_url(url, prep.pin_ip)
    client = _http_client(request)
    try:
        upstream = await client.send(
            client.build_request(
                request.method,
                url,
                headers=headers,
                content=request.stream(),
                extensions={"sni_hostname": prep.host} if prep.pin_ip else None,
            ),
            stream=True,
            follow_redirects=False,
        )
    except httpx.HTTPError:
        return _denied(request, lease, 502, "upstream_unreachable")

    if _check_redirect(request, upstream, prep.host, headers):
        await upstream.aclose()
        return _denied(request, lease, 502, "redirect_offhost")

    response_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "x-accel-buffering"
    }
    elapsed_ms = int((time.monotonic() - prep.started) * 1000)

    async def _body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    def _audit_done() -> None:
        _audit_broker(
            request,
            lease.lease_id,
            "infer",
            f"{request.method} {request.url.path} {upstream.status_code} ms={elapsed_ms}",
        )

    return StreamingResponse(
        _body(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(_audit_done),
    )


def create_app(
    *,
    umrs_db_pool: Any,
    unseal_privkey: str,
    sealed_store: Any | None = None,
    broker: Any | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Starlette:
    """App factory (tests inject fakes; production wires the real pool/key)."""
    from umrs.jobs.secret_broker import SecretBroker

    app = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route(
                "/s/{sid}/{path:path}",
                proxy_request,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
            ),
        ]
    )
    app.state.umrs_db_pool = umrs_db_pool
    app.state.unseal_privkey = unseal_privkey
    app.state.sealed_store = sealed_store or SealedSourceStore(umrs_db_pool)
    app.state.broker = broker or SecretBroker(db_pool=umrs_db_pool)
    app.state.http_client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=600.0, pool=10.0),
        http2=_HTTP2_AVAILABLE,  # provider offers h2 else http/1.1 (design §2.2)
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )
    # R2-3: per-session rate cap, off by default (UMRS_INFER_RPM env may set it).
    try:
        app.state.rate_limiter = SessionRateLimiter(
            rpm=int(os.environ.get("UMRS_INFER_RPM", "0"))
        )
    except ValueError:
        logger.warning("UMRS_INFER_RPM unparsable — rate limiter OFF")
        app.state.rate_limiter = SessionRateLimiter(0)
    # Logging discipline: filter tokens out of every proxy log line.
    for handler in logging.getLogger("umrs.infer_proxy").handlers or [
        logging.NullHandler()
    ]:
        handler.addFilter(SecretRedactingFilter())
    return app
