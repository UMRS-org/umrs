"""Inference proxy app (R1-2, design §3.3, D-1/D-3/D-6).

Starlette ASGI service. Per request:

  1. session token → live lease read (no cache); binding checks
     (session/sid/ip/source).
  2. Source-class gate (ADR-062/D-51): a lease whose project is
     ``data_class=sensitive`` never reaches a public-kind source —
     403 ``source_class_refused`` (no project claim → no gate) — unless
     the source's host is on the site's institutional-provider allowlist
     (``UMRS_INSTITUTIONAL_PROVIDER_HOSTS``, Rikyu by default; ADR-062
     amendment 2026-10-15), which a sensitive project may use.
  3. Path policy per provider kind; 403 path_denied with audit on miss.
  4. Unseal the provider key (private half lives only here).
  5. Re-resolve allow_host, reject non-global IPs; strip all inbound
     credential shapes; inject the provider credential; stream.

Upstream 3xx never followed; off-host Location → 502 redirect_offhost.
Audit per request via the broker audit table (no bodies, no headers, no
tokens). No per-user labels in metrics.

ADR-099 (RS-1 lane Q): after the lease verdict and the per-session limiter,
the quota enforcer (when wired) refuses a metered call over the lease
owner's daily token or per-minute request cap with a typed 429
(``quota.tokens_per_day_exceeded`` / ``quota.rpm_exceeded``), fail closed as
``quota.unavailable`` (503). Every relayed call writes one
``llm_usage_log`` row (``origin='infer_proxy'``) after the answer was sent;
the token counts come from a bounded head/tail copy of the answer, so the
upstream is asked for an uncompressed (``identity``) answer.

Workstation leases (ADR-062 amendment 2, S-D37-3): ``/w/{sid}/{path}`` is
the route the edge's ``/gw/infer/`` relays to; it admits only a
``kind=workstation`` lease (``/s/`` refuses one). On it the source must be
the lease owner's own sealed row (403 ``source_not_owned``), the class gate
fails closed (a public provider needs the project's ``data_class`` to read
``non_sensitive``, else 403 ``source_class_refused``; an institutional host
stays admitted), the site source is not served (``/gw/llm`` serves it), and
edge-added identity and forwarding headers plus credential-shaped query
parameters are stripped before the provider is called.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from umrs.quotas.usage_tap import PromptMeter, UsageTap, record_proxy_usage
from umrs.security.provider_url import _is_global_unicast as _ip_is_public
from umrs.security.seal import SealedBlob, SealError, unseal_credential
from umrs.security.sealed_sources import SealedSourceStore
from umrs.shared.project_data_class import (
    PUBLIC_PROVIDER_KINDS,
    data_class_for_project,
    is_institutional_url,
)
from umrs.shared.site_source import SITE_PATH_PREFIX, SITE_SOURCE_ID

from . import site_models
from .leases import (
    check_route_lease,
    extract_session_token,
    is_workstation_lease,
    lease_by_token,
)
from .policies import allows, inject_credential, policy_for, strip_credential_headers
from .urljoin import join_provider_url

if TYPE_CHECKING:
    from umrs.quotas.enforcer import QuotaEnforcer

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

# Calls the quotas meter: listing models costs no tokens (and a worker's
# bootstrap lists them), so reads stay open at the cap.
_UNMETERED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Providers whose streams report usage only when asked (stream_options).
_OPENAI_STYLE_KINDS = frozenset({"openai", "openai_compat", "azure_openai"})


class SecretRedactingFilter(logging.Filter):
    """Never log tokens/keys even inside exception traces (design §3.3.4)."""

    _PATTERNS = (
        re.compile(r"(?i)bearer\s+\S+"),
        re.compile(
            r"(?i)(authorization|x-api-key|x-goog-api-key|api-key"
            r"|x-umrs-host-token)[:=]\s*\S+"
        ),
        re.compile(r"sk-[A-Za-z0-9_-]{4,}"),
        # ADR-098 D-5: host tokens and enrollment tokens.
        re.compile(r"umrs-(host\.v1\.|enroll-)\S+"),
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


# Headers the edge (nginx, the auth sidecar) or the client adds that must not
# reach a public provider from the workstation route: identity claims,
# forwarding chains (the user's address) and the source selector.
_EDGE_HEADER_PREFIXES = ("x-forwarded-", "x-verified-", "x-umrs-", "x-r4-")
_EDGE_HEADERS = frozenset(
    {"forwarded", "x-real-ip", "x-user-id", "x-project-id", "x-request-id", "via"}
)
# Credential-shaped query parameters (Google's ``?key=`` and its kin).
_CREDENTIAL_QUERY_KEYS = frozenset({"key", "api_key", "api-key", "access_token"})


def _strip_edge_headers(headers: dict) -> dict:
    return {
        name: value
        for name, value in headers.items()
        if name not in _EDGE_HEADERS and not name.startswith(_EDGE_HEADER_PREFIXES)
    }


def _request_path(request: Request, workstation: bool) -> str:
    """``/<path>[?query]``; the workstation route drops credential params."""
    path = "/" + request.path_params.get("path", "")
    if not request.query_params:
        return path
    if not workstation:
        return path + "?" + str(request.query_params)
    items = [
        (key, value)
        for key, value in request.query_params.multi_items()
        if key.lower() not in _CREDENTIAL_QUERY_KEYS
    ]
    return path + ("?" + urlencode(items) if items else "")


def _verbatim_error_status(reason: str) -> int:
    return 401 if reason.startswith(("token", "lease", "ip", "sid", "host_")) else 403


def _request_host(request: Request) -> tuple[str, Any]:
    """(verified host id or '', refusal) from the optional host token,
    checked before the lease (ADR-098 D-5/D-6)."""
    from umrs.hosts.identity import (
        HostAuthError,
        host_keys_from_state,
        presented_host,
    )

    try:
        store = host_keys_from_state(request.app.state)
        return presented_host(request.headers, store), None
    except HostAuthError as exc:
        return "", _json_error(exc.status, exc.reason)


def _source_header(request: Request) -> str | None:
    """``x-umrs-source`` or ``x-r4-source`` (U3 S6b); None when neither."""
    from umrs.shared.compat_names import header

    return header(request.headers, "source") or None


def _prepare(request: Request, *, workstation: bool = False) -> _Prepared:
    """Token → lease verdict → sealed source → policy + host check."""
    sid = request.path_params["sid"]
    token = extract_session_token(request.headers)
    if not token:
        return _Prepared(error=_json_error(401, "token_invalid"))
    host, refusal = _request_host(request)
    if refusal is not None:
        return _Prepared(error=refusal)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    lease = lease_by_token(pool, token)
    client_ip = _client_ip(request)
    verdict = check_route_lease(
        lease,
        workstation=workstation,
        sid=sid,
        client_ip=client_ip,
        source_override=_source_header(request),
        host=host,
    )
    logger.info(
        "infer_eval sid=%s client=%s verdict=%s path=%s method=%s",
        sid,
        client_ip,
        verdict.reason or "ok",
        "/" + request.path_params.get("path", ""),
        request.method,
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
    return _resolve_source(request, pool, verdict.lease, verdict.source_id)


def _resolve_source(
    request: Request, pool: Any, lease: Any, source_id: str
) -> _Prepared:
    """Sealed source → class gate → policy + host check (the verdict passed)."""
    store = _store_from(request)
    full = store.fetch_for_unseal(source_id) if store else None
    if full is None:
        return _Prepared(error=_denied(request, lease, 410, "source_tombstoned"))
    refusal = _gate_refusal(pool, lease, full)
    if refusal:
        return _Prepared(error=_denied(request, lease, 403, refusal))
    path = _request_path(request, is_workstation_lease(lease))
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


def _gate_refusal(pool: Any, lease: Any, full: Any) -> str:
    """The refusal reason for this lease and source, ``""`` when admitted."""
    if is_workstation_lease(lease):
        return _workstation_refusal(pool, lease, full)
    return "source_class_refused" if _source_class_refused(pool, lease, full) else ""


def _workstation_refusal(pool: Any, lease: Any, full: Any) -> str:
    """S-D37-3: the edge-reachable route fails closed.

    The source must be the lease owner's own row (a lease minted through the
    generic broker API may name any source id). A public-kind source outside
    the site's institutional allowlist needs the project to read
    ``non_sensitive``: an unknown class (no row, no pool, a failed lookup)
    refuses, unlike the sandbox route, which never invents a class.
    """
    if str(getattr(full, "owner", "") or "") != lease.owner_id:
        return "source_not_owned"
    if full.provider_kind not in PUBLIC_PROVIDER_KINDS:
        return ""
    if is_institutional_url(getattr(full, "allow_host", "")):
        return ""
    if data_class_for_project(pool, lease.project) != "non_sensitive":
        return "source_class_refused"
    return ""


def _source_class_refused(pool: Any, lease: Any, full: Any) -> bool:
    """ADR-062 (D-51 lease seam): True when the lease's project is
    ``data_class=sensitive`` and the resolved source is a public-kind
    provider. No project claim on the lease → no gate (pre-gate
    behavior); lookup failure/absent row → no gate (never invented)."""
    if not getattr(lease, "project", ""):
        return False
    if full.provider_kind not in PUBLIC_PROVIDER_KINDS:
        return False
    # ADR-062 amendment (2026-10-15): an institutional host (the site's
    # allowlist — Rikyu by default) is admitted for sensitive projects.
    if is_institutional_url(getattr(full, "allow_host", "")):
        return False
    return data_class_for_project(pool, lease.project) == "sensitive"


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
    request: Request, policy: Any, api_key: str, allow_host: str, workstation: bool
) -> dict:
    headers = {}
    for name, value in request.headers.raw:
        decoded = name.decode("latin-1").lower()
        if decoded in _HOP_BY_HOP:
            continue
        headers[decoded] = value.decode("latin-1")
    headers = strip_credential_headers(headers)
    if workstation:
        headers = _strip_edge_headers(headers)
    headers = inject_credential(policy, headers, api_key)
    headers["accept-encoding"] = "identity"  # the usage tap reads plain bytes
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


def _quota_refusal(request: Request, lease: Any) -> JSONResponse | None:
    """ADR-099 D-5: the typed 429 (or fail-closed 503) before unsealing."""
    enforcer = getattr(request.app.state, "quota_enforcer", None)
    if enforcer is None or request.method in _UNMETERED_METHODS:
        return None
    denial = enforcer.check(lease)
    if denial is None:
        return None
    _denied(request, lease, denial.status, denial.code)
    return denial.to_response()


def _rate_or_quota_refusal(request: Request, lease: Any) -> JSONResponse | None:
    """The per-session limiter (R2-3), then the owner's quotas (ADR-099 D-5)."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None and not limiter.allow(request.path_params["sid"]):
        return _denied(request, lease, 429, "rate_denied")
    return _quota_refusal(request, lease)


_FALSE_WORDS = frozenset({"", "0", "false", "f", "no", "n", "off"})


def _lax_bool(value: Any) -> bool:
    """``stream`` as a lenient server reads it (vLLM's pydantic takes ``1``
    and ``"true"`` as true): a false word, zero or null is false, anything
    else true, so an odd spelling never streams unmetered."""
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_WORDS
    return bool(value)


def with_stream_usage(payload: Any) -> bool:
    """Ask a streamed OpenAI-style completion for its usage chunk (ADR-099
    D-4): ``stream_options.include_usage = true`` when ``stream`` is true.

    Without it an OpenAI or vLLM stream carries no usage, the row records
    zero tokens and the daily token cap never binds. A ``stream`` that is
    not a JSON boolean (``1``, ``"true"``) is sent as the boolean it reads
    as (Amendment 2). True when ``payload`` was changed."""
    if not isinstance(payload, dict) or "stream" not in payload:
        return False
    raw = payload["stream"]
    payload["stream"] = _lax_bool(raw)
    changed = raw is not payload["stream"]
    if not payload["stream"]:
        return changed
    options = payload.get("stream_options")
    options = dict(options) if isinstance(options, dict) else {}
    if options.get("include_usage") is True:
        return changed
    options["include_usage"] = True
    payload["stream_options"] = options
    return True


def _metered_completion(request: Request, prep: _Prepared) -> bool:
    """A POST to an OpenAI-style completions path (where ``stream`` lives)."""
    kind = str(getattr(prep.full, "provider_kind", "") or "")
    path = str(prep.path or "").split("?", 1)[0].rstrip("/")
    return (
        request.method == "POST"
        and kind in _OPENAI_STYLE_KINDS
        and path.endswith("completions")
    )


class _Sent:
    """Request bytes sent upstream: the prompt side of a usage estimate
    (``meter``: images and other base64 runs at a fixed cost, Amendment 3)."""

    def __init__(self) -> None:
        self.size = 0
        self.meter = PromptMeter()

    async def counted(self, stream: Any) -> Any:
        async for chunk in stream:
            self.size += len(chunk)
            self.meter.feed(chunk)
            yield chunk

    def body(self, raw: bytes) -> bytes:
        self.size = len(raw)
        self.meter = PromptMeter()
        self.meter.feed(raw)
        return raw


def _sent_body(raw: bytes) -> _Sent:
    sent = _Sent()
    sent.body(raw)
    return sent


async def _upstream_content(request: Request, prep: _Prepared, sent: _Sent) -> Any:
    """The body sent upstream: streamed through unchanged, except a streamed
    OpenAI-style completion, which is re-encoded with ``include_usage``."""
    if not _metered_completion(request, prep):
        return sent.counted(request.stream())
    raw = await request.body()
    try:
        payload = json.loads(raw)
    except ValueError:
        return sent.body(raw)
    changed = with_stream_usage(payload)
    return sent.body(json.dumps(payload).encode() if changed else raw)


async def _send_upstream(
    request: Request, prep: _Prepared, headers: dict, sent: _Sent
) -> Any:
    """The streamed provider call: pinned IP, original Host + SNI, no redirects."""
    url = join_provider_url(prep.full.base_url, prep.path)
    if prep.pin_ip:
        url = _pin_url(url, prep.pin_ip)
    content = await _upstream_content(request, prep, sent)
    client = _http_client(request)
    return await client.send(
        client.build_request(
            request.method,
            url,
            headers=headers,
            content=content,
            extensions={"sni_hostname": prep.host} if prep.pin_ip else None,
        ),
        stream=True,
        follow_redirects=False,
    )


async def proxy_request(request: Request) -> Any:
    """ANY /s/{sid}/{path:path} — the credential-injecting forward."""
    if request.path_params.get("path", "").startswith(SITE_PATH_PREFIX):
        return await _proxy_site(request)
    return await _relay_sealed(request, workstation=False)


async def proxy_workstation(request: Request) -> Any:
    """ANY /w/{sid}/{path:path} — the edge's ``/gw/infer/`` (S-D37-3).

    Only a workstation lease; the site source is ``/gw/llm``'s, not this."""
    if request.path_params.get("path", "").startswith(SITE_PATH_PREFIX):
        return _json_error(403, "path_denied")
    return await _relay_sealed(request, workstation=True)


async def _relay_sealed(request: Request, *, workstation: bool) -> Any:
    """Lease → sealed source → quota → unseal → relay (one usage row)."""
    prep = _prepare(request, workstation=workstation)
    if prep.error is not None:
        return prep.error
    lease = prep.lease

    refusal = _rate_or_quota_refusal(request, lease)
    if refusal is not None:
        return refusal

    try:
        api_key = _unseal_api_key(request, prep.full)
    except SealError:
        return _denied(request, lease, 500, "unseal_failed")

    headers = _upstream_headers(
        request, prep.policy, api_key, prep.full.allow_host, workstation
    )
    sent = _Sent()
    try:
        upstream = await _send_upstream(request, prep, headers, sent)
    except httpx.HTTPError:
        return _denied(request, lease, 502, "upstream_unreachable")

    if _check_redirect(request, upstream, prep.host, headers):
        await upstream.aclose()
        return _denied(request, lease, 502, "redirect_offhost")

    meta = _RelayMeta(
        source_label=str(prep.full.id),
        provider_kind=str(prep.full.provider_kind or ""),
        started=prep.started,
        sent=sent,
    )
    return _relayed(request, lease, upstream, meta)


@dataclass(frozen=True)
class _RelayMeta:
    """What the access line and the usage row say about a relayed call."""

    source_label: str
    provider_kind: str
    started: float
    model: str = ""
    served_model: str = ""
    sent: _Sent = field(default_factory=_Sent)


def _is_streaming(request: Request, upstream: Any) -> bool:
    content_type = upstream.headers.get("content-type", "")
    path = str(request.path_params.get("path", "")).lower()
    return "event-stream" in content_type or "stream" in path


def _relayed(
    request: Request, lease: Any, upstream: Any, meta: _RelayMeta
) -> StreamingResponse:
    """Stream the upstream answer back; one access line, one audit row and
    one usage row (ADR-099 D-4) once it was sent."""
    response_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "x-accel-buffering"
    }
    elapsed_ms = int((time.monotonic() - meta.started) * 1000)
    tap = UsageTap(
        provider_kind=meta.provider_kind, streaming=_is_streaming(request, upstream)
    )
    once = threading.Lock()

    async def _body():
        try:
            async for chunk in upstream.aiter_raw():
                tap.feed(chunk)
                yield chunk
            tap.ended = True
        finally:
            await upstream.aclose()

    def _finish() -> None:
        """The access line, audit row and usage row: once per call."""
        if once.acquire(blocking=False):
            _audit_done()

    def _abort() -> None:
        if not once.locked():  # the background task did not run
            _record_off_loop(_finish)

    def _audit_done() -> None:
        # Access log contract §3.3.4 via logger.warning (uvicorn default
        # surfaces it): ts sid source_id method path status ms reason.
        logger.warning(
            "infer_access ts=%.3f sid=%s source_id=%s method=%s path=%s status=%s ms=%d reason=ok",
            time.time(),
            request.path_params["sid"],
            meta.source_label,
            request.method,
            request.url.path,
            upstream.status_code,
            elapsed_ms,
        )
        _audit_broker(
            request,
            lease.lease_id,
            "infer",
            f"{request.method} {request.url.path} {upstream.status_code} ms={elapsed_ms}",
        )
        _record_usage(request, lease, upstream.status_code, tap, meta)

    return _RelayedResponse(
        _body(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(_finish),
        on_abort=_abort,
    )


class _RelayedResponse(StreamingResponse):
    """A relayed stream whose rows are written even when it never completes.

    Starlette runs the background task only after the last byte was sent: a
    client that disconnects mid-stream (or a cancelled relay) would skip the
    usage row and dodge the token cap (ADR-099 D-4). ``on_abort`` runs when
    the response ends without its background task having run."""

    def __init__(self, *args: Any, on_abort: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_abort = on_abort

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._on_abort()


def _record_off_loop(fn: Any) -> None:
    """Run ``fn`` in a worker thread without awaiting it: called from a
    ``finally`` that may be cancelled (a client disconnect), where an
    ``await`` would be cancelled again and the usage row lost."""
    try:
        asyncio.get_running_loop().run_in_executor(None, fn)
    except RuntimeError:  # no running loop
        fn()


def _record_usage(
    request: Request, lease: Any, status: int, tap: UsageTap, meta: _RelayMeta
) -> None:
    """One ``llm_usage_log`` row for the relayed call (never raises)."""
    try:
        model = meta.model or tap.model()
        usage, estimated = tap.metered(
            request_bytes=meta.sent.size,
            generation=request.method not in _UNMETERED_METHODS,
            prompt_tokens=meta.sent.meter.tokens(),
        )
        record_proxy_usage(
            getattr(request.app.state, "umrs_db_pool", None),
            lease=lease,
            session_id=request.path_params["sid"],
            model=model,
            served_model=meta.served_model or tap.model() or model,
            source_id=meta.source_label,
            path=request.url.path,
            status=status,
            streaming=tap.streaming,
            duration_ms=(time.monotonic() - meta.started) * 1000.0,
            usage=usage,
            estimated=estimated,
        )
    except Exception:  # accounting never breaks the relay
        logger.warning("infer usage row not written", exc_info=True)


# ── the site source: EndpointRegistry rows (ADR-092 slice 8b, D-63) ───────


async def _proxy_site(request: Request) -> Any:
    """``/s/{sid}/site/...``: the lease's own project's registry rows."""
    token = extract_session_token(request.headers)
    if not token:
        return _json_error(401, "token_invalid")
    host, refusal = _request_host(request)
    if refusal is not None:
        return refusal
    pool = getattr(request.app.state, "umrs_db_pool", None)
    verdict = check_route_lease(
        lease_by_token(pool, token),
        workstation=False,
        sid=request.path_params["sid"],
        client_ip=_client_ip(request),
        source_override=SITE_SOURCE_ID,
        host=host,
    )
    if verdict.deny():
        status = _verbatim_error_status(verdict.reason)
        return _denied(request, verdict.lease, status, verdict.reason)
    refusal = _rate_or_quota_refusal(request, verdict.lease)
    if refusal is not None:
        return refusal
    path = "/" + request.path_params["path"][len(SITE_PATH_PREFIX) :]
    try:
        return await _site_forward(request, pool, verdict.lease, path)
    except site_models.SiteSourceRefused as exc:
        return _denied(request, verdict.lease, exc.status, exc.reason)


def _site_registry(request: Request) -> Any:
    registry = getattr(request.app.state, "site_registry", None)
    return registry if registry is not None else site_models.default_registry()


async def _site_forward(request: Request, pool: Any, lease: Any, path: str) -> Any:
    registry = _site_registry(request)
    project = str(getattr(lease, "project", "") or "")
    if request.method == "GET" and path == site_models.SITE_MODELS_PATH:
        return JSONResponse(site_models.models_listing(registry, project))
    if request.method != "POST" or path not in site_models.SITE_PATHS:
        raise site_models.SiteSourceRefused(403, "path_denied")
    started = time.monotonic()
    payload, model = site_models.request_model(await request.body())
    endpoint = site_models.resolve(registry, project, model)
    _check_site_data_class(pool, project, endpoint)
    payload["model"] = site_models.served_name(endpoint)
    with_stream_usage(payload)  # a vLLM stream reports usage only when asked
    upstream = await _site_send(request, lease, endpoint, path, payload)
    meta = _RelayMeta(
        source_label=f"site:{endpoint.job_id}",
        provider_kind="openai",
        started=started,
        model=str(model or ""),
        served_model=str(payload["model"] or ""),
        sent=_sent_body(json.dumps(payload).encode()),
    )
    return _relayed(request, lease, upstream, meta)


def _check_site_data_class(pool: Any, project: str, endpoint: Any) -> None:
    """ADR-062: a sensitive project never reaches a public row."""
    if not site_models.is_public_row(endpoint):
        return
    if data_class_for_project(pool, project) == "sensitive":
        raise site_models.SiteSourceRefused(403, "source_class_refused")


async def _site_send(
    request: Request, lease: Any, endpoint: Any, path: str, payload: dict[str, Any]
) -> Any:
    """POST to the row's upstream (streamed); no redirect is ever followed."""
    client = _http_client(request)
    try:
        upstream = await client.send(
            client.build_request(
                "POST",
                site_models.upstream_url(endpoint, path),
                headers=_site_headers(request, lease, endpoint),
                content=json.dumps(payload).encode(),
            ),
            stream=True,
            follow_redirects=False,
        )
    except httpx.HTTPError:
        raise site_models.SiteSourceRefused(502, "upstream_unreachable") from None
    if 300 <= upstream.status_code < 400:
        await upstream.aclose()
        raise site_models.SiteSourceRefused(502, "redirect_offhost")
    return upstream


def _site_headers(request: Request, lease: Any, endpoint: Any) -> dict[str, str]:
    """Inbound credentials stripped; the site's own service key injected."""
    headers = {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in request.headers.raw
        if name.decode("latin-1").lower() not in _HOP_BY_HOP
    }
    headers = strip_credential_headers(headers)
    headers.pop("x-project-id", None)
    headers.pop("x-user-id", None)
    headers["content-type"] = "application/json"
    headers["accept-encoding"] = "identity"  # the usage tap reads plain bytes
    headers["host"] = f"{endpoint.host}:{int(endpoint.port)}"
    key = site_models.service_key()
    if key:
        headers["authorization"] = f"Bearer {key}"
    headers["x-project-id"] = str(lease.project or "")
    headers["x-user-id"] = str(lease.owner_id or "")
    return headers


_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def create_app(
    *,
    umrs_db_pool: Any,
    unseal_privkey: str,
    sealed_store: Any | None = None,
    broker: Any | None = None,
    http_client: httpx.AsyncClient | None = None,
    site_registry: Any | None = None,
    quota_enforcer: QuotaEnforcer | None = None,
) -> Starlette:
    """App factory (tests inject fakes; production wires the real pool/key).

    ``quota_enforcer`` None = no quota checks (ADR-099 D-5 is opt-in here;
    ``__main__`` wires it)."""
    from umrs.jobs.secret_broker import SecretBroker

    app = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/s/{sid}/{path:path}", proxy_request, methods=_METHODS),
            Route("/w/{sid}/{path:path}", proxy_workstation, methods=_METHODS),
        ]
    )
    app.state.umrs_db_pool = umrs_db_pool
    app.state.unseal_privkey = unseal_privkey
    app.state.sealed_store = sealed_store or SealedSourceStore(umrs_db_pool)
    app.state.broker = broker or SecretBroker(db_pool=umrs_db_pool)
    # ADR-092 slice 8b: the site source's registry (None = the process one).
    app.state.site_registry = site_registry
    app.state.quota_enforcer = quota_enforcer
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
    # Logging discipline: one stdout handler if none exists, redact filters on all.
    proxy_logger = logging.getLogger("umrs.infer_proxy")
    if not proxy_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s infer_access %(message)s")
        )
        proxy_logger.addHandler(stream_handler)
    proxy_logger.setLevel(os.environ.get("UMRS_INFER_ACCESS_LEVEL", "info").upper())
    proxy_logger.propagate = True
    for handler in proxy_logger.handlers:
        handler.addFilter(SecretRedactingFilter())
    return app
