"""Egress fetch + content scanning for KNOWLEDGE_INGEST remote sources.

Supported kinds: ``http_url``, ``git_repo``.

Policy:
  - Fail-closed: empty ``allow_domains`` blocks all remote fetches.
  - Domain allowlist enforced on every hop, redirects included.
  - Private, loopback, link-local and metadata addresses refused (K1.1).
  - Content scanned against PII/credential patterns before storage.

Stdlib-only: ``http.client`` over a socket pinned to the checked address.

Environment variables:
  EGRESS_TIMEOUT_SEC  – per-connect / per-read timeout, default 30
  EGRESS_TOTAL_SEC    – whole fetch (every hop and the body), default 120
  EGRESS_MAX_BYTES    – max content size, default 10MB (10 * 1024 * 1024)
"""

from __future__ import annotations

import os
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_EGRESS_TIMEOUT_SEC: int = int(os.environ.get("EGRESS_TIMEOUT_SEC", "30"))

# K1.1 review (2026-10-24): a socket timeout bounds each read, not the fetch;
# a server trickling one byte per read held a worker for as long as it liked.
_EGRESS_TOTAL_SEC: int = int(os.environ.get("EGRESS_TOTAL_SEC", "120"))

_EGRESS_MAX_BYTES: int = int(os.environ.get("EGRESS_MAX_BYTES", str(10 * 1024 * 1024)))

# Default block patterns (PII, credentials, secrets).
_DEFAULT_BLOCK_PATTERNS: list[str] = [
    r"(?i)(password|passwd|pwd)\s*[:=]",
    r"(?i)(secret|secret_key)\s*[:=]",
    r"(?i)(api_key|apikey)\s*[:=]",
    r"(?i)(token|auth_token|access_token)\s*[:=]",
    r"(?i)(private_key)\s*[:=]",
    r"(?i)(aws_secret_access_key)\s*[:=]",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result of content scanning."""

    passed: bool
    blocked_reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class FetchResult:
    """Result of an egress fetch."""

    url: str
    content: bytes | None
    status_code: int | None
    size_bytes: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Content scanning
# ---------------------------------------------------------------------------


def scan_content(
    content: bytes,
    block_patterns: list[str] | None = None,
) -> ScanResult:
    """Scan ``content`` for PII / credential patterns.

    Returns ``ScanResult(passed=True)`` when clean, or
    ``ScanResult(passed=False, blocked_reason=...)`` when blocked.

    Patterns are matched against the content decoded as UTF-8 (with fallback
    to latin-1 so that bytes-based scans never crash).
    """
    patterns = list(block_patterns) if block_patterns else list(_DEFAULT_BLOCK_PATTERNS)
    matched: list[str] = []

    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    # Limit scan to first 500 KB to avoid O(n) cost on large payloads.
    scan_window = text[:500_000]

    for pat in patterns:
        try:
            if re.search(pat, scan_window):
                matched.append(pat)
        except re.error:
            continue

    if matched:
        pattern_summary = "; ".join(matched[:3])
        return ScanResult(
            passed=False,
            blocked_reason=f"content blocked: {len(matched)} pattern(s) matched — {pattern_summary}",
            matched_patterns=matched,
        )

    # Additional heuristic: check for email-like addresses in non-text content.
    if len(matched) == 0 and len(content) < 100_000:
        emails = re.findall(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", scan_window
        )
        # If more than 5 unique emails in a short window, flag it.
        unique_emails = set(emails)
        if len(unique_emails) > 5:
            return ScanResult(
                passed=False,
                blocked_reason=f"content blocked: {len(unique_emails)} unique email addresses detected",
                matched_patterns=["email_density"],
            )

    return ScanResult(passed=True)


# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------
#
# K1.1(d) (knowledge-sharing plan, 2026-10-24). Before this, ``fetch_url``
# handed the URL to ``urllib.request.urlopen``: no private-address block,
# redirects followed after one domain check on the FIRST URL only, any
# scheme urllib speaks, and the whole body read before the size cap. Any
# project member could reach internal addresses through a KNOWLEDGE_INGEST
# job. Now every hop (the first URL and each redirect target) must:
#   * be http(s) to a host the allowlist names (``*.suffix`` patterns allowed);
#   * resolve only to global unicast addresses (no private, loopback,
#     link-local, multicast, reserved, unspecified or cloud-metadata address);
# and the connection goes to the address that was checked (pinned, so a DNS
# answer cannot change between the check and the connect). The body is
# streamed and refused once it passes the cap. The fetch is direct: an
# HTTP(S)_PROXY in the environment is not used, because the address check
# means nothing once a proxy resolves the name (ADR-015 amendment
# 2026-10-24).
#
# Review of K1.1 (2026-10-24) added: one deadline for the whole fetch
# (EGRESS_TOTAL_SEC); a body shorter than its Content-Length is refused, not
# stored; an https hop never redirects down to http; and only ports 80/443
# are fetched unless the allowlist entry names the port (``host:8443``).

_MAX_REDIRECTS: int = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_CHUNK_BYTES = 64 * 1024
_ALLOWED_SCHEMES = ("http", "https")
_DEFAULT_PORTS = frozenset({80, 443})
# Named for the refusal message; each is also non-global, so the generic
# check would refuse it anyway.
_METADATA_ADDRESSES = frozenset(
    {"169.254.169.254", "169.254.170.2", "100.100.100.200", "fd00:ec2::254"}
)


class EgressRefused(Exception):
    """A hop the egress policy refuses (named reason, never fetched)."""


class _Budget:
    """The per-read timeout and the one deadline the whole fetch shares."""

    def __init__(self, timeout: float, total: float) -> None:
        self.timeout = float(timeout)
        self.total = float(total)
        self._end = time.monotonic() + self.total

    def left(self) -> float:
        """Seconds the next connect or read may wait; refuses once spent."""
        remaining = self._end - time.monotonic()
        if remaining <= 0:
            raise EgressRefused(f"fetch took longer than {self.total:g}s in total")
        return min(self.timeout, remaining)


def _split_port(pattern: str) -> tuple[str, int | None]:
    """``host:port`` → (host, port); anything else (IPv6 literals included,
    which hold more than one colon) is a host pattern with no port."""
    host, sep, port = pattern.rpartition(":")
    if sep and port.isdigit() and ":" not in host:
        return host, int(port)
    return pattern, None


def _host_matches(name: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        return name.endswith(pattern[1:])
    return bool(pattern) and name == pattern


def _normal(text: str) -> str:
    return str(text).strip().lower().rstrip(".")


def domain_allowed(host: str, patterns: list[str] | tuple[str, ...]) -> bool:
    """True when *host* equals a pattern or sits under a ``*.suffix`` one.

    Case and one trailing dot are ignored. ``*.example.org`` admits
    ``a.example.org`` (and deeper) but not ``example.org`` itself, as in
    ADR-015's allowlist. A ``:port`` on a pattern is ignored here
    (:func:`port_allowed` reads it).
    """
    name = _normal(host)
    if not name:
        return False
    return any(_host_matches(name, _split_port(_normal(raw))[0]) for raw in patterns)


def port_allowed(host: str, port: int, patterns: list[str] | tuple[str, ...]) -> bool:
    """80 and 443 for any allowed host; another port only when an allowlist
    entry matching *host* names it (``docs.example.org:8443``)."""
    if port in _DEFAULT_PORTS:
        return True
    name = _normal(host)
    for raw in patterns:
        pattern, named = _split_port(_normal(raw))
        if named == port and _host_matches(name, pattern):
            return True
    return False


def intersect_allowlists(
    caller: list[str] | None, site: list[str] | tuple[str, ...]
) -> list[str]:
    """The caller's list narrowed to what the site list admits.

    ``None`` (the caller gave no list) means the site list. A caller entry
    survives only when the site list admits it, so a caller can narrow the
    site allowlist but never widen or replace it (ADR-015, K1.1).
    """
    site_list = [str(d) for d in site if str(d).strip()]
    if caller is None:
        return site_list
    return [d for d in caller if str(d) in site_list or domain_allowed(str(d), site_list)]


def address_refusal(address: str) -> str:
    """Why *address* may not be fetched, or ``""`` when it is global unicast."""
    import ipaddress

    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return f"unparseable address {address!r}"
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if str(ip) in _METADATA_ADDRESSES:
        return f"cloud metadata address {ip}"
    for label in ("loopback", "link_local", "private", "multicast", "unspecified", "reserved"):
        if getattr(ip, f"is_{label}"):
            return f"{label.replace('_', '-')} address {ip}"
    if not ip.is_global:
        return f"non-global address {ip}"
    return ""


def _resolve_host(host: str, port: int) -> list[str]:
    """Every address *host* resolves to (tests replace this; no network)."""
    import socket

    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def _port(parsed: Any) -> int:
    """The URL's port, or its scheme's default."""
    if parsed.port:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else 80


def _addresses_for(host: str, port: int) -> list[str]:
    """An IP literal as itself; a name through the resolver (named failure)."""
    import ipaddress

    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    try:
        addresses = _resolve_host(host, port)
    except OSError as exc:
        raise EgressRefused(f"cannot resolve {host!r}: {exc}") from exc
    if not addresses:
        raise EgressRefused(f"{host!r} resolved to no address")
    return addresses


def _check_hop(url: str, domains: list[str]) -> tuple[Any, str]:
    """Validate one hop; return ``(parsed_url, pinned_address)`` or raise.

    Every address the name resolves to must be global: one private answer
    among public ones refuses the hop (a rebinding name cannot slip one in).
    The port must be 80/443 or one the matching allowlist entry names.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise EgressRefused(f"scheme {parsed.scheme!r} not allowed (http/https only)")
    host = parsed.hostname or ""
    if not host:
        raise EgressRefused("invalid URL: no domain")
    if not domain_allowed(host, domains):
        raise EgressRefused(f"domain {host!r} not in allowlist {domains}")
    addresses = _addresses_for(host, _port(parsed))
    for address in addresses:
        reason = address_refusal(address)
        if reason:
            raise EgressRefused(f"{host!r} resolves to a {reason}")
    port = _port(parsed)
    if not port_allowed(host, port, domains):
        raise EgressRefused(
            f"port {port} of {host!r} not allowed (80/443 only, unless the "
            f"allowlist names it: '{host}:{port}')"
        )
    return parsed, addresses[0]


def _open_connection(
    scheme: str, host: str, port: int, address: str, timeout: float
) -> Any:
    """An HTTP(S) connection to the CHECKED address (TLS still names *host*)."""
    import http.client
    import socket

    tls = ssl.create_default_context()

    class _PinnedHTTP(http.client.HTTPConnection):
        def connect(self) -> None:
            self.sock = socket.create_connection((address, self.port), self.timeout)

    class _PinnedHTTPS(http.client.HTTPSConnection):
        def connect(self) -> None:
            raw = socket.create_connection((address, self.port), self.timeout)
            self.sock = tls.wrap_socket(raw, server_hostname=self.host)

    if scheme == "https":
        return _PinnedHTTPS(host, port, timeout=timeout, context=tls)
    return _PinnedHTTP(host, port, timeout=timeout)


def _request_hop(url: str, domains: list[str], budget: _Budget) -> Any:
    """Send one GET after the hop check; return the open response (its
    socket kept as ``egress_sock``, so each body read gets the budget)."""
    parsed, address = _check_hop(url, domains)
    conn = _open_connection(
        parsed.scheme, parsed.hostname, _port(parsed), address, budget.left()
    )
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    conn.request("GET", path, headers={"User-Agent": "UMRS-KnowledgeIngest/1.0"})
    sock = conn.sock
    resp = conn.getresponse()
    resp.egress_sock = sock
    return resp


def _next_hop(url: str, resp: Any) -> str:
    """The redirect target; an https hop never goes down to http."""
    from urllib.parse import urljoin, urlparse

    location = resp.getheader("Location") or ""
    if not location:
        raise EgressRefused(f"redirect {resp.status} without a Location")
    target = urljoin(url, location)
    if urlparse(url).scheme == "https" and urlparse(target).scheme != "https":
        raise EgressRefused(f"redirect from https down to {target!r} refused")
    return target


def _follow(url: str, domains: list[str], budget: _Budget) -> Any:
    """GET *url*, re-checking domain and address on EVERY redirect hop."""
    for _hop in range(_MAX_REDIRECTS + 1):
        resp = _request_hop(url, domains, budget)
        if resp.status not in _REDIRECT_CODES:
            return resp
        resp.close()
        url = _next_hop(url, resp)
    raise EgressRefused(f"more than {_MAX_REDIRECTS} redirects")


def _declared_length(resp: Any, cap: int) -> int | None:
    """The Content-Length, when given; refused up front over *cap*."""
    declared = str(resp.getheader("Content-Length") or "").strip()
    if not declared.isdigit():
        return None
    if int(declared) > cap:
        raise EgressRefused(f"content too large: {declared} bytes (max {cap})")
    return int(declared)


def _arm(resp: Any, budget: _Budget) -> None:
    """Bound the next socket read by what the budget has left."""
    sock = getattr(resp, "egress_sock", None)
    if sock is not None:
        sock.settimeout(budget.left())
    else:
        budget.left()


def _read_chunk(resp: Any, budget: _Budget) -> bytes:
    """One read bounded by the budget; a read that ran the deadline out is
    refused by name, not reported as a bare socket timeout.

    A closed response has nothing left: since Python 3.11 http.client closes
    the socket once the body is read, and arming it then raises EBADF (CI
    2026-10-24, run 40094838994; the 3.10 dev venv never saw it)."""
    if resp.isclosed():
        return b""
    _arm(resp, budget)
    try:
        return resp.read1(_CHUNK_BYTES)
    except TimeoutError:
        budget.left()
        raise


def _read_capped(resp: Any, cap: int, budget: _Budget) -> bytes:
    """Stream the body under the cap and the deadline; a body shorter than
    its Content-Length is refused (a dropped connection is not a document)."""
    declared = _declared_length(resp, cap)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = _read_chunk(resp, budget)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise EgressRefused(f"content too large: over {cap} bytes (max {cap})")
        chunks.append(chunk)
    if declared is not None and total != declared:
        raise EgressRefused(f"truncated body: {total} of {declared} bytes")
    return b"".join(chunks)


def _download(url: str, domains: list[str], budget: _Budget) -> FetchResult:
    """Follow, check the status, read under the cap (raises EgressRefused)."""
    resp = _follow(url, domains, budget)
    try:
        if resp.status >= 400:
            return FetchResult(
                url=url,
                content=None,
                status_code=resp.status,
                error=f"HTTP {resp.status}: {resp.reason}",
            )
        raw = _read_capped(resp, _EGRESS_MAX_BYTES, budget)
    finally:
        resp.close()
    return FetchResult(url=url, content=raw, status_code=resp.status, size_bytes=len(raw))


def fetch_url(
    url: str,
    allow_domains: list[str] | None = None,
    timeout: int | None = None,
    block_patterns: list[str] | None = None,
) -> FetchResult:
    """Fetch content from ``url`` under the egress policy, then scan it.

    Parameters:
        url: The URL to fetch.
        allow_domains: Domain allowlist (fail-closed when empty/None); the
            caller passes the site list already narrowed by the requester's.
        timeout: Per-connect / per-read timeout in seconds (default from
            env); the whole fetch is bounded by ``EGRESS_TOTAL_SEC``.
        block_patterns: Content scan patterns (default PII/credential set).

    Returns:
        FetchResult with content, or ``content=None`` and a named error.
    """
    domains = [str(d) for d in allow_domains] if allow_domains else []
    if not domains:
        return FetchResult(
            url=url,
            content=None,
            status_code=None,
            error="egress blocked: allow_domains is empty (fail-closed)",
        )
    try:
        budget = _Budget(timeout or _EGRESS_TIMEOUT_SEC, _EGRESS_TOTAL_SEC)
        fetched = _download(url, domains, budget)
    except EgressRefused as exc:
        return FetchResult(url=url, content=None, status_code=None, error=f"egress blocked: {exc}")
    except Exception as exc:  # network faults come back by name
        return FetchResult(url=url, content=None, status_code=None, error=f"fetch failed: {exc}")
    if fetched.content is None:
        return fetched
    scan = scan_content(fetched.content, block_patterns=block_patterns)
    if not scan.passed:
        return FetchResult(
            url=url,
            content=None,
            status_code=fetched.status_code,
            size_bytes=fetched.size_bytes,
            error=f"content blocked: {scan.blocked_reason}",
        )
    return fetched


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
