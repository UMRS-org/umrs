"""Egress fetch + content scanning for KNOWLEDGE_INGEST remote sources.

Supported kinds: ``http_url``, ``git_repo``.

Policy:
  - Fail-closed: empty ``allow_domains`` blocks all remote fetches.
  - Domain allowlist enforced on every hop, redirects included.
  - Private, loopback, link-local and metadata addresses refused (K1.1).
  - Content scanned against PII/credential patterns before storage.

Stdlib-only: ``http.client`` over a socket pinned to the checked address.

Environment variables:
  EGRESS_TIMEOUT_SEC  – per-request timeout, default 30
  EGRESS_MAX_BYTES    – max content size, default 10MB (10 * 1024 * 1024)
"""

from __future__ import annotations

import os
import re
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_EGRESS_TIMEOUT_SEC: int = int(os.environ.get("EGRESS_TIMEOUT_SEC", "30"))

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
# means nothing once a proxy resolves the name.

_MAX_REDIRECTS: int = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_CHUNK_BYTES = 64 * 1024
_ALLOWED_SCHEMES = ("http", "https")
# Named for the refusal message; each is also non-global, so the generic
# check would refuse it anyway.
_METADATA_ADDRESSES = frozenset(
    {"169.254.169.254", "169.254.170.2", "100.100.100.200", "fd00:ec2::254"}
)


class EgressRefused(Exception):
    """A hop the egress policy refuses (named reason, never fetched)."""


def domain_allowed(host: str, patterns: list[str] | tuple[str, ...]) -> bool:
    """True when *host* equals a pattern or sits under a ``*.suffix`` one.

    Case and one trailing dot are ignored. ``*.example.org`` admits
    ``a.example.org`` (and deeper) but not ``example.org`` itself, as in
    ADR-015's allowlist.
    """
    name = host.strip().lower().rstrip(".")
    if not name:
        return False
    for raw in patterns:
        pattern = str(raw).strip().lower().rstrip(".")
        if pattern.startswith("*."):
            if name.endswith(pattern[1:]):
                return True
        elif pattern and name == pattern:
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
    return parsed, addresses[0]


def _open_connection(
    scheme: str, host: str, port: int, address: str, timeout: int
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


def _request_hop(url: str, domains: list[str], timeout: int) -> Any:
    """Send one GET after the hop check; return the open response."""
    parsed, address = _check_hop(url, domains)
    conn = _open_connection(parsed.scheme, parsed.hostname, _port(parsed), address, timeout)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    conn.request("GET", path, headers={"User-Agent": "UMRS-KnowledgeIngest/1.0"})
    return conn.getresponse()


def _follow(url: str, domains: list[str], timeout: int) -> Any:
    """GET *url*, re-checking domain and address on EVERY redirect hop."""
    from urllib.parse import urljoin

    for _hop in range(_MAX_REDIRECTS + 1):
        resp = _request_hop(url, domains, timeout)
        if resp.status not in _REDIRECT_CODES:
            return resp
        location = resp.getheader("Location") or ""
        resp.close()
        if not location:
            raise EgressRefused(f"redirect {resp.status} without a Location")
        url = urljoin(url, location)
    raise EgressRefused(f"more than {_MAX_REDIRECTS} redirects")


def _read_capped(resp: Any, cap: int) -> bytes:
    """Stream the body; refuse as soon as it passes *cap* bytes."""
    declared = resp.getheader("Content-Length")
    if declared and declared.isdigit() and int(declared) > cap:
        raise EgressRefused(f"content too large: {declared} bytes (max {cap})")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = resp.read(_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > cap:
            raise EgressRefused(f"content too large: over {cap} bytes (max {cap})")
        chunks.append(chunk)


def _download(url: str, domains: list[str], timeout: int) -> FetchResult:
    """Follow, check the status, read under the cap (raises EgressRefused)."""
    resp = _follow(url, domains, timeout)
    try:
        if resp.status >= 400:
            return FetchResult(
                url=url,
                content=None,
                status_code=resp.status,
                error=f"HTTP {resp.status}: {resp.reason}",
            )
        raw = _read_capped(resp, _EGRESS_MAX_BYTES)
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
        timeout: Request timeout in seconds (default from env).
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
        fetched = _download(url, domains, timeout or _EGRESS_TIMEOUT_SEC)
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
