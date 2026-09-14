"""Server-side validation for pushed provider base URLs (R0-3).

The client-side check is untrusted: this module is the authority for
§3.1.3 of the remote-session credential design. A pushed ``base_url``
must be an HTTPS endpoint that cannot point at infrastructure
(SSRF-guard layer one; the inference proxy re-resolves per request as
layer two).

Public surface:

    validate_provider_base_url(url, *, trusted_http_hosts=()) -> ProviderUrl

Raises ``ProviderUrlError`` with a stable ``reason`` code, or returns a
normalized ``ProviderUrl`` with ``base_url`` and ``allow_host``.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

# Query keys that look like credentials carried in the URL itself.
FORBIDDEN_QUERY_KEYS = frozenset(
    {"key", "api_key", "api-key", "apikey", "token", "access_token", "sig", "signature"}
)

# Hostnames that must never be a provider base for a pushed credential.
FORBIDDEN_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".infra")
FORBIDDEN_HOST_EXACT = frozenset({"localhost", "host.containers.internal"})

MAX_URL_LENGTH = 2048


class ProviderUrlError(ValueError):
    """Validation failure with a stable reason code (never a free-form leak)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ProviderUrl:
    """Validated, normalized provider endpoint."""

    base_url: str  # e.g. "https://api.openai.com/v1"
    allow_host: str  # e.g. "https://api.openai.com" or "https://host:8443"


def _is_global_unicast(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    return not (isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None)


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _reject_numeric_tricks(host: str) -> None:
    """Reject decimal/hex IP literals that urlsplit accepts as DNS names."""
    if host.startswith(("0x", "0X")):
        raise ProviderUrlError("host_numeric_literal")
    segments = host.split(".")
    if len(segments) > 1 and all(seg.isdigit() and seg != "" for seg in segments):
        raise ProviderUrlError("host_numeric_literal")
    if host.isdigit() and host != "":
        raise ProviderUrlError("host_numeric_literal")


def _check_dns_name(hostname: str) -> None:
    """infra suffixes; every resolved address must be global unicast."""
    if hostname in FORBIDDEN_HOST_EXACT or hostname.endswith(FORBIDDEN_HOST_SUFFIXES):
        raise ProviderUrlError("host_infrastructure")
    try:
        infos = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)
    except socket.gaierror:
        raise ProviderUrlError("host_unresolvable") from None
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise ProviderUrlError("host_unresolvable")
    for addr in addresses:
        if not _is_global_unicast(addr):
            raise ProviderUrlError("address_not_global")


def _check_shape(parts, text: str) -> None:
    """URL length/credentials-in-URL/fragment/query rules (§3.1.3 rules 0–2)."""
    if not text:
        raise ProviderUrlError("url_empty")
    if len(text) > MAX_URL_LENGTH:
        raise ProviderUrlError("url_too_long")
    if parts.username or parts.password:
        raise ProviderUrlError("userinfo_forbidden")
    if parts.fragment:
        raise ProviderUrlError("fragment_forbidden")
    query_keys = {k.lower() for k, _ in parse_qsl(parts.query, keep_blank_values=True)}
    if query_keys & FORBIDDEN_QUERY_KEYS:
        raise ProviderUrlError("credential_query_key")


def _check_scheme(
    scheme: str, hostname: str, trusted_http_hosts: tuple[str, ...]
) -> None:
    """HTTPS required unless the host is an operator-trusted on-prem (rule 1)."""
    trusted = hostname in {h.lower().rstrip(".") for h in trusted_http_hosts}
    if scheme != "https" and not (scheme == "http" and trusted):
        raise ProviderUrlError("scheme_not_https")


def _check_ip_literal(parsed, hostname: str) -> None:
    """Literal IP must be global unicast incl. the IPv4-mapped v6 case (rule 4)."""
    if not _is_global_unicast(hostname):
        raise ProviderUrlError("address_not_global")
    mapped = parsed.ipv4_mapped if isinstance(parsed, ipaddress.IPv6Address) else None
    if mapped is not None and not _is_global_unicast(str(mapped)):
        raise ProviderUrlError("address_not_global")


def _normalize(parts, scheme: str, hostname: str, parsed) -> ProviderUrl:
    """Lowercased host, trailing-/ stripped path, bracketed v6 (rules 6–7)."""
    host_str = (
        f"[{hostname}]" if isinstance(parsed, ipaddress.IPv6Address) else hostname
    )
    allow_host = f"{scheme}://{host_str}" + (f":{parts.port}" if parts.port else "")
    base_url = allow_host + parts.path.rstrip("/")
    if parts.query:
        base_url += f"?{parts.query}"
    return ProviderUrl(base_url=base_url, allow_host=allow_host)


def validate_provider_base_url(
    url: str,
    *,
    trusted_http_hosts: tuple[str, ...] = (),
) -> ProviderUrl:
    """Validate + normalize a pushed provider base URL.

    Rules (design §3.1.3): https only (unless the host is an operator-
    trusted on-prem one), no userinfo/fragment/credential-looking query,
    hostname is not infrastructure, every resolved address is global
    unicast, numeric-literal tricks rejected.
    """
    text = (url or "").strip()
    parts = urlsplit(text)
    _check_shape(parts, text)

    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ProviderUrlError("host_missing")
    parsed = _parse_ip(hostname)
    if parsed is None:
        _reject_numeric_tricks(hostname)

    _check_scheme(scheme, hostname, trusted_http_hosts)
    if parsed is not None:
        _check_ip_literal(parsed, hostname)
    else:
        _check_dns_name(hostname)

    return _normalize(parts, scheme, hostname, parsed)
