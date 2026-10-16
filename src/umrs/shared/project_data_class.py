"""ADR-062 model-locality seam: the per-project data_class lookup + the
public-provider kind vocabulary (D-51).

Both enforcement points share this module so ONE definition of
"public kind" and ONE never-invented lookup serve every seam:

* the venue-start gate (``venues/frontend_service.py``),
* the inference-proxy lease gate (``infer_proxy/app.py``).

The lookup never blocks its caller: pool-less, absent row, or a lookup
failure all answer ``""`` (no gate) — a class is never invented.
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import os
import re
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# ADR-062 model locality: every user-pushable provider kind is PUBLIC; a
# sensitive project's sessions/leases must not reach any of them
# (self-deployed models need no pushed source).
PUBLIC_PROVIDER_KINDS = frozenset(
    {"openai", "openai_compat", "anthropic", "google", "azure_openai"}
)

# ADR-062 amendment (2026-10-15): the third provider class, INSTITUTIONAL —
# an inference service run by the user's own institution outside UMRS
# (Rikyu first). A sensitive project may use it; commercial and other
# public providers stay refused. Membership comes from this allowlist the
# SITE keeps, never from a user label: UMRS_INSTITUTIONAL_PROVIDER_HOSTS
# (comma/space separated) replaces the packaged default; empty = none.
# Every process that runs a gate must see the same value — on umrs-front
# the deploy role passes the one .env line to the servers and umrs-infer
# (tests/test_institutional_allowlist_parity.py).
DEFAULT_INSTITUTIONAL_PROVIDER_HOSTS = frozenset({"api.rikyu.r-ccs.riken.jp"})

_HTTP_SCHEMES = ("http", "https")
_LABEL = re.compile(r"^[a-z0-9_]([a-z0-9_-]*[a-z0-9_])?$")


def _canonical_host(host: str) -> str:
    """Lowercased host, one trailing dot dropped, IPs in canonical form.

    ``""`` when *host* is not a DNS name or an IP literal (wildcards,
    empty labels, non-ASCII): such a host never matches the allowlist.
    """
    text = host.strip().lower().strip("[]").removesuffix(".")
    if not text:
        return ""
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        pass
    if len(text) > 253 or not text.isascii():
        return ""
    labels = text.split(".")
    if not all(len(label) <= 63 and _LABEL.match(label) for label in labels):
        return ""
    return text


def _split_http_url(text: str) -> Any:
    """``urlsplit`` of an http(s) URL (a bare ``host[:port]`` gets https).

    ``None`` for another scheme, userinfo, an unparseable or non-numeric
    port — the shapes on which two URL parsers can disagree on the host.
    """
    if "://" not in text:
        text = "https://" + text
    try:
        parts = urlsplit(text)
        parts.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        return None
    if parts.scheme.lower() not in _HTTP_SCHEMES or "@" in parts.netloc:
        return None
    return parts


def _allowlist_entry_host(entry: str) -> str:
    """One allowlist entry → its bare host, or ``""`` when unusable.

    Operators write hosts, base URLs or ``host:port``: the scheme, port,
    path, trailing dot and IPv6 brackets are dropped. Userinfo, a query or
    fragment, wildcards and other schemes are not hosts.
    """
    text = entry.strip()
    try:  # a bare IPv6 literal has colons that are not a port
        return str(ipaddress.ip_address(text.strip("[]")))
    except ValueError:
        pass
    parts = _split_http_url(text)
    if parts is None or parts.query or parts.fragment:
        return ""
    return _canonical_host(parts.hostname or "")


_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")
_REJECTION_MARKS = (
    ("@", "carries a user name or password"),
    ("?", "carries a query"),
    ("#", "carries a fragment"),
    ("*", "is a wildcard"),
)


def _loggable_entry(entry: str) -> str:
    """*entry* safe for a log line: userinfo masked, all after the host cut.

    A rejected entry may be a URL with a credential in it (a password, an
    api_key query, a token path), so the warning keeps only the scheme and
    the host part. Everything up to the LAST ``@`` is masked (a password may
    itself hold ``@``, ``/`` or ``?``); over-masking is the safe side.
    """
    scheme = _SCHEME.match(entry)
    prefix = scheme.group(0) if scheme else ""
    rest = entry[len(prefix) :]
    if "@" in rest:
        prefix += "***@"
        rest = rest.rsplit("@", 1)[1]
    authority = re.split(r"[/?#]", rest, maxsplit=1)[0]
    cut = rest[len(authority) : len(authority) + 1]
    return prefix + authority + (cut + "***" if cut else "")


def _rejection_reason(entry: str) -> str:
    """Why *entry* is not a host, in a few words (never quoting it)."""
    why = next((why for mark, why in _REJECTION_MARKS if mark in entry), "")
    scheme = _SCHEME.match(entry)
    if not why and scheme and scheme.group(1).lower() not in _HTTP_SCHEMES:
        why = "is not an http(s) URL"
    return why or "is not a host name"


@functools.lru_cache(maxsize=16)
def _parse_allowlist(raw: str) -> frozenset[str]:
    """The env value → normalized hosts (cached: one warning per value;
    the lease gate reads the list on every proxied request)."""
    hosts = set()
    for entry in raw.replace(",", " ").split():
        host = _allowlist_entry_host(entry)
        if host:
            hosts.add(host)
        else:
            logger.warning(
                "UMRS_INSTITUTIONAL_PROVIDER_HOSTS: ignoring %r — it %s (write "
                "bare hosts such as api.example.ac.jp; exact match, no wildcards)",
                _loggable_entry(entry),
                _rejection_reason(entry),
            )
    return frozenset(hosts)


def institutional_provider_hosts() -> frozenset[str]:
    """The site's institutional-provider allowlist (bare host names).

    Unset → the packaged default; set but empty → no hosts; otherwise the
    listed hosts replace the default. Entries are normalized to bare
    lowercase hosts; an unusable entry is ignored with a warning.
    """
    raw = os.environ.get("UMRS_INSTITUTIONAL_PROVIDER_HOSTS")
    if raw is None:
        return DEFAULT_INSTITUTIONAL_PROVIDER_HOSTS
    return _parse_allowlist(raw)


def url_host(url: str) -> str:
    """The canonical host of an http(s) URL — no port, no brackets, no
    trailing dot.

    ``""`` when unparseable, for another scheme, or when the URL carries
    userinfo (``https://a:b@host`` — parsers disagree on such hosts, so it
    never matches: fail closed). A scheme-less ``host[:port]`` is accepted
    (sealed sources store allow_host with a scheme, but config rows vary).
    """
    text = str(url or "").strip()
    if not text:
        return ""
    parts = _split_http_url(text)
    if parts is None:
        return ""
    return _canonical_host(parts.hostname or "")


def is_institutional_url(url: Any, hosts: frozenset[str] | None = None) -> bool:
    """True when *url*'s host is on the institutional allowlist.

    Exact host match only — no suffix games (``not-rikyu.r-ccs.riken.jp``
    is not ``api.rikyu.r-ccs.riken.jp``). Loopback is NOT institutional:
    the amendment's local class is a workstation-side notion; a remote
    gateway cannot see the client's machine (ADR-046).
    """
    host = url_host(str(url or ""))
    if not host:
        return False
    return host in (hosts if hosts is not None else institutional_provider_hosts())


def default_project_data_class() -> str:
    """The class a NEW project row takes (UMRS_PROJECT_DEFAULT_DATA_CLASS).

    ``"sensitive"`` unless the deployment opts out — the stand-alone
    single-host profile sets ``"non_sensitive"`` (ADR-062 amendment D-4).
    An invalid value falls back to ``"sensitive"``: a typo never widens.
    Existing rows are never rewritten by this (INSERT-time only).
    """
    raw = os.environ.get("UMRS_PROJECT_DEFAULT_DATA_CLASS", "").strip()
    if not raw:
        return "sensitive"
    if raw not in ("sensitive", "non_sensitive"):
        logger.warning(
            "invalid UMRS_PROJECT_DEFAULT_DATA_CLASS %r — using 'sensitive'", raw
        )
        return "sensitive"
    return raw


def data_class_for_project(pool: Any, project: str) -> str:
    """The project row's ``data_class`` (migration 0107).

    Returns ``""`` when *project* is empty, *pool* is None, the row is
    absent, or the lookup fails — the caller's gate never blocks on a
    lookup failure and never invents a class.
    """
    if pool is None or not project:
        return ""
    try:
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT data_class FROM projects WHERE project_id = %s",
                (project,),
            )
            row = cur.fetchone()
            return str(row[0]) if row and row[0] else ""
        finally:
            pool.putconn(conn)
    except Exception:  # the gate never blocks on lookup failure
        logger.warning("data_class lookup failed for %s", project, exc_info=True)
        return ""
