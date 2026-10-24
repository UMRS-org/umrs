"""The site's knowledge-ingest allowlist, as the fetcher applies it (2026-10-25).

One source: ``extra.knowledge_ingest.allowed_domains`` in the front's
``site.yaml``. Each entry is one of

* ``example.org`` (that host), ``*.example.org`` (its subdomains),
  ``host:8443`` (that host on a port other than 80/443), as before;
* ``@recommended``: the curated list packaged beside this module
  (``ingest_domains_recommended.yaml``), combinable with other entries;
* ``*``: every public host. The network guards in ``egress_fetch`` all stay
  (global unicast addresses only, pinned connection, ports 80/443, every
  redirect hop re-checked, size and time caps, content scan).

``extra.knowledge_ingest.denied_networks`` (CIDR strings) names address
space no hop may resolve into, whatever the list admits: the site's own
network when it is public address space (RIKEN's 134.160.0.0/16 is global
unicast, so the global check alone admitted the intranet; review
2026-10-25).

:func:`effective_site_allowlist` is the one place entries are expanded: the
ingest resolver (``sources.SourceResolver``) and the read-only route
``GET /umrs/k2/ingest/allowlist`` both call it. Refused entries (an unknown
``@token``, ``*:port``) are dropped with one log line each; the rest apply.
:func:`announce_site_allowlist` writes those lines when the server starts.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Any

from .egress_fetch import (
    ALLOW_ALL,
    _normal,
    _split_port,
    domain_allowed,
    not_listed,
    parse_networks,
    port_allowed,
)

logger = logging.getLogger(__name__)

RECOMMENDED_TOKEN = "@recommended"
RECOMMENDED_FILE = "ingest_domains_recommended.yaml"

#: Log lines already written (each refused entry and the ``*`` warning are
#: logged once per process, not on every ingest).
_REPORTED: set[str] = set()


def _report_once(level: int, message: str) -> None:
    if message not in _REPORTED:
        _REPORTED.add(message)
        logger.log(level, message)


@functools.lru_cache(maxsize=1)
def recommended_domains() -> tuple[str, ...]:
    """The packaged recommended list, read from the installed package.

    A file that cannot be read or is not a list of strings expands to
    nothing (fail-closed for ``@recommended``; other entries still apply).
    """
    from importlib.resources import files

    import yaml

    try:
        text = (
            files("umrs.knowledge")
            .joinpath(RECOMMENDED_FILE)
            .read_text(encoding="utf-8")
        )
        loaded = yaml.safe_load(text)
    except Exception:  # a broken package file refuses the token
        logger.exception(
            "cannot read %s: '@recommended' admits nothing", RECOMMENDED_FILE
        )
        return ()
    if not isinstance(loaded, list) or not all(isinstance(d, str) for d in loaded):
        logger.error(
            "%s is not a list of host entries: '@recommended' admits nothing",
            RECOMMENDED_FILE,
        )
        return ()
    return tuple(d.strip() for d in loaded if d.strip())


def _expand(entry: str) -> tuple[str, ...]:
    """One configured entry → the entries it stands for (none when refused)."""
    if not entry:
        return ()
    if entry.lower() == RECOMMENDED_TOKEN:
        return recommended_domains()
    if entry.startswith("@"):
        _report_once(
            logging.ERROR,
            f"knowledge_ingest.allowed_domains: unknown entry {entry!r} refused "
            f"(known: {RECOMMENDED_TOKEN!r}); the other entries apply",
        )
        return ()
    host, port = _split_port(_normal(entry))
    if host.rstrip(".") != ALLOW_ALL:
        return (entry,)
    if port is not None:
        _report_once(
            logging.ERROR,
            f"knowledge_ingest.allowed_domains: {entry!r} refused: a port is opened "
            "per host ('docs.example.org:8443'), never on every host",
        )
        return ()
    # "*." and "*.." admit every host at fetch time (the fetcher strips the
    # dots): they are "*", and reported as such (review 2026-10-25).
    return (ALLOW_ALL,)


def effective_site_allowlist(raw: Any) -> list[str]:
    """The configured entries expanded: ``@recommended`` replaced by the
    packaged list, refused entries dropped, duplicates removed (first kept).

    *raw* is the ``allowed_domains`` value as configured: a list, or one
    string; anything else (absent included) is the empty list (fail-closed).
    """
    entries: list[str] = []
    for entry in configured_list(raw):
        for item in _expand(entry):
            if item not in entries:
                entries.append(item)
    if ALLOW_ALL in entries:
        _report_once(
            logging.WARNING,
            "knowledge_ingest.allowed_domains contains '*': knowledge ingest may fetch "
            "every public web site (non-global addresses and the site's "
            "denied_networks stay blocked)",
        )
    return entries


def configured_list(raw: Any) -> list[str]:
    """The ``allowed_domains`` value as a list of non-empty strings."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(entry).strip() for entry in raw if str(entry).strip()]


def _ingest_block(config: Any) -> dict[str, Any]:
    """``extra.knowledge_ingest`` of a site config (``{}`` without one)."""
    extra = getattr(config, "extra", None)
    block = extra.get("knowledge_ingest") if isinstance(extra, dict) else None
    return block if isinstance(block, dict) else {}


def configured_entries(config: Any) -> list[str]:
    """``extra.knowledge_ingest.allowed_domains`` of a site config, raw."""
    return configured_list(_ingest_block(config).get("allowed_domains"))


def site_denied_networks(config: Any) -> list[str]:
    """``extra.knowledge_ingest.denied_networks`` as canonical CIDR strings.

    An entry that is not a network or an address is dropped with one log
    line; the rest apply (an unreadable entry cannot deny anything, so it is
    said loudly rather than refusing every fetch on the front).
    """
    networks, refused = parse_networks(_ingest_block(config).get("denied_networks"))
    for entry in refused:
        _report_once(
            logging.ERROR,
            f"knowledge_ingest.denied_networks: {entry!r} is not a network "
            "(e.g. '134.160.0.0/16') and denies nothing",
        )
    return [str(network) for network in networks]


def site_allowlist(config: Any) -> list[str]:
    """The effective allowlist of a site config (``[]`` without one)."""
    return effective_site_allowlist(configured_entries(config))


def _front_dir(site_name: str) -> str:
    return site_name if site_name.endswith("-front") else f"{site_name}-front"


def how_to_change(site_name: str) -> str:
    """One sentence a user can act on: who changes the list, and where."""
    return (
        "Ask your UMRS operator to edit extra.knowledge_ingest.allowed_domains in the "
        f"site.yaml of site '{site_name}' (on umrs-front: umrs/deploy/"
        f"{_front_dir(site_name)}/site.yaml) and redeploy; '*' allows every public web "
        "site (non-global addresses and the site's denied_networks stay blocked)."
    )


_SCP_GIT = re.compile(r"^[^\s/@:]+@[^\s/@:]+:(?!//)")


def _scheme_refusal(text: str) -> str:
    """Why the fetcher would refuse *text* before any host check, or ``""``
    (it fetches http and https only; review 2026-10-25)."""
    if "://" in text:
        scheme = text.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            return f"scheme {scheme!r} is not fetched (http/https only)"
        return ""
    if _SCP_GIT.match(text):
        return (
            f"{text!r} (scp-style git address) is not fetched: use the https clone URL"
        )
    return ""


def _url_parts(url: str) -> tuple[str, int | None, str]:
    """``(host, port, error)`` of a URL or a bare host name."""
    from urllib.parse import urlparse

    text = url.strip()
    refusal = _scheme_refusal(text)
    if refusal:
        return "", None, refusal
    parsed = urlparse(text if "://" in text else f"https://{text}")
    try:
        port = parsed.port
    except ValueError:
        return "", None, f"invalid port in {url!r}"
    host = parsed.hostname or ""
    if not host:
        return "", None, f"no host in {url!r}"
    default = 443 if parsed.scheme == "https" else 80
    return host, port or default, ""


def check_url(url: str, entries: list[str]) -> dict[str, Any]:
    """Would the allowlist admit *url*'s host and port? Names only: the
    address, size and content checks run when the fetch happens."""
    host, port, error = _url_parts(url)
    result: dict[str, Any] = {"url": url, "host": host, "port": port, "allowed": False}
    if error:
        return {**result, "reason": error}
    if not domain_allowed(host, entries):
        return {**result, "reason": not_listed(host)}
    if not port_allowed(host, int(port or 0), entries):
        return {
            **result,
            "reason": f"port {port} of {host} is not allowed (80/443 only, unless an "
            f"entry names '{host}:{port}')",
        }
    return {
        **result,
        "allowed": True,
        "reason": "on the allowlist; the network checks (public address, size, "
        "content scan) run when the fetch happens",
    }


def load_site_config() -> Any:
    """The running site config, or ``None`` when none loads."""
    try:
        from umrs.orchestrator.site_config import get_site_config

        return get_site_config()
    except Exception:  # no config → empty allowlist
        logger.warning(
            "site config unavailable: the ingest allowlist is empty", exc_info=True
        )
        return None


def allowlist_view(config: Any, check: str | None = None) -> dict[str, Any]:
    """What ``GET /umrs/k2/ingest/allowlist`` answers."""
    configured = configured_entries(config)
    entries = effective_site_allowlist(configured)
    site = str(getattr(config, "name", "") or "unknown")
    view: dict[str, Any] = {
        "allow_all": ALLOW_ALL in entries,
        "entries": entries,
        "configured": configured,
        "denied_networks": site_denied_networks(config),
        "how_to_change": how_to_change(site),
        "site": site,
    }
    if check is not None:
        view["check"] = check_url(check, entries)
    return view


def announce_site_allowlist(config: Any) -> list[str]:
    """Expand the site's list once at server start, so a refused entry, the
    ``*`` warning and the list in force reach the log at deploy time, not at
    the first ingest (review 2026-10-25)."""
    entries = site_allowlist(config)
    denied = site_denied_networks(config)
    scope = (
        "every public web site" if ALLOW_ALL in entries else f"{len(entries)} entries"
    )
    logger.info(
        "knowledge ingest allowlist: %s; denied networks: %s",
        scope,
        ", ".join(denied) or "none",
    )
    return entries


__all__ = [
    "RECOMMENDED_FILE",
    "RECOMMENDED_TOKEN",
    "allowlist_view",
    "announce_site_allowlist",
    "check_url",
    "configured_entries",
    "configured_list",
    "effective_site_allowlist",
    "how_to_change",
    "load_site_config",
    "recommended_domains",
    "site_allowlist",
    "site_denied_networks",
]
