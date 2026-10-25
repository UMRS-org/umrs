"""Knowledge URL ingest under the ADR-099 web policy (2026-10-25, wave 79).

URL and git ingest used to be bound by the front's
``extra.knowledge_ingest.allowed_domains`` alone (``ingest_allowlist``).
Since wave 79 the web policy that binds the agent's browsing (ADR-099 D-1,
D-3: ``web.allow`` modes and the ``web.domains`` cap, folded from the site,
project and user admin rules and narrowed by the project owner's rule)
binds ingest too, per project and per user. No second list and no new
store: the rules are the ones ``session_policy_rules`` already holds, read
through :func:`umrs.session_policy.effective_limits_for`.

The policy's widest allowed mode decides (the person's own narrowing is the
request's ``allow_domains``, which only narrows):

* ``none`` only: URL and git ingest are refused, naming the layer that set it;
* ``allowlist``: the host, and every redirect hop, must sit inside the
  ``web.domains`` cap, matched like the agent's browse guard (squid
  ``dstdomain``: ``a.org`` that host, ``.a.org`` it and its subdomains).
  Without a cap the person names the sites per request: the request's
  ``allow_domains``, else the URL's own host (redirects stay on it); each
  named entry must be a domain a session open accepts (``a.org`` or
  ``*.a.org``; no ``*``, no ``*.com``), else the ingest is refused;
* ``public``: the front list alone.

The result is always intersected with the front list, and every network
guard of ``egress_fetch`` (global unicast only, pinned connection, every
redirect hop re-checked, ports 80/443, size and time caps,
``denied_networks``) still applies on top. Rules that cannot be read refuse
the ingest (fail closed, like a session start's ``sessionpolicy.unavailable``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .egress_fetch import ALLOW_ALL, _normal, _split_port, domain_allowed

logger = logging.getLogger(__name__)

#: Who may change each layer, in words a refused user can act on.
_LAYER_WORDS = {
    "default": "the site default",
    "site:admin": "the site's admin rule",
    "project:admin": "an admin rule for this project",
    "user:admin": "an admin rule for this user",
    "project:owner": "the project owner's rule",
}


@dataclass(frozen=True)
class IngestRequester:
    """Whose policy binds an ingest: the job's owner and its project."""

    user_id: str
    project_id: str


@dataclass(frozen=True)
class IngestWebPolicy:
    """The web policy an ingest of ``user_id`` in ``project_id`` runs under."""

    mode: str
    domains: tuple[str, ...] = ()
    set_by: str = "default"
    domains_set_by: str = "default"
    project_id: str = ""
    user_id: str = ""
    venue: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "domains": list(self.domains),
            "set_by": self.set_by,
            "domains_set_by": self.domains_set_by,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "venue": self.venue,
            "per_request": self.mode == "allowlist" and not self.domains,
        }


class IngestPolicyUnavailable(RuntimeError):
    """The web policy could not be read: the ingest is refused (fail closed)."""


def layer_words(layer: str) -> str:
    """``project:owner`` → "the project owner's rule (project:owner)"."""
    words = _LAYER_WORDS.get(layer)
    return f"{words} ({layer})" if words else layer or "an unnamed layer"


def _text(mapping: Mapping[str, Any], key: str, default: str = "") -> str:
    return str(mapping.get(key) or default)


def _widest_mode(web: Mapping[str, Any]) -> str:
    allow = list(web.get("allow") or [])
    return next((m for m in ("public", "allowlist") if m in allow), "none")


def policy_from_limits(limits: Mapping[str, Any]) -> IngestWebPolicy:
    """The ingest policy of ADR-099 effective limits: their widest mode."""
    web = limits.get("web") or {}
    origin = limits.get("origin") or {}
    mode = _widest_mode(web)
    domains = tuple(web.get("domains") or ()) if mode == "allowlist" else ()
    return IngestWebPolicy(
        mode=mode,
        domains=domains,
        set_by=_text(origin, "web.allow", "default"),
        domains_set_by=_text(origin, "web.domains", "default"),
        project_id=_text(limits, "project_id"),
        user_id=_text(limits, "user_id"),
        venue=_text(limits, "venue"),
    )


def default_rule_store() -> Any:
    """The rule store the server's other readers use: Postgres when the
    environment names a database, else in memory (no database: no rules,
    the site defaults apply, as for a venue service without a pool)."""
    from umrs.session_policy import PolicyRuleStore
    from umrs.shared.pg import get_default_sync_pool, pg_configured_from_env

    if not pg_configured_from_env():
        return PolicyRuleStore()
    return PolicyRuleStore(get_default_sync_pool())


def resolve_ingest_web_policy(
    *,
    venue: str,
    requester: IngestRequester,
    store: Any = None,
    env: Mapping[str, str] | None = None,
) -> IngestWebPolicy:
    """The ADR-099 policy of *requester* at *venue*; unreadable rules or an
    unknown site name raise :class:`IngestPolicyUnavailable`."""
    from umrs.session_policy import effective_limits_for

    if not venue:
        raise IngestPolicyUnavailable(
            "this server's site name is unknown, so the project's web policy "
            "(ADR-099) cannot be read; URL ingest is refused until it can"
        )
    try:
        limits = effective_limits_for(
            store if store is not None else default_rule_store(),
            venue=venue,
            user_id=requester.user_id,
            project_id=requester.project_id,
            env=env,
        )
    except Exception as exc:  # any read failure fails closed
        logger.warning("session policy rules unreadable for ingest", exc_info=True)
        raise IngestPolicyUnavailable(
            f"the project's web policy (ADR-099 session policy rules) is unreadable "
            f"({type(exc).__name__}); URL ingest is refused until it can be read"
        ) from exc
    return policy_from_limits(limits)


# ── squid dstdomain caps as fetcher patterns ───────────────────────────────


def squid_patterns(domains: Sequence[str]) -> list[str]:
    """``web.domains`` (squid ``dstdomain``) as fetcher patterns: ``a.org``
    stays that host, ``.a.org`` becomes ``a.org`` and ``*.a.org``."""
    out: list[str] = []
    for raw in domains:
        entry = _normal(raw)
        if entry.startswith("."):
            out.extend((entry[1:], f"*{entry}"))
        elif entry:
            out.append(entry)
    return [p for p in dict.fromkeys(out) if p and p != "*"]


def _covers(outer: str, inner: str) -> bool:
    """Does fetcher pattern *outer* admit every host *inner* admits?"""
    if outer == ALLOW_ALL:
        return True
    if inner == ALLOW_ALL:
        return False
    if inner.startswith("*."):
        return outer.startswith("*.") and inner.endswith(outer[1:])
    return domain_allowed(inner, [outer])


def _pair(entry: str, pattern: str) -> str:
    """The narrower of a front entry and a policy pattern, or ``""`` when
    neither holds the other (a front ``host:port`` keeps its port)."""
    host = _normal(_split_port(_normal(entry))[0])
    if _covers(pattern, host):
        return entry
    if _covers(host, pattern):
        return pattern
    return ""


def intersect_patterns(site: Sequence[str], policy: Sequence[str]) -> list[str]:
    """The hosts both lists admit, as fetcher patterns (front order)."""
    out: list[str] = []
    for entry in site:
        for pattern in policy:
            kept = _pair(str(entry), _normal(pattern))
            if kept and kept not in out:
                out.append(kept)
    return out


def policy_bound(
    policy: IngestWebPolicy,
    site: Sequence[str],
    host: str,
    caller: Sequence[str] | None,
) -> list[str]:
    """The front list narrowed by *policy* (``[]`` for ``none``)."""
    if policy.mode == "public":
        return list(site)
    if policy.mode != "allowlist":
        return []
    if policy.domains:
        chosen = squid_patterns(policy.domains)
    else:
        chosen = per_request_patterns(caller, host)
    return intersect_patterns(site, chosen)


# ── per-request sites (allowlist with no cap) ──────────────────────────────


def _caller_patterns(raw: Any) -> list[str]:
    """One request ``allow_domains`` entry as fetcher patterns, or ``[]``
    when a session open would refuse it as a domain (review fix, wave 79).

    ``allow_domains`` holds fetcher patterns (ADR-015): ``a.org`` that host,
    ``*.a.org`` its subdomains. The domain must pass a session open's
    ``web.domains`` check (``session_policy.model.normalize_domain``, with
    ``*.a.org`` checked as ``.a.org``), so ``*``, ``*.com`` and ``com`` are
    refused. The squid ``.a.org`` form is refused here too: the fetcher would
    read it as a literal host and fetch nothing.
    """
    from umrs.session_policy.model import PolicyError, normalize_domain

    text = _normal(raw) if isinstance(raw, str) else ""
    if text.startswith("."):
        return []
    try:
        normalize_domain(text[1:] if text.startswith("*.") else text, "allow_domains")
    except PolicyError:
        return []
    return [text]


def invalid_caller_domains(caller: Sequence[str] | None) -> list[str]:
    """The request's ``allow_domains`` entries a session open would refuse."""
    return [str(raw) for raw in caller or () if not _caller_patterns(raw)]


def per_request_patterns(caller: Sequence[str] | None, host: str) -> list[str]:
    """The sites an uncapped ``allowlist`` request names: its valid
    ``allow_domains`` (at most ``MAX_DOMAINS``), else the URL's own host."""
    from umrs.session_policy.model import MAX_DOMAINS

    if not caller:
        return [host] if host else []
    if len(caller) > MAX_DOMAINS:
        return []
    out: list[str] = []
    for raw in caller:
        out.extend(p for p in _caller_patterns(raw) if p not in out)
    return out


# ── refusals ───────────────────────────────────────────────────────────────

_HOW = (
    "the project owner (PUT /umrs/projects/{project}/session-policy) or an "
    "admin ('umrs-server admin policy set') changes it; see 'r4 knowledge "
    "allowlist'"
)


def _who(policy: IngestWebPolicy) -> str:
    project = repr(policy.project_id) if policy.project_id else "(no project)"
    return f"project {project}" + (
        f", user {policy.user_id!r}" if policy.user_id else ""
    )


def off_refusal(policy: IngestWebPolicy) -> str:
    """The refusal for a policy whose only mode is ``none``."""
    return (
        f"web access is off for {_who(policy)} (ADR-099 web policy 'none', set by "
        f"{layer_words(policy.set_by)}), so URL and git ingest are refused; "
        + _HOW.format(project=policy.project_id or "<project>")
    )


def outside_refusal(policy: IngestWebPolicy, host: str) -> str:
    """The refusal for a host outside an ``allowlist`` policy."""
    if not policy.domains:
        return (
            f"domain {host!r} is not among the sites this request names "
            f"(allow_domains): {_who(policy)} allows allowlist web access only "
            f"(set by {layer_words(policy.set_by)}); name the site in allow_domains"
        )
    return (
        f"domain {host!r} is outside the web policy of {_who(policy)}: "
        f"allowlist {', '.join(policy.domains)} (set by "
        f"{layer_words(policy.domains_set_by)}), the sites the agent may browse "
        "there; " + _HOW.format(project=policy.project_id or "<project>")
    )


def caller_refusal(policy: IngestWebPolicy, caller: Sequence[str] | None) -> str:
    """Why an uncapped ``allowlist`` refuses the request's ``allow_domains``
    (entries a session open would refuse, or too many), or ``""``."""
    from umrs.session_policy.model import MAX_DOMAINS

    if policy.mode != "allowlist" or policy.domains or not caller:
        return ""
    if len(caller) > MAX_DOMAINS:
        return (
            f"allow_domains names {len(caller)} sites; {_who(policy)} allows "
            f"allowlist web access only, which takes at most {MAX_DOMAINS} sites"
        )
    bad = invalid_caller_domains(caller)
    if not bad:
        return ""
    return (
        f"allow_domains {', '.join(repr(b) for b in bad)} names no single site: "
        f"{_who(policy)} allows allowlist web access only (set by "
        f"{layer_words(policy.set_by)}), so each entry must be a domain as at a "
        "session open (a.org, or *.a.org for its subdomains; no '*', no '*.com')"
    )


def ingest_refusal(
    policy: IngestWebPolicy,
    site: Sequence[str],
    host: str,
    caller: Sequence[str] | None = None,
) -> str:
    """Why *policy* refuses ingesting from *host*, or ``""`` (the front list
    and the network guards are checked by their own code)."""
    if policy.mode == "none":
        return off_refusal(policy)
    if policy.mode == "public":
        return ""
    refused = caller_refusal(policy, caller)
    if refused or not host:
        return refused
    if domain_allowed(host, policy_bound(policy, site, host, caller)):
        return ""
    return outside_refusal(policy, host)


def policy_summary(policy: IngestWebPolicy) -> str:
    """One sentence for ``r4 knowledge allowlist``."""
    by = layer_words(policy.set_by)
    if policy.mode == "none":
        return off_refusal(policy)
    if policy.mode == "public":
        return f"web policy 'public' for {_who(policy)} (set by {by}): the site list applies"
    if not policy.domains:
        return (
            f"web policy 'allowlist' for {_who(policy)} (set by {by}), with no domain "
            "cap: each ingest fetches only the sites its request names (allow_domains, "
            "else the URL's own host)"
        )
    return (
        f"web policy 'allowlist' for {_who(policy)}: {', '.join(policy.domains)} "
        f"(set by {layer_words(policy.domains_set_by)}); ingest fetches only hosts on "
        "both this list and the site list"
    )


__all__ = [
    "IngestPolicyUnavailable",
    "IngestRequester",
    "IngestWebPolicy",
    "caller_refusal",
    "default_rule_store",
    "ingest_refusal",
    "intersect_patterns",
    "invalid_caller_domains",
    "layer_words",
    "off_refusal",
    "outside_refusal",
    "per_request_patterns",
    "policy_bound",
    "policy_from_limits",
    "policy_summary",
    "resolve_ingest_web_policy",
    "squid_patterns",
]
