"""Effective limits and the resolved session policy (ADR-099 D-1..D-3).

Fold per field (D-3):

1. the site default (code plus env, :func:`site_defaults`);
2. replaced by the admin ``site`` rule when it sets the field;
3. replaced by the admin ``project`` and ``user`` rules that set it; when
   both do, the stricter wins (min for numbers, intersection for
   ``web.allow`` and the domain cap);
4. narrowed by the project owner's rule (never widened);
5. narrowed by the user's web choice (:func:`resolve_session_policy`).

``null`` in a rule means "no limit at this layer": it replaces a lower
layer's number in steps 2 and 3 (an admin can lift one user's cap) and never
narrows in step 4. A rule row for this site beats the all-sites row
(``venue = ''``) at the same key.

``none`` stays choosable whatever the caps say: turning the web off is
never wider than any cap. A domain cap removes ``public``; an empty
intersection of domain caps removes ``allowlist`` too.
"""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .model import (
    RULE_FIELDS,
    WEB_MODES,
    PolicyError,
    PolicyRule,
    WebChoice,
    domain_within,
    limit_value,
    normalize_domains,
)

logger = logging.getLogger(__name__)

SCHEMA = 1
_NO_CAP = object()  # domain cap "absent" while folding (None is "no limit")

ENV_IDLE = "UMRS_VENUE_IDLE_SECONDS"
ENV_WALL_CLOCK = "UMRS_VENUE_MAX_LIFETIME_SECONDS"
ENV_CONCURRENT = "UMRS_SESSION_CONCURRENT_PER_USER"
ENV_CHOICE_REQUIRED = "UMRS_SESSION_WEB_CHOICE_REQUIRED"
ENV_LEGACY_WEB = "UMRS_VENUE_WEB_POLICY_DEFAULT"


DEFAULT_CONCURRENT_PER_USER = 12
_UNLIMITED_WORDS = frozenset({"none", "null", "unlimited"})


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def concurrent_per_user(env: Mapping[str, str]) -> int | None:
    """``UMRS_SESSION_CONCURRENT_PER_USER``, the one place it is parsed.

    Empty: 12. A positive integer: that cap. ``none`` / ``unlimited`` /
    ``null``: no cap (null, the policy's "no limit"). Anything else (``0``,
    a negative number, a typo) is a misconfiguration: logged, and the
    default 12 applies, so the env never means "block every start" nor
    silently "no cap" (ADR-099 D-1: null = no limit, 0 is refused)."""
    raw = (env.get(ENV_CONCURRENT) or "").strip()
    if not raw:
        return DEFAULT_CONCURRENT_PER_USER
    if raw.lower() in _UNLIMITED_WORDS:
        return None
    value = limit_value(int(raw)) if raw.lstrip("-").isdigit() else None
    if value is None:
        logger.warning(
            "%s=%r is not a positive integer or 'unlimited'; using %d",
            ENV_CONCURRENT,
            raw,
            DEFAULT_CONCURRENT_PER_USER,
        )
        return DEFAULT_CONCURRENT_PER_USER
    return value


def site_defaults(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Step 1: the D-1 table's site defaults (today's values)."""
    env = os.environ if env is None else env
    return {
        "web": {"allow": list(WEB_MODES), "domains": None, "suggested_domains": []},
        "lifetime": {
            "idle_s": _env_int(env, ENV_IDLE, 30 * 60),
            "wall_clock_s": _env_int(env, ENV_WALL_CLOCK, 7 * 86_400),
        },
        "budget": {
            "tokens_per_day_user": None,
            "tokens_per_day_project": None,
            "rpm_user": None,
        },
        "sessions": {
            "concurrent_user": concurrent_per_user(env),
            "concurrent_project": None,
        },
    }


def web_choice_required(env: Mapping[str, str] | None = None) -> bool:
    """``UMRS_SESSION_WEB_CHOICE_REQUIRED`` (default true, D-2)."""
    env = os.environ if env is None else env
    raw = (env.get(ENV_CHOICE_REQUIRED) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def legacy_web_default(env: Mapping[str, str] | None = None) -> str:
    """``UMRS_VENUE_WEB_POLICY_DEFAULT``; read only when no choice is required."""
    env = os.environ if env is None else env
    return (env.get(ENV_LEGACY_WEB) or "").strip() or "public"


# ── combination ────────────────────────────────────────────────────────────


def _min_number(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _cap_intersection(a: Any, b: Any) -> Any:
    if a is _NO_CAP:
        return b
    if b is _NO_CAP:
        return a
    return [
        d
        for d in dict.fromkeys([*a, *b])
        if domain_within(d, a) and domain_within(d, b)
    ]


def _strictest(kind: str, a: Any, b: Any) -> Any:
    if kind == "modes":
        a = list(WEB_MODES) if a is None else a
        b = list(WEB_MODES) if b is None else b
        return [m for m in WEB_MODES if m in a and m in b]
    if kind == "domains":
        return _cap_intersection(a, b)
    return _min_number(a, b)


def _rule_value(kind: str, value: Any) -> Any:
    """A rule's value in fold form (a domain cap: list, or _NO_CAP)."""
    if kind == "domains" and not value:
        return _NO_CAP
    return value


def _pick(
    rules: Sequence[PolicyRule], venue: str, scope: str, subject: str, layer: str
) -> PolicyRule | None:
    """The rule at one key; this site's row beats the all-sites row."""
    if scope != "site" and not subject:
        return None
    found: dict[str, PolicyRule] = {
        r.venue: r
        for r in rules
        if (r.scope, r.subject, r.layer) == (scope, subject, layer)
        and r.venue in (venue, "")
    }
    return found.get(venue) or found.get("")


def _layers(
    rules: Sequence[PolicyRule], venue: str, user_id: str, project_id: str
) -> dict[str, PolicyRule | None]:
    return {
        "site:admin": _pick(rules, venue, "site", "", "admin"),
        "project:admin": _pick(rules, venue, "project", project_id, "admin"),
        "user:admin": _pick(rules, venue, "user", user_id, "admin"),
        "project:owner": _pick(rules, venue, "project", project_id, "owner"),
    }


def _setting(rule: PolicyRule | None, section: str, key: str) -> tuple[bool, Any]:
    if rule is None:
        return False, None
    part = rule.doc.get(section)
    if not isinstance(part, dict) or key not in part:
        return False, None
    return True, part[key]


def _narrow(
    kind: str, current: tuple[Any, str], other: tuple[Any, str]
) -> tuple[Any, str]:
    """``current`` narrowed by ``other``; the origin follows the stricter.

    On a tie with the site default the narrowing layer is recorded: a rule
    equal to the default must not read as ``default``, which a running
    session re-reads from the live env (ADR-099 D-7, Amendment 3)."""
    combined = _strictest(kind, current[0], other[0])
    keep = combined == current[0] and (combined != other[0] or current[1] != "default")
    return (combined, current[1]) if keep else (combined, other[1])


def _fold_limit(
    kind: str,
    default: Any,
    layers: dict[str, PolicyRule | None],
    section: str,
    key: str,
) -> tuple[Any, str]:
    value: tuple[Any, str] = (_rule_value(kind, default), "default")
    found, raw = _setting(layers["site:admin"], section, key)
    if found:
        value = (_rule_value(kind, raw), "site:admin")
    explicit = [
        (_rule_value(kind, raw), name)
        for name in ("project:admin", "user:admin")
        for found, raw in [_setting(layers[name], section, key)]
        if found
    ]
    if explicit:
        value = explicit[0] if len(explicit) == 1 else _narrow(kind, *explicit)
    found, raw = _setting(layers["project:owner"], section, key)
    if found and raw is not None:
        value = _narrow(kind, value, (_rule_value(kind, raw), "project:owner"))
    return value


def _fold_hint(
    default: Any, layers: dict[str, PolicyRule | None], section: str, key: str
) -> tuple[Any, str]:
    """A hint (``web.suggested_domains``): the most specific setter wins."""
    value: tuple[Any, str] = (default, "default")
    for name in ("site:admin", "project:admin", "user:admin", "project:owner"):
        found, raw = _setting(layers[name], section, key)
        if found and raw is not None:
            value = (list(raw), name)
    return value


def _allowed_modes(allow: Any) -> list[str]:
    """``web.allow`` in mode order (None = every mode); ``none`` always."""
    allowed = WEB_MODES if allow is None else allow
    return [m for m in WEB_MODES if m in allowed or m == "none"]


def _removed_by_cap(cap: Any) -> set[str]:
    """A cap removes ``public``; an empty cap removes ``allowlist`` too."""
    if cap is _NO_CAP:
        return set()
    return {"public"} if cap else {"public", "allowlist"}


def _apply_domain_cap(web: dict[str, Any], origin: dict[str, str]) -> None:
    """The fold's web section in output form; the domain cap's origin
    explains a mode it removed."""
    cap = web["domains"]
    allow = _allowed_modes(web["allow"])
    removed = _removed_by_cap(cap)
    if removed & set(allow):
        origin["web.allow"] = origin["web.domains"]
    web["allow"] = [m for m in allow if m not in removed]
    if cap is _NO_CAP:
        web["domains"] = []
        return
    web["domains"] = list(cap)
    web["suggested_domains"] = [
        d for d in web["suggested_domains"] if domain_within(d, cap)
    ]


def effective_limits(
    rules: Sequence[PolicyRule],
    *,
    venue: str,
    user_id: str,
    project_id: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Steps 1-4 of the fold: what a session of ``user_id`` in ``project_id``
    at ``venue`` may get, with the layer that set each value (``origin``)."""
    layers = _layers(rules, venue, user_id, project_id)
    out: dict[str, Any] = {"schema": SCHEMA, "venue": venue, "user_id": user_id}
    out["project_id"] = project_id
    origin: dict[str, str] = {}
    for section, fields in RULE_FIELDS.items():
        part: dict[str, Any] = {}
        for key, kind in fields.items():
            default = defaults.get(section, {}).get(key)
            value = (
                _fold_hint(default, layers, section, key)
                if key == "suggested_domains"
                else _fold_limit(kind, default, layers, section, key)
            )
            part[key], origin[f"{section}.{key}"] = value
        out[section] = part
    _apply_domain_cap(out["web"], origin)
    out["origin"] = origin
    return out


# ── the owner layer (D-3: may only narrow) ─────────────────────────────────


def _widening(field_name: str, requested: Any, cap: Any, capped_by: str) -> PolicyError:
    return PolicyError(
        "widening_refused",
        f"{field_name}: {requested!r} is wider than the effective {cap!r} "
        f"(set by {capped_by})",
        field=field_name,
        requested=requested,
        cap=cap,
        capped_by=capped_by,
    )


def _wider_modes(value: list[str], cap: list[str]) -> list[str]:
    return [m for m in value if m not in cap and m != "none"]


def _outside_cap(value: list[str], cap: list[str]) -> list[str]:
    return [d for d in value if cap and not domain_within(d, cap)]


def _owner_field_problem(
    kind: str, value: Any, cap: Any, field_name: str, capped_by: str
) -> PolicyError | None:
    """The refusal of one owner field wider than the admin value, else None."""
    if kind == "modes":
        wider = _wider_modes(value, cap)
    elif kind == "domains":
        wider = _outside_cap(value, cap)
    else:
        wider = value if cap is not None and value > cap else None
    return _widening(field_name, wider, cap, capped_by) if wider else None


def check_owner_rule(doc: dict[str, Any], admin_limits: dict[str, Any]) -> None:
    """Refuse an owner rule wider than the admin-effective value
    (``admin_limits``: :func:`effective_limits` over the admin layer)."""
    for section, part in doc.items():
        for key, value in part.items():
            kind = RULE_FIELDS[section][key]
            if value is None or key == "suggested_domains":
                continue
            field_name = f"{section}.{key}"
            cap_key = "allow" if kind == "modes" else key
            problem = _owner_field_problem(
                kind,
                value,
                admin_limits[section][cap_key],
                field_name,
                admin_limits["origin"].get(field_name, "default"),
            )
            if problem is not None:
                raise problem


# ── the choice (step 5) ────────────────────────────────────────────────────


def _choice_required(web: dict[str, Any]) -> PolicyError:
    """The refusal of a start without a web choice. The message alone says
    everything a client needs (the typed detail stays server-side; clients
    read the structured caps from ``GET …/venue/policy``, ADR-099 §4)."""
    domains = ", ".join(web["domains"]) or "any"
    suggested = ", ".join(web["suggested_domains"]) or "none"
    return PolicyError(
        "web_choice_required",
        "choose the agent's web access for this session: web_policy "
        f"{' | '.join(web['allow'])} (allowlist domains within: {domains}; "
        f"suggested: {suggested}); r4: --web off|allowlist|public "
        "or R4_WEB_ACCESS",
        choices=list(web["allow"]),
        domains_cap=list(web["domains"]),
        suggested_domains=list(web["suggested_domains"]),
    )


def _legacy_domains(web: dict[str, Any], sent: list[str]) -> list[str]:
    """The allowlist a legacy start gets: the domains the client sent that
    fall inside the cap, else the cap itself, else the suggested domains."""
    cap = web["domains"]
    inside = [d for d in sent if not cap or domain_within(d, cap)]
    return inside or list(cap or web["suggested_domains"])


def _legacy_choice(
    web: dict[str, Any], legacy_default: str, sent: list[str]
) -> tuple[str, list[str]]:
    """``UMRS_VENUE_WEB_POLICY_DEFAULT`` clipped to the caps: the widest
    allowed mode at or below it (allowlist only with domains to use)."""
    order = ["public", "allowlist", "none"]
    start = order.index(legacy_default) if legacy_default in order else 0
    domains = _legacy_domains(web, sent)
    for mode in order[start:]:
        if mode in web["allow"] and (mode != "allowlist" or domains):
            return mode, domains if mode == "allowlist" else []
    return "none", []


def _check_choice(limits: dict[str, Any], mode: str, domains: list[str]) -> None:
    web, origin = limits["web"], limits["origin"]
    if mode not in web["allow"]:
        raise _widening("web.mode", mode, web["allow"], origin.get("web.allow", ""))
    if mode != "allowlist":
        return
    if not domains:
        raise PolicyError(
            "invalid",
            "web_policy allowlist needs at least one domain (web_allow_domains)",
            field="web.domains",
        )
    outside = [
        d for d in domains if web["domains"] and not domain_within(d, web["domains"])
    ]
    if outside:
        raise _widening(
            "web.domains", outside, web["domains"], origin.get("web.domains", "")
        )


def _chosen(
    limits: dict[str, Any], choice: WebChoice, require_choice: bool, legacy_default: str
) -> tuple[str, list[str], str]:
    mode = (choice.mode or "").strip()
    if mode:
        if mode not in WEB_MODES:
            raise PolicyError(
                "invalid",
                f"web_policy must be one of {', '.join(WEB_MODES)}",
                field="web.mode",
            )
        return mode, normalize_domains(list(choice.domains), "web.domains"), "choice"
    if require_choice:
        raise _choice_required(limits["web"])
    sent = normalize_domains(list(choice.domains), "web.domains")
    legacy_mode, legacy_domains = _legacy_choice(limits["web"], legacy_default, sent)
    return legacy_mode, legacy_domains, "default"


def resolve_session_policy(
    limits: dict[str, Any],
    choice: WebChoice,
    *,
    require_choice: bool,
    legacy_default: str,
    now: datetime,
) -> dict[str, Any]:
    """The resolved session policy: ``limits`` with ``web`` replaced by the
    checked choice (typed :class:`PolicyError` on refusal)."""
    mode, domains, tag = _chosen(limits, choice, require_choice, legacy_default)
    _check_choice(limits, mode, domains)
    policy = {
        key: copy.deepcopy(limits[key])
        for key in (
            "schema",
            "venue",
            "user_id",
            "project_id",
            "lifetime",
            "budget",
            "sessions",
        )
    }
    policy["web"] = {"mode": mode, "domains": domains if mode == "allowlist" else []}
    origin = {k: v for k, v in limits["origin"].items() if not k.startswith("web.")}
    origin["web.mode"] = origin["web.domains"] = tag
    policy["origin"] = origin
    stamp = now.astimezone(timezone.utc).replace(microsecond=0)
    policy["resolved_at"] = stamp.isoformat().replace("+00:00", "Z")
    return policy
