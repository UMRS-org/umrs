"""Session policy rule documents (ADR-099 D-1, D-3).

A rule is a partial policy document an admin (any scope) or a project owner
(project scope) stores in ``session_policy_rules`` (migration 0125). This
module holds the vocabulary, the typed refusal and the validation every
write and every web choice goes through. Pure: no database, no FastAPI.

Rule document (every key optional; ``null`` = no limit at this layer)::

    {"web": {"allow": [...], "domains": [...], "suggested_domains": [...]},
     "lifetime": {"idle_s": int, "wall_clock_s": int},
     "budget": {"tokens_per_day_user": int, "tokens_per_day_project": int,
                "rpm_user": int},
     "sessions": {"concurrent_user": int, "concurrent_project": int}}

Counts and token budgets are integers >= 1 or ``null`` (no limit): ``0`` is
refused (``sessionpolicy.invalid``) rather than read as either "no limit" or
"block", and every enforcer reads a value through :func:`limit_value`.

``web.domains`` is a domain CAP (squid ``dstdomain`` entries); an empty
list is the same as no cap. ``web.suggested_domains`` is a hint for the
session-open question, never a limit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

WEB_MODES: tuple[str, ...] = ("none", "allowlist", "public")
SCOPES: tuple[str, ...] = ("site", "project", "user")
LAYERS: tuple[str, ...] = ("admin", "owner")

MAX_DOMAINS = 64
MIN_LIFETIME_S = 60
# Counts and token budgets: null = no limit, so 0 would only mean "block
# everything", which RS-1 does not offer (a disabled account is the block).
MIN_COUNT = 1
_DOMAIN_RE = re.compile(r"^\.?[a-z0-9-]+(\.[a-z0-9-]+)+$")

# section -> field -> kind ("modes" | "domains" | "count" | "seconds")
RULE_FIELDS: dict[str, dict[str, str]] = {
    "web": {"allow": "modes", "domains": "domains", "suggested_domains": "domains"},
    "lifetime": {"idle_s": "seconds", "wall_clock_s": "seconds"},
    "budget": {
        "tokens_per_day_user": "count",
        "tokens_per_day_project": "count",
        "rpm_user": "count",
    },
    "sessions": {"concurrent_user": "count", "concurrent_project": "count"},
}


class PolicyError(Exception):
    """Typed refusal; ``code`` is the ``sessionpolicy`` name without the
    family (``invalid``, ``web_choice_required``, ``widening_refused``,
    ``forbidden``, ``not_found``, ``unavailable``)."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = dict(detail)


@dataclass(frozen=True)
class PolicyRule:
    venue: str
    scope: str
    subject: str
    layer: str
    doc: dict[str, Any] = field(default_factory=dict)
    set_by: str = ""
    note: str = ""
    updated_at: datetime | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.venue, self.scope, self.subject, self.layer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "scope": self.scope,
            "subject": self.subject,
            "layer": self.layer,
            "doc": self.doc,
            "set_by": self.set_by,
            "note": self.note,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True)
class WebChoice:
    mode: str  # "" = no choice
    domains: tuple[str, ...] = ()


def _invalid(field_name: str, message: str) -> PolicyError:
    return PolicyError("invalid", message, field=field_name)


def check_rule_key(scope: str, subject: str, layer: str) -> None:
    """The (scope, subject, layer) shape the table's constraints enforce."""
    if scope not in SCOPES:
        raise _invalid("scope", f"scope must be one of {', '.join(SCOPES)}")
    if layer not in LAYERS:
        raise _invalid("layer", f"layer must be one of {', '.join(LAYERS)}")
    if layer == "owner" and scope != "project":
        raise _invalid("layer", "an owner rule is a project rule")
    if scope == "site" and subject:
        raise _invalid("subject", "a site rule has no subject")
    if scope != "site" and not subject:
        raise _invalid("subject", f"a {scope} rule names its {scope}")


def normalize_domain(raw: Any, field_name: str) -> str:
    """One squid ``dstdomain`` entry, normalized like the egress pod's
    allowlist file (``frontend_podman.normalize_allow_domain``)."""
    from umrs.venues.frontend_podman import normalize_allow_domain

    if not isinstance(raw, str):
        raise _invalid(field_name, f"{field_name}: domains are strings")
    domain = normalize_allow_domain(raw)
    if not _DOMAIN_RE.match(domain):
        raise _invalid(field_name, f"{field_name}: {raw!r} is not a domain")
    return domain


def normalize_domains(raw: Any, field_name: str) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        raise _invalid(field_name, f"{field_name} must be a list of domains")
    if len(raw) > MAX_DOMAINS:
        raise _invalid(field_name, f"{field_name}: at most {MAX_DOMAINS} domains")
    out: list[str] = []
    for item in raw:
        domain = normalize_domain(item, field_name)
        if domain not in out:
            out.append(domain)
    return out


def _modes(raw: Any, field_name: str) -> list[str]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise _invalid(field_name, f"{field_name} must be a non-empty list of modes")
    unknown = [m for m in raw if m not in WEB_MODES]
    if unknown:
        raise _invalid(
            field_name, f"{field_name}: unknown mode(s) {unknown}; use {WEB_MODES}"
        )
    return [m for m in WEB_MODES if m in raw]


def _integer(raw: Any, field_name: str, minimum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise _invalid(field_name, f"{field_name} must be an integer or null")
    if raw < minimum:
        raise _invalid(
            field_name,
            f"{field_name} must be at least {minimum} (null = no limit at this layer)",
        )
    return raw


def _field_value(kind: str, raw: Any, field_name: str) -> Any:
    if raw is None:
        return None
    if kind == "modes":
        return _modes(raw, field_name)
    if kind == "domains":
        return normalize_domains(raw, field_name)
    minimum = MIN_LIFETIME_S if kind == "seconds" else MIN_COUNT
    return _integer(raw, field_name, minimum)


def limit_value(value: Any) -> int | None:
    """A count limit as every enforcer reads it (ADR-099 D-1, one rule):
    ``null`` (or absent) is no limit, a positive integer is the limit. ``0``
    and negatives are refused at validation (:func:`validate_rule_doc`), so
    they never come from a rule; one that reaches a reader anyway (a row
    edited by hand) reads as no limit, never as a block."""
    if isinstance(value, bool) or not isinstance(value, int) or value < MIN_COUNT:
        return None
    return value


def _section(name: str, raw: Any, scope: str) -> dict[str, Any]:
    fields = RULE_FIELDS.get(name)
    if fields is None:
        raise _invalid(name, f"unknown policy section {name!r}")
    if not isinstance(raw, dict):
        raise _invalid(name, f"{name} must be an object")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        field_name = f"{name}.{key}"
        if key not in fields:
            raise _invalid(field_name, f"unknown policy field {field_name!r}")
        if scope == "user" and key.endswith("_project"):
            raise _invalid(field_name, f"{field_name} is not a user-rule field")
        out[key] = _field_value(fields[key], value, field_name)
    return out


def validate_rule_doc(doc: Any, *, scope: str, layer: str) -> dict[str, Any]:
    """The normalized rule document, else ``PolicyError("invalid")``."""
    if scope not in SCOPES:
        raise _invalid("scope", f"scope must be one of {', '.join(SCOPES)}")
    if layer not in LAYERS:
        raise _invalid("layer", f"layer must be one of {', '.join(LAYERS)}")
    if layer == "owner" and scope != "project":
        raise _invalid("layer", "an owner rule is a project rule")
    if not isinstance(doc, dict):
        raise _invalid("doc", "a policy rule document is a JSON object")
    return {name: _section(name, raw, scope) for name, raw in doc.items()}


def domain_within(domain: str, cap: Sequence[str]) -> bool:
    """Is ``domain`` inside one cap entry (squid ``dstdomain`` semantics)?

    Inside ``c`` when it equals ``c``, or ``c`` starts with ``.`` and the
    domain is ``c[1:]``, a subdomain of it, or a ``.``-entry under it.
    """
    for entry in cap:
        if domain == entry:
            return True
        if entry.startswith(".") and (domain == entry[1:] or domain.endswith(entry)):
            return True
    return False
