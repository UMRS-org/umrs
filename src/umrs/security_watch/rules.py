"""Versioned rules pack for the Security Watch (ADR-088 D-4.1, DEC-D61-088W1).

A rules pack is configuration, not code. The default pack ships as package
data (``rules/default.yaml``); a site overlay is a YAML file of the same
shape that changes a default rule by its ``id`` (field by field, so
``{id: x, threshold: 50}`` or ``{id: x, enabled: false}`` is enough) or adds
a new, complete rule. The effective ``rules_version`` names both packs.

Validation is closed: unknown keys, unknown detectors, unknown sources,
events the audit stream does not carry (D-3, telemetry first) and the
watch's own ``security.*`` events are refused. A malformed rule is refused
with :class:`RulePackError`, which names the file and the rule id, so an
operator sees exactly what to fix. Nothing in a rule is ever interpolated
into SQL: :mod:`umrs.security_watch.detectors` maps every field to a fixed
column or a bound parameter.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from umrs.audit.security_events import AUDIT_FAMILIES

DEFAULT_PACK = "rules/default.yaml"
SITE_RULES_ENV = "UMRS_SECURITY_WATCH_RULES"

SEVERITIES = ("info", "low", "medium", "high", "critical")
DETECTORS = ("burst", "match")
SOURCES = ("audit_events", "gateway_audit_log")
RESERVED_PREFIX = "security."

#: group_by values per source; ``details.<field>`` is also allowed on
#: audit_events when every listed event's schema names the field.
GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "audit_events": ("actor", "project_id", "outcome"),
    "gateway_audit_log": ("client_ip", "user_id", "path"),
}

MIN_WINDOW = 60
MAX_WINDOW = 7 * 24 * 3600
MAX_THRESHOLD = 1_000_000
MAX_RULES = 200
MAX_EVENTS = 16
MAX_FILE_BYTES = 256 * 1024

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_DETAILS_PREFIX = "details."

_REQUIRED = ("id", "severity", "source", "detector", "events", "window_seconds")
_OPTIONAL = ("description", "group_by", "threshold", "where", "outcome", "enabled")
_KNOWN_KEYS = frozenset(_REQUIRED + _OPTIONAL)


class RulePackError(ValueError):
    """A rules pack or one of its rules is refused (file and rule id named)."""

    def __init__(self, source: str, rule_id: str, reason: str) -> None:
        self.source = source
        self.rule_id = rule_id
        self.reason = reason
        super().__init__(f"security watch rules {source}: rule {rule_id!r}: {reason}")


@dataclass(frozen=True)
class Rule:
    """One validated detector rule."""

    id: str
    severity: str
    source: str
    detector: str
    events: tuple[str, ...]
    window_seconds: int
    group_by: str = "actor"
    threshold: int = 1
    where: tuple[tuple[str, str], ...] = ()
    outcome: str = ""
    description: str = ""
    enabled: bool = True
    origin: str = "default"


@dataclass(frozen=True)
class RulePack:
    """The effective rules: default pack plus an optional site overlay."""

    rules_version: str
    rules: tuple[Rule, ...] = field(default_factory=tuple)

    def enabled(self) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.enabled)

    def get(self, rule_id: str) -> Rule | None:
        return next((rule for rule in self.rules if rule.id == rule_id), None)


# ── field checks (each raises ValueError with the reason) ──────────────────


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _check_choice(value: Any, choices: tuple[str, ...], name: str) -> str:
    _need(
        isinstance(value, str) and value in choices,
        f"{name} must be one of {list(choices)}",
    )
    return str(value)


def _check_int(value: Any, low: int, high: int, name: str) -> int:
    ok = isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
    _need(ok, f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _check_events(value: Any, source: str) -> tuple[str, ...]:
    ok = isinstance(value, list) and 0 < len(value) <= MAX_EVENTS
    _need(ok, f"events must be a list of 1..{MAX_EVENTS} names")
    for event in value:
        _check_event(event, source)
    return tuple(dict.fromkeys(value))


def _check_event(event: Any, source: str) -> None:
    _need(
        isinstance(event, str) and bool(_CODE_RE.fullmatch(event)),
        f"bad event name {event!r}",
    )
    _need(
        not event.startswith(RESERVED_PREFIX),
        f"{event!r}: the watch never watches its own events",
    )
    if source == "audit_events":
        _need(
            event in AUDIT_FAMILIES,
            f"{event!r} is not an audit family the stream carries",
        )


def _schema_fields(events: tuple[str, ...]) -> set[str]:
    """Details fields every listed event's schema names."""
    sets = [set(AUDIT_FAMILIES.get(event, {})) for event in events]
    return set.intersection(*sets) if sets else set()


def _check_group_by(value: Any, source: str, events: tuple[str, ...]) -> str:
    _need(isinstance(value, str), "group_by must be a string")
    if value in GROUP_COLUMNS[source]:
        return str(value)
    name = str(value).removeprefix(_DETAILS_PREFIX)
    ok = source == "audit_events" and value.startswith(_DETAILS_PREFIX)
    _need(ok and name in _schema_fields(events), f"group_by {value!r} is not allowed")
    return str(value)


def _where_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    _need(
        isinstance(value, (str, int)),
        "where values must be strings, integers or booleans",
    )
    return str(value)


def _check_where(
    value: Any, source: str, events: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    _need(source == "audit_events", "where is only allowed on audit_events")
    _need(
        isinstance(value, Mapping) and len(value) <= 8,
        "where must be a mapping of up to 8 fields",
    )
    allowed = _schema_fields(events)
    for name in value:
        _need(name in allowed, f"where field {name!r} is not in the events' schema")
    return tuple(sorted((str(k), _where_value(v)) for k, v in value.items()))


def _check_threshold(raw: Mapping[str, Any], detector: str) -> int:
    if detector == "match":
        _need("threshold" not in raw, "a match rule takes no threshold")
        return 1
    _need("threshold" in raw, "a burst rule needs a threshold")
    return _check_int(raw["threshold"], 2, MAX_THRESHOLD, "threshold")


def _check_text(
    raw: Mapping[str, Any], key: str, pattern: re.Pattern[str] | None
) -> str:
    value = raw.get(key, "")
    _need(isinstance(value, str) and len(value) <= 200, f"{key} must be a short string")
    if pattern is not None and value:
        _need(bool(pattern.fullmatch(value)), f"{key} {value!r} is malformed")
    return str(value)


def _build_rule(raw: Mapping[str, Any], origin: str) -> Rule:
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    _need(not unknown, f"unknown keys {unknown}")
    missing = [key for key in _REQUIRED if key not in raw]
    _need(not missing, f"missing keys {missing}")
    source = _check_choice(raw["source"], SOURCES, "source")
    detector = _check_choice(raw["detector"], DETECTORS, "detector")
    events = _check_events(raw["events"], source)
    enabled = raw.get("enabled", True)
    _need(isinstance(enabled, bool), "enabled must be a boolean")
    return Rule(
        id=str(raw["id"]),
        severity=_check_choice(raw["severity"], SEVERITIES, "severity"),
        source=source,
        detector=detector,
        events=events,
        window_seconds=_check_int(
            raw["window_seconds"], MIN_WINDOW, MAX_WINDOW, "window_seconds"
        ),
        group_by=_check_group_by(
            raw.get("group_by", _default_group(source)), source, events
        ),
        threshold=_check_threshold(raw, detector),
        where=_check_where(raw.get("where"), source, events),
        outcome=_check_text(raw, "outcome", _CODE_RE),
        description=_check_text(raw, "description", None),
        enabled=enabled,
        origin=origin,
    )


def _default_group(source: str) -> str:
    return GROUP_COLUMNS[source][0]


# ── pack loading ──────────────────────────────────────────────────────────


def _rule_id(raw: Any) -> str:
    rule_id = raw.get("id") if isinstance(raw, Mapping) else None
    return rule_id if isinstance(rule_id, str) and _ID_RE.fullmatch(rule_id) else ""


def _parse_pack(text: str, source: str) -> tuple[str, list[Mapping[str, Any]]]:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RulePackError(
            source, "-", f"not valid YAML ({exc.__class__.__name__})"
        ) from None
    if not isinstance(doc, Mapping) or set(doc) - {"rules_version", "rules"}:
        raise RulePackError(
            source, "-", "a pack is a mapping of rules_version and rules"
        )
    version = doc.get("rules_version")
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise RulePackError(source, "-", "rules_version must be a short version string")
    rules = doc.get("rules") or []
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        raise RulePackError(source, "-", f"rules must be a list of at most {MAX_RULES}")
    return version, _checked_entries(rules, source)


def _checked_entries(rules: list[Any], source: str) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    for index, raw in enumerate(rules):
        rule_id = _rule_id(raw)
        if not rule_id:
            raise RulePackError(
                source, f"#{index}", "every rule needs an id ([a-z0-9-], 2..64)"
            )
        if rule_id in seen:
            raise RulePackError(source, rule_id, "duplicate rule id")
        seen.add(rule_id)
    return list(rules)


def _validated(raw: Mapping[str, Any], source: str, origin: str) -> Rule:
    try:
        return _build_rule(raw, origin)
    except ValueError as exc:
        raise RulePackError(source, str(raw.get("id")), str(exc)) from None


def _default_text() -> str:
    return (
        resources.files(__package__).joinpath(DEFAULT_PACK).read_text(encoding="utf-8")
    )


def _read_site_file(path: Path) -> str:
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RulePackError(
                str(path), "-", f"file larger than {MAX_FILE_BYTES} bytes"
            )
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RulePackError(
            str(path), "-", f"cannot read ({exc.__class__.__name__})"
        ) from None


def _overlay(
    base: dict[str, dict[str, Any]], entries: list[Mapping[str, Any]]
) -> dict[str, tuple[dict[str, Any], str]]:
    """Site entries merged onto the default rules by id (field by field)."""
    merged = {rule_id: (dict(raw), "default") for rule_id, raw in base.items()}
    for raw in entries:
        rule_id = _rule_id(raw)
        previous = merged.get(rule_id, ({}, "site"))[0]
        merged[rule_id] = ({**previous, **raw}, "site")
    return merged


def load_rule_pack(site_path: str | os.PathLike[str] | None = None) -> RulePack:
    """The effective rules pack: the packaged default, plus ``site_path``.

    Raises :class:`RulePackError` naming the file and the rule id when the
    default pack or the site overlay holds a malformed rule.
    """
    version, defaults = _parse_pack(_default_text(), DEFAULT_PACK)
    base = {_rule_id(raw): dict(raw) for raw in defaults}
    if site_path is None:
        rules = [_validated(raw, DEFAULT_PACK, "default") for raw in base.values()]
        return RulePack(rules_version=version, rules=tuple(rules))
    site = Path(site_path)
    site_version, entries = _parse_pack(_read_site_file(site), str(site))
    merged = _overlay(base, entries)
    rules = [
        _validated(raw, str(site) if origin == "site" else DEFAULT_PACK, origin)
        for raw, origin in merged.values()
    ]
    return RulePack(rules_version=f"{version}+site.{site_version}", rules=tuple(rules))


def site_rules_path(env: Mapping[str, str] | None = None) -> Path | None:
    """The site overlay named by ``UMRS_SECURITY_WATCH_RULES``, if any."""
    value = (os.environ if env is None else env).get(SITE_RULES_ENV, "").strip()
    return Path(value) if value else None
