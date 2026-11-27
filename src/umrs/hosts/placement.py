"""Host placement (ADR-098 D-7, D-9; decision SH-D5, slice SH-H2).

The user picks the venue class (``frontend``); placement picks the host from
this site's registry, in three stages, then the least loaded:

1. **resources**: the host's role and launch mode, and its reported
   ``capacity`` (``cpus``, ``memory_gb``, ``max_sessions``) against what the
   sandbox asks for and what already runs there. A capacity key the host
   does not report is no limit.
2. **labels and the isolation floor**: every label the session policy
   requires (ADR-099 ``placement.labels``: ``key`` or ``key=value``), and,
   for a ``data_class=sensitive`` project, ``isolation.egress == "nft"`` or a
   sensitive tier the *site* declares for its inline host. A host reporting
   nothing is ``crun`` / ``squid-only`` (reported, never assumed: D-7); an
   agent host cannot declare the site tier for itself.
3. **least loaded**: fewest live sessions, then host id (deterministic).

A pin (``cpu:<host>``, an admin or project choice) is validated against
``site.yaml`` and the registry by the caller and here: an undeclared,
unregistered, draining or down pin, or one that fails a stage, is refused
``host_unavailable``; it is never re-placed on another host. No eligible
host is ``no_host``. Callers map both to their own error family.

Pure: no database, no extension import (ADR-117 D-3). The registry rows and
the per-host load come in as arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from umrs.hosts.registry import PLACEABLE_STATES, HostRecord
from umrs.hosts.runtime import LAUNCH_MODES

HOST_UNAVAILABLE = "host_unavailable"
NO_HOST = "no_host"

#: The isolation a host reporting nothing gets (D-7: never assume better).
DEFAULT_ISOLATION: dict[str, str] = {"runtime": "crun", "egress": "squid-only"}
#: ``isolation.tier`` value a site declares for its inline front (D-7 "a front
#: tier the site declares"); honoured on ``inline`` hosts only.
SENSITIVE_TIER = "sensitive"

STAGES = ("role", "launch_mode", "resources", "labels", "isolation")


class PlacementRefused(Exception):
    """No host for this sandbox; ``code`` is ``host_unavailable`` or ``no_host``."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = dict(detail)


@dataclass(frozen=True)
class PlacementRequest:
    """What one sandbox needs from its host."""

    role: str = "session"
    #: A pinned host id (``cpu:<host>`` without the prefix); "" = place.
    pin: str = ""
    #: Declared pinnable hosts (``site.yaml``); consulted only with a pin.
    declared: frozenset[str] = frozenset()
    cpus: float = 0.0
    memory_gb: float = 0.0
    #: Required labels, ``key`` or ``key=value`` (ADR-099 ``placement.labels``).
    labels: tuple[str, ...] = ()
    sensitive: bool = False
    #: Launch modes this server can drive (``agent`` arrives with SH-H3).
    launch_modes: tuple[str, ...] = LAUNCH_MODES


# ── parsing what hosts report ──────────────────────────────────────────────


def _number(value: Any) -> float | None:
    """A reported capacity number; absent or unreadable = no limit."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def parse_cpus(value: Any) -> float:
    """``"8"`` / ``8`` / ``"1.5"`` → float; anything else → 0 (asks nothing)."""
    return _number(value) or 0.0


def parse_memory_gb(value: Any) -> float:
    """A podman ``--memory`` size (``16g``, ``512m``, ``1t``, bytes) in GiB."""
    text = str(value or "").strip().lower()
    units = {"k": 1 / 1024**2, "m": 1 / 1024, "g": 1.0, "t": 1024.0}
    scale = units.get(text[-1:]) if text else None
    raw = text[:-1] if scale is not None else text
    number = _number(raw)
    if number is None:
        return 0.0
    return number * scale if scale is not None else number / 1024**3


def effective_isolation(host: HostRecord) -> dict[str, str]:
    """The host's reported isolation over :data:`DEFAULT_ISOLATION`."""
    reported = {k: str(v) for k, v in (host.isolation or {}).items() if v}
    return {**DEFAULT_ISOLATION, **reported}


# ── stages ─────────────────────────────────────────────────────────────────


def _role_problem(host: HostRecord, req: PlacementRequest, load: int) -> str:
    return "" if req.role in host.roles else f"no {req.role} role"


def _mode_problem(host: HostRecord, req: PlacementRequest, load: int) -> str:
    if host.launch_mode in req.launch_modes:
        return ""
    return f"launch mode {host.launch_mode} is not available on this server"


def _over(capacity: Mapping[str, Any], key: str, wanted: float) -> bool:
    limit = _number(capacity.get(key))
    return limit is not None and wanted > limit


def _resource_problem(host: HostRecord, req: PlacementRequest, load: int) -> str:
    capacity = host.capacity or {}
    if _over(capacity, "cpus", req.cpus):
        return f"needs {req.cpus:g} cpus, host has {capacity['cpus']}"
    if _over(capacity, "memory_gb", req.memory_gb):
        return f"needs {req.memory_gb:g} GiB, host has {capacity['memory_gb']}"
    if _over(capacity, "max_sessions", load + 1):
        return f"full ({load} of {capacity['max_sessions']} sessions)"
    return ""


def label_matches(labels: Mapping[str, Any], required: str) -> bool:
    """``key`` = the label is set (not false/empty); ``key=value`` = equal."""
    key, sep, value = required.partition("=")
    if key not in labels:
        return False
    have = labels[key]
    if sep:
        text = str(have).lower() if isinstance(have, bool) else str(have)
        return text == value
    return have not in (False, None, "")


def _label_problem(host: HostRecord, req: PlacementRequest, load: int) -> str:
    missing = [r for r in req.labels if not label_matches(host.labels or {}, r)]
    return f"missing label(s) {', '.join(missing)}" if missing else ""


def meets_isolation_floor(host: HostRecord) -> bool:
    """May a ``data_class=sensitive`` session land here (D-7)?"""
    isolation = effective_isolation(host)
    if isolation["egress"] == "nft":
        return True
    return host.launch_mode == "inline" and isolation.get("tier") == SENSITIVE_TIER


def _isolation_problem(host: HostRecord, req: PlacementRequest, load: int) -> str:
    if not req.sensitive or meets_isolation_floor(host):
        return ""
    isolation = effective_isolation(host)
    return (
        f"isolation {isolation['runtime']}/{isolation['egress']} is below the "
        "floor of a sensitive project (egress=nft or a site-declared "
        "sensitive tier)"
    )


_CHECKS = (
    ("role", _role_problem),
    ("launch_mode", _mode_problem),
    ("resources", _resource_problem),
    ("labels", _label_problem),
    ("isolation", _isolation_problem),
)


def host_problem(host: HostRecord, req: PlacementRequest, load: int) -> tuple[str, str]:
    """``(stage, reason)`` of the first stage ``host`` fails, ``("", "")``."""
    for stage, check in _CHECKS:
        reason = check(host, req, load)
        if reason:
            return stage, reason
    return "", ""


# ── the inline host's profile (site-declared) ──────────────────────────────

_PROFILE_KEYS = ("labels", "capacity", "isolation")


def inline_host_profile(extra: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``labels``/``capacity``/``isolation`` of the front's own host, from
    ``site.yaml`` ``extra.hosts.inline`` (each a mapping; anything else is
    ignored). Isolation defaults to :data:`DEFAULT_ISOLATION`: the inline
    front is ``squid-only`` unless the site declares otherwise (ADR-098
    host contract)."""
    hosts = extra.get("hosts") if isinstance(extra, Mapping) else None
    inline = hosts.get("inline") if isinstance(hosts, Mapping) else None
    inline = inline if isinstance(inline, Mapping) else {}
    profile = {
        key: dict(inline[key]) if isinstance(inline.get(key), Mapping) else {}
        for key in _PROFILE_KEYS
    }
    profile["isolation"] = {**DEFAULT_ISOLATION, **profile["isolation"]}
    return profile


# ── placement ──────────────────────────────────────────────────────────────


def _load_of(load: Mapping[str, int], host_id: str) -> int:
    return int(load.get(host_id, 0) or 0)


def _pinned(
    hosts: Sequence[HostRecord], req: PlacementRequest, load: Mapping[str, int]
) -> HostRecord:
    pin = req.pin
    if pin not in req.declared:
        raise PlacementRefused(
            HOST_UNAVAILABLE,
            f"host {pin!r} is not declared at this site "
            "(site.yaml extra.venues.cpu_hosts)",
            pin=pin,
            reason="undeclared",
        )
    host = next((h for h in hosts if h.host_id == pin), None)
    if host is None or host.state not in PLACEABLE_STATES:
        state = host.state if host is not None else "unregistered"
        raise PlacementRefused(
            HOST_UNAVAILABLE,
            f"pinned host {pin!r} is {state}; a pinned session is never "
            "placed elsewhere",
            pin=pin,
            reason=state,
        )
    stage, reason = host_problem(host, req, _load_of(load, pin))
    if stage:
        raise PlacementRefused(
            HOST_UNAVAILABLE,
            f"pinned host {pin!r} refused ({stage}): {reason}",
            pin=pin,
            reason=stage,
        )
    return host


def _no_host(refused: dict[str, int], considered: int) -> PlacementRefused:
    if not considered:
        why = "no active host is registered at this site"
    else:
        why = "every active host was refused: " + ", ".join(
            f"{stage} ({count})" for stage, count in refused.items()
        )
    return PlacementRefused(
        NO_HOST,
        f"no host for this session: {why}",
        considered=considered,
        refused=refused,
    )


def place(
    hosts: Sequence[HostRecord],
    req: PlacementRequest,
    load: Mapping[str, int] | None = None,
) -> HostRecord:
    """The host this sandbox runs on, else :class:`PlacementRefused`.

    ``hosts`` are this site's registry rows (every state: a pin to a
    draining or down host is named as such); ``load`` maps host id to live
    sessions."""
    load = load or {}
    if req.pin:
        return _pinned(hosts, req, load)
    active = [h for h in hosts if h.state in PLACEABLE_STATES]
    refused: dict[str, int] = {}
    eligible: list[HostRecord] = []
    for host in active:
        stage, _ = host_problem(host, req, _load_of(load, host.host_id))
        if stage:
            refused[stage] = refused.get(stage, 0) + 1
        else:
            eligible.append(host)
    if not eligible:
        raise _no_host(refused, len(active))
    return min(eligible, key=lambda h: (_load_of(load, h.host_id), h.host_id))


__all__ = [
    "DEFAULT_ISOLATION",
    "HOST_UNAVAILABLE",
    "NO_HOST",
    "SENSITIVE_TIER",
    "STAGES",
    "PlacementRefused",
    "PlacementRequest",
    "effective_isolation",
    "host_problem",
    "inline_host_profile",
    "label_matches",
    "meets_isolation_floor",
    "parse_cpus",
    "parse_memory_gb",
    "place",
]
