"""Venue placement vocabulary + declaration gates (D-48/D-49, ADR-060/061).

One grammar for "where does this run" shared by session venues and MCP
registry placements:

    local | frontend | cluster:<name> | cpu:<host> | trusted:<host>

cpu:<host> and trusted:<host> are valid only when declared in
``site.yaml`` under ``extra.venues.cpu_hosts`` / ``extra.venues.trusted_hosts``
— an undeclared host is refused by name, fail-closed when the site config
is unreadable (never a silent frontend/session-local placement).
"""

from __future__ import annotations

import re

PLACEMENT_RE = re.compile(
    r"^(local|frontend|cluster:[a-z0-9][a-z0-9-]{0,62}|"
    r"cpu:[a-z0-9][a-z0-9.-]{0,62}|"
    r"trusted:[a-z0-9][a-z0-9.-]{0,62})$"
)

TRUST_CLASSES = frozenset({"local-safe", "front-end", "trusted-host-only"})


class PlacementError(ValueError):
    """Bad placement grammar or undeclared host."""


def _site_venue_hosts(key: str) -> set[str]:
    """Declared venue hosts from site config (fail-closed: empty on error)."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        cfg = load_site_config()
    except Exception:  # noqa: BLE001 — fail-closed placement
        return set()
    venues = cfg.extra.get("venues") if isinstance(cfg.extra, dict) else None
    hosts = (venues or {}).get(key) if isinstance(venues, dict) else None
    if not isinstance(hosts, list):
        return set()
    return {str(h).strip() for h in hosts if str(h).strip()}


def declared_cpu_hosts() -> set[str]:
    """site.yaml extra.venues.cpu_hosts."""
    return _site_venue_hosts("cpu_hosts")


def declared_trusted_hosts() -> set[str]:
    """site.yaml extra.venues.trusted_hosts."""
    return _site_venue_hosts("trusted_hosts")


def validate_placement(placement: str) -> str:
    """Validate grammar + declaration. Returns the placement on success."""
    if not PLACEMENT_RE.match(placement or ""):
        raise PlacementError(
            f"invalid placement {placement!r}: expected local | frontend | "
            "cluster:<name> | cpu:<host> | trusted:<host>"
        )
    for prefix, hosts in (
        ("cpu:", declared_cpu_hosts()),
        ("trusted:", declared_trusted_hosts()),
    ):
        if placement.startswith(prefix):
            host = placement[len(prefix) :]
            if host not in hosts:
                raise PlacementError(
                    f"{prefix.rstrip(':')} host {host!r} is not declared "
                    f"(site.yaml extra.venues.{prefix.rstrip(':')}_hosts)"
                )
    return placement


def validate_trust_class(placement: str, trust_class: str) -> str:
    """Cross-check a trust class against its placement (ADR-061 map)."""
    if trust_class not in TRUST_CLASSES:
        raise PlacementError(
            f"invalid trust_class {trust_class!r}: expected one of {sorted(TRUST_CLASSES)}"
        )
    if trust_class == "trusted-host-only" and not placement.startswith("trusted:"):
        raise PlacementError(
            f"trust_class 'trusted-host-only' requires placement trusted:<host>, "
            f"got {placement!r}"
        )
    if trust_class == "front-end" and (placement.startswith(("cpu:", "trusted:"))):
        raise PlacementError(
            f"trust_class 'front-end' forbids placement {placement!r} "
            "(local | frontend only)"
        )
    return trust_class
