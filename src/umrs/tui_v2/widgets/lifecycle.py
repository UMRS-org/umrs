"""Lifecycle helpers for monitor v2 panes.

Pure functions over plain dicts — no widget state, no Textual dependency —
so stages, transitions, and filtering are directly unit-testable.
"""

from __future__ import annotations

from typing import Any

# Ordered provisioning stages shown in endpoint detail views.
_STAGES: tuple[str, ...] = ("submitted", "provisioning", "serving")

# Raw status → stage index.
_STAGE_BY_STATUS: dict[str, int] = {
    "queued": 0,
    "pending": 0,
    "submitted": 0,
    "provisioning": 1,
    "allocated": 1,
    "starting": 1,
    "running": 2,
    "ready": 2,
    "serving": 2,
    "active": 2,
}

_FAILURE_STATUSES = frozenset({"failed", "error", "lost", "cancelled", "canceled"})
_HALT_STATUSES = frozenset({"stopped", "stopping"})


def stage_tracker(status: str) -> str:
    """Return a rich-markup staged checklist for an endpoint/endpoint-job status.

    Example (provisioning): ``✓ submitted → ◐ provisioning → ○ serving``
    """
    key = (status or "").strip().lower()
    if key in _FAILURE_STATUSES:
        return f"[red]✗ provisioning {key}[/red]"
    if key in _HALT_STATUSES:
        return "[bright_black]○ provisioning stopped[/bright_black]"
    index = _STAGE_BY_STATUS.get(key)
    if index is None:
        return ""
    parts = []
    for pos, label in enumerate(_STAGES):
        if pos < index or (pos == index == len(_STAGES) - 1):
            parts.append(f"[green]✓ {label}[/green]")
        elif pos == index:
            parts.append(f"[cyan]◐ {label}[/cyan]")
        else:
            parts.append(f"[dim]○ {label}[/dim]")
    return " [dim]→[/dim] ".join(parts)


def build_status_map(
    items: list[dict[str, Any]],
    id_key: str = "job_id",
) -> dict[str, str]:
    """Build ``{id: status}`` for transition tracking across refreshes."""
    return {
        str(item[id_key]): str(item.get("status", ""))
        for item in items
        if item.get(id_key)
    }


def detect_transitions(
    previous: dict[str, str],
    current: dict[str, str],
    watch: tuple[str, ...] = ("running", "failed"),
) -> list[tuple[str, str, str]]:
    """Return ``(id, old, new)`` for statuses that changed into a watched state.

    First refresh (empty ``previous``) is treated as the baseline and fires
    nothing — otherwise every row would toast on startup.
    """
    if not previous:
        return []
    events: list[tuple[str, str, str]] = []
    for key, new in current.items():
        old = previous.get(key)
        if old is None or old == new or new not in watch:
            continue
        events.append((key, old, new))
    return events


def filter_by_status(
    items: list[dict[str, Any]],
    status_filter: str,
) -> list[dict[str, Any]]:
    """Filter items by status; ``all`` (or empty) returns everything."""
    key = (status_filter or "all").strip().lower()
    if key == "all":
        return list(items)
    return [item for item in items if str(item.get("status", "")).lower() == key]
