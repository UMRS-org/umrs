"""Coloured status pill widget.

Renders a UMRS job/endpoint status as a coloured ``Static`` label.  The
mapping mirrors the legacy dashboard's :data:`STATUS_STYLES` table.
"""

from __future__ import annotations

from textual.widgets import Static

# Mirror the legacy STATUS_STYLES mapping from dashboard.py.
STATUS_STYLES: dict[str, str] = {
    "running": "bold yellow",
    "queued": "cyan",
    "planned": "dim cyan",
    "completed": "green",
    "failed": "bold red",
    "cancelled": "dim",
}


def status_style(status: str) -> str:
    return STATUS_STYLES.get(status, "")


class StatusBadge(Static):
    """A compact coloured status label."""

    DEFAULT_CSS = """
    StatusBadge {
        padding: 0 1;
        width: auto;
    }
    """

    def __init__(self, status: str) -> None:
        super().__init__()
        self._status = status

    def render(self) -> str:
        style = status_style(self._status)
        return f"[{style}]{self._status}[/{style}]" if style else self._status
