"""Stats bar widget showing live counts and connection status.

Docked at the top of the screen below the logo, this bar gives the user an
at-a-glance summary of the system state: connection status and live counts
of jobs, endpoints, models, and workflows.
"""

from __future__ import annotations

from textual.widgets import Static

_INIT_TEXT = (
    "[dim]● Unknown[/dim]  │  Jobs: [yellow]0[/yellow]  │  "
    "Endpoints: [yellow]0[/yellow]  │  Models: [yellow]0[/yellow]  │  "
    "Workflows: [yellow]0[/yellow]"
)


class StatsBar(Static):
    """Top-of-screen stats bar with connection status and counts."""

    DEFAULT_CSS = """
    StatsBar {
        dock: top;
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 2;
    }
    """

    def __init__(self) -> None:
        super().__init__(_INIT_TEXT)
        self._connected: bool | None = None
        self._counts: dict[str, int] = {}

    def set_connection(self, connected: bool | None) -> None:
        self._connected = connected
        self._render()

    def set_counts(self, counts: dict[str, int]) -> None:
        self._counts = counts
        self._render()

    def _render(self) -> None:
        parts: list[str] = []
        if self._connected is True:
            parts.append("[green]● Connected[/green]")
        elif self._connected is False:
            parts.append("[red]● Disconnected[/red]")
        else:
            parts.append("[dim]● Unknown[/dim]")
        jobs = self._counts.get("jobs", 0)
        eps = self._counts.get("endpoints", 0)
        models = self._counts.get("models", 0)
        wfs = self._counts.get("workflows", 0)
        parts.append(f"Jobs: [yellow]{jobs}[/yellow]")
        parts.append(f"Endpoints: [yellow]{eps}[/yellow]")
        parts.append(f"Models: [yellow]{models}[/yellow]")
        parts.append(f"Workflows: [yellow]{wfs}[/yellow]")
        self.update("  │  ".join(parts))
