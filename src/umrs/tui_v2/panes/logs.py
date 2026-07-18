"""Logs tab pane v2 — client-side log view from status data."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class LogsPane(Static):
    """Logs tab pane v2."""

    DEFAULT_CSS = """
    LogsPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    LogsPane > .content {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        overflow-y: auto;
        font: monospace;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._content = Static("", classes="content")

    def compose(self):
        yield self._content

    def update_data(self, data: Any) -> None:
        self._content.update("[dim]Logs tab — no data fetched[/dim]")
