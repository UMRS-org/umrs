"""Errors tab pane v2."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class ErrorsPane(Static):
    """Errors tab pane v2."""

    DEFAULT_CSS = """
    ErrorsPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    ErrorsPane > .content {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._content = Static("", classes="content")

    def compose(self):
        yield self._content

    def update_data(self, data: Any) -> None:
        if data is None or not isinstance(data, dict):
            self._content.update("[dim]No error data[/dim]")
            return
        error_logs = data.get("error_logs", [])
        if not error_logs:
            self._content.update("[dim]No errors found[/dim]")
            return
        lines = [f"[bold]Errors ({len(error_logs)})[/bold]"]
        for entry in error_logs[:100]:
            ts = (entry.get("timestamp") or "")[:19]
            level = entry.get("level", "?")
            msg = entry.get("message", "?")
            src = entry.get("source", "")
            lines.append(f"[{ts}] {level} [{src}] {msg}")
        self._content.update("\n".join(lines))
