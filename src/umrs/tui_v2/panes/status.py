"""Status tab pane v2 — server status display."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class StatusPane(Static):
    """Status tab pane v2."""

    DEFAULT_CSS = """
    StatusPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    StatusPane > .content {
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
        if data is None:
            self._content.update("[red]No status data[/red]")
            return
        if isinstance(data, dict):
            lines = self._format_status(data)
            self._content.update("\n".join(lines))
        else:
            self._content.update(str(data))

    def _format_status(self, data: dict) -> list[str]:
        lines = []
        summary = data.get("summary", {}) or {}
        summaries = data.get("summaries", {}) or {}

        if summary:
            lines.append("[bold]Global Summary[/bold]")
            for k, v in summary.items():
                if isinstance(v, str) and len(v) > 200:
                    lines.append(f"  {k}: {v[:200]}...")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

        if summaries:
            lines.append(f"[bold]Per-User Summaries ({len(summaries)} users)[/bold]")
            for user, entry in list(summaries.items())[:20]:
                if isinstance(entry, dict):
                    status = entry.get("status", "?")
                    updated = (
                        (entry.get("last_updated_at") or "")[:19]
                        if entry.get("last_updated_at")
                        else ""
                    )
                    lines.append(f"  {user}: [{status}] {updated}")
        else:
            lines.append("  [dim](no per-user summaries)[/dim]")

        return lines
