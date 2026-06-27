"""Logs tab pane — client-side activity log viewer."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable, Static


class LogsPane(Static):
    """Logs tab pane."""

    DEFAULT_CSS = """
    LogsPane { layout: vertical; padding: 0 1; }
    LogsPane > .summary { height: 1; color: $text; }
    LogsPane DataTable { height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._entries: list[dict[str, str]] = []
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="logs-table")

    def compose(self):
        yield self._summary_widget
        yield self._table

    def on_mount(self) -> None:
        pass  # columns set up in _render_table

    def update_data(self, data: Any) -> None:
        # Logs are client-side; data is the list of entries.
        if data is None:
            return
        self._entries = data if isinstance(data, list) else []
        self._render_table()
        self._render_summary()

    def append(self, level: str, message: str) -> None:
        import time
        self._entries.append({
            "time": time.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        })
        self._render_table()
        self._render_summary()

    def clear(self) -> None:
        self._entries.clear()
        self._render_table()
        self._render_summary()

    def _render_table(self) -> None:
        self._table.clear(columns=True)
        self._table.add_column("Time", width=10)
        self._table.add_column("Level", width=7)
        self._table.add_column("Message", width=60)
        for e in self._entries:
            self._table.add_row(
                e.get("time", ""),
                e.get("level", "INFO"),
                e.get("message", ""),
            )

    def _render_summary(self) -> None:
        self._summary_widget.update(f"entries={len(self._entries)}")
