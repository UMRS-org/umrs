"""Status tab pane — server task summaries."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable, Static


class StatusPane(Static):
    """Status tab pane."""

    DEFAULT_CSS = """
    StatusPane { layout: vertical; padding: 0 1; }
    StatusPane > .summary { height: 1; color: $text; }
    StatusPane DataTable { height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._status_data: dict[str, Any] = {}
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="status-table")

    def compose(self):
        yield self._summary_widget
        yield self._table

    def update_data(self, data: Any) -> None:
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        self._status_data = data if isinstance(data, dict) else {}
        self._render_table()
        self._render_summary()

    def _render_table(self) -> None:
        self._table.clear(columns=True)
        self._table.add_column("#", width=3)
        self._table.add_column("Task", width=16)
        self._table.add_column("State", width=10)
        self._table.add_column("Last Update", width=20)
        self._table.add_column("Preview", width=40)
        tasks = self._status_data.get("tasks", [])
        for idx, task in enumerate(tasks):
            state = task.get("state", "unknown")
            preview = str(task.get("preview", ""))[:40]
            self._table.add_row(
                str(idx + 1),
                str(task.get("name", "?")),
                str(state),
                str(task.get("last_update", "?"))[:19],
                preview,
                key=str(task.get("name", idx)),
            )

    def _render_summary(self) -> None:
        client = self._status_data.get("client", "")
        pending = len(self._status_data.get("uncompleted_key_list", []))
        parts = []
        if client:
            parts.append(f"client={client}")
        parts.append(f"pending={pending}")
        self._summary_widget.update("  ".join(parts))
