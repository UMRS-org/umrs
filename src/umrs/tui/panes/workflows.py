"""Workflows tab pane — registered workflow registry."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.widgets import DataTable, Static


class WorkflowsPane(Static):
    """Workflows tab pane."""

    DEFAULT_CSS = """
    WorkflowsPane { layout: vertical; padding: 0 1; }
    WorkflowsPane > .summary { height: 1; color: $text; }
    WorkflowsPane DataTable { height: 1fr; }
    WorkflowsPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    """

    BINDINGS = [
        ("enter", "show_detail", "Detail"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._workflows: list[dict[str, Any]] = []
        self._selected_name: str | None = None
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="workflows-table", cursor_type="row")
        self._detail = Static("", classes="detail")

    def compose(self):
        yield self._summary_widget
        yield self._table
        yield self._detail

    def update_data(self, data: Any) -> None:
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        wfs = data.get("workflows", []) if isinstance(data, dict) else []
        self._workflows = wfs
        self._render_table()
        self._render_summary()
        self._render_detail()

    def _render_table(self) -> None:
        saved_cursor = self._table.cursor_row
        self._table.clear(columns=True)
        self._table.add_column("#", width=3)
        self._table.add_column("Name", width=24)
        self._table.add_column("Version", width=10)
        self._table.add_column("Scope", width=8)
        self._table.add_column("Owner", width=12)
        self._table.add_column("Steps", width=6)
        self._table.add_column("Registered", width=12)
        for idx, wf in enumerate(self._workflows):
            steps = wf.get("steps", [])
            self._table.add_row(
                str(idx + 1),
                str(wf.get("name", "?")),
                str(wf.get("version", "?")),
                str(wf.get("scope", "?")),
                str(wf.get("user_id", "—")),
                str(len(steps)),
                str(wf.get("registered_at", "?"))[:19],
                key=str(wf.get("name", idx)),
            )
        if saved_cursor is not None and self._workflows:
            self._table.move_cursor(row=min(saved_cursor, len(self._workflows) - 1))

    def _render_summary(self) -> None:
        total = len(self._workflows)
        self._summary_widget.update(f"Workflows: {total}")

    def _render_detail(self) -> None:
        if not self._selected_name:
            self._detail.update("")
            return
        wf = next((w for w in self._workflows if w.get("name") == self._selected_name), None)
        if not wf:
            self._detail.update("")
            return
        lines = [f"[bold]Workflow: {self._selected_name}[/bold]"]
        for k in ("name", "version", "scope", "user_id", "source_path",
                  "description", "registered_at"):
            v = wf.get(k)
            if v is not None:
                lines.append(f"{k}: {v}")
        steps = wf.get("steps", [])
        if steps:
            lines.append("\nSteps:")
            for i, s in enumerate(steps):
                lines.append(f"  {i + 1}. {s.get('name', '?')} — {s.get('entrypoint', '?')}")
        self._detail.update("\n".join(lines))

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self._selected_name = event.row_key.value
            self._render_detail()

    @on(DataTable.RowHighlighted)
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value:
            self._selected_name = event.row_key.value
            self._render_detail()

    def action_show_detail(self) -> None:
        """Show detail for the currently highlighted row when Enter is pressed."""
        cursor_row = self._table.cursor_row
        if cursor_row is None:
            return
        row_key = self._table.get_row_key_at(cursor_row)
        if row_key is None or row_key.value is None:
            return
        self._selected_name = str(row_key.value)
        self._render_detail()
