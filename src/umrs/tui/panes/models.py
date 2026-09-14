"""Models tab pane — registered model catalog."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.widgets import DataTable, Static


class ModelsPane(Static):
    """Models tab pane."""

    DEFAULT_CSS = """
    ModelsPane { layout: vertical; padding: 0 1; }
    ModelsPane > .summary { height: 1; color: $text; }
    ModelsPane DataTable { height: 1fr; }
    ModelsPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    """

    BINDINGS = [
        ("enter", "show_detail", "Detail"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._models: list[dict[str, Any]] = []
        self._selected_name: str | None = None
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="models-table", cursor_type="row")
        self._detail = Static("", classes="detail")

    def compose(self):
        yield self._summary_widget
        yield self._table
        yield self._detail

    def update_data(self, data: Any) -> None:
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        models = data.get("models", []) if isinstance(data, dict) else []
        self._models = models
        self._render_table()
        self._render_summary()
        self._render_detail()

    def _render_table(self) -> None:
        saved_cursor = self._table.cursor_row
        self._table.clear(columns=True)
        self._table.add_column("#", width=3)
        self._table.add_column("Name", width=24)
        self._table.add_column("Type", width=10)
        self._table.add_column("Scope", width=8)
        self._table.add_column("HF ID", width=20)
        self._table.add_column("Registered", width=12)
        for idx, m in enumerate(self._models):
            self._table.add_row(
                str(idx + 1),
                str(m.get("name", "?")),
                str(m.get("model_type", m.get("type", "?"))),
                str(m.get("scope", "?")),
                str(m.get("huggingface_id", "—")),
                str(m.get("registered_at", "?"))[:19],
                key=str(m.get("name", idx)),
            )
        if saved_cursor is not None and self._models:
            self._table.move_cursor(row=min(saved_cursor, len(self._models) - 1))

    def _render_summary(self) -> None:
        total = len(self._models)
        self._summary_widget.update(f"Models: {total}")

    def _render_detail(self) -> None:
        if not self._selected_name:
            self._detail.update("")
            return
        m = next(
            (x for x in self._models if x.get("name") == self._selected_name), None
        )
        if not m:
            self._detail.update("")
            return
        lines = [f"[bold]Model: {self._selected_name}[/bold]"]
        for k in (
            "name",
            "model_type",
            "scope",
            "storage_ref",
            "huggingface_id",
            "description",
            "registered_by",
            "registered_at",
        ):
            v = m.get(k)
            if v is not None:
                lines.append(f"{k}: {v}")
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
