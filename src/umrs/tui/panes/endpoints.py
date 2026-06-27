"""Endpoints tab pane — endpoint provisioning jobs with sortable DataTable."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.widgets import DataTable, Static


class EndpointsPane(Static):
    """Endpoints tab pane."""

    DEFAULT_CSS = """
    EndpointsPane {
        layout: vertical;
        padding: 0 1;
    }
    EndpointsPane > .summary { height: 1; color: $text; }
    EndpointsPane DataTable { height: 1fr; }
    EndpointsPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    """

    BINDINGS = [
        ("enter", "show_detail", "Detail"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._endpoints: list[dict[str, Any]] = []
        self._selected_id: str | None = None
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="endpoints-table", cursor_type="row")
        self._detail = Static("", classes="detail")

    def compose(self):
        yield self._summary_widget
        yield self._table
        yield self._detail

    def update_data(self, data: Any) -> None:
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        eps = data.get("endpoints", []) if isinstance(data, dict) else []
        self._endpoints = eps
        self._render_table()
        self._render_summary()
        self._render_detail()

    def _render_table(self) -> None:
        self._table.clear(columns=True)
        self._table.add_column("#", width=3)
        self._table.add_column("Job ID", width=12)
        self._table.add_column("Status", width=10)
        self._table.add_column("Model", width=24)
        self._table.add_column("Idle TTL", width=9)
        self._table.add_column("Backend", width=12)
        self._table.add_column("Created", width=12)
        for idx, ep in enumerate(self._endpoints):
            spec = ep.get("spec") or {}
            backend = ""
            for att in reversed(ep.get("attempts") or []):
                if att.get("backend_ref"):
                    backend = str(att["backend_ref"])
                    break
            self._table.add_row(
                str(idx + 1),
                str(ep.get("job_id", ""))[-8:],
                str(ep.get("status", "?")),
                str(spec.get("model", "—")),
                str(spec.get("idle_ttl_sec", "—")),
                backend or "—",
                str(ep.get("created_at", "?"))[:19],
                key=str(ep.get("job_id", idx)),
            )

    def _render_summary(self) -> None:
        total = len(self._endpoints)
        running = sum(1 for e in self._endpoints if e.get("status") == "running")
        parts = [f"Endpoints: {total}"]
        if running:
            parts.append(f"{running} running")
        self._summary_widget.update("  │  ".join(parts))

    def _render_detail(self) -> None:
        if not self._selected_id:
            self._detail.update("")
            return
        ep = next((e for e in self._endpoints if e.get("job_id") == self._selected_id), None)
        if not ep:
            self._detail.update("")
            return
        spec = ep.get("spec") or {}
        lines = [f"[bold]Endpoint {self._selected_id}[/bold]"]
        lines.append(f"Status: {ep.get('status', '?')}")
        lines.append(f"Project: {ep.get('project_id', '?')}")
        lines.append(f"Model: {spec.get('model', '—')}")
        lines.append(f"Model path: {spec.get('model_path', '—')}")
        lines.append(f"Idle TTL: {spec.get('idle_ttl_sec', '—')}")
        lines.append(f"Tier: {spec.get('tier', '—')}")
        lines.append(f"GPU count: {spec.get('gpu_count', '—')}")
        lines.append(f"Max model len: {spec.get('max_model_len', '—')}")
        attempts = ep.get("attempts") or []
        if attempts:
            latest = attempts[-1]
            lines.append(f"\nLatest attempt #{latest.get('attempt_no', '?')}:")
            lines.append(f"  Status: {latest.get('status', '—')}")
            lines.append(f"  Backend ref: {latest.get('backend_ref', '—')}")
            lines.append(f"  Started: {latest.get('started_at', '—')}")
        self._detail.update("\n".join(lines))

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self._selected_id = event.row_key.value
            self._render_detail()

    @on(DataTable.RowHighlighted)
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value:
            self._selected_id = event.row_key.value
            self._render_detail()

    def action_show_detail(self) -> None:
        """Show detail for the currently highlighted row when Enter is pressed."""
        cursor_row = self._table.cursor_row
        if cursor_row is None:
            return
        row_key = self._table.get_row_key_at(cursor_row)
        if row_key is None or row_key.value is None:
            return
        self._selected_id = str(row_key.value)
        self._render_detail()
