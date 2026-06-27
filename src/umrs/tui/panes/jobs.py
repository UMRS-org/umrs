"""Jobs tab pane — durable job queue with sortable DataTable + detail panel."""

from __future__ import annotations

import json
from typing import Any

from textual import on
from textual.widgets import DataTable, Static

from ..widgets.status_badge import status_style


class JobsPane(Static):
    """Jobs tab pane."""

    DEFAULT_CSS = """
    JobsPane {
        layout: vertical;
        padding: 0 1;
    }
    JobsPane > .summary { height: 1; color: $text; }
    JobsPane DataTable { height: 1fr; }
    JobsPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    """

    BINDINGS = [
        ("enter", "show_detail", "Detail"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._jobs: list[dict[str, Any]] = []
        self._filter_status: str | None = None
        self._filter_project: str | None = None
        self._filter_job_type: str | None = None
        self._selected_job_id: str | None = None
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="jobs-table", cursor_type="row")
        self._detail = Static("", classes="detail")

    def compose(self):
        yield self._summary_widget
        yield self._table
        yield self._detail

    def update_data(self, data: Any) -> None:
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        self._jobs = jobs
        self._render_table()
        self._render_summary()
        self._render_detail()

    def _render_table(self) -> None:
        self._table.clear(columns=True)
        self._table.add_column("#", width=3)
        self._table.add_column("Job ID", width=12)
        self._table.add_column("Type", width=14)
        self._table.add_column("Status", width=10)
        self._table.add_column("User", width=10)
        self._table.add_column("Attempts", width=8)
        self._table.add_column("Created", width=12)
        self._table.add_column("Updated", width=12)
        for idx, job in enumerate(self._jobs):
            row = (
                str(idx + 1),
                str(job.get("job_id", "?"))[-8:],
                str(job.get("job_type", "?")),
                str(job.get("status", "?")),
                str(job.get("user_id", "?")),
                str(len(job.get("attempts", []))),
                str(job.get("created_at", "?"))[:19],
                str(job.get("updated_at", "?"))[:19],
            )
            self._table.add_row(*row, key=str(job.get("job_id", idx)))

    def _render_summary(self) -> None:
        total = len(self._jobs)
        running = sum(1 for j in self._jobs if j.get("status") == "running")
        failed = sum(1 for j in self._jobs if j.get("status") == "failed")
        parts = [f"Jobs: {total}"]
        if running:
            parts.append(f"{running} running")
        if failed:
            parts.append(f"{failed} failed")
        self._summary_widget.update("  │  ".join(parts))

    def _render_detail(self) -> None:
        if not self._selected_job_id:
            self._detail.update("")
            return
        job = next((j for j in self._jobs if j.get("job_id") == self._selected_job_id), None)
        if not job:
            self._detail.update("")
            return
        lines = [f"[bold]Job {self._selected_job_id}[/bold]"]
        lines.append(f"Type: {job.get('job_type', '?')}")
        lines.append(f"Status: {job.get('status', '?')}")
        lines.append(f"Project: {job.get('project_id', '?')}")
        lines.append(f"Policy: {job.get('restart_policy', '?')}")
        lines.append(f"Max attempts: {job.get('desired_max_attempts', '?')}")
        spec_str = json.dumps(job.get("spec", {}), indent=2)
        if len(spec_str) > 300:
            spec_str = spec_str[:300] + "..."
        lines.append(f"Spec:\n{spec_str}")
        attempts = job.get("attempts", [])
        if attempts:
            lines.append("\nAttempts:")
            for att in attempts:
                lines.append(
                    f"  #{att.get('attempt_no', '?')} "
                    f"{att.get('status', '?')} "
                    f"{att.get('started_at', '?')[:19]} → "
                    f"{att.get('ended_at', '?')[:19]}"
                )
        self._detail.update("\n".join(lines))

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self._selected_job_id = event.row_key.value
            self._render_detail()

    @on(DataTable.RowHighlighted)
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value:
            self._selected_job_id = event.row_key.value
            self._render_detail()

    def action_show_detail(self) -> None:
        """Show detail for the currently highlighted row when Enter is pressed."""
        cursor_row = self._table.cursor_row
        if cursor_row is None:
            return
        row_key = self._table.get_row_key_at(cursor_row)
        if row_key is None or row_key.value is None:
            return
        self._selected_job_id = str(row_key.value)
        self._render_detail()
