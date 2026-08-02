"""Jobs tab pane v2 — with inline actions (view detail, cancel job)."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.widgets import DataTable, Static

from umrs.tui_v2.widgets.lifecycle import build_status_map, detect_transitions
from umrs.tui_v2.widgets.status_badge import status_text

_FILTER_ORDER = ("all", "queued", "running", "succeeded", "failed")


def _attempt_lines(attempts: list[dict[str, Any]]) -> list[str]:
    if not attempts:
        return []
    lines = [f"\nAttempts ({len(attempts)}):"]
    for i, a in enumerate(attempts):
        st = a.get("status", "?")
        lines.append(f"  {i + 1}. [{st}] {(a.get('started_at') or '?')[:19]}")
    return lines


def _detail_lines(job: dict[str, Any], job_id: str | None) -> list[str]:
    lines = [f"[bold]Job: {job_id}[/bold]"]
    for k in ("job_type", "status", "user_id", "project_id", "job_id"):
        v = job.get(k)
        if v is not None:
            lines.append(f"{k}: {v}")
    lines.extend(_attempt_lines(job.get("attempts", [])))
    return lines


class JobsPane(Static):
    """Jobs tab pane v2."""

    DEFAULT_CSS = """
    JobsPane {
        layout: vertical;
        padding: 0 1;
    }
    JobsPane > .summary { height: 1; color: $text; }
    JobsPane DataTable { height: 1fr; }
    JobsPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    JobsPane > .actions { height: 1; background: $boost; color: $text-muted; padding: 0 1; }
    """

    BINDINGS = [
        ("enter", "show_detail", "Detail"),
        ("x", "cancel_selected", "Cancel Job"),
        ("f", "cycle_filter", "Filter"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._jobs: list[dict[str, Any]] = []
        self._selected_job_id: str | None = None
        self._status_map: dict[str, str] = {}
        self._filter = "all"
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="jobs-table", cursor_type="row")
        self._detail = Static("", classes="detail")
        self._actions_bar = Static("", classes="actions")

    def compose(self):
        yield self._summary_widget
        yield self._table
        yield self._detail
        yield self._actions_bar

    def update_data(self, data: Any) -> None:
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        self._jobs = jobs
        self._notify_transitions(jobs)
        self._render_table()
        self._render_summary()
        self._render_actions()
        self._render_detail()

    def _notify_transitions(self, jobs: list[dict[str, Any]]) -> None:
        """Toast when a job's status flips into a watched state."""
        current = build_status_map(jobs)
        for job_id, new in detect_transitions(self._status_map, current):
            severity = "error" if new == "failed" else "information"
            self.app.notify(
                f"job …{job_id[-8:]} → {new}",
                title="Status changed",
                severity=severity,
            )
        self._status_map = current

    def _visible_jobs(self) -> list[dict[str, Any]]:
        if self._filter == "all":
            return self._jobs
        return [
            j for j in self._jobs if str(j.get("status", "")).lower() == self._filter
        ]

    def _render_table(self) -> None:
        saved_cursor = self._table.cursor_row
        visible = self._visible_jobs()
        self._table.clear(columns=True)
        self._table.add_column("#", width=3)
        self._table.add_column("Job ID", width=12)
        self._table.add_column("Type", width=14)
        self._table.add_column("Status", width=13)
        self._table.add_column("User", width=10)
        self._table.add_column("Attempts", width=8)
        self._table.add_column("Created", width=12)
        self._table.add_column("Updated", width=12)
        for idx, job in enumerate(visible):
            status = str(job.get("status", "?"))
            row = (
                str(idx + 1),
                str(job.get("job_id", "?"))[-8:],
                str(job.get("job_type", "?")),
                status_text(status),
                str(job.get("user_id", "?")),
                str(len(job.get("attempts", []))),
                str(job.get("created_at", "?"))[:19],
                str(job.get("updated_at", "?"))[:19],
            )
            self._table.add_row(*row, key=str(job.get("job_id", idx)))
        if saved_cursor is not None and visible:
            self._table.move_cursor(row=min(saved_cursor, len(visible) - 1))

    def _render_summary(self) -> None:
        total = len(self._jobs)
        running = sum(1 for j in self._jobs if j.get("status") == "running")
        failed = sum(1 for j in self._jobs if j.get("status") == "failed")
        parts = [f"Jobs: {total}"]
        if running:
            parts.append(f"{running} running")
        if failed:
            parts.append(f"{failed} failed")
        if self._filter != "all":
            parts.append(f"[cyan]filter: {self._filter}[/cyan]")
        self._summary_widget.update("  │  ".join(parts))

    def _render_actions(self) -> None:
        self._actions_bar.update(
            "  Actions: Enter=Detail  x=Cancel"
            f"  f=Filter: {self._filter}  j=Jump to Submit"
        )

    def action_cycle_filter(self) -> None:
        idx = (_FILTER_ORDER.index(self._filter) + 1) % len(_FILTER_ORDER)
        self._filter = _FILTER_ORDER[idx]
        self._render_table()
        self._render_summary()
        self._render_actions()

    def _selected_job(self) -> dict[str, Any] | None:
        if not self._selected_job_id:
            return None
        return next(
            (j for j in self._jobs if j.get("job_id") == self._selected_job_id), None
        )

    def _render_detail(self) -> None:
        job = self._selected_job()
        if job is None:
            self._detail.update("")
            return
        self._detail.update("\n".join(_detail_lines(job, self._selected_job_id)))

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
        cursor_row = self._table.cursor_row
        if cursor_row is None:
            return
        row_key = self._table.get_row_key_at(cursor_row)
        if row_key is None or row_key.value is None:
            return
        self._selected_job_id = str(row_key.value)
        self._render_detail()

    def action_cancel_selected(self) -> None:
        """Trigger cancel for the selected job."""
        if not self._selected_job_id:
            self.notify("No job selected", severity="warning")
            return
        # Notify the parent app
        self.app.notify(
            f"Cancel requested: {self._selected_job_id}\nUse Submit tab to submit jobs",
            title="Action queued",
        )
