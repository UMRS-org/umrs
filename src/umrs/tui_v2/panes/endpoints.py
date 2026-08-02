"""Endpoints tab pane v2 — with inline actions (provision, stop)."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.widgets import DataTable, Static

from umrs.tui_v2.widgets.lifecycle import (
    build_status_map,
    detect_transitions,
    stage_tracker,
)
from umrs.tui_v2.widgets.status_badge import status_text

_FILTER_ORDER = ("all", "queued", "provisioning", "running", "failed")


def _fmt_duration(seconds: int | None) -> str:
    if not seconds:
        return ""
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _attempt_lines(attempts: list[dict[str, Any]]) -> list[str]:
    if not attempts:
        return []
    latest = attempts[-1]
    lines = [f"\nLatest attempt: {latest.get('status', '?')}"]
    if latest.get("backend_ref"):
        lines.append(f"  Backend: {latest['backend_ref']}")
    port = latest.get("outputs", {}).get("endpoint_port")
    if port:
        lines.append(f"  Port: {port}")
    return lines


def _detail_lines(ep: dict[str, Any], job_id: str | None) -> list[str]:
    spec = ep.get("spec", {}) or {}
    lines = [f"[bold]Endpoint: {job_id}[/bold]"]
    tracker = stage_tracker(str(ep.get("status", "")))
    if tracker:
        lines.append(tracker)
    for k in ("status", "user_id", "project_id"):
        v = ep.get(k)
        if v is not None:
            lines.append(f"{k}: {v}")
    for k in ("model", "idle_ttl_sec", "model_path", "gpu_count", "max_model_len"):
        v = spec.get(k)
        if v is not None:
            lines.append(f"{k}: {v}")
    lines.extend(_attempt_lines(ep.get("attempts", [])))
    return lines


class EndpointsPane(Static):
    """Endpoints tab pane v2."""

    DEFAULT_CSS = """
    EndpointsPane {
        layout: vertical;
        padding: 0 1;
    }
    EndpointsPane > .summary { height: 1; color: $text; }
    EndpointsPane DataTable { height: 1fr; }
    EndpointsPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    EndpointsPane > .actions { height: 1; background: $boost; color: $text-muted; padding: 0 1; }
    """

    BINDINGS = [
        ("enter", "show_detail", "Detail"),
        ("s", "stop_selected", "Stop Endpoint"),
        ("f", "cycle_filter", "Filter"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._endpoints: list[dict[str, Any]] = []
        self._selected_job_id: str | None = None
        self._status_map: dict[str, str] = {}
        self._filter = "all"
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="endpoints-table", cursor_type="row")
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
        endpoints = data.get("endpoints", []) or data.get("jobs", [])
        self._endpoints = endpoints
        self._notify_transitions(endpoints)
        self._render_table()
        self._render_summary()
        self._render_actions()
        self._render_detail()

    def _notify_transitions(self, endpoints: list[dict[str, Any]]) -> None:
        """Toast when an endpoint's status flips into a watched state."""
        current = build_status_map(endpoints)
        for job_id, new in detect_transitions(self._status_map, current):
            severity = "error" if new == "failed" else "information"
            self.app.notify(
                f"endpoint …{job_id[-8:]} → {new}",
                title="Status changed",
                severity=severity,
            )
        self._status_map = current

    def _visible_endpoints(self) -> list[dict[str, Any]]:
        if self._filter == "all":
            return self._endpoints
        return [
            ep
            for ep in self._endpoints
            if str(ep.get("status", "")).lower() == self._filter
        ]

    def _render_table(self) -> None:
        saved_cursor = self._table.cursor_row
        visible = self._visible_endpoints()
        self._table.clear(columns=True)
        self._table.add_column("#", width=3)
        self._table.add_column("Job ID", width=12)
        self._table.add_column("Status", width=13)
        self._table.add_column("Model", width=24)
        self._table.add_column("Idle TTL", width=8)
        self._table.add_column("Backend", width=16)
        for idx, ep in enumerate(visible):
            spec = ep.get("spec", {}) or {}
            status = ep.get("status", "")
            # Find backend_ref from latest attempt
            backend = ""
            for attempt in reversed(ep.get("attempts", [])):
                br = attempt.get("backend_ref")
                if br:
                    backend = str(br)
                    break
            row = (
                str(idx + 1),
                str(ep.get("job_id", "?"))[-8:],
                status_text(str(status)),
                str(spec.get("model", "?")),
                _fmt_duration(spec.get("idle_ttl_sec")),
                backend or "-",
            )
            self._table.add_row(*row, key=str(ep.get("job_id", idx)))
        if saved_cursor is not None and visible:
            self._table.move_cursor(row=min(saved_cursor, len(visible) - 1))

    def _render_summary(self) -> None:
        total = len(self._endpoints)
        running = sum(1 for e in self._endpoints if e.get("status") == "running")
        parts = [f"Endpoints: {total}"]
        if running:
            parts.append(f"{running} running")
        if self._filter != "all":
            parts.append(f"[cyan]filter: {self._filter}[/cyan]")
        self._summary_widget.update("  │  ".join(parts))

    def _render_actions(self) -> None:
        self._actions_bar.update(
            "  Actions: Enter=Detail  s=Stop"
            f"  f=Filter: {self._filter}  → Submit tab for provisioning"
        )

    def action_cycle_filter(self) -> None:
        idx = (_FILTER_ORDER.index(self._filter) + 1) % len(_FILTER_ORDER)
        self._filter = _FILTER_ORDER[idx]
        self._render_table()
        self._render_summary()
        self._render_actions()

    def _selected_endpoint(self) -> dict[str, Any] | None:
        if not self._selected_job_id:
            return None
        return next(
            (e for e in self._endpoints if e.get("job_id") == self._selected_job_id),
            None,
        )

    def _render_detail(self) -> None:
        ep = self._selected_endpoint()
        if ep is None:
            self._detail.update("")
            return
        self._detail.update("\n".join(_detail_lines(ep, self._selected_job_id)))

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

    def action_stop_selected(self) -> None:
        """Trigger stop for the selected endpoint."""
        if not self._selected_job_id:
            self.notify("No endpoint selected", severity="warning")
            return
        self.app.notify(
            f"Stop requested: {self._selected_job_id}",
            title="Action queued",
        )
