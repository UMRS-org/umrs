"""Workflow Logs tab pane — stream execution logs for a workflow run.

This pane displays the log timeline for a workflow execution, including:
- Workflow lifecycle events (submitted, started, finished)
- Per-step events (started, routing, succeeded/failed)
- Error messages

The pane auto-scrolls to the latest log entry and supports incremental
updates — new entries are appended without re-rendering the entire table.
"""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable, Static


class WorkflowLogsPane(Static):
    """Workflow Logs tab pane — streams execution logs."""

    DEFAULT_CSS = """
    WorkflowLogsPane { layout: vertical; padding: 0 1; }
    WorkflowLogsPane > .summary { height: 1; color: $text; }
    WorkflowLogsPane DataTable { height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._execution_id: str | None = None
        self._logs: list[dict[str, Any]] = []
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="workflow-logs-table", cursor_type="row")

    def compose(self):
        yield self._summary_widget
        yield self._table

    def on_mount(self) -> None:
        self._setup_table()

    def _setup_table(self) -> None:
        self._table.clear(columns=True)
        self._table.add_column("Timestamp", width=26)
        self._table.add_column("Level", width=7)
        self._table.add_column("Source", width=20)
        self._table.add_column("Message", width=60)

    @property
    def execution_id(self) -> str | None:
        return self._execution_id

    def set_execution_id(self, execution_id: str | None) -> None:
        """Set the execution ID to stream logs for.

        Clears the current log entries and re-renders.
        """
        self._execution_id = execution_id
        self._logs.clear()
        self._setup_table()
        self._render_summary()

    def update_data(self, data: Any) -> None:
        """Update the pane with workflow log data.

        Args:
            data: Either a list of log entries, or a dict with
                  a ``logs`` key containing the list.
        """
        if data is None:
            return
        if isinstance(data, dict):
            self._logs = data.get("logs", [])
        elif isinstance(data, list):
            self._logs = data
        else:
            self._logs = []
        self._setup_table()
        for entry in self._logs:
            self._table.add_row(
                str(entry.get("timestamp", "")),
                str(entry.get("level", "INFO")),
                str(entry.get("source", "")),
                str(entry.get("message", "")),
            )
        self._render_summary()
        self._auto_scroll()

    def append_log(self, entry: dict[str, Any]) -> None:
        """Append a single log entry and auto-scroll."""
        self._logs.append(entry)
        self._table.add_row(
            str(entry.get("timestamp", "")),
            str(entry.get("level", "INFO")),
            str(entry.get("source", "")),
            str(entry.get("message", "")),
        )
        self._render_summary()
        self._auto_scroll()

    def clear(self) -> None:
        self._logs.clear()
        self._setup_table()
        self._render_summary()

    def _render_summary(self) -> None:
        exec_label = self._execution_id or "(none)"
        self._summary_widget.update(
            f"Execution: {exec_label}  │  Log entries: {len(self._logs)}"
        )

    def _auto_scroll(self) -> None:
        """Scroll to the last row to show the latest log entry."""
        try:
            row_count = self._table.row_count
            if row_count > 0:
                last_key = self._table.coordinate_to_cell_key(
                    self._table.get_coordinate(row_count - 1, 0)
                ).row_key
                self._table.move_cursor(row=last_key)
        except Exception:
            pass
