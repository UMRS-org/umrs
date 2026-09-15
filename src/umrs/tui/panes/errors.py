"""Errors tab pane — error logs for failed workflow executions.

This pane surfaces error logs for failed Attempts, including:
- Workflow-level error messages
- Per-step error details for failed steps
- Error log entries with timestamps

The pane fetches error logs from the server via the
``WORKFLOW_ERRORS`` command and displays them in a scrollable table.
"""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable, Static


class ErrorsPane(Static):
    """Errors tab pane — shows error logs for failed workflow executions."""

    DEFAULT_CSS = """
    ErrorsPane { layout: vertical; padding: 0 1; }
    ErrorsPane > .summary { height: 1; color: $text; }
    ErrorsPane DataTable { height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._error_logs: list[dict[str, Any]] = []
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="errors-table", cursor_type="row")

    def compose(self):
        yield self._summary_widget
        yield self._table

    def on_mount(self) -> None:
        pass  # columns set up in _render_table

    def update_data(self, data: Any) -> None:
        """Update the pane with error log data.

        Args:
            data: Either a list of error log entries, or a dict with
                  an ``error_logs`` key containing the list.
        """
        if data is None:
            return
        if isinstance(data, dict):
            self._error_logs = data.get("error_logs", [])
        elif isinstance(data, list):
            self._error_logs = data
        else:
            self._error_logs = []
        self._render_table()
        self._render_summary()

    def clear(self) -> None:
        self._error_logs.clear()
        self._render_table()
        self._render_summary()

    def _render_table(self) -> None:
        self._table.clear(columns=True)
        self._table.add_column("Timestamp", width=26)
        self._table.add_column("Level", width=7)
        self._table.add_column("Source", width=20)
        self._table.add_column("Message", width=60)
        for e in self._error_logs:
            self._table.add_row(
                str(e.get("timestamp", "")),
                str(e.get("level", "ERROR")),
                str(e.get("source", "")),
                str(e.get("message", "")),
            )

    def _render_summary(self) -> None:
        self._summary_widget.update(f"Error logs: {len(self._error_logs)}")
