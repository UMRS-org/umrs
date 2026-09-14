"""Workflow logs tab pane v2 — streaming execution logs."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class WorkflowLogsPane(Static):
    """Workflow logs tab pane v2."""

    DEFAULT_CSS = """
    WorkflowLogsPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    WorkflowLogsPane > .header {
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    WorkflowLogsPane > .content {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._exec_id: str | None = None
        self._header = Static("", classes="header")
        self._content = Static("", classes="content")

    def compose(self):
        yield self._header
        yield self._content

    def set_execution_id(self, exec_id: str | None) -> None:
        self._exec_id = exec_id
        self._render_header()

    def update_data(self, data: Any) -> None:
        if data is None:
            self._content.update("[red]No logs[/red]")
            return
        if not isinstance(data, dict):
            self._content.update(str(data))
            return
        self._render_execution(data)

    def _render_execution(self, execution: dict) -> None:
        """Render execution status and steps."""
        status = execution.get("status", "?")
        exec_id = execution.get("execution_id", self._exec_id or "?")
        progress = execution.get("progress", 0.0)
        lines = [f"[bold]Execution: {exec_id}[/bold]"]
        lines.append(f"  Status: {status}  Progress: {progress:.0%}")
        self._add_time_lines(execution, lines)
        self._add_error_line(execution, lines)
        self._add_steps_line(execution, lines)
        self._render_logs(execution, lines)
        self._content.update("\n".join(lines))

    def _add_time_lines(self, execution: dict, lines: list[str]) -> None:
        """Add submitted/started/finished times."""
        for key, label in [
            ("submitted_at", "Submitted"),
            ("started_at", "Started"),
            ("finished_at", "Finished"),
        ]:
            val = execution.get(key)
            if val:
                lines.append(f"  {label}: {val[:19]}")

    def _add_error_line(self, execution: dict, lines: list[str]) -> None:
        """Add error line if present."""
        error = execution.get("error")
        if error:
            lines.append(f"\n  [red]Error:[/red] {error}")

    def _add_steps_line(self, execution: dict, lines: list[str]) -> None:
        """Add steps section."""
        steps = execution.get("steps", [])
        if steps:
            lines.append(f"\n[bold]Steps ({len(steps)}):[/bold]")
            for s in steps:
                st = s.get("status", "?")
                name = s.get("name", "?")
                prog = s.get("progress", 0.0)
                marker = "✓" if st == "succeeded" else "✗" if st == "failed" else "→"
                lines.append(f"  {marker} {name}: {st} {prog:.0%}")

    def _render_logs(self, execution: dict, lines: list[str]) -> None:
        """Render logs section."""
        logs = execution.get("logs", [])
        if logs:
            lines.append(f"\n[bold]Logs ({len(logs)}):[/bold]")
            for log in logs[-200:]:  # Last 200 lines
                ts = (log.get("timestamp") or "")[:19]
                level = log.get("level", "INFO")
                src = log.get("source", "")
                msg = log.get("message", "")
                color = ""
                if level == "ERROR":
                    color = "[red]"
                elif level == "WARN":
                    color = "[yellow]"
                lines.append(
                    f"{color}[{ts}] {level} [{src}] {msg}[/]"
                    if color
                    else f"[{ts}] {level} [{src}] {msg}"
                )

        self._content.update("\n".join(lines))

    def _render_header(self) -> None:
        if self._exec_id:
            self._header.update(f"  Execution: {self._exec_id}  |  r=Refresh  q=Back")
        else:
            self._header.update(
                "  No execution selected — select a workflow to stream logs"
            )
