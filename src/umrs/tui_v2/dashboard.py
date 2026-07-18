"""UMRS Monitor TUI v2 Dashboard shim.

Maintains backward compatibility: same constructor signature as v1
Dashboard, but delegates to MonitorAppV2.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class DashboardV2:
    """Monitor v2 — delegates to MonitorAppV2 for interactive mode."""

    def __init__(
        self,
        control_mq: Any,
        client_id: str,
        interval: float = 3.0,
        status_filter: str | None = None,
        project_filter: str | None = None,
        *,
        initial_tab: str = "jobs",
        job_type_filter: str | None = None,
        config_path: str | None = None,
        user_cfg: dict[str, Any] | None = None,
        connection_info: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
        workflow_id: str | None = None,
    ) -> None:
        self._mq = control_mq
        self._client_id = client_id
        self._interval = interval
        self._status_filter = status_filter
        self._project_filter = project_filter
        self._initial_tab = initial_tab
        self._job_type_filter = job_type_filter
        self._config_path = config_path
        self._user_cfg = user_cfg or {}
        self._connection_info = connection_info or {}
        self._capabilities = capabilities or {}
        self._workflow_id = workflow_id

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the interactive TUI (blocks until quit)."""
        from umrs.tui_v2.app import MonitorAppV2

        app = MonitorAppV2(
            control_mq=self._mq,
            client_id=self._client_id,
            interval=self._interval,
            status_filter=self._status_filter,
            project_filter=self._project_filter,
            initial_tab=self._initial_tab,
            job_type_filter=self._job_type_filter,
            config_path=self._config_path,
            user_cfg=self._user_cfg,
            connection_info=self._connection_info,
            capabilities=self._capabilities,
            workflow_id=self._workflow_id,
        )
        app.run()

    # ------------------------------------------------------------------
    def render_once(self, console: Console | None = None) -> None:
        """Non-interactive one-shot render (same contract as v1)."""
        out_console = console or Console()
        try:
            from umrs.client.control_api import UMRSControlAPI
        except ImportError:
            out_console.print("[red]Cannot import UMRSControlAPI[/red]")
            return

        api = UMRSControlAPI(self._mq, self._client_id, self._config_path)
        try:
            parsed = api.list_jobs()
        except Exception as exc:
            out_console.print(f"[red]Error fetching jobs: {exc}[/red]")
            return

        jobs = parsed.get("jobs", []) if isinstance(parsed, dict) else []
        table = Table(title="UMRS Monitor v2 — Jobs (one-shot)", show_lines=False, header_style="bold")
        table.add_column("#", width=3, justify="right")
        table.add_column("Job ID", min_width=10)
        table.add_column("Type", min_width=10)
        table.add_column("Status", min_width=10)
        table.add_column("User", min_width=8)
        table.add_column("Attempts", justify="right", width=8)
        table.add_column("Created", min_width=8)

        for idx, job in enumerate(jobs):
            table.add_row(
                str(idx + 1),
                str(job.get("job_id", "?"))[-8:],
                str(job.get("job_type", "?")),
                str(job.get("status", "?")),
                str(job.get("user_id", "?")),
                str(len(job.get("attempts", []))),
                str(job.get("created_at", "?"))[:19],
            )
        out_console.print(table)
