"""
UMRS Jobs Dashboard — Rich-based live-refreshing TUI.

Usage (via client subcommand):
    python src/client_umrs.py monitor [--interval 3] [--status running] [--project default]

Keyboard shortcuts:
    q       Quit
    r       Force refresh
    j / ↓   Move selection down
    k / ↑   Move selection up
    Enter   Toggle detail view for selected job
    f       Cycle status filter (all → running → failed → completed → queued → all)
    /       Toggle compact / wide table
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .keybindings import KeyReader

# ---------------------------------------------------------------------------
# Status colour mapping
# ---------------------------------------------------------------------------
STATUS_STYLES: Dict[str, str] = {
    "running": "bold yellow",
    "queued": "cyan",
    "planned": "dim cyan",
    "completed": "green",
    "failed": "bold red",
    "cancelled": "dim",
}

FILTER_CYCLE = [None, "running", "failed", "completed", "queued", "planned", "cancelled"]


def _style_status(status: str) -> Text:
    style = STATUS_STYLES.get(status, "")
    return Text(status, style=style)


def _short_id(job_id: str) -> str:
    """Return last 8 chars of a job-id for compact display."""
    return job_id[-8:] if len(job_id) > 8 else job_id


def _ago(iso_str: Optional[str]) -> str:
    """Human-readable relative time from ISO timestamp."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now(timezone.utc).replace(tzinfo=None) - dt.replace(tzinfo=None)
        secs = int(delta.total_seconds())
        if secs < 0:
            return "just now"
        if secs < 60:
            return f"{secs}s ago"
        mins = secs // 60
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except (ValueError, TypeError):
        return str(iso_str)[:19]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class Dashboard:
    """Live-refreshing jobs dashboard."""

    def __init__(
        self,
        control_mq: Any,
        client_id: str,
        interval: float = 3.0,
        status_filter: Optional[str] = None,
        project_filter: Optional[str] = None,
        console: Optional[Console] = None,
    ):
        self._mq = control_mq
        self._client_id = client_id
        self._interval = interval
        self._status_filter = status_filter
        self._project_filter = project_filter
        self._console = console or Console()

        self._jobs: List[Dict[str, Any]] = []
        self._cursor = 0
        self._detail_job_id: Optional[str] = None  # expanded job
        self._compact = True
        self._quit = threading.Event()
        self._force_refresh = threading.Event()
        self._last_error: Optional[str] = None
        self._last_refresh: Optional[float] = None

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------
    def _fetch_jobs(self) -> None:
        filters: Dict[str, Any] = {}
        if self._status_filter:
            filters["status"] = self._status_filter
        if self._project_filter:
            filters["project_id"] = self._project_filter
        try:
            resp = self._mq.request_job_list(self._client_id, filters)
            if resp.get("status") == "ok":
                self._jobs = resp.get("jobs", [])
                self._last_error = None
            else:
                self._last_error = resp.get("message", "unknown error")
        except Exception as e:
            self._last_error = str(e)
        self._last_refresh = time.time()
        # Clamp cursor
        if self._jobs:
            self._cursor = max(0, min(self._cursor, len(self._jobs) - 1))
        else:
            self._cursor = 0

    # ------------------------------------------------------------------
    # Keyboard handler
    # ------------------------------------------------------------------
    def _move_down(self) -> None:
        self._cursor = min(self._cursor + 1, max(len(self._jobs) - 1, 0))

    def _move_up(self) -> None:
        self._cursor = max(self._cursor - 1, 0)

    def _toggle_detail(self) -> None:
        if self._jobs:
            jid = self._jobs[self._cursor].get("job_id")
            self._detail_job_id = None if self._detail_job_id == jid else jid

    def _cycle_filter(self) -> None:
        idx = FILTER_CYCLE.index(self._status_filter) if self._status_filter in FILTER_CYCLE else 0
        self._status_filter = FILTER_CYCLE[(idx + 1) % len(FILTER_CYCLE)]
        self._force_refresh.set()

    def _on_key(self, key: str) -> None:
        _KEY_ACTIONS = {
            "q": lambda: self._quit.set(),
            "Q": lambda: self._quit.set(),
            "\x03": lambda: self._quit.set(),
            "r": lambda: self._force_refresh.set(),
            "R": lambda: self._force_refresh.set(),
            "j": self._move_down,
            "down": self._move_down,
            "k": self._move_up,
            "up": self._move_up,
            "\r": self._toggle_detail,
            "\n": self._toggle_detail,
            "f": self._cycle_filter,
            "/": lambda: setattr(self, "_compact", not self._compact),
        }
        action = _KEY_ACTIONS.get(key)
        if action:
            action()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _build_jobs_table(self) -> Table:
        table = Table(
            title=None,
            expand=True,
            show_lines=False,
            padding=(0, 1),
            header_style="bold bright_white",
        )
        table.add_column("#", width=3, justify="right")
        table.add_column("Job ID", min_width=10)
        table.add_column("Type", min_width=10)
        table.add_column("Status", min_width=10)
        table.add_column("User", min_width=8)
        if not self._compact:
            table.add_column("Project", min_width=8)
        table.add_column("Attempts", justify="right", width=8)
        table.add_column("Created", min_width=8)
        table.add_column("Updated", min_width=8)

        for idx, job in enumerate(self._jobs):
            is_selected = idx == self._cursor
            row_style = "reverse" if is_selected else ""
            n_attempts = len(job.get("attempts", []))

            cols: list = [
                str(idx + 1),
                _short_id(job.get("job_id", "?")),
                job.get("job_type", "?"),
                _style_status(job.get("status", "?")),
                job.get("user_id", "?"),
            ]
            if not self._compact:
                cols.append(job.get("project_id", "?"))
            cols.extend([
                str(n_attempts),
                _ago(job.get("created_at")),
                _ago(job.get("updated_at")),
            ])
            table.add_row(*cols, style=row_style)

        return table

    def _build_detail_panel(self) -> Optional[Panel]:
        if not self._detail_job_id or not self._jobs:
            return None
        job = next((j for j in self._jobs if j.get("job_id") == self._detail_job_id), None)
        if not job:
            return None

        # Attempts table
        att_table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold")
        att_table.add_column("#", width=3, justify="right")
        att_table.add_column("Attempt ID", min_width=12)
        att_table.add_column("Status", min_width=10)
        att_table.add_column("Started", min_width=14)
        att_table.add_column("Ended", min_width=14)
        att_table.add_column("Exit Reason")

        for att in job.get("attempts", []):
            att_table.add_row(
                str(att.get("attempt_no", "?")),
                _short_id(att.get("attempt_id", "?")),
                _style_status(att.get("status", "?")),
                _ago(att.get("started_at")),
                _ago(att.get("ended_at")),
                att.get("exit_reason") or "—",
            )

        # Spec summary
        spec_str = json.dumps(job.get("spec", {}), indent=2)
        if len(spec_str) > 200:
            spec_str = spec_str[:200] + "..."

        subtitle = (
            f"type={job.get('job_type')}  project={job.get('project_id')}  "
            f"policy={job.get('restart_policy', '?')}  max_attempts={job.get('desired_max_attempts', '?')}"
        )

        from rich.console import Group
        detail_group = Group(
            att_table,
            Text(f"\nSpec: {spec_str}", style="dim"),
        )
        return Panel(
            detail_group,
            title=f"[bold]Job {self._detail_job_id}[/bold]",
            subtitle=subtitle,
            border_style="bright_blue",
        )

    def _build_display(self) -> Layout:
        layout = Layout()

        # Header bar
        filter_label = self._status_filter or "all"
        hdr_parts = [
            f"[bold]UMRS Jobs Monitor[/bold]",
            f"filter=[cyan]{filter_label}[/cyan]",
            f"jobs=[yellow]{len(self._jobs)}[/yellow]",
        ]
        if self._last_refresh:
            hdr_parts.append(f"refreshed {_ago(datetime.fromtimestamp(self._last_refresh, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'))}")
        if self._last_error:
            hdr_parts.append(f"[red]err: {self._last_error}[/red]")
        header = Text.from_markup("  ".join(hdr_parts))

        help_text = Text.from_markup(
            "[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]j/k[/dim] navigate  "
            "[dim]Enter[/dim] detail  [dim]f[/dim] filter  [dim]/[/dim] compact"
        )

        jobs_table = self._build_jobs_table()
        detail = self._build_detail_panel()

        if detail:
            layout.split_column(
                Layout(header, size=1),
                Layout(Panel(jobs_table, border_style="bright_white"), ratio=2),
                Layout(detail, ratio=1),
                Layout(help_text, size=1),
            )
        else:
            layout.split_column(
                Layout(header, size=1),
                Layout(Panel(jobs_table, border_style="bright_white"), ratio=1),
                Layout(help_text, size=1),
            )

        return layout

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the dashboard (blocks until the user quits)."""
        reader = KeyReader(self._on_key)
        reader.start()

        self._fetch_jobs()

        try:
            with Live(
                self._build_display(),
                console=self._console,
                refresh_per_second=4,
                screen=True,
            ) as live:
                while not self._quit.is_set():
                    # Wait for interval or force-refresh
                    self._force_refresh.wait(timeout=self._interval)
                    self._force_refresh.clear()
                    if self._quit.is_set():
                        break
                    self._fetch_jobs()
                    live.update(self._build_display())
        finally:
            reader.stop()

    # ------------------------------------------------------------------
    # Non-interactive rendering (for tests / piped output)
    # ------------------------------------------------------------------
    def render_once(self) -> None:
        """Fetch and print the dashboard once, then exit."""
        self._fetch_jobs()
        self._console.print(self._build_display())
