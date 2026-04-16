"""
UMRS Monitor — Multi-tab Rich TUI for live system monitoring.

Usage (via client subcommand):
    umrs-client monitor [--interval 3] [--status running] [--project demo]
    umrs-client monitor --tab status
    umrs-client monitor -c conf_umrs.json   # enables in-TUI reload

Tabs:
    1  Jobs        Durable job queue with detail expansion
    2  Status      Server task summaries (recompute/update/check/reload)
    3  Models      Registered model catalog
    4  Workflows   Registered workflows and their steps
    5  Connection  Active connection profile + server health

Keyboard shortcuts (global):
    1-5         Switch tab
    q / Ctrl+C  Quit
    r           Force refresh
    ?           Toggle help overlay

Keyboard shortcuts (Jobs tab):
    j / down    Move selection down
    k / up      Move selection up
    Enter       Toggle detail panel for selected job
    f           Cycle status filter
    /           Toggle compact / wide table
    c           Cancel selected job

Keyboard shortcuts (Status tab):
    R           Trigger server reload (requires --conf)

Keyboard shortcuts (Models / Workflows tabs):
    j / down    Move selection down
    k / up      Move selection up
    Enter       Toggle detail panel
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .keybindings import KeyReader

# ── Tab constants ─────────────────────────────────────────────────────
TAB_JOBS = "jobs"
TAB_STATUS = "status"
TAB_MODELS = "models"
TAB_WORKFLOWS = "workflows"
TAB_CONNECTION = "connection"

TAB_ORDER = [TAB_JOBS, TAB_STATUS, TAB_MODELS, TAB_WORKFLOWS, TAB_CONNECTION]
TAB_LABELS = {
    TAB_JOBS: "Jobs",
    TAB_STATUS: "Status",
    TAB_MODELS: "Models",
    TAB_WORKFLOWS: "Workflows",
    TAB_CONNECTION: "Connection",
}

# ── Status colour mapping ────────────────────────────────────────────
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
    """Return last 8 chars of an id for compact display."""
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


def _ts_short(ts: Optional[float]) -> str:
    """Compact HH:MM:SS from unix timestamp."""
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# ── Dashboard ────────────────────────────────────────────────────────
class Dashboard:
    """Multi-tab live-refreshing UMRS monitor."""

    def __init__(
        self,
        control_mq: Any,
        client_id: str,
        interval: float = 3.0,
        status_filter: Optional[str] = None,
        project_filter: Optional[str] = None,
        console: Optional[Console] = None,
        *,
        initial_tab: str = TAB_JOBS,
        config_path: Optional[str] = None,
        user_cfg: Optional[Dict[str, Any]] = None,
        connection_info: Optional[Dict[str, Any]] = None,
    ):
        self._mq = control_mq
        self._client_id = client_id
        self._interval = interval
        self._status_filter = status_filter
        self._project_filter = project_filter
        self._console = console or Console()

        # Tab state
        self._tab = initial_tab if initial_tab in TAB_ORDER else TAB_JOBS

        # Config / reload support
        self._config_path = config_path
        self._user_cfg = user_cfg or {}

        # Connection info (from ResolvedConnection.to_dict())
        self._connection_info = connection_info or {}

        # ── per-tab data caches ──
        # Jobs
        self._jobs: List[Dict[str, Any]] = []
        self._jobs_cursor = 0
        self._detail_job_id: Optional[str] = None
        self._compact = True

        # Status
        self._status_data: Dict[str, Any] = {}

        # Models
        self._models: List[Dict[str, Any]] = []
        self._models_cursor = 0
        self._model_detail_name: Optional[str] = None

        # Workflows
        self._workflows: List[Dict[str, Any]] = []
        self._wf_cursor = 0
        self._wf_detail_name: Optional[str] = None

        # Connection
        self._ping_ok: Optional[bool] = None
        self._ping_ms: Optional[float] = None

        # UI state
        self._quit = threading.Event()
        self._force_refresh = threading.Event()
        self._last_error: Optional[str] = None
        self._last_refresh: Optional[float] = None
        self._show_help = False
        self._flash_message: Optional[str] = None
        self._flash_expire: float = 0

        # Track which tabs have been fetched at least once
        self._fetched: Dict[str, bool] = {}

    # ==================================================================
    # Data fetching — per tab
    # ==================================================================
    def _fetch_active_tab(self) -> None:
        """Fetch data only for the currently active tab."""
        try:
            if self._tab == TAB_JOBS:
                self._fetch_jobs()
            elif self._tab == TAB_STATUS:
                self._fetch_status()
            elif self._tab == TAB_MODELS:
                self._fetch_models()
            elif self._tab == TAB_WORKFLOWS:
                self._fetch_workflows()
            elif self._tab == TAB_CONNECTION:
                self._fetch_connection()
        except Exception as e:
            self._last_error = str(e)
        self._last_refresh = time.time()
        self._fetched[self._tab] = True

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
        if self._jobs:
            self._jobs_cursor = max(0, min(self._jobs_cursor, len(self._jobs) - 1))
        else:
            self._jobs_cursor = 0

    def _fetch_status(self) -> None:
        try:
            raw = self._mq.request_status(self._client_id, "all")
            parsed = json.loads(raw)
            if parsed.get("status") == "ok":
                self._status_data = parsed
                self._last_error = None
            else:
                self._last_error = parsed.get("message", "status unavailable")
        except Exception as e:
            self._last_error = str(e)

    def _fetch_models(self) -> None:
        from umrs.core.TaskCmdStatus import TaskCmd
        try:
            raw = self._mq.request_model_command(self._client_id, TaskCmd.MODEL_LIST, "")
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                self._models = parsed
                self._last_error = None
            elif isinstance(parsed, dict):
                self._models = parsed.get("models", []) if "models" in parsed else []
                if parsed.get("status") == "error":
                    self._last_error = parsed.get("message")
                else:
                    self._last_error = None
            else:
                self._models = []
        except Exception as e:
            self._last_error = str(e)
        if self._models:
            self._models_cursor = max(0, min(self._models_cursor, len(self._models) - 1))
        else:
            self._models_cursor = 0

    def _fetch_workflows(self) -> None:
        from umrs.core.TaskCmdStatus import TaskCmd
        try:
            raw = self._mq.request_model_command(self._client_id, TaskCmd.LIST_WORKFLOWS, "")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                self._workflows = parsed.get("workflows", [])
                self._last_error = None
            elif isinstance(parsed, list):
                self._workflows = parsed
                self._last_error = None
            else:
                self._workflows = []
        except Exception as e:
            self._last_error = str(e)
        if self._workflows:
            self._wf_cursor = max(0, min(self._wf_cursor, len(self._workflows) - 1))
        else:
            self._wf_cursor = 0

    def _fetch_connection(self) -> None:
        t0 = time.monotonic()
        try:
            self._ping_ok = self._mq.is_server_ready(timeout_ms=3000)
            self._ping_ms = (time.monotonic() - t0) * 1000
            self._last_error = None
        except Exception as e:
            self._ping_ok = False
            self._ping_ms = None
            self._last_error = str(e)

    # ==================================================================
    # Actions
    # ==================================================================
    def _cancel_selected_job(self) -> None:
        if not self._jobs:
            return
        job = self._jobs[self._jobs_cursor]
        jid = job.get("job_id")
        status = job.get("status", "")
        if status in ("completed", "failed", "cancelled"):
            self._set_flash(f"Job {_short_id(jid)} already {status}")
            return
        from umrs.core.TaskCmdStatus import TaskCmd
        try:
            raw = self._mq.request_model_command(
                self._client_id, TaskCmd.JOB_CANCEL, json.dumps({"job_id": jid})
            )
            parsed = json.loads(raw)
            if parsed.get("status") == "ok":
                self._set_flash(f"Cancelled job {_short_id(jid)}")
            else:
                self._set_flash(f"Cancel failed: {parsed.get('message', '?')}")
        except Exception as e:
            self._set_flash(f"Cancel error: {e}")
        self._force_refresh.set()

    def _do_reload(self) -> None:
        if not self._config_path or not os.path.isfile(self._config_path):
            self._set_flash("No config file (use --conf/-c to enable reload)")
            return
        from umrs.core.TaskCmdStatus import TaskCmd
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            upload_user = cfg_json.get("username") or self._client_id
        except Exception:
            upload_user = self._client_id

        try:
            data_mq = self._mq.get_data_queue(upload_user)
            try:
                data_mq.send_file(self._config_path)
                self._mq.notify_server(upload_user, TaskCmd.FILE, 1)
            finally:
                try:
                    data_mq.close()
                except Exception:
                    pass

            ok = self._mq.notify_server(self._client_id, TaskCmd.RELOAD, 0)
            if ok:
                self._set_flash("Reload triggered successfully")
            else:
                self._set_flash("Reload command rejected by server")
        except Exception as e:
            self._set_flash(f"Reload error: {e}")
        self._force_refresh.set()

    def _set_flash(self, msg: str, duration: float = 5.0) -> None:
        self._flash_message = msg
        self._flash_expire = time.time() + duration

    # ==================================================================
    # Keyboard handler
    # ==================================================================
    def _move_down(self) -> None:
        if self._tab == TAB_JOBS:
            self._jobs_cursor = min(self._jobs_cursor + 1, max(len(self._jobs) - 1, 0))
        elif self._tab == TAB_MODELS:
            self._models_cursor = min(self._models_cursor + 1, max(len(self._models) - 1, 0))
        elif self._tab == TAB_WORKFLOWS:
            self._wf_cursor = min(self._wf_cursor + 1, max(len(self._workflows) - 1, 0))

    def _move_up(self) -> None:
        if self._tab == TAB_JOBS:
            self._jobs_cursor = max(self._jobs_cursor - 1, 0)
        elif self._tab == TAB_MODELS:
            self._models_cursor = max(self._models_cursor - 1, 0)
        elif self._tab == TAB_WORKFLOWS:
            self._wf_cursor = max(self._wf_cursor - 1, 0)

    def _toggle_detail(self) -> None:
        if self._tab == TAB_JOBS and self._jobs:
            jid = self._jobs[self._jobs_cursor].get("job_id")
            self._detail_job_id = None if self._detail_job_id == jid else jid
        elif self._tab == TAB_MODELS and self._models:
            name = self._models[self._models_cursor].get("name")
            self._model_detail_name = None if self._model_detail_name == name else name
        elif self._tab == TAB_WORKFLOWS and self._workflows:
            name = self._workflows[self._wf_cursor].get("name")
            self._wf_detail_name = None if self._wf_detail_name == name else name

    def _cycle_filter(self) -> None:
        idx = FILTER_CYCLE.index(self._status_filter) if self._status_filter in FILTER_CYCLE else 0
        self._status_filter = FILTER_CYCLE[(idx + 1) % len(FILTER_CYCLE)]
        self._force_refresh.set()

    def _switch_tab(self, tab: str) -> None:
        if tab != self._tab:
            self._tab = tab
            self._show_help = False
            if not self._fetched.get(tab):
                self._force_refresh.set()

    def _on_key(self, key: str) -> None:
        # Tab switching via number keys
        tab_keys = {"1": TAB_JOBS, "2": TAB_STATUS, "3": TAB_MODELS, "4": TAB_WORKFLOWS, "5": TAB_CONNECTION}
        if key in tab_keys:
            self._switch_tab(tab_keys[key])
            return

        # Global keys
        global_actions = {
            "q": lambda: self._quit.set(),
            "Q": lambda: self._quit.set(),
            "\x03": lambda: self._quit.set(),
            "r": lambda: self._force_refresh.set(),
            "?": lambda: setattr(self, "_show_help", not self._show_help),
        }
        action = global_actions.get(key)
        if action:
            action()
            return

        # Per-tab keys
        if self._tab == TAB_JOBS:
            jobs_actions = {
                "j": self._move_down,
                "down": self._move_down,
                "k": self._move_up,
                "up": self._move_up,
                "\r": self._toggle_detail,
                "\n": self._toggle_detail,
                "f": self._cycle_filter,
                "/": lambda: setattr(self, "_compact", not self._compact),
                "c": self._cancel_selected_job,
            }
            act = jobs_actions.get(key)
            if act:
                act()
        elif self._tab == TAB_STATUS:
            if key == "R":
                self._do_reload()
        elif self._tab in (TAB_MODELS, TAB_WORKFLOWS):
            nav_actions = {
                "j": self._move_down,
                "down": self._move_down,
                "k": self._move_up,
                "up": self._move_up,
                "\r": self._toggle_detail,
                "\n": self._toggle_detail,
            }
            act = nav_actions.get(key)
            if act:
                act()

    # ==================================================================
    # Rendering — Header and Footer
    # ==================================================================
    def _build_tab_bar(self) -> Text:
        bar = Text()
        for i, tab_id in enumerate(TAB_ORDER):
            key = str(i + 1)
            label = TAB_LABELS[tab_id]
            if tab_id == self._tab:
                bar.append(f" {key}:{label} ", style="bold white on blue")
            else:
                bar.append(f" {key}:{label} ", style="dim")
            bar.append(" ")

        # Spacer + health indicator
        bar.append("  ")
        if self._ping_ok is True:
            bar.append("● ", style="bold green")
            bar.append("connected", style="green")
        elif self._ping_ok is False:
            bar.append("● ", style="bold red")
            bar.append("disconnected", style="red")
        else:
            bar.append("● ", style="dim")
            bar.append("unknown", style="dim")

        if self._last_refresh:
            bar.append(f"  refreshed {_ts_short(self._last_refresh)}", style="dim")

        # Flash message
        if self._flash_message and time.time() < self._flash_expire:
            bar.append(f"  │ {self._flash_message}", style="bold yellow")
        elif self._flash_message:
            self._flash_message = None

        if self._last_error:
            bar.append(f"  [err: {self._last_error}]", style="bold red")

        return bar

    def _build_footer(self) -> Text:
        if self._tab == TAB_JOBS:
            return Text.from_markup(
                "[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]j/k[/dim] navigate  "
                "[dim]Enter[/dim] detail  [dim]f[/dim] filter  [dim]/[/dim] compact  "
                "[dim]c[/dim] cancel  [dim]1-5[/dim] tabs  [dim]?[/dim] help"
            )
        elif self._tab == TAB_STATUS:
            reload_hint = "[dim]R[/dim] reload" if self._config_path else "[dim strikethrough]R reload (no --conf)[/dim strikethrough]"
            return Text.from_markup(
                f"[dim]q[/dim] quit  [dim]r[/dim] refresh  {reload_hint}  "
                "[dim]1-5[/dim] tabs  [dim]?[/dim] help"
            )
        elif self._tab in (TAB_MODELS, TAB_WORKFLOWS):
            return Text.from_markup(
                "[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]j/k[/dim] navigate  "
                "[dim]Enter[/dim] detail  [dim]1-5[/dim] tabs  [dim]?[/dim] help"
            )
        else:
            return Text.from_markup(
                "[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]1-5[/dim] tabs  [dim]?[/dim] help"
            )

    # ==================================================================
    # Rendering — Help overlay
    # ==================================================================
    def _build_help_panel(self) -> Panel:
        t = Table(show_header=True, header_style="bold", expand=True, show_lines=False, padding=(0, 2))
        t.add_column("Key", style="bold cyan", width=12)
        t.add_column("Action")
        t.add_column("Tab", style="dim")

        t.add_row("1-5", "Switch tab", "Global")
        t.add_row("q / Ctrl+C", "Quit", "Global")
        t.add_row("r", "Force refresh", "Global")
        t.add_row("?", "Toggle this help", "Global")
        t.add_row("", "", "")
        t.add_row("j / ↓", "Move selection down", "Jobs, Models, Workflows")
        t.add_row("k / ↑", "Move selection up", "Jobs, Models, Workflows")
        t.add_row("Enter", "Toggle detail view", "Jobs, Models, Workflows")
        t.add_row("f", "Cycle status filter", "Jobs")
        t.add_row("/", "Toggle compact/wide", "Jobs")
        t.add_row("c", "Cancel selected job", "Jobs")
        t.add_row("R", "Trigger server reload", "Status (requires --conf)")

        return Panel(t, title="[bold]Keyboard Shortcuts[/bold]", border_style="bright_cyan", padding=(1, 2))

    # ==================================================================
    # Rendering — Jobs tab
    # ==================================================================
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
            is_selected = idx == self._jobs_cursor
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

    def _build_job_detail_panel(self) -> Optional[Panel]:
        if not self._detail_job_id or not self._jobs:
            return None
        job = next((j for j in self._jobs if j.get("job_id") == self._detail_job_id), None)
        if not job:
            return None

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

        spec_str = json.dumps(job.get("spec", {}), indent=2)
        if len(spec_str) > 300:
            spec_str = spec_str[:300] + "..."

        subtitle = (
            f"type={job.get('job_type')}  project={job.get('project_id')}  "
            f"policy={job.get('restart_policy', '?')}  max_attempts={job.get('desired_max_attempts', '?')}"
        )

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

    def _build_jobs_tab(self) -> Layout:
        layout = Layout()
        filter_label = self._status_filter or "all"
        summary = Text.from_markup(
            f"  filter=[cyan]{filter_label}[/cyan]  total=[yellow]{len(self._jobs)}[/yellow]"
        )

        jobs_table = self._build_jobs_table()
        detail = self._build_job_detail_panel()

        if not self._jobs:
            empty = Panel(
                Text("No jobs found. Submit jobs with: umrs-client job submit <type>", style="dim"),
                border_style="dim",
            )
            layout.split_column(
                Layout(summary, size=1),
                Layout(empty, ratio=1),
            )
        elif detail:
            layout.split_column(
                Layout(summary, size=1),
                Layout(Panel(jobs_table, border_style="bright_white"), ratio=2),
                Layout(detail, ratio=1),
            )
        else:
            layout.split_column(
                Layout(summary, size=1),
                Layout(Panel(jobs_table, border_style="bright_white"), ratio=1),
            )
        return layout

    # ==================================================================
    # Rendering — Status tab
    # ==================================================================
    def _build_status_tab(self) -> Layout:
        layout = Layout()

        if not self._status_data:
            layout.split_column(
                Layout(Panel(
                    Text("Loading server status...", style="dim"),
                    border_style="dim",
                ), ratio=1),
            )
            return layout

        client_label = self._status_data.get("client", "")
        summaries = self._status_data.get("summaries", {})
        uncompleted = self._status_data.get("uncompleted_key_list")

        table = Table(expand=True, show_lines=True, padding=(0, 1), header_style="bold bright_white")
        table.add_column("Task", min_width=16, style="bold")
        table.add_column("Generated At", min_width=18)
        table.add_column("Summary", ratio=1)

        task_order = ["process_objects", "wait_output", "recompute", "update", "check", "reload"]
        seen = set()
        for task_key in task_order:
            entry = summaries.get(task_key)
            if entry:
                seen.add(task_key)
                gen_at = entry.get("generated_at", "—")
                summary_text = entry.get("summary", "(no summary)")
                table.add_row(task_key, str(gen_at), summary_text)
        for k, v in summaries.items():
            if k not in seen and isinstance(v, dict):
                gen_at = v.get("generated_at", "—")
                summary_text = v.get("summary", "(no summary)")
                table.add_row(k, str(gen_at), summary_text)

        if not summaries:
            table.add_row("—", "—", "(no status entries available)")

        info_parts = []
        if client_label:
            info_parts.append(f"client=[cyan]{client_label}[/cyan]")
        if uncompleted:
            info_parts.append(f"pending=[yellow]{len(uncompleted)}[/yellow]")
        else:
            info_parts.append("pending=[green]0[/green]")

        if self._config_path:
            reload_hint = f"  config=[green]{os.path.basename(self._config_path)}[/green]  [dim]press R to reload[/dim]"
        else:
            reload_hint = "  [dim]use --conf to enable reload[/dim]"

        header = Text.from_markup(f"  {'  '.join(info_parts)}{reload_hint}")

        parts: list = [
            Layout(header, size=1),
            Layout(Panel(table, title="[bold]Task Summaries[/bold]", border_style="bright_white"), ratio=2),
        ]

        if uncompleted:
            uc_table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold")
            uc_table.add_column("Uncompleted Task", style="yellow")
            for item in uncompleted:
                uc_table.add_row(str(item))
            parts.append(
                Layout(Panel(uc_table, title="[bold]Pending Tasks[/bold]", border_style="yellow"), ratio=1)
            )

        layout.split_column(*parts)
        return layout

    # ==================================================================
    # Rendering — Models tab
    # ==================================================================
    def _build_models_tab(self) -> Layout:
        layout = Layout()

        table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold bright_white")
        table.add_column("#", width=3, justify="right")
        table.add_column("Name", min_width=14, style="bold")
        table.add_column("Type", min_width=10)
        table.add_column("Scope", min_width=8)
        table.add_column("HuggingFace ID", min_width=14)
        table.add_column("Registered By", min_width=10)
        table.add_column("Registered At", min_width=10)

        for idx, m in enumerate(self._models):
            is_selected = idx == self._models_cursor
            row_style = "reverse" if is_selected else ""
            table.add_row(
                str(idx + 1),
                m.get("name", "?"),
                m.get("model_type", m.get("type", "?")),
                m.get("scope", "?"),
                m.get("huggingface_id", "—"),
                m.get("registered_by", "—"),
                _ago(m.get("registered_at")),
                style=row_style,
            )

        summary = Text.from_markup(f"  total=[yellow]{len(self._models)}[/yellow]")
        detail = self._build_model_detail_panel()

        if not self._models:
            empty = Panel(
                Text("No models registered. Use: umrs-client model register ...", style="dim"),
                border_style="dim",
            )
            layout.split_column(
                Layout(summary, size=1),
                Layout(empty, ratio=1),
            )
        elif detail:
            layout.split_column(
                Layout(summary, size=1),
                Layout(Panel(table, border_style="bright_white"), ratio=2),
                Layout(detail, ratio=1),
            )
        else:
            layout.split_column(
                Layout(summary, size=1),
                Layout(Panel(table, border_style="bright_white"), ratio=1),
            )
        return layout

    def _build_model_detail_panel(self) -> Optional[Panel]:
        if not self._model_detail_name or not self._models:
            return None
        model = next((m for m in self._models if m.get("name") == self._model_detail_name), None)
        if not model:
            return None

        detail_table = Table(show_header=False, expand=True, padding=(0, 2))
        detail_table.add_column("Field", style="bold", width=18)
        detail_table.add_column("Value", ratio=1)

        for key in ["name", "model_type", "scope", "storage_ref", "huggingface_id",
                     "description", "registered_by", "registered_at"]:
            val = model.get(key)
            if val is not None:
                detail_table.add_row(key, str(val))

        shown = {"name", "model_type", "type", "scope", "storage_ref", "huggingface_id",
                 "description", "registered_by", "registered_at"}
        for k, v in model.items():
            if k not in shown:
                detail_table.add_row(k, str(v))

        return Panel(
            detail_table,
            title=f"[bold]Model: {self._model_detail_name}[/bold]",
            border_style="bright_blue",
        )

    # ==================================================================
    # Rendering — Workflows tab
    # ==================================================================
    def _build_workflows_tab(self) -> Layout:
        layout = Layout()

        table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold bright_white")
        table.add_column("#", width=3, justify="right")
        table.add_column("Name", min_width=14, style="bold")
        table.add_column("Version", min_width=8)
        table.add_column("Scope", min_width=8)
        table.add_column("Owner", min_width=10)
        table.add_column("Steps", width=6, justify="right")
        table.add_column("Registered At", min_width=10)

        for idx, wf in enumerate(self._workflows):
            is_selected = idx == self._wf_cursor
            row_style = "reverse" if is_selected else ""
            steps = wf.get("steps", [])
            table.add_row(
                str(idx + 1),
                wf.get("name", "?"),
                wf.get("version", "?"),
                wf.get("scope", "?"),
                wf.get("user_id", "—"),
                str(len(steps)),
                _ago(wf.get("registered_at")),
                style=row_style,
            )

        summary = Text.from_markup(f"  total=[yellow]{len(self._workflows)}[/yellow]")
        detail = self._build_workflow_detail_panel()

        if not self._workflows:
            empty = Panel(
                Text("No workflows registered. Use: umrs-client upload-workflow <path>", style="dim"),
                border_style="dim",
            )
            layout.split_column(
                Layout(summary, size=1),
                Layout(empty, ratio=1),
            )
        elif detail:
            layout.split_column(
                Layout(summary, size=1),
                Layout(Panel(table, border_style="bright_white"), ratio=2),
                Layout(detail, ratio=1),
            )
        else:
            layout.split_column(
                Layout(summary, size=1),
                Layout(Panel(table, border_style="bright_white"), ratio=1),
            )
        return layout

    def _build_workflow_detail_panel(self) -> Optional[Panel]:
        if not self._wf_detail_name or not self._workflows:
            return None
        wf = next((w for w in self._workflows if w.get("name") == self._wf_detail_name), None)
        if not wf:
            return None

        parts = []

        meta_table = Table(show_header=False, expand=True, padding=(0, 2))
        meta_table.add_column("Field", style="bold", width=14)
        meta_table.add_column("Value", ratio=1)
        for key in ["name", "version", "scope", "user_id", "source_path", "description", "registered_at"]:
            val = wf.get(key)
            if val is not None:
                meta_table.add_row(key, str(val))
        parts.append(meta_table)

        steps = wf.get("steps", [])
        if steps:
            steps_table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold")
            steps_table.add_column("#", width=3, justify="right")
            steps_table.add_column("Step Name", min_width=14)
            steps_table.add_column("Entrypoint", min_width=14)
            for i, step in enumerate(steps):
                steps_table.add_row(
                    str(i + 1),
                    step.get("name", "?"),
                    step.get("entrypoint", "?"),
                )
            parts.append(Text(""))
            parts.append(steps_table)

        reqs = wf.get("requirements", [])
        if reqs:
            parts.append(Text(f"\nRequirements: {', '.join(reqs)}", style="dim"))

        return Panel(
            Group(*parts),
            title=f"[bold]Workflow: {self._wf_detail_name}[/bold]",
            border_style="bright_blue",
        )

    # ==================================================================
    # Rendering — Connection tab
    # ==================================================================
    def _build_connection_tab(self) -> Layout:
        layout = Layout()
        info = self._connection_info

        conn_table = Table(show_header=False, expand=True, padding=(0, 2), show_lines=False)
        conn_table.add_column("Property", style="bold", width=22)
        conn_table.add_column("Value", ratio=1)
        conn_table.add_column("Source", style="dim", width=24)

        sources = info.get("sources", {})

        rows = [
            ("Profile", info.get("profile_name") or "—", info.get("profile_source") or "—"),
            ("Mode", info.get("mode") or "—", ""),
            ("Server Host", info.get("server_host") or "127.0.0.1", sources.get("server_host", "")),
            ("Control Port", str(info.get("control_port", 5555)), sources.get("control_port", "")),
            ("Data Base Port", str(info.get("data_base_port", 5556)), sources.get("data_base_port", "")),
            ("Tunnel Mode", str(info.get("tunnel_mode", False)), sources.get("tunnel_mode", "")),
            ("Health Timeout (ms)", str(info.get("health_check_timeout_ms", 5000)), sources.get("health_check_timeout_ms", "")),
            ("MinIO URL", info.get("minio_url") or "—", ""),
            ("RAG URL", info.get("rag_url") or "—", ""),
        ]
        ssh_host = info.get("ssh_host")
        if ssh_host:
            rows.append(("SSH Host", ssh_host, ""))
        cmd_hint = info.get("command_hint")
        if cmd_hint:
            rows.append(("SSH Command Hint", cmd_hint, ""))

        for label, value, source in rows:
            conn_table.add_row(label, str(value), str(source))

        # Health panel
        health_parts = []
        if self._ping_ok is True:
            health_parts.append(Text("● Server is READY", style="bold green"))
            if self._ping_ms is not None:
                health_parts.append(Text(f"  Ping latency: {self._ping_ms:.0f} ms", style="green"))
        elif self._ping_ok is False:
            health_parts.append(Text("● Server is UNREACHABLE", style="bold red"))
            if self._last_error:
                health_parts.append(Text(f"  Error: {self._last_error}", style="red"))
        else:
            health_parts.append(Text("● Server health unknown (press r to check)", style="dim"))

        health_parts.append(Text(f"\n  Client ID: {self._client_id}", style="dim"))
        if self._config_path:
            health_parts.append(Text(f"  Config: {self._config_path}", style="dim"))

        forwards = info.get("local_forwards", [])
        if forwards:
            fwd_table = Table(show_header=True, expand=True, padding=(0, 1), header_style="bold")
            fwd_table.add_column("Local Port", width=12, justify="right")
            fwd_table.add_column("Remote Host", min_width=14)
            fwd_table.add_column("Remote Port", width=12, justify="right")
            for fwd in forwards:
                fwd_table.add_row(
                    str(fwd.get("local_port", "?")),
                    fwd.get("remote_host", "?"),
                    str(fwd.get("remote_port", "?")),
                )
            layout.split_column(
                Layout(Panel(Group(*health_parts), title="[bold]Server Health[/bold]",
                             border_style="green" if self._ping_ok else "red"), size=6),
                Layout(Panel(conn_table, title="[bold]Connection Profile[/bold]",
                             border_style="bright_white"), ratio=2),
                Layout(Panel(fwd_table, title="[bold]SSH Port Forwards[/bold]",
                             border_style="bright_white"), ratio=1),
            )
        else:
            layout.split_column(
                Layout(Panel(Group(*health_parts), title="[bold]Server Health[/bold]",
                             border_style="green" if self._ping_ok else "red"), size=6),
                Layout(Panel(conn_table, title="[bold]Connection Profile[/bold]",
                             border_style="bright_white"), ratio=1),
            )
        return layout

    # ==================================================================
    # Rendering — Display compositor
    # ==================================================================
    def _build_display(self) -> Layout:
        layout = Layout()

        tab_bar = self._build_tab_bar()
        footer = self._build_footer()

        if self._tab == TAB_JOBS:
            content = self._build_jobs_tab()
        elif self._tab == TAB_STATUS:
            content = self._build_status_tab()
        elif self._tab == TAB_MODELS:
            content = self._build_models_tab()
        elif self._tab == TAB_WORKFLOWS:
            content = self._build_workflows_tab()
        elif self._tab == TAB_CONNECTION:
            content = self._build_connection_tab()
        else:
            content = Layout(Text("Unknown tab", style="red"))

        if self._show_help:
            help_panel = self._build_help_panel()
            layout.split_column(
                Layout(tab_bar, size=1),
                Layout(help_panel, ratio=1),
                Layout(footer, size=1),
            )
        else:
            layout.split_column(
                Layout(tab_bar, size=1),
                Layout(content, ratio=1),
                Layout(footer, size=1),
            )

        return layout

    # ==================================================================
    # Main event loop
    # ==================================================================
    def run(self) -> None:
        """Start the dashboard (blocks until the user quits)."""
        reader = KeyReader(self._on_key)
        reader.start()

        self._fetch_active_tab()

        try:
            with Live(
                self._build_display(),
                console=self._console,
                screen=True,
                auto_refresh=False,
                vertical_overflow="crop",
            ) as live:
                next_fetch = time.monotonic() + self._interval
                while not self._quit.is_set():
                    live.update(self._build_display())
                    live.refresh()

                    # Short poll for responsive key handling; data is
                    # fetched only every self._interval seconds.
                    self._quit.wait(timeout=0.15)
                    if self._quit.is_set():
                        break
                    now = time.monotonic()
                    if self._force_refresh.is_set() or now >= next_fetch:
                        self._force_refresh.clear()
                        self._fetch_active_tab()
                        next_fetch = now + self._interval
        finally:
            reader.stop()

    # ------------------------------------------------------------------
    # Non-interactive rendering (for tests / piped output)
    # ------------------------------------------------------------------
    def render_once(self) -> None:
        """Fetch and print the dashboard once, then exit."""
        self._fetch_active_tab()
        self._console.print(self._build_display())
