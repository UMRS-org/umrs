"""
UMRS Monitor — Multi-tab Rich TUI for live system monitoring.

Usage (via client subcommand):
    umrs-client monitor [--interval 3] [--status running] [--project demo]
    umrs-client monitor --tab status
    umrs-client monitor --tab endpoints
    umrs-client monitor -c conf_umrs.json   # enables in-TUI reload

Tabs:
    1  Jobs        Durable job queue with detail expansion
    2  Endpoints   Endpoint provisioning jobs (filtered view)
    3  Status      Server task summaries (recompute/update/check/reload)
    4  Models      Registered model catalog
    5  Workflows   Registered workflows and their steps
    6  Connection  Active connection profile + server health
    7  Logs        Client-side activity log

Keyboard shortcuts (global):
    1-7         Switch tab
    q / Ctrl+C  Quit
    r           Force refresh
    a           Open quick actions
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

import contextlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from umrs.client.control_api import MAINTENANCE_ACTIONS, UMRSControlAPI

from .keybindings import KeyReader

# ── Tab constants ─────────────────────────────────────────────────────
TAB_JOBS = "jobs"
TAB_ENDPOINTS = "endpoints"
TAB_STATUS = "status"
TAB_MODELS = "models"
TAB_WORKFLOWS = "workflows"
TAB_CONNECTION = "connection"
TAB_LOGS = "logs"

TAB_ORDER = [TAB_JOBS, TAB_ENDPOINTS, TAB_STATUS, TAB_MODELS, TAB_WORKFLOWS, TAB_CONNECTION, TAB_LOGS]
TAB_LABELS = {
    TAB_JOBS: "Jobs",
    TAB_ENDPOINTS: "Endpoints",
    TAB_STATUS: "Status",
    TAB_MODELS: "Models",
    TAB_WORKFLOWS: "Workflows",
    TAB_CONNECTION: "Connection",
    TAB_LOGS: "Logs",
}

# ── Maximum activity log entries ─────────────────────────────────────
MAX_LOG_ENTRIES = 200

# ── Spinner frames (braille dots, cycles at ~6.7 fps) ────────────────
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ── ASCII art logo ──────────────────────────────────────────────────
# UMRS text + compact warp pipe
UMRS_ART = r"""
 ██╗   ██╗███╗   ███╗██████╗ ███████╗
 ██║   ██║████╗ ████║██╔══██╗██╔════╝
 ██║   ██║██╔████╔██║██████╔╝███████╗
 ██║   ██║██║╚██╔╝██║██╔══██╗╚════██║
 ╚██████╔╝██║ ╚═╝ ██║██║  ██║███████║
  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝
 Bring your agentic workflow to Stage 1-2
"""

PIPE_ART = r"""
███████████████████
█▒▒    ░▒  ▒▒░▒▒▒░█
█░▒    ░▒  ▒▒░▒▒▒░█
█░▒    ░▒  ▒▒░▒▒▒░█
  █░▒   ░▒  ▒▒▒░█
  █░▒   ░▒  ▒▒▒░█
  █░▒   ░▒  ▒▒▒░█
"""

LOGO_HEIGHT = max(
        len(UMRS_ART.strip("\n").split("\n")),
        len(PIPE_ART.strip("\n").split("\n")),
) + 2

# ── Status colour mapping ────────────────────────────────────────────
STATUS_STYLES: dict[str, str] = {
    "running": "bold yellow",
    "queued": "cyan",
    "planned": "dim cyan",
    "completed": "green",
    "failed": "bold red",
    "cancelled": "dim",
}

FILTER_CYCLE = [None, "running", "failed", "completed", "queued", "planned", "cancelled"]
STATUS_TASK_ORDER = ["process_objects", "wait_output", "recompute", "update", "check", "reload"]
JOB_TYPE_CHOICES = ["endpoint_provision", "index_build", "cleanup"]
MODEL_TYPE_CHOICES = ["llm", "embedding", "vision"]
MODEL_SCOPE_CHOICES = ["user", "project", "system"]
MODEL_FILTER_SCOPE_CHOICES = ["all", "user", "project", "system"]
WORKFLOW_SCOPE_CHOICES = ["user", "system"]
WORKFLOW_FILTER_SCOPE_CHOICES = ["all", "user", "system"]
BOOL_CHOICES = ["false", "true"]


def _style_status(status: str) -> Text:
    style = STATUS_STYLES.get(status, "")
    return Text(status, style=style)


def _short_id(job_id: str) -> str:
    """Return last 8 chars of an id for compact display."""
    return job_id[-8:] if len(job_id) > 8 else job_id


def _ago(iso_str: str | None) -> str:
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


def _ts_short(ts: float | None) -> str:
    """Compact HH:MM:SS from unix timestamp."""
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _one_line(text: str | None, *, limit: int = 72) -> str:
    """Collapse multiline text into a single preview line."""
    collapsed = " ".join((text or "").split()) or "(no summary)"
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _iso_label(iso_str: str | None) -> str:
    if not iso_str:
        return "—"
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(iso_str)


def _canonical_status_key(task_key: str | None) -> str:
    return str(task_key or "").strip().lower()


def _iso_sort_value(iso_str: str | None) -> tuple[int, str]:
    if not iso_str:
        return (0, "")
    try:
        parsed = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        return (1, parsed.isoformat())
    except (ValueError, TypeError):
        return (0, str(iso_str))


def _latest_iso(*values: str | None) -> str | None:
    present = [value for value in values if value]
    if not present:
        return None
    return max(present, key=_iso_sort_value)


def _status_entry_rank(entry: dict[str, Any]) -> tuple[int, int, int, tuple[int, str]]:
    data = entry.get("data") or []
    summary = str(entry.get("summary") or "")
    timestamp = entry.get("last_updated_at") or entry.get("generated_at")
    return (1 if data else 0, len(data), len(summary), _iso_sort_value(timestamp))


def _merge_status_entries(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    preferred = candidate if _status_entry_rank(candidate) >= _status_entry_rank(existing) else existing
    fallback = existing if preferred is candidate else candidate
    merged = dict(fallback)
    merged.update(preferred)
    merged["data"] = preferred.get("data") or fallback.get("data") or []
    merged["summary"] = preferred.get("summary") or fallback.get("summary") or "(no summary)"
    merged["generated_at"] = _latest_iso(existing.get("generated_at"), candidate.get("generated_at"))
    merged["last_updated_at"] = _latest_iso(
        existing.get("last_updated_at") or existing.get("generated_at"),
        candidate.get("last_updated_at") or candidate.get("generated_at"),
    )
    return merged


def _bool_from_text(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ── Dashboard ────────────────────────────────────────────────────────
class Dashboard:
    """Multi-tab live-refreshing UMRS monitor."""

    def __init__(
        self,
        control_mq: Any,
        client_id: str,
        interval: float = 3.0,
        status_filter: str | None = None,
        project_filter: str | None = None,
        console: Console | None = None,
        *,
        initial_tab: str = TAB_JOBS,
        job_type_filter: str | None = None,
        config_path: str | None = None,
        user_cfg: dict[str, Any] | None = None,
        connection_info: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
    ):
        self._mq = control_mq
        self._api = UMRSControlAPI(control_mq, client_id, config_path)
        self._client_id = client_id
        self._interval = interval
        self._status_filter = status_filter
        self._project_filter = project_filter
        self._job_type_filter = job_type_filter
        self._console = console or Console()

        # Tab state
        self._tab = initial_tab if initial_tab in TAB_ORDER else TAB_JOBS

        # Config / reload support
        self._config_path = config_path
        self._user_cfg = user_cfg or {}

        # Connection info (from ResolvedConnection.to_dict())
        self._connection_info = connection_info or {}
        self._capabilities = capabilities or {}

        # ── per-tab data caches ──
        # Jobs
        self._jobs: list[dict[str, Any]] = []
        self._jobs_cursor = 0
        self._jobs_scroll_offset = 0
        self._detail_job_id: str | None = None
        self._compact = True

        # Status
        self._status_data: dict[str, Any] = {}
        self._status_items: list[dict[str, Any]] = []
        self._status_cursor = 0
        self._status_scroll_offset = 0

        # Models
        self._models: list[dict[str, Any]] = []
        self._models_cursor = 0
        self._models_scroll_offset = 0
        self._model_detail_name: str | None = None
        self._model_scope_filter: str | None = None

        # Endpoints
        self._endpoints: list[dict[str, Any]] = []
        self._endpoints_cursor = 0
        self._endpoints_scroll_offset = 0
        self._endpoint_detail_id: str | None = None

        # Workflows
        self._workflows: list[dict[str, Any]] = []
        self._wf_cursor = 0
        self._wf_scroll_offset = 0
        self._wf_detail_name: str | None = None
        self._workflow_scope_filter: str | None = None
        self._workflow_user_filter: str | None = None
        self._workflow_name_filter: str | None = None

        # Connection
        self._ping_ok: bool | None = None
        self._ping_ms: float | None = None

        # UI state
        self._quit = threading.Event()
        self._force_refresh = threading.Event()
        self._last_error: str | None = None
        self._last_refresh: float | None = None
        self._show_help = False
        self._flash_message: str | None = None
        self._flash_expire: float = 0

        # Track which tabs have been fetched at least once
        self._fetched: dict[str, bool] = {}

        # Spinner animation frame counter
        self._spinner_frame = 0
        self._is_fetching = False

        # Activity log (client-side event capture)
        self._log_entries: list[dict[str, str]] = []
        self._log_cursor = 0
        self._log_scroll_offset = 0
        self._log_auto_scroll = True

        # Quick Actions overlay
        self._show_actions = False
        self._action_cursor = 0

        # Stats summary cache
        self._stats: dict[str, Any] = {}

        # Text search state
        self._search_active = False
        self._search_query = ""

        # Command prompt state
        self._command_active = False
        self._command_query = ""

        # Structured form overlay state
        self._form_active = False
        self._form_title = ""
        self._form_submit_action = ""
        self._form_fields: list[dict[str, Any]] = []
        self._form_cursor = 0

    # ==================================================================
    # Data fetching — per tab
    # ==================================================================
    def _fetch_active_tab(self) -> None:
        """Fetch data only for the currently active tab."""
        self._is_fetching = True
        try:
            if self._tab == TAB_JOBS:
                self._fetch_jobs()
            elif self._tab == TAB_ENDPOINTS:
                self._fetch_endpoints()
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
            self._log("ERROR", f"Fetch {self._tab}: {e}")
        self._last_refresh = time.time()
        self._fetched[self._tab] = True
        self._is_fetching = False
        self._update_stats()

    def _fetch_jobs(self) -> None:
        filters: dict[str, Any] = {}
        if self._status_filter:
            filters["status"] = self._status_filter
        if self._project_filter:
            filters["project_id"] = self._project_filter
        if self._job_type_filter:
            filters["job_type"] = self._job_type_filter
        try:
            resp = self._mq.request_job_list(self._client_id, filters)
            if resp.get("status") == "ok":
                self._jobs = resp.get("jobs", [])
                self._last_error = None
                self._log("INFO", f"Fetched {len(self._jobs)} jobs")
            else:
                self._last_error = resp.get("message", "unknown error")
        except Exception as e:
            self._last_error = str(e)
        if self._jobs:
            self._jobs_cursor = max(0, min(self._jobs_cursor, len(self._jobs) - 1))
            self._jobs_scroll_offset = min(self._jobs_scroll_offset, max(len(self._jobs) - 1, 0))
        else:
            self._jobs_cursor = 0
            self._jobs_scroll_offset = 0

    def _fetch_endpoints(self) -> None:
        try:
            parsed = self._api.list_endpoints()
            if parsed.get("status") == "ok":
                self._endpoints = parsed.get("endpoints", [])
                self._last_error = None
                self._log("INFO", f"Fetched {len(self._endpoints)} endpoints")
            else:
                self._last_error = parsed.get("message", "unknown error")
        except Exception as e:
            self._last_error = str(e)
        if self._endpoints:
            self._endpoints_cursor = max(0, min(self._endpoints_cursor, len(self._endpoints) - 1))
            self._endpoints_scroll_offset = min(self._endpoints_scroll_offset, max(len(self._endpoints) - 1, 0))
        else:
            self._endpoints_cursor = 0
            self._endpoints_scroll_offset = 0

    def _fetch_status(self) -> None:
        try:
            parsed = self._api.request_status("all")
            if parsed.get("status") == "ok":
                self._status_data = parsed
                self._status_items = self._build_status_items(parsed)
                if self._status_items:
                    self._status_cursor = max(0, min(self._status_cursor, len(self._status_items) - 1))
                    self._status_scroll_offset = min(self._status_scroll_offset, max(len(self._status_items) - 1, 0))
                else:
                    self._status_cursor = 0
                    self._status_scroll_offset = 0
                self._last_error = None
            else:
                self._last_error = parsed.get("message", "status unavailable")
        except Exception as e:
            self._last_error = str(e)

    def _fetch_models(self) -> None:
        try:
            parsed = self._api.list_models(scope=self._model_scope_filter)
            self._models = parsed.get("models", [])
            self._last_error = parsed.get("message") if parsed.get("status") == "error" else None
        except Exception as e:
            self._last_error = str(e)
        if self._models:
            self._models_cursor = max(0, min(self._models_cursor, len(self._models) - 1))
            self._models_scroll_offset = min(self._models_scroll_offset, max(len(self._models) - 1, 0))
        else:
            self._models_cursor = 0
            self._models_scroll_offset = 0

    def _fetch_workflows(self) -> None:
        try:
            parsed = self._api.list_workflows(
                scope=self._workflow_scope_filter,
                user=self._workflow_user_filter,
                name=self._workflow_name_filter,
            )
            self._workflows = parsed.get("workflows", [])
            self._last_error = parsed.get("message") if parsed.get("status") == "error" else None
        except Exception as e:
            self._last_error = str(e)
        if self._workflows:
            self._wf_cursor = max(0, min(self._wf_cursor, len(self._workflows) - 1))
            self._wf_scroll_offset = min(self._wf_scroll_offset, max(len(self._workflows) - 1, 0))
        else:
            self._wf_cursor = 0
            self._wf_scroll_offset = 0

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
        try:
            parsed = self._api.cancel_job(jid)
            if parsed.get("status") == "ok":
                self._set_flash(f"Cancelled job {_short_id(jid)}")
                self._log("INFO", f"Cancelled job {_short_id(jid)}")
            else:
                self._set_flash(f"Cancel failed: {parsed.get('message', '?')}")
                self._log("WARN", f"Cancel failed: {parsed.get('message', '?')}")
        except Exception as e:
            self._set_flash(f"Cancel error: {e}")
        self._force_refresh.set()

    def _stop_selected_endpoint(self) -> None:
        if not self._endpoints:
            return
        ep = self._endpoints[self._endpoints_cursor]
        jid = ep.get("job_id")
        ep_status = ep.get("status", "")
        if ep_status in ("completed", "failed", "cancelled"):
            self._set_flash(f"Endpoint {_short_id(jid)} already {ep_status}")
            return
        try:
            parsed = self._api.stop_endpoint(jid)
            if parsed.get("status") == "ok":
                self._set_flash(f"Stopped endpoint {_short_id(jid)}")
                self._log("INFO", f"Stopped endpoint {_short_id(jid)}")
            else:
                self._set_flash(f"Stop failed: {parsed.get('message', '?')}")
                self._log("WARN", f"Stop failed: {parsed.get('message', '?')}")
        except Exception as e:
            self._set_flash(f"Stop error: {e}")
        self._force_refresh.set()

    def _do_reload(self) -> None:
        if not self._config_path or not os.path.isfile(self._config_path):
            self._set_flash("No config file (use --conf/-c to enable reload)")
            return
        try:
            parsed = self._api.maintenance("reload")
            if parsed.get("status") == "ok":
                self._set_flash("Reload triggered successfully")
                self._log("INFO", "Server reload triggered")
            else:
                self._set_flash("Reload command rejected by server")
                self._log("WARN", "Server reload rejected")
        except Exception as e:
            self._set_flash(f"Reload error: {e}")
        self._force_refresh.set()

    def _run_maintenance(self, action: str) -> None:
        try:
            parsed = self._api.maintenance(action)
            if parsed.get("status") == "ok":
                self._set_flash(f"{action.title()} triggered")
                self._log("INFO", f"Triggered {action}")
            else:
                self._set_flash(f"{action.title()} failed: {parsed.get('message', '?')}")
                self._log("WARN", f"{action} failed: {parsed.get('message', '?')}")
        except Exception as e:
            self._set_flash(f"{action.title()} error: {e}")
            self._log("ERROR", f"{action} error: {e}")
        self._force_refresh.set()

    def _ping_server(self) -> None:
        try:
            parsed = self._api.ping(timeout_ms=3000)
            ready = bool(parsed.get("server_ready"))
            self._ping_ok = ready
            self._set_flash("Server is ready" if ready else "Server is unreachable")
            self._log("INFO", f"Ping result: {'ready' if ready else 'unreachable'}")
        except Exception as e:
            self._ping_ok = False
            self._set_flash(f"Ping error: {e}")
            self._log("ERROR", f"Ping error: {e}")

    def _run_selected_workflow(self) -> None:
        if not self._workflows:
            self._set_flash("No workflow selected")
            return
        workflow_name = self._workflows[self._wf_cursor].get("name")
        if not workflow_name:
            self._set_flash("Selected workflow has no name")
            return
        try:
            parsed = self._api.run_workflow(workflow_name=workflow_name, params={})
            if parsed.get("status") == "ok":
                self._set_flash(f"Workflow {workflow_name} started")
                self._log("INFO", f"Started workflow {workflow_name}")
            else:
                self._set_flash(f"Workflow failed: {parsed.get('message', '?')}")
                self._log("WARN", f"Workflow failed: {parsed.get('message', '?')}")
        except Exception as e:
            self._set_flash(f"Workflow error: {e}")
            self._log("ERROR", f"Workflow error: {e}")
        self._force_refresh.set()

    def _build_status_items(self, parsed: dict[str, Any]) -> list[dict[str, Any]]:
        summaries = parsed.get("summaries") or {}
        pending = {_canonical_status_key(item) for item in (parsed.get("uncompleted_key_list") or [])}
        normalized: dict[str, dict[str, Any]] = {}
        for task_key, entry in summaries.items():
            if not isinstance(entry, dict):
                continue
            canonical_key = _canonical_status_key(task_key)
            if not canonical_key:
                continue
            existing = normalized.get(canonical_key)
            normalized[canonical_key] = _merge_status_entries(existing, entry) if existing else dict(entry)

        items: list[dict[str, Any]] = []
        seen = set()
        for task_key in STATUS_TASK_ORDER:
            entry = normalized.get(task_key)
            if isinstance(entry, dict):
                seen.add(task_key)
                items.append(self._make_status_item(task_key, entry, pending))
        for task_key in sorted(normalized.keys()):
            if task_key in seen:
                continue
            entry = normalized.get(task_key)
            if isinstance(entry, dict):
                items.append(self._make_status_item(task_key, entry, pending))
        return items

    def _make_status_item(self, task_key: str, entry: dict[str, Any], pending: set[str]) -> dict[str, Any]:
        generated_at = entry.get("generated_at")
        last_updated_at = entry.get("last_updated_at") or generated_at
        process_data = entry.get("data") or []
        process_rows = self._iter_status_process_rows(process_data)
        summary = entry.get("summary") or "(no summary)"
        process_count = len(process_data)
        work_count = 0
        object_count = 0
        buckets = set()
        thread_ids = set()
        users = set()
        for process_entry in process_data:
            thread_id = process_entry.get("thread_id")
            if thread_id not in (None, ""):
                thread_ids.add(str(thread_id))
            works = process_entry.get("work") or []
            work_count += len(works)
            for work in works:
                bucket = work.get("bucket")
                if bucket:
                    buckets.add(str(bucket))
                user = work.get("user")
                if user:
                    users.add(str(user))
                with contextlib.suppress(TypeError, ValueError):
                    object_count += int(work.get("nb_objects", 0))

        if process_rows:
            preview_parts = [f"{process_count} process" + ("es" if process_count != 1 else "")]
            if work_count:
                preview_parts.append(f"{work_count} work item" + ("s" if work_count != 1 else ""))
            preview_parts.append(f"{object_count} object" + ("s" if object_count != 1 else ""))
            if buckets:
                preview_parts.append(f"{len(buckets)} bucket" + ("s" if len(buckets) != 1 else ""))
            preview = " • ".join(preview_parts)
        else:
            preview = _one_line(summary)

        return {
            "key": task_key,
            "generated_at": generated_at,
            "generated_label": str(generated_at or "—"),
            "last_updated_at": last_updated_at,
            "last_updated_label": _iso_label(last_updated_at),
            "age": _ago(last_updated_at),
            "summary": summary,
            "preview": preview,
            "data": process_data,
            "process_rows": process_rows,
            "process_count": process_count,
            "work_count": work_count,
            "object_count": object_count,
            "bucket_count": len(buckets),
            "bucket_names": sorted(buckets),
            "thread_ids": sorted(thread_ids),
            "user_count": len(users),
            "pending": _canonical_status_key(task_key) in pending,
        }

    @staticmethod
    def _iter_status_process_rows(status_data: list[dict[str, Any]]) -> list[list[str]]:
        rows: list[list[str]] = []
        for entry in status_data:
            thread_id = str(entry.get("thread_id", "—"))
            status = str(entry.get("status", "—"))
            updated_label = _iso_label(entry.get("updated_at"))
            works = entry.get("work") or []
            if not works:
                rows.append([thread_id, "—", "—", "—", "—", "—", status, updated_label])
                continue
            for work in works:
                rows.append(
                    [
                        thread_id,
                        str(work.get("work_type", "—")),
                        str(work.get("user", "—")),
                        str(work.get("nb_objects", "—")),
                        str(work.get("bucket", "—")),
                        f"{work.get('data_base', '—')}/{work.get('table_name', '—')}",
                        status,
                        updated_label,
                    ]
                )
        return rows

    def _refresh_selected_status(self) -> None:
        if not self._status_items:
            self._force_refresh.set()
            return
        selected = self._status_items[self._status_cursor]
        key = selected["key"]
        try:
            parsed = self._api.request_status(key)
            if parsed.get("status") != "ok":
                self._set_flash(f"Status refresh failed: {parsed.get('message', '?')}")
                return
            summaries = self._status_data.setdefault("summaries", {})
            if "summary" in parsed:
                summaries[key] = parsed["summary"]
            self._status_items = self._build_status_items(self._status_data)
            self._set_flash(f"Refreshed status: {key}")
            self._log("INFO", f"Refreshed status: {key}")
        except Exception as e:
            self._set_flash(f"Status refresh error: {e}")
            self._log("ERROR", f"Status refresh error: {e}")

    def _apply_control_result(self, command_text: str, parsed: dict[str, Any]) -> None:
        normalized = command_text.strip().lower()
        if parsed.get("status") == "ok":
            self._set_flash(f"Executed: {command_text}")
            self._log("INFO", f"Command ok: {command_text}")
        else:
            self._set_flash(f"Command failed: {parsed.get('message', '?')}")
            self._log("WARN", f"Command failed: {command_text} -> {parsed.get('message', '?')}")

        if "server_ready" in parsed:
            self._ping_ok = bool(parsed.get("server_ready"))
        if "summaries" in parsed or "summary" in parsed:
            if "summaries" in parsed:
                self._status_data = parsed
            elif self._status_items and "summary" in parsed:
                key = self._status_items[self._status_cursor]["key"]
                self._status_data.setdefault("summaries", {})[key] = parsed["summary"]
            self._status_items = self._build_status_items(self._status_data)
            self._switch_tab(TAB_STATUS)
        if "models" in parsed:
            self._models = parsed.get("models", [])
            self._models_cursor = min(self._models_cursor, max(len(self._models) - 1, 0))
            self._switch_tab(TAB_MODELS)
        if "workflows" in parsed:
            self._workflows = parsed.get("workflows", [])
            self._wf_cursor = min(self._wf_cursor, max(len(self._workflows) - 1, 0))
            self._switch_tab(TAB_WORKFLOWS)
        if normalized.startswith(("maintenance ", "job ", "workflow run", "model register", "model download")):
            self._force_refresh.set()

    def _start_command_prompt(self, default: str = "") -> None:
        self._command_active = True
        self._command_query = default

    def _start_form(self, title: str, submit_action: str, fields: list[dict[str, Any]]) -> None:
        self._form_active = True
        self._form_title = title
        self._form_submit_action = submit_action
        self._form_fields = [dict(field) for field in fields]
        self._form_cursor = 0

    def _close_form(self) -> None:
        self._form_active = False
        self._form_title = ""
        self._form_submit_action = ""
        self._form_fields = []
        self._form_cursor = 0

    def _cycle_form_choice(self, delta: int) -> None:
        if not self._form_fields:
            return
        field = self._form_fields[self._form_cursor]
        choices = field.get("choices") or []
        if not choices:
            return
        current = str(field.get("value", choices[0]))
        try:
            idx = choices.index(current)
        except ValueError:
            idx = 0
        field["value"] = choices[(idx + delta) % len(choices)]

    def _validate_required_fields(self, values: dict[str, str]) -> None:
        for field in self._form_fields:
            name = field["name"]
            if field.get("required") and not values.get(name, ""):
                raise ValueError(f"{field.get('label', name)} is required")
            choices = field.get("choices") or []
            if values.get(name, "") and choices and values[name] not in choices:
                raise ValueError(
                    f"{field.get('label', name)} must be one of: {', '.join(choices)}"
                )

    def _execute_form(self) -> None:
        values = {
            field["name"]: str(field.get("value", "")).strip()
            for field in self._form_fields
        }
        try:
            self._validate_required_fields(values)
            if self._form_submit_action == "submit_job_form":
                spec_raw = values.get("spec", "")
                spec = json.loads(spec_raw) if spec_raw else {}
                if spec and not isinstance(spec, dict):
                    raise ValueError("spec must be a JSON object")
                parsed = self._api.submit_job(
                    job_type=values["job_type"],
                    spec=spec,
                    project=values.get("project") or None,
                    max_attempts=int(values.get("max_attempts") or 3),
                    keep_alive=_bool_from_text(values.get("keep_alive", "false")),
                    partition=values.get("partition") or None,
                    node=values.get("node") or None,
                )
                self._apply_control_result(f"job submit {values['job_type']}", parsed)
            elif self._form_submit_action == "model_filter_form":
                scope = values.get("scope") or "all"
                self._model_scope_filter = None if scope == "all" else scope
                self._switch_tab(TAB_MODELS)
                self._force_refresh.set()
                self._set_flash("Updated model filters")
                self._log("INFO", f"Model filter scope={self._model_scope_filter or 'all'}")
            elif self._form_submit_action == "workflow_filter_form":
                scope = values.get("scope") or "all"
                self._workflow_scope_filter = None if scope == "all" else scope
                self._workflow_user_filter = values.get("user") or None
                self._workflow_name_filter = values.get("name") or None
                self._switch_tab(TAB_WORKFLOWS)
                self._force_refresh.set()
                self._set_flash("Updated workflow filters")
                self._log(
                    "INFO",
                    "Workflow filters updated"
                    f" scope={self._workflow_scope_filter or 'all'}"
                    f" user={self._workflow_user_filter or '*'}"
                    f" name={self._workflow_name_filter or '*'}",
                )
            elif self._form_submit_action == "run_workflow_form":
                params_raw = values.get("params", "")
                params = json.loads(params_raw) if params_raw else {}
                if params and not isinstance(params, dict):
                    raise ValueError("params must be a JSON object")
                parsed = self._api.run_workflow(
                    workflow_name=values["workflow_name"],
                    params=params,
                    bucket=values.get("bucket") or None,
                )
                self._apply_control_result(f"workflow run {values['workflow_name']}", parsed)
            elif self._form_submit_action == "model_download_form":
                parsed = self._api.download_model_server_side(
                    repo_id=values["repo_id"],
                    name=values.get("name") or None,
                    scope=values.get("scope") or "user",
                )
                self._apply_control_result(f"model download {values['repo_id']}", parsed)
            elif self._form_submit_action == "model_register_form":
                parsed = self._api.register_model(
                    name=values["name"],
                    path=values["path"],
                    model_type=values.get("model_type") or "embedding",
                    scope=values.get("scope") or "user",
                    description=values.get("description") or None,
                    huggingface_id=values.get("huggingface_id") or None,
                )
                self._apply_control_result(f"model register {values['name']}", parsed)
            else:
                raise ValueError(f"Unknown form action: {self._form_submit_action}")
            self._close_form()
        except Exception as e:
            self._set_flash(f"Form error: {e}")
            self._log("ERROR", f"Form error ({self._form_submit_action}): {e}")

    def _open_submit_job_form(self) -> None:
        self._start_form(
            "Submit Job",
            "submit_job_form",
            [
                {"name": "job_type", "label": "Job Type", "value": "endpoint_provision", "required": True, "choices": JOB_TYPE_CHOICES},
                {"name": "spec", "label": "Spec JSON", "value": "{}", "required": False},
                {"name": "project", "label": "Project", "value": "", "required": False},
                {"name": "max_attempts", "label": "Max Attempts", "value": "3", "required": False},
                {"name": "keep_alive", "label": "Keep Alive", "value": "false", "required": False, "choices": BOOL_CHOICES},
                {"name": "partition", "label": "Partition", "value": "", "required": False},
                {"name": "node", "label": "Node", "value": "", "required": False},
            ],
        )

    def _open_run_workflow_form(self) -> None:
        workflow_name = ""
        if self._workflows:
            workflow_name = self._workflows[self._wf_cursor].get("name", "")
        bucket_choices = ["", *sorted(self._capabilities.get("bucket_aliases") or [])]
        self._start_form(
            "Run Workflow",
            "run_workflow_form",
            [
                {"name": "workflow_name", "label": "Workflow", "value": workflow_name, "required": True},
                {"name": "params", "label": "Params JSON", "value": "{}", "required": False},
                {"name": "bucket", "label": "Bucket", "value": "", "required": False, "choices": bucket_choices},
            ],
        )

    def _open_model_download_form(self) -> None:
        self._start_form(
            "Download Model",
            "model_download_form",
            [
                {"name": "repo_id", "label": "Repo ID", "value": "", "required": True},
                {"name": "name", "label": "Name", "value": "", "required": False},
                {"name": "scope", "label": "Scope", "value": "user", "required": False, "choices": MODEL_SCOPE_CHOICES},
            ],
        )

    def _open_model_register_form(self) -> None:
        default_name = ""
        if self._models:
            default_name = self._models[self._models_cursor].get("name", "")
        self._start_form(
            "Register Model",
            "model_register_form",
            [
                {"name": "name", "label": "Name", "value": default_name, "required": True},
                {"name": "path", "label": "Path", "value": "", "required": True},
                {"name": "model_type", "label": "Type", "value": "embedding", "required": False, "choices": MODEL_TYPE_CHOICES},
                {"name": "scope", "label": "Scope", "value": "user", "required": False, "choices": MODEL_SCOPE_CHOICES},
                {"name": "description", "label": "Description", "value": "", "required": False},
                {"name": "huggingface_id", "label": "HF ID", "value": "", "required": False},
            ],
        )

    def _open_model_filter_form(self) -> None:
        self._start_form(
            "Filter Models",
            "model_filter_form",
            [
                {
                    "name": "scope",
                    "label": "Scope",
                    "value": self._model_scope_filter or "all",
                    "required": False,
                    "choices": MODEL_FILTER_SCOPE_CHOICES,
                },
            ],
        )

    def _open_workflow_filter_form(self) -> None:
        self._start_form(
            "Filter Workflows",
            "workflow_filter_form",
            [
                {
                    "name": "scope",
                    "label": "Scope",
                    "value": self._workflow_scope_filter or "all",
                    "required": False,
                    "choices": WORKFLOW_FILTER_SCOPE_CHOICES,
                },
                {"name": "user", "label": "User", "value": self._workflow_user_filter or "", "required": False},
                {"name": "name", "label": "Name", "value": self._workflow_name_filter or "", "required": False},
            ],
        )

    def _set_flash(self, msg: str, duration: float = 5.0) -> None:
        self._flash_message = msg
        self._flash_expire = time.time() + duration

    def _log(self, level: str, message: str) -> None:
        """Append an entry to the activity log."""
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self._log_entries.append(entry)
        if len(self._log_entries) > MAX_LOG_ENTRIES:
            self._log_entries = self._log_entries[-MAX_LOG_ENTRIES:]
        if self._log_auto_scroll:
            self._log_cursor = max(len(self._log_entries) - 1, 0)

    def _update_stats(self) -> None:
        """Compute summary stats from cached data."""
        job_counts: dict[str, int] = {}
        for j in self._jobs:
            s = j.get("status", "unknown")
            job_counts[s] = job_counts.get(s, 0) + 1
        self._stats = {
            "total_jobs": len(self._jobs),
            "job_counts": job_counts,
            "total_endpoints": len(self._endpoints),
            "total_models": len(self._models),
            "total_workflows": len(self._workflows),
            "connected": self._ping_ok,
        }

    # ==================================================================
    # Viewport scrolling helpers
    # ==================================================================
    def _visible_page_size(self, *, with_detail: bool = False) -> int:
        """Return the number of table rows that fit in the terminal.

        The layout has fixed-size elements that consume terminal rows:
          LOGO_HEIGHT (9) + tab_bar (1) + stats_bar (1) + footer (1)
          + summary line (1) + list/table chrome (6)
        Total fixed overhead = LOGO_HEIGHT + 9.

        Rich tables render extra chrome inside the outer panel beyond the
        visible body rows, so the viewport budget needs to leave room for
        those lines as well or scrolling starts a couple of rows too late.

        When a detail panel is open the content area is split ratio 2:1,
        so the table panel gets only ⅔ of the remaining space.
        """
        try:
            term_h = self._console.size.height
        except Exception:
            term_h = 24
        overhead = LOGO_HEIGHT + 9  # 18 for default LOGO_HEIGHT=9
        available = term_h - overhead
        if with_detail:
            available = (available * 2) // 3
        return max(available, 3)

    def _current_page_size(self) -> int:
        """Page size for the currently active tab, accounting for detail panels."""
        detail = False
        if self._tab == TAB_JOBS and self._detail_job_id:
            detail = True
        elif self._tab == TAB_ENDPOINTS and self._endpoint_detail_id:
            detail = True
        elif self._tab == TAB_MODELS and self._model_detail_name:
            detail = True
        elif self._tab == TAB_WORKFLOWS and self._wf_detail_name:
            detail = True
        return self._visible_page_size(with_detail=detail)

    @staticmethod
    def _adjust_scroll_offset(cursor: int, offset: int, page_size: int) -> int:
        """Return a new scroll offset that keeps *cursor* within the visible window."""
        if cursor < offset:
            return cursor
        if cursor >= offset + page_size:
            return cursor - page_size + 1
        return offset

    # ==================================================================
    # Keyboard handler
    # ==================================================================
    def _move_down(self) -> None:
        page = self._current_page_size()
        if self._tab == TAB_JOBS:
            self._jobs_cursor = min(self._jobs_cursor + 1, max(len(self._jobs) - 1, 0))
            self._jobs_scroll_offset = self._adjust_scroll_offset(self._jobs_cursor, self._jobs_scroll_offset, page)
        elif self._tab == TAB_ENDPOINTS:
            self._endpoints_cursor = min(self._endpoints_cursor + 1, max(len(self._endpoints) - 1, 0))
            self._endpoints_scroll_offset = self._adjust_scroll_offset(self._endpoints_cursor, self._endpoints_scroll_offset, page)
        elif self._tab == TAB_STATUS:
            self._status_cursor = min(self._status_cursor + 1, max(len(self._status_items) - 1, 0))
            self._status_scroll_offset = self._adjust_scroll_offset(self._status_cursor, self._status_scroll_offset, page)
        elif self._tab == TAB_MODELS:
            self._models_cursor = min(self._models_cursor + 1, max(len(self._models) - 1, 0))
            self._models_scroll_offset = self._adjust_scroll_offset(self._models_cursor, self._models_scroll_offset, page)
        elif self._tab == TAB_WORKFLOWS:
            self._wf_cursor = min(self._wf_cursor + 1, max(len(self._workflows) - 1, 0))
            self._wf_scroll_offset = self._adjust_scroll_offset(self._wf_cursor, self._wf_scroll_offset, page)
        elif self._tab == TAB_LOGS:
            self._log_auto_scroll = False
            self._log_cursor = min(self._log_cursor + 1, max(len(self._log_entries) - 1, 0))
            self._log_scroll_offset = self._adjust_scroll_offset(self._log_cursor, self._log_scroll_offset, page)

    def _move_up(self) -> None:
        page = self._current_page_size()
        if self._tab == TAB_JOBS:
            self._jobs_cursor = max(self._jobs_cursor - 1, 0)
            self._jobs_scroll_offset = self._adjust_scroll_offset(self._jobs_cursor, self._jobs_scroll_offset, page)
        elif self._tab == TAB_ENDPOINTS:
            self._endpoints_cursor = max(self._endpoints_cursor - 1, 0)
            self._endpoints_scroll_offset = self._adjust_scroll_offset(self._endpoints_cursor, self._endpoints_scroll_offset, page)
        elif self._tab == TAB_STATUS:
            self._status_cursor = max(self._status_cursor - 1, 0)
            self._status_scroll_offset = self._adjust_scroll_offset(self._status_cursor, self._status_scroll_offset, page)
        elif self._tab == TAB_MODELS:
            self._models_cursor = max(self._models_cursor - 1, 0)
            self._models_scroll_offset = self._adjust_scroll_offset(self._models_cursor, self._models_scroll_offset, page)
        elif self._tab == TAB_WORKFLOWS:
            self._wf_cursor = max(self._wf_cursor - 1, 0)
            self._wf_scroll_offset = self._adjust_scroll_offset(self._wf_cursor, self._wf_scroll_offset, page)
        elif self._tab == TAB_LOGS:
            self._log_auto_scroll = False
            self._log_cursor = max(self._log_cursor - 1, 0)
            self._log_scroll_offset = self._adjust_scroll_offset(self._log_cursor, self._log_scroll_offset, page)

    def _toggle_detail(self) -> None:
        if self._tab == TAB_JOBS and self._jobs:
            jid = self._jobs[self._jobs_cursor].get("job_id")
            self._detail_job_id = None if self._detail_job_id == jid else jid
        elif self._tab == TAB_ENDPOINTS and self._endpoints:
            jid = self._endpoints[self._endpoints_cursor].get("job_id")
            self._endpoint_detail_id = None if self._endpoint_detail_id == jid else jid
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
        # If search is active, handle search input
        if self._search_active:
            if key in ("\r", "\n", "\x1b", "esc"):
                self._search_active = False
            elif key == "\x7f":  # backspace
                self._search_query = self._search_query[:-1]
            elif len(key) == 1 and key.isprintable():
                self._search_query += key
            return

        if self._form_active:
            if key in ("\x1b", "esc"):
                self._close_form()
            elif key in ("l", "right", "]"):
                self._cycle_form_choice(1)
            elif key in ("h", "left", "["):
                self._cycle_form_choice(-1)
            elif key in ("j", "down", "\t"):
                self._form_cursor = min(self._form_cursor + 1, len(self._form_fields) - 1)
            elif key in ("k", "up"):
                self._form_cursor = max(self._form_cursor - 1, 0)
            elif key in ("\r", "\n"):
                if self._form_cursor == len(self._form_fields) - 1:
                    self._execute_form()
                else:
                    self._form_cursor = min(self._form_cursor + 1, len(self._form_fields) - 1)
            elif key == "\x7f":
                if self._form_fields:
                    value = str(self._form_fields[self._form_cursor].get("value", ""))
                    self._form_fields[self._form_cursor]["value"] = value[:-1]
            elif len(key) == 1 and key.isprintable() and self._form_fields:
                field = self._form_fields[self._form_cursor]
                if field.get("choices"):
                    # Field with choices: ignore free text, use h/l to cycle
                    pass
                else:
                    value = str(field.get("value", ""))
                    field["value"] = value + key
            return

        # If quick actions overlay is open, handle it
        if self._show_actions:
            if key in ("\x1b", "esc", "a", "q"):
                self._show_actions = False
            elif key in ("j", "down"):
                self._action_cursor = min(self._action_cursor + 1, len(self._get_actions()) - 1)
            elif key in ("k", "up"):
                self._action_cursor = max(self._action_cursor - 1, 0)
            elif key in ("\r", "\n"):
                self._execute_action(self._action_cursor)
                self._show_actions = False
            return

        # Tab switching via number keys
        tab_keys = {
            "1": TAB_JOBS, "2": TAB_ENDPOINTS, "3": TAB_STATUS, "4": TAB_MODELS,
            "5": TAB_WORKFLOWS, "6": TAB_CONNECTION, "7": TAB_LOGS,
        }
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
            "a": lambda: self._toggle_actions(),
            "s": lambda: self._start_search(),
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
            if key in ("j", "down"):
                self._move_down()
            elif key in ("k", "up"):
                self._move_up()
            elif key in ("\r", "\n"):
                self._refresh_selected_status()
            elif key == "R":
                self._do_reload()
        elif self._tab in (TAB_ENDPOINTS, TAB_MODELS, TAB_WORKFLOWS):
            nav_actions = {
                "j": self._move_down,
                "down": self._move_down,
                "k": self._move_up,
                "up": self._move_up,
                "\r": self._toggle_detail,
                "\n": self._toggle_detail,
                "/": lambda: setattr(self, "_compact", not self._compact),
            }
            act = nav_actions.get(key)
            if act:
                act()
            # Endpoints-specific: 'c' cancels / stops the selected endpoint
            if self._tab == TAB_ENDPOINTS and key == "c":
                self._stop_selected_endpoint()
        elif self._tab == TAB_LOGS:
            log_actions = {
                "j": self._move_down,
                "down": self._move_down,
                "k": self._move_up,
                "up": self._move_up,
                "G": lambda: self._log_scroll_end(),
                "g": lambda: self._log_scroll_top(),
            }
            act = log_actions.get(key)
            if act:
                act()

    def _toggle_actions(self) -> None:
        self._show_actions = not self._show_actions
        self._action_cursor = 0

    def _start_search(self) -> None:
        self._search_active = True
        self._search_query = ""

    def _log_scroll_end(self) -> None:
        self._log_cursor = max(len(self._log_entries) - 1, 0)
        self._log_auto_scroll = True
        page = self._visible_page_size()
        self._log_scroll_offset = max(len(self._log_entries) - page, 0)

    def _log_scroll_top(self) -> None:
        self._log_cursor = 0
        self._log_scroll_offset = 0
        self._log_auto_scroll = False

    def _get_actions(self) -> list[dict[str, str]]:
        """Return available command actions for current context."""
        actions = [
            {"label": "Force Refresh", "key": "r", "action": "refresh"},
            {"label": "Ping Server", "key": "", "action": "ping"},
            {"label": "Toggle Help", "key": "?", "action": "help"},
            {"label": "Submit Job...", "key": "", "action": "submit_job_form"},
            {"label": "Download Model...", "key": "", "action": "model_download_form"},
            {"label": "Register Model...", "key": "", "action": "model_register_form"},
        ]
        allowed_actions = self._capabilities.get("allowed_actions", {})
        if self._capabilities.get("is_admin"):
            for label, action in (
                ("Trigger Recompute", "recompute"),
                ("Trigger Update", "update"),
                ("Run Consistency Check", "check"),
            ):
                if allowed_actions.get(action, True):
                    actions.append({"label": label, "key": "", "action": action})
        if self._config_path:
            actions.append({"label": "Reload Server Config", "key": "R", "action": "reload"})
        if self._tab == TAB_STATUS and self._status_items:
            actions.append({"label": f"Refresh Status {self._status_items[self._status_cursor]['key']}", "key": "Enter", "action": "refresh_status"})
        if self._tab == TAB_JOBS and self._jobs:
            job = self._jobs[self._jobs_cursor]
            if job.get("status") in ("running", "queued", "planned"):
                actions.append({"label": f"Cancel Job {_short_id(job.get('job_id', ''))}", "key": "c", "action": "cancel_job"})
        if self._tab == TAB_ENDPOINTS and self._endpoints:
            ep = self._endpoints[self._endpoints_cursor]
            if ep.get("status") in ("running", "queued", "planned"):
                actions.append({"label": f"Stop Endpoint {_short_id(ep.get('job_id', ''))}", "key": "c", "action": "stop_endpoint"})
        if self._tab == TAB_WORKFLOWS and self._workflows:
            workflow_name = self._workflows[self._wf_cursor].get("name", "selected")
            actions.append({"label": f"Run Workflow {workflow_name}", "key": "", "action": "run_selected_workflow"})
            actions.append({"label": f"Run Workflow {workflow_name} With Params...", "key": "", "action": "run_workflow_form"})
        if self._tab == TAB_MODELS:
            actions.append({"label": "Filter Models...", "key": "", "action": "model_filter_form"})
        if self._tab == TAB_WORKFLOWS:
            actions.append({"label": "Filter Workflows...", "key": "", "action": "workflow_filter_form"})
        actions.extend([
            {"label": "Show Jobs Tab", "key": "1", "action": "show_jobs"},
            {"label": "Show Endpoints Tab", "key": "2", "action": "show_endpoints"},
            {"label": "Show Status Tab", "key": "3", "action": "show_status"},
            {"label": "Show Models Tab", "key": "4", "action": "show_models"},
            {"label": "Show Workflows Tab", "key": "5", "action": "show_workflows"},
            {"label": "Show Connection Tab", "key": "6", "action": "show_connection"},
            {"label": "Show Logs Tab", "key": "7", "action": "show_logs"},
        ])
        actions.append({"label": "Clear Log", "key": "", "action": "clear_log"})
        return actions

    def _execute_action(self, idx: int) -> None:
        """Execute a quick action by index."""
        actions = self._get_actions()
        if idx < 0 or idx >= len(actions):
            return
        action = actions[idx]["action"]
        if action == "refresh":
            self._force_refresh.set()
        elif action == "ping":
            self._ping_server()
        elif action == "help":
            self._show_help = not self._show_help
        elif action == "submit_job_form":
            self._open_submit_job_form()
        elif action in MAINTENANCE_ACTIONS:
            self._run_maintenance(action)
        elif action == "reload":
            self._do_reload()
        elif action == "run_workflow_form":
            self._open_run_workflow_form()
        elif action == "refresh_status":
            self._refresh_selected_status()
        elif action == "cancel_job":
            self._cancel_selected_job()
        elif action == "stop_endpoint":
            self._stop_selected_endpoint()
        elif action == "model_download_form":
            self._open_model_download_form()
        elif action == "model_register_form":
            self._open_model_register_form()
        elif action == "model_filter_form":
            self._open_model_filter_form()
        elif action == "workflow_filter_form":
            self._open_workflow_filter_form()
        elif action == "run_selected_workflow":
            self._run_selected_workflow()
        elif action == "show_jobs":
            self._switch_tab(TAB_JOBS)
        elif action == "show_endpoints":
            self._switch_tab(TAB_ENDPOINTS)
        elif action == "show_status":
            self._switch_tab(TAB_STATUS)
        elif action == "show_models":
            self._switch_tab(TAB_MODELS)
        elif action == "show_workflows":
            self._switch_tab(TAB_WORKFLOWS)
        elif action == "show_connection":
            self._switch_tab(TAB_CONNECTION)
        elif action == "show_logs":
            self._switch_tab(TAB_LOGS)
        elif action == "clear_log":
            self._log_entries.clear()
            self._log_cursor = 0
            self._set_flash("Activity log cleared")

    # ==================================================================
    # Rendering — Header and Footer
    # ==================================================================
    def _build_logo(self) -> Text:
        """Render the UMRS ASCII-art logo with a compact pipe."""
        logo = Text()
        umrs_lines = UMRS_ART.strip("\n").split("\n")
        pipe_lines = PIPE_ART.strip("\n").split("\n")
        umrs_width = max(len(line) for line in umrs_lines)
        total_lines = max(len(umrs_lines), len(pipe_lines))

        for index in range(total_lines):
            umrs_line = umrs_lines[index] if index < len(umrs_lines) else ""
            pipe_line = pipe_lines[index] if index < len(pipe_lines) else ""
            logo.append(umrs_line.ljust(umrs_width + 4), style="bold bright_cyan")
            logo.append(pipe_line, style="bold green")
            logo.append("\n")
        return logo

    def _build_tab_bar(self) -> Text:
        bar = Text()
        # Animated spinner
        frame = SPINNER_FRAMES[self._spinner_frame % len(SPINNER_FRAMES)]
        self._spinner_frame += 1
        if self._is_fetching:
            bar.append(f" {frame} ", style="bold bright_cyan")
        else:
            bar.append(f" {frame} ", style="dim cyan")
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
        if self._search_active:
            return Text.from_markup(
                f"[bold cyan]Search:[/bold cyan] {self._search_query}[blink]_[/blink]  "
                "[dim]Enter[/dim] confirm  [dim]Esc[/dim] cancel"
            )

        if self._form_active:
            field = self._form_fields[self._form_cursor] if self._form_fields else {"label": "field"}
            choices = field.get("choices") or []
            choice_hint = f"  [dim]choices[/dim]={','.join(choices)}  [dim]h/l[/dim] cycle" if choices else ""
            return Text.from_markup(
                f"[bold magenta]Form:[/bold magenta] {self._form_title}  "
                f"[dim]field[/dim]={field.get('label', '?')}  "
                "[dim]j/k/Tab[/dim] move  [dim]Enter[/dim] next-submit  [dim]Esc[/dim] cancel"
                f"{choice_hint}"
            )

        common = "[dim]a[/dim] actions  [dim]?[/dim] help  [dim]1-6[/dim] tabs"
        if self._tab == TAB_JOBS:
            return Text.from_markup(
                "[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]j/k[/dim] navigate  "
                "[dim]Enter[/dim] detail  [dim]f[/dim] filter  [dim]/[/dim] compact  "
                f"[dim]c[/dim] cancel  {common}"
            )
        elif self._tab == TAB_STATUS:
            reload_hint = "[dim]R[/dim] reload" if self._config_path else "[dim strikethrough]R reload (no --conf)[/dim strikethrough]"
            return Text.from_markup(
                f"[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]j/k[/dim] navigate  [dim]Enter[/dim] refresh item  {reload_hint}  [dim]a[/dim] actions  [dim]?[/dim] help"
            )
        elif self._tab in (TAB_MODELS, TAB_WORKFLOWS):
            return Text.from_markup(
                "[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]j/k[/dim] navigate  "
                f"[dim]Enter[/dim] detail  [dim]/[/dim] compact  {common}"
            )
        elif self._tab == TAB_LOGS:
            return Text.from_markup(
                "[dim]q[/dim] quit  [dim]r[/dim] refresh  [dim]j/k[/dim] scroll  "
                f"[dim]g[/dim] top  [dim]G[/dim] bottom  {common}"
            )
        else:
            return Text.from_markup(
                f"[dim]q[/dim] quit  [dim]r[/dim] refresh  {common}"
            )

    # ==================================================================
    # Rendering — Help overlay
    # ==================================================================
    def _build_help_panel(self) -> Panel:
        t = Table(show_header=True, header_style="bold", expand=True, show_lines=False, padding=(0, 2))
        t.add_column("Key", style="bold cyan", width=12)
        t.add_column("Action")
        t.add_column("Tab", style="dim")

        t.add_row("1-7", "Switch tab", "Global")
        t.add_row("q / Ctrl+C", "Quit", "Global")
        t.add_row("r", "Force refresh", "Global")
        t.add_row("a", "Quick actions menu", "Global")
        t.add_row("s", "Text search", "Global")
        t.add_row("Tab", "Move to next form field", "Forms")
        t.add_row("h / l", "Cycle form choices", "Forms")
        t.add_row("?", "Toggle this help", "Global")
        t.add_row("", "", "")
        t.add_row("j / ↓", "Move selection down", "Jobs, Endpoints, Status, Models, Workflows, Logs")
        t.add_row("k / ↑", "Move selection up", "Jobs, Endpoints, Status, Models, Workflows, Logs")
        t.add_row("Enter", "Toggle detail view", "Jobs, Endpoints, Models, Workflows")
        t.add_row("Enter", "Refresh selected status", "Status")
        t.add_row("f", "Cycle status filter", "Jobs")
        t.add_row("/", "Toggle compact/wide", "Jobs, Endpoints, Models, Workflows")
        t.add_row("c", "Cancel selected job / stop endpoint", "Jobs, Endpoints")
        t.add_row("R", "Trigger server reload", "Status (requires --conf)")
        t.add_row("g / G", "Scroll to top / bottom", "Logs")

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

        page = self._visible_page_size(with_detail=bool(self._detail_job_id))
        self._jobs_scroll_offset = self._adjust_scroll_offset(self._jobs_cursor, self._jobs_scroll_offset, page)
        start = self._jobs_scroll_offset
        end = min(start + page, len(self._jobs))

        for idx in range(start, end):
            job = self._jobs[idx]
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

    def _build_job_detail_panel(self) -> Panel | None:
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
        type_label = self._job_type_filter or "all"
        total = len(self._jobs)
        pos_hint = f"  [dim]{self._jobs_cursor + 1}/{total}[/dim]" if total else ""
        summary = Text.from_markup(
            f"  filter=[cyan]{filter_label}[/cyan]  type=[cyan]{type_label}[/cyan]  total=[yellow]{total}[/yellow]{pos_hint}"
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
    # Rendering — Endpoints tab
    # ==================================================================
    @staticmethod
    def _format_idle_ttl(seconds: int | None) -> str:
        if not seconds:
            return ""
        if seconds % 86400 == 0:
            return f"{seconds // 86400}d"
        if seconds % 3600 == 0:
            return f"{seconds // 3600}h"
        if seconds % 60 == 0:
            return f"{seconds // 60}m"
        return f"{seconds}s"

    def _build_endpoints_tab(self) -> Layout:
        layout = Layout()

        table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold bright_white")
        table.add_column("#", width=3, justify="right")
        table.add_column("Job ID", min_width=10)
        table.add_column("Status", min_width=10)
        table.add_column("Model", min_width=20, style="bold")
        table.add_column("Idle TTL", width=9)
        table.add_column("Backend", min_width=12)
        table.add_column("Created", min_width=10)

        page = self._visible_page_size(with_detail=bool(self._endpoint_detail_id))
        self._endpoints_scroll_offset = self._adjust_scroll_offset(self._endpoints_cursor, self._endpoints_scroll_offset, page)
        start = self._endpoints_scroll_offset
        end = min(start + page, len(self._endpoints))

        for idx in range(start, end):
            ep = self._endpoints[idx]
            is_selected = idx == self._endpoints_cursor
            row_style = "reverse" if is_selected else ""
            spec = ep.get("spec") or {}
            ep_status = ep.get("status", "?")
            backend = ""
            for attempt in reversed(ep.get("attempts") or []):
                if attempt.get("backend_ref"):
                    backend = str(attempt["backend_ref"])
                    break
            table.add_row(
                str(idx + 1),
                _short_id(ep.get("job_id", "")),
                _style_status(ep_status),
                spec.get("model", "—"),
                self._format_idle_ttl(spec.get("idle_ttl_sec")),
                Text(backend or "—", style="dim"),
                _ago(ep.get("created_at")),
                style=row_style,
            )

        total = len(self._endpoints)
        running = sum(1 for e in self._endpoints if e.get("status") == "running")
        summary = Text.from_markup(f"  total=[yellow]{total}[/yellow]")
        if running:
            summary.append_text(Text.from_markup(f"  [green]{running} running[/green]"))
        if total:
            summary.append_text(Text.from_markup(f"  [dim]{self._endpoints_cursor + 1}/{total}[/dim]"))

        detail = self._build_endpoint_detail_panel()

        if not self._endpoints:
            empty = Panel(
                Text("No endpoint jobs found. Use: umrs-client endpoint up ...", style="dim"),
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

    def _build_endpoint_detail_panel(self) -> Panel | None:
        if not self._endpoint_detail_id or not self._endpoints:
            return None
        ep = next((e for e in self._endpoints if e.get("job_id") == self._endpoint_detail_id), None)
        if not ep:
            return None

        detail_table = Table(show_header=False, expand=True, padding=(0, 2))
        detail_table.add_column("Field", style="bold", width=18)
        detail_table.add_column("Value", ratio=1)

        spec = ep.get("spec") or {}
        detail_table.add_row("job_id", ep.get("job_id", ""))
        detail_table.add_row("status", ep.get("status", ""))
        detail_table.add_row("project_id", ep.get("project_id", ""))
        detail_table.add_row("model", spec.get("model", "—"))
        detail_table.add_row("model_path", spec.get("model_path", "—"))
        detail_table.add_row("idle_ttl", self._format_idle_ttl(spec.get("idle_ttl_sec")) or "—")
        detail_table.add_row("tier", spec.get("tier", "—"))
        detail_table.add_row("gpu_count", str(spec.get("gpu_count", "—")))
        detail_table.add_row("max_model_len", str(spec.get("max_model_len", "—")))
        detail_table.add_row("created_at", _iso_label(ep.get("created_at")))

        attempts = ep.get("attempts") or []
        if attempts:
            latest = attempts[-1]
            detail_table.add_row("", "")
            detail_table.add_row("attempt #", str(latest.get("attempt_number", len(attempts))))
            detail_table.add_row("attempt_status", latest.get("status", "—"))
            detail_table.add_row("backend_ref", latest.get("backend_ref") or "—")
            detail_table.add_row("started_at", _iso_label(latest.get("started_at")))

        return Panel(
            detail_table,
            title=f"[bold]Endpoint: {_short_id(ep.get('job_id', ''))}[/bold]",
            border_style="bright_blue",
        )

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
        uncompleted = self._status_data.get("uncompleted_key_list") or []

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

        if not self._status_items:
            layout.split_column(
                Layout(header, size=1),
                Layout(Panel(Text("No status entries available.", style="dim"), border_style="dim"), ratio=1),
            )
            return layout

        selected = self._status_items[self._status_cursor]

        list_table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold bright_white")
        list_table.add_column("#", width=3, justify="right")
        list_table.add_column("Task", min_width=14, style="bold")
        list_table.add_column("Last Update", min_width=19)
        list_table.add_column("Age", width=10)
        list_table.add_column("State", width=9)
        list_table.add_column("Preview", ratio=1)

        # Status tab always shows a detail panel beside the list (split_row 2:3)
        page = self._visible_page_size()
        self._status_scroll_offset = self._adjust_scroll_offset(self._status_cursor, self._status_scroll_offset, page)
        start = self._status_scroll_offset
        end = min(start + page, len(self._status_items))

        for idx in range(start, end):
            item = self._status_items[idx]
            row_style = "reverse" if idx == self._status_cursor else ""
            state = "pending" if item["pending"] else "ready"
            state_style = "yellow" if item["pending"] else "green"
            list_table.add_row(
                str(idx + 1),
                item["key"],
                item["last_updated_label"],
                item["age"],
                Text(state, style=state_style),
                item["preview"],
                style=row_style,
            )

        meta_table = Table(show_header=False, expand=True, padding=(0, 2))
        meta_table.add_column("Field", style="bold", width=14)
        meta_table.add_column("Value", ratio=1)
        meta_table.add_row("Task", selected["key"])
        meta_table.add_row("Generated", selected["generated_label"])
        meta_table.add_row("Last Update", selected["last_updated_label"])
        meta_table.add_row("Age", selected["age"])
        meta_table.add_row("State", "pending" if selected["pending"] else "ready")
        meta_table.add_row("Preview", selected["preview"])
        if selected["thread_ids"]:
            meta_table.add_row("Threads", ", ".join(selected["thread_ids"]))
        if selected["bucket_names"]:
            meta_table.add_row("Bucket Names", ", ".join(selected["bucket_names"]))
        if selected["process_rows"]:
            meta_table.add_row("Processes", str(selected["process_count"]))
            meta_table.add_row("Work Items", str(selected["work_count"]))
            meta_table.add_row("Objects", str(selected["object_count"]))
            meta_table.add_row("Buckets", str(selected["bucket_count"]))
        if client_label:
            meta_table.add_row("Client", client_label)

        detail_group = [meta_table]
        process_rows = selected["process_rows"]
        if process_rows:
            process_table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold bright_white")
            process_table.add_column("Thread", min_width=8)
            process_table.add_column("Work", min_width=12)
            process_table.add_column("User", min_width=8)
            process_table.add_column("Objects", width=8, justify="right")
            process_table.add_column("Bucket", min_width=12)
            process_table.add_column("DB/Table", min_width=18)
            process_table.add_column("Status", min_width=8)
            process_table.add_column("Last Update", min_width=19)
            for row in process_rows:
                process_table.add_row(*row)
            detail_group.extend([Text(""), process_table])
        elif selected["summary"]:
            detail_group.extend([Text(""), Text(selected["summary"])])
        if uncompleted:
            detail_group.extend([
                Text(""),
                Text(f"Pending tasks: {', '.join(map(str, uncompleted))}", style="yellow"),
            ])

        body = Layout()
        body.split_row(
            Layout(Panel(list_table, title="[bold]Status Tasks[/bold]", border_style="bright_white"), ratio=2),
            Layout(Panel(Group(*detail_group), title=f"[bold]Details: {selected['key']}[/bold]", border_style="bright_blue"), ratio=3),
        )

        layout.split_column(
            Layout(header, size=1),
            Layout(body, ratio=1),
        )
        return layout

    # ==================================================================
    # Helpers — Models tab
    # ==================================================================
    @staticmethod
    def _colorize_model_type(model_type: str) -> str:
        """Return Rich-markup colored model type string."""
        colors = {"llm": "bright_green", "embedding": "bright_cyan", "vision": "bright_magenta"}
        color = colors.get(model_type, "white")
        return f"[{color}]{model_type}[/{color}]"

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

        page = self._visible_page_size(with_detail=bool(self._model_detail_name))
        self._models_scroll_offset = self._adjust_scroll_offset(self._models_cursor, self._models_scroll_offset, page)
        start = self._models_scroll_offset
        end = min(start + page, len(self._models))

        for idx in range(start, end):
            m = self._models[idx]
            is_selected = idx == self._models_cursor
            row_style = "reverse" if is_selected else ""
            table.add_row(
                str(idx + 1),
                m.get("name", "?"),
                self._colorize_model_type(m.get("model_type", m.get("type", "?"))),
                m.get("scope", "?"),
                m.get("huggingface_id", "—"),
                m.get("registered_by", "—"),
                _ago(m.get("registered_at")),
                style=row_style,
            )

        total = len(self._models)
        summary = Text.from_markup(f"  total=[yellow]{total}[/yellow]")
        if total:
            summary.append_text(Text.from_markup(f"  [dim]{self._models_cursor + 1}/{total}[/dim]"))
        if self._model_scope_filter:
            summary.append(f"  scope={self._model_scope_filter}")
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

    def _build_model_detail_panel(self) -> Panel | None:
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

        page = self._visible_page_size(with_detail=bool(self._wf_detail_name))
        self._wf_scroll_offset = self._adjust_scroll_offset(self._wf_cursor, self._wf_scroll_offset, page)
        start = self._wf_scroll_offset
        end = min(start + page, len(self._workflows))

        for idx in range(start, end):
            wf = self._workflows[idx]
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

        total = len(self._workflows)
        summary = Text.from_markup(f"  total=[yellow]{total}[/yellow]")
        if total:
            summary.append_text(Text.from_markup(f"  [dim]{self._wf_cursor + 1}/{total}[/dim]"))
        if self._workflow_scope_filter:
            summary.append(f"  scope={self._workflow_scope_filter}")
        if self._workflow_user_filter:
            summary.append(f"  user={self._workflow_user_filter}")
        if self._workflow_name_filter:
            summary.append(f"  name={self._workflow_name_filter}")
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

    def _build_workflow_detail_panel(self) -> Panel | None:
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
    # Rendering — Logs tab (3A: Activity log viewer)
    # ==================================================================
    def _build_logs_tab(self) -> Layout:
        layout = Layout()

        LOG_LEVEL_STYLES = {"INFO": "green", "WARN": "yellow", "ERROR": "bold red", "DEBUG": "dim"}

        table = Table(expand=True, show_lines=False, padding=(0, 1), header_style="bold bright_white")
        table.add_column("Time", width=10, style="dim")
        table.add_column("Level", width=7)
        table.add_column("Message", ratio=1)

        entries = self._log_entries
        # Apply search filter if active
        if self._search_query:
            q = self._search_query.lower()
            entries = [e for e in entries if q in e.get("message", "").lower()]

        # Clamp cursor to filtered entries to prevent out-of-bounds selection
        if entries:
            self._log_cursor = min(self._log_cursor, len(entries) - 1)
        else:
            self._log_cursor = 0

        page = self._visible_page_size()
        if self._log_auto_scroll and entries:
            # Keep the view pinned to the bottom when auto-scrolling
            self._log_cursor = len(entries) - 1
            self._log_scroll_offset = max(len(entries) - page, 0)
        else:
            self._log_scroll_offset = self._adjust_scroll_offset(self._log_cursor, self._log_scroll_offset, page)
        start = self._log_scroll_offset
        end = min(start + page, len(entries))

        for idx in range(start, end):
            entry = entries[idx]
            is_selected = idx == self._log_cursor
            level = entry.get("level", "INFO")
            level_style = LOG_LEVEL_STYLES.get(level, "")
            row_style = "reverse" if is_selected else ""
            table.add_row(
                entry.get("time", ""),
                Text(level, style=level_style),
                entry.get("message", ""),
                style=row_style,
            )

        count = len(entries)
        scroll_hint = "[dim]auto-scroll[/dim]" if self._log_auto_scroll else f"[dim]{self._log_cursor + 1}/{count}[/dim]"
        summary = Text.from_markup(f"  entries=[yellow]{count}[/yellow]  {scroll_hint}")

        if not entries:
            empty = Panel(
                Text("No activity logged yet. Events will appear here as you interact.", style="dim"),
                border_style="dim",
            )
            layout.split_column(
                Layout(summary, size=1),
                Layout(empty, ratio=1),
            )
        else:
            layout.split_column(
                Layout(summary, size=1),
                Layout(Panel(table, border_style="bright_white"), ratio=1),
            )
        return layout

    # ==================================================================
    # Rendering — Quick Actions overlay (3B)
    # ==================================================================
    def _build_actions_overlay(self) -> Panel:
        actions = self._get_actions()
        t = Table(show_header=False, expand=True, padding=(0, 2))
        t.add_column("", width=3)
        t.add_column("Action", ratio=1)
        t.add_column("Key", width=8, style="dim cyan")

        for idx, act in enumerate(actions):
            marker = "▸" if idx == self._action_cursor else " "
            style = "bold white" if idx == self._action_cursor else ""
            t.add_row(marker, act["label"], act.get("key", ""), style=style)

        return Panel(
            t,
            title="[bold]Command Actions[/bold]  [dim]j/k navigate, Enter select, a/Esc close[/dim]",
            border_style="bright_yellow",
            padding=(1, 2),
        )

    def _build_form_overlay(self) -> Panel:
        form_table = Table(show_header=False, expand=True, padding=(0, 2))
        form_table.add_column("", width=3)
        form_table.add_column("Field", width=16, style="bold")
        form_table.add_column("Value", ratio=1)

        for idx, field in enumerate(self._form_fields):
            marker = "▸" if idx == self._form_cursor else " "
            style = "bold white" if idx == self._form_cursor else ""
            suffix = " *" if field.get("required") else ""
            value = str(field.get("value", ""))
            if idx == self._form_cursor:
                value = value + "_"
            form_table.add_row(marker, f"{field.get('label', field['name'])}{suffix}", value, style=style)

        return Panel(
            form_table,
            title=f"[bold]{self._form_title}[/bold]  [dim]Enter moves/submit, Tab next, Esc close[/dim]",
            border_style="bright_magenta",
            padding=(1, 2),
        )

    # ==================================================================
    # Rendering — Stats bar (3C: persistent summary)
    # ==================================================================
    def _build_stats_bar(self) -> Text:
        bar = Text()
        s = self._stats
        if not s:
            return bar

        bar.append("  ")
        # Jobs summary
        total_jobs = s.get("total_jobs", 0)
        counts = s.get("job_counts", {})
        running = counts.get("running", 0)
        failed = counts.get("failed", 0)
        bar.append(f"Jobs: {total_jobs}", style="bold")
        if running:
            bar.append(f" ({running} running)", style="yellow")
        if failed:
            bar.append(f" ({failed} failed)", style="red")

        bar.append("  │  ", style="dim")
        # Endpoints
        bar.append(f"Endpoints: {s.get('total_endpoints', 0)}", style="bold")
        bar.append("  │  ", style="dim")
        # Models
        bar.append(f"Models: {s.get('total_models', 0)}", style="bold")
        bar.append("  │  ", style="dim")
        # Workflows
        bar.append(f"Workflows: {s.get('total_workflows', 0)}", style="bold")

        return bar

    # ==================================================================
    # Rendering — Display compositor
    # ==================================================================
    def _build_display(self) -> Layout:
        layout = Layout()

        logo = self._build_logo()
        tab_bar = self._build_tab_bar()
        footer = self._build_footer()
        stats_bar = self._build_stats_bar()

        if self._tab == TAB_JOBS:
            content = self._build_jobs_tab()
        elif self._tab == TAB_ENDPOINTS:
            content = self._build_endpoints_tab()
        elif self._tab == TAB_STATUS:
            content = self._build_status_tab()
        elif self._tab == TAB_MODELS:
            content = self._build_models_tab()
        elif self._tab == TAB_WORKFLOWS:
            content = self._build_workflows_tab()
        elif self._tab == TAB_CONNECTION:
            content = self._build_connection_tab()
        elif self._tab == TAB_LOGS:
            content = self._build_logs_tab()
        else:
            content = Layout(Text("Unknown tab", style="red"))

        if self._show_actions:
            actions_panel = self._build_actions_overlay()
            layout.split_column(
                Layout(logo, size=LOGO_HEIGHT),
                Layout(tab_bar, size=1),
                Layout(stats_bar, size=1),
                Layout(actions_panel, ratio=1),
                Layout(footer, size=1),
            )
        elif self._form_active:
            form_panel = self._build_form_overlay()
            layout.split_column(
                Layout(logo, size=LOGO_HEIGHT),
                Layout(tab_bar, size=1),
                Layout(stats_bar, size=1),
                Layout(form_panel, ratio=1),
                Layout(footer, size=1),
            )
        elif self._show_help:
            help_panel = self._build_help_panel()
            layout.split_column(
                Layout(logo, size=LOGO_HEIGHT),
                Layout(tab_bar, size=1),
                Layout(stats_bar, size=1),
                Layout(help_panel, ratio=1),
                Layout(footer, size=1),
            )
        else:
            layout.split_column(
                Layout(logo, size=LOGO_HEIGHT),
                Layout(tab_bar, size=1),
                Layout(stats_bar, size=1),
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

        # Always fetch connection status first so the tab bar shows
        # "connected" / "disconnected" instead of "unknown".
        self._fetch_connection()
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
