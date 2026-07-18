"""UMRS Monitor TUI v2 — main application.

Adds action tabs (data-transfer, jobs-submit, config-editor), inline
actions on existing tables (run/stop/cancel/upload/download),
progress indicators, confirmation dialogs, and actions history.
"""

from __future__ import annotations

import json
import time
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Click
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from .data.fetchers import Fetchers
from .data.store import MonitorStore
from .forms.screens import (
    ActionConfirmScreen,
    FilterScreen,
    UploadFormScreen,
    SubmitFormScreen,
    RunFormScreen,
)
from .prefs import MonitorPrefs

# ── Tab → pane class mapping ──────────────────────────────────
from .panes import (
    ConnectionPane,
    EndpointsPane,
    ErrorsPane,
    JobsPane,
    LogsPane,
    ModelsPane,
    StatusPane,
    WorkflowLogsPane,
    DataTransferPane,
    JobsSubmitPane,
    ConfigEditorPane,
)

_PANE_TYPES: dict[str, type] = {
    "jobs": JobsPane,
    "endpoints": EndpointsPane,
    "status": StatusPane,
    "models": ModelsPane,
    "workflows": None,  # imported below
    "connection": ConnectionPane,
    "logs": LogsPane,
    "errors": ErrorsPane,
    "workflow_logs": WorkflowLogsPane,
    "data_transfer": DataTransferPane,
    "jobs_submit": JobsSubmitPane,
    "config_editor": ConfigEditorPane,
}

# ── ASCII art logo ────────────────────────────────────────────
UMRS_V2_ART = r"""
 ██╗   ██╗███╗   ███╗██████╗ ███████╗
 ██║   ██║████╗ ████║██╔══██╗██╔════╝
 ██║   ██║██╔████╔██║██████╔╝███████╗
 ██║   ██║██║╚██╔╝██║██╔══██╗╚════██║
 ╚██████╔╝██║ ╚═╝ ██║██║  ██║███████║
  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝
 v2 — Monitor & Actions
"""

_STATS_INIT = "● Unknown  │  Jobs: 0  │  Endpoints: 0  │  Models: 0  │  Workflows: 0"
_FOOTER_INIT = "1-12 tabs │ r refresh │ a action │ q quit │ ? help │ Ctrl+S save"


class MonitorAppV2(App):
    """Interactive UMRS monitor with action capabilities."""

    CSS_PATH = "monitor.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("ctrl+s", "save_prefs", "Save Prefs"),
        Binding("ctrl+p", "command_palette", "Palette"),
        Binding("1", "switch_tab('jobs')", "Jobs"),
        Binding("2", "switch_tab('endpoints')", "Endpoints"),
        Binding("3", "switch_tab('status')", "Status"),
        Binding("4", "switch_tab('models')", "Models"),
        Binding("5", "switch_tab('workflows')", "Workflows"),
        Binding("6", "switch_tab('connection')", "Connection"),
        Binding("7", "switch_tab('logs')", "Logs"),
        Binding("8", "switch_tab('errors')", "Errors"),
        Binding("9", "switch_tab('workflow_logs')", "WF Logs"),
        Binding("0", "switch_tab('data_transfer')", "Data"),
        Binding("a", "switch_tab('jobs_submit')", "Submit"),
        Binding("b", "switch_tab('config_editor')", "Config"),
        Binding("slash", "focus_filter", "Filter", show=False),
        Binding("question", "toggle_help", "Help", show=False),
    ]

    def __init__(
        self,
        control_mq: Any,
        client_id: str,
        *,
        interval: float = 3.0,
        status_filter: str | None = None,
        project_filter: str | None = None,
        initial_tab: str = "jobs",
        job_type_filter: str | None = None,
        config_path: str | None = None,
        user_cfg: dict[str, Any] | None = None,
        connection_info: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
        workflow_id: str | None = None,
    ) -> None:
        super().__init__()
        self._mq = control_mq
        self._client_id = client_id
        self._interval = interval
        self._status_filter = status_filter
        self._project_filter = project_filter
        self._job_type_filter = job_type_filter
        self._config_path = config_path
        self._user_cfg = user_cfg or {}
        self._connection_info = connection_info or {}
        self._capabilities = capabilities or {}
        self._initial_tab = initial_tab
        self._workflow_id = workflow_id

        # Data layer
        self._store = MonitorStore()
        from umrs.client.control_api import UMRSControlAPI
        self._api = UMRSControlAPI(control_mq, client_id, config_path)
        self._fetchers = Fetchers(
            store=self._store,
            control_api=self._api,
            control_mq=control_mq,
            client_id=client_id,
        )

        # Prefs
        self._prefs = MonitorPrefs.load()

        # Cached counts
        self._counts: dict[str, int] = {}
        self._connected: bool | None = None

        # Actions history
        self._actions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # UI composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        # Top bar
        with Horizontal(id="topbar"):
            yield Static(UMRS_V2_ART, id="logo", markup=False)
            yield Static(_STATS_INIT, id="stats")
        # Connection status bar
        yield Static("● Checking...", id="connection-status")
        # Tabbed content
        with TabbedContent(initial=self._initial_tab):
            with TabPane("Jobs", id="jobs"):
                yield JobsPane()
            with TabPane("Endpoints", id="endpoints"):
                yield EndpointsPane()
            with TabPane("Status", id="status"):
                yield StatusPane()
            with TabPane("Models", id="models"):
                yield ModelsPane()
            with TabPane("Workflows", id="workflows"):
                yield self._create_workflows_pane()
            with TabPane("Connection", id="connection"):
                yield ConnectionPane()
            with TabPane("Logs", id="logs"):
                yield LogsPane()
            with TabPane("Errors", id="errors"):
                yield ErrorsPane()
            with TabPane("WF Logs", id="workflow_logs"):
                yield WorkflowLogsPane()
            with TabPane("Data", id="data_transfer"):
                yield DataTransferPane()
            with TabPane("Submit", id="jobs_submit"):
                yield JobsSubmitPane()
            with TabPane("Config", id="config_editor"):
                yield ConfigEditorPane()
        # Footer
        yield Static(_FOOTER_INIT, id="footer")

    def _create_workflows_pane(self):
        """Create workflows pane with inline actions."""
        from .panes.workflows import WorkflowsPaneV2
        return WorkflowsPaneV2()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        # Wire connection info
        try:
            pane = self.query_one(ConnectionPane)
            pane.set_connection_info(self._connection_info)
        except NoMatches:
            pass
        # Restore filters from prefs
        self._status_filter = self._prefs.status_filter
        self._project_filter = self._prefs.project_filter
        self._job_type_filter = self._prefs.job_type_filter
        self._fetchers.status_filter = self._status_filter
        self._fetchers.project_filter = self._project_filter
        self._fetchers.job_type_filter = self._job_type_filter
        # Start refresh loop
        self._refresh_active_tab()
        self.set_interval(self._interval, self._refresh_active_tab)

    # ------------------------------------------------------------------
    # Refresh logic
    # ------------------------------------------------------------------
    def _refresh_active_tab(self) -> None:
        tab = self._active_tab_name()
        self._refresh_connection()
        if tab == "jobs":
            self._refresh_jobs()
        elif tab == "endpoints":
            self._refresh_endpoints()
        elif tab == "status":
            self._refresh_status()
        elif tab == "models":
            self._refresh_models()
        elif tab == "workflows":
            self._refresh_workflows()
        elif tab == "connection":
            self._refresh_connection()
        elif tab == "logs":
            pass
        elif tab == "errors":
            self._refresh_errors()
        elif tab == "workflow_logs":
            self._refresh_workflow_logs()

    @work
    async def _refresh_jobs(self) -> None:
        await self._fetchers.fetch_jobs()
        data = self._store.tab("jobs").data
        self._update_pane("jobs", data)
        self._update_counts(data, "jobs")

    @work
    async def _refresh_endpoints(self) -> None:
        await self._fetchers.fetch_endpoints()
        data = self._store.tab("endpoints").data
        self._update_pane("endpoints", data)
        self._update_counts(data, "endpoints")

    @work
    async def _refresh_status(self) -> None:
        await self._fetchers.fetch_status()
        data = self._store.tab("status").data
        self._update_pane("status", data)

    @work
    async def _refresh_models(self) -> None:
        await self._fetchers.fetch_models()
        data = self._store.tab("models").data
        self._update_pane("models", data)
        self._update_counts(data, "models")

    @work
    async def _refresh_workflows(self) -> None:
        await self._fetchers.fetch_workflows()
        data = self._store.tab("workflows").data
        self._update_pane("workflows", data)
        self._update_counts(data, "workflows")

    @work
    async def _refresh_errors(self) -> None:
        await self._fetchers.fetch_errors()
        data = self._store.tab("errors").data
        self._update_pane("errors", data)

    @work
    async def _refresh_workflow_logs(self) -> None:
        exec_id = self._workflow_id
        if not exec_id:
            try:
                pane = self.query_one(WorkflowLogsPane)
                pane.set_execution_id(None)
            except NoMatches:
                pass
            return
        await self._fetchers.fetch_workflow_logs(exec_id)
        data = self._store.tab("workflow_logs").data
        if data is not None:
            try:
                pane = self.query_one(WorkflowLogsPane)
                pane.set_execution_id(exec_id)
                pane.update_data(data)
            except NoMatches:
                pass

    @work
    async def _refresh_connection(self) -> None:
        await self._fetchers.fetch_connection()
        data = self._store.tab("connection").data
        if isinstance(data, dict):
            self._connected = bool(data.get("ping_ok"))
        self._update_pane("connection", data)
        self._render_stats()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_pane(self, tab: str, data: Any) -> None:
        if tab == "workflows":
            try:
                pane = self.query_one(".WorkflowsPaneV2")
                pane.update_data(data)
            except NoMatches:
                pass
            return
        pane_type = _PANE_TYPES.get(tab)
        if pane_type is None:
            return
        try:
            pane = self.query_one(pane_type)
            pane.update_data(data)
        except NoMatches:
            pass

    def _update_counts(self, data: Any, tab: str) -> None:
        if isinstance(data, dict):
            if tab == "jobs":
                self._counts["jobs"] = len(data.get("jobs", []))
            elif tab == "endpoints":
                self._counts["endpoints"] = len(data.get("endpoints", []))
            elif tab == "models":
                self._counts["models"] = len(data.get("models", []))
            elif tab == "workflows":
                self._counts["workflows"] = len(data.get("workflows", []))
        self._render_stats()

    def _render_stats(self) -> None:
        try:
            stats = self.query_one("#stats", Static)
            parts: list[str] = []
            if self._connected is True:
                parts.append("● Connected")
            elif self._connected is False:
                parts.append("● Disconnected")
            else:
                parts.append("● Unknown")
            jobs = self._counts.get("jobs", 0)
            eps = self._counts.get("endpoints", 0)
            models = self._counts.get("models", 0)
            wfs = self._counts.get("workflows", 0)
            parts.append(f"Jobs: {jobs}")
            parts.append(f"Endpoints: {eps}")
            parts.append(f"Models: {models}")
            parts.append(f"Workflows: {wfs}")
            stats.update("  │  ".join(parts))
        except NoMatches:
            pass

    def _render_connection_status(self) -> None:
        try:
            el = self.query_one("#connection-status", Static)
            if self._connected is True:
                el.update("● Connected to UMRS")
            elif self._connected is False:
                el.update("● Disconnected from UMRS")
            else:
                el.update("● Checking...")
        except NoMatches:
            pass

    def _active_tab_name(self) -> str:
        try:
            tc = self.query_one(TabbedContent)
            return tc.active
        except NoMatches:
            return "jobs"

    def _record_action(self, action_type: str, target: str, status: str, details: str = "") -> None:
        self._actions.append({
            "time": time.strftime("%H:%M:%S"),
            "action": action_type,
            "target": target,
            "status": status,
            "details": details,
        })
        # Keep last 50 actions
        if len(self._actions) > 50:
            self._actions.pop(0)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_refresh(self) -> None:
        """Force refresh the active tab."""
        self._refresh_active_tab()
        self.notify("Refreshed", severity="information")

    def action_refresh_all(self) -> None:
        """Force refresh all tabs."""
        self._refresh_jobs()
        self._refresh_endpoints()
        self._refresh_status()
        self._refresh_models()
        self._refresh_workflows()
        self._refresh_errors()
        self.notify("All tabs refreshed", severity="information")

    def action_save_prefs(self) -> None:
        self._prefs.active_tab = self._active_tab_name()
        self._prefs.status_filter = self._status_filter
        self._prefs.project_filter = self._project_filter
        self._prefs.interval = self._interval
        self._prefs.save()
        self.notify("Preferences saved")

    def action_switch_tab(self, tab: str) -> None:
        try:
            tc = self.query_one(TabbedContent)
            tc.active = tab
        except NoMatches:
            pass

    def action_focus_filter(self) -> None:
        tab = self._active_tab_name()
        current = ""
        if tab == "jobs":
            current = self._status_filter or ""
        self.push_screen(FilterScreen(tab, current), self._on_filter_result)

    def _on_filter_result(self, result: str | None) -> None:
        if result is None:
            return
        tab = self._active_tab_name()
        if tab == "jobs":
            self._status_filter = result or None
            self._fetchers.status_filter = self._status_filter
        self._refresh_active_tab()

    def action_toggle_help(self) -> None:
        help_text = (
            "[bold]UMRS Monitor v2 — Keybindings[/bold]\n\n"
            "[cyan]1[/cyan]   Jobs          [cyan]2[/cyan]  Endpoints\n"
            "[cyan]3[/cyan]   Status        [cyan]4[/cyan]  Models\n"
            "[cyan]5[/cyan]   Workflows     [cyan]6[/cyan]  Connection\n"
            "[cyan]7[/cyan]   Logs          [cyan]8[/cyan]  Errors\n"
            "[cyan]9[/cyan]   WF Logs       [cyan]0[/cyan]  Data Transfer\n"
            "[cyan]a[/cyan]   Submit Job    [cyan]b[/cyan]  Config Editor\n"
            "[cyan]r[/cyan]   Refresh       [cyan]q[/cyan]  Quit\n"
            "[cyan]Ctrl+E[/cyan] Export tab  [cyan]Ctrl+S[/cyan] Save prefs\n"
            "[cyan]slash[/cyan] Filter       [cyan]?[/cyan]  Help\n"
            "\n[bold]Tables[/bold]\n"
            "  [cyan]Enter[/cyan]  Show detail / run action\n"
            "  Click header to sort\n"
            "  Scroll with mouse wheel\n"
        )
        self.notify(help_text, title="Help", timeout=15)

    def action_export(self) -> None:
        """Export current tab data to JSON."""
        from ..tui.export import export_tab as export_fn
        tab = self._active_tab_name()
        data = self._store.tab(tab).data
        if data is None:
            self.notify("No data to export", severity="warning")
            return
        try:
            path = export_fn(tab, data, fmt="json")
            self.notify(f"Exported to {path}")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
