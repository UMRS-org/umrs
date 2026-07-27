"""UMRS Monitor TUI v2 — main application.

Adds action tabs (data-transfer, jobs-submit, config-editor), inline
actions on existing tables (run/stop/cancel/upload/download),
progress indicators, confirmation dialogs, and actions history.

Entry point: ``MonitorAppV2`` (use via DashboardV2.run())
"""

from __future__ import annotations

import time
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import (
    Static,
    TabbedContent,
    TabPane,
)

from .data.fetchers import Fetchers
from .data.store import MonitorStore
from .forms.screens import (
    FilterScreen,
)
from .models import OperationQueue
from .op_api import OperationAPI
from .panes import (
    ArtifactsPane,
    ConnectionPane,
    DataTransferPane,
    EndpointsPane,
    ErrorsPane,
    JobsPane,
    LogsPane,
    ModelsPane,
    StatusPane,
    WorkflowLogsPane,
    WorkflowsPane,
    ConfigEditorPane,
)
from .prefs import MonitorPrefs
from .screens import (
    ArtifactBrowser,
    DownloadWizard,
    ExperienceLauncher,
    ScenarioRunner,
    UploadWizard,
    WorkflowWizard,
)

_PANE_TYPES: dict[str, type] = {
    "jobs": JobsPane,
    "endpoints": EndpointsPane,
    "status": StatusPane,
    "models": ModelsPane,
    "workflows": WorkflowsPane,
    "connection": ConnectionPane,
    "logs": LogsPane,
    "errors": ErrorsPane,
    "workflow_logs": WorkflowLogsPane,
    "data_transfer": DataTransferPane,
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
            Binding("b", "switch_tab('config_editor')", "Config"),
        Binding("slash", "focus_filter", "Filter", show=False),
        Binding("u", "show_upload", "Upload", show=False),
        Binding("e", "show_experience", "Experience", show=False),
        Binding("w", "show_workflow", "Workflow", show=False),
        Binding("d", "show_download", "Download", show=False),
        Binding("x", "show_artifacts", "Artifacts", show=False),
        Binding("g", "start_demo", "Demo", show=False),
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

        # Background operations
        self._op_queue = OperationQueue()
        self._op_api = OperationAPI(
            control_api=self._api,
            control_mq=control_mq,
            client_id=client_id,
            operation_queue=self._op_queue,
            config_path=config_path,
        )
        self._op_queue.register_callback(self._on_operation_progress)

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
                yield WorkflowsPane()
            with TabPane("Connection", id="connection"):
                yield ConnectionPane()
            with TabPane("Logs", id="logs"):
                yield LogsPane()
            with TabPane("Errors", id="errors"):
                yield ErrorsPane()
            with TabPane("Artifacts", id="artifacts"):
                yield ArtifactsPane()
            with TabPane("WF Logs", id="workflow_logs"):
                yield WorkflowLogsPane()
            with TabPane("Data", id="data_transfer"):
                yield DataTransferPane()
            with TabPane("Config", id="config_editor"):
                yield ConfigEditorPane()
        # Progress container (hidden when no operations)
        yield Static("", id="progress-bar")

        # Footer
        yield Static(_FOOTER_INIT, id="footer")

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
        """Update the pane for ``tab`` with ``data``."""
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

    def _record_action(
        self, action_type: str, target: str, status: str, details: str = ""
    ) -> None:
        self._actions.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "action": action_type,
                "target": target,
                "status": status,
                "details": details,
            }
        )
        # Keep last 50 actions
        if len(self._actions) > 50:
            self._actions.pop(0)

    # ------------------------------------------------------------------
    # Background operations
    # ------------------------------------------------------------------

    def _resolve_bucket_aliases(self) -> list[str]:
        """Resolve configured bucket aliases from config."""
        aliases: list[str] = []
        try:
            import os

            cfg_path = self._config_path or os.path.expanduser("~/.umrs/config.json")
            if os.path.exists(cfg_path):
                import json

                with open(cfg_path) as f:
                    cfg = json.load(f)
                # Buckets are under 'buckets' key, each entry has 'alias' or 'name'
                buckets = cfg.get("buckets", cfg.get("bucket_aliases", []))
                for b in buckets:
                    if isinstance(b, dict):
                        alias = b.get("alias", b.get("name", ""))
                        if alias:
                            aliases.append(alias)
                    elif isinstance(b, str):
                        aliases.append(b)
        except Exception:
            pass
        if not aliases:
            # Default buckets for HMAE demo
            aliases = ["demo-hmae", "demo-hmae-mae", "demo-hmae-features"]
        return sorted(set(aliases))

    def action_show_upload(self) -> None:
        """Show the upload wizard."""
        try:
            buckets = self._resolve_bucket_aliases()
            self.push_screen(UploadWizard(buckets=buckets))
        except Exception as e:
            self.notify(f"Cannot open upload: {e}", severity="error")

    def action_show_experience(self) -> None:
        """Show the experience launcher."""
        try:
            self.push_screen(ExperienceLauncher())
        except Exception as e:
            self.notify(f"Cannot open launcher: {e}", severity="error")

    def action_show_workflow(self) -> None:
        """Show the workflow wizard."""
        try:
            self.push_screen(WorkflowWizard())
        except Exception as e:
            self.notify(f"Cannot open workflow: {e}", severity="error")

    def action_show_download(self) -> None:
        """Show the download wizard."""
        try:
            buckets = self._resolve_bucket_aliases()
            self.push_screen(DownloadWizard(buckets=buckets))
        except Exception as e:
            self.notify(f"Cannot open download: {e}", severity="error")

    def action_show_artifacts(self) -> None:
        """Show the artifact browser."""
        try:
            self.push_screen(ArtifactBrowser())
        except Exception as e:
            self.notify(f"Cannot open artifacts: {e}", severity="error")

    def action_start_demo(self, scenario_name: str = "RAG Pipeline") -> None:
        """Start the scenario runner with operation API."""
        try:
            screen = ScenarioRunner(
                scenario_name=scenario_name,
                operation_api=self._op_api,
                operation_queue=self._op_queue,
            )
            self.push_screen(screen)
        except Exception as e:
            self.notify(f"Cannot start demo: {e}", severity="error")

    def _on_operation_progress(self, operation) -> None:
        """Callback from OperationQueue on progress updates."""
        self._render_progress_bar()
        if operation.completed:
            if operation.status.value == "succeeded":
                self.notify(f"✓ {operation.title}: Done", severity="information")
            elif operation.status.value == "failed":
                self.notify(f"✗ {operation.title}: {operation.error}", severity="error")
            elif operation.status.value == "cancelled":
                self.notify(f"Cancelled: {operation.title}", severity="warning")

    def _render_progress_bar(self) -> None:
        """Render progress bar for active operations."""
        try:
            bar_widget = self.query_one("#progress-bar", Static)
            ops = self._op_queue.active()
            if ops:
                op = ops[-1]  # Show latest active operation
                bar = op.progress.bar
                msg = op.progress.message
                pct = op.progress.percent
                bar_widget.update(
                    f"  [{op.status.value}] {op.title}  [{bar}] {pct:.0f}%  {msg}"
                )
                bar_widget.add_class("visible")
            else:
                bar_widget.update("")
                bar_widget.remove_class("visible")
        except NoMatches:
            pass

    def action_cancel_operation(self) -> None:
        """Cancel the currently active operation."""
        active = self._op_queue.active()
        if active:
            self._op_queue.cancel(active[0].operation_id)
            self._render_progress_bar()
            self.notify(f"Cancelled: {active[0].title}", severity="warning")
        else:
            self.notify("No active operation to cancel", severity="information")

    def action_op_history(self) -> None:
        """Show operation history (completed operations)."""
        completed = self._op_queue.completed()
        if not completed:
            self.notify("No completed operations yet")
            return
        lines = ["[bold]Recent Operations[/bold]"]
        for op in completed[-10:]:
            status = op.status.value.upper()
            dur = f"{op.duration_seconds:.1f}s"
            lines.append(f"  [{status}] {op.title}  ({dur})")
        self.notify("\n".join(lines), title="Operation History", timeout=10)

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
        # Apply filter based on active tab
        if tab == "jobs":
            self._status_filter = result or None
            self._fetchers.status_filter = self._status_filter
        elif tab == "endpoints":
            self._prefs.project_filter = result or None
            self._fetchers.project_filter = self._prefs.project_filter
        elif tab == "models":
            self._prefs.project_filter = result or None
            self._fetchers.project_filter = self._prefs.project_filter
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
