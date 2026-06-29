"""Textual-based UMRS monitor TUI.

Replaces the legacy monolithic Rich ``Live`` dashboard with a decomposed
Textual widget app.  Mouse support, scroll, sortable ``DataTable``, command
palette, CSS theming, and modal screens come from Textual itself.
"""

from __future__ import annotations

from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static, TabbedContent, TabPane

from .data.fetchers import Fetchers
from .data.store import MonitorStore
from .export import export_tab
from .panes import (
    ConnectionPane,
    EndpointsPane,
    JobsPane,
    LogsPane,
    ModelsPane,
    StatusPane,
    WorkflowsPane,
)
from .prefs import MonitorPrefs

# Tab → pane class mapping (used for type-safe pane queries).
_PANE_TYPES: dict[str, type] = {
    "jobs": JobsPane,
    "endpoints": EndpointsPane,
    "status": StatusPane,
    "models": ModelsPane,
    "workflows": WorkflowsPane,
    "connection": ConnectionPane,
    "logs": LogsPane,
}

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

# Combined logo: UMRS art and warp pipe placed horizontally side-by-side.
UMRS_LOGO = "\n".join(
    f"{u}  {p}"
    for u, p in zip(
        UMRS_ART.strip("\n").split("\n"),
        PIPE_ART.strip("\n").split("\n"),
    )
)

_STATS_INIT = "● Unknown  │  Jobs: 0  │  Endpoints: 0  │  Models: 0  │  Workflows: 0"
_FOOTER_INIT = "1-7 tabs │ r refresh │ q quit │ ? help │ Ctrl+E export │ Ctrl+P palette"


class MonitorApp(App):
    """Main Textual app for the UMRS monitor."""

    CSS_PATH = "monitor.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("ctrl+e", "export", "Export"),
        Binding("ctrl+s", "save_prefs", "Save Prefs"),
        Binding("ctrl+p", "command_palette", "Palette"),
        Binding("1", "switch_tab('jobs')", "Jobs"),
        Binding("2", "switch_tab('endpoints')", "Endpoints"),
        Binding("3", "switch_tab('status')", "Status"),
        Binding("4", "switch_tab('models')", "Models"),
        Binding("5", "switch_tab('workflows')", "Workflows"),
        Binding("6", "switch_tab('connection')", "Connection"),
        Binding("7", "switch_tab('logs')", "Logs"),
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
        # Cached counts for the stats bar.
        self._counts: dict[str, int] = {}
        self._connected: bool | None = None

    # ------------------------------------------------------------------
    # UI composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        # Logo: UMRS ASCII art + compact warp pipe
        yield Static(UMRS_LOGO, id="logo", markup=False)
        yield Static(_STATS_INIT, id="stats")
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
        yield Static(_FOOTER_INIT, id="footer")

    def on_mount(self) -> None:
        # Wire up connection info to the connection pane.
        try:
            pane = self.query_one(ConnectionPane)
            pane.set_connection_info(self._connection_info)
        except Exception:
            pass
        # Restore filters from saved prefs.
        self._status_filter = self._prefs.status_filter
        self._project_filter = self._prefs.project_filter
        self._job_type_filter = self._prefs.job_type_filter
        self._fetchers.status_filter = self._status_filter
        self._fetchers.project_filter = self._project_filter
        self._fetchers.job_type_filter = self._job_type_filter
        # Start the refresh loop.
        self._refresh_active_tab()
        self.set_interval(self._interval, self._refresh_active_tab)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Refresh the newly-activated tab immediately on switch."""
        self._refresh_active_tab()

    # ------------------------------------------------------------------
    # Refresh logic
    # ------------------------------------------------------------------
    def _refresh_active_tab(self) -> None:
        tab = self._active_tab_name()
        # Always poll connection status so the stats bar stays current.
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
            pass  # logs are client-side

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
    async def _refresh_connection(self) -> None:
        await self._fetchers.fetch_connection()
        data = self._store.tab("connection").data
        if isinstance(data, dict):
            self._connected = bool(data.get("ping_ok"))
        self._update_pane("connection", data)
        self._render_stats()

    def _update_pane(self, tab: str, data: Any) -> None:
        """Update the pane for ``tab`` with ``data``.

        Queries for the pane by **type** (not by CSS ID) to avoid
        ambiguity with the ``TabPane`` wrapper which shares the same ID.
        """
        pane_type = _PANE_TYPES.get(tab)
        if pane_type is None:
            return
        try:
            pane = self.query_one(pane_type)
            pane.update_data(data)
        except Exception:
            pass

    def _update_counts(self, data: Any, tab: str) -> None:
        """Update the stats bar with counts from the fetched data."""
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
        """Render the stats bar with current counts and connection status."""
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
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tab helpers
    # ------------------------------------------------------------------
    def _active_tab_name(self) -> str:
        try:
            tc = self.query_one(TabbedContent)
            return tc.active
        except Exception:
            return "jobs"

    def action_switch_tab(self, tab: str) -> None:
        try:
            tc = self.query_one(TabbedContent)
            tc.active = tab
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_refresh(self) -> None:
        self._refresh_active_tab()

    def action_export(self) -> None:
        tab = self._active_tab_name()
        data = self._store.tab(tab).data
        if data is None:
            self.notify("No data to export", severity="warning")
            return
        path = export_tab(tab, data, fmt="json")
        self.notify(f"Exported to {path}")

    def action_save_prefs(self) -> None:
        self._prefs.active_tab = self._active_tab_name()
        self._prefs.status_filter = self._status_filter
        self._prefs.project_filter = self._project_filter
        self._prefs.interval = self._interval
        self._prefs.save()
        self.notify("Preferences saved")

    def action_toggle_help(self) -> None:
        help_text = (
            "[bold]UMRS Monitor — Keybindings[/bold]\n\n"
            "[cyan]1-7[/cyan]       Switch tab\n"
            "[cyan]r[/cyan]          Force refresh\n"
            "[cyan]q / Ctrl+C[/cyan] Quit\n"
            "[cyan]Ctrl+E[/cyan]    Export current tab\n"
            "[cyan]Ctrl+S[/cyan]    Save preferences\n"
            "[cyan]Ctrl+P[/cyan]    Command palette\n"
            "[cyan]?[/cyan]          Toggle this help\n"
            "\n[bold]DataTable[/bold]\n"
            "  Click or arrow keys to highlight a row\n"
            "  [cyan]Enter[/cyan]     Show detail for highlighted row\n"
            "  Click column header to sort\n"
            "  Scroll with mouse wheel\n"
            "\n[dim]Preferences are saved to ~/.umrs/monitor.json[/dim]"
        )
        self.notify(help_text, title="Help", timeout=10)

    def action_focus_filter(self) -> None:
        """Open a filter input dialog for the current tab."""
        from .forms.screens import FilterScreen

        tab = self._active_tab_name()
        current = ""
        if tab == "jobs":
            current = self._status_filter or ""
        screen = FilterScreen(tab, current)
        self.push_screen(screen, self._on_filter_result)

    def _on_filter_result(self, result: str | None) -> None:
        """Apply the filter string returned by FilterScreen."""
        if result is None:
            return  # user cancelled
        tab = self._active_tab_name()
        if tab == "jobs":
            self._status_filter = result or None
            self._fetchers.status_filter = self._status_filter
        self._refresh_active_tab()
