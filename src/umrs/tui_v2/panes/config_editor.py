"""Config editor tab v2 — view and edit user config."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Label,
    Static,
)

from ..forms.screens import FormScreen, FormSchema, FormField


class ConfigEditorPane(Static):
    """Config editor tab v2 — view current config and run maintenance actions."""

    DEFAULT_CSS = """
    ConfigEditorPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    ConfigEditorPane > .summary { height: 1; color: $text; }
    ConfigEditorPane > .content {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        overflow-y: auto;
    }
    ConfigEditorPane > .actions-bar {
        height: auto;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
        margin-top: 1;
        layout: horizontal;
    }
    ConfigEditorPane > .actions-bar > Button {
        margin-right: 1;
        width: 14;
    }
    """

    BINDINGS = [
        ("r", "reload_config", "Reload"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._summary_widget = Static("", classes="summary")
        self._content = Static("", classes="content")
        self._reload_btn = Button("Reload Config", id="btn-reload", variant="primary")
        self._actions_bar = Static(
            "  Actions: r=Reload  |  Reload server config without restart",
            classes="actions-bar",
        )

    def compose(self) -> ComposeResult:
        yield self._summary_widget
        yield self._content
        yield self._actions_bar

    def on_mount(self) -> None:
        self._reload_btn.on_click(self._on_reload_click)
        self._render_config()

    def update_data(self, data: Any) -> None:
        self._summary_widget.update("  User config & server maintenance")
        self._render_config()

    def _render_config(self) -> None:
        lines = [
            "[bold]User Config[/bold]",
            "  No user config loaded (requires -c/--conf flag)",
            "",
            "[bold]Server Config[/bold]",
            "  Read-only — managed on server side",
            "",
            "[bold]Maintenance[/bold]",
            "  • Reload — re-reads config without restart",
            "  • Recompute — recompute embeddings",
            "  • Update — update missing embeddings",
            "  • Check — check S3/DB consistency",
            "",
            "[bold]Experiences[/bold]",
            "  Add/remove via: umrs-client experience add/remove",
            "  Or edit the config JSON file directly",
            "",
            "[bold]Connection Profiles[/bold]",
            "  Switch with: umrs-client profile use <name>",
            "  List with: umrs-client profile list",
        ]
        self._content.update("\n".join(lines))

    def _on_reload_click(self) -> None:
        self.app.notify("Config reload requested")

    def action_reload_config(self) -> None:
        self._on_reload_click()
