"""Artifacts tab pane v2 — artifact registry browser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import on
from textual.widgets import DataTable, Static


class ArtifactsPane(Static):
    """Artifacts tab pane v2."""

    DEFAULT_CSS = """
    ArtifactsPane {
        layout: vertical;
        padding: 0 1;
    }
    ArtifactsPane > .summary { height: 1; color: $text; }
    ArtifactsPane DataTable { height: 1fr; }
    ArtifactsPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    ArtifactsPane > .actions {
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("enter", "show_detail", "Detail"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._artifacts: list[dict[str, Any]] = []
        self._selected_artifact: dict[str, Any] | None = None
        self._summary_widget = Static("", classes="summary")
        self._table = DataTable(id="artifacts-table", cursor_type="row")
        self._detail = Static("", classes="detail")
        self._actions_bar = Static(
            "  Actions: Enter=Detail  r=Refresh  a=Artifacts", classes="actions"
        )

    def compose(self):
        yield self._summary_widget
        yield self._table
        yield self._detail
        yield self._actions_bar

    def on_mount(self) -> None:
        """Configure the table columns."""
        self._table.add_column("#", width=3)
        self._table.add_column("Name", width=15)
        self._table.add_column("Type", width=10)
        self._table.add_column("Status", width=10)
        self._table.add_column("Size", width=8)
        self._table.add_column("Created", width=10)

    def update_data(self, data: Any) -> None:
        """Update with artifact data from fetcher."""
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        artifacts = data.get("artifacts", []) if isinstance(data, dict) else []
        self._artifacts = artifacts
        self._render_table()
        self._render_summary()
        self._render_detail()

    def _render_table(self) -> None:
        """Render the artifacts table."""
        saved_cursor = self._table.cursor_row
        self._table.clear(columns=True)
        for idx, artifact in enumerate(self._artifacts):
            name = artifact.get("name", "?")
            artifact_type = artifact.get("artifact_type", "?")
            size = artifact.get("size_bytes", 0)
            size_str = self._format_size(size)
            created = artifact.get("created_at", "?")
            status = artifact.get("status", "active")
            self._table.add_row(
                str(idx + 1), name, artifact_type, status, size_str, str(created)
            )
        if 0 <= saved_cursor < len(self._artifacts):
            self._table.cursor_row = saved_cursor

    def _render_summary(self) -> None:
        """Render the summary line."""
        count = len(self._artifacts)
        total_size = sum(a.get("size_bytes", 0) for a in self._artifacts)
        self._summary_widget.update(
            f"  {count} artifact{'s' if count != 1 else ''}  ·  "
            f"Total: {self._format_size(total_size)}"
        )

    def _render_detail(self) -> None:
        """Render detail view for selected artifact."""
        if self._selected_artifact:
            a = self._selected_artifact
            uri = a.get("uri", "")
            size = self._format_size(a.get("size_bytes", 0))
            lines = [
                "[bold]Artifact[/bold]",
                f"  Name: {a.get('name', '?')}",
                f"  Type: {a.get('artifact_type', '?')}",
                f"  Status: {a.get('status', '?')}",
                f"  Size: {size}",
                f"  URI: {uri}",
                f"  Created: {a.get('created_at', '?')}",
            ]
            # Local results preview (if URI is a file path)
            if uri and not uri.startswith(("s3://", "http://", "https://")):
                local_path = Path(uri)
                if local_path.is_dir():
                    children = list(local_path.iterdir())[:20]
                    lines.append("")
                    lines.append("[bold]Local Contents[/bold]")
                    for child in sorted(children):
                        child_size = (
                            self._format_size(child.stat().st_size)
                            if child.is_file()
                            else "    "
                        )
                        marker = "/" if child.is_dir() else " "
                        lines.append(f"  [{marker}] {child.name:<30} {child_size}")
                    if len(children) > 20:
                        total = len(list(local_path.iterdir()))
                        lines.append(f"  ... and {total - 20} more")
            self._detail.update("\n".join(lines))
        else:
            self._detail.update("Select an artifact for details")

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f}MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        row_key = event.row_key
        if row_key is not None:
            idx = event._row_cache.index(row_key) if row_key in event._row_cache else 0
            if idx < len(self._artifacts):
                self._selected_artifact = self._artifacts[idx]
                self._render_detail()

    def action_show_detail(self) -> None:
        """Show detail for selected artifact."""
        if self._selected_artifact:
            self._render_detail()

    def action_refresh(self) -> None:
        """Refresh artifact list."""
        self.notify("Refreshing artifacts...", severity="information")
