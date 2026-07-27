"""Artifact browser — browse local results directories, preview, copy paths."""

from __future__ import annotations

import os
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Static


class ArtifactBrowser(ModalScreen[dict | None]):
    """Browse local results directories, preview, copy paths."""

    DEFAULT_CSS = """
    ArtifactBrowser {
        layout: grid;
        grid-size: 1;
        grid-gutter: 1 2;
        padding: 0 1;
    }

    ArtifactBrowser > .title {
        dock: top;
        height: 1;
        background: $primary;
        color: $text-on-primary;
        padding: 0 2;
        text-align: center;
        text-style: bold;
    }

    ArtifactBrowser > Vertical {
        height: 1fr;
        layout: vertical;
        margin: 0 1;
    }

    ArtifactBrowser > .field {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }

    ArtifactBrowser > .field Label {
        width: 6;
        text-align: right;
        padding-right: 1;
        color: $text;
    }

    ArtifactBrowser > .field > #path {
        width: 40;
    }

    ArtifactBrowser > .preview {
        height: 10;
        border: round $primary;
        padding: 1;
        margin-bottom: 1;
    }

    ArtifactBrowser > .actions {
        dock: bottom;
        height: 1;
        layout: horizontal;
        horizontal-align: center;
    }

    ArtifactBrowser > .actions Button {
        width: 14;
        margin: 0 1;
    }

    ArtifactBrowser DataTable {
        border: none;
    }

    ArtifactBrowser DataTable:focus {
        border: solid $primary;
    }
    """

    BINDINGS = [("escape", "dismiss_screen", "Close")]

    def __init__(self, initial_path: str = "") -> None:
        super().__init__()
        self._current_path = Path(initial_path) if initial_path else Path.home()
        self._selected: tuple[str, str, str, str] | None = None

    def compose(self) -> ComposeResult:
        yield Static("  Results Browser", classes="title")

        with Vertical():
            with Horizontal(classes="field"):
                yield Label("Path:")
                yield Input(
                    placeholder="~/results",
                    id="path",
                    value=str(self._current_path),
                )

            table = DataTable(id="files")
            table.add_columns("Name", "Size", "Type", "Actions")
            yield table

            yield Static("", classes="preview", id="preview")

            with Horizontal(classes="actions"):
                yield Button("Refresh", id="btn-refresh", variant="default")
                yield Button("Copy Path", id="btn-copy", variant="default")
                yield Button("Open Dir", id="btn-opendir", variant="default")
                yield Button("Back", id="btn-back", variant="default")

    def on_mount(self) -> None:
        self._load_directory()

    def _load_directory(self) -> None:
        """Load directory contents into the file table."""
        try:
            path = Path(self._current_path)
            if not path.exists():
                self.notify(f"Path not found: {path}", severity="warning")
                return

            table = self.query_one("#files", DataTable)
            table.clear()

            entries: list[tuple[str, str, str, str]] = []

            if path.parent != path:
                entries.append(("..", "DIR", "DIR", "Back"))

            for entry in sorted(path.iterdir()):
                if entry.is_file():
                    size = self._fmt_size(entry.stat().st_size)
                    ext = self._get_type(entry.suffix)
                    actions = self._get_actions(entry)
                    entries.append((entry.name, size, ext, actions))
                elif entry.is_dir():
                    entries.append((entry.name + "/", "DIR", "DIR", "→"))

            for name, size, ext, actions in entries:
                table.add_row(name, size, ext, actions)

            self.query_one("#preview", Static).update("")

        except PermissionError:
            self.notify("Permission denied", severity="error")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def _fmt_size(self, size_bytes: int) -> str:
        """Format file size in human-readable form."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        return f"{size_bytes / (1024 * 1024):.1f}MB"

    def _get_type(self, suffix: str) -> str:
        """Map file extension to type badge."""
        types: dict[str, str] = {
            ".png": "PNG", ".jpg": "JPG", ".jpeg": "JPG", ".gif": "GIF",
            ".json": "JSON", ".pkl": "PKL", ".pt": "PT",
            ".pth": "PT", ".npy": "NPY", ".txt": "TXT",
        }
        return types.get(suffix.lower(), suffix.upper().lstrip("."))

    def _get_actions(self, path: Path) -> str:
        """Get available actions for a file."""
        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".gif"):
            return "[Preview]"
        return "[Copy]"

    @on(DataTable.RowSelected, "#files")
    def _on_select(self, event: DataTable.RowSelected) -> None:
        """Handle file/directory selection."""
        row = event.cursor_row
        if row is None:
            return

        table = self.query_one("#files", DataTable)
        try:
            name = table[row][0]  # type: ignore[index]
            file_type = table[row][2]

            if name == "..":
                self._current_path = self._current_path.parent
                self._load_directory()
                return

            if file_type == "DIR":
                self._current_path = self._current_path / name.rstrip("/")
                self._load_directory()
                return

            self._selected = (name, table[row][1], file_type, table[row][3])  # type: ignore[index]
            self._show_preview(name, file_type)

        except (IndexError, KeyError, TypeError):
            pass

    def _show_preview(self, name: str, file_type: str) -> None:
        """Show preview for selected file."""
        preview = self.query_one("#preview", Static)

        if file_type in ("PNG", "JPG", "GIF"):
            preview.update(f"[bold]Preview:[/bold] {name}")
        else:
            preview.update(f"[italic]{name} — no preview available[/italic]")

    @on(Click, "#btn-refresh")
    def _on_refresh(self) -> None:
        self._load_directory()

    @on(Click, "#btn-copy")
    def _on_copy(self) -> None:
        if not self._selected:
            self.notify("No file selected", severity="warning")
            return
        name = self._selected[0]
        full_path = str(self._current_path / name)
        try:
            import pyperclip
            pyperclip.copy(full_path)
            self.notify(f"Copied: {full_path}")
        except ImportError:
            self.notify(f"Path: {full_path}", severity="information")

    def _open_current_dir(self) -> None:
        """Open current directory in file manager."""
        path = str(self._current_path)
        if os.name == "posix":
            os.system(f"open '{path}' &")
        else:
            os.system(f"xdg-open '{path}' &")

    @on(Click, "#btn-opendir")
    def _on_opendir(self) -> None:
        self._open_current_dir()

    @on(Click, "#btn-back")
    def _on_back(self) -> None:
        if self._current_path.parent != self._current_path:
            self._current_path = self._current_path.parent
            self._load_directory()
        else:
            self.dismiss(None)

    def on_mount(self) -> None:  # noqa: F811
        self.focus()
        self.query_one("#path", Input).focus()
