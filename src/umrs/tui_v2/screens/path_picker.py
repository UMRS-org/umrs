"""Path picker — browse local filesystem to select a directory or file."""

from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Static


class PathPickerScreen(ModalScreen[str | None]):
    """Browse filesystem to select a path."""

    DEFAULT_CSS = """
    PathPickerScreen {
        layout: grid;
        grid-size: 1;
        grid-gutter: 1 2;
        padding: 0 1;
    }

    PathPickerScreen > .title {
        dock: top;
        height: 1;
        background: $primary;
        color: $text-on-primary;
        padding: 0 2;
        text-align: center;
        text-style: bold;
    }

    PathPickerScreen > Vertical {
        height: 1fr;
        layout: vertical;
        margin: 0 1;
    }

    PathPickerScreen > .field {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }

    PathPickerScreen > .field Label {
        width: 5;
        text-align: right;
        padding-right: 1;
        color: $text;
    }

    PathPickerScreen > .field > #path {
        width: 45;
    }

    PathPickerScreen > DataTable {
        border: none;
    }

    PathPickerScreen > DataTable:focus {
        border: solid $primary;
    }

    PathPickerScreen > .actions {
        dock: bottom;
        height: 1;
        layout: horizontal;
        horizontal-align: center;
    }

    PathPickerScreen > .actions Button {
        width: 12;
        margin: 0 1;
    }
    """

    BINDINGS = [("escape", "dismiss_cancel", "Cancel")]

    def __init__(self, initial_path: str = "") -> None:
        super().__init__()
        self._current_path = Path(initial_path) if initial_path else Path.home()
        self._selected: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("  Path Picker", classes="title")

        with Vertical():
            with Horizontal(classes="field"):
                yield Label("Dir:")
                yield Input(
                    placeholder="~/path",
                    id="path",
                    value=str(self._current_path),
                )

            table = DataTable(id="files")
            table.add_columns("Name", "Type", "")
            yield table

            with Horizontal(classes="actions"):
                yield Button("Cancel", id="btn-cancel", variant="default")
                yield Button("Up", id="btn-up", variant="default")
                yield Button("Select", id="btn-select", variant="primary")

    def on_mount(self) -> None:
        self._load_directory()

    def _load_directory(self) -> None:
        """Load directory contents into the table."""
        try:
            path = Path(self._current_path)
            if not path.exists():
                self.notify(f"Path not found: {path}", severity="warning")
                return

            table = self.query_one("#files", DataTable)
            table.clear()

            entries: list[tuple[str, str, str]] = []

            if path.parent != path:
                entries.append(("..", "DIR", ""))

            for entry in sorted(path.iterdir()):
                if entry.is_dir():
                    entries.append((entry.name + "/", "DIR", ""))
                elif entry.is_file():
                    entries.append((entry.name, "FILE", ""))

            for name, etype, _ in entries:
                table.add_row(name, etype, "")

        except PermissionError:
            self.notify("Permission denied", severity="error")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    @on(Click, "#btn-up")
    def on_up_click(self) -> None:
        self._current_path = self._current_path.parent
        self._load_directory()

    @on(Click, "#btn-cancel")
    def on_cancel_click(self) -> None:
        self.dismiss(None)

    @on(Click, "#btn-select")
    def on_select_click(self) -> None:
        if self._selected is not None:
            self.dismiss(self._selected)
        else:
            self.dismiss(str(self._current_path))

    @on(DataTable.CellSelected, "#files")
    def on_cell_selected(self, event: DataTable.CellSelected) -> None:
        name = str(event.cell_value)
        if name == "..":
            self._current_path = self._current_path.parent
        else:
            candidate = self._current_path / name
            if candidate.is_dir():
                self._current_path = candidate
            else:
                self._selected = str(candidate)
        self._load_directory()
