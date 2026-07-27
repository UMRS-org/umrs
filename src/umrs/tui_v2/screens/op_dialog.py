"""Shared operational dialog base for the UMRS TUI.

Provides a single-screen modal with all fields visible, inline validation,
progress display, and an action row ([Run]/[Cancel]). Subclasses declare
FIELDS and implement submit(values).

Used as the base for upload/download/run workflow/run experience dialogs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.signal import Signal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


@dataclass
class FieldSpec:
    """A single input field in an operational dialog."""

    name: str
    label: str
    placeholder: str = ""
    valid: bool = True


class OpDialog(ModalScreen):
    """Base modal screen with fields, progress, and action row.

    Subclasses:
        - define FIELDS: ClassVar[list[FieldSpec]]
        - implement submit(values: dict) -> None
        - optionally override on_submit_validated(values) for pre-flight validation
    """

    DEFAULT_CSS = """
    OpDialog, .OpDialog-Container {
        layout: vertical;
        align: center middle;
        width: 100%;
        height: 100%;
    }

    .OpDialog-Container {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        align: center middle;
    }

    .OpDialog-Title {
        text-align: center;
        width: 100%;
        content-align: center middle;
        color: $primary;
        text-style: bold;
    }

    .OpDialog-Field {
        width: 100%;
        height: auto;
        margin: 1 0;
    }

    .OpDialog-Label {
        width: 100%;
        text-align: left;
        color: $foreground;
        text-style: bold;
    }

    .OpDialog-Input {
        width: 100%;
    }

    .OpDialog-Progress {
        width: 100%;
        height: 3;
        margin: 1 0;
        display: none;
    }

    .OpDialog-Progress.visible {
        display: block;
    }

    .OpDialog-ProgressText {
        width: 100%;
        text-align: center;
        color: $primary;
    }

    .OpDialog-ProgressBar {
        width: 100%;
        height: 1;
        background: $surface;
        border: none;
    }

    .OpDialog-ProgressFill {
        width: 100%;
        height: 100%;
        background: $primary;
    }

    .OpDialog-Error {
        width: 100%;
        height: 1;
        color: $error;
        text-align: center;
        display: none;
    }

    .OpDialog-Error.visible {
        display: block;
    }

    .OpDialog-Actions {
        width: 100%;
        height: auto;
        margin-top: 1;
        layout: horizontal;
        justify: right;
    }

    .OpDialog-Actions Button {
        width: 10;
    }
    """

    # Signal emitted when submit values are valid and user pressed Run/Enter
    # Bound at runtime (Textual requires Signal(owner, name) init)
    action_submitted: Signal

    # Subclasses override
    FIELDS: ClassVar[list[FieldSpec]] = []
    TITLE: ClassVar[str] = "Operational Dialog"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inputs: dict[str, Input] = {}
        self._errors_visible = False
        self.action_submitted = Signal(self, "action_submitted", dict)

    def compose(self) -> ComposeResult:
        """Build the dialog layout."""
        yield Static(self.TITLE, classes="OpDialog-Title")

        for spec in self.FIELDS:
            with Vertical(classes="OpDialog-Field"):
                yield Static(spec.label, classes="OpDialog-Label")
                inp = Input(
                    placeholder=spec.placeholder,
                    id=f"field_{spec.name}",
                    classes="OpDialog-Input",
                )
                self._inputs[spec.name] = inp
                yield inp

        yield Static("", classes="OpDialog-ProgressText", id="progress_text")
        yield Static("░░░░░░░░░░", classes="OpDialog-Progress", id="progress_bar")
        yield Static("", classes="OpDialog-Error", id="error_msg")

        with Horizontal(classes="OpDialog-Actions"):
            yield Button("Run", id="btn_run", variant="primary")
            yield Button("Cancel", id="btn_cancel", variant="default")

    def on_mount(self) -> None:
        """Focus the first input."""
        first = next(iter(self._inputs.values()), None)
        if first:
            first.focus()

    # -- public helpers --------------------------------------------------------

    def set_progress(self, pct: float, msg: str = "") -> None:
        """Show progress bar with percentage."""
        pb = self.query_one("#progress_bar", Static)
        pt = self.query_one("#progress_text", Static)
        pb.display = True
        pb.classes = "OpDialog-Progress visible"
        pb.renderable = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        pt.renderable = f"{pct:.0f}% — {msg}" if msg else f"{pct:.0f}%"

    def set_error(self, msg: str) -> None:
        """Show an inline error message."""
        err = self.query_one("#error_msg", Static)
        err.renderable = msg
        err.classes = "OpDialog-Error visible"
        self._errors_visible = True

    def clear_error(self) -> None:
        """Hide inline error."""
        err = self.query_one("#error_msg", Static)
        err.renderable = ""
        err.classes = "OpDialog-Error"
        self._errors_visible = False

    def disable_run(self) -> None:
        """Disable the Run button (e.g. during background work)."""
        btn = self.query_one("#btn_run", Button)
        btn.disabled = True

    def enable_run(self) -> None:
        """Re-enable the Run button."""
        btn = self.query_one("#btn_run", Button)
        btn.disabled = False

    # -- value extraction ------------------------------------------------------

    def get_values(self) -> dict:
        """Extract current input values."""
        return {name: inp.value for name, inp in self._inputs.items()}

    # -- event handlers --------------------------------------------------------

    @on(Button.Pressed, "#btn_run")
    def on_run(self) -> None:
        """Handle Run button press — validate and emit submitted signal."""
        self.clear_error()
        values = self.get_values()
        # Check all required fields are non-empty
        for spec in self.FIELDS:
            if spec.valid and not values.get(spec.name, "").strip():
                self.set_error(f"Required field: {spec.label}")
                self.query_one(f"#field_{spec.name}", Input).focus()
                return
        # Let subclass do additional validation
        ok = self.on_submit_validated(values)
        if ok:
            self.action_submitted.emit(values)

    def on_submit_validated(self, values: dict) -> bool:
        """Override for additional pre-flight validation.

        Returns True if values are valid (proceed to emit), False otherwise.
        Use set_error() to communicate why validation failed.
        """
        return True

    @on(Button.Pressed, "#btn_cancel")
    def on_cancel(self) -> None:
        """Handle Cancel — dismiss the dialog."""
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        """Enter = submit, Esc = cancel."""
        if event.key == "enter":
            self.on_run()
            event.stop()
        elif event.key == "escape":
            self.on_cancel()
            event.stop()
