"""Download dialog — single-screen file/dir download operation.

Replaces the old multi-step DownloadWizard.  All fields visible at once;
Enter submits, Esc cancels.  Progress shown inline.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Button, Input, Label, Static

from .op_dialog import FieldSpec, OpDialog


class DownloadDialog(OpDialog):
    """Single-screen download dialog backed by OperationAPI.download."""

    DEFAULT_CSS = """
    DownloadDialog {
        layout: vertical;
        align: center middle;
        width: 100%;
        height: 100%;
    }

    .Download-Container {
        width: 50;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    .Download-Title {
        text-align: center;
        width: 100%;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    .Download-Field {
        width: 100%;
        height: auto;
        margin: 1 0;
        layout: vertical;
    }

    .Download-Label {
        color: $foreground;
        text-style: bold;
        margin-bottom: 0;
    }

    .Download-Input {
        width: 100%;
    }

    .Download-Actions {
        width: 100%;
        margin-top: 1;
        layout: horizontal;
        justify: right;
    }

    .Download-Actions Button {
        width: 10;
    }
    """

    FIELDS: list[FieldSpec] = [
        FieldSpec(
            name="source",
            label="Source:",
            placeholder="s3://bucket/key or execution-id",
            valid=True,
        ),
        FieldSpec(
            name="dest",
            label="Destination:",
            placeholder="/path/to/dest",
            valid=True,
        ),
        FieldSpec(
            name="overwrite",
            label="Overwrite (y/n):",
            placeholder="n",
            valid=False,
        ),
    ]

    TITLE: str = "Download from Bucket"

    def __init__(
        self,
        buckets: list[str] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._buckets = buckets or []

    def compose(self) -> ComposeResult:
        """Build the download dialog layout."""
        yield Static(self.TITLE, classes="Download-Title")

        for spec in self.FIELDS:
            with Vertical(classes="Download-Field"):
                yield Label(spec.label, classes="Download-Label")
                inp = Input(
                    placeholder=spec.placeholder,
                    id=f"field_{spec.name}",
                    classes="Download-Input",
                )
                self._inputs[spec.name] = inp
                yield inp

        yield Static("", classes="OpDialog-ProgressText", id="progress_text")
        yield Static("██████████", classes="OpDialog-Progress", id="progress_bar")
        yield Static("", classes="OpDialog-Error", id="error_msg")

        with Horizontal(classes="Download-Actions"):
            yield Button("Run", id="btn_run", variant="primary")
            yield Button("Cancel", id="btn_cancel", variant="default")

    def on_key(self, event: Key) -> None:
        """Enter = submit, Esc = cancel."""
        if event.key == "enter":
            self.on_run()
            event.stop()
        elif event.key == "escape":
            self.on_cancel()
            event.stop()


# Compatibility alias
DownloadWizard = DownloadDialog
