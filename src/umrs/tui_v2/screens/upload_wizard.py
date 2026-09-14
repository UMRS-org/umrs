"""Upload dialog — single-screen file/dir upload operation.

Replaces the old multi-step UploadWizard.  All fields visible at once;
Enter submits, Esc cancels.  Progress shown inline.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Button, Input, Label, Static

from .op_dialog import FieldSpec, OpDialog


class UploadDialog(OpDialog):
    """Single-screen upload dialog backed by OperationAPI.upload."""

    DEFAULT_CSS = """
    UploadDialog {
        layout: vertical;
        align: center middle;
        width: 100%;
        height: 100%;
    }

    .Upload-Container {
        width: 50;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    .Upload-Title {
        text-align: center;
        width: 100%;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    .Upload-Field {
        width: 100%;
        height: auto;
        margin: 1 0;
        layout: vertical;
    }

    .Upload-Label {
        color: $foreground;
        text-style: bold;
        margin-bottom: 0;
    }

    .Upload-Input {
        width: 100%;
    }

    .Upload-Preview {
        width: 100%;
        height: 1;
        color: $success;
        text-align: center;
    }

    .Upload-Actions {
        width: 100%;
        margin-top: 1;
        layout: horizontal;
        justify: right;
    }

    .Upload-Actions Button {
        width: 10;
    }
    """

    FIELDS: list[FieldSpec] = [
        FieldSpec(
            name="source",
            label="Source path:",
            placeholder="/path/to/file or /path/to/dir",
            valid=True,
        ),
        FieldSpec(
            name="bucket",
            label="Bucket:",
            placeholder="s3-bucket-name",
            valid=True,
        ),
        FieldSpec(
            name="max_files",
            label="Max files:",
            placeholder="1000 (optional)",
            valid=False,
        ),
    ]

    TITLE: str = "Upload to Bucket"

    def __init__(
        self,
        buckets: list[str] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._buckets = buckets or []

    # -- preview ---------------------------------------------------------------

    def on_input_updated(self, widget: Input) -> None:
        """Update preview when source path changes."""
        if widget.id == "field_source" or widget.id == "field_max_files":
            self._check_source()

    def _check_source(self) -> None:
        """Validate source path and show file count preview."""
        preview = self.query_one("#preview", Static)
        source_val = self._inputs.get("source")
        bucket_val = self._inputs.get("bucket")
        if source_val and bucket_val:
            source = source_val.value or ""
            if source:
                try:
                    from pathlib import Path

                    p = Path(source)
                    if p.is_dir():
                        count = len(list(p.iterdir()))
                        preview.renderable = f"{count} items"
                    elif p.is_file():
                        preview.renderable = f"1 file, {p.stat().st_size} bytes"
                    else:
                        preview.renderable = "Path does not exist"
                    preview.classes = "Upload-Preview"
                except OSError:
                    preview.renderable = "Cannot read path"
                    preview.classes = "Upload-Preview"

    def compose(self) -> ComposeResult:
        """Build the upload dialog layout."""
        yield Static(self.TITLE, classes="Upload-Title")

        for spec in self.FIELDS:
            with Vertical(classes="Upload-Field"):
                yield Label(spec.label, classes="Upload-Label")
                inp = Input(
                    placeholder=spec.placeholder,
                    id=f"field_{spec.name}",
                    classes="Upload-Input",
                )
                self._inputs[spec.name] = inp
                yield inp

        yield Static("", id="preview", classes="Upload-Preview")
        yield Static("", classes="OpDialog-ProgressText", id="progress_text")
        yield Static("██████████", classes="OpDialog-Progress", id="progress_bar")
        yield Static("", classes="OpDialog-Error", id="error_msg")

        with Horizontal(classes="Upload-Actions"):
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
UploadWizard = UploadDialog
