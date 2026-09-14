"""Workflow dialogs — single-screen operational dialogs.

Replaces the old multi-step WorkflowWizard with:
- RunWorkflowDialog: select workflow name, bucket, params, run
- RegisterWorkflowDialog: upload workflow from directory path

All fields visible at once; Enter submits, Esc cancels.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Button, Input, Label, Static

from .op_dialog import FieldSpec, OpDialog


class RunWorkflowDialog(OpDialog):
    """Single-screen workflow run dialog."""

    DEFAULT_CSS = """
    RunWorkflowDialog {
        layout: vertical;
        align: center middle;
        width: 100%;
        height: 100%;
    }

    .RunWorkflow-Container {
        width: 50;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    .RunWorkflow-Title {
        text-align: center;
        width: 100%;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    .RunWorkflow-Field {
        width: 100%;
        height: auto;
        margin: 1 0;
        layout: vertical;
    }

    .RunWorkflow-Label {
        color: $foreground;
        text-style: bold;
        margin-bottom: 0;
    }

    .RunWorkflow-Input {
        width: 100%;
    }

    .RunWorkflow-Actions {
        width: 100%;
        margin-top: 1;
        layout: horizontal;
        justify: right;
    }

    .RunWorkflow-Actions Button {
        width: 10;
    }
    """

    FIELDS: list[FieldSpec] = [
        FieldSpec(
            name="workflow",
            label="Workflow:",
            placeholder="workflow-name",
            valid=True,
        ),
        FieldSpec(
            name="bucket",
            label="Bucket:",
            placeholder="s3-bucket-name",
            valid=True,
        ),
        FieldSpec(
            name="params",
            label="Params JSON:",
            placeholder='{"key":"value"} (optional)',
            valid=False,
        ),
    ]

    TITLE: str = "Run Workflow"

    def compose(self) -> ComposeResult:
        """Build the dialog layout."""
        yield Static(self.TITLE, classes="RunWorkflow-Title")

        for spec in self.FIELDS:
            with Vertical(classes="RunWorkflow-Field"):
                yield Label(spec.label, classes="RunWorkflow-Label")
                inp = Input(
                    placeholder=spec.placeholder,
                    id=f"field_{spec.name}",
                    classes="RunWorkflow-Input",
                )
                self._inputs[spec.name] = inp
                yield inp

        yield Static("", classes="OpDialog-ProgressText", id="progress_text")
        yield Static(
            "██████████",
            classes="OpDialog-Progress",
            id="progress_bar",
        )
        yield Static("", classes="OpDialog-Error", id="error_msg")

        with Horizontal(classes="RunWorkflow-Actions"):
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


class RegisterWorkflowDialog(OpDialog):
    """Single-screen workflow registration dialog.

    Validates workflow.json + steps/ directory exist before enabling submit.
    """

    DEFAULT_CSS = """
    RegisterWorkflowDialog {
        layout: vertical;
        align: center middle;
        width: 100%;
        height: 100%;
    }

    .RegisterWorkflow-Container {
        width: 50;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    .RegisterWorkflow-Title {
        text-align: center;
        width: 100%;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    .RegisterWorkflow-Field {
        width: 100%;
        height: auto;
        margin: 1 0;
        layout: vertical;
    }

    .RegisterWorkflow-Label {
        color: $foreground;
        text-style: bold;
        margin-bottom: 0;
    }

    .RegisterWorkflow-Input {
        width: 100%;
    }

    .RegisterWorkflow-Validation {
        width: 100%;
        height: 1;
        color: $success;
        text-align: center;
    }

    .RegisterWorkflow-Actions {
        width: 100%;
        margin-top: 1;
        layout: horizontal;
        justify: right;
    }

    .RegisterWorkflow-Actions Button {
        width: 10;
    }
    """

    FIELDS: list[FieldSpec] = [
        FieldSpec(
            name="directory",
            label="Directory path:",
            placeholder="/path/to/workflow/dir",
            valid=True,
        ),
        FieldSpec(
            name="name",
            label="Workflow name:",
            placeholder="workflow-name",
            valid=True,
        ),
    ]

    TITLE: str = "Register Workflow"

    def _validate_path(self) -> None:
        """Check workflow.json + steps/ exist."""
        validation = self.query_one("#validation", Static)
        dir_val = self._inputs.get("directory")
        name_val = self._inputs.get("name")
        if dir_val and name_val:
            source = dir_val.value or ""
            if source:
                try:
                    from pathlib import Path

                    p = Path(source)
                    manifest = p / "workflow.json"
                    steps_dir = p / "steps"
                    if manifest.is_file() and steps_dir.is_dir():
                        validation.renderable = "OK: manifest + steps/"
                        validation.classes = "RegisterWorkflow-Validation"
                    elif not manifest.is_file():
                        validation.renderable = "Missing: workflow.json"
                        validation.classes = "RegisterWorkflow-Validation"
                    else:
                        validation.renderable = "Missing: steps/"
                        validation.classes = "RegisterWorkflow-Validation"
                except OSError:
                    validation.renderable = "Cannot read path"
                    validation.classes = "RegisterWorkflow-Validation"

    def on_input_updated(self, widget: Input) -> None:
        """Validate on input change."""
        if widget.id in ("field_directory", "field_name"):
            self._validate_path()

    def on_submit_validated(self, values: dict) -> bool:
        """Require validation before submit."""
        dir_val = values.get("directory", "")
        if dir_val:
            try:
                from pathlib import Path

                p = Path(dir_val)
                if not (p / "workflow.json").is_file():
                    self.set_error("Missing workflow.json in directory")
                    self.query_one("#field_directory", Input).focus()
                    return False
                if not (p / "steps").is_dir():
                    self.set_error("Missing steps/ directory")
                    self.query_one("#field_directory", Input).focus()
                    return False
            except OSError as exc:
                self.set_error(str(exc))
                self.query_one("#field_directory", Input).focus()
                return False
        return True

    def compose(self) -> ComposeResult:
        """Build the dialog layout."""
        yield Static(self.TITLE, classes="RegisterWorkflow-Title")

        for spec in self.FIELDS:
            with Vertical(classes="RegisterWorkflow-Field"):
                yield Label(spec.label, classes="RegisterWorkflow-Label")
                inp = Input(
                    placeholder=spec.placeholder,
                    id=f"field_{spec.name}",
                    classes="RegisterWorkflow-Input",
                )
                self._inputs[spec.name] = inp
                yield inp

        yield Static("", id="validation", classes="RegisterWorkflow-Validation")
        yield Static("", classes="OpDialog-ProgressText", id="progress_text")
        yield Static(
            "██████████",
            classes="OpDialog-Progress",
            id="progress_bar",
        )
        yield Static("", classes="OpDialog-Error", id="error_msg")

        with Horizontal(classes="RegisterWorkflow-Actions"):
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


# Compatibility aliases
WorkflowWizard = RunWorkflowDialog
