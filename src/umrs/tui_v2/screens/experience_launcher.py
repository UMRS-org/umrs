"""Experience launcher — single-screen operational dialog.

Replaces the old multi-step ExperienceLauncher with:
- RunExperienceDialog: select experience name, bucket, overrides, run

All fields visible at once; Enter submits, Esc cancels.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Button, Input, Label, Static

from .op_dialog import FieldSpec, OpDialog


class RunExperienceDialog(OpDialog):
    """Single-screen experience launch dialog."""

    DEFAULT_CSS = """
    RunExperienceDialog {
        layout: vertical;
        align: center middle;
        width: 100%;
        height: 100%;
    }

    .RunExperience-Container {
        width: 50;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    .RunExperience-Title {
        text-align: center;
        width: 100%;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    .RunExperience-Field {
        width: 100%;
        height: auto;
        margin: 1 0;
        layout: vertical;
    }

    .RunExperience-Label {
        color: $foreground;
        text-style: bold;
        margin-bottom: 0;
    }

    .RunExperience-Input {
        width: 100%;
    }

    .RunExperience-Actions {
        width: 100%;
        margin-top: 1;
        layout: horizontal;
        justify: right;
    }

    .RunExperience-Actions Button {
        width: 10;
    }
    """

    FIELDS: list[FieldSpec] = [
        FieldSpec(
            name="name",
            label="Experience:",
            placeholder="experience-name",
            valid=True,
        ),
        FieldSpec(
            name="bucket",
            label="Bucket:",
            placeholder="s3-bucket-name",
            valid=True,
        ),
        FieldSpec(
            name="overrides",
            label="Overrides JSON:",
            placeholder='{"key":"value"} (optional)',
            valid=False,
        ),
    ]

    TITLE: str = "Run Experience"

    def compose(self) -> ComposeResult:
        """Build the dialog layout."""
        yield Static(self.TITLE, classes="RunExperience-Title")

        for spec in self.FIELDS:
            with Vertical(classes="RunExperience-Field"):
                yield Label(spec.label, classes="RunExperience-Label")
                inp = Input(
                    placeholder=spec.placeholder,
                    id=f"field_{spec.name}",
                    classes="RunExperience-Input",
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

        with Horizontal(classes="RunExperience-Actions"):
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
ExperienceLauncher = RunExperienceDialog
