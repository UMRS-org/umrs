"""Schema-validated modal form screens for monitor v2 actions.

Provides form screens for:
- Action confirmation (run, stop, cancel)
- File upload/download
- Job submission
- Workflow execution
- Filter input
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    Select,
)


@dataclass
class FormField:
    """Declarative form field definition."""

    name: str
    label: str
    kind: str = "text"  # "text", "select", "bool"
    value: Any = ""
    choices: list[tuple[str, Any]] | None = None
    required: bool = False
    placeholder: str = ""


@dataclass
class FormSchema:
    """Schema for a modal form."""

    title: str
    fields: list[FormField]
    submit_label: str = "Submit"
    cancel_label: str = "Cancel"
    json_schema: dict[str, Any] | None = None


class FilterScreen(ModalScreen[str | None]):
    """A simple filter input modal screen."""

    DEFAULT_CSS = """
    FilterScreen {
        align: center middle;
    }
    FilterScreen > Vertical {
        width: 60;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    FilterScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    FilterScreen .hint {
        color: $text-disabled;
        margin-top: 1;
    }
    """

    def __init__(self, tab: str, current_filter: str = "") -> None:
        super().__init__()
        self._tab = tab
        self._current = current_filter

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Filter {self._tab}", classes="title")
            yield Input(
                value=self._current,
                id="filter-input",
                placeholder="Enter filter value (empty to clear)",
            )
            yield Label("Press Enter to apply, Esc to cancel", classes="hint")
            with Horizontal():
                yield Button("Apply", id="apply", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    @on(Button.Pressed, "#apply")
    def _on_apply(self) -> None:
        value = self.query_one("#filter-input", Input).value
        self.dismiss(value or None)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)


class ActionConfirmScreen(ModalScreen[dict[str, Any]]):
    """Confirmation dialog for destructive actions."""

    DEFAULT_CSS = """
    ActionConfirmScreen {
        align: center middle;
    }
    ActionConfirmScreen > Vertical {
        width: 50;
        height: auto;
        border: round $warning;
        padding: 1 2;
        background: $surface;
    }
    ActionConfirmScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    ActionConfirmScreen Label.message {
        color: $text;
        margin-bottom: 1;
    }
    ActionConfirmScreen .buttons {
        margin-top: 2;
        height: 3;
        layout: horizontal;
    }
    ActionConfirmScreen .buttons Button {
        margin-right: 2;
    }
    """

    def __init__(
        self,
        action: str,
        target: str,
        confirm_label: str = "Confirm",
    ) -> None:
        super().__init__()
        self._action = action
        self._target = target
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                f"Confirm {self._action}",
                classes="title",
            )
            yield Label(
                f"Are you sure you want to {self._action} on target {self._target}?",
                classes="message",
            )
            with Horizontal(classes="buttons"):
                yield Button(
                    self._confirm_label,
                    id="confirm",
                    variant="primary",
                )
                yield Button("Cancel", id="cancel", variant="default")

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self) -> None:
        self.dismiss(
            {"action": self._action, "target": self._target, "confirmed": True}
        )

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(
            {"action": self._action, "target": self._target, "confirmed": False}
        )


class UploadFormScreen(ModalScreen[dict[str, Any] | None]):
    """Form screen for upload/download actions."""

    DEFAULT_CSS = """
    UploadFormScreen {
        align: center middle;
    }
    UploadFormScreen > Vertical {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    UploadFormScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    UploadFormScreen .field-label {
        margin-top: 1;
        color: $text;
    }
    UploadFormScreen .buttons {
        margin-top: 2;
        height: 3;
        layout: horizontal;
    }
    UploadFormScreen .buttons Button {
        margin-right: 2;
    }
    """

    def __init__(self, kind: str = "upload") -> None:
        super().__init__()
        self._kind = kind  # "upload" or "download"

    def compose(self) -> ComposeResult:
        source_label = "File Path" if self._kind == "upload" else "Bucket Alias"
        dest_label = "Bucket Alias" if self._kind == "upload" else "Local Path"
        with Vertical():
            yield Label(f"{self._kind.title()} Form", classes="title")
            yield Label(source_label, classes="field-label")
            yield Input(
                placeholder=f"{source_label.lower()}...",
                id="source",
            )
            yield Label(dest_label, classes="field-label")
            yield Input(
                placeholder=f"{dest_label.lower()}...",
                id="dest",
            )
            with Horizontal(classes="buttons"):
                yield Button("Submit", id="submit", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    @on(Button.Pressed, "#submit")
    def _on_submit(self) -> None:
        source = self.query_one("#source", Input).value
        dest = self.query_one("#dest", Input).value
        if not source or not dest:
            self.notify("Fill both fields", severity="warning")
            return
        self.dismiss(
            {
                "kind": self._kind,
                "source": source,
                "dest": dest,
            }
        )

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)


class SubmitFormScreen(ModalScreen[dict[str, Any] | None]):
    """Form screen for job submission."""

    DEFAULT_CSS = """
    SubmitFormScreen {
        align: center middle;
    }
    SubmitFormScreen > Vertical {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    SubmitFormScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    SubmitFormScreen .field-label {
        margin-top: 1;
        color: $text;
    }
    SubmitFormScreen .buttons {
        margin-top: 2;
        height: 3;
        layout: horizontal;
    }
    SubmitFormScreen .buttons Button {
        margin-right: 2;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._job_types = [
            ("processing", "Text Processing"),
            ("embedding", "Embedding"),
            ("text-analysis", "Text Analysis"),
            ("vision", "Vision"),
        ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Submit Job", classes="title")
            yield Label("Job Type", classes="field-label")
            yield Select(
                self._job_types,
                value="processing",
                id="job-type",
            )
            yield Label("Model", classes="field-label")
            yield Input(
                placeholder="model_name_or_path",
                id="model",
            )
            yield Label("Bucket (optional)", classes="field-label")
            yield Input(
                placeholder="bucket_alias",
                id="bucket",
            )
            with Horizontal(classes="buttons"):
                yield Button("Submit", id="submit", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    @on(Button.Pressed, "#submit")
    def _on_submit(self) -> None:
        job_type = self.query_one("#job-type", Select).value
        model = self.query_one("#model", Input).value
        bucket = self.query_one("#bucket", Input).value
        if not job_type or not model:
            self.notify("Fill job type and model", severity="warning")
            return
        self.dismiss(
            {
                "job_type": job_type,
                "model": model,
                "bucket": bucket or None,
            }
        )

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)


class RunFormScreen(ModalScreen[dict[str, Any] | None]):
    """Form screen for workflow execution."""

    DEFAULT_CSS = """
    RunFormScreen {
        align: center middle;
    }
    RunFormScreen > Vertical {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    RunFormScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    RunFormScreen .field-label {
        margin-top: 1;
        color: $text;
    }
    RunFormScreen .buttons {
        margin-top: 2;
        height: 3;
        layout: horizontal;
    }
    RunFormScreen .buttons Button {
        margin-right: 2;
    }
    """

    def __init__(self, workflow_name: str = "") -> None:
        super().__init__()
        self._workflow_name = workflow_name

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Run Workflow", classes="title")
            yield Label("Workflow Name", classes="field-label")
            yield Input(
                value=self._workflow_name,
                id="workflow-name",
            )
            yield Label("Bucket (optional)", classes="field-label")
            yield Input(
                placeholder="bucket_alias",
                id="bucket",
            )
            with Horizontal(classes="buttons"):
                yield Button("Run", id="submit", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    @on(Button.Pressed, "#submit")
    def _on_submit(self) -> None:
        workflow_name = self.query_one("#workflow-name", Input).value
        bucket = self.query_one("#bucket", Input).value
        if not workflow_name:
            self.notify("Enter workflow name", severity="warning")
            return
        self.dismiss(
            {
                "workflow_name": workflow_name,
                "bucket": bucket or None,
            }
        )

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)


class ConfigFormScreen(ModalScreen[dict[str, Any] | None]):
    """Form screen for config operations (init, add experience, remove)."""

    DEFAULT_CSS = """
    ConfigFormScreen {
        align: center middle;
    }
    ConfigFormScreen > Vertical {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    ConfigFormScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    ConfigFormScreen .field-label {
        margin-top: 1;
        color: $text;
    }
    ConfigFormScreen .buttons {
        margin-top: 2;
        height: 3;
        layout: horizontal;
    }
    ConfigFormScreen .buttons Button {
        margin-right: 2;
    }
    """

    def __init__(
        self,
        operation: str = "init",
        xp_name: str = "",
    ) -> None:
        super().__init__()
        self._operation = operation  # "init", "add", "remove", "use"
        self._xp_name = xp_name

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Config {self._operation.title()}", classes="title")

            if self._operation == "init":
                yield Label("Username", classes="field-label")
                yield Input(placeholder="username", id="username")
                yield Label("Full Name", classes="field-label")
                yield Input(placeholder="John Doe", id="fullname")
            elif self._operation == "add":
                yield Label("XP Name", classes="field-label")
                yield Input(placeholder="my-experience", id="xp-name")
                yield Label("Experience JSON", classes="field-label")
                yield Input(placeholder="/path/to/experience.json", id="xp-file")
            elif self._operation == "remove":
                yield Label("XP Names to Remove", classes="field-label")
                yield Input(
                    value=self._xp_name,
                    placeholder="xp1,xp2 (comma-separated)",
                    id="xp-names",
                )
            elif self._operation == "use":
                yield Label("Profile Name", classes="field-label")
                yield Input(
                    placeholder="profile-name",
                    id="profile",
                )

            with Horizontal(classes="buttons"):
                yield Button("Submit", id="submit", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    @on(Button.Pressed, "#submit")
    def _on_submit(self) -> None:
        result: dict[str, Any] = {"operation": self._operation}

        if self._operation == "init":
            username = self.query_one("#username", Input).value
            fullname = self.query_one("#fullname", Input).value
            if not username:
                self.notify("Enter username", severity="warning")
                return
            result["username"] = username
            result["fullname"] = fullname
        elif self._operation == "add":
            xp_name = self.query_one("#xp-name", Input).value
            xp_file = self.query_one("#xp-file", Input).value
            if not xp_name or not xp_file:
                self.notify("Fill XP name and file", severity="warning")
                return
            result["xp_name"] = xp_name
            result["xp_file"] = xp_file
        elif self._operation == "remove":
            xp_names = self.query_one("#xp-names", Input).value
            if not xp_names:
                self.notify("Enter XP names", severity="warning")
                return
            result["xp_names"] = [n.strip() for n in xp_names.split(",")]
        elif self._operation == "use":
            profile = self.query_one("#profile", Input).value
            if not profile:
                self.notify("Enter profile name", severity="warning")
                return
            result["profile"] = profile

        self.dismiss(result)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)
