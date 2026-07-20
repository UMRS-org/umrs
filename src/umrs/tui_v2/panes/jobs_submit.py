"""Jobs submit tab v2 — form-based job submission."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Input,
    Label,
    Select,
    Static,
)

from ..forms.screens import SubmitFormScreen


class JobsSubmitPane(Static):
    """Jobs submit tab v2 — form for submitting new jobs."""

    DEFAULT_CSS = """
    JobsSubmitPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    JobsSubmitPane > .summary { height: 1; color: $text; }
    JobsSubmitPane > .form-section {
        height: auto;
        border: round $primary;
        padding: 0 1;
        margin-top: 1;
    }
    JobsSubmitPane > .actions-bar {
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
        margin-top: 1;
    }
    JobsSubmitPane .form-row {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    JobsSubmitPane .form-row Label {
        width: 18;
        text-align: right;
        padding-right: 1;
    }
    JobsSubmitPane .form-row Input {
        width: 40;
    }
    JobsSubmitPane .form-row Select {
        width: 20;
    }
    JobsSubmitPane .form-row Button {
        width: 12;
        margin-left: 2;
    }
    """

    BINDINGS = [
        ("enter", "submit_job", "Submit Job"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._summary_widget = Static("", classes="summary")
        self._actions_bar = Static("  Actions: Enter=Submit  |  Fill form below", classes="actions-bar")

        # Form fields
        self._job_type = Select(
            [("embedding", "embedding"), ("inference", "inference"),
             ("training", "training"), ("text_analysis", "text_analysis")],
            value="embedding",
            id="job-type",
        )
        self._model = Input(placeholder="model_name or model_id", id="model-name")
        self._project = Input(placeholder="project_id (optional)", id="project")
        self._partition = Input(placeholder="slurm partition (optional)", id="partition")
        self._reservation = Input(placeholder="slurm reservation (optional)", id="reservation")
        self._nodes = Input(placeholder="1", id="nodes", value="1")
        self._gpus = Input(placeholder="1", id="gpus", value="1")
        self._max_attempts = Input(placeholder="1", id="max-attempts", value="1")
        self._keep_alive = Static("Keep alive: OFF", classes="toggle", id="keep-alive")
        self._submit_btn = Button("Submit Job", id="btn-submit", variant="primary")

        self._keep_alive_enabled = False

    def compose(self) -> ComposeResult:
        yield self._summary_widget
        with Vertical(classes="form-section"):
            with Horizontal(classes="form-row"):
                yield Label("Job type:")
                yield self._job_type
            with Horizontal(classes="form-row"):
                yield Label("Model:")
                yield self._model
            with Horizontal(classes="form-row"):
                yield Label("Project:")
                yield self._project
            with Horizontal(classes="form-row"):
                yield Label("Partition:")
                yield self._partition
            with Horizontal(classes="form-row"):
                yield Label("Reservation:")
                yield self._reservation
            with Horizontal(classes="form-row"):
                yield Label("Nodes:")
                yield self._nodes
            with Horizontal(classes="form-row"):
                yield Label("GPUs:")
                yield self._gpus
            with Horizontal(classes="form-row"):
                yield Label("Max attempts:")
                yield self._max_attempts
            with Horizontal(classes="form-row"):
                yield Label("Keep alive:")
                yield self._keep_alive
            with Horizontal(classes="form-row"):
                yield Static()  # Spacer
                yield self._submit_btn
        yield self._actions_bar

    def on_mount(self) -> None:
        self._submit_btn.on_click(self._on_submit_click)
        self._summary_widget.update("  Submit a new durable job to UMRS")

    def update_data(self, data: Any) -> None:
        self._summary_widget.update("  Submit a new durable job to UMRS")

    def _on_submit_click(self) -> None:
        job_type = self._job_type.value
        model = self._model.value or "default"
        project = self._project.value or None
        partition = self._partition.value or None
        reservation = self._reservation.value or None
        max_attempts = int(self._max_attempts.value or "1")
        self._submit_job(job_type, model, project, partition, reservation, max_attempts)

    def _submit_job(
        self,
        job_type: str,
        model: str,
        project: str | None,
        partition: str | None,
        reservation: str | None,
        max_attempts: int,
    ) -> None:
        """Submit a job — delegated to the app's action executor."""
        self.app.notify(
            f"Job submit requested:\n  type={job_type}\n  model={model}\n  project={project or '-'}"
        )

    def action_submit_job(self) -> None:
        self._on_submit_click()

    def action_toggle_keep_alive(self) -> None:
        self._keep_alive_enabled = not self._keep_alive_enabled
        status = "ON" if self._keep_alive_enabled else "OFF"
        self._keep_alive.update(f"Keep alive: {status}")
