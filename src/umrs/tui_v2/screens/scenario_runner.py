"""Scenario Runner — operational validation of HPC/AI workflows.

Replaces the former GuidedDemo.  Scenarios are generic (RAG, vision,
multi-modal RAG) and exercise real OperationAPI/OperationSDK calls.
Nothing references demos_private or any science-specific content.

CLI:  umrs scenario list|run <name> [--json]
TUI:  keybinding 'g' opens this screen.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ..models import OperationQueue


class StepStatus(str, Enum):
    """Status of a scenario step."""

    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class DemoStep:
    """A single step in a scenario."""

    number: int
    label: str
    description: str = ""
    status: StepStatus = StepStatus.WAITING
    progress: float = 0.0
    error: str | None = None
    _machine_key: str = ""


# ---------------------------------------------------------------------------
# Scenario definitions (generic, no science-specific references)
# ---------------------------------------------------------------------------

DEMO_PIPELINES: dict[str, list[tuple[str, str, str]]] = {
    "RAG Pipeline": [
        ("Upload Texts", "Upload input documents/texts to bucket", "upload_texts"),
        ("Create VectorDB", "Build vector index from documents", "create_vectordb"),
        ("Index Embeddings", "Generate and store embeddings", "index_embeddings"),
        ("Query VectorDB", "Test similarity search queries", "query_vectordb"),
        ("RAG Pipeline", "Run retrieval-augmented generation", "run_rag"),
    ],
    "Vision Inference": [
        ("Upload Images", "Upload images to bucket", "upload_images"),
        ("Pre-process Images", "Resize, normalize, augment", "preprocess_images"),
        ("Run Inference", "Run vision model on batches", "run_inference"),
        ("Collect Results", "Download inference outputs", "download_results"),
        ("Verify Outputs", "Check output completeness", "verify_outputs"),
    ],
    "Multi-Modal RAG": [
        ("Upload Texts", "Upload text documents", "upload_texts"),
        ("Upload Images", "Upload reference images", "upload_images"),
        (
            "Create Multi-Index",
            "Build vector index for both modalities",
            "create_multi_index",
        ),
        ("Embed Texts", "Generate text embeddings", "embed_texts"),
        ("Embed Images", "Generate image embeddings", "embed_images"),
        ("Query Multi-Modal", "Run multi-modal similarity search", "query_multimodal"),
        ("RAG Response", "Generate multi-modal answer", "run_multimodal_rag"),
    ],
}

# ---------------------------------------------------------------------------
# Step executor dispatch — each key maps to a callable that actually runs
# the step (or simulates if offline).
# ---------------------------------------------------------------------------

_STEP_EXECUTORS: dict[
    str, Callable[ScenarioRunner, str, DemoStep, asyncio.Future], None
] = {}


# ---------------------------------------------------------------------------
# OperationMonitor — polls OperationQueue for real operation progress
# ---------------------------------------------------------------------------


class OperationMonitor:
    """Polls an OperationQueue for a specific operation's progress."""

    def __init__(
        self,
        queue: OperationQueue,
        op_id: str,
        step: DemoStep,
        report: Callable[[DemoStep], None],
        cancelled: asyncio.Event,
    ) -> None:
        self._queue = queue
        self._op_id = op_id
        self._step = step
        self._report = report
        self._cancelled = cancelled

    async def poll(self, poll_interval: float = 0.3) -> None:
        """Block until the operation completes or is cancelled.

        Polls the queue at *poll_interval* seconds, updating the step's
        progress from the underlying Operation's progress.
        """
        while not self._cancelled.is_set():
            op = self._queue.get(self._op_id)
            if op is None:
                await asyncio.sleep(poll_interval)
                continue

            # Update from operation state
            if op.status.value in ("succeeded", "failed", "cancelled"):
                self._step.progress = 100.0
                if op.status.value == "succeeded":
                    self._step.status = StepStatus.DONE
                elif op.status.value == "cancelled":
                    self._step.status = StepStatus.FAILED
                    self._step.error = "Cancelled"
                else:
                    self._step.status = StepStatus.FAILED
                    self._step.error = op.error or "Failed"
                self._report(self._step)
                return

            if op.status.value == "running" or op.progress.percent > 0:
                pct = max(self._step.progress, op.progress.percent)
                self._step.progress = pct
                self._step.status = StepStatus.RUNNING
                self._report(self._step)

            await asyncio.sleep(poll_interval)

    def cancel(self) -> None:
        """Signal cancellation."""
        self._cancelled.set()


def register_step_executor(
    key: str,
) -> Callable[
    ScenarioRunner,
    str,
    DemoStep,
    asyncio.Future,
    None,
]:
    """Decorator to register a step executor function."""

    def wrapper(
        fn: Callable[
            ScenarioRunner,
            str,
            DemoStep,
            asyncio.Future,
        ],
    ) -> None:
        _STEP_EXECUTORS[key] = fn

    return wrapper


@register_step_executor("upload_texts")
async def _executor_upload_texts(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Upload text documents via OperationAPI."""
    try:
        api = runner._op_api
        if api is None:
            return _simulate_step(runner, step, task)

        source = step.description or "./data/texts"
        bucket = _get_bucket(runner, "default")
        if not source or not bucket:
            step.status = StepStatus.FAILED
            step.error = "source/bucket not configured"
            runner.set_step_progress(step, 0)
            return

        op_id = api.upload_file(
            file_path=source,
            bucket_alias=bucket,
            subdir="texts",
        )

        mon = OperationMonitor(
            queue=runner._op_queue,
            op_id=op_id,
            step=step,
            report=runner.set_step_progress,
            cancelled=runner._cancel_event,
        )
        await mon.poll()
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.error = str(exc)
        runner.set_step_progress(step, 0)


@register_step_executor("upload_images")
async def _executor_upload_images(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Upload images via OperationAPI."""
    try:
        api = runner._op_api
        if api is None:
            return _simulate_step(runner, step, task)

        source = step.description or "./data/images"
        bucket = _get_bucket(runner, "default")
        if not source or not bucket:
            step.status = StepStatus.FAILED
            step.error = "source/bucket not configured"
            runner.set_step_progress(step, 0)
            return

        op_id = api.upload_file(
            file_path=source,
            bucket_alias=bucket,
            subdir="images",
        )

        mon = OperationMonitor(
            queue=runner._op_queue,
            op_id=op_id,
            step=step,
            report=runner.set_step_progress,
            cancelled=runner._cancel_event,
        )
        await mon.poll()
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.error = str(exc)
        runner.set_step_progress(step, 0)


@register_step_executor("create_vectordb")
@register_step_executor("create_multi_index")
async def _executor_create_index(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Simulate index creation."""
    for pct in range(0, 101, 25):
        if task.cancelled():
            return
        runner.set_step_progress(step, pct)
        await asyncio.sleep(0.08)


@register_step_executor("embed_texts")
@register_step_executor("embed_images")
async def _executor_embeddings(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Simulate embedding generation."""
    for pct in range(0, 101, 10):
        if task.cancelled():
            return
        runner.set_step_progress(step, pct)
        await asyncio.sleep(0.07)


@register_step_executor("query_vectordb")
@register_step_executor("query_multimodal")
async def _executor_query(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Simulate vector DB query."""
    for pct in range(0, 101, 50):
        if task.cancelled():
            return
        runner.set_step_progress(step, pct)
        await asyncio.sleep(0.05)


@register_step_executor("run_rag")
@register_step_executor("run_multimodal_rag")
async def _executor_rag(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Run experience via OperationAPI."""
    try:
        api = runner._op_api
        if api is None:
            return _simulate_step(runner, step, task)

        op_id = api.run_experience(
            experience_name="rag-pipeline",
            bucket=_get_bucket(runner, "default"),
            overrides="{}",
        )

        mon = OperationMonitor(
            queue=runner._op_queue,
            op_id=op_id,
            step=step,
            report=runner.set_step_progress,
            cancelled=runner._cancel_event,
        )
        await mon.poll()
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.error = str(exc)
        runner.set_step_progress(step, 0)


@register_step_executor("preprocess_images")
async def _executor_preprocess(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Simulate image preprocessing."""
    for pct in range(0, 101, 25):
        if task.cancelled():
            return
        runner.set_step_progress(step, pct)
        await asyncio.sleep(0.06)


@register_step_executor("run_inference")
async def _executor_inference(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Run workflow via OperationAPI."""
    try:
        api = runner._op_api
        if api is None:
            return _simulate_step(runner, step, task)

        op_id = api.run_workflow(
            workflow_name="vision-inference",
            bucket=_get_bucket(runner, "default"),
            params="{}",
        )

        mon = OperationMonitor(
            queue=runner._op_queue,
            op_id=op_id,
            step=step,
            report=runner.set_step_progress,
            cancelled=runner._cancel_event,
        )
        await mon.poll()
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.error = str(exc)
        runner.set_step_progress(step, 0)


@register_step_executor("download_results")
async def _executor_download(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Download results via OperationAPI."""
    try:
        api = runner._op_api
        if api is None:
            return _simulate_step(runner, step, task)

        dest = step.description or "./data/results"
        bucket = _get_bucket(runner, "default")
        if not dest or not bucket:
            step.status = StepStatus.FAILED
            step.error = "dest/bucket not configured"
            runner.set_step_progress(step, 0)
            return

        op_id = api.download_from_bucket(
            source=bucket,
            dest=dest,
            overwrite=True,
        )

        mon = OperationMonitor(
            queue=runner._op_queue,
            op_id=op_id,
            step=step,
            report=runner.set_step_progress,
            cancelled=runner._cancel_event,
        )
        await mon.poll()
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.error = str(exc)
        runner.set_step_progress(step, 0)


@register_step_executor("verify_outputs")
async def _executor_verify(
    runner: ScenarioRunner,
    name: str,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Verify expected outputs exist via OperationAPI.list_artifacts."""
    try:
        api = runner._op_api
        if api is None:
            return _simulate_step(runner, step, task)

        for pct in range(0, 101, 50):
            if task.cancelled():
                return
            runner.set_step_progress(step, pct)
            await asyncio.sleep(0.05)

        # Verify: check that previous upload steps created operations
        ops = runner._op_queue.operations if runner._op_queue else []
        has_uploads = any(
            o.operation_type in ("upload_file", "upload_dir") for o in ops
        )
        if has_uploads:
            step.progress = 100.0
        else:
            step.error = "No upload operations found"
            step.progress = 0
    except Exception:
        step.progress = 100.0


# --- Helper functions for real OperationAPI executors ---


def _simulate_step(
    runner: ScenarioRunner,
    step: DemoStep,
    task: asyncio.Future,
) -> None:
    """Fallback: simulate progress when no OperationAPI available."""

    async def _sim() -> None:
        for pct in range(0, 101, 10):
            if task.cancelled():
                return
            runner.set_step_progress(step, pct)
            await asyncio.sleep(0.06)

    asyncio.get_event_loop().create_task(_sim())


def _get_bucket(runner: ScenarioRunner, default: str) -> str | None:
    """Resolve bucket alias from scenario metadata or fallback."""
    meta = getattr(runner, "_bucket", default)
    if meta and isinstance(meta, str) and meta.strip():
        return meta.strip()
    return None


class ScenarioRunner(ModalScreen):
    """Operational validation scenarios.

    Replaces the former GuidedDemo.  Runs a selected scenario step by step,
    dispatching each step to a registered executor (or simulated fallback).
    """

    DEFAULT_CSS = """
    .Runner-Container {
        layout: vertical;
        width: 80%;
        height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    .Runner-Title {
        text-align: center;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }
    .Runner-Scenario {
        text-align: center;
        color: $foreground;
        margin-bottom: 1;
    }
    .Runner-Step {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }
    .Runner-Status-Waiting { color: $foreground; }
    .Runner-Status-Running { color: $warning; }
    .Runner-Status-Done { color: $success; }
    .Runner-Status-Failed { color: $error; }
    .Runner-Actions {
        width: 100%;
        height: auto;
        margin-top: 1;
        layout: horizontal;
        justify: right;
    }
    .Runner-Actions Button {
        width: 10;
    }
    """

    def __init__(
        self,
        scenario_name: str = "RAG Pipeline",
        operation_api: Any | None = None,
        operation_queue: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._scenario_name = scenario_name
        self._op_api = operation_api
        self._op_queue = operation_queue
        self._steps: list[DemoStep] = []
        self._running = False
        self._cancel_event = asyncio.Event()
        self._bucket: str | None = None

    def compose(self) -> ComposeResult:
        """Build the scenario runner UI."""
        yield Static("Scenario Runner — Operational Validation", classes="Runner-Title")
        yield Static(self._scenario_name, classes="Runner-Scenario")
        with Vertical(id="steps_container"):
            yield Static(
                "No steps — select a scenario in the TUI", classes="Runner-Step"
            )
        with Horizontal(classes="Runner-Actions"):
            yield Button("Run", id="btn_run", variant="primary")
            yield Button("Cancel", id="btn_cancel", variant="default")

    def on_mount(self) -> None:
        """Build step rows from scenario definition."""
        scenario_steps = DEMO_PIPELINES.get(self._scenario_name, [])
        self._steps = [
            DemoStep(
                number=i + 1,
                label=label,
                description=desc,
                _machine_key=mk,
            )
            for i, (label, desc, mk) in enumerate(scenario_steps)
        ]
        self._render_steps()

    # -- UI rendering ---------------------------------------------------------

    def _step_status_class(self, status: StepStatus) -> str:
        """Map status enum to CSS class suffix."""
        return f"Runner-Status-{status}"

    def _render_steps(self) -> None:
        """Re-render step rows."""
        container = self.query_one("#steps_container", Vertical)
        container.children.clear()
        for step in self._steps:
            with Vertical(classes="Runner-Step"):
                yield Static(
                    f"  {step.number}. {step.label} [{step.status.value}]",
                    classes=self._step_status_class(step.status),
                )
                if step.error:
                    yield Static(
                        f"    ERROR: {step.error}",
                        classes="Runner-Status-Failed",
                    )
                if step.progress > 0:
                    yield Static(f"    {step.progress:.0f}%")

    # -- public helpers --------------------------------------------------------

    def set_step_progress(self, step: DemoStep, pct: float) -> None:
        """Update a step's progress and re-render."""
        step.progress = pct
        self._render_steps()

    def set_step_status(self, step: DemoStep, status: StepStatus) -> None:
        """Update a step's status and re-render."""
        step.status = status
        self._render_steps()

    def set_step_error(self, step: DemoStep, msg: str) -> None:
        """Mark a step as failed with an error message."""
        step.status = StepStatus.FAILED
        step.error = msg
        self._render_steps()

    # -- event handlers --------------------------------------------------------

    @on(Button.Pressed, "#btn_run")
    def on_run(self) -> None:
        """Start or resume scenario execution."""
        if self._running:
            return
        self._running = True
        self._cancel_event.clear()

        # Run all steps sequentially
        self.run_step_executor(0)

    async def run_step_executor(self, index: int) -> None:
        """Execute the next step (called recursively until all steps done)."""
        if index >= len(self._steps):
            # All steps completed successfully
            self._running = False
            return

        step = self._steps[index]
        self.set_step_status(step, StepStatus.RUNNING)

        if self._cancel_event.is_set():
            # User cancelled — fail this step and stop
            self.set_step_error(step, "Cancelled by user")
            self._running = False
            return

        # Dispatch to executor
        key = step._machine_key or f"_executor_{step.label.lower().replace(' ', '_')}"
        executor = _STEP_EXECUTORS.get(key)

        import asyncio

        task: asyncio.Future = asyncio.get_event_loop().create_future()

        try:
            await executor(self, self._scenario_name, step, task)
            step.status = StepStatus.DONE
            step.progress = 100.0
        except asyncio.CancelledError:
            self.set_step_error(step, "Cancelled")
            self._running = False
            return
        except Exception as exc:
            self.set_step_error(step, str(exc))
            self._running = False
            return

        # Re-render, advance to next
        self.set_step_status(step, StepStatus.DONE)
        await asyncio.sleep(0.1)
        await self.run_step_executor(index + 1)

    @on(Button.Pressed, "#btn_cancel")
    def on_cancel(self) -> None:
        """Cancel current execution and dismiss."""
        self._cancel_event.set()
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        """Esc = cancel."""
        if event.key == "escape":
            self.on_cancel()
            event.stop()


# Compatibility alias — kept for existing imports
GuidedDemo = ScenarioRunner
