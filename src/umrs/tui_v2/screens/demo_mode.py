"""Guided demo mode — pre-configured AI pipeline walkthroughs.

Covers general HPC/AI workflows:
1. VectorDB / RAG pipeline from input texts/images
2. Vision model inference
3. Complete multi-modal RAG pipeline with VLM usage
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class StepStatus(str, Enum):
    """Status of a demo step."""

    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class DemoStep:
    """A single step in the demo pipeline."""

    number: int
    label: str
    description: str = ""
    status: StepStatus = StepStatus.WAITING
    progress: float = 0.0
    error: str | None = None


# Pre-configured general AI demo pipelines (no specific scientific projects)
DEMO_PIPELINES = {
    "RAG Pipeline": [
        ("Upload Texts", "Upload input documents/texts to bucket"),
        ("Create VectorDB", "Build vector index from documents"),
        ("Index Embeddings", "Generate and store embeddings"),
        ("Query VectorDB", "Test similarity search queries"),
        ("RAG Pipeline", "Run retrieval-augmented generation"),
        ("Download Results", "Retrieve generated outputs"),
        ("Verify Quality", "Check output quality and completeness"),
        ("Complete", "RAG pipeline finished ✓"),
    ],
    "Vision Inference": [
        ("Upload Images", "Upload test images to bucket"),
        ("Load Vision Model", "Provision vision model endpoint"),
        ("Preprocess", "Resize, normalize, augment images"),
        ("Run Inference", "Execute vision model predictions"),
        ("Parse Results", "Extract detections/classifications"),
        ("Visualize", "Generate prediction visualizations"),
        ("Download Results", "Retrieve outputs from bucket"),
        ("Complete", "Vision pipeline finished ✓"),
    ],
    "Multi-Modal RAG": [
        ("Upload Texts & Images", "Upload documents + images to bucket"),
        ("Create Text Embeddings", "Build vector index for text"),
        ("Create Image Embeddings", "Build vector index for images"),
        ("Cross-Modal Index", "Link text and image embeddings"),
        ("Load VLM", "Provision vision-language model"),
        ("Query Multi-Modal", "Search with text+image queries"),
        ("VLM Generation", "Generate responses from VLM"),
        ("Download Results", "Retrieve all outputs"),
        ("Complete", "Multi-modal RAG finished ✓"),
    ],
}


class GuidedDemo(ModalScreen[bool]):
    """Guided demo mode for AI/HPC pipeline walkthroughs."""

    DEFAULT_CSS = """
    GuidedDemo {
        layout: grid;
        grid-size: 1;
        grid-gutter: 1 2;
        padding: 0 1;
    }

    GuidedDemo > .title {
        dock: top;
        height: 1;
        background: $primary;
        color: $text-on-primary;
        padding: 0 2;
        text-align: center;
        text-style: bold;
    }

    GuidedDemo > Vertical {
        height: 1fr;
        layout: vertical;
        margin: 0 1;
    }

    GuidedDemo .step {
        height: 2;
        margin-bottom: 1;
    }

    GuidedDemo .step.running {
        background: $primary 20%;
    }

    GuidedDemo .step.failed {
        background: $error 20%;
    }

    GuidedDemo .step.done {
        background: $success 20%;
    }

    GuidedDemo > .progress-bar {
        dock: bottom;
        height: 2;
        background: $surface;
        padding: 0 1;
    }

    GuidedDemo > .controls {
        dock: bottom;
        height: 1;
        layout: horizontal;
        horizontal-align: center;
    }

    GuidedDemo > .controls Button {
        width: 12;
        margin: 0 1;
    }
    """

    BINDINGS = [("escape", "dismiss_screen", "Cancel")]

    def __init__(
        self,
        pipeline_name: str = "RAG Pipeline",
        initial_step: int = 0,
    ) -> None:
        super().__init__()
        self._pipeline_name = pipeline_name
        self._steps: list[DemoStep] = [
            DemoStep(i + 1, label, desc)
            for i, (label, desc) in enumerate(
                DEMO_PIPELINES.get(pipeline_name, DEMO_PIPELINES["RAG Pipeline"])
            )
        ]
        self._current_step = initial_step
        self._running = False
        self._cancelled = False
        self._start_time: float = 0.0

    def compose(self) -> ComposeResult:
        yield Static(f"  Demo: {self._pipeline_name}", classes="title")

        with Vertical():
            for step in self._steps:
                yield Static("", classes="step", id=f"step-{step.number}")

            yield Static("", classes="progress-bar", id="progress-bar")

            with Horizontal(classes="controls"):
                yield Button("Pause", id="btn-pause", variant="default")
                yield Button("Reset", id="btn-reset", variant="warning")
                yield Button("Next Step", id="btn-next", variant="default")

    def on_mount(self) -> None:
        self._render_steps()
        self._run_pipeline()

    def _render_steps(self) -> None:
        """Render all step status lines."""
        for step in self._steps:
            widget = self.query_one(f"#step-{step.number}", Static)
            status_icon = {
                StepStatus.WAITING: "—",
                StepStatus.RUNNING: "→",
                StepStatus.DONE: "✓",
                StepStatus.FAILED: "✗",
            }[step.status]

            status_text = {
                StepStatus.WAITING: "[dim]Waiting[/dim]",
                StepStatus.RUNNING: "[yellow]Running[/yellow]",
                StepStatus.DONE: "[green]Done[/green]",
                StepStatus.FAILED: "[red]Failed[/red]",
            }[step.status]

            progress_str = (
                f"  {step.progress:.0f}%" if step.status == StepStatus.RUNNING else ""
            )

            widget.update(
                f"  Step {step.number}/{len(self._steps)}  {step.label:<20} "
                f"{status_icon} {status_text[step.status]}{progress_str}"
            )
            widget.classes = f"step {step.status.value}"

        self._update_progress_bar()

    def _update_progress_bar(self) -> None:
        """Update overall progress bar."""
        total = len(self._steps)
        done = sum(1 for s in self._steps if s.status == StepStatus.DONE)
        pct = done / total if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = time.time() - self._start_time if self._start_time else 0
        eta = max(0, elapsed / max(pct, 0.01) * (1 - pct) - elapsed) if pct < 1 else 0

        progress = self.query_one("#progress-bar", Static)
        progress.update(
            f"  Progress: [{bar}] {pct:.0%}  Elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s"
        )

    async def _run_pipeline(self) -> None:
        """Run through all steps automatically."""
        self._running = True
        self._start_time = time.time()

        for i, step in enumerate(self._steps):
            if self._cancelled:
                break
            if step.status != StepStatus.WAITING:
                continue

            self._current_step = i
            step.status = StepStatus.RUNNING
            self._render_steps()

            # Execute step (simulated — wire to real op_api calls)
            success = await self._execute_step(step)

            if success:
                step.status = StepStatus.DONE
                step.progress = 100.0
            else:
                step.status = StepStatus.FAILED
                step.error = "Step failed"

            self._render_steps()

            # Auto-advance delay
            if success and not self._cancelled:
                await asyncio.sleep(2)

        self._running = False
        if not self._cancelled:
            self.dismiss(True)

    async def _execute_step(self, step: DemoStep) -> bool:
        """Execute a single step. Returns True on success."""
        # Simulate async work with progress updates
        for progress_pct in range(0, 101, 10):
            if self._cancelled:
                return False
            step.progress = progress_pct
            self._render_steps()
            await asyncio.sleep(0.3)
        return True

    @on(Click, "#btn-pause")
    def _on_pause(self) -> None:
        """Pause/resume pipeline."""
        self._running = not self._running
        pause_btn = self.query_one("#btn-pause", Button)
        pause_btn.label = "Resume" if not self._running else "Pause"
        if self._running:
            asyncio.create_task(self._run_pipeline())

    @on(Click, "#btn-reset")
    def _on_reset(self) -> None:
        """Reset all steps."""
        self._cancelled = True
        for step in self._steps:
            step.status = StepStatus.WAITING
            step.progress = 0.0
            step.error = None
        self._current_step = 0
        self._running = False
        self._start_time = 0.0
        self._render_steps()
        self.query_one("#btn-pause", Button).label = "Pause"
        self.notify("Demo reset", severity="information")

    @on(Click, "#btn-next")
    def _on_next(self) -> None:
        """Skip to next step."""
        if self._current_step < len(self._steps) - 1:
            self._current_step += 1
            self.notify(
                f"Jumped to step {self._current_step + 1}", severity="information"
            )

    def on_mount(self) -> None:  # noqa: F811
        self.focus()
