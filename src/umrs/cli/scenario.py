"""UMRS CLI — scenario runner subcommand.

Provides headless execution of operational validation scenarios:
- ``umrs scenario list`` — list available scenarios
- ``umrs scenario run <name>`` — run a scenario headlessly

Scenarios exercise real OperationAPI/OperationSDK calls when connected
to an UMRS deployment. Without a connection, steps are simulated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status of a scenario step."""

    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class DemoStepResult:
    """Result of executing a scenario step."""

    number: int
    label: str
    status: str
    progress: float
    error: str | None = None


@dataclass
class ScenarioResult:
    """Result of running an entire scenario."""

    name: str
    steps: list[DemoStepResult]
    total_time: float
    success: bool


# ---------------------------------------------------------------------------
# Scenario definitions (from scenario_runner.py)
# ---------------------------------------------------------------------------

_DEMO_PIPELINES: dict[str, list[tuple[str, str, str]]] = {
    "RAG Pipeline": [
        ("Upload Texts", "Upload input documents/texts to bucket", "upload_texts"),
        ("Create VectorDB", "Build vector index from documents", "create_vectordb"),
        (
            "Index Embeddings",
            "Generate and store embeddings",
            "index_embeddings",
        ),
        ("Query VectorDB", "Test similarity search queries", "query_vectordb"),
        ("RAG Pipeline", "Run retrieval-augmented generation", "run_rag"),
    ],
    "Vision Inference": [
        ("Upload Images", "Upload images to bucket", "upload_images"),
        (
            "Pre-process Images",
            "Resize, normalize, augment",
            "preprocess_images",
        ),
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
        (
            "Query Multi-Modal",
            "Run multi-modal similarity search",
            "query_multimodal",
        ),
        (
            "RAG Response",
            "Generate multi-modal answer",
            "run_multimodal_rag",
        ),
    ],
}


def _simulate_step(
    label: str,
    progress_interval: float = 0.05,
) -> DemoStepResult:
    """Simulate a step by sleeping and reporting progress."""
    start = time.monotonic()
    step = DemoStepResult(number=0, label=label, status="running", progress=0.0)
    try:
        for pct in range(0, 101, 10):
            time.sleep(progress_interval)
            step.progress = float(pct)
        step.status = "done"
        step.progress = 100.0
    except KeyboardInterrupt:
        step.status = "failed"
        step.error = "Cancelled"
    elapsed = time.monotonic() - start
    if step.status == "done" and elapsed < 0.5:
        time.sleep(0.5 - elapsed)
    return step


def _get_api() -> Any:
    """Try to import OperationAPI. Returns None if not available."""
    try:
        from ..tui_v2.models import OperationQueue
        from ..tui_v2.op_api import OperationAPI

        op_queue = OperationQueue()
        return OperationAPI(
            control_api=None,
            control_mq=None,
            client_id="cli",
            operation_queue=op_queue,
        ), op_queue
    except ImportError:
        return None, None


def _run_scenario_step(
    api: Any | None,
    queue: Any | None,
    number: int,
    label: str,
    description: str,
    machine_key: str,
) -> DemoStepResult:
    """Run a single scenario step (real or simulated)."""
    step = DemoStepResult(number=number, label=label, status="running", progress=0.0)

    # Map machine keys to OperationAPI methods
    BUCKET = "default"
    op_methods: dict[str, tuple[str, dict[str, Any]]] = {
        "upload_texts": (
            "upload_file",
            {
                "file_path": description or "./data/texts",
                "bucket_alias": BUCKET,
                "subdir": "texts",
            },
        ),
        "upload_images": (
            "upload_file",
            {
                "file_path": description or "./data/images",
                "bucket_alias": BUCKET,
                "subdir": "images",
            },
        ),
        "download_results": (
            "download_from_bucket",
            {
                "bucket_alias": BUCKET,
                "s3_path": "results",
                "local_path": description or "./data/results",
            },
        ),
        "run_inference": (
            "run_workflow",
            {"workflow_name": "vision-inference", "bucket_alias": BUCKET},
        ),
        "run_rag": (
            "run_experience",
            {"experience_name": "rag-pipeline", "bucket_alias": BUCKET},
        ),
        "run_multimodal_rag": (
            "run_experience",
            {"experience_name": "rag-pipeline", "bucket_alias": BUCKET},
        ),
    }

    if api is None or machine_key not in op_methods:
        # Simulated step
        return _simulate_step(label)

    method_name, kwargs = op_methods[machine_key]
    try:
        op_id = getattr(api, method_name)(**kwargs)
        # Poll the queue for progress
        deadline = time.monotonic() + 30  # 30s timeout
        while time.monotonic() < deadline:
            op = queue.get(op_id) if queue else None
            if op is None:
                time.sleep(0.3)
                continue
            if op.status.value in ("succeeded", "failed", "cancelled"):
                step.progress = 100.0
                step.status = "done" if op.status.value == "succeeded" else "failed"
                if op.status.value != "succeeded":
                    step.error = op.error or "Operation failed"
                return step
            pct = float(op.progress.percent)
            step.progress = max(step.progress, pct)
            time.sleep(0.3)
        # Timeout — mark as failed
        step.status = "failed"
        step.error = "Timed out after 30s"
    except Exception as exc:
        step.status = "failed"
        step.error = str(exc)

    return step


def cmd_scenario_list(args: argparse.Namespace, config: dict) -> None:
    """List available scenarios."""
    json_output = getattr(args, "json", False)
    if json_output:
        result = {
            "scenarios": list(_DEMO_PIPELINES.keys()),
            "details": {
                name: [
                    {"label": step[0], "description": step[1], "key": step[2]}
                    for step in steps
                ]
                for name, steps in _DEMO_PIPELINES.items()
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print("Available scenarios:")
        print()
        for name, steps in _DEMO_PIPELINES.items():
            print(f"  {name}  ({len(steps)} steps)")
            for label, desc, _ in steps:
                print(f"    - {label}: {desc}")
            print()


def cmd_scenario_run(args: argparse.Namespace, config: dict) -> None:
    """Run a scenario headlessly."""
    name = args.name
    if name not in _DEMO_PIPELINES:
        print(f"Unknown scenario: {name}", file=sys.stderr)
        print(f"Available: {', '.join(_DEMO_PIPELINES.keys())}", file=sys.stderr)
        sys.exit(1)

    json_output = getattr(args, "json", False)
    api, queue = _get_api()

    steps = _DEMO_PIPELINES[name]
    results: list[DemoStepResult] = []
    success = True
    start_time = time.monotonic()

    for idx, (label, desc, mk) in enumerate(steps, 1):
        result = _run_scenario_step(api, queue, idx, label, desc, mk)
        results.append(result)
        if result.status == "failed":
            success = False

        if json_output:
            print(json.dumps(asdict(result)))
        else:
            _STATUS_MAP: dict[str, str] = {"done": "+", "running": "...", "failed": "x"}
            status_icon = _STATUS_MAP.get(result.status, "?")
            msg = f"  [{status_icon}] {result.label}: {result.progress:.0f}%"
            if result.error:
                msg += f" ({result.error})"
            print(msg)

    total_time = time.monotonic() - start_time

    if json_output:
        overall = ScenarioResult(
            name=name,
            steps=results,
            total_time=round(total_time, 2),
            success=success,
        )
        print(json.dumps(asdict(overall)))
    else:
        print()
        overall_status = "SUCCESS" if success else "FAILED"
        print(f"  {overall_status}: {name} in {total_time:.1f}s")
        if not success:
            failed = [r for r in results if r.status == "failed"]
            for r in failed:
                print(f"    Failed: {r.label} — {r.error}")

    sys.exit(0 if success else 1)


def build_scenario_parser(subparsers: Any) -> None:
    """Build the scenario subcommand parser.

    Adds a ``scenario`` subcommand with ``list`` and ``run`` sub-subcommands.
    """
    # scenario subcommand
    scenario_parser = subparsers.add_parser(
        "scenario", help="Operational validation scenarios"
    )
    scenario_subparsers = scenario_parser.add_subparsers(dest="scenario_command")

    # list subcommand
    list_parser = scenario_subparsers.add_parser(
        "list", help="List available scenarios"
    )
    list_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    list_parser.set_defaults(func=cmd_scenario_list)

    # run subcommand
    run_parser = scenario_subparsers.add_parser("run", help="Run a scenario")
    run_parser.add_argument("name", help="Scenario name")
    run_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    run_parser.set_defaults(func=cmd_scenario_run)
