"""Integration: video-ai YOLO→Aggregate through run_workflow_record.

Requires the demo tree under ``ai4s-platform/demos/video_ai_workflow``.
Uses the in-memory S3 shim with bucket-aware parent handoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from umrs.server.step_router import reset_feedback_store
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import run_workflow_record

DEMO_ROOT = Path(__file__).resolve().parents[2] / "demos" / "video_ai_workflow" / "workflow"
FIX = DEMO_ROOT / "fixtures"


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    reset_feedback_store()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


@pytest.mark.skipif(
    not (DEMO_ROOT / "yolo_step.py").is_file(),
    reason="demos/video_ai_workflow not present",
)
@pytest.mark.skipif(
    not (FIX / "video_inventory.csv").is_file(),
    reason="video_ai fixtures missing",
)
def test_video_ai_yolo_aggregate_via_workflow_runner(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("pandas")
    pytest.importorskip("yaml")

    # Ensure sample video exists for staging.
    sample = FIX / "sample.mp4"
    if not sample.is_file():
        import subprocess

        subprocess.check_call([sys.executable, str(FIX / "make_sample_video.py")])

    # Seed in-memory blobs by monkeypatching record after create is awkward;
    # instead pass params and pre-seed via a tiny wrapper around put on first step.
    # The runner starts with empty blobs — seed them on the record before run.
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="video-ai-yolo",
        user_id="u",
        params={
            "inventory_key": "data/video_inventory.csv",
            "analysis_plan_key": "data/analysis_plan.csv",
            "video_keys": ["videos/sample.mp4"],
            "runtime": {"mode": "mock"},
        },
    )
    record.results.setdefault("blobs", {})
    record.results["blobs"]["wf-runner/data/video_inventory.csv"] = (
        FIX / "video_inventory.csv"
    ).read_bytes()
    record.results["blobs"]["wf-runner/data/analysis_plan.csv"] = (
        FIX / "analysis_plan.csv"
    ).read_bytes()
    record.results["blobs"]["wf-runner/videos/sample.mp4"] = sample.read_bytes()

    workflow_json = {
        "name": "video-ai-yolo",
        "execution_mode": "in_process",
        "steps": [
            {
                "name": "yolo",
                "entrypoint": "yolo_step:YoloStep",
                "depends_on": [],
                "config": {"runtime": {"mode": "mock"}},
            },
            {
                "name": "aggregate",
                "entrypoint": "aggregate_step:AggregateStep",
                "depends_on": ["yolo"],
            },
        ],
    }

    run_workflow_record(
        record,
        source_path=str(DEMO_ROOT),
        workflow_json=workflow_json,
    )

    assert record.status == "succeeded", record.error
    blobs = record.results.get("blobs", {})
    assert any(k.startswith("wf-runner-output/") and k.endswith("detections.csv") for k in blobs)
    assert any(k.startswith("wf-runner-output/") and k.endswith("all_metrics.csv") for k in blobs)
    # Source bucket remains distinct from output bucket.
    assert any(k.startswith("wf-runner/") for k in blobs)
