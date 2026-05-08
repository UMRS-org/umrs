"""End-to-end test for the in-process workflow runner.

Uses ``UMRS_VISION_MOCK=1`` and ``UMRS_LLM_MOCK=1`` so no external
services are required.  Validates that the new step templates
(``VisionStep``, ``OCRStep``, ``LLMStep``) compose through the
``WorkflowExecutionRegistry`` and produce the expected structured
output.
"""

from __future__ import annotations

import json
import os

import pytest

from umrs.server.step_router import reset_feedback_store
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import run_workflow_record


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    reset_feedback_store()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.setenv("UMRS_VISION_MOCK", "1")
    monkeypatch.setenv("UMRS_LLM_MOCK", "1")
    yield


def _figure_workflow_json():
    return {
        "name": "scientific-figure-understanding",
        "version": "0.1",
        "steps": [
            {
                "name": "vision_dense",
                "entrypoint": (
                    "umrs.pipeline_sdk.templates.vision_step:VisionStep"
                ),
                "config": {"mode": "dense"},
            },
            {
                "name": "vision_ocr",
                "entrypoint": (
                    "umrs.pipeline_sdk.templates.ocr_step:OCRStep"
                ),
                "config": {},
            },
            {
                "name": "reason",
                "entrypoint": (
                    "umrs.pipeline_sdk.templates.llm_step:LLMStep"
                ),
                "depends_on": ["vision_dense", "vision_ocr"],
                "config": {
                    "model": "rivault-llm-static",
                    "system_prompt": "You are a scientific figure analyst.",
                    "prompt_template": (
                        "DESC: {{ steps.vision_dense.dense_description }}\n"
                        "OCR: {{ steps.vision_ocr.ocr_text }}"
                    ),
                    "output_schema": {
                        "type": "object",
                        "properties": {"finding": {"type": "string"}},
                    },
                },
            },
        ],
    }


def test_runner_executes_three_step_dag(tmp_path):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="scientific-figure-understanding",
        user_id="alice",
        params={"image": "fake.png"},
    )

    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_figure_workflow_json(),
    )

    assert record.status == "succeeded", record.error
    assert record.progress == 1.0

    step_names = [s.name for s in record.steps.values()]
    assert {"vision_dense", "vision_ocr", "reason"} <= set(step_names)
    for step in record.steps.values():
        assert step.status == "succeeded"

    final = record.results["final"]["extra"]
    # The mock LLM returns a JSON string when output_schema is set.
    assert "response_text" in final
    assert "structured" in final
    assert isinstance(final["structured"], dict)
    assert final["structured"].get("_mock") is True

    # Vision step records the perception via S3 shim.
    artifacts = list(record.artifacts)
    assert any("perception.json" in key for key in artifacts)
    assert any("response.json" in key for key in artifacts)


def test_runner_marks_failure_on_unknown_entrypoint(tmp_path):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="bad",
        user_id="alice",
    )

    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "bad",
            "steps": [
                {
                    "name": "bogus",
                    "entrypoint": "umrs.does_not_exist:Nope",
                    "config": {},
                }
            ],
        },
    )

    assert record.status == "failed"
    assert "bogus" in (record.error or "")


def test_runner_respects_cancel_request(tmp_path):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="cancel-me",
        user_id="alice",
    )
    record.cancel_requested = True
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_figure_workflow_json(),
    )
    assert record.status == "cancelled"


def test_runner_records_routing_decisions_per_step(tmp_path, monkeypatch):
    """Each step's routing decision must be recorded in record.results['_routing']."""
    monkeypatch.setenv("UMRS_STEP_ROUTER_MOCK", "1")

    registry = get_execution_registry()
    record = registry.create(
        workflow_name="scientific-figure-understanding",
        user_id="alice",
        params={"image": "fake.png"},
    )

    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_figure_workflow_json(),
    )

    assert record.status == "succeeded", record.error
    routing = record.results.get("_routing", {})
    # Every step must have a decision logged.
    assert set(routing.keys()) == {"vision_dense", "vision_ocr", "reason"}
    for step_name, dec in routing.items():
        assert dec["resolved_via"] == "mock"
        assert dec["api_base"] == "http://router-mock.invalid"


def test_runner_fails_step_when_routing_strict_mode_unmatched(tmp_path, monkeypatch):
    """A step that declares ``requires`` must fail when the registry yields nothing."""
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    from umrs.server.step_router import StepRouter

    # Router with empty registry -> StepRoutingError on first capability step.
    router = StepRouter(
        find_models=lambda caps, mt: [],
        find_endpoint=lambda *a, **k: None,
    )

    registry = get_execution_registry()
    record = registry.create(
        workflow_name="needs-vision",
        user_id="alice",
    )

    workflow = {
        "name": "needs-vision",
        "steps": [
            {
                "name": "vd",
                "entrypoint": (
                    "umrs.pipeline_sdk.templates.vision_step:VisionStep"
                ),
                "config": {"mode": "dense"},
                "requires": ["vision"],
                "model_type": "vision",
            },
        ],
    }

    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=workflow,
        router=router,
    )

    assert record.status == "failed"
    assert "vd" in (record.error or "")
    assert "vision" in (record.error or "")
    # The step itself was marked failed with a routing message.
    vd_step = record.steps["vd"]
    assert vd_step.status == "failed"


def test_runner_injects_routing_into_step_config(tmp_path, monkeypatch):
    """The routing decision must be visible in ctx.config['_routing']."""
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    seen: dict = {}

    # Build a fake Step that records what it was given. We register it as
    # an importable module so the runner's _load_step_class can find it.
    import sys
    import types

    from umrs.pipeline_sdk.step import Step, StepContext, StepResult

    class _ProbeStep(Step):
        name = "probe"

        def setup(self, config):
            self._config = dict(config)

        def run(self, ctx: StepContext) -> StepResult:
            seen["routing"] = ctx.config.get("_routing")
            seen["api_base"] = ctx.config.get("api_base")
            seen["model"] = ctx.config.get("model")
            return StepResult(extra={"ok": True})

    probe_module = types.ModuleType("umrs_test_probe_module")
    probe_module._ProbeStep = _ProbeStep  # type: ignore[attr-defined]
    sys.modules["umrs_test_probe_module"] = probe_module
    monkeypatch.setattr(
        "sys.modules", sys.modules
    )  # ensure module visible during import

    from umrs.server.step_router import StepRouter

    router = StepRouter(
        find_models=lambda caps, mt: [
            {"name": "vlm-system", "scope": "system", "model_type": "vision"}
        ],
        find_endpoint=lambda model_name, *a, **k: {
            "litellm_url": "http://gw/llm/system",
            "served_model_name": "Qwen/Qwen2.5-VL",
            "job_id": "ep-sys-1",
        },
    )

    registry = get_execution_registry()
    record = registry.create(workflow_name="probe", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "probe",
            "steps": [
                {
                    "name": "probe",
                    "entrypoint": "umrs_test_probe_module:_ProbeStep",
                    "config": {},
                    "requires": ["vision"],
                    "model_type": "vision",
                },
            ],
        },
        router=router,
    )
    assert record.status == "succeeded", record.error
    assert seen["api_base"] == "http://gw/llm/system"
    assert seen["model"] == "Qwen/Qwen2.5-VL"
    assert seen["routing"]["resolved_via"] == "registry_capability"
    assert seen["routing"]["endpoint_id"] == "ep-sys-1"
