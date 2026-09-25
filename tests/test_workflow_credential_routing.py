"""Credentials never follow a caller-chosen route (2026-09-25).

- A trigger param set `api_base` on any step that did not pin one, and the
  routed platform key then rode to that host as a Bearer token.
- The routing decision, key and header values included, was stored in the
  execution's results, which WORKFLOW_RESULTS returns to the caller.
"""

from __future__ import annotations

import sys
import types

import pytest
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.step_router import reset_feedback_store
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import _assemble_step_config, run_workflow_record

_SEEN: dict = {}


class _ProbeStep(Step):
    name = "probe"

    def setup(self, config):
        self._config = dict(config)

    def run(self, ctx: StepContext) -> StepResult:
        _SEEN.update(
            api_base=ctx.config.get("api_base"),
            api_key=ctx.config.get("api_key"),
            routing=ctx.config.get("_routing"),
        )
        return StepResult(extra={"ok": True})


@pytest.fixture(autouse=True)
def _probe(monkeypatch):
    module = types.ModuleType("umrs_test_credential_probe")
    module._ProbeStep = _ProbeStep  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umrs_test_credential_probe", module)
    reset_execution_registry_for_tests()
    reset_feedback_store()
    _SEEN.clear()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


def _run(tmp_path, params):
    record = get_execution_registry().create(workflow_name="probe", user_id="alice", params=params)
    workflow = {
        "name": "probe",
        "version": "1",
        "steps": [
            {"name": "probe", "entrypoint": "umrs_test_credential_probe:_ProbeStep", "config": {}}
        ],
    }
    run_workflow_record(record, source_path=str(tmp_path), workflow_json=workflow)
    assert record.status == "succeeded", record.error
    return record


def test_trigger_params_cannot_set_routing_or_credential_keys():
    config = _assemble_step_config(
        {"name": "s", "config": {}},
        {"api_base": "http://attacker", "api_key": "x", "headers": {}, "_routing": {}, "q": 1},
    )
    assert "api_base" not in config and "api_key" not in config
    assert "headers" not in config and "_routing" not in config
    assert config["q"] == 1


def test_a_param_api_base_never_receives_the_platform_key(tmp_path, monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway:8080/v1")
    _run(tmp_path, {"api_base": "http://attacker.example/v1"})
    assert _SEEN["api_base"] == "http://gateway:8080/v1"
    assert _SEEN["api_key"] == "sk-platform"  # only ever for the platform base


def test_results_and_step_introspection_carry_no_key_values(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_STEP_ROUTER_MOCK", "1")  # mock decision carries sk-mock
    record = _run(tmp_path, {})
    stored = record.results["_routing"]["probe"]
    assert stored["api_key"] == "<redacted>"
    assert all(v == "<redacted>" for v in stored["headers"].values())
    assert _SEEN["routing"]["api_key"] == "<redacted>"
    assert _SEEN["api_key"] == "sk-mock"  # the step itself still gets its key
