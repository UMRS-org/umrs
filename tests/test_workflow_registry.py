"""
Unit tests for Workflow Registry (Phase 2).

Tests:
- register_workflow() — basic registration, duplicate update
- _handle_list_workflows() — listing with scope/user/name filters
- LIST_WORKFLOWS constant exists in TaskCmd
- StepSpec.extends field
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from umrs.server.workflow_registry_handler import (
    _handle_list_workflows,
    _load_catalog,
    _save_catalog,
    handle_workflow_command,
    register_workflow,
)
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.pipeline_sdk.workflow import StepSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path):
    """Redirect the workflow catalog to a temporary directory."""
    catalog_file = str(tmp_path / "workflow_catalog.json")
    with patch("umrs.server.workflow_registry_handler._catalog_path", return_value=catalog_file):
        yield catalog_file


def _sample_workflow_json(steps=None, description="test workflow", version="1.0"):
    """Return a minimal workflow.json dict."""
    if steps is None:
        steps = [
            {"name": "preprocess", "entrypoint": "my_mod:Preprocess"},
            {"name": "embed", "entrypoint": "my_mod:Embed"},
        ]
    return {
        "name": "sample_wf",
        "version": version,
        "description": description,
        "steps": steps,
        "requirements": ["numpy"],
    }


# ---------------------------------------------------------------------------
# TaskCmd constant
# ---------------------------------------------------------------------------


class TestTaskCmdConstant:
    def test_list_workflows_exists(self):
        assert hasattr(TaskCmd, "LIST_WORKFLOWS")
        assert TaskCmd.LIST_WORKFLOWS == "LIST_WORKFLOWS"


# ---------------------------------------------------------------------------
# register_workflow
# ---------------------------------------------------------------------------


class TestRegisterWorkflow:
    def test_basic_register(self):
        wf = _sample_workflow_json()
        result = register_workflow(
            workflow_name="wf1",
            version="1.0",
            user_id="alice",
            source_path="/tmp/wf1",
            workflow_json=wf,
        )
        assert result["status"] == "ok"
        assert "registered" in result["message"].lower() or "updated" in result["message"].lower()
        assert result["workflow"]["name"] == "wf1"
        assert len(result["workflow"]["steps"]) == 2

    def test_rejects_unsupported_scope(self):
        result = register_workflow(
            workflow_name="wf1",
            version="1.0",
            user_id="alice",
            source_path="/tmp/wf1",
            workflow_json=_sample_workflow_json(),
            scope="project",
        )
        assert result["status"] == "error"
        assert "user" in result["message"]
        assert "system" in result["message"]

    def test_duplicate_update(self):
        wf = _sample_workflow_json(description="v1")
        register_workflow("wf1", "1.0", "alice", "/tmp/wf1", wf)

        wf2 = _sample_workflow_json(description="v2")
        result = register_workflow("wf1", "1.0", "alice", "/tmp/wf1", wf2)
        assert result["status"] == "ok"
        assert result["workflow"]["description"] == "v2"

        # Should still be 1 entry, not 2
        raw = _handle_list_workflows("alice", "")
        parsed = json.loads(raw)
        assert parsed["count"] == 1

    def test_different_version_creates_new_entry(self):
        wf = _sample_workflow_json()
        register_workflow("wf1", "1.0", "alice", "/tmp/wf1", wf)
        register_workflow("wf1", "2.0", "alice", "/tmp/wf1_v2", wf)

        raw = _handle_list_workflows("alice", "")
        parsed = json.loads(raw)
        assert parsed["count"] == 2

    def test_different_user_creates_new_entry(self):
        wf = _sample_workflow_json()
        register_workflow("wf1", "1.0", "alice", "/tmp/wf1", wf)
        register_workflow("wf1", "1.0", "bob", "/tmp/wf1", wf)

        raw = _handle_list_workflows("_", "")
        parsed = json.loads(raw)
        assert parsed["count"] == 2

    def test_steps_metadata_extracted(self):
        wf = _sample_workflow_json(
            steps=[
                {"name": "step_a", "entrypoint": "mod:A", "config": {"x": 1}},
            ]
        )
        result = register_workflow("wf1", "1.0", "alice", "/tmp/wf1", wf)
        steps = result["workflow"]["steps"]
        assert len(steps) == 1
        assert steps[0]["name"] == "step_a"
        assert steps[0]["entrypoint"] == "mod:A"


# ---------------------------------------------------------------------------
# _handle_list_workflows
# ---------------------------------------------------------------------------


class TestListWorkflows:
    def _populate(self):
        """Register several workflows for filtering tests."""
        register_workflow("wf_a", "1.0", "alice", "/tmp/a", _sample_workflow_json(), scope="user")
        register_workflow("wf_b", "1.0", "bob", "/tmp/b", _sample_workflow_json(), scope="system")
        register_workflow("wf_c", "2.0", "alice", "/tmp/c", _sample_workflow_json(), scope="system")

    def test_list_all(self):
        self._populate()
        raw = _handle_list_workflows("_", "")
        parsed = json.loads(raw)
        assert parsed["status"] == "ok"
        assert parsed["count"] == 3

    def test_filter_by_scope(self):
        self._populate()
        raw = _handle_list_workflows("_", json.dumps({"scope": "system"}))
        parsed = json.loads(raw)
        assert parsed["count"] == 2

    def test_filter_by_user(self):
        self._populate()
        raw = _handle_list_workflows("_", json.dumps({"user_id": "alice"}))
        parsed = json.loads(raw)
        assert parsed["count"] == 2

    def test_filter_by_name_substring(self):
        self._populate()
        raw = _handle_list_workflows("_", json.dumps({"name": "wf_b"}))
        parsed = json.loads(raw)
        assert parsed["count"] == 1
        assert parsed["workflows"][0]["name"] == "wf_b"

    def test_combined_filters(self):
        self._populate()
        raw = _handle_list_workflows("_", json.dumps({"scope": "system", "user_id": "alice"}))
        parsed = json.loads(raw)
        assert parsed["count"] == 1
        assert parsed["workflows"][0]["name"] == "wf_c"

    def test_empty_result(self):
        raw = _handle_list_workflows("_", "")
        parsed = json.loads(raw)
        assert parsed["count"] == 0
        assert parsed["workflows"] == []

    def test_invalid_json_payload_treated_as_no_filter(self):
        self._populate()
        raw = _handle_list_workflows("_", "not-json")
        parsed = json.loads(raw)
        assert parsed["count"] == 3


# ---------------------------------------------------------------------------
# handle_workflow_command dispatcher
# ---------------------------------------------------------------------------


class TestHandleWorkflowCommand:
    def test_list_workflows_dispatches(self):
        register_workflow("wf1", "1.0", "alice", "/tmp/wf1", _sample_workflow_json())
        raw = handle_workflow_command(TaskCmd.LIST_WORKFLOWS, "alice", "")
        parsed = json.loads(raw)
        assert parsed["status"] == "ok"
        assert parsed["count"] == 1

    def test_unknown_command_returns_error(self):
        raw = handle_workflow_command("UNKNOWN_CMD", "alice", "")
        parsed = json.loads(raw)
        assert parsed["status"] == "error"


# ---------------------------------------------------------------------------
# StepSpec.extends field
# ---------------------------------------------------------------------------


class TestStepSpecExtends:
    def test_extends_default_none(self):
        spec = StepSpec(name="s1", entrypoint="mod:Cls")
        assert spec.extends is None

    def test_extends_set(self):
        spec = StepSpec(name="s1", entrypoint="mod:Cls", extends="base_step@1.0")
        assert spec.extends == "base_step@1.0"

    def test_extends_in_to_dict(self):
        spec = StepSpec(name="s1", entrypoint="mod:Cls", extends="base@2.0")
        d = spec.to_dict()
        assert d["extends"] == "base@2.0"

    def test_extends_absent_when_none(self):
        spec = StepSpec(name="s1", entrypoint="mod:Cls")
        d = spec.to_dict()
        assert "extends" not in d

    def test_extends_roundtrip(self):
        original = StepSpec(name="s1", entrypoint="mod:Cls", extends="base@1.0", config={"k": "v"})
        d = original.to_dict()
        restored = StepSpec.from_dict(d)
        assert restored.extends == "base@1.0"
        assert restored.name == "s1"
        assert restored.config == {"k": "v"}

    def test_from_dict_without_extends(self):
        d = {"name": "s1", "entrypoint": "mod:Cls", "config": {}}
        spec = StepSpec.from_dict(d)
        assert spec.extends is None
