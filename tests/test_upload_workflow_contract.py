"""The upload contract (2026-10-02, r4_contracts.workflow_package).

An uploaded workflow.json must carry a description and an explicit flow
graph: every step lists ``depends_on`` ([] for a step that starts the run).
A refusal names every problem at once and says how to fix each; advice
(parameters no schema declares, a placeholder nothing fills) never refuses
an upload and comes back in the notice.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.pipeline_sdk.workflow import WorkflowDefinition
from umrs.server.file_handling import handle_upload_workflow

_LLM = "umrs.pipeline_sdk.templates.llm_step:LLMStep"


def _workflow(**overrides):
    wf = {
        "name": "contract",
        "version": "1.0",
        "description": "Segments an image, then measures each region.",
        "source": {"type": "local_path", "path": "."},
        "steps": [
            {"name": "segment", "entrypoint": "steps:Segment", "depends_on": []},
            {"name": "measure", "entrypoint": "steps:Measure", "depends_on": ["segment"]},
        ],
    }
    wf.update(overrides)
    return wf


def _zip(tmp_path: Path, workflow: dict) -> str:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("workflow.json", json.dumps(workflow))
        zf.writestr("steps.py", "class Segment: pass\nclass Measure: pass\n")
    path = tmp_path / "upload.zip"
    path.write_bytes(buffer.getvalue())
    return str(path)


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail(f"host process {a!r}"))


def _upload(tmp_path, workflow):
    return handle_upload_workflow("alice", _zip(tmp_path, workflow), "contract")


def test_a_described_explicit_graph_is_accepted(tmp_path):
    ok, dest, notice = _upload(tmp_path, _workflow())
    assert ok, notice
    assert notice is None
    assert Path(dest, "workflow.json").is_file()


def test_no_description_is_refused_with_what_to_write(tmp_path):
    ok, dest, message = _upload(tmp_path, _workflow(description="  "))
    assert not ok and dest is None
    assert "description: missing" in message
    assert "what it produces" in message


def test_a_step_without_depends_on_is_refused_with_a_hint(tmp_path):
    wf = _workflow()
    del wf["steps"][1]["depends_on"]
    ok, _dest, message = _upload(tmp_path, wf)
    assert not ok
    assert 'steps[1] "measure".depends_on: missing' in message
    assert '["segment"]' in message  # the step before it, offered as the likely edge


def test_the_old_implicit_chain_is_refused_and_every_problem_named(tmp_path):
    # No depends_on anywhere (the runner's declaration-order chain) and no
    # description: one refusal lists all three problems.
    wf = _workflow(description="")
    for step in wf["steps"]:
        del step["depends_on"]
    ok, _dest, message = _upload(tmp_path, wf)
    assert not ok
    assert "(1) description: missing" in message
    assert '(2) steps[0] "segment".depends_on' in message
    assert '(3) steps[1] "measure".depends_on' in message


@pytest.mark.parametrize(
    "deps, needle",
    [
        (["segmnt"], '"segmnt" is not a step of this workflow'),
        (["measure"], "a step cannot wait for itself"),
        ("segment", "must be a list of step names"),
    ],
)
def test_broken_edges_are_refused(tmp_path, deps, needle):
    wf = _workflow()
    wf["steps"][1]["depends_on"] = deps
    ok, _dest, message = _upload(tmp_path, wf)
    assert not ok and needle in message


def test_a_cycle_is_refused_with_its_path(tmp_path):
    wf = _workflow()
    wf["steps"][0]["depends_on"] = ["measure"]
    ok, _dest, message = _upload(tmp_path, wf)
    assert not ok
    assert "cycle" in message and "→" in message


def test_a_gate_step_needs_no_entrypoint(tmp_path):
    wf = _workflow()
    wf["steps"].append({"name": "review", "gate": True, "depends_on": ["measure"]})
    ok, _dest, notice = _upload(tmp_path, wf)
    assert ok, notice
    parsed = WorkflowDefinition.from_dict(wf)
    assert parsed.steps[-1].gate and parsed.steps[-1].entrypoint == ""


def test_advice_rides_the_notice_and_never_refuses(tmp_path):
    wf = _workflow(
        steps=[
            {
                "name": "ask",
                "entrypoint": _LLM,
                "depends_on": [],
                "config": {
                    "model": "{{model}}",
                    "prompt_template": "Answer {{ params.question }} for {{ params.project_id }}.",
                },
            }
        ]
    )
    ok, _dest, notice = _upload(tmp_path, wf)
    assert ok
    assert "{{ model }} is not filled in at run time" in notice
    assert "the prompts use question but no parameter_schema declares" in notice
    assert "project_id" not in notice.split("the prompts use", 1)[1]  # filled by the platform


def test_a_declared_parameter_silences_the_advice(tmp_path):
    wf = _workflow(
        parameter_schema={"type": "object", "properties": {"question": {"type": "string"}}},
        steps=[
            {
                "name": "ask",
                "entrypoint": _LLM,
                "depends_on": [],
                "config": {"prompt_template": "Answer {{ params.question }}."},
            }
        ],
    )
    ok, _dest, notice = _upload(tmp_path, wf)
    assert ok and notice is None


def test_http_upload_answers_400_with_the_problems(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
    from r4_contracts.handshake_hmac import sign_claims

    app = FastAPI()
    app.include_router(router)
    headers = {
        "X-User-ID": "alice",
        "X-Verified-Project-ID": "project-a",
        "X-Verified-Is-Admin": "true",
        "X-Verified-HMAC": sign_claims("test-secret", "alice", "project-a", True),
    }
    wf = _workflow(description="")
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=str(tmp_path / "catalog.json"),
    ):
        response = TestClient(app).post(
            "/umrs/workflows/upload",
            headers=headers,
            data={"workflow_name": "contract"},
            files={"file": ("contract.zip", Path(_zip(tmp_path, wf)).read_bytes())},
        )
    assert response.status_code == 400, response.text
    assert "description: missing" in response.text


def test_the_contract_knows_exactly_the_runner_enums():
    from r4_contracts.workflow_package import EXECUTION_MODES, SOURCE_TYPES
    from umrs.pipeline_sdk.workflow import ExecutionMode, SourceType

    assert set(SOURCE_TYPES) == {t.value for t in SourceType}
    assert set(EXECUTION_MODES) == {m.value for m in ExecutionMode}


def test_an_uppercase_enum_is_refused_in_plain_words(tmp_path):
    ok, _dest, message = _upload(tmp_path, _workflow(source={"type": "LOCAL_PATH", "path": "."}, execution_mode="IN_PROCESS"))
    assert not ok
    assert '"LOCAL_PATH" is not one of local_path' in message
    assert '"IN_PROCESS" is not one of in_process' in message
