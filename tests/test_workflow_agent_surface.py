"""ADR-105 S1 (umrs half): typed workflow parameters and the agent surface.

- D-3: ``_validate_params_against_schema`` is full JSON Schema (types, enums,
  ``additionalProperties``, nested ``required``); the typed 400 keeps
  ``missing`` and adds ``errors``; a workflow whose own schema is invalid
  falls back to the required-keys check; platform-injected keys
  (``input_files``, ``project_id``) are invisible to a schema that does not
  declare them.
- D-1: the agent-exposed routes carry explicit, unique ``operation_id``s
  named after the R4 native tool each backs.
- D-1 "type before expose": the workflow response models describe today's
  payloads without changing them (undeclared keys survive).
"""

from __future__ import annotations

import json
from collections import Counter
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from umrs.api.workflow_models import (
    WorkflowDescribeResponse,
    WorkflowExecutionResponse,
    WorkflowListResponse,
    WorkflowResultsResponse,
    WorkflowTriggerResponse,
)
from umrs.api.workflows import _validate_params_against_schema, router
from umrs.server.workflow_registry_handler import register_workflow

# ── D-3: full JSON Schema validation ───────────────────────────────────────

_SCHEMA = {
    "type": "object",
    "properties": {
        "bucket": {"type": "string"},
        "steps": {"type": "integer"},
        "mode": {"enum": ["fast", "accurate"]},
        "grid": {
            "type": "object",
            "properties": {"nx": {"type": "integer"}},
            "required": ["nx"],
        },
    },
    "required": ["bucket"],
    "additionalProperties": False,
}


def _refusal(params: dict, schema: dict) -> dict:
    with pytest.raises(HTTPException) as caught:
        _validate_params_against_schema(params, schema)
    assert caught.value.status_code == 400
    detail = caught.value.detail
    assert detail["code"] == "workflow_parameters_invalid"
    return detail


def test_valid_params_pass():
    _validate_params_against_schema(
        {"bucket": "b", "steps": 3, "mode": "fast", "grid": {"nx": 4}}, _SCHEMA
    )


def test_type_mismatch_is_refused_with_path():
    detail = _refusal({"bucket": "b", "steps": "three"}, _SCHEMA)
    assert detail["missing"] == []
    assert detail["errors"] == [
        {"path": "/steps", "message": "'three' is not of type 'integer'"}
    ]
    assert "/steps" in detail["message"]


def test_enum_violation_is_refused():
    detail = _refusal({"bucket": "b", "mode": "slow"}, _SCHEMA)
    assert [row["path"] for row in detail["errors"]] == ["/mode"]


def test_additional_properties_false_is_enforced():
    detail = _refusal({"bucket": "b", "typo_param": 1}, _SCHEMA)
    assert "typo_param" in detail["errors"][0]["message"]


def test_nested_required_is_enforced():
    detail = _refusal({"bucket": "b", "grid": {}}, _SCHEMA)
    assert detail["missing"] == []
    assert detail["errors"] == [{"path": "/grid", "message": "'nx' is a required property"}]


def test_missing_list_is_kept_beside_errors():
    detail = _refusal({"steps": 1}, _SCHEMA)
    assert detail["missing"] == ["bucket"]
    assert detail["message"] == "Missing required workflow parameters: bucket"
    assert {"path": "", "message": "'bucket' is a required property"} in detail["errors"]


def test_errors_are_capped_at_twenty():
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    detail = _refusal({f"p{i:02d}": "x" for i in range(30)}, schema)
    assert len(detail["errors"]) == 20


def test_invalid_workflow_schema_falls_back_to_required_keys():
    # "float" is not a JSON Schema type: the schema itself is invalid.
    schema = {
        "type": "object",
        "properties": {"dt": {"type": "float"}},
        "required": ["bucket"],
    }
    _validate_params_against_schema({"bucket": "b", "dt": "not-a-number"}, schema)
    detail = _refusal({"dt": 0.1}, schema)
    assert detail["missing"] == ["bucket"]
    assert detail["errors"] == []


def test_unresolvable_ref_falls_back_instead_of_500():
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "https://example.invalid/schema.json"}},
    }
    _validate_params_against_schema({"x": 1}, schema)


def test_injected_keys_are_hidden_from_an_undeclaring_schema():
    schema = {
        "type": "object",
        "properties": {"bucket": {"type": "string"}},
        "additionalProperties": False,
    }
    _validate_params_against_schema(
        {"bucket": "b", "input_files": ["k1"], "project_id": "p"}, schema
    )


def test_injected_keys_are_validated_when_declared():
    schema = {
        "type": "object",
        "properties": {"input_files": {"type": "array", "items": {"type": "string"}}},
        "additionalProperties": False,
    }
    detail = _refusal({"input_files": [1], "project_id": "p"}, schema)
    assert detail["errors"] == [{"path": "/input_files/0", "message": "1 is not of type 'string'"}]


# ── route level ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _handshake_secret(monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")


@pytest.fixture(autouse=True)
def _tmp_catalog(tmp_path):
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=str(tmp_path / "workflow_catalog.json"),
    ):
        yield


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _auth_headers(user: str = "alice", project: str = "project-a") -> dict[str, str]:
    from r4_contracts.handshake_hmac import sign_claims

    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims("test-secret", user, project, False),
    }


def _register_strict_workflow(tmp_path) -> None:
    workflow_json = {
        "name": "relax",
        "version": "1.0",
        "description": "strict schema",
        "source": {"type": "local_path", "path": "."},
        "steps": [{"name": "s1", "entrypoint": "m:S1", "depends_on": []}],
        "parameter_schema": {
            "type": "object",
            "properties": {"steps": {"type": "integer"}},
            "additionalProperties": False,
        },
    }
    workflow_dir = tmp_path / "relax"
    workflow_dir.mkdir()
    (workflow_dir / "workflow.json").write_text(json.dumps(workflow_json))
    register_workflow(
        workflow_name="relax",
        version="1.0",
        user_id="alice",
        source_path=str(workflow_dir),
        workflow_json=workflow_json,
        scope="user",
    )


def test_trigger_refuses_a_type_mismatch(client, tmp_path):
    _register_strict_workflow(tmp_path)
    response = client.post(
        "/umrs/workflows/relax/trigger",
        headers=_auth_headers(),
        json={"parameters": {"steps": "ten"}},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "workflow_parameters_invalid"
    assert detail["errors"][0]["path"] == "/steps"


def test_trigger_accepts_input_files_under_additional_properties_false(client, tmp_path):
    _register_strict_workflow(tmp_path)
    with patch(
        "umrs.api.workflows.handle_workflow_command",
        return_value=json.dumps({"status": "ok", "execution_id": "wf-1"}),
    ) as run:
        response = client.post(
            "/umrs/workflows/relax/trigger",
            headers=_auth_headers(),
            json={"parameters": {"steps": 10}, "input_files": ["raw/a.dat"]},
        )
    assert response.status_code == 200, response.text
    sent = json.loads(run.call_args.args[2])["params"]
    assert sent["input_files"] == ["raw/a.dat"]
    assert sent["project_id"] == "project-a"


# ── D-1: explicit operation ids on the agent-exposed routes ────────────────

AGENT_OPERATIONS: dict[str, tuple[str, str]] = {
    "workflow_list": ("get", "/umrs/workflows"),
    "workflow_describe": ("get", "/umrs/workflows/{workflow_name}"),
    "workflow_run": ("post", "/umrs/workflows/{workflow_name}/trigger"),
    "workflow_status": ("get", "/umrs/workflows/executions/{execution_id}"),
    "workflow_results": ("get", "/umrs/workflows/executions/{execution_id}/results"),
    "workflow_cancel": ("post", "/umrs/workflows/executions/{execution_id}/cancel"),
    "job_list": ("get", "/umrs/jobs"),
    "job_status": ("get", "/umrs/jobs/{job_id}"),
    "job_cancel": ("post", "/umrs/jobs/{job_id}/cancel"),
    "job_retry": ("post", "/umrs/jobs/{job_id}/retry"),
    "rag_query": ("post", "/umrs/rag/query"),
    "rag_list_documents": ("get", "/umrs/rag/documents"),
    "rag_upload": ("post", "/umrs/rag/upload"),
}


@pytest.fixture(scope="module")
def openapi_spec() -> dict:
    """The real sidecar app's OpenAPI, without a database pool."""
    from umrs.server import http_sidecar

    original = http_sidecar._build_db_pool
    http_sidecar._build_db_pool = lambda _app: None
    try:
        return http_sidecar._create_app().openapi()
    finally:
        http_sidecar._build_db_pool = original


def _operations(spec: dict) -> list[tuple[str, str, str]]:
    return [
        (op["operationId"], method, path)
        for path, ops in spec["paths"].items()
        for method, op in ops.items()
        if isinstance(op, dict) and "operationId" in op
    ]


@pytest.mark.parametrize("operation_id", sorted(AGENT_OPERATIONS))
def test_agent_operation_id_is_on_its_route_and_unique(openapi_spec, operation_id):
    found = [(m, p) for op, m, p in _operations(openapi_spec) if op == operation_id]
    assert found == [AGENT_OPERATIONS[operation_id]]


def test_agent_operations_carry_a_response_schema(openapi_spec):
    for method, path in AGENT_OPERATIONS.values():
        ok = openapi_spec["paths"][path][method]["responses"]["200"]
        assert ok["content"]["application/json"]["schema"], (method, path)


def test_no_other_route_reuses_an_agent_operation_id(openapi_spec):
    counts = Counter(op for op, _m, _p in _operations(openapi_spec))
    assert {name: counts[name] for name in AGENT_OPERATIONS} == dict.fromkeys(
        AGENT_OPERATIONS, 1
    )


# ── D-1: response models keep today's payloads byte-for-byte in content ───

_DURABLE_JOB = {"job_id": "j-1", "status": "queued", "created_at": "2026-11-02T00:00:00"}
_EXECUTION = {
    "execution_id": "wf-1",
    "workflow_name": "relax",
    "user_id": "alice",
    "status": "running",
    "submitted_at": 1730505600.5,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "progress": 0,
    "params": {"steps": 10},
    "steps": [{"name": "s1", "status": "running", "extra": {}}],
    "artifacts": [],
    "durable_job_id": "j-1",
    "durable_job": _DURABLE_JOB,
    "cancel_requested": False,
    "identity_provenance": "gateway_verified",
    "gates": {},
    "future_key": {"nested": [1, 2]},
}

ROUND_TRIPS = [
    (
        WorkflowListResponse,
        {"status": "ok", "count": 1, "workflows": [{"name": "relax", "shared": True}]},
    ),
    (
        WorkflowDescribeResponse,
        {
            "status": "ok",
            "contract_version": "1.0.0",
            "workflow_name": "relax",
            "version": "1.0",
            "description": "",
            "scope": "user",
            "user_id": "alice",
            "parameter_schema": {"type": "object", "properties": {}},
            "steps": [{"name": "s1"}],
            "requirements": [],
            "triggers": [],
            "undeclared": "kept",
        },
    ),
    (
        WorkflowTriggerResponse,
        {
            "status": "ok",
            "action": "run-workflow",
            "workflow_name": "relax",
            "source_path": "/w/relax",
            "steps": 2,
            "params": {"steps": 10},
            "execution_id": "wf-1",
            "durable_job_id": None,
            "message": "accepted",
            "undeclared": 3,
        },
    ),
    (WorkflowExecutionResponse, {"status": "ok", "execution": _EXECUTION}),
    (
        WorkflowResultsResponse,
        {
            "status": "ok",
            "execution_id": "wf-1",
            "execution_status": "succeeded",
            "progress": 1.0,
            "error": None,
            "results": {"steps": {"s1": {"energy": -1}}, "final": None, "extra": 1},
            "artifacts": [{"key": "out.json", "size_bytes": 12}],
            "durable_job_id": "j-1",
            "durable_job": _DURABLE_JOB,
            "undeclared": True,
        },
    ),
]


@pytest.mark.parametrize(
    ("model", "payload"), ROUND_TRIPS, ids=[m.__name__ for m, _ in ROUND_TRIPS]
)
def test_response_model_round_trips_payload_unchanged(model, payload):
    dumped = model.model_validate(payload).model_dump(mode="json", exclude_unset=True)
    assert dumped == payload


def test_list_route_serialises_handler_payload_unchanged(client):
    payload = {
        "status": "ok",
        "count": 1,
        "workflows": [{"name": "relax", "scope": "user"}],
        "undeclared": {"k": [1, None]},
    }
    with patch(
        "umrs.api.workflows.handle_workflow_command", return_value=json.dumps(payload)
    ):
        response = client.get("/umrs/workflows", headers=_auth_headers())
    assert response.status_code == 200
    assert response.json() == payload


def test_trigger_route_does_not_add_absent_keys(client, tmp_path):
    _register_strict_workflow(tmp_path)
    payload = {"status": "ok", "execution_id": "wf-9", "legacy_field": "x"}
    with patch(
        "umrs.api.workflows.handle_workflow_command", return_value=json.dumps(payload)
    ):
        response = client.post(
            "/umrs/workflows/relax/trigger",
            headers=_auth_headers(),
            json={"parameters": {"steps": 1}},
        )
    assert response.status_code == 200
    assert response.json() == payload
