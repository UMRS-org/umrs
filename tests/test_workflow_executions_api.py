"""Tests for the new workflow execution status / cancel HTTP routes."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.workflows import router
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _registry():
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


def test_executions_status_404_for_unknown(client):
    response = client.get("/umrs/workflows/executions/does-not-exist")
    assert response.status_code == 404


def test_executions_status_returns_record(client):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"image": "x.png"},
    )

    response = client.get(f"/umrs/workflows/executions/{record.execution_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["execution"]["execution_id"] == record.execution_id
    assert body["execution"]["status"] == "submitted"


def test_executions_results_includes_artifacts(client):
    registry = get_execution_registry()
    record = registry.create(workflow_name="wf", user_id="alice")
    # Pretend a step produced an artifact.
    record.results.setdefault("blobs", {})["wf-runner/foo.json"] = b"{}"
    record.add_artifact("wf-runner/foo.json")
    record.update(status="succeeded", progress=1.0)

    response = client.get(
        f"/umrs/workflows/executions/{record.execution_id}/results"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["execution_status"] == "succeeded"
    assert body["artifacts"][0]["key"] == "wf-runner/foo.json"
    assert body["artifacts"][0]["size_bytes"] == 2


def test_cancel_marks_submitted_record_as_cancelled(client):
    registry = get_execution_registry()
    record = registry.create(workflow_name="wf", user_id="alice")
    response = client.post(
        f"/umrs/workflows/executions/{record.execution_id}/cancel"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["execution"]["status"] == "cancelled"


def test_cancel_unknown_returns_404(client):
    response = client.post("/umrs/workflows/executions/does-not-exist/cancel")
    assert response.status_code == 404
