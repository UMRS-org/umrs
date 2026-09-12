"""U1 dashboard overview endpoint tests (additive sections)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.web_dashboard import _CONNECTOR_TYPES, router
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
def _clean_registry(monkeypatch):
    # Backend pin: never let a leaked DB_HOST/DATABASE_URL flip the suite
    # onto the PG-backed registry (auto mode). These tests assert the
    # in-memory contract.
    monkeypatch.setenv("UMRS_WORKFLOW_EXECUTION_REGISTRY", "memory")
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


def _create_execution(status: str = "running") -> None:
    registry = get_execution_registry()
    record = registry.create(workflow_name="wf-test", user_id="tester")
    record.update(status=status)


def test_overview_includes_connector_inventory(client):
    resp = client.get("/umrs/web/dashboard/overview")
    assert resp.status_code == 200
    connectors = resp.json()["connectors"]
    # U3: probe-driven partition — every registered type lands in exactly
    # one bucket, matching environment loadability.
    assert set(connectors["available_types"]) | set(connectors["unavailable"]) == set(
        _CONNECTOR_TYPES
    )
    assert set(connectors["available_types"]).isdisjoint(connectors["unavailable"])
    assert connectors["sync_state"] is None
    # Per-connector probe detail is present for the full registry.
    details = connectors["details"]
    assert set(details) == set(_CONNECTOR_TYPES)
    for info in details.values():
        assert {"name", "class", "available"} <= set(info)


def test_overview_workflow_counts_empty_registry(client):
    body = client.get("/umrs/web/dashboard/overview").json()
    workflows = body["workflows"]
    assert workflows == {
        "total": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }


def test_overview_workflow_counts_and_terminal_mapping(client):
    _create_execution("running")
    _create_execution("succeeded")  # terminal success maps to "completed"
    _create_execution("failed")
    _create_execution("cancelled")
    _create_execution("pending")  # non-terminal, non-running → running bucket
    body = client.get("/umrs/web/dashboard/overview").json()
    workflows = body["workflows"]
    assert workflows["total"] == 5
    assert workflows["running"] == 2
    assert workflows["completed"] == 1
    assert workflows["failed"] == 1
    assert workflows["cancelled"] == 1


def test_overview_workflows_null_when_registry_missing(client, monkeypatch):
    import umrs.api.web_dashboard as wd

    registry_cls = get_execution_registry().__class__
    _create_execution("running")
    monkeypatch.setattr(
        f"{registry_cls.__module__}.get_execution_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )
    body = client.get("/umrs/web/dashboard/overview").json()
    assert body["workflows"] is None
    # Connectors remain intact when the workflow registry is down.
    assert set(body["connectors"]["available_types"]) | set(
        body["connectors"]["unavailable"]
    ) == set(_CONNECTOR_TYPES)


def test_overview_response_still_has_original_fields(client):
    body = client.get("/umrs/web/dashboard/overview").json()
    for key in (
        "total_jobs",
        "running_jobs",
        "queued_jobs",
        "completed_jobs",
        "failed_jobs",
        "total_models",
        "available_models",
        "staging_models",
        "deprecated_models",
        "system_health",
    ):
        assert key in body
