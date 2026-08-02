"""U6 tests — web dashboard panels + SSE snapshot channels."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.web_dashboard import router
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.state.umrs_db_pool = None
    app.include_router(router)
    return app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


def _seed_executions() -> None:
    registry = get_execution_registry()
    first = registry.create(workflow_name="wf-1", user_id="alice")
    first.update(status="running", started_at=1.0)
    second = registry.create(workflow_name="wf-2", user_id="bob")
    second.update(status="failed", finished_at=2.0)


class TestWorkflowsPanel:
    def test_counts_and_recent(self, client):
        _seed_executions()
        resp = client.get("/umrs/web/dashboard/workflows")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["counts"] == {"running": 1, "failed": 1}
        assert len(payload["recent"]) == 2
        names = {r["workflow_name"] for r in payload["recent"]}
        assert names == {"wf-1", "wf-2"}
        assert payload["degraded"] is False

    def test_empty_registry_yields_zero_counts(self, client):
        resp = client.get("/umrs/web/dashboard/workflows")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["counts"] == {}
        assert payload["recent"] == []

    def test_degraded_when_registry_raises(self, client, monkeypatch):
        def _boom():
            raise RuntimeError("registry down")

        monkeypatch.setattr("umrs.server.workflow_execution_registry.get_execution_registry", _boom)
        resp = client.get("/umrs/web/dashboard/workflows")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["degraded"] is True
        assert payload["counts"] == {}


class TestConnectorsPanel:
    def test_seeded_rows(self, client, monkeypatch):
        from umrs.connectors.sync_state import InMemorySyncStateStore, SyncStateRecord

        store = InMemorySyncStateStore()
        store.put(
            SyncStateRecord(
                name="gdrive",
                connector_type="gdrive",
                last_status="ok",
                last_sync_at="2026-08-02T00:00",
                docs_synced=12,
                errors_count=1,
                dlq=[{"doc": "a.txt"}],
            )
        )
        monkeypatch.setattr("umrs.connectors.sync_state.build_default_store", lambda: store)
        resp = client.get("/umrs/web/dashboard/connectors")
        assert resp.status_code == 200
        payload = resp.json()
        row = payload["connectors"]["gdrive"]
        assert row == {
            "status": "ok",
            "last_sync_at": "2026-08-02T00:00",
            "docs_synced": 12,
            "errors": 1,
            "dlq": 1,
        }

    def test_null_safe_when_store_fails(self, client, monkeypatch):
        def _boom():
            raise RuntimeError("store down")

        monkeypatch.setattr("umrs.connectors.sync_state.build_default_store", _boom)
        resp = client.get("/umrs/web/dashboard/connectors")
        payload = resp.json()
        assert resp.status_code == 200
        assert payload["connectors"] is None
        assert payload["degraded"] is True


class _FakeStore:
    def __init__(self) -> None:
        self.artifacts = [type("A", (), {"status": "active"})() for _ in range(2)]

    async def list_artifacts(self, **_kwargs):
        return self.artifacts, 5

    async def list_irs(self, **_kwargs):
        return [1, 2, 3]

    async def list_nodes(self, **_kwargs):
        return [1, 2]

    async def list_edges(self, **_kwargs):
        return [1]


class _FakeVerification:
    async def list_reports(self, **_kwargs):
        return [type("R", (), {"confidence": 0.9})()]


class TestKnowledgePanel:
    async def test_payload_aggregates(self):
        from umrs.api.web_dashboard import _knowledge_panel_payload

        result = await _knowledge_panel_payload("proj-a", _FakeStore(), _FakeVerification())
        assert result.project_id == "proj-a"
        assert result.artifact_count == 5
        assert result.active_artifact_count == 2
        assert result.document_count == 3
        assert result.node_count == 2
        assert result.edge_count == 1
        assert result.last_confidence == 0.9

    def test_route_503_degraded_without_pool(self, client):
        resp = client.get("/umrs/web/dashboard/knowledge", params={"project_id": "proj-a"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["degraded"] is True
        assert "database unavailable" in payload["degraded_reason"]

    def test_route_requires_project(self, client):
        resp = client.get("/umrs/web/dashboard/knowledge")
        assert resp.status_code == 422


class TestSseChannels:
    async def test_snapshot_stream_first_frame(self):
        from umrs.api.web_dashboard import _sse_snapshot_stream

        async def collect():
            frames: list[str] = []
            async for frame in _sse_snapshot_stream(
                snapshot_fn=lambda: {"ready": True, "n": 2},
                channel="workflows",
                interval=0.01,
            ):
                frames.append(frame)
                if len(frames) >= 2:
                    break
            return frames

        frames = await collect()
        header, data = frames[0].split("\n", 1)
        assert header.startswith("event: workflows-snapshot")
        assert json.loads(data.removeprefix("data: ").strip()) == {"ready": True, "n": 2}
        assert frames[1].startswith(": keepalive")

    def test_routes_registered(self, client):
        # Streams hang forever under TestClient HTTP, so route presence is
        # verified via the OpenAPI schema instead of consuming the stream.
        schema = client.get("/openapi.json").json()
        assert "/umrs/web/dashboard/sse/workflows" in schema["paths"]
        assert "/umrs/web/dashboard/sse/intake" in schema["paths"]
