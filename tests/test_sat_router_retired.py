"""R5 tail (ADR-026): every /umrs/knowledge-graph/* route goes 410."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.knowledge_graph import router

_HEADERS = {"X-User-ID": "u1", "X-Project-ID": "proj-a"}


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def test_get_node_returns_410():
    client = _client()
    response = client.get("/umrs/knowledge-graph/nodes/n-1", headers=_HEADERS)
    assert response.status_code == 410


def test_post_node_returns_410_with_guidance():
    client = _client()
    response = client.post(
        "/umrs/knowledge-graph/nodes",
        json={"node_id": "n-1", "node_type": "concept"},
        headers=_HEADERS,
    )
    assert response.status_code == 410
    detail = response.json()["detail"].lower()
    assert "retired" in detail
    assert "canonical" in detail or "multigraph" in detail


def test_error_detail_points_at_multigraph():
    response = _client().post("/umrs/knowledge-graph/edges", json={}, headers=_HEADERS)
    assert "/umrs/knowledge/graph" in response.json()["detail"]
