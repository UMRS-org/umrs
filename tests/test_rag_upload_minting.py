"""ADR-027 P1a: knowledge node minting on rag uploads.

Service returns chunk_details; proxy mints canonical document/chunk nodes
(fail-soft). Upload success must never depend on knowledge minting.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import rag as rag_module
from umrs.api.rag import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry


def _create_client(project_id: str = "test-project") -> TestClient:
    registry = EndpointRegistry()
    registry.register(
        project_id=project_id,
        model="m",
        job_id="job-1",
        host="compute-01",
        port=11434,
        backend="slurm",
    )
    registry.mark_ready("job-1")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


async def _fake_post_ok(self, url, **kwargs):
    return httpx.Response(
        status_code=200,
        json={
            "document_id": "doc-abc",
            "filename": "notes.txt",
            "status": "indexed",
            "chunks": 2,
            "chunk_details": [
                {"s3_object": "uploads/notes.txt#chunk0", "excerpt": "alpha"},
                {"s3_object": "uploads/notes.txt#chunk1", "excerpt": "beta"},
            ],
        },
        headers={"content-type": "application/json"},
    )


def _upload_regression_shims(monkeypatch):
    """Keep the upload test hermetic: governance/audit side channels off."""
    monkeypatch.setattr("umrs.api.governance.check_quota", lambda *a, **k: None)
    monkeypatch.setattr("umrs.api.governance.record_resource_created", lambda *a, **k: None)
    monkeypatch.setattr("umrs.api.audit_events.emit_audit", lambda *a, **k: None)


def test_upload_mints_knowledge_nodes(monkeypatch):
    _upload_regression_shims(monkeypatch)
    client = _create_client()
    captured: dict = {}

    async def fake_mint(request, **kwargs):
        captured.update(kwargs)
        return 2

    monkeypatch.setattr(rag_module, "_mint_knowledge_nodes_best_effort", fake_mint)
    with patch.object(httpx.AsyncClient, "post", _fake_post_ok):
        response = client.post(
            "/umrs/rag/upload",
            files={"file": ("notes.txt", b"alpha beta", "text/plain")},
            headers={"x-user-id": "u1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-abc"
    assert body["knowledge_nodes_minted"] == 2
    assert captured["document_id"] == "doc-abc"
    assert captured["project_id"] == "test-project"
    assert len(captured["chunk_details"]) == 2


def test_upload_survives_mint_failure(monkeypatch):
    _upload_regression_shims(monkeypatch)
    client = _create_client()

    async def boom(request, **kwargs):
        raise RuntimeError("pool down")

    # Wrapper returns 0 on inner failure; patch the inner function itself.
    monkeypatch.setattr(rag_module, "_mint_knowledge_nodes", boom)
    with patch.object(httpx.AsyncClient, "post", _fake_post_ok):
        response = client.post(
            "/umrs/rag/upload",
            files={"file": ("notes.txt", b"alpha beta", "text/plain")},
            headers={"x-user-id": "u1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    assert response.json()["knowledge_nodes_minted"] == 0


class _FakeStore:
    def __init__(self, pool):
        self.artifacts: list = []
        self.nodes: list = []
        self.edges: list = []

    async def insert_artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))
        return True

    async def upsert_node(self, node):
        self.nodes.append(node)

    async def upsert_edge(self, edge):
        self.edges.append(edge)


def _mk_chunk_details(n: int, name: str = "big.txt") -> list[dict]:
    return [
        {
            "s3_object": f"uploads/{name}#chunk{i}",
            "excerpt": f"chunk {i}",
            "chunk_md5": f"m{i}",
        }
        for i in range(n)
    ]


def _assert_minted_graph(store) -> None:
    """Shape assertions for a 1-doc + 100-chunk mint result."""
    _assert_doc_topology(store)
    _assert_edges_and_graph(store)


def _assert_doc_topology(store) -> None:
    assert len(store.artifacts) == 1
    assert len(store.nodes) == 101
    _assert_doc_nodes(store)
    _assert_chunk_nodes(store)


def _assert_doc_nodes(store) -> None:
    doc_nodes = [n for n in store.nodes if n.node_type == "file"]
    assert [n.node_id for n in doc_nodes] == ["ragdoc-doc-big"]
    assert doc_nodes[0].content["document_id"] == "doc-big"


def _assert_chunk_nodes(store) -> None:
    chunk_nodes = [n for n in store.nodes if n.node_type == "entity"]
    assert chunk_nodes[0].content["document_id"] == "doc-big"
    assert chunk_nodes[99].content["chunk_index"] == 99


def _assert_edges_and_graph(store) -> None:
    from umrs.knowledge.graph_registry import default_graph_id

    assert len(store.edges) == 100
    assert _part_of_to_doc(store.edges, "ragdoc-doc-big")
    assert _same_default_graph(store.nodes, "test-project", default_graph_id)


def _part_of_to_doc(edges, doc_id: str) -> bool:
    part_of = all(e.relation_type == "part_of" for e in edges)
    targets = all(e.target_id == doc_id for e in edges)
    return part_of and targets


def _same_default_graph(nodes, project: str, default_graph_id) -> bool:
    return all(n.graph_id == default_graph_id(project) for n in nodes)


async def test_mint_caps_at_100_chunks(monkeypatch):
    store_holder: dict = {}

    class FactoryStore(_FakeStore):
        def __init__(self, pool):
            super().__init__(pool)
            store_holder["store"] = self

    async def fake_pool(request):
        return object()

    monkeypatch.setattr("umrs.api._asyncpg_pool.knowledge_asyncpg_pool", fake_pool)
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", FactoryStore)

    minted = await rag_module._mint_knowledge_nodes(
        request=None,
        user_id="u1",
        project_id="test-project",
        document_id="doc-big",
        filename="big.txt",
        mime_type="text/plain",
        chunk_details=_mk_chunk_details(150),
    )

    assert minted == 100
    _assert_minted_graph(store_holder["store"])
