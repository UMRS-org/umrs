"""K1.3 (knowledge-sharing plan, 2026-10-24): one listing truth.

``GET /rag/documents`` and ``GET /rag/stats`` answer from the spine through
the graph-readable predicate when the upload facade is central (the
default). Before K1.3 both asked the endpoint corpus, which never held an
upload (WP-15): ``docs list`` was empty and ``stats`` counted 0 after a
successful upload, and ``/stats`` counted the caller's own corpus under any
requested name. On the CI-shape Postgres; skipped without one.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from test_k1_contributions_pg import (  # noqa: F401 — sync_pool/world: fixtures
    _SECRET,
    _Acquired,
    _client,
    _db_url,
    _ingest,
    sync_pool,
    world,
)

pytestmark = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture()
def env(request: Any) -> tuple[Any, Any]:
    """K1.2's ``world`` (a project, alice's personal graph, a shared graph)
    and its Postgres pool."""
    return request.getfixturevalue("world"), request.getfixturevalue("sync_pool")


def _rag_client(user: str, w: Any, pool: Any, monkeypatch: Any, *, admin: bool = False) -> Any:
    """The rag router on the K1.2 test app, as *user* of *w*'s project (or,
    with *admin*, a platform admin whose verified project is another one)."""
    from umrs.api.rag import router
    from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

    from r4_contracts.handshake_hmac import sign_claims

    monkeypatch.delenv("UMRS_RAG_UPLOAD_FACADE", raising=False)
    client = _client(user, w.project, pool, monkeypatch)
    client.app.include_router(router)
    client.app.dependency_overrides[get_registry] = EndpointRegistry
    if admin:
        home = f"admin-home-{uuid.uuid4().hex[:6]}"
        client.headers.update(
            {
                "X-Verified-Project-ID": home,
                "X-Verified-Is-Admin": "true",
                "X-Verified-HMAC": sign_claims(_SECRET, user, home, True),
            }
        )
    return client


def _chunk(project: str, artifact_id: str, index: int) -> None:
    async def _run() -> None:
        async with _Acquired() as conn:
            await conn.execute(
                "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, "
                "content, content_hash, graph_id) VALUES ($1, $2, $3, 'entity', $4, 'h', '')",
                f"node-chunk-{artifact_id}-{index}",
                project,
                f"chunk {index}",
                {"kind": "chunk", "artifact_id": artifact_id, "chunk_index": index},
            )

    asyncio.run(_run())


def _docs(client: Any, **params: Any) -> dict[str, dict[str, Any]]:
    resp = client.get("/umrs/rag/documents", params=params)
    assert resp.status_code == 200, resp.text
    return {d["document_id"]: d for d in resp.json()}


def _stats(client: Any, **params: Any) -> dict[str, Any]:
    resp = client.get("/umrs/rag/stats", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_an_upload_shows_in_docs_list_and_stats_with_its_contributor(env, monkeypatch) -> None:
    w, pool = env
    aid = _ingest(w, user="alice", graph=w.default, content="listing")
    _chunk(w.project, aid, 0)
    _chunk(w.project, aid, 1)
    bob = _rag_client("bob", w, pool, monkeypatch)
    doc = _docs(bob)[aid]
    assert doc["filename"] == "listing.md"
    assert doc["chunks"] == 2
    assert doc["metadata"]["contributors"] == ["alice"]
    assert doc["metadata"]["graphs"] == [w.default]
    stats = _stats(bob)
    assert (stats["documents"], stats["chunks"], stats["source"]) == (1, 2, "spine")


def test_a_personal_item_is_hidden_from_another_member_in_docs_and_stats(env, monkeypatch) -> None:
    w, pool = env
    private = _ingest(w, user="alice", graph=w.personal, content="diary-l")
    _chunk(w.project, private, 0)
    public = _ingest(w, user="alice", graph=w.default, content="paper-l")
    bob = _rag_client("bob", w, pool, monkeypatch)
    assert set(_docs(bob)) == {public}
    assert (_stats(bob)["documents"], _stats(bob)["chunks"]) == (1, 0)
    alice = _rag_client("alice", w, pool, monkeypatch)
    assert set(_docs(alice)) == {public, private}
    assert _stats(alice)["documents"] == 2


def test_one_listing_truth_for_ls_docs_list_and_stats(env, monkeypatch) -> None:
    """``r4 knowledge ls`` (``/k2/artifacts``), ``docs list`` and ``stats``
    agree, and none shows another member's personal item."""
    w, pool = env
    shared = _ingest(w, user="alice", graph=w.shared, content="methods-l")
    _ingest(w, user="alice", graph=w.personal, content="private-l")
    bob = _rag_client("bob", w, pool, monkeypatch)
    listed = bob.get("/umrs/k2/artifacts")
    assert listed.status_code == 200, listed.text
    ls_ids = {a["artifact_id"] for a in listed.json()["artifacts"]}
    assert ls_ids == set(_docs(bob)) == {shared}
    assert _stats(bob)["documents"] == len(ls_ids)


def test_stats_counts_the_requested_project(env, monkeypatch) -> None:
    """``/stats`` dropped ``project_id``: it counted the caller's own corpus."""
    w, pool = env
    _ingest(w, user="alice", graph=w.default, content="other-project")
    admin = _rag_client("root-admin", w, pool, monkeypatch, admin=True)
    assert _stats(admin, project_id=w.project)["documents"] == 1
    assert _stats(admin)["documents"] == 0  # its own (empty) project


def test_the_endpoint_facade_keeps_the_old_proxy(env, monkeypatch) -> None:
    w, pool = env
    _ingest(w, user="alice", graph=w.default, content="proxied")
    bob = _rag_client("bob", w, pool, monkeypatch)
    monkeypatch.setenv("UMRS_RAG_UPLOAD_FACADE", "endpoint")
    # No endpoint registered: the proxy answers, not the spine.
    assert bob.get("/umrs/rag/documents").status_code >= 400


# ── review of K1 part B (2026-10-24) ──────────────────────────────────────


def _graph_chunk(project: str, artifact_id: str, index: int, graph: str) -> None:
    async def _run() -> None:
        async with _Acquired() as conn:
            await conn.execute(
                "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, "
                "content, content_hash, graph_id) VALUES ($1, $2, $3, 'entity', $4, 'h', $5)",
                f"node-chunk-{graph}-{artifact_id}-{index}",
                project,
                f"chunk {index}",
                {"kind": "chunk", "artifact_id": artifact_id, "chunk_index": index},
                graph,
            )

    asyncio.run(_run())


def test_chunks_in_another_members_personal_graph_are_not_counted(env, monkeypatch) -> None:
    """The same bytes in the default graph and alice's personal graph: bob's
    chunk counts were doubled by (and leaked the size of) her copy."""
    w, pool = env
    aid = _ingest(w, user="alice", graph=w.default, content="twice")
    assert _ingest(w, user="alice", graph=w.personal, content="twice") == aid
    _graph_chunk(w.project, aid, 0, w.default)
    _graph_chunk(w.project, aid, 0, w.personal)
    bob = _rag_client("bob", w, pool, monkeypatch)
    assert _docs(bob)[aid]["chunks"] == 1
    assert _stats(bob)["chunks"] == 1
    assert _docs(_rag_client("alice", w, pool, monkeypatch))[aid]["chunks"] == 2


def test_one_document_answers_from_the_spine(env, monkeypatch) -> None:
    """``GET /rag/documents/{id}`` proxied the endpoint corpus: the ids the
    listing gives were 404 there."""
    w, pool = env
    public = _ingest(w, user="alice", graph=w.default, content="one-doc")
    private = _ingest(w, user="alice", graph=w.personal, content="one-private")
    bob = _rag_client("bob", w, pool, monkeypatch)
    resp = bob.get(f"/umrs/rag/documents/{public}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["filename"], body["metadata"]["contributors"]) == ("one-doc.md", ["alice"])
    assert bob.get(f"/umrs/rag/documents/{private}").status_code == 404
    assert bob.get("/umrs/rag/documents/art-none").status_code == 404
    assert _rag_client("alice", w, pool, monkeypatch).get(
        f"/umrs/rag/documents/{private}"
    ).status_code == 200
