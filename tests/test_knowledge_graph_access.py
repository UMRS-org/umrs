"""Knowledge graph access scopes (ADR-080).

A graph row carries ``access``: several user:/project:/circle:/system
entries; any match admits READS beyond the tier ladder; lifecycle writes
stay on the ladder. Absent list = the historical tier behavior exactly.
"""

from __future__ import annotations

import json

from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.knowledge_graphs import router as graphs_router
from umrs.knowledge.graph_registry import GraphRegistryStore, _row_to_graph

_IDENT = {"caller_user_id": "alice", "caller_project_id": "proj-a"}


def _graph_row(**overrides):
    row = {
        "graph_id": "g-abc",
        "name": "experiment-history",
        "description": "History",
        "owner_id": "alice",
        "project_id": "proj-a",
        "kind": "research",
        "status": "active",
        "ontology_id": None,
        "ontology_version": None,
        "provenance": {},
        "access": [],
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _client(script):
    app = FastAPI()
    app.include_router(graphs_router)
    pool = FakePool(script)
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    headers = {"x-user-id": user, "x-verified-project-id": project}
    if admin:
        headers["x-verified-is-admin"] = "true"
    return headers


# ── registry store ─────────────────────────────────────────────────────────


def test_row_to_graph_normalizes_access():
    meta = _row_to_graph(_graph_row(access=json.dumps(["user:bob", "circle:proj-a/lab"])))
    assert meta.access == ["user:bob", "circle:proj-a/lab"]
    assert meta.to_dict()["access"] == ["user:bob", "circle:proj-a/lab"]
    legacy = _row_to_graph(_graph_row())
    legacy_dict = {k: v for k, v in _graph_row().items() if k != "access"}
    assert _row_to_graph(legacy_dict).access == []
    assert legacy.access == []


def test_set_access_sql_rounds_through():
    pool = FakePool([None, _graph_row(access=["user:bob"])])
    store = GraphRegistryStore(pool)
    meta = store.set_access("g-abc", ["user:bob"])
    sql, params = pool.record[0]
    assert "UPDATE knowledge_graphs SET access" in sql
    assert params == (json.dumps(["user:bob"]), "g-abc")
    assert meta is not None and meta.access == ["user:bob"]


def test_list_shared_graphs_only_returns_access_carrying_rows():
    pool = FakePool([[_graph_row(access=["system"])]])
    store = GraphRegistryStore(pool)
    shared = store.list_shared_graphs()
    sql, _ = pool.record[0]
    assert "access != '[]'::jsonb" in sql
    assert [m.access for m in shared] == [["system"]]


# ── routes: reads beyond the tier ladder ───────────────────────────────────


class TestSharedGraphReads:
    def test_user_scope_opens_the_graph_to_the_named_user(self):
        shared = _graph_row(access=["user:bob"])
        client, _ = _client([shared, []])  # resolve + ontology block
        resp = client.get("/umrs/knowledge/graphs/g-abc", headers=_headers("bob", "proj-b"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["access"] == ["user:bob"]

    def test_unrelated_caller_gets_404_no_disclosure(self):
        client, _ = _client([_graph_row(access=["user:bob"])])
        resp = client.get("/umrs/knowledge/graphs/g-abc", headers=_headers("heidi", "proj-b"))
        assert resp.status_code == 404

    def test_circle_scope_via_active_membership(self):
        shared = _graph_row(access=["circle:proj-a/lab"])
        # resolve; then the lazy circles lookup admits; then the ontology block
        client, pool = _client([shared, [("proj-a/lab",)], []])
        resp = client.get("/umrs/knowledge/graphs/g-abc", headers=_headers("carol", "proj-b"))
        assert resp.status_code == 200, resp.text
        assert any("FROM circle_members" in sql for sql, _ in pool.record)

    def test_list_includes_shared_graphs_for_admitted_callers(self):
        shared = _graph_row(access=["user:bob"])
        # unscoped list: own-project rows (none), then the shared candidates
        client, _ = _client([[], [shared]])
        resp = client.get("/umrs/knowledge/graphs", headers=_headers("bob", "proj-b"))
        assert resp.status_code == 200, resp.text
        assert [g["graph_id"] for g in resp.json()["graphs"]] == ["g-abc"]

    def test_list_hides_shared_graphs_from_others(self):
        shared = _graph_row(access=["user:bob"])
        client, _ = _client([[], [shared]])
        resp = client.get("/umrs/knowledge/graphs", headers=_headers("heidi", "proj-b"))
        assert resp.json()["graphs"] == []


# ── routes: setting the list ───────────────────────────────────────────────


class TestSetGraphAccess:
    def test_owner_replaces_the_list(self):
        rows = _graph_row()
        updated = _graph_row(access=["user:bob", "circle:proj-a/lab"])
        # resolve, the circle exists (find_active_circle), set_access update, re-read
        client, pool = _client([rows, ("proj-a", "lab"), None, updated])
        resp = client.put(
            "/umrs/knowledge/graphs/g-abc/access",
            json={"access": ["user:bob", "circle:proj-a/lab"]},
            params=_IDENT,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["access"] == ["user:bob", "circle:proj-a/lab"]
        assert any("UPDATE knowledge_graphs SET access" in sql for sql, _ in pool.record)

    def test_foreign_caller_cannot_set(self):
        client, _ = _client([_graph_row()])
        resp = client.put(
            "/umrs/knowledge/graphs/g-abc/access",
            json={"access": ["user:bob"]},
            params={"caller_user_id": "bob", "caller_project_id": "proj-b"},
        )
        assert resp.status_code == 404  # not writable → not disclosed

    def test_system_entry_is_admin_only(self):
        client, _ = _client([_graph_row()])
        resp = client.put(
            "/umrs/knowledge/graphs/g-abc/access",
            json={"access": ["system"]},
            params=_IDENT,
        )
        assert resp.status_code == 403

    def test_admin_may_set_system(self):
        rows = _graph_row()
        updated = _graph_row(access=["system"])
        client, _ = _client([rows, None, updated])
        resp = client.put(
            "/umrs/knowledge/graphs/g-abc/access",
            json={"access": ["system"]},
            params={**_IDENT, "is_admin": True},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["access"] == ["system"]

    def test_bad_scope_is_a_typed_403_or_literal_rejected(self):
        client, _ = _client([_graph_row()])
        resp = client.put(
            "/umrs/knowledge/graphs/g-abc/access",
            json={"access": ["garbage"]},
            params=_IDENT,
        )
        assert resp.status_code == 403

    def test_get_access_route_shows_the_list_to_readers(self):
        shared = _graph_row(access=["user:bob"])
        client, _ = _client([shared])
        resp = client.get("/umrs/knowledge/graphs/g-abc/access", headers=_headers("bob", "proj-b"))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"graph_id": "g-abc", "access": ["user:bob"]}
