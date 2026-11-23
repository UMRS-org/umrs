"""ADR-118 D-8 (K-14, slice KN-S4): the outside-project share gate.

``PUT /umrs/knowledge/graphs/{id}/access`` let any graph writer add an
audience outside the graph's project. Now an ADDED outside entry needs the
project owner or an admin AND a ``non_sensitive`` project; in-project
audiences stay open to every writer, and the response names the effective
readers (circle rosters included).
"""

from __future__ import annotations

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import knowledge_graphs
from umrs.api.knowledge_graphs import router as graphs_router

_URL = "/umrs/knowledge/graphs/g-abc/access"
_NO_ROW = None  # a script item that reads as "no row"


def _graph_row(**overrides):
    row = {
        "graph_id": "g-abc",
        "name": "experiment-history",
        "description": "History",
        "owner_id": "bob",
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


def _put(client, access, user="bob", admin=False):
    params = {"caller_user_id": user, "caller_project_id": "proj-a"}
    if admin:
        params["is_admin"] = True
    return client.put(_URL, json={"access": access}, params=params)


@pytest.fixture
def members(monkeypatch):
    """Live project members (ADR-097) the gate sees: user → projects."""
    table: dict[str, list[str]] = {"bob": ["proj-a"], "carol": ["proj-a"]}

    def fake(request, user, project_id):
        del request
        return project_id in table.get(user, [])

    monkeypatch.setattr(knowledge_graphs, "_live_member", fake, raising=False)
    return table


def _refusal(resp):
    assert resp.status_code == 403, resp.text
    assert resp.headers["x-platform-error-code"] == "knowledgegraphs.forbidden"
    return resp.json()["detail"]


@pytest.mark.usefixtures("members")
class TestNonOwnerWriter:
    """bob writes proj-a's graph; alice owns proj-a."""

    def test_outside_project_share_is_refused(self):
        client, pool = _client([_graph_row(), ("alice", "non_sensitive")])
        message = _refusal(_put(client, ["project:proj-b"]))
        assert "limited to the project owner or an admin" in message
        assert "(project:proj-b)" in message
        assert not any("UPDATE knowledge_graphs" in sql for sql, _ in pool.record)

    def test_foreign_circle_and_non_member_user_are_outside(self):
        client, _ = _client([_graph_row(), ("proj-b", "lab"), ("alice", "non_sensitive")])
        message = _refusal(_put(client, ["circle:proj-b/lab", "user:mallory"]))
        assert "(circle:proj-b/lab, user:mallory)" in message

    def test_in_project_circle_share_still_works(self):
        updated = _graph_row(access=["circle:proj-a/lab"])
        script = [
            _graph_row(),
            ("proj-a", "lab"),  # the circle exists
            ("alice", "sensitive"),  # project facts
            _NO_ROW,  # UPDATE
            updated,  # re-read
            [("carol",), ("dave",)],  # roster
        ]
        client, _ = _client(script)
        resp = _put(client, ["circle:proj-a/lab"])
        assert resp.status_code == 200, resp.text
        readers = resp.json()["effective_readers"]
        assert readers["entries"] == [
            {"scope": "circle:proj-a/lab", "kind": "circle", "users": ["carol", "dave"]}
        ]
        assert readers["tier"] == "live members of project proj-a"

    def test_member_user_and_own_project_are_inside(self):
        updated = _graph_row(access=["user:carol", "project:proj-a"])
        client, _ = _client([_graph_row(), ("alice", "sensitive"), _NO_ROW, updated])
        resp = _put(client, ["user:carol", "project:proj-a"])
        assert resp.status_code == 200, resp.text

    def test_kept_outside_entries_are_not_added(self):
        stored = ["project:proj-b", "user:carol"]
        updated = _graph_row(access=["project:proj-b"])
        # nothing is added: no facts read, the writer may narrow
        client, pool = _client([_graph_row(access=stored), _NO_ROW, updated])
        resp = _put(client, ["project:proj-b"])
        assert resp.status_code == 200, resp.text
        assert not any("FROM projects" in sql for sql, _ in pool.record)


@pytest.mark.usefixtures("members")
class TestDataClass:
    def test_owner_of_a_sensitive_project_is_refused_by_name(self):
        client, _ = _client([_graph_row(), ("bob", "sensitive")])
        message = _refusal(_put(client, ["project:proj-b"]))
        assert "project 'proj-a' has data_class 'sensitive'" in message

    def test_admin_is_refused_for_a_sensitive_project(self):
        client, _ = _client([_graph_row(), ("alice", "sensitive")])
        message = _refusal(_put(client, ["system"], user="root", admin=True))
        assert "data_class 'sensitive'" in message

    def test_unknown_project_row_fails_closed(self):
        client, _ = _client([_graph_row(), _NO_ROW])
        message = _refusal(_put(client, ["project:proj-b"], user="root", admin=True))
        assert "data_class 'unknown'" in message

    def test_admin_may_share_a_non_sensitive_project(self):
        updated = _graph_row(access=["system"])
        client, _ = _client([_graph_row(), ("alice", "non_sensitive"), _NO_ROW, updated])
        resp = _put(client, ["system"], user="root", admin=True)
        assert resp.status_code == 200, resp.text
        assert resp.json()["effective_readers"]["entries"] == [
            {"scope": "system", "kind": "system", "audience": "every authenticated caller"}
        ]


@pytest.mark.usefixtures("members")
class TestStandalone:
    def test_single_owner_can_share(self):
        """P1: one human owns the project; the stack's class is non_sensitive."""
        access = ["project:proj-b", "user:friend"]
        updated = _graph_row(owner_id="scientist", access=access)
        script = [
            _graph_row(owner_id="scientist"),
            ("scientist", "non_sensitive"),
            _NO_ROW,
            updated,
        ]
        client, _ = _client(script)
        resp = _put(client, access, user="scientist")
        assert resp.status_code == 200, resp.text
        assert resp.json()["access"] == access


@pytest.mark.usefixtures("members")
def test_personal_graph_keeps_its_owner_ladder():
    """A project-less (personal) graph has no project to leave: unchanged."""
    personal = _graph_row(kind="user", project_id=None, owner_id="bob")
    updated = _graph_row(kind="user", project_id=None, access=["user:x"])
    client, pool = _client([personal, _NO_ROW, updated])
    resp = _put(client, ["user:x"])
    assert resp.status_code == 200, resp.text
    assert not any("FROM projects" in sql for sql, _ in pool.record)
    assert resp.json()["effective_readers"]["tier"] == "its owner bob"


class _RollbackConn:
    def rollback(self):
        return None


class _Pool:
    def getconn(self):
        return _RollbackConn()

    def putconn(self, conn):
        del conn


def test_unreadable_membership_is_outside(monkeypatch):
    """ADR-097 live memberships unreadable (None) → not a member."""
    from umrs.api import projects

    monkeypatch.setattr(projects, "live_memberships", lambda *_: None)
    request = type("R", (), {})()
    request.app = type("A", (), {})()
    request.app.state = type("S", (), {"umrs_db_pool": _Pool()})()
    assert knowledge_graphs._live_member(request, "carol", "proj-a") is False
    monkeypatch.setattr(projects, "live_memberships", lambda *_: ["proj-a"])
    assert knowledge_graphs._live_member(request, "carol", "proj-a") is True
