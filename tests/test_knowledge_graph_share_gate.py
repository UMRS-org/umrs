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
from umrs.api import _access, knowledge_graphs
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
    # live_memberships lists owned projects too: alice owns proj-a
    table: dict[str, list[str]] = {"alice": ["proj-a"], "bob": ["proj-a"], "carol": ["proj-a"]}

    def fake(request, user, project_id):
        del request
        return project_id in table.get(user, [])

    monkeypatch.setattr(_access, "_live_member", fake, raising=False)
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
            [("alice",), ("carol",)],  # roster the gate checks: owner + member
            _NO_ROW,  # UPDATE
            updated,  # re-read
            [("alice",), ("carol",)],  # roster for effective_readers
        ]
        client, pool = _client(script)
        resp = _put(client, ["circle:proj-a/lab"])
        assert resp.status_code == 200, resp.text
        readers = resp.json()["effective_readers"]
        assert readers["entries"] == [
            {
                "scope": "circle:proj-a/lab",
                "kind": "circle",
                "users": ["alice", "carol"],
                "not_admitted": [],
            }
        ]
        assert readers["tier"] == "live members of project proj-a"
        roster_reads = [sql for sql, _ in pool.record if "FROM circle_members" in sql]
        assert len(roster_reads) == 2  # the gate read the roster before the write

    def test_effective_readers_name_own_project_outsiders_not_admitted(self):
        """KN-S4b: a stored own-project circle that grew an outsider admits
        only project insiders, so effective_readers must not list mallory as
        a reader. A circle of another project keeps its whole roster."""
        stored = ["circle:proj-a/lab", "circle:proj-b/lab"]
        updated = _graph_row(access=stored)
        script = [
            _graph_row(access=stored),  # nothing added: no share-gate reads
            ("proj-a", "lab"),  # the circles exist
            ("proj-b", "lab"),
            _NO_ROW,  # UPDATE
            updated,  # re-read
            [("carol",), ("mallory",)],  # proj-a/lab roster
            [("dave",)],  # proj-b/lab roster
        ]
        client, _ = _client(script)
        resp = _put(client, stored)
        assert resp.status_code == 200, resp.text
        assert resp.json()["effective_readers"]["entries"] == [
            {
                "scope": "circle:proj-a/lab",
                "kind": "circle",
                "users": ["carol"],
                "not_admitted": ["mallory"],
            },
            {"scope": "circle:proj-b/lab", "kind": "circle", "users": ["dave"]},
        ]

    def test_in_project_circle_holding_an_outsider_is_outside(self):
        """KN-S4 review: ``circle:proj-a/mallory`` (a personal circle anyone
        may open) or a named circle a manager grew with an outsider is not
        inside proj-a just because of its prefix."""
        script = [
            _graph_row(),
            ("proj-a", "mallory"),  # the circle exists
            ("alice", "sensitive"),  # project facts
            [("mallory",)],  # roster: not a live member of proj-a
        ]
        client, pool = _client(script)
        message = _refusal(_put(client, ["circle:proj-a/mallory"]))
        assert "(circle:proj-a/mallory)" in message
        assert not any("UPDATE knowledge_graphs" in sql for sql, _ in pool.record)

    def test_in_project_circle_with_unreadable_roster_is_outside(self, monkeypatch):
        """A roster that cannot be read fails closed."""
        monkeypatch.setattr(knowledge_graphs, "_circle_roster", lambda *_: None)
        client, _ = _client([_graph_row(), ("proj-a", "lab"), ("alice", "sensitive")])
        message = _refusal(_put(client, ["circle:proj-a/lab"]))
        assert "(circle:proj-a/lab)" in message

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


def test_one_membership_read_for_both_gates():
    """KN-S4 and KN-S4b share ``_access.user_in_project``: the share gate
    keeps no second copy of the ADR-097 read."""
    assert not hasattr(knowledge_graphs, "_live_member")
    assert knowledge_graphs.user_in_project is _access.user_in_project


def test_unreadable_membership_is_outside(monkeypatch):
    """ADR-097 live memberships unreadable (None) → not a member."""
    from umrs.api import projects

    monkeypatch.setattr(projects, "live_memberships", lambda *_: None)
    request = type("R", (), {})()
    request.app = type("A", (), {})()
    request.app.state = type("S", (), {"umrs_db_pool": _Pool()})()
    assert _access._live_member(request, "carol", "proj-a") is False
    monkeypatch.setattr(projects, "live_memberships", lambda *_: ["proj-a"])
    assert _access._live_member(request, "carol", "proj-a") is True


# ── Real Postgres (CI: UMRS_TEST_DB_URL on a migrated umrs_test) ─────────────
#
# The FakePool tests script every read; these run the gate's own SQL
# (project facts, active-circle roster) against the migrated schema. Rows are
# inserted in the test's transaction, which ``db_connection`` rolls back.
# ``_live_member`` stays the ``members`` fixture: its real path reads the
# user store, covered by ``test_unreadable_membership_is_outside``.


class _OneConnPool:
    def __init__(self, conn):
        self._conn = conn

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        del conn


def _ns(**fields):
    return type("NS", (), fields)()


@pytest.fixture
def pg_request(db_connection):
    """kn-s4-a (sensitive, owner alice): an active circle of members, an
    active circle holding mallory, and a retired circle holding mallory."""
    cur = db_connection.cursor()
    cur.execute(
        "INSERT INTO projects (project_id, owner_id, data_class)"
        " VALUES ('kn-s4-a', 'alice', 'sensitive')"
    )
    cur.execute(
        "INSERT INTO circles (project_id, circle_id, owner_id, status) VALUES"
        " ('kn-s4-a', 'lab', 'alice', 'active'), ('kn-s4-a', 'leak', 'bob', 'active'),"
        " ('kn-s4-a', 'old', 'bob', 'retired')"
    )
    cur.execute(
        "INSERT INTO circle_members (project_id, circle_id, user_id, added_by) VALUES"
        " ('kn-s4-a', 'lab', 'carol', 'alice'), ('kn-s4-a', 'leak', 'mallory', 'bob'),"
        " ('kn-s4-a', 'old', 'mallory', 'bob')"
    )
    return _ns(app=_ns(state=_ns(umrs_db_pool=_OneConnPool(db_connection))))


@pytest.mark.integration
@pytest.mark.usefixtures("members")
class TestRealPostgres:
    def test_facts_and_rosters_read_the_schema(self, pg_request):
        facts = knowledge_graphs._project_share_facts(pg_request, "kn-s4-a")
        assert facts == ("alice", "sensitive")
        assert knowledge_graphs._circle_roster(pg_request, "kn-s4-a/lab") == ["carol"]
        assert knowledge_graphs._circle_roster(pg_request, "kn-s4-a/old") == []

    def test_member_circle_passes_and_outsider_circle_is_refused(self, members, pg_request):
        from fastapi import HTTPException

        members["carol"] = ["kn-s4-a"]
        bob = _ns(user_id="bob", is_admin=False)
        graph = _ns(graph_id="g-pg", project_id="kn-s4-a", access=[])
        knowledge_graphs._enforce_share_gate(pg_request, graph, ["circle:kn-s4-a/lab"], bob)
        knowledge_graphs._enforce_share_gate(pg_request, graph, ["circle:kn-s4-a/old"], bob)
        with pytest.raises(HTTPException) as refused:
            knowledge_graphs._enforce_share_gate(
                pg_request, graph, ["circle:kn-s4-a/leak"], bob
            )
        assert refused.value.status_code == 403
        assert "circle:kn-s4-a/leak" in str(refused.value.detail)
