"""KN-S4b (ADR-118 D-8, read side): a circle of the graph's own project
admits only a reader inside that project.

A circle manager may add any user to a circle. Until 2026-11-24 a graph of
``proj-a`` shared with ``circle:proj-a/lab`` then admitted that outsider at
read time: ``graph_read_allowed`` fell back to the bare access list with no
membership check, past the KN-S4 write-time gate (which only sees the
roster when the share is made). Now such an entry admits a reader only when
the reader is verified into the project, a live member (its owner included,
ADR-097) or an admin. Other entries (``user:``, ``project:``, ``system``,
circles of another project) are unchanged.
"""

from __future__ import annotations

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import _access
from umrs.api.knowledge_graphs import router as graphs_router
from umrs.api.knowledge_proposals import router as proposals_router
from umrs.shared.access_scopes import ResourceCaller

_GRAPH_URL = "/umrs/knowledge/graphs/g-abc"
_LAB = "circle:proj-a/lab"
_IN_LAB = [("proj-a/lab",)]  # the caller's active circles (lazy lookup)


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
        "access": [_LAB],
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _client(script, router=graphs_router):
    app = FastAPI()
    app.include_router(router)
    pool = FakePool(script)
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    headers = {"x-user-id": user, "x-verified-project-id": project}
    if admin:
        headers["x-verified-is-admin"] = "true"
    return headers


@pytest.fixture
def members(monkeypatch):
    """Live project members (ADR-097): user → projects; every ask recorded."""
    table: dict[str, list[str]] = {"alice": ["proj-a"], "carol": ["proj-a", "proj-b"]}
    asked: list[tuple[str, str]] = []

    def fake(raw_request, user, project_id):
        del raw_request
        asked.append((user, project_id))
        return project_id in table.get(user, [])

    monkeypatch.setattr(_access, "_live_member", fake, raising=False)
    return asked


@pytest.mark.usefixtures("members")
class TestGraphRead:
    def test_outsider_added_to_the_project_circle_is_refused(self):
        """mallory sits on proj-a/lab's roster but is no member of proj-a."""
        client, _ = _client([_graph_row(), _IN_LAB, []])
        resp = client.get(_GRAPH_URL, headers=_headers("mallory", "proj-x"))
        assert resp.status_code == 404, resp.text

    def test_live_member_acting_in_another_project_still_reads(self):
        client, _ = _client([_graph_row(), _IN_LAB, []])
        resp = client.get(_GRAPH_URL, headers=_headers("carol", "proj-b"))
        assert resp.status_code == 200, resp.text

    def test_outsider_is_refused_the_access_list_and_content_too(self):
        client, _ = _client([_graph_row(), _IN_LAB])
        resp = client.get(f"{_GRAPH_URL}/access", headers=_headers("mallory", "proj-x"))
        assert resp.status_code == 404, resp.text

    def test_circle_of_another_project_is_unchanged(self, members):
        """An outside share the owner made (KN-S4 gate) keeps its audience."""
        row = _graph_row(access=["circle:proj-b/lab"])
        client, _ = _client([row, [("proj-b/lab",)], []])
        resp = client.get(_GRAPH_URL, headers=_headers("mallory", "proj-x"))
        assert resp.status_code == 200, resp.text
        assert members == []  # no membership asked for a foreign circle

    def test_named_user_entry_is_unchanged(self, members):
        row = _graph_row(access=[_LAB, "user:mallory"])
        client, pool = _client([row, []])
        resp = client.get(_GRAPH_URL, headers=_headers("mallory", "proj-x"))
        assert resp.status_code == 200, resp.text
        assert members == []
        assert not any("FROM circle_members" in sql for sql, _ in pool.record)

    def test_verified_project_member_reads_on_the_tier_without_lookups(self, members):
        """Stand-alone single owner: the tier ladder decides, nothing else."""
        client, pool = _client([_graph_row(), []])
        resp = client.get(_GRAPH_URL, headers=_headers("alice", "proj-a"))
        assert resp.status_code == 200, resp.text
        assert members == []
        assert not any("FROM circle_members" in sql for sql, _ in pool.record)


class TestListAndStats:
    @pytest.mark.usefixtures("members")
    def test_list_hides_the_graph_from_the_outsider(self):
        client, _ = _client([[], [_graph_row()], _IN_LAB])
        resp = client.get("/umrs/knowledge/graphs", headers=_headers("mallory", "proj-x"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["graphs"] == []

    def test_list_asks_membership_once_per_request(self, members):
        rows = [_graph_row(), _graph_row(graph_id="g-def")]
        # the list builds one caller per shared row: one circle lookup each
        client, _ = _client([[], rows, list(_IN_LAB), list(_IN_LAB)])
        resp = client.get("/umrs/knowledge/graphs", headers=_headers("carol", "proj-b"))
        assert [g["graph_id"] for g in resp.json()["graphs"]] == ["g-abc", "g-def"]
        assert members == [("carol", "proj-a")]

    @pytest.mark.usefixtures("members")
    def test_proposal_stats_do_not_count_the_graph_for_the_outsider(self):
        row = {
            "target_graph_id": "g-abc",
            "accepted": 5,
            "declined": 3,
            "pending": 1,
            "kind": "research",
            "project_id": "proj-a",
            "owner_id": "alice",
            "access": [_LAB],
        }
        client, _ = _client([[row], _IN_LAB], router=proposals_router)
        resp = client.get(
            "/umrs/knowledge/proposals/stats", headers=_headers("mallory", "proj-x")
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["decided"] == 0


class _Request:
    """A request carrying only an app pool and a state (the helper's needs)."""

    def __init__(self, pool) -> None:
        if pool is not None:
            pool.conn.rollback = lambda: None  # psycopg2 parity
        self.app = type("App", (), {"state": type("S", (), {"umrs_db_pool": pool})()})()
        self.state = type("State", (), {})()


def _caller(user="mallory", project="proj-x", admin=False) -> ResourceCaller:
    return ResourceCaller(user_id=user, project_id=project, is_admin=admin)


class TestProjectInsider:
    def test_owner_through_live_memberships_is_inside(self, monkeypatch):
        """``live_memberships`` lists owned projects: the owner is inside."""
        from umrs.api import projects

        monkeypatch.setattr(projects, "live_memberships", lambda _store, _user: ["proj-a"])
        request = _Request(FakePool([]))
        assert _access.project_insider(request, _caller("alice"), "proj-a")

    def test_unreadable_membership_is_outside(self, monkeypatch):
        from umrs.api import projects

        def unreadable(_store, _user):
            raise RuntimeError("users row unreadable")

        monkeypatch.setattr(projects, "live_memberships", unreadable)
        assert not _access.project_insider(_Request(FakePool([])), _caller(), "proj-a")

    def test_none_membership_is_outside(self, monkeypatch):
        from umrs.api import projects

        monkeypatch.setattr(projects, "live_memberships", lambda _store, _user: None)
        assert not _access.project_insider(_Request(FakePool([])), _caller(), "proj-a")

    def test_no_pool_is_outside(self):
        assert not _access.project_insider(_Request(None), _caller(), "proj-a")

    def test_admin_and_verified_caller_are_inside_without_a_lookup(self):
        request = _Request(None)
        assert _access.project_insider(request, _caller(admin=True), "proj-a")
        assert _access.project_insider(request, _caller(project="proj-a"), "proj-a")
