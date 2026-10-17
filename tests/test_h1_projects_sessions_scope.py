"""H-1 item 5 + Appendix A.2 (hardening plan, 2026-10-17): projects, circles,
research sessions and companion turns/approvals.

- projects ``GET /{project_id}`` and ``/budget`` took no identity although
  annotated ``# auth: verified``: any caller read any project's row and caps.
  Now owner, member (verified into the project or affiliated through a
  circle) or admin; anyone else reads "not found".
- circles ``GET /{project_id}/{circle_id}`` and ``/members`` took no identity:
  same rule as the project circle listing (admin, project owner, affiliated
  caller) plus the circle's own owner and members.
- research sessions: ``POST /{sid}/loops`` linked any ``loop_id`` (another
  user's or another project's loop) into the caller's session.
- companion: ``GET /turns/{attempt_id}`` and ``/events`` returned turn content
  without an owner check; ownerless approvals were listed to and decidable by
  anyone. Review 2026-10-18: an ownerless intent on an OWNED loop is the loop
  owner's, and a turn cannot begin on an ownerless worker (it could never be
  read back).
"""

from __future__ import annotations

from typing import Any

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-projects-sessions-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


def _client(router, pool: Any) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool
    return TestClient(app)


# ── projects ─────────────────────────────────────────────────────────────────


class _ProjectsConn:
    """projects + project_budgets + circle affiliation, by SQL shape."""

    def __init__(self) -> None:
        self.projects: dict[str, tuple] = {}
        self.budgets: dict[str, tuple] = {}
        self.affiliations: set[tuple[str, str]] = set()

    def cursor(self) -> _ProjectsCursor:
        return _ProjectsCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def add(self, pid: str, owner: str) -> None:
        self.projects[pid] = (pid, pid, owner, "active", None, None, None, "sensitive", "", "[]")
        self.budgets[pid] = (pid, 8.0, 3.0, None)


class _ProjectsCursor:
    def __init__(self, conn: _ProjectsConn) -> None:
        self._conn = conn
        self._rows: list = []

    def execute(self, sql: str, params: Any = None) -> None:
        if "FROM circle_members cm" in sql:
            hit = (params[0], params[1]) in self._conn.affiliations
            self._rows = [(1,)] if hit else []
        elif "project_budgets" in sql:
            row = self._conn.budgets.get(params[0])
            self._rows = [row] if row else []
        else:
            row = self._conn.projects.get(params[0])
            self._rows = [row] if row else []

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None


class _ProjectsPool:
    def __init__(self) -> None:
        self.conn = _ProjectsConn()

    def getconn(self):
        return self.conn

    def putconn(self, _conn) -> None:
        pass


@pytest.fixture
def projects():
    from umrs.api import projects as mod

    pool = _ProjectsPool()
    pool.conn.add("proj-b", owner="bob")
    return _client(mod.router, pool), pool.conn


@pytest.mark.parametrize("path", ["/umrs/projects/proj-b", "/umrs/projects/proj-b/budget"])
def test_another_users_project_reads_as_not_found(projects, path):
    client, _ = projects
    assert client.get(path, headers=_headers("alice", "proj-a")).status_code == 404


@pytest.mark.parametrize("path", ["/umrs/projects/proj-b", "/umrs/projects/proj-b/budget"])
def test_project_reads_need_identity(projects, path):
    client, _ = projects
    assert client.get(path).status_code == 401


@pytest.mark.parametrize(
    "headers",
    [
        _headers("bob", "proj-x"),  # owner
        _headers("carol", "proj-b"),  # verified into the project
        _headers("dave", "proj-a"),  # affiliated through a circle
        _headers("root", "proj-a", admin=True),  # admin
    ],
)
def test_owner_member_and_admin_read_the_project(projects, headers):
    client, conn = projects
    conn.affiliations.add(("proj-b", "dave"))
    row = client.get("/umrs/projects/proj-b", headers=headers)
    assert row.status_code == 200
    assert row.json()["project"]["owner_id"] == "bob"
    budget = client.get("/umrs/projects/proj-b/budget", headers=headers)
    assert budget.status_code == 200
    assert budget.json()["budget"]["max_wall_clock_hours"] == 3.0


def test_forged_admin_flag_is_rejected(projects):
    client, _ = projects
    forged = {**_headers("alice", "proj-a"), "X-Verified-Is-Admin": "true"}
    assert client.get("/umrs/projects/proj-b", headers=forged).status_code == 401


def test_unsigned_membership_header_does_not_open_a_project(projects):
    # Review 2026-10-18 (decided, not changed): X-Verified-Project-Memberships
    # is outside the gateway HMAC, so it grants nothing here, as in
    # access_scope.CallerScope. A token-claim-only member reads "not found";
    # the r4 client's fail-open reading of that is deferred to the r4 lane.
    client, _ = projects
    claimed = {**_headers("alice", "proj-a"), "X-Verified-Project-Memberships": "proj-a,proj-b"}
    assert client.get("/umrs/projects/proj-b", headers=claimed).status_code == 404


# ── circles ──────────────────────────────────────────────────────────────────


@pytest.fixture
def circles():
    from test_api_circles import _Pool
    from umrs.api import circles as mod

    pool = _Pool()
    conn = pool._conn
    conn.projects["proj-b"] = ("proj-b", "bob", "active")
    conn.circles[("proj-b", "team")] = (
        "proj-b",
        "team",
        "Team",
        "erin",
        "active",
        None,
        None,
        None,
    )
    conn.members_add("proj-b", "team", "erin", "admin", "erin")
    conn.members_add("proj-b", "team", "frank", "member", "erin")
    return _client(mod.router, pool)


@pytest.mark.parametrize("path", ["/umrs/circles/proj-b/team", "/umrs/circles/proj-b/team/members"])
def test_unaffiliated_caller_cannot_read_a_circle(circles, path):
    resp = circles.get(path, headers=_headers("alice", "proj-a"))
    assert resp.status_code == 404
    assert "frank" not in resp.text


@pytest.mark.parametrize("path", ["/umrs/circles/proj-b/team", "/umrs/circles/proj-b/team/members"])
def test_circle_reads_need_identity(circles, path):
    assert circles.get(path).status_code == 401


@pytest.mark.parametrize(
    "headers",
    [
        _headers("frank", "proj-a"),  # member
        _headers("erin", "proj-a"),  # circle owner
        _headers("bob", "proj-a"),  # project owner
        _headers("root", "proj-a", admin=True),  # admin
    ],
)
def test_members_owners_and_admin_read_the_circle(circles, headers):
    assert circles.get("/umrs/circles/proj-b/team", headers=headers).status_code == 200
    roster = circles.get("/umrs/circles/proj-b/team/members", headers=headers)
    assert roster.status_code == 200
    assert {m["user_id"] for m in roster.json()["members"]} == {"erin", "frank"}


# ── research sessions: attach loop ──────────────────────────────────────────


class _SessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, Any] = {}
        self.links: list[Any] = []

    def get(self, session_id):
        return self.sessions.get(session_id)

    def attach_loop(self, link) -> None:
        self.links.append(link)


@pytest.fixture
def sessions(monkeypatch):
    from umrs.api import loop as loop_api
    from umrs.api import research_sessions as mod
    from umrs.knowledge.loop import ResearchLoop
    from umrs.research_sessions import ResearchSession

    store = _SessionStore()
    store.sessions["s-a"] = ResearchSession(session_id="s-a", project_id="proj-a", title="A")
    monkeypatch.setattr(mod, "_store", store)
    # A memory store of its own: in a full run an earlier test may have bound
    # the module store to Postgres, which rejects these bare loops.
    from umrs.knowledge.loop_store import LoopStore

    monkeypatch.setattr(loop_api, "_loop_store", LoopStore.memory())
    for loop_id, project, owner in (
        ("loop-bob", "proj-a", "bob"),
        ("loop-other-project", "proj-b", "alice"),
        ("loop-alice", "proj-a", "alice"),
    ):
        loop_api._loop_store.save(
            ResearchLoop(
                loop_id=loop_id,
                project_id=project,
                research_goal="g",
                metadata={"owner_user_id": owner},
            )
        )
    yield TestClient(_app_with(mod.router)), store
    loop_api.clear_loop_store()


def _app_with(router) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _attach(client, loop_id: str, venue: str = "umrs", headers=None):
    return client.post(
        "/umrs/research/sessions/s-a/loops",
        json={"loop_id": loop_id, "venue": venue},
        headers=headers or _headers("alice", "proj-a"),
    )


@pytest.mark.parametrize("loop_id", ["loop-bob", "loop-other-project", "loop-ghost"])
def test_attach_refuses_a_loop_the_caller_does_not_own(sessions, loop_id):
    client, store = sessions
    assert _attach(client, loop_id).status_code == 404
    assert store.links == []


def test_attach_cannot_relabel_a_server_loop_as_local(sessions):
    client, store = sessions
    assert _attach(client, "loop-bob", venue="local").status_code == 404
    assert store.links == []


def test_attach_own_loop_and_client_held_local_loop(sessions):
    client, store = sessions
    assert _attach(client, "loop-alice").status_code == 201
    assert _attach(client, "local-loop-1", venue="local").status_code == 201
    assert [lnk.loop_id for lnk in store.links] == ["loop-alice", "local-loop-1"]


def test_admin_attaches_any_loop_of_the_sessions_project(sessions):
    client, _ = sessions
    admin = _headers("root", "proj-a", admin=True)
    assert _attach(client, "loop-bob", headers=admin).status_code == 201


# ── companion turns ─────────────────────────────────────────────────────────


def _worker_row(owner: str = "alice") -> dict:
    return {
        "registry_key": f"{owner}@proj-a",
        "owner_user_id": owner,
        "project_id": "proj-a",
        "job_id": "job-w1",
        "state": "active",
        "attempt_index": 0,
        "attempts_max": 6,
    }


def _turn_row(**overrides) -> dict:
    row = {
        "attempt_id": "at-1",
        "job_id": "job-w1",
        "registry_key": "alice@proj-a",
        "session_id": "s-1",
        "request_token": "rt-1",
        "user_message_hash": "h",
        "state": "committed",
        "committed_response_ref": "attempt:at-1",
        "response_content": "alice's private answer",
        "parent_attempt_id": None,
        "research_session_id": None,
    }
    row.update(overrides)
    return row


def _companion(script: list) -> TestClient:
    from umrs.api import companion

    return _client(companion.router, FakePool(script))


def test_another_users_turn_reads_as_not_found():
    client = _companion([_turn_row(), _worker_row("alice")])
    resp = client.get("/umrs/companion/turns/at-1", headers=_headers("bob", "proj-a"))
    assert resp.status_code == 404
    assert "private answer" not in resp.text


def test_another_users_turn_events_stream_as_not_found():
    client = _companion([_turn_row(), _worker_row("alice"), _turn_row()])
    resp = client.get("/umrs/companion/turns/at-1/events", headers=_headers("bob", "proj-a"))
    assert "private answer" not in resp.text
    assert "attempt_not_found" in resp.text


def test_turn_without_a_worker_row_reads_as_not_found():
    client = _companion([_turn_row(), None])
    resp = client.get("/umrs/companion/turns/at-1", headers=_headers("alice", "proj-a"))
    assert resp.status_code == 404


def test_turn_reads_need_identity():
    client = _companion([_turn_row(), _worker_row("alice")])
    assert client.get("/umrs/companion/turns/at-1").status_code == 401


def test_turns_cannot_begin_on_an_ownerless_worker():
    # Review 2026-10-18: a turn on an ownerless worker could be begun by any
    # caller yet read back by nobody; beginning now uses the read rule.
    client = _companion([_worker_row(""), None, _turn_row()])
    resp = client.post(
        "/umrs/companion/chat",
        json={"registry_key": "@proj-a", "message": "hi", "request_token": "rt-1"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 404


def test_owner_reads_own_turn_and_events():
    alice = _headers("alice", "proj-a")
    client = _companion([_turn_row(), _worker_row("alice")])
    assert client.get("/umrs/companion/turns/at-1", headers=alice).json()["response_content"] == (
        "alice's private answer"
    )
    client = _companion([_turn_row(), _worker_row("alice"), _turn_row()])
    assert "private answer" in client.get("/umrs/companion/turns/at-1/events", headers=alice).text


# ── companion approvals: ownerless rows ─────────────────────────────────────


@pytest.fixture
def approvals(monkeypatch):
    from umrs.api import loop as loop_api
    from umrs.knowledge import tool_approvals
    from umrs.knowledge.loop import ResearchLoop
    from umrs.knowledge.loop_store import LoopStore

    tool_approvals.clear_all()
    # Own memory store (see the sessions fixture).
    monkeypatch.setattr(loop_api, "_loop_store", LoopStore.memory())
    loop_api._loop_store.save(
        ResearchLoop(loop_id="loop-b", project_id="proj-b", research_goal="g", metadata={})
    )
    loop_api._loop_store.save(
        ResearchLoop(
            loop_id="loop-carol",
            project_id="proj-b",
            research_goal="g",
            metadata={"owner_user_id": "carol"},
        )
    )
    tool_approvals.request_approval("loop-b", 1, {"kind": "workflow_trigger"}, approval_id="ap-b")
    tool_approvals.request_approval(
        "loop-carol", 1, {"kind": "workflow_trigger"}, approval_id="ap-c"
    )
    tool_approvals.request_approval(
        "loop-gone", 1, {"kind": "workflow_trigger"}, approval_id="ap-gone"
    )
    yield _companion([]), tool_approvals
    tool_approvals.clear_all()
    loop_api.clear_loop_store()


def _decide(client, approval_id: str, headers):
    return client.post(
        f"/umrs/companion/approvals/{approval_id}/decision",
        json={"decision": "once"},
        headers=headers,
    )


@pytest.mark.parametrize("approval_id", ["ap-b", "ap-gone"])
def test_ownerless_approval_of_another_project_reads_as_not_found(approvals, approval_id):
    client, inbox = approvals
    assert _decide(client, approval_id, _headers("alice", "proj-a")).status_code == 404
    assert inbox.get(approval_id)["status"] == "pending"


def test_ownerless_approval_is_hidden_from_other_projects(approvals):
    client, _ = approvals
    listed = client.get("/umrs/companion/approvals/pending", headers=_headers("alice", "proj-a"))
    assert listed.status_code == 200
    assert listed.json()["pending"] == []


def test_ownerless_approval_belongs_to_the_loops_project(approvals):
    client, _ = approvals
    bob = _headers("bob", "proj-b")
    listed = client.get("/umrs/companion/approvals/pending", headers=bob).json()["pending"]
    assert [r["id"] for r in listed] == ["ap-b"]
    assert _decide(client, "ap-b", bob).status_code == 200
    admin = _headers("root", "proj-a", admin=True)
    assert _decide(client, "ap-gone", admin).status_code == 200


def test_ownerless_approval_on_an_owned_loop_is_the_loop_owners(approvals):
    # Review 2026-10-18: an ownerless intent on carol's loop was decidable by
    # any writer of proj-b ("always" then flipped carol's loop to auto).
    client, inbox = approvals
    bob = _headers("bob", "proj-b")
    listed = client.get("/umrs/companion/approvals/pending", headers=bob).json()["pending"]
    assert "ap-c" not in [r["id"] for r in listed]
    assert _decide(client, "ap-c", bob).status_code == 404
    assert inbox.get("ap-c")["status"] == "pending"
    carol = _headers("carol", "proj-b")
    listed = client.get("/umrs/companion/approvals/pending", headers=carol).json()["pending"]
    assert "ap-c" in [r["id"] for r in listed]
    assert _decide(client, "ap-c", carol).status_code == 200
