"""Loop routes are owner-scoped (defect fixed 2026-09-24).

Only loop START checked identity; list/status/iterations/snapshot/state/
append/stop/pause/resume took any caller, so any authenticated user holding
a loop id could read another user's research goal or stop / pause / resume /
overwrite their loop, and the list disclosed every loop on the installation.
Another user's loop is now a 404 (existence not disclosed); admins pass.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.loop import router
from umrs.knowledge.engine_loop_runner import _RUNNERS


@pytest.fixture(autouse=True)
def _mvp_auth_and_engine(monkeypatch):
    """Mvp auth (identity headers accepted unsigned) + simulated engine."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    from umrs.api import loop as loop_api
    from umrs.knowledge import engine_loop_runner
    from umrs.knowledge.loop import AutonomousResearchLoop

    loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
    monkeypatch.setattr(
        loop_api, "_registered_workflow_names", lambda _owner, _project=None, **_kw: ["wf"]
    )
    monkeypatch.setattr(engine_loop_runner, "register", lambda *_a, **_k: None)
    loop_api._configure_hosting()
    yield
    loop_api.clear_loop_store()


def _client(user: str, admin: bool = False) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    headers = {"X-User-ID": user, "X-Verified-Project-ID": "proj-x"}
    if admin:
        headers["X-Verified-Is-Admin"] = "true"
    return TestClient(app, headers=headers)


@pytest.fixture()
def alice_loop() -> str:
    resp = _client("alice").post(
        "/umrs/loops",
        json={"project_id": "proj-x", "research_goal": "alice's secret goal", "max_iterations": 3},
    )
    assert resp.status_code == 201, resp.text
    loop_id = resp.json()["loop_id"]
    _RUNNERS.pop(loop_id, None)  # keep it RUNNING for the surface calls
    return loop_id


_READS = ("", "/iterations", "/snapshot")
_WRITES = (("post", "/stop"), ("post", "/pause"), ("post", "/resume"))


@pytest.mark.parametrize("suffix", _READS)
def test_another_user_cannot_read_the_loop(alice_loop, suffix):
    resp = _client("bob").get(f"/umrs/loops/{alice_loop}{suffix}")
    assert resp.status_code == 404
    assert "secret" not in resp.text


@pytest.mark.parametrize(("method", "suffix"), _WRITES)
def test_another_user_cannot_control_the_loop(alice_loop, method, suffix):
    resp = getattr(_client("bob"), method)(f"/umrs/loops/{alice_loop}{suffix}")
    assert resp.status_code == 404


def test_another_user_cannot_overwrite_state_or_append(alice_loop):
    bob = _client("bob")
    assert (
        bob.put(f"/umrs/loops/{alice_loop}/state", json={"current_iteration": 2}).status_code == 404
    )
    append = bob.post(
        f"/umrs/loops/{alice_loop}/iterations",
        json={"iteration_number": 1, "status": "completed"},
    )
    assert append.status_code in (404, 422)  # never applied to alice's loop
    assert append.status_code != 200


def test_the_list_shows_only_the_callers_loops(alice_loop):
    bob_ids = {row["loop_id"] for row in _client("bob").get("/umrs/loops").json()["loops"]}
    alice_ids = {row["loop_id"] for row in _client("alice").get("/umrs/loops").json()["loops"]}
    assert alice_loop not in bob_ids
    assert alice_loop in alice_ids


def test_list_rows_carry_the_project_id(alice_loop):
    """2026-10-01: the work pane scopes loops by the row's project_id —
    the list ships the loop's real project, never a client guess."""
    rows = _client("alice").get("/umrs/loops").json()["loops"]
    mine = {row["loop_id"]: row for row in rows}[alice_loop]
    assert mine["project_id"] == "proj-x"


def test_owner_and_admin_pass(alice_loop):
    assert _client("alice").get(f"/umrs/loops/{alice_loop}").status_code == 200
    admin = _client("root", admin=True)
    assert admin.get(f"/umrs/loops/{alice_loop}").status_code == 200
    assert alice_loop in {r["loop_id"] for r in admin.get("/umrs/loops").json()["loops"]}


# ── D-52: the loop records the companion session that started it ────────────


def _start(client: TestClient, **headers) -> str:
    resp = client.post(
        "/umrs/loops",
        json={"project_id": "proj-x", "research_goal": "g", "max_iterations": 2},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    _RUNNERS.pop(resp.json()["loop_id"], None)
    return resp.json()["loop_id"]


def test_start_records_the_companion_session_and_the_list_filters_by_it():
    alice = _client("alice")
    mine = _start(alice, **{"x-r4-session-id": "sess-a1"})
    other = _start(alice)
    status = alice.get(f"/umrs/loops/{mine}").json()
    assert status["metadata"]["companion_session_id"] == "sess-a1"
    assert status["metadata"]["trace_id"] == mine
    rows = alice.get("/umrs/loops", params={"companion_session_id": "sess-a1"}).json()["loops"]
    assert [r["loop_id"] for r in rows] == [mine]
    assert other not in {r["loop_id"] for r in rows}


def test_malformed_session_ids_are_not_recorded():
    alice = _client("alice")
    loop_id = _start(alice, **{"x-r4-session-id": "bad id; drop"})
    assert "companion_session_id" not in alice.get(f"/umrs/loops/{loop_id}").json()["metadata"]


def test_the_session_filter_never_reaches_another_users_loops():
    _start(_client("alice"), **{"x-r4-session-id": "shared-name"})
    rows = _client("bob").get("/umrs/loops", params={"companion_session_id": "shared-name"})
    assert rows.json()["loops"] == []


# ── a loop runs in a project its starter may write (2026-10-09) ─────────────


def test_a_loop_cannot_be_started_in_another_project():
    """The body named the project and nothing bound it to the caller: a loop
    read, wrote and spent model capacity in any project the caller named."""
    resp = _client("alice").post(
        "/umrs/loops", json={"project_id": "proj-other", "research_goal": "g", "max_iterations": 1}
    )
    assert resp.status_code == 403
    assert resp.headers["x-platform-error-code"] == "loops.forbidden"


def test_an_admin_may_start_a_loop_in_any_project():
    resp = _client("root", admin=True).post(
        "/umrs/loops", json={"project_id": "proj-other", "research_goal": "g", "max_iterations": 1}
    )
    assert resp.status_code == 201, resp.text


def test_a_loop_needs_a_project():
    resp = _client("alice").post(
        "/umrs/loops", json={"project_id": " ", "research_goal": "g", "max_iterations": 1}
    )
    assert resp.status_code == 400
