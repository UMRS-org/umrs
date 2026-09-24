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

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from umrs.api.loop import router  # noqa: E402
from umrs.knowledge.engine_loop_runner import _RUNNERS  # noqa: E402


@pytest.fixture(autouse=True)
def _mvp_auth_and_engine(monkeypatch):
    """Mvp auth (identity headers accepted unsigned) + simulated engine."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    from umrs.api import loop as loop_api
    from umrs.knowledge import engine_loop_runner
    from umrs.knowledge.loop import AutonomousResearchLoop

    loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
    monkeypatch.setattr(loop_api, "_registered_workflow_names", lambda _owner: ["wf"])
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


def test_owner_and_admin_pass(alice_loop):
    assert _client("alice").get(f"/umrs/loops/{alice_loop}").status_code == 200
    admin = _client("root", admin=True)
    assert admin.get(f"/umrs/loops/{alice_loop}").status_code == 200
    assert alice_loop in {r["loop_id"] for r in admin.get("/umrs/loops").json()["loops"]}
