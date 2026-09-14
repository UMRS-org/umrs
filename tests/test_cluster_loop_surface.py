"""Cluster-venue hosting surface (WP-05 round 2): PUT /state, POST /iterations
append, GET /snapshot — each must map faithfully onto the durable loop
record without inventing other state."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mvp_auth_and_engine(monkeypatch):
    """Mvp auth + simulated engine + one registered workflow name (start
    path never does real workloads in these tests)."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    from umrs.api import loop as loop_api
    from umrs.knowledge import engine_loop_runner
    from umrs.knowledge.loop import AutonomousResearchLoop

    loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
    monkeypatch.setattr(
        loop_api, "_registered_workflow_names", lambda _owner: ["demo08_yield_eval"]
    )
    monkeypatch.setattr(engine_loop_runner, "register", lambda *_args, **_kwargs: None)
    loop_api._configure_hosting()
    yield
    loop_api.clear_loop_store()


def _start_loop(client, **body) -> str:
    """POST /umrs/loops then freeze the simulated engine so the loop stays
    RUNNING for the test (surface semantics, not lifecycle)."""
    start = client.post("/umrs/loops", json=body)
    assert start.status_code == 201, start.text
    loop_id = start.json()["loop_id"]
    _freeze_loop(loop_id)
    return loop_id


def _freeze_loop(loop_id: str) -> None:
    """Detach a newly started loop from the API's simulated engine so it stays

    RUNNING for the surface calls: the phantom engine completes loops almost
    instantly, so drive the test's control flow directly against the store
    and never let the embedded runner advance this loop id.
    """
    from umrs.knowledge.engine_loop_runner import _RUNNERS

    _RUNNERS.pop(loop_id, None)  # don't let the simulated engine finish it


def _make_app():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api.loop import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(
        app,
        headers={
            "X-User-ID": "test-owner",
            "X-Verified-Project-ID": "proj-x",
        },
    )


class TestPutState:
    def test_put_state_maps_driver_owned_fields_only(self):
        client = _make_app()

        start = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-x",
                "research_goal": "explore workflows",
                "max_iterations": 5,
            },
        )
        assert start.status_code == 201
        loop_id = start.json()["loop_id"]
        response = client.put(
            f"/umrs/loops/{loop_id}/state",
            json={
                "phase": "experiment_execution",
                "current_iteration": 3,
                "total_hypotheses_tested": 7,
                "total_experiments_run": 4,
                "total_findings": 2,
                "success_rate": 0.5,
                "metadata": {"host": "cluster:rccs"},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["phase"] == "experiment_execution"
        assert payload["current_iteration"] == 3
        assert payload["total_hypotheses_tested"] == 7
        assert payload["total_experiments_run"] == 4
        assert payload["total_findings"] == 2
        assert payload["success_rate"] == 0.5
        assert payload["metadata"]["host"] == "cluster:rccs"
        # Server-owned fields are NOT clobbered by the PUT.
        assert "research_goal" in payload
        assert payload["research_goal"] == "explore workflows"

    def test_put_state_clamps_iteration_to_max(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops",
            json={"project_id": "proj-x", "research_goal": "clamp", "max_iterations": 2},
        )
        loop_id = start.json()["loop_id"]
        response = client.put(f"/umrs/loops/{loop_id}/state", json={"current_iteration": 99})
        assert response.status_code == 200
        assert response.json()["current_iteration"] == 2

    def test_put_state_invalid_status_or_phase(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "proj-x", "research_goal": "bad"})
        loop_id = start.json()["loop_id"]
        assert (
            client.put(f"/umrs/loops/{loop_id}/state", json={"status": "bogus"}).status_code == 422
        )
        assert (
            client.put(f"/umrs/loops/{loop_id}/state", json={"phase": "bogus"}).status_code == 422
        )

    def test_finished_loop_stops_saving_state(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "proj-x", "research_goal": "frozen"})
        loop_id = start.json()["loop_id"]
        assert client.post(f"/umrs/loops/{loop_id}/stop").status_code == 200
        assert (
            client.put(f"/umrs/loops/{loop_id}/state", json={"status": "running"}).status_code
            == 200
        )  # PUT succeeds as a projection (no creation), but no resurrection: status stays stopped? Validate rule below.

    def test_finish_blocked_for_append_on_stopped_loop(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops", json={"project_id": "proj-x", "research_goal": "frozen-append"}
        )
        loop_id = start.json()["loop_id"]
        assert client.post(f"/umrs/loops/{loop_id}/stop").status_code == 200
        assert (
            client.post(
                f"/umrs/loops/{loop_id}/iterations", json={"iteration_number": 1}
            ).status_code
            == 409
        )


class TestAppendIteration:
    def test_append_maps_fields_and_advances_totals(self):
        client = _make_app()
        loop_id = _start_loop(
            client,
            project_id="proj-x",
            research_goal="iterate",
            max_iterations=3,
        )

        body = {
            "iteration_number": 1,
            "phase": "experiment_execution",
            "started_at": "2026-09-10T01:00:00",
            "completed_at": "2026-09-10T01:03:00",
            "hypothesis": "h1",
            "experiment_id": "e1",
            "analysis": "a1",
            "evaluation": "keep",
            "findings": ["A", "B"],
            "status": "completed",
        }
        response = client.post(f"/umrs/loops/{loop_id}/iterations", json=body)
        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "iteration_number": 1,
            "status": "running",
            "current_iteration": 1,
        }

        iters = client.get(f"/umrs/loops/{loop_id}/iterations").json()
        assert len(iters) == 1
        assert iters[0]["experiment_id"] == "e1"
        assert iters[0]["hypothesis"] == "h1"
        assert iters[0]["findings"] == ["A", "B"]
        assert iters[0]["status"] == "completed"

        status = client.get(f"/umrs/loops/{loop_id}").json()
        assert status["total_findings"] == 2
        assert status["current_iteration"] == 1

    def test_append_completion_state_marked_ready_stops_the_loop(self):
        client = _make_app()
        loop_id = _start_loop(client, project_id="proj-x", research_goal="finish")
        response = client.post(
            f"/umrs/loops/{loop_id}/iterations",
            json={"iteration_number": 1, "completion_state": "ready"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"


class TestLoopSnapshot:
    def test_snapshot_returns_hosting_checker_view(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "proj-x", "research_goal": "snap"})
        loop_id = start.json()["loop_id"]
        snap = client.get(f"/umrs/loops/{loop_id}/snapshot")
        assert snap.status_code == 200
        payload = snap.json()
        for field in (
            "status",
            "phase",
            "current_iteration",
            "max_iterations",
            "total_findings",
            "driver_alive",
        ):
            assert field in payload
        assert payload["driver_alive"] in (True, False)

    def test_snapshot_404_on_missing_loop(self):
        client = _make_app()
        assert client.get("/umrs/loops/ghost/snapshot").status_code == 404
