"""Tests for UMRS Loop REST API (TASK-036).

Covers:
  - POST /umrs/loops — start a loop with/without approval
  - GET /umrs/loops — list loops
  - GET /umrs/loops/{id} — loop status
  - GET /umrs/loops/{id}/iterations — list iterations
  - POST /umrs/loops/{id}/stop — stop a loop
  - POST /umrs/loops/{id}/pause — pause a loop
  - POST /umrs/loops/{id}/resume — resume a loop
  - Unapproved project rejection (403)
"""

from __future__ import annotations

import pytest


def _try_fastapi():
    try:
        pytest.importorskip("fastapi")
    except Exception:
        pytest.skip("fastapi not available")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    return FastAPI, TestClient


@pytest.fixture(autouse=True)
def _clean_loop_store():
    from umrs.api.loop import clear_loop_store
    clear_loop_store()
    yield
    clear_loop_store()


def _make_app():
    FastAPI, TestClient = _try_fastapi()
    from umrs.api.loop import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestStartLoop:
    def test_unapproved_project_rejected(self):
        """Core TASK-036 acceptance: unapproved project refuses to start.

        research-workflow spec §12 — Loop criteria.
        """
        from unittest.mock import patch

        from umrs.knowledge.readiness import (
            ConfidenceLevel,
            IssueSeverity,
            ReadinessIssue,
            ReadinessLevel,
            ResearchReadinessReport,
        )

        unapproved_report = ResearchReadinessReport(
            project_id="unapproved-proj",
            readiness_level=ReadinessLevel.PARTIALLY_READY,
            confidence_level=ConfidenceLevel.MEDIUM,
            issues=[
                ReadinessIssue(
                    severity=IssueSeverity.WARNING,
                    category="missing_dataset",
                    description="Missing dataset",
                    recommendation="Add dataset",
                )
            ],
            recommendation="Improve coverage",
            next_steps=["Add data"],
        )

        client = _make_app()

        # The handler does: from umrs.knowledge.project import ResearchProject
        # which is now imported at module level. Patching umrs.api.loop.ResearchProject
        # affects the name in the module namespace that the handler uses.
        with patch("umrs.api.loop.ResearchProject") as mock_cls:
            mock_inst = mock_cls.return_value
            mock_inst.check_readiness.return_value = unapproved_report
            mock_inst.knowledge_store.list_artifacts.return_value = ["artifact1"]

            resp = client.post(
                "/umrs/loops",
                json={
                    "project_id": "unapproved-proj",
                    "research_goal": "Test goal",
                    "max_iterations": 5,
                    "approved": True,
                },
            )
            assert resp.status_code == 403
            detail = resp.json().get("detail", "")
            assert "not approved" in detail.lower()

    def test_trust_mode_no_approval_check(self):
        """approved=False (default) = trust the client."""
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-trust",
                "research_goal": "Trust test",
                "max_iterations": 3,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["project_id"] == "proj-trust"
        assert body["status"] == "running"
        assert body["loop_id"].startswith("loop-")

    def test_validation_rejects_empty_goal(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj", "research_goal": "", "max_iterations": 5},
        )
        assert resp.status_code == 422

    def test_validation_rejects_max_iterations_0(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj", "research_goal": "G", "max_iterations": 0},
        )
        assert resp.status_code == 422

    def test_validation_rejects_max_iterations_101(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj", "research_goal": "G", "max_iterations": 101},
        )
        assert resp.status_code == 422

    def test_approved_flag_skips_check_on_fresh_project(self):
        """Fresh project with no artifacts — trust the client."""
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "fresh",
                "research_goal": "Fresh",
                "max_iterations": 2,
                "approved": True,
            },
        )
        assert resp.status_code == 201


class TestListLoops:
    def test_empty(self):
        client = _make_app()
        resp = client.get("/umrs/loops")
        assert resp.status_code == 200
        assert resp.json()["loops"] == []

    def test_with_loops(self):
        client = _make_app()
        client.post("/umrs/loops", json={"project_id": "p1", "research_goal": "G1"})
        resp = client.get("/umrs/loops")
        assert resp.status_code == 200
        assert len(resp.json()["loops"]) == 1


class TestLoopStatus:
    def test_existing_loop(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops",
            json={"project_id": "p-status", "research_goal": "Status test"},
        )
        lid = start.json()["loop_id"]

        resp = client.get(f"/umrs/loops/{lid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loop_id"] == lid
        assert body["status"] == "running"

    def test_not_found(self):
        client = _make_app()
        resp = client.get("/umrs/loops/nonexistent")
        assert resp.status_code == 404


class TestIterations:
    def test_empty(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops",
            json={"project_id": "p-iter", "research_goal": "Iter"},
        )
        lid = start.json()["loop_id"]
        resp = client.get(f"/umrs/loops/{lid}/iterations")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_after_run(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops",
            json={"project_id": "p-iter-run", "research_goal": "Iter run", "max_iterations": 1},
        )
        lid = start.json()["loop_id"]
        resp = client.get(f"/umrs/loops/{lid}/iterations")
        assert resp.status_code == 200
        for it in resp.json():
            assert "iteration_id" in it
            assert "phase" in it


class TestLoopControl:
    def test_stop(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-stop", "research_goal": "S"})
        lid = start.json()["loop_id"]
        resp = client.post(f"/umrs/loops/{lid}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_pause(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-pause", "research_goal": "S"})
        lid = start.json()["loop_id"]
        resp = client.post(f"/umrs/loops/{lid}/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_resume(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-resume", "research_goal": "S"})
        lid = start.json()["loop_id"]
        client.post(f"/umrs/loops/{lid}/pause")
        resp = client.post(f"/umrs/loops/{lid}/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_resume_not_paused_fails(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-rn", "research_goal": "S"})
        lid = start.json()["loop_id"]
        resp = client.post(f"/umrs/loops/{lid}/resume")
        assert resp.status_code == 400
