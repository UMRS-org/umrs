"""Route tests for the knowledge graph registry + proposals APIs (0069)."""

from __future__ import annotations

from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.knowledge_graphs import router as graphs_router
from umrs.api.knowledge_proposals import router as proposals_router
from umrs.knowledge.graph_registry import default_graph_id

_IDENTITY = {"caller_user_id": "alice", "caller_project_id": "proj-a"}


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
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _proposal_row(**overrides):
    row = {
        "proposal_id": "gp-1",
        "target_graph_id": "g-abc",
        "reason": "gap",
        "missing_information": ["x"],
        "source_artifacts": [],
        "proposed_operations": [],
        "created_by": "agent",
        "session_id": "s-1",
        "status": "pending",
        "decided_by": "",
        "decided_at": None,
        "execution": None,
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _client(script):
    app = FastAPI()
    app.include_router(graphs_router)
    app.include_router(proposals_router)
    pool = FakePool(script)
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


class TestGraphRegistryApi:
    def test_create_graph_201(self):
        client, pool = _client([None, _graph_row()])
        resp = client.post(
            "/umrs/knowledge/graphs",
            json={
                "name": "experiment-history",
                "description": "History",
                "project_id": "proj-a",
                "kind": "research",
            },
            params=_IDENTITY,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "experiment-history"
        assert body["owner_id"] == "alice"
        assert "INSERT INTO knowledge_graphs" in pool.record[0][0]

    def test_list_with_project_materializes_default_graph(self):
        default_id = default_graph_id("proj-a")
        client, pool = _client(
            [
                None,
                _graph_row(graph_id=default_id, name="proj-a"),
                [_graph_row(graph_id=default_id, name="proj-a")],
            ]
        )
        resp = client.get("/umrs/knowledge/graphs", params={"project_id": "proj-a"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["graphs"][0]["graph_id"] == default_id
        assert "ON CONFLICT (graph_id) DO NOTHING" in pool.record[0][0]

    def test_get_unknown_graph_404(self):
        client, _ = _client([None, None])
        resp = client.get("/umrs/knowledge/graphs/g-nope")
        assert resp.status_code == 404

    def test_patch_archive_is_soft_delete(self):
        client, pool = _client(
            [
                _graph_row(),  # get_graph existence check
                None,  # set_status update
                _graph_row(status="archived"),  # set_status re-read
                _graph_row(status="archived"),  # update_graph (no fields) re-read
            ]
        )
        resp = client.patch("/umrs/knowledge/graphs/g-abc", json={"status": "archived"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "archived"
        assert "UPDATE knowledge_graphs SET status" in pool.record[1][0]

    def test_patch_invalid_status_400(self):
        client, _ = _client([_graph_row()])
        resp = client.patch("/umrs/knowledge/graphs/g-abc", json={"status": "deleted"})
        assert resp.status_code == 400


class TestProposalApi:
    def test_create_proposal_requires_existing_graph(self):
        client, _ = _client([None, None])
        resp = client.post(
            "/umrs/knowledge/graphs/g-nope/proposals", json={"reason": "gap"}, params=_IDENTITY
        )
        assert resp.status_code == 404

    def test_create_then_approve_flow(self):
        client, pool = _client(
            [
                _graph_row(),  # _require_graph
                None,  # create insert
                _proposal_row(),  # create re-read
                _proposal_row(),  # decision: get_proposal
                None,  # decision: update
                _proposal_row(status="approved", decided_by="bob"),
            ]
        )
        created = client.post(
            "/umrs/knowledge/graphs/g-abc/proposals",
            json={
                "reason": "missing experiment X",
                "missing_information": ["No experiment X data"],
                "source_artifacts": ["exp-x.json"],
                "proposed_operations": [{"op": "knowledge_ingest"}],
                "session_id": "s-1",
            },
            params=_IDENTITY,
        )
        assert created.status_code == 201, created.text
        proposal_id = created.json()["proposal_id"]
        assert created.json()["status"] == "pending"

        decided = client.post(
            f"/umrs/knowledge/proposals/{proposal_id}/decision",
            json={"approved": True},
            params={"caller_user_id": "bob", "caller_project_id": "proj-a"},
        )
        assert decided.status_code == 200, decided.text
        body = decided.json()
        assert body["status"] == "approved"
        assert body["decided_by"] == "bob"
        assert "AND status = 'pending'" in pool.record[4][0]

    def test_double_decision_409(self):
        client, _ = _client([_proposal_row(status="approved")])
        resp = client.post(
            "/umrs/knowledge/proposals/gp-1/decision", json={"approved": False}, params=_IDENTITY
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "proposal_already_decided"

    def test_decline_persisted(self):
        client, _ = _client(
            [
                _proposal_row(),
                None,
                _proposal_row(status="declined", decided_by="bob"),
            ]
        )
        resp = client.post(
            "/umrs/knowledge/proposals/gp-1/decision", json={"approved": False}, params=_IDENTITY
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "declined"
