"""Server-side agent-mutation guard tests (0069 proposals gate).

The k2 ingest route is the enforcement point under test:
  - human actors (no x-r4-actor header) keep existing behavior
  - agent actors need an approved proposal targeting the project's
    default graph; provenance is recorded on execution
"""

from __future__ import annotations

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.knowledge_v2 import router as k2_router
from umrs.knowledge.graph_registry import default_graph_id


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def submit_job_sync(self, *, job_type, project_id, spec):
        self.calls.append({"job_type": job_type, "project_id": project_id, "spec": spec})
        return "job-1"


@pytest.fixture
def runtime(monkeypatch) -> _FakeRuntime:
    fake = _FakeRuntime()
    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: fake)
    return fake


def _client(script) -> tuple[TestClient, FakePool]:
    app = FastAPI()
    app.include_router(k2_router)
    pool = FakePool(script)
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


def _proposal_row(**overrides):
    row = {
        "proposal_id": "gp-1",
        "target_graph_id": default_graph_id("proj-a"),
        "reason": "gap",
        "missing_information": [],
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


def _graph_row(**overrides):
    row = {
        "graph_id": "g-exp",
        "name": "experiment-history",
        "description": "Experiments",
        "owner_id": "alice",
        "project_id": "proj-a",
        "kind": "experiment",
        "status": "active",
        "ontology_id": None,
        "ontology_version": None,
        "provenance": {},
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _ingest(
    client: TestClient,
    *,
    actor: str | None = None,
    proposal_id: str | None = None,
    graph_id: str | None = None,
) -> object:
    headers = {"X-Project-ID": "proj-a"}
    if actor:
        headers["x-r4-actor"] = actor
        headers["x-r4-session-id"] = "s-1"
    body: dict = {"sources": [{"kind": "local_file", "uri": "/data/exp.json"}]}
    if proposal_id:
        body["proposal_id"] = proposal_id
    if graph_id:
        body["graph_id"] = graph_id
    return client.post("/umrs/k2/ingest", json=body, headers=headers)


@pytest.mark.usefixtures("runtime")
class TestKnowledgeMutationGuard:
    def test_human_actor_without_proposal_is_allowed(self, runtime):
        client, _ = _client([])
        resp = _ingest(client)
        assert resp.status_code == 200, resp.text
        assert resp.json()["job_id"] == "job-1"
        assert runtime.calls[0]["spec"].get("proposal_id") is None

    def test_agent_without_proposal_is_rejected_409(self, runtime):
        client, _ = _client([])
        resp = _ingest(client, actor="agent")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "proposal_required"
        assert runtime.calls == []

    def test_agent_with_pending_proposal_is_rejected_409(self, runtime):
        client, _ = _client([_proposal_row(status="pending")])
        resp = _ingest(client, actor="agent", proposal_id="gp-1")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "proposal_not_approved"
        assert runtime.calls == []

    def test_agent_with_unknown_proposal_is_404(self, runtime):
        client, _ = _client([None])
        resp = _ingest(client, actor="agent", proposal_id="gp-nope")
        assert resp.status_code == 404
        assert runtime.calls == []

    def test_agent_with_mismatched_target_is_rejected_409(self, runtime):
        client, _ = _client([_proposal_row(status="approved", target_graph_id="g-other")])
        resp = _ingest(client, actor="agent", proposal_id="gp-1")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "proposal_target_mismatch"
        assert runtime.calls == []

    def test_agent_with_approved_proposal_executes_and_records(self, runtime):
        client, pool = _client(
            [
                _proposal_row(status="approved"),  # guard: get_proposal
                None,  # mark_executed: update
                _proposal_row(status="executed"),  # mark_executed: re-read
            ]
        )
        resp = _ingest(client, actor="agent", proposal_id="gp-1")
        assert resp.status_code == 200, resp.text
        # Job spec carries the proposal for job-level provenance
        assert runtime.calls[0]["spec"]["proposal_id"] == "gp-1"
        # Execution provenance written to the proposal row
        executed_sql = pool.record[1][0]
        assert "status = 'executed'" in executed_sql
        assert "AND status = 'approved'" in executed_sql

    def test_agent_named_graph_with_matching_proposal_executes(self, runtime):
        client, pool = _client(
            [
                _graph_row(),  # registry resolve (id hit)
                _proposal_row(
                    status="approved",  # guard: proposal
                    proposal_id="gp-1",
                    target_graph_id="g-exp",
                ),
                None,  # mark_executed update
                _proposal_row(status="executed"),  # mark_executed re-read
            ]
        )
        resp = _ingest(client, actor="agent", proposal_id="gp-1", graph_id="experiment-history")
        assert resp.status_code == 200, resp.text
        assert runtime.calls[0]["spec"]["graph_id"] == "g-exp"
        assert runtime.calls[0]["spec"]["proposal_id"] == "gp-1"

    def test_named_graph_proposal_mismatch_is_409(self, runtime):
        client, _ = _client(
            [
                _graph_row(),  # registry resolve
                _proposal_row(
                    status="approved",  # proposal targets default
                    proposal_id="gp-1",
                ),
            ]
        )
        resp = _ingest(client, actor="agent", proposal_id="gp-1", graph_id="experiment-history")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "proposal_target_mismatch"
        assert runtime.calls == []

    def test_unknown_named_graph_is_400_before_guard(self, runtime):
        client, _ = _client([None, None])  # resolve: id miss + name miss
        resp = _ingest(client, graph_id="no-such-graph")
        assert resp.status_code == 400
        assert runtime.calls == []

    def test_agent_direct_node_write_is_blocked_without_proposal(self):
        from umrs.api.knowledge_graph import (
            _project_graphs,
            register_project_graph,
        )
        from umrs.api.knowledge_graph import (
            router as kg_router,
        )
        from umrs.knowledge_graph_api import InMemoryKnowledgeGraph

        register_project_graph("proj-a", InMemoryKnowledgeGraph())
        try:
            app = FastAPI()
            app.include_router(kg_router)
            app.state.umrs_db_pool = FakePool([])
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/umrs/knowledge-graph/nodes",
                json={"node_id": "n-1", "node_type": "concept"},
                headers={"X-Project-ID": "proj-a", "x-r4-actor": "agent"},
            )
            # R5 tail: sat facade retired — gate now 410s before mutation
            # guard runs; canonical path lives on /umrs/knowledge/graph.
            assert resp.status_code == 410, resp.text
        finally:
            _project_graphs.pop("proj-a", None)
