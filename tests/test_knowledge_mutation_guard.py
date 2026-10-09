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

    async def submit_job_sync(self, *, job_type, project_id, spec, user_id=""):
        self.calls.append(
            {"job_type": job_type, "project_id": project_id, "spec": spec, "user_id": user_id}
        )
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
    body: dict = {"sources": [{"kind": "inline_text", "uri": "exp.json", "content": "{}"}]}
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

    def test_ingest_job_is_owned_by_its_caller(self, runtime):
        """An ownerless job was 403 to its own submitter on `r4 job describe` (2026-10-09)."""
        client, _ = _client([])
        resp = client.post(
            "/umrs/k2/ingest",
            json={"sources": [{"kind": "inline_text", "uri": "n.md", "content": "x"}]},
            headers={"X-Project-ID": "proj-a", "x-user-id": "scientist"},
        )
        assert resp.status_code == 200, resp.text
        assert runtime.calls[0]["user_id"] == "scientist"

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
        """Facade deleted (ADR-026 wave C): direct writes are 404 (no route).

        The retired facade's register_project_graph store is gone with it —
        the coverage stays honest: hitting the old path must not succeed.
        """
        from umrs.knowledge.canonical_graph_view import default_view

        _ = default_view(store=None)  # construction shape kept for smoke
        app = FastAPI()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/umrs/knowledge-graph/nodes",
            json={"node_id": "n-1", "node_type": "concept"},
            headers={"X-Project-ID": "proj-a", "x-r4-actor": "agent"},
        )
        assert resp.status_code == 404, resp.text


@pytest.mark.usefixtures("runtime")
class TestRemoteSafeSourceKinds:
    """The HTTP route never lets a caller name a server path or bucket key:
    the job runs inside UMRS, so those would read the server's files / any
    key with server credentials into the caller's project."""

    @pytest.mark.parametrize(
        "source",
        [
            {"kind": "local_file", "uri": "/proc/self/environ"},
            {"kind": "local_dir", "uri": "/run/secrets"},
            {"kind": "local", "uri": "/etc"},
            {"kind": "s3_uri", "uri": "s3://umrs/other-project/secret.pdf"},
            {"uri": "/etc/passwd"},  # missing kind → the job defaults to local_file
        ],
    )
    def test_server_side_kinds_refused_by_name(self, runtime, source):
        client, _ = _client([])
        resp = client.post(
            "/umrs/k2/ingest", json={"sources": [source]}, headers={"X-Project-ID": "proj-a"}
        )
        assert resp.status_code == 400, resp.text
        from r4_contracts.errors import ERROR_CODE_HEADER

        assert resp.headers.get(ERROR_CODE_HEADER) == "knowledge2.source_kind_refused"
        assert "/umrs/knowledge/upload" in resp.text
        assert runtime.calls == []  # nothing submitted

    def test_one_bad_source_refuses_the_whole_request(self, runtime):
        client, _ = _client([])
        resp = client.post(
            "/umrs/k2/ingest",
            json={
                "sources": [
                    {"kind": "inline_text", "uri": "a.txt", "content": "ok"},
                    {"kind": "local_file", "uri": "/etc/shadow"},
                ]
            },
            headers={"X-Project-ID": "proj-a"},
        )
        assert resp.status_code == 400
        assert "sources[1]" in resp.text
        assert runtime.calls == []

    @pytest.mark.parametrize(
        "source",
        [
            {"kind": "inline_text", "uri": "notes.md", "content": "# hi"},
            {"kind": "http_url", "uri": "https://example.org/paper"},
            {"kind": "git_repo", "uri": "https://github.com/o/r"},
        ],
    )
    def test_remote_safe_kinds_pass(self, source):
        client, _ = _client([])
        resp = client.post(
            "/umrs/k2/ingest",
            json={"sources": [source], "allow_domains": ["example.org", "github.com"]},
            headers={"X-Project-ID": "proj-a"},
        )
        assert resp.status_code == 200, resp.text


@pytest.mark.usefixtures("runtime")
class TestIngestTargetAccess:
    """A caller ingests only into a graph it may write (review of KG-05,
    2026-10-09): any project's graph was a valid target, by id or by the
    oldest graph of that name anywhere, so rows could be injected into
    another project's graph views and searches."""

    def test_another_projects_graph_is_an_unknown_target(self):
        client, _ = _client([_graph_row(graph_id="g-b", project_id="proj-b", owner_id="bob")])
        from r4_contracts.errors import ERROR_CODE_HEADER

        resp = _ingest(client, graph_id="g-b")
        assert resp.status_code == 400
        assert resp.headers.get(ERROR_CODE_HEADER) == "knowledge2.unknown_graph"

    def test_a_shared_name_targets_the_callers_own_graph(self, runtime):
        foreign = _graph_row(graph_id="g-b", project_id="proj-b", owner_id="bob")
        client, _ = _client([None, [foreign, _graph_row()]])
        resp = _ingest(client, graph_id="experiment-history")
        assert resp.status_code in (200, 201, 202), resp.text
        assert runtime.calls[0]["spec"]["graph_id"] == "g-exp"
