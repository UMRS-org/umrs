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


class TestGraphContentApi:
    def _node_row(self, **overrides):
        row = {
            "node_id": "n1",
            "project_id": "proj-a",
            "label": "hypothesis-1",
            "node_type": "hypothesis",
            "graph_id": "g-abc",
            "created_at": None,
        }
        row.update(overrides)
        return row

    def _edge_row(self, **overrides):
        row = {
            "edge_id": "e1",
            "source_id": "n1",
            "target_id": "n2",
            "relation_type": "supports",
            "graph_id": "g-abc",
            "created_at": None,
        }
        row.update(overrides)
        return row

    def test_content_200_webview_shape(self):
        script = [
            _graph_row(),
            [
                self._node_row(),
                self._node_row(node_id="n2", label="finding-1", node_type="finding"),
            ],
            [self._edge_row()],
        ]
        client, _ = _client(script)
        resp = client.get("/umrs/knowledge/graphs/g-abc/content")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["graph_id"] == "g-abc"
        assert body["name"] == "experiment-history"
        assert body["truncated"] is False
        assert [n["id"] for n in body["nodes"]] == ["n1", "n2"]
        assert body["nodes"][0]["type"] == "hypothesis"
        assert body["nodes"][0]["name"] == "hypothesis-1"
        assert body["edges"] == [
            {
                "from": "n1",
                "to": "n2",
                "label": "supports",
                "type": "supports",
                "graph_id": "g-abc",
                "created_at": None,
            }
        ]

    def test_content_default_graph_legacy_fallback(self):
        default_id = default_graph_id("proj-a")
        script = [
            _graph_row(graph_id=default_id),
            [self._node_row(graph_id="")],
            [],
        ]
        client, pool = _client(script)
        resp = client.get(f"/umrs/knowledge/graphs/{default_id}/content")
        assert resp.status_code == 200, resp.text
        node_sql, node_params = pool.record[1]
        assert "graph_id = '' AND project_id = %s" in node_sql
        assert node_params[1] is True
        assert node_params[2] == "proj-a"

    def test_content_non_default_no_legacy_fallback(self):
        client, pool = _client(
            [
                _graph_row(graph_id="g-custom"),
                [],
                [],
            ]
        )
        resp = client.get("/umrs/knowledge/graphs/g-custom/content")
        assert resp.status_code == 200, resp.text
        _, node_params = pool.record[1]
        assert node_params[1] is False
        assert resp.json()["nodes"] == []

    def test_content_unknown_graph_404(self):
        client, _ = _client([None, None])
        resp = client.get("/umrs/knowledge/graphs/g-nope/content")
        assert resp.status_code == 404

    def test_content_resolves_by_name(self):
        script = [
            None,
            _graph_row(),
            [self._node_row()],
            [],
        ]
        client, _ = _client(script)
        resp = client.get("/umrs/knowledge/graphs/experiment-history/content")
        assert resp.status_code == 200, resp.text
        assert resp.json()["graph_id"] == "g-abc"


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


class TestOntologyTraversalBlock:
    def test_graph_detail_carries_resolved_default_ontology(self):
        """B-part-2 (ADR-049): traversal detail exposes the resolved ontology."""
        graph = _graph_row()
        client, _ = _client([graph])
        resp = client.get("/umrs/knowledge/graphs/g-abc")
        assert resp.status_code == 200, resp.text
        block = resp.json()["ontology"]
        assert block["graph_id"] == "g-abc"
        assert block["ontology"]["ontology_id"] == "r4-default"
        assert "transitive_relations" in block["ontology"]

    def test_graph_detail_carries_inline_registered_custom_ontology(self):
        """Inline provenance graphs resolve to their custom ontology."""
        from umrs.knowledge.ontology import Ontology

        custom = Ontology(
            ontology_id="lab-x",
            version="2",
            entity_types=frozenset({"concept"}),
            relation_types=frozenset({"supports"}),
        )
        graph = _graph_row(
            ontology_id="lab-x",
            ontology_version="2",
            provenance={"ontology": custom.to_dict()},
        )
        client, _ = _client([graph])
        resp = client.get("/umrs/knowledge/graphs/g-abc")
        assert resp.status_code == 200, resp.text
        block = resp.json()["ontology"]
        assert block["ontology"]["ontology_id"] == "lab-x"

    def test_graph_detail_unknown_reference_named_not_smudged(self):
        graph = _graph_row(ontology_id="ghost-o", ontology_version="1")
        client, _ = _client([graph])
        resp = client.get("/umrs/knowledge/graphs/g-abc")
        assert resp.status_code == 200, resp.text
        block = resp.json()["ontology"]
        assert "ontology_error" in block
