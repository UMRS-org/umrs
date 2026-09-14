"""Unit tests for the knowledge graph registry + proposal stores (0069)."""

from __future__ import annotations

import pytest

# splice removed (ADR-053): pytest auto-inserts tests/ for sibling helpers; umrs editable
from fake_pg_pool import FakePool  # noqa: E402
from umrs.knowledge.graph_registry import (  # noqa: E402
    GraphRegistryStore,
    ProposalStore,
    default_graph_id,
)


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
        "reason": "missing experiment data",
        "missing_information": ["No info about experiment X"],
        "source_artifacts": ["exp.json"],
        "proposed_operations": ["ingest"],
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


class TestDefaultGraphId:
    def test_deterministic_and_project_scoped(self):
        assert default_graph_id("proj-a") == default_graph_id("proj-a")
        assert default_graph_id("proj-a") != default_graph_id("proj-b")
        assert default_graph_id("proj-a").startswith("g-")


class TestRegistryStore:
    def test_ensure_default_graph_inserts_idempotent_row(self):
        pool = FakePool(
            script=[
                None,
                _graph_row(graph_id=default_graph_id("proj-a"), name="proj-a", kind="project"),
            ]
        )
        store = GraphRegistryStore(pool)
        graph = store.ensure_default_graph("proj-a")
        sql, params = pool.record[0]
        assert "ON CONFLICT (graph_id) DO NOTHING" in sql
        assert params[0] == default_graph_id("proj-a")
        assert params[4] == "proj-a"
        assert graph.graph_id == default_graph_id("proj-a")

    def test_create_graph_uses_uuid_and_kind(self):
        pool = FakePool(script=[None, _graph_row()])
        graph = GraphRegistryStore(pool).create_graph(
            "experiment-history",
            description="History",
            owner_id="alice",
            project_id="proj-a",
            kind="research",
        )
        sql, params = pool.record[0]
        assert "INSERT INTO knowledge_graphs" in sql
        assert params[1] == "experiment-history"
        assert params[5] == "research"
        assert graph.name == "experiment-history"

    def test_resolve_graph_name_fallback(self):
        pool = FakePool(script=[None, _graph_row()])
        store = GraphRegistryStore(pool)
        graph = store.resolve_graph("Experiment-History")
        assert graph is not None
        by_name_sql, params = pool.record[1]
        assert "lower(name) = lower(%s)" in by_name_sql
        assert params == ("Experiment-History",)

    def test_list_graphs_filters_project_and_status(self):
        pool = FakePool(script=[[_graph_row()]])
        graphs = GraphRegistryStore(pool).list_graphs(project_id="proj-a", status="active")
        sql, params = pool.record[0]
        assert "project_id = %s" in sql and "status = %s" in sql
        assert params == ("proj-a", "active", 200)
        assert len(graphs) == 1

    def test_search_graphs_matches_name_description_id(self):
        pool = FakePool(script=[[_graph_row()]])
        graphs = GraphRegistryStore(pool).search_graphs("experiment")
        sql, params = pool.record[0]
        assert "name ILIKE %s OR description ILIKE %s OR graph_id ILIKE %s" in sql
        assert params == ("%experiment%", "%experiment%", "%experiment%", 50)
        assert len(graphs) == 1

    def test_set_status_rejects_unknown_status(self):
        with pytest.raises(ValueError, match="invalid graph status"):
            GraphRegistryStore(FakePool()).set_status("g-abc", "deleted")

    def test_update_graph_patches_only_given_fields(self):
        pool = FakePool(script=[None, _graph_row(description="new")])
        graph = GraphRegistryStore(pool).update_graph("g-abc", description="new")
        sql, params = pool.record[0]
        assert "description = %s" in sql
        assert "kind = %s" not in sql
        assert params == ("new", "g-abc")
        assert graph is not None and graph.description == "new"


class TestProposalStore:
    def test_create_proposal_pending(self):
        pool = FakePool(script=[None, _proposal_row()])
        proposal = ProposalStore(pool).create_proposal(
            "g-abc",
            reason="missing data",
            missing_information=["gap"],
            source_artifacts=["a.json"],
            created_by="agent",
            session_id="s-1",
        )
        sql, params = pool.record[0]
        assert "INSERT INTO knowledge_graph_proposals" in sql
        assert params[1] == "g-abc"
        assert proposal.status == "pending"

    def test_decide_moves_pending_to_approved(self):
        pool = FakePool(script=[None, _proposal_row(status="approved", decided_by="bob")])
        proposal = ProposalStore(pool).decide("gp-1", approved=True, decided_by="bob")
        sql, params = pool.record[0]
        assert "WHERE proposal_id = %s AND status = 'pending'" in sql
        assert params == ("approved", "bob", "gp-1")
        assert proposal is not None and proposal.status == "approved"

    def test_mark_executed_only_from_approved(self):
        pool = FakePool(script=[None, _proposal_row(status="executed")])
        proposal = ProposalStore(pool).mark_executed(
            "gp-1", {"operation": "knowledge_ingest", "job_id": "j-1"}
        )
        sql, params = pool.record[0]
        assert "status = 'executed'" in sql
        assert "AND status = 'approved'" in sql
        assert proposal is not None and proposal.status == "executed"

    def test_list_proposals_filters_target_and_status(self):
        pool = FakePool(script=[[_proposal_row()]])
        proposals = ProposalStore(pool).list_proposals(target_graph_id="g-abc", status="pending")
        sql, params = pool.record[0]
        assert "target_graph_id = %s" in sql and "status = %s" in sql
        assert params == ("g-abc", "pending", 100)
        assert len(proposals) == 1
