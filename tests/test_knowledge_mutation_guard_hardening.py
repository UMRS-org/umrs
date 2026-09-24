"""Mutation-guard hardening tests (0069, Phase 5 re-audit).

Two agent bypass paths closed:
  - POST /umrs/knowledge/restore wrote persistent graph content with no
    proposal requirement for agent actors
  - POST/PATCH /umrs/knowledge/graphs accepted agent actors (graph spam /
    archive abuse) — registry lifecycle is human-only now
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.bundle_restore import router as restore_router
from umrs.api.knowledge_graphs import router as graphs_router
from umrs.knowledge.graph_registry import default_graph_id


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


def _engine():
    engine = MagicMock()
    engine.import_project = AsyncMock(return_value=SimpleNamespace(to_dict=lambda: {"nodes": 1}))
    return engine


def _restore_client(monkeypatch, script, engine):
    app = FastAPI()
    app.include_router(restore_router)
    app.state.umrs_db_pool = FakePool(script)

    import umrs.api._asyncpg_pool as pool_mod
    import umrs.knowledge.bundle_import as bundle_import

    monkeypatch.setattr(pool_mod, "knowledge_asyncpg_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(bundle_import, "KnowledgeImportEngine", lambda **_kwargs: engine)
    return TestClient(app, raise_server_exceptions=False), app.state.umrs_db_pool


@pytest.fixture
def bundle_dir(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"project_id": "proj-a"}))
    return str(tmp_path)


class TestRestoreGuard:
    def test_agent_restore_without_proposal_is_409(self, monkeypatch, bundle_dir):
        engine = _engine()
        client, _ = _restore_client(monkeypatch, [], engine)
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": bundle_dir},
            headers={"x-r4-actor": "agent", "x-r4-session-id": "s-1"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "proposal_required"
        engine.import_project.assert_not_called()

    def test_agent_restore_target_mismatch_is_409(self, monkeypatch, bundle_dir):
        engine = _engine()
        client, _ = _restore_client(
            monkeypatch,
            [_proposal_row(status="approved", target_graph_id="g-other")],
            engine,
        )
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": bundle_dir, "proposal_id": "gp-1"},
            headers={"x-r4-actor": "agent"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "proposal_target_mismatch"
        engine.import_project.assert_not_called()

    def test_agent_restore_approved_executes_with_provenance(self, monkeypatch, bundle_dir):
        engine = _engine()
        client, pool = _restore_client(
            monkeypatch,
            [
                _proposal_row(status="approved"),  # guard: get_proposal
                None,  # mark_executed: update
                _proposal_row(status="executed"),  # mark_executed: re-read
            ],
            engine,
        )
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": bundle_dir, "proposal_id": "gp-1"},
            headers={"x-r4-actor": "agent", "x-r4-session-id": "s-1"},
        )
        assert resp.status_code == 200, resp.text
        engine.import_project.assert_awaited_once()
        assert "status = 'executed'" in pool.record[1][0]

    def test_human_restore_unchanged(self, monkeypatch, bundle_dir):
        engine = _engine()
        client, _ = _restore_client(monkeypatch, [], engine)
        resp = client.post("/umrs/knowledge/restore", json={"bundle_dir": bundle_dir})
        assert resp.status_code == 200, resp.text
        engine.import_project.assert_awaited_once()


class TestRestoreIsOperatorOnly:
    """bundle_dir resolves on the UMRS host (server rights; write_upgrade
    rewrites its manifest) and project_id retargets anywhere — so server-path
    restore is admin-only; the legacy anonymous dev scope keeps working."""

    def test_non_admin_caller_refused_by_name(self, monkeypatch, bundle_dir):
        from r4_contracts.errors import ERROR_CODE_HEADER

        engine = _engine()
        client, _ = _restore_client(monkeypatch, [], engine)
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": bundle_dir, "project_id": "someone-elses-project"},
            headers={"x-user-id": "mallory", "x-verified-project-id": "proj-m"},
        )
        assert (resp.status_code, resp.headers.get(ERROR_CODE_HEADER)) == (403, "bundle.forbidden")
        engine.import_project.assert_not_called()

    def test_legacy_project_scope_refused(self, monkeypatch, bundle_dir):
        engine = _engine()
        client, _ = _restore_client(monkeypatch, [], engine)
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": bundle_dir},
            headers={"x-project-id": "proj-a"},
        )
        assert resp.status_code == 403
        engine.import_project.assert_not_called()

    def test_admin_restores(self, monkeypatch, bundle_dir):
        engine = _engine()
        client, _ = _restore_client(monkeypatch, [], engine)
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": bundle_dir},
            headers={
                "x-user-id": "ops",
                "x-verified-project-id": "proj-a",
                "x-verified-is-admin": "true",
            },
        )
        assert resp.status_code == 200, resp.text
        engine.import_project.assert_awaited_once()


class TestLintNeverReadsServerPaths:
    def _client(self):
        from umrs.api.knowledge_v2 import router as k2_router

        app = FastAPI()
        app.include_router(k2_router)
        return TestClient(app)

    def test_files_lane_refused(self, tmp_path):
        from r4_contracts.errors import ERROR_CODE_HEADER

        secret = tmp_path / "server_only.py"
        secret.write_text("def f(:\n    pass\n")  # would surface as a syntax finding
        resp = self._client().post(
            "/umrs/k2/lint", json={"files": [str(secret)]}, headers={"X-Project-ID": "p"}
        )
        assert (resp.status_code, resp.headers.get(ERROR_CODE_HEADER)) == (
            400,
            "knowledge2.source_kind_refused",
        )
        assert "server_only" not in resp.text

    def test_source_code_lane_lints(self):
        resp = self._client().post(
            "/umrs/k2/lint",
            json={"source_code": {"a.py": "def f(:\n    pass\n"}},
            headers={"X-Project-ID": "p"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["total"] >= 1


def _graph_client(script):
    app = FastAPI()
    app.include_router(graphs_router)
    app.state.umrs_db_pool = FakePool(script)
    return TestClient(app), app.state.umrs_db_pool


class TestRegistryWriteGuard:
    def test_agent_create_is_403(self):
        client, _ = _graph_client([])
        resp = client.post(
            "/umrs/knowledge/graphs",
            json={"name": "spam-graph"},
            params={"caller_user_id": "alice", "caller_project_id": "proj-a"},
            headers={"x-r4-actor": "agent"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"] == "registry_write_forbidden_for_agent"

    def test_agent_update_and_archive_is_403(self):
        client, _ = _graph_client([])
        resp = client.patch(
            "/umrs/knowledge/graphs/g-abc",
            json={"status": "archived"},
            headers={"x-r4-actor": "agent"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"] == "registry_write_forbidden_for_agent"
