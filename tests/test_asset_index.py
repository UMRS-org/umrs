"""Asset index (Review F4): model invariants + fanout composition + router."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.assets import router as assets_router
from umrs.asset_index import KINDS, AssetRow, build_asset_index
from umrs.asset_index.fanout import _scope_of, model_access_context


def _artifact(**over):
    fields = dict(
        id=uuid.uuid4(),
        artifact_type="dataset",
        name="seismic-raw",
        owner_id="alice",
        project_id="p1",
        uri="s3://bucket/file.csv",
        execution_id="job-7",
        state="registered",
    )
    fields.update(over)
    return SimpleNamespace(**fields)


class _FakeArtifactStore:
    def __init__(self, arts=(), lineage_map=None):
        self._arts = list(arts)
        self._lineage = lineage_map or {}

    def list(self, query):
        return self._arts

    def lineage(self, artifact_id, depth=1):
        return self._lineage.get(str(artifact_id), [])

    def close(self):
        pass


class TestFanout:
    def test_artifacts_compose_with_lineage_edges(self):
        art = _artifact()
        store = _FakeArtifactStore(
            [art],
            lineage_map={str(art.id): [SimpleNamespace(parent_artifact_id=uuid.uuid4(), relation="derived_from")]},
        )
        rows, warnings = build_asset_index(artifact_store=store)
        assert not warnings
        assert len(rows) == 1
        row = rows[0]
        assert row.kind == "artifact"
        assert row.scope == "project:p1"
        assert row.portable is True
        relations = {e.relation for e in row.lineage}
        assert relations == {"derived_from", "executed_by"}

    def test_sessions_index_loops_as_edges(self):
        sess = SimpleNamespace(
            session_id="sess-1",
            project_id="p1",
            title="MERMAID",
            research_goal="digital filter",
            status="active",
            loops=[SimpleNamespace(loop_id="loop-9")],
        )
        store = SimpleNamespace(list=lambda project_id=None: [sess])
        rows, _ = build_asset_index(session_store=store)
        assert rows[0].lineage[0].to_kind == "loop"

    def test_loops_link_back_to_sessions(self):
        loop = SimpleNamespace(
            loop_id="loop-9",
            project_id="p1",
            research_goal="goal",
            research_session_id="sess-1",
            status="running",
            venue="umrs",
        )
        store = SimpleNamespace(list_all=lambda: [loop])
        rows, _ = build_asset_index(loop_store=store)
        assert rows[0].lineage[0].relation == "registered_by"

    def test_jobs_compose(self):
        job = SimpleNamespace(
            job_id="job-1", project_id="p1", state="running", job_type="KNOWLEDGE_INGEST",
            execution_authority="slurm",
        )
        store = SimpleNamespace(list_jobs=lambda limit=500: [job])
        rows, _ = build_asset_index(job_store=store)
        assert rows[0].scope == "project:p1"
        assert rows[0].metadata["authority"] == "slurm"

    def test_failed_collector_degrades_to_warning(self):
        class _Boom:
            def list(self, query):
                raise RuntimeError("pg down")

        rows, warnings = build_asset_index(artifact_store=_Boom())
        assert rows == []
        assert [w.kind for w in warnings] == ["artifact"]

    def test_kinds_filter(self):
        rows, warnings = build_asset_index(
            kinds=["artifact"],
            artifact_store=_FakeArtifactStore([_artifact()]),
            job_store=SimpleNamespace(list_jobs=lambda limit=500: [SimpleNamespace(job_id="x")]),
        )
        assert {r.kind for r in rows} == {"artifact"}

    def test_kind_value_error(self):
        with pytest.raises(ValueError):
            AssetRow(kind="teapot", asset_id="x", scope="system", status="ok")

    def test_scope_derivation_table(self):
        assert _scope_of(scope="system") == "system"
        assert _scope_of(scope="project:p1") == "project:p1"
        assert _scope_of(scope="group:lab-ml") == "group:lab-ml"
        assert _scope_of(project_id="p2") == "project:p2"
        assert _scope_of(owner="alice") == "user:alice"
        assert _scope_of() == "system"

    def test_model_access_context_helper(self):
        ctx = model_access_context()
        assert ctx.is_admin is True

    def test_kinds_constant_complete(self):
        assert set(KINDS) == {"artifact", "model", "mcp_server", "research_session", "loop", "job"}


class TestAssetsRouter:
    @pytest.fixture()
    def api(self, monkeypatch):
        from umrs.api import assets as assets_api

        art = _artifact()
        monkeypatch.setattr(
            assets_api, "_gather_stores", lambda: {"artifact_store": _FakeArtifactStore([art])}
        )
        app = FastAPI()
        app.include_router(assets_router)
        return TestClient(app), art

    def test_snapshot_returns_rows(self, api):
        client, art = api
        res = client.get("/umrs/assets")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["count"] == 1
        assert body["rows"][0]["asset_id"] == str(art.id)

    def test_snapshot_kind_filter(self, api):
        client, _ = api
        res = client.get("/umrs/assets", params={"kinds": "model"})
        assert res.status_code == 200
        assert res.json()["count"] == 0

    def test_get_one_asset(self, api):
        client, art = api
        res = client.get(f"/umrs/assets/artifact/{art.id}")
        assert res.status_code == 200
        assert res.json()["row"]["title"] == "seismic-raw"

    def test_get_missing_asset_404(self, api):
        client, _ = api
        res = client.get(f"/umrs/assets/artifact/{uuid.uuid4()}")
        assert res.status_code == 404
