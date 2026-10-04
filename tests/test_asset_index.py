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
            lineage_map={
                str(art.id): [
                    SimpleNamespace(parent_artifact_id=uuid.uuid4(), relation="derived_from")
                ]
            },
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
            job_id="job-1",
            project_id="p1",
            state="running",
            job_type="KNOWLEDGE_INGEST",
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


class TestCallerVisibility:
    """Read-side authorization: rows invisible to a caller are filtered."""

    def _caller(self, **over):
        fields = dict(user_id="alice", project_id="p1", is_admin=False, anonymous=False)
        fields.update(over)
        return SimpleNamespace(**fields)

    def _rows(self):
        return [
            AssetRow(kind="artifact", asset_id="sys", scope="system", status="ok"),
            AssetRow(kind="artifact", asset_id="mine", scope="project:p1", status="ok"),
            AssetRow(kind="artifact", asset_id="theirs", scope="project:p2", status="ok"),
            AssetRow(kind="artifact", asset_id="own", scope="user:alice", status="ok"),
            AssetRow(kind="artifact", asset_id="notown", scope="user:bob", status="ok"),
            AssetRow(kind="artifact", asset_id="grp", scope="group:lab", status="ok"),
        ]

    def _run(self, caller, rows):
        from umrs.asset_index.fanout import _visible_row

        return [r.asset_id for r in rows if _visible_row(r, caller)]

    def test_non_admin_sees_own_scopes_only(self):
        assert self._run(self._caller(), self._rows()) == ["sys", "mine", "own"]

    def test_admin_sees_everything(self):
        assert len(self._run(self._caller(is_admin=True), self._rows())) == 6

    def test_none_caller_is_in_process_full_visibility(self):
        assert len(self._run(None, self._rows())) == 6

    def test_group_rows_hidden_for_non_admins(self):
        assert "grp" not in self._run(self._caller(), self._rows())

    def test_bad_scope_strings_fail_closed(self):
        rows = [AssetRow(kind="artifact", asset_id="x", scope="cosmic", status="ok")]
        assert self._run(self._caller(), rows) == []


class TestAssetsRouter:
    @pytest.fixture()
    def api(self, monkeypatch):
        from umrs.api import assets as assets_api
        from umrs.api._scope_identity import get_caller_scope

        art = _artifact()
        monkeypatch.setattr(
            assets_api, "_gather_stores", lambda: {"artifact_store": _FakeArtifactStore([art])}
        )
        app = FastAPI()
        app.include_router(assets_router)
        app.dependency_overrides[get_caller_scope] = lambda: SimpleNamespace(
            user_id="alice", project_id="p1", is_admin=False, anonymous=False
        )
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


class TestLoopArtifactLineage:
    """ADR-036 D4 (WP-05): loop → artifact edges derive from iteration artifact_ids."""

    def test_loop_rows_carry_produced_edges_deduped(self):
        loop = SimpleNamespace(
            loop_id="loop-9",
            project_id="p1",
            research_goal="digital filter",
            status="stopped",
            research_session_id="",
            metadata={"hosting": "job", "durable_job_id": "job-77"},
            iterations=[
                SimpleNamespace(artifact_ids=["finding-1", "wf-art-1"]),
                SimpleNamespace(artifact_ids=["wf-art-1", "finding-2"]),
                SimpleNamespace(artifact_ids=[]),
            ],
        )
        store = SimpleNamespace(list_all=lambda: [loop])
        rows, warnings = build_asset_index(loop_store=store)
        assert not warnings
        row = rows[0]
        produced = [(e.to_kind, e.to_id, e.relation) for e in row.lineage]
        assert produced == [
            ("artifact", "finding-1", "produced"),
            ("artifact", "wf-art-1", "produced"),
            ("artifact", "finding-2", "produced"),
        ]
        assert row.metadata["hosting"] == "job"
        assert row.metadata["durable_job_id"] == "job-77"

    def test_loop_without_iterations_has_no_artifact_edges(self):
        loop = SimpleNamespace(
            loop_id="loop-0", project_id="p1", research_goal="g", status="running"
        )
        rows, _ = build_asset_index(loop_store=SimpleNamespace(list_all=lambda: [loop]))
        assert rows[0].lineage == ()
        assert rows[0].metadata["hosting"] == ""


# ── ADR-086 (2026-10-05): circle-aware visibility and the /patch wire lane.


class TestCircleVisibilityNow:
    """Circles stop being invisible when the caller carries memberships."""

    def _caller(self):
        return SimpleNamespace(user_id="alice", project_id="p1", is_admin=False, anonymous=False)

    def _rows(self):
        return [
            AssetRow(kind="artifact", asset_id="grp-active", scope="circle:p1/lab", status="ok"),
            AssetRow(kind="artifact", asset_id="grp-other", scope="circle:p1/ops", status="ok"),
        ]

    def test_circle_rows_visible_for_active_memberships(self):
        from umrs.asset_index.fanout import _visible_row

        seen = [
            r.asset_id
            for r in self._rows()
            if _visible_row(r, self._caller(), circle_refs=frozenset({"p1/lab"}))
        ]
        assert seen == ["grp-active"]

    def test_group_scoped_rows_grant_nothing(self):
        # ADR-086: superseded group: scopes are fail-closed in the fanout too
        from umrs.asset_index.fanout import _visible_row

        legacy = AssetRow(kind="artifact", asset_id="old", scope="group:lab", status="ok")
        assert not _visible_row(legacy, self._caller(), circle_refs=frozenset({"p1/lab"}))

    def test_builder_threads_caller_circles(self):
        # default () keeps the fail-closed shape exactly (legacy behavior)
        rows, _ = build_asset_index(
            artifact_store=None,
            caller=self._caller(),
        )
        assert rows == []
        grp = AssetRow(kind="artifact", asset_id="grp", scope="circle:p1/lab", status="ok")
        from umrs.asset_index.fanout import _visible_row

        assert not _visible_row(grp, self._caller())
        assert _visible_row(grp, self._caller(), circle_refs=frozenset({"p1/lab"}))


class TestCatalogPatchRoute:
    """POST /patch wire lane: caller-held base, server-side diff, no cookies."""

    def _client(self, store_rows):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from umrs.api.assets import router as assets_router

        app = FastAPI()
        app.include_router(assets_router)

        import umrs.api.assets as mod

        original_gather = mod._gather_stores
        mod._gather_stores = lambda: dict(store_rows)
        self._restore = lambda: setattr(mod, "_gather_stores", original_gather)
        return TestClient(app)

    def teardown_method(self):
        if hasattr(self, "_restore"):
            self._restore()

    def _caller_headers(self):
        return {
            "x-user-id": "alice",
            "x-verified-project-id": "p1",
        }

    def test_patch_from_empty_base_reports_all_as_upserts_and_hash_matches(self):
        rows = [
            AssetRow(kind="model", asset_id="m1", scope="system", status="ok"),
            AssetRow(kind="job", asset_id="j9", scope="project:p1", status="ok"),
        ]

        import umrs.api.assets as mod
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(mod.router)
        was = mod.build_asset_index
        mod_rows = list(rows)
        mod.build_asset_index = lambda **_kw: (mod_rows, [])
        self._restore = lambda: setattr(mod, "build_asset_index", was)

        client = TestClient(app)
        resp = client.post("/umrs/assets/patch", json={"base": {}}, headers=self._caller_headers())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert sorted(r["asset_id"] for r in body["upserts"]) == ["j9", "m1"]
        assert body["tombstones"] == []
        assert body["same"] is False
        # snapshot route carries the same snapshot_hash additive
        snap = client.get("/umrs/assets", headers=self._caller_headers())
        assert snap.status_code == 200
        assert snap.json()["snapshot_hash"] == body["snapshot_hash"]

    def test_patch_same_base_short_circuits(self):
        import umrs.api.assets as mod
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from umrs.api.assets import _row_shape_hash

        rows = [AssetRow(kind="model", asset_id="m1", scope="system", status="ok")]
        base_rows = list(rows)
        key = "model:m1"
        base_hash = _row_shape_hash(base_rows[0].to_dict())

        app = FastAPI()
        app.include_router(mod.router)
        was = mod.build_asset_index
        mod.build_asset_index = lambda **_kw: (base_rows, [])
        self._restore = lambda: setattr(mod, "build_asset_index", was)

        client = TestClient(app)
        resp = client.post(
            "/umrs/assets/patch",
            json={"base": {key: base_hash}},
            headers=self._caller_headers(),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["same"] is True
        assert body["upserts"] == []
        assert body["tombstones"] == []

    def test_patch_tombstones_gone_rows(self):
        import umrs.api.assets as mod
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(mod.router)
        was = mod.build_asset_index
        mod.build_asset_index = lambda **_kw: ([], [])
        self._restore = lambda: setattr(mod, "build_asset_index", was)

        client = TestClient(app)
        resp = client.post(
            "/umrs/assets/patch",
            json={"base": {"model:ghost": "abc123"}},
            headers=self._caller_headers(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tombstones"] == ["model:ghost"]

    def test_row_shape_hash_is_order_stable(self):
        from umrs.api.assets import _row_shape_hash

        a = {"b": 1, "a": {"z": [1, 2], "q": None}}
        b = {"a": {"q": None, "z": [1, 2]}, "b": 1}
        assert _row_shape_hash(a) == _row_shape_hash(b)
