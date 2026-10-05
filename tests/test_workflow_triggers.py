"""W4 tests — catalog ingest event → workflow trigger dispatch.

Covers the matcher, the per-entry dispatch, the full fan-out over the
workflow catalog, and the best-effort wiring from
``data_catalog.triggers.trigger_post_upload_ingestion``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from umrs.data_catalog import triggers as catalog_triggers
from umrs.server import workflow_triggers

# ── Matcher ─────────────────────────────────────────────────────────────────


class TestTriggerMatches:
    def _event(self):
        return {"project_id": "proj-a", "bucket": "uploads", "s3_key": "img/ct.tif"}

    def test_bare_ingest_trigger_matches_everything(self):
        assert workflow_triggers._trigger_matches({"on": "ingest"}, **self._event())

    def test_wrong_event_type_rejected(self):
        assert not workflow_triggers._trigger_matches({"on": "cron"}, **self._event())

    def test_project_mismatch_rejected(self):
        assert not workflow_triggers._trigger_matches(
            {"on": "ingest", "project": "proj-b"}, **self._event()
        )

    def test_bucket_mismatch_rejected(self):
        assert not workflow_triggers._trigger_matches(
            {"on": "ingest", "bucket": "scratch"}, **self._event()
        )

    def test_prefix_must_match_key_start(self):
        assert workflow_triggers._trigger_matches(
            {"on": "ingest", "prefix": "img/"}, **self._event()
        )
        assert not workflow_triggers._trigger_matches(
            {"on": "ingest", "prefix": "video/"}, **self._event()
        )

    def test_wildcard_values_match(self):
        trig = {"on": "ingest", "project": "*", "bucket": "*"}
        assert workflow_triggers._trigger_matches(trig, **self._event())


# ── Catalog fan-out ─────────────────────────────────────────────────────────


def _write_workflow_dir(base: Path, name: str, triggers: list) -> Path:
    wf_dir = base / name
    wf_dir.mkdir(parents=True)
    (wf_dir / "workflow.json").write_text(
        json.dumps({"name": name, "version": "1.0", "triggers": triggers})
    )
    return wf_dir


def _write_catalog(path: Path, entries: list) -> None:
    path.write_text(json.dumps({"workflows": entries}))


def test_stem_key_only_scopes_run_to_the_uploaded_file(catalog_env, tmp_path):
    """`stem_key_only: true` → run params carry only_videos=<uploaded stem>,
    so batch uploads fan out as per-file runs instead of whole-bucket rescans."""
    catalog_file, calls = catalog_env
    wf_dir = _write_workflow_dir(
        tmp_path,
        "wf-auto",
        [{"on": "ingest", "bucket": "demo", "prefix": "opossum_", "stem_key_only": True}],
    )
    _write_catalog(
        catalog_file,
        [{"name": "wf-auto", "user_id": "alice", "source_path": str(wf_dir)}],
    )
    ids = workflow_triggers.dispatch_catalog_event(
        project_id="p", bucket="demo",
        s3_key="opossum_215_A/cam/20260325_030811.mp4", owner="alice",
    )
    assert ids == ["exec-1"]
    params = json.loads(calls[0]["payload"])["params"]
    assert params["only_videos"] == "20260325_030811"


@pytest.fixture
def catalog_env(tmp_path, monkeypatch):
    """Temp workflow catalog + captured trigger calls."""

    calls: list[dict] = []

    def _fake_handle(cmd, client_id, payload, job_backend=None):
        calls.append({"cmd": cmd, "client_id": client_id, "payload": payload})
        return json.dumps({"execution_id": f"exec-{len(calls)}", "status": "accepted"})

    catalog_file = tmp_path / "catalog.json"
    monkeypatch.setattr(
        "umrs.server.workflow_registry_handler._catalog_path",
        lambda: str(catalog_file),
    )
    monkeypatch.setattr(
        "umrs.server.workflow_registry_handler.handle_workflow_command",
        _fake_handle,
    )
    return catalog_file, calls


class TestDispatchCatalogEvent:
    def test_matching_workflow_is_triggered(self, tmp_path, catalog_env):
        catalog_file, calls = catalog_env
        wf_dir = _write_workflow_dir(
            tmp_path, "ct-analysis", [{"on": "ingest", "bucket": "uploads"}]
        )
        _write_catalog(
            Path(catalog_file),
            [
                {
                    "name": "ct-analysis",
                    "user_id": "alice",
                    "scope": "user",
                    "source_path": str(wf_dir),
                }
            ],
        )

        ids = workflow_triggers.dispatch_catalog_event(
            project_id="proj-a", bucket="uploads", s3_key="img/ct.tif", owner="alice"
        )

        assert ids == ["exec-1"]
        payload = json.loads(calls[0]["payload"])
        assert payload["workflow_name"] == "ct-analysis"
        assert payload["params"]["ingest_s3_key"] == "img/ct.tif"
        assert payload["params"]["bucket"] == "uploads"

    def test_non_matching_workflow_ignored(self, tmp_path, catalog_env):
        catalog_file, calls = catalog_env
        wf_dir = _write_workflow_dir(
            tmp_path, "video-analysis", [{"on": "ingest", "bucket": "videos"}]
        )
        _write_catalog(
            Path(catalog_file),
            [
                {
                    "name": "video-analysis",
                    "user_id": "bob",
                    "scope": "user",
                    "source_path": str(wf_dir),
                }
            ],
        )

        ids = workflow_triggers.dispatch_catalog_event(
            project_id="proj-a", bucket="uploads", s3_key="img/ct.tif", owner="alice"
        )
        assert ids == []
        assert calls == []

    def test_catalog_unavailable_returns_empty(self, monkeypatch):
        def _boom():
            raise RuntimeError("catalog down")

        monkeypatch.setattr("umrs.server.workflow_registry_handler._load_catalog", _boom)
        ids = workflow_triggers.dispatch_catalog_event(
            project_id="p", bucket="b", s3_key="k", owner="o"
        )
        assert ids == []

    def test_entry_without_triggers_skipped(self, tmp_path, catalog_env):
        catalog_file, calls = catalog_env
        wf_dir = tmp_path / "plain"
        wf_dir.mkdir()
        (wf_dir / "workflow.json").write_text(json.dumps({"name": "plain", "version": "1.0"}))
        _write_catalog(
            Path(catalog_file),
            [
                {
                    "name": "plain",
                    "user_id": "alice",
                    "scope": "user",
                    "source_path": str(wf_dir),
                }
            ],
        )
        ids = workflow_triggers.dispatch_catalog_event(
            project_id="p", bucket="b", s3_key="k", owner="o"
        )
        assert ids == []
        assert calls == []


# ── Post-upload wiring ──────────────────────────────────────────────────────


class TestPostUploadWiring:
    def test_ingest_enqueue_triggers_dispatch(self, monkeypatch):
        dispatched: list[dict] = []

        monkeypatch.setattr(
            catalog_triggers,
            "_enqueue_ingestion_job",
            lambda **_: "job-1",
        )
        import umrs.api.catalog_store as catalog_store_module

        monkeypatch.setattr(
            catalog_store_module,
            "get_optional_catalog_meta_store",
            lambda: None,
        )

        def _fake_dispatch(**kwargs):
            dispatched.append(kwargs)
            return ["exec-9"]

        monkeypatch.setattr(
            "umrs.server.workflow_triggers.dispatch_catalog_event",
            _fake_dispatch,
        )

        job_id = catalog_triggers.trigger_post_upload_ingestion(
            None,
            "uploads",
            s3_key="img/ct.tif",
            object_etag="etag-1",
            project_id="proj-a",
            owner="alice",
        )

        assert job_id == "job-1"
        assert dispatched == [
            {
                "project_id": "proj-a",
                "bucket": "uploads",
                "s3_key": "img/ct.tif",
                "owner": "alice",
            }
        ]

    def test_dispatch_failure_never_breaks_upload(self, monkeypatch):
        monkeypatch.setattr(
            catalog_triggers,
            "_enqueue_ingestion_job",
            lambda **_: "job-1",
        )
        import umrs.api.catalog_store as catalog_store_module

        monkeypatch.setattr(
            catalog_store_module,
            "get_optional_catalog_meta_store",
            lambda: None,
        )

        def _boom_dispatch(**kwargs):
            raise RuntimeError("dispatcher down")

        monkeypatch.setattr(
            "umrs.server.workflow_triggers.dispatch_catalog_event",
            _boom_dispatch,
        )

        job_id = catalog_triggers.trigger_post_upload_ingestion(
            None,
            "uploads",
            s3_key="img/ct.tif",
            object_etag="etag-1",
            project_id="proj-a",
            owner="alice",
        )
        assert job_id == "job-1"
