"""D-40 transport: POST /umrs/knowledge/import (bundle bytes → caller's project)
and the export's project guard + archive publish."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.bundle_restore import router as restore_router

from r4_contracts.errors import ERROR_CODE_HEADER


def _archive(manifest: dict | None = None, extra=None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = json.dumps(manifest or {"project_id": "origin-proj"}).encode()
        info = tarfile.TarInfo("bundle/manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        if extra:
            extra(tar)
    return buf.getvalue()


@pytest.fixture
def engine(monkeypatch):
    import umrs.api._asyncpg_pool as pool_mod
    import umrs.knowledge.bundle_import as bundle_import

    seen = {}
    eng = MagicMock()

    async def _import(bundle_dir, **kwargs):
        seen["manifest"] = json.loads((Path(bundle_dir) / "manifest.json").read_text())
        seen["bundle_dir"] = bundle_dir
        seen.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"node_count": 1, "verified": False})

    eng.import_project = AsyncMock(side_effect=_import)
    monkeypatch.setattr(pool_mod, "knowledge_asyncpg_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(bundle_import, "KnowledgeImportEngine", lambda **_kw: eng)
    eng.seen = seen
    return eng


def _client():
    app = FastAPI()
    app.include_router(restore_router)
    return TestClient(app, raise_server_exceptions=False)


def _post(archive: bytes, **params):
    return _client().post(
        "/umrs/knowledge/import",
        files={"file": ("shared.kb.tar.gz", archive, "application/gzip")},
        params=params,
        headers={"X-Project-ID": "caller-proj"},
    )


def test_import_lands_in_the_callers_project_verified(engine):
    resp = _post(_archive())
    assert resp.status_code == 200, resp.text
    assert resp.json()["project_id"] == "caller-proj"
    seen = engine.seen
    assert (seen["target_project_id"], seen["verify"], seen["allow_unsigned"]) == (
        "caller-proj",
        True,
        False,
    )
    assert seen["manifest"] == {"project_id": "origin-proj"}  # unpacked from the bytes
    assert not Path(seen["bundle_dir"]).exists()  # private temp dir cleaned up


def test_allow_unsigned_is_an_explicit_opt_out(engine):
    assert _post(_archive(), allow_unsigned="true").status_code == 200
    assert engine.seen["allow_unsigned"] is True


def test_hostile_archive_refused_typed_before_import(engine):
    def evil(tar):
        info = tarfile.TarInfo("../../etc/cron.d/x")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))

    resp = _post(_archive(extra=evil))
    assert (resp.status_code, resp.headers.get(ERROR_CODE_HEADER)) == (400, "bundle.bad_request")
    engine.import_project.assert_not_called()


def test_garbage_bytes_refused(engine):
    resp = _post(b"definitely not a tarball")
    assert resp.status_code == 400
    engine.import_project.assert_not_called()


def test_agent_import_needs_a_proposal(engine):
    resp = _client().post(
        "/umrs/knowledge/import",
        files={"file": ("b.kb.tar.gz", _archive(), "application/gzip")},
        headers={"X-Project-ID": "caller-proj", "x-r4-actor": "agent", "x-r4-session-id": "s"},
    )
    assert resp.status_code == 409, resp.text
    assert "proposal_required" in resp.text
    engine.import_project.assert_not_called()


# ── export: caller's project only; archive published under its prefix ───────


def _export_client(monkeypatch):
    from umrs.api import knowledge as kmod

    runtime = SimpleNamespace(submit_job_sync=AsyncMock(return_value="job-9"))
    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: runtime)
    app = FastAPI()
    app.include_router(kmod.router)
    return TestClient(app, raise_server_exceptions=False), runtime


def test_export_of_another_project_refused(monkeypatch):
    client, runtime = _export_client(monkeypatch)
    resp = client.post(
        "/umrs/knowledge/export",
        json={"project_id": "someone-else"},
        headers={"X-Project-ID": "caller-proj"},
    )
    assert (resp.status_code, resp.headers.get(ERROR_CODE_HEADER)) == (
        409,
        "knowledge.project_mismatch",
    )
    runtime.submit_job_sync.assert_not_called()


def test_export_returns_the_download_key(monkeypatch):
    client, _runtime = _export_client(monkeypatch)
    resp = client.post(
        "/umrs/knowledge/export",
        json={"project_id": "caller-proj"},
        headers={"X-Project-ID": "caller-proj"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["archive_key"] == "caller-proj/knowledge-exports/job-9.kb.tar.gz"


def _export_result(tmp_path, **over):
    from umrs.knowledge.bundle_export import KnowledgeExportResult

    bundle = tmp_path / "proj-a"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    return KnowledgeExportResult(bundle_path=str(bundle), project_id="proj-a", **over)


def test_publish_uploads_under_the_project_prefix(tmp_path, monkeypatch):
    from umrs.jobs.knowledge_adapters import KnowledgeExportBackend

    uploads = []
    client = SimpleNamespace(upload_file=lambda _path, bucket, key: uploads.append((bucket, key)))
    config = SimpleNamespace(bucket="umrs", get_client=lambda: client)
    monkeypatch.setattr("umrs.shared.storage.s3_config.get_s3_config", lambda: config)
    result = _export_result(tmp_path)
    KnowledgeExportBackend._publish_archive(result, "job-9")
    assert uploads == [("umrs", "proj-a/knowledge-exports/job-9.kb.tar.gz")]
    assert (result.archive_key, result.degraded) == (
        "proj-a/knowledge-exports/job-9.kb.tar.gz",
        False,
    )


def test_publish_failure_is_named_degraded(tmp_path, monkeypatch):
    from umrs.jobs.knowledge_adapters import KnowledgeExportBackend

    def boom():
        raise RuntimeError("s3 down")

    monkeypatch.setattr(
        "umrs.shared.storage.s3_config.get_s3_config",
        lambda: SimpleNamespace(bucket="b", get_client=boom),
    )
    result = _export_result(tmp_path)
    KnowledgeExportBackend._publish_archive(result, "job-9")
    assert (result.archive_key, result.degraded) == ("", True)
    assert "archive publish failed: s3 down" in result.degraded_reason


def test_publish_skips_a_degraded_export(tmp_path):
    from umrs.jobs.knowledge_adapters import KnowledgeExportBackend

    result = _export_result(tmp_path, degraded=True, degraded_reason="pool down")
    KnowledgeExportBackend._publish_archive(result, "job-9")
    assert (result.archive_key, result.degraded_reason) == ("", "pool down")


def test_repeat_exports_get_distinct_specs(monkeypatch):
    """The spec-derived idempotency key deduped every later export of a
    project onto its first one ("Job already exists", live 2026-09-24)."""
    client, runtime = _export_client(monkeypatch)
    for _ in range(2):
        resp = client.post(
            "/umrs/knowledge/export",
            json={"project_id": "caller-proj"},
            headers={"X-Project-ID": "caller-proj"},
        )
        assert resp.status_code == 200, resp.text
    specs = [c.kwargs["spec"] for c in runtime.submit_job_sync.call_args_list]
    assert specs[0]["request_id"] != specs[1]["request_id"]
