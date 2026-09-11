"""WP-15 dual-corpus dwell pins: central upload + facade switch (fail-without-change)."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.files import get_project_id
from umrs.api.knowledge_upload import router


class _FakeS3Client:
    def __init__(self, fail: bool = False):
        self.puts: list[dict] = []
        self.fail = fail

    def put_object(self, **kwargs):
        if self.fail:
            raise RuntimeError("minio wedged")
        self.puts.append(kwargs)


def _app(monkeypatch, *, s3=None, runtime=None):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_project_id] = lambda: "proj-9"

    fake_config = SimpleNamespace(
        bucket="umrs-files",
        get_client=lambda: s3 or _FakeS3Client(),
    )
    monkeypatch.setattr(
        "umrs.api.knowledge_upload.get_s3_config", lambda: fake_config
    )
    monkeypatch.setattr(
        "umrs.api.knowledge_upload._resolve_ingest_target", lambda *a, **k: "graph-default"
    )
    monkeypatch.setattr(
        "umrs.api.knowledge_upload.enforce_graph_mutation_authorization",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("umrs.api.knowledge_upload.guard_ingest", lambda *a, **k: None)

    class _Runtime:
        def __init__(self):
            self.calls = []

        async def submit_job_sync(self, **kwargs):
            self.calls.append(kwargs)
            return runtime or "job-central-1"

    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: _Runtime())
    return app


def _post(app, monkeypatch, filename="note.pdf", data=b"hello-bytes"):
    client = TestClient(app, raise_server_exceptions=False)
    return client.post(
        "/umrs/knowledge/upload",
        files={"file": (filename, io.BytesIO(data), "application/pdf")},
        headers={"x-user-id": "u-1"},
    )


def test_path_traversal_sanitized_and_staged_under_prefix(monkeypatch):
    s3 = _FakeS3Client()
    _post(_app(monkeypatch, s3=s3), monkeypatch, filename="../../etc/passwd")
    assert s3.puts, "no staging call"
    key = s3.puts[0]["Key"]
    assert key.startswith("knowledge-uploads/proj-9/")
    assert "../" not in key.split("/", 3)[-1]  # suffix segment carries no traversal
    assert key.endswith("passwd")


def test_ingest_spec_carries_s3_uri_and_graph(monkeypatch):
    s3 = _FakeS3Client()
    resp = _post(_app(monkeypatch, s3=s3), monkeypatch)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["job_id"] == "job-central-1"
    assert body["s3_key"].startswith("knowledge-uploads/proj-9/")
    assert body["graph_id"] == "graph-default"
    assert body["dwell_contract"].startswith("central")


def test_s3_fault_is_503_never_silent_fallback(monkeypatch):
    resp = _post(_app(monkeypatch, s3=_FakeS3Client(fail=True)), monkeypatch)
    assert resp.status_code == 503


def test_oversize_is_413(monkeypatch):
    resp = _post(
        _app(monkeypatch), monkeypatch,
        filename="big.bin", data=b"x" * (512 * 1024 * 1024 + 1),
    )
    assert resp.status_code == 413


def test_missing_filename_is_400(monkeypatch):
    from io import BytesIO

    app = _app(monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/umrs/knowledge/upload",
        files={"file": ("", BytesIO(b"x"), "application/octet-stream")},
        headers={"x-user-id": "u-1"},
    )
    assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# rag facade switch — dual-corpus dwell defaulting to central
# ---------------------------------------------------------------------------


class TestRagFacadeDwell:
    def _file(self, name="x.pdf"):
        return io.BytesIO(b"bytes"), {"file": (name, io.BytesIO(b"bytes"), "application/pdf")}

    def test_default_routes_to_central(self, monkeypatch):
        monkeypatch.delenv("UMRS_RAG_UPLOAD_FACADE", raising=False)
        from umrs.api import rag

        called = {}

        async def fake_knowledge_upload(*, request, file, graph_id, project_id):
            called["graph_id"] = graph_id
            called["project_id"] = project_id
            return {"job_id": "job-central-1"}

        monkeypatch.setattr(
            "umrs.api.rag._require_rag_scope", lambda *_a, **_k: ("u-1", "proj-9")
        )
        # Facade's inner "from .knowledge_upload import knowledge_upload"
        # resolves the attribute on this module — patch there.
        monkeypatch.setattr(
            "umrs.api.knowledge_upload.knowledge_upload", fake_knowledge_upload
        )

        import asyncio

        resp = asyncio.run(
            rag.upload_document(MagicMock(), MagicMock(filename="x.pdf"), registry=None)
        )
        assert called["project_id"] == "proj-9"
        assert resp.document_id == "job-central-1"
        assert resp.status == "queued-central"

    def test_endpoint_optin_keeps_legacy(self, monkeypatch):
        monkeypatch.setenv("UMRS_RAG_UPLOAD_FACADE", "endpoint")
        from umrs.api import rag

        monkeypatch.setattr(
            "umrs.api.rag._require_rag_scope", lambda *_a, **_k: ("u-1", "proj-9")
        )
        monkeypatch.setattr(
            "umrs.api.rag._discover_rag_endpoint", lambda *_a, **_k: "http://rag-svc"
        )

        class _FakeClientCls:
            def __init__(self, *a, **k):
                self.responses = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kwargs):
                resp = MagicMock(status_code=200)
                resp.json = lambda: {"document_id": "doc-1", "chunks": 3}
                return resp

        monkeypatch.setattr("umrs.api.rag.httpx.AsyncClient", _FakeClientCls)

        import asyncio
        from unittest.mock import AsyncMock

        request = MagicMock()
        file = MagicMock(filename="x.pdf", content_type="application/pdf")
        file.read = AsyncMock(return_value=b"bytes")
        resp = asyncio.run(rag.upload_document(request, file, registry=None))
        assert resp.document_id == "doc-1"
