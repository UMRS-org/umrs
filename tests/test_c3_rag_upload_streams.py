"""Cycle-3 hard lane, N4 (2026-10-21): a RAG upload is streamed, never held
in UMRS memory whole.

Found live (lane stack, repro ``.stab-cycle3/hard/repro/n4_rag_rss.py``,
peak RSS from VmHWM): POST /umrs/rag/upload of 1 GiB through the endpoint
facade (``UMRS_RAG_UPLOAD_FACADE=endpoint``) raised the UMRS process's peak
RSS by 1026 MiB, twice: the proxy did ``content = await file.read()`` before
posting it on. Through the default central facade, 400 MiB raised it by
412 MiB: knowledge_upload staged with ``await file.read()`` + put_object.
The gateway allows 10 GB on /gw/rag/. The RAG service itself must hold the
text to chunk it, so it now refuses an oversize part from its size first.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from umrs.api import knowledge_upload, rag, rag_service

CHUNK = 1024 * 1024
BODY = bytes(range(256)) * (5 * CHUNK // 256) + b"tail"


@pytest.fixture
def whole_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every whole-body ``UploadFile.read()`` the routes make."""
    reads: list[int] = []
    original = UploadFile.read

    async def recording_read(self: UploadFile, size: int = -1) -> bytes:
        data = await original(self, size)
        if size is None or size < 0:
            reads.append(len(data))
        return data

    monkeypatch.setattr(UploadFile, "read", recording_read)
    return reads


def _endpoint_app(monkeypatch: pytest.MonkeyPatch, forwarded: list[bytes]) -> TestClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(await request.aread())
        return httpx.Response(200, json={"document_id": "doc-1", "chunks": 1})

    real_client = httpx.AsyncClient

    def client(**kw: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kw)

    async def success(_request: Any, **kw: Any) -> rag.UploadResponse:
        return rag.UploadResponse(
            document_id=kw["result"]["document_id"],
            filename=kw["filename"],
            status="success",
            chunks=1,
            knowledge_nodes_minted=0,
        )

    monkeypatch.setenv("UMRS_RAG_UPLOAD_FACADE", "endpoint")
    monkeypatch.setattr(rag.httpx, "AsyncClient", client)
    monkeypatch.setattr(rag, "_require_rag_scope", lambda *_a, **_k: ("alice", "pa"))
    monkeypatch.setattr(rag, "_discover_rag_endpoint", lambda *_a: "http://rag")
    monkeypatch.setattr(rag, "_upload_success", success)
    monkeypatch.setattr("umrs.api.governance.guard_ingest", lambda *_a, **_k: None)
    app = FastAPI()
    app.include_router(rag.router)
    app.dependency_overrides[rag.get_registry] = lambda: object()
    return TestClient(app)


def test_endpoint_facade_streams_the_part_to_the_rag_service(
    monkeypatch: pytest.MonkeyPatch, whole_reads: list[int]
) -> None:
    forwarded: list[bytes] = []
    client = _endpoint_app(monkeypatch, forwarded)
    r = client.post("/umrs/rag/upload", files={"file": ("big.txt", BODY, "text/plain")})
    assert r.status_code == 200, r.text
    assert whole_reads == [], "the proxy read the whole upload into memory"
    assert len(forwarded) == 1 and BODY in forwarded[0]


class _Part:
    """An UploadFile-like part whose whole-body read is a test failure."""

    def __init__(self, data: bytes, size: int | None) -> None:
        self.file = io.BytesIO(data)
        self.size = size
        self.filename = "notes.txt"
        self.content_type = "text/plain"

    async def read(self, _size: int = -1) -> bytes:
        raise AssertionError("the part was read whole")


class _StreamingS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.largest_read = 0

    def upload_fileobj(self, fileobj: Any, _bucket: str, key: str, **_kw: Any) -> None:
        parts = []
        while chunk := fileobj.read(CHUNK):
            self.largest_read = max(self.largest_read, len(chunk))
            parts.append(chunk)
        self.objects[key] = b"".join(parts)

    def delete_object(self, **kw: str) -> None:
        self.deleted.append(kw["Key"])


def test_central_staging_streams_and_still_hashes() -> None:
    s3 = _StreamingS3()
    digest, size = asyncio.run(
        knowledge_upload._stage_upload_bytes(
            _Part(BODY, len(BODY)), bucket="b", staged_key="k", client=s3
        )
    )
    assert s3.objects["k"] == BODY
    assert s3.largest_read <= CHUNK
    assert (digest, size) == (hashlib.sha256(BODY).hexdigest(), len(BODY))


def test_central_staging_refuses_a_part_of_unknown_size_once_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(knowledge_upload, "_KNOWLEDGE_UPLOAD_MAX_BYTES", 2 * CHUNK)
    s3 = _StreamingS3()
    with pytest.raises(HTTPException) as err:
        asyncio.run(
            knowledge_upload._stage_upload_bytes(
                _Part(BODY, None), bucket="b", staged_key="k", client=s3
            )
        )
    assert err.value.status_code == 413
    assert s3.deleted == ["k"]


def test_rag_service_refuses_an_oversize_part_before_reading_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UMRS_RAG_UPLOAD_MAX_BYTES", str(CHUNK))
    with pytest.raises(HTTPException) as err:
        asyncio.run(rag_service._read_upload_text(_Part(BODY, len(BODY))))
    assert err.value.status_code == 413
