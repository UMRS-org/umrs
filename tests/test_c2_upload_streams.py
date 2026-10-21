"""Cycle-2 umrs-api lane (2026-10-21): a project-namespace upload streams to
the object store instead of being held in memory whole.

Found live (gateway -> umrs, ``r4 files upload`` path): POST /gw/files/upload
of a 1 GiB file raised the UMRS process's peak RSS by ~1 GiB, because the
route did ``content = await file.read()`` before ``put_object``. nginx allows
any size on that location (``client_max_body_size 0``), and r4 never gets a
presigned URL (Phase 2l), so every dataset upload went through this path; a
dataset larger than the host's free memory would take down the process that
also runs the ZMQ plane and the durable runtime. The owned-bucket branch of
the same route already streamed (``_HashingReader`` + ``upload_fileobj``).
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from umrs.api import files

BUCKET = "test-bucket"
CHUNK = 1024 * 1024
SIZE = 6 * CHUNK + 123


class _StreamingS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[str] = []

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):  # noqa: N803 — boto3's name
        assert bucket == BUCKET
        self.calls.append("upload_fileobj")
        parts = []
        while chunk := fileobj.read(CHUNK):
            parts.append(chunk)
        self.objects[key] = b"".join(parts)
        self.extra = ExtraArgs

    def put_object(self, *, Bucket, Key, Body, **_kw):  # noqa: N803 — boto3's names
        assert Bucket == BUCKET
        self.calls.append("put_object")
        self.objects[Key] = Body
        return {"ETag": '"' + hashlib.md5(Body).hexdigest() + '"'}


class _Config:
    bucket = BUCKET

    def __init__(self, client) -> None:
        self._client = client

    def get_client(self):
        return self._client


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    s3 = _StreamingS3()
    whole_reads: list[int] = []
    ingests: list[tuple] = []
    original_read = UploadFile.read

    async def recording_read(self, size: int = -1) -> bytes:
        data = await original_read(self, size)
        if size is None or size < 0:
            whole_reads.append(len(data))
        return data

    monkeypatch.setattr(UploadFile, "read", recording_read)
    monkeypatch.setattr(files, "_fire_post_upload_ingest", lambda *a: ingests.append(a))
    app = FastAPI()
    app.include_router(files.router)
    app.dependency_overrides[files.get_project_id] = lambda: "pa"
    app.dependency_overrides[files.get_project_s3_config] = lambda: _Config(s3)
    return TestClient(app), s3, whole_reads, ingests


BODY = bytes(range(256)) * (SIZE // 256) + b"x" * (SIZE % 256)


@pytest.fixture
def uploaded(harness):
    client, s3, whole_reads, ingests = harness
    r = client.post(
        "/umrs/files/upload",
        files={"file": ("data.bin", BODY, "application/octet-stream")},
        data={"purpose": "uploads"},
    )
    assert r.status_code == 200, r.text
    return r.json(), s3, whole_reads, ingests


def test_project_upload_is_never_read_whole(uploaded) -> None:
    _body, s3, whole_reads, _ingests = uploaded
    assert whole_reads == [], "the route read the whole upload into memory"
    assert s3.calls == ["upload_fileobj"]


def test_streamed_upload_keeps_bytes_metadata_and_etag(uploaded) -> None:
    body, s3, _whole_reads, ingests = uploaded
    assert body["key"].startswith("pa/uploads/")
    assert s3.objects[body["key"]] == BODY
    assert body["size"] == len(BODY)
    assert s3.extra["Metadata"] == {"project_id": "pa", "filename": "data.bin"}
    # the ingest trigger still gets the content's md5 as the object etag
    assert ingests[0][3] == hashlib.md5(BODY).hexdigest()
