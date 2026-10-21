"""Cycle-2 umrs-api lane (2026-10-21): an upload over its size cap is
refused from the multipart part's size, before its bytes are read into
memory.

Found live through the gateway: POST /gw/knowledge/upload of a 700 MiB file
answered the right 413 ("Upload exceeds 512 MiB limit") but only after
``await file.read()``: peak RSS +683 MiB. POST /gw/workflows/upload of a
600 MiB zip did the same (+610 MiB) against its 500 MiB cap. nginx allows
any size on /gw/knowledge/ and 10 GB on /gw/workflows/, so the refusal
path itself could exhaust memory. Starlette has spooled the part to disk
and knows its size; the routes now check that first, and the workflow zip
is copied to its temp file in chunks.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import HTTPException
from umrs.api import knowledge_upload, workflows


class _SpooledPart:
    """An UploadFile-like part whose whole-body read is a test failure."""

    def __init__(self, size: int, data: bytes = b"") -> None:
        self.size = size
        self.filename = "part.bin"
        self.content_type = "application/octet-stream"
        self.file = io.BytesIO(data)
        self.whole_reads = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            self.whole_reads += 1
        return self.file.read(size)


class _NoS3:
    def put_object(self, **_kw):
        raise AssertionError("an oversize upload must not be staged")


def test_oversize_knowledge_upload_is_413_without_reading(monkeypatch) -> None:
    monkeypatch.setattr(knowledge_upload, "_KNOWLEDGE_UPLOAD_MAX_BYTES", 1024)
    part = _SpooledPart(size=4096)
    with pytest.raises(HTTPException) as err:
        asyncio.run(
            knowledge_upload._stage_upload_bytes(
                part, bucket="b", staged_key="k", client=_NoS3()
            )
        )
    assert err.value.status_code == 413
    assert part.whole_reads == 0


def test_oversize_workflow_zip_is_413_without_reading(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(workflows, "_MAX_WORKFLOW_UPLOAD_BYTES", 1024)
    part = _SpooledPart(size=4096, data=b"z" * 4096)
    with pytest.raises(HTTPException) as err:
        asyncio.run(workflows._spool_upload(part, str(tmp_path / "w.zip")))
    assert err.value.status_code == 413
    assert part.whole_reads == 0


def test_workflow_zip_is_copied_in_chunks(tmp_path) -> None:
    data = b"PK" + b"q" * (3 * 1024 * 1024)
    part = _SpooledPart(size=len(data), data=data)
    dest = tmp_path / "w.zip"
    asyncio.run(workflows._spool_upload(part, str(dest)))
    assert dest.read_bytes() == data
    assert part.whole_reads == 0
