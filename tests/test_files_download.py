"""Unit tests for UMRS GET /umrs/files/download/{path:path} (Phase-2l).

Verifies the streaming semantics — S3 responses never leave the platform;
the route does its own S3 fetch and streams bytes to the caller.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from botocore.exceptions import ClientError


class _FakeBody(io.BytesIO):
    def read(self, n: int = -1) -> bytes:
        return super().read(n)


class _FakeS3:
    def __init__(self, payload: bytes, content_type: str = "text/csv"):
        self._payload = payload
        self._content_type = content_type
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def head_object(self, **kwargs):
        self.calls.append(("head_object", kwargs))
        return {"ContentType": self._content_type, "ContentLength": len(self._payload)}

    def get_object(self, **kwargs):
        self.calls.append(("get_object", kwargs))
        return {"Body": _FakeBody(self._payload)}


class _FakeS3Config:
    def __init__(self, s3):
        self.bucket = "r4-data"
        self._s3 = s3

    def get_client(self):
        return self._s3


@pytest.mark.asyncio
async def test_download_stream_roundtrip():
    from umrs.api.files import download_file_stream

    payload = b"a,b,c\n1,2,3\n"
    cfg = _FakeS3Config(_FakeS3(payload))

    resp = await download_file_stream(
        path="default/results/x/a.csv",
        bucket=None,
        project_id="default",
        config=cfg,  # type: ignore[arg-type]
    )

    # Streaming shape
    assert resp.media_type == "text/csv"
    assert resp.headers["content-disposition"].endswith('filename="a.csv"')
    assert resp.headers["content-length"] == str(len(payload))

    # Read the stream & verify bytes
    parts = []
    async for chunk in resp.body_iterator:
        parts.append(chunk)
    assert b"".join(parts) == payload

    # S3 was called on the SAME key server-side (no presigned URL in response),
    # namespaced under default.
    assert cfg._s3.calls[0][1] == {"Bucket": "r4-data", "Key": "default/results/x/a.csv"}
    assert cfg._s3.calls[1][0] == "get_object"


@pytest.mark.asyncio
async def test_download_404_on_missing_file():
    from umrs.api.files import download_file_stream

    class _FakeS3Client:
        def head_object(self, **_kwargs):
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    cfg = _FakeS3Config(_FakeS3Client())

    with pytest.raises(Exception) as exc:
        await download_file_stream(
            path="default/results/nope.csv",
            bucket=None,
            project_id="default",
            config=cfg,  # type: ignore[arg-type]
        )
    assert "404" in str(exc.value) or "File not found" in str(exc.value)
