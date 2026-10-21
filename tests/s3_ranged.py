"""A GET as S3 answers it — ranged when asked — for the in-memory S3 fakes.

Cycle-3 P1 (2026-10-22): the knowledge chain reads only the bytes it uses (a
ranged GET for understand, embed and retrieval) and copies staged uploads
server side, so the fakes answer ``Range`` and ``copy_object`` like a store.

Cycle-3 memfix #6: like the real store on the paths that code depends on
(checked against SeaweedFS 4.47 by ``test_s3_fake_parity.py``): a range whose
first byte is past the end — any range over an empty object — is a 416
``InvalidRange``; an ETag is the quoted MD5 of a single-part body; a copy
whose ``CopySourceIfMatch`` names another ETag is a 412 ``PreconditionFailed``
and stores nothing. The fakes answered 200 with ``bytes 0--1/0`` and copied
whatever the precondition said.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

from botocore.exceptions import ClientError


def etag_of(body: bytes) -> str:
    """A single-part object's ETag: its quoted MD5."""
    return f'"{hashlib.md5(body).hexdigest()}"'  # noqa: S324 — the S3 ETag, not a security hash


def _client_error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


def ranged(body: bytes, range_header: str = "") -> dict[str, Any]:
    """``get_object``'s answer for *body*, honouring ``Range: bytes=a-b``."""
    if not range_header:
        return {"Body": io.BytesIO(body), "ContentLength": len(body), "ETag": etag_of(body)}
    first, _, last = range_header.removeprefix("bytes=").partition("-")
    start = int(first)
    if start >= len(body):
        raise _client_error("InvalidRange", 416, "GetObject")
    part = body[start : int(last) + 1]
    return {
        "Body": io.BytesIO(part),
        "ContentLength": len(part),
        "ContentRange": f"bytes {start}-{start + len(part) - 1}/{len(body)}",
        "ETag": etag_of(body),
    }


def check_copy_precondition(source: bytes, kwargs: dict[str, Any]) -> None:
    """Raise as S3 does when ``CopySourceIfMatch`` is not *source*'s ETag."""
    wanted = kwargs.get("CopySourceIfMatch")
    if wanted and wanted != etag_of(source):
        raise _client_error("PreconditionFailed", 412, "CopyObject")


class FakeS3:
    """The smallest object store the parity test drives: one dict of
    ``(bucket, key) -> bytes``."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **_kw: Any) -> dict[str, Any]:  # noqa: N803
        self.objects[(Bucket, Key)] = Body
        return {"ETag": etag_of(Body)}

    def get_object(self, Bucket: str, Key: str, Range: str = "") -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", 404, "GetObject")
        return ranged(self.objects[(Bucket, Key)], Range)

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", 404, "HeadObject")
        body = self.objects[(Bucket, Key)]
        return {"ContentLength": len(body), "ETag": etag_of(body)}

    def copy_object(self, Bucket: str, Key: str, CopySource: dict, **kw: Any) -> dict[str, Any]:  # noqa: N803
        source = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        check_copy_precondition(source, kw)
        self.objects[(Bucket, Key)] = source
        return {"CopyObjectResult": {"ETag": etag_of(source)}}
