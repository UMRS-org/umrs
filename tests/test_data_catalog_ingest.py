"""Tests for bounded data catalog ingestion."""

from __future__ import annotations

import io
from typing import Any

from botocore.exceptions import ClientError
from catalog_store_fake import FakeCatalogMetaStore
from umrs.data_catalog.constants import EXTRACTED_TEXT_PREFIX
from umrs.data_catalog.ingest import ingest_object
from umrs.data_catalog.schema import CatalogState


def test_ingest_object_reads_only_configured_sample():
    s3_key = "proj-1/uploads/big.txt"
    client = _CatalogS3Client({s3_key: b"abcdefghijklmnopqrstuvwxyz"})
    store = FakeCatalogMetaStore()

    metrics = ingest_object(
        client,
        "bucket",
        s3_key=s3_key,
        object_etag="abc",
        project_id="proj-1",
        owner="user-1",
        max_extract_bytes=4,
        meta_store=store,
    )

    assert client.object_ranges[s3_key] == ["bytes=0-3"]
    assert metrics["sampled_bytes"] == 4
    assert metrics["total_size"] == 26

    meta = store.get("proj-1", s3_key)
    assert meta is not None
    assert meta.state == CatalogState.READY
    assert "perception_skipped" in meta.tags
    assert "catalog_sample_truncated" in meta.tags

    extracted_keys = [key for key in client.objects if key.startswith(EXTRACTED_TEXT_PREFIX)]
    assert len(extracted_keys) == 1
    assert client.objects[extracted_keys[0]] == b"abcd"


class _CatalogS3Client:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.object_ranges: dict[str, list[str]] = {}

    def head_object(self, *, Bucket: str, Key: str):  # noqa: N803 - boto3 shape
        if Key not in self.objects:
            raise _missing_key(Key)
        return {"ContentLength": len(self.objects[Key]), "ETag": '"abc"'}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):  # noqa: N803
        if Key not in self.objects:
            raise _missing_key(Key)
        content = self.objects[Key]
        if Range is not None:
            self.object_ranges.setdefault(Key, []).append(Range)
            start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            content = content[start : end + 1]
        return {"Body": io.BytesIO(content)}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes | str,
        ContentType: str | None = None,
    ) -> dict[str, Any]:  # noqa: N803
        self.objects[Key] = Body.encode("utf-8") if isinstance(Body, str) else Body
        return {"ETag": '"abc"'}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _CatalogPaginator(self.objects)


class _CatalogPaginator:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803 - boto3 shape
        contents = [{"Key": key} for key in self.objects if key.startswith(Prefix)]
        return [{"Contents": contents}]


def _missing_key(key: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": f"missing {key}"}},
        "GetObject",
    )
