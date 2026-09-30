"""Tests for catalog query scoring, ACL helpers, and bounded reads."""

import io

from catalog_store_fake import FakeCatalogMetaStore
from umrs.data_catalog.query import _allowed, _score_meta, read_catalog_object
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState


def test_allowed_by_acl_scope():
    meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/file.txt",
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
    )
    assert _allowed(meta, "proj-1") is True
    assert _allowed(meta, "proj-2") is False


def test_score_meta_matches_summary():
    meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/protein_results.csv",
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        summary="protein folding analysis",
        filename_signals=["protein_results.csv"],
    )
    assert _score_meta(meta, ["protein"]) >= 1.0


def test_read_catalog_object_uses_s3_range_request():
    s3_key = "proj-1/uploads/file.txt"
    meta = CatalogObjectMeta(
        s3_key=s3_key,
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
    )
    store = FakeCatalogMetaStore()
    store.upsert(meta, "proj-1")
    client = _RangeS3Client(s3_key, b"abcdef")

    payload = read_catalog_object(
        client,
        "bucket",
        store,
        s3_key=s3_key,
        project_id="proj-1",
        offset=2,
        length=3,
    )

    assert client.object_range == "bytes=2-4"
    assert payload["text"] == "cde"
    assert payload["length"] == 3
    assert payload["total_size"] == 6
    assert payload["truncated"] is True


class _RangeS3Client:
    def __init__(self, s3_key: str, content: bytes):
        self.s3_key = s3_key
        self.content = content
        self.object_range: str | None = None

    def head_object(self, *, Bucket: str, Key: str):
        assert Key == self.s3_key
        return {"ContentLength": len(self.content), "ETag": '"abc"'}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):
        assert Key == self.s3_key
        assert Range is not None
        self.object_range = Range
        start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
        start = int(start_raw)
        end = int(end_raw)
        return {"Body": io.BytesIO(self.content[start : end + 1])}
