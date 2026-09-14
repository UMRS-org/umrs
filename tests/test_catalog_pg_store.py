"""Tests for CatalogMetaStore with in-memory fake (unit) and optional Postgres."""

from __future__ import annotations

from catalog_store_fake import FakeCatalogMetaStore
from umrs.data_catalog.pg_store import derive_project_id, reconcile_project
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState


def _ready_meta(s3_key: str, project_id: str, summary: str) -> CatalogObjectMeta:
    return CatalogObjectMeta(
        s3_key=s3_key,
        object_etag="etag-1",
        state=CatalogState.READY,
        owner="user-1",
        acl_scope=project_id,
        updated_at="2026-01-01T00:00:00",
        summary=summary,
        deterministic_summary=summary,
        detected_modality="structured",
        tags=["csv"],
    )


def test_upsert_and_get_round_trip():
    store = FakeCatalogMetaStore()
    meta = _ready_meta("proj-1/uploads/a.csv", "proj-1", "alpha dataset")
    store.upsert(meta, "proj-1")

    loaded = store.get("proj-1", "proj-1/uploads/a.csv")
    assert loaded is not None
    assert loaded.summary == "alpha dataset"


def test_search_ranking_prefers_matching_summary():
    store = FakeCatalogMetaStore()
    store.upsert(_ready_meta("proj-1/uploads/a.csv", "proj-1", "alpha dataset"), "proj-1")
    store.upsert(_ready_meta("proj-1/uploads/b.csv", "proj-1", "unrelated"), "proj-1")

    hits = store.search("proj-1", "alpha", limit=5)
    assert hits
    assert hits[0].meta.s3_key == "proj-1/uploads/a.csv"


def test_delete_removes_row():
    store = FakeCatalogMetaStore()
    meta = _ready_meta("proj-1/uploads/a.csv", "proj-1", "alpha")
    store.upsert(meta, "proj-1")
    assert store.delete("proj-1", "proj-1/uploads/a.csv")
    assert store.get("proj-1", "proj-1/uploads/a.csv") is None


def test_derive_project_id_from_acl_scope():
    meta = CatalogObjectMeta.pending(
        s3_key="proj-1/uploads/x.csv",
        object_etag="e",
        owner="u",
        acl_scope="proj-1",
    )
    assert derive_project_id(meta) == "proj-1"


def test_reconcile_deletes_missing_s3_objects():
    store = FakeCatalogMetaStore()
    store.upsert(_ready_meta("proj-1/uploads/gone.csv", "proj-1", "gone"), "proj-1")
    store.upsert(_ready_meta("proj-1/uploads/live.csv", "proj-1", "live"), "proj-1")

    class _S3:
        def head_object(self, *, Bucket: str, Key: str):  # noqa: N803
            if Key.endswith("gone.csv"):
                raise OSError("missing")
            return {}

    result = reconcile_project(store, _S3(), "bucket", "proj-1")
    assert "proj-1/uploads/gone.csv" in (result.missing_keys or [])
    assert store.get("proj-1", "proj-1/uploads/live.csv") is not None


def test_backfill_parses_legacy_sidecar():
    from umrs.data_catalog.legacy_s3_store import write_meta

    class _S3:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs):  # noqa: N803
            self.objects[Key] = Body

        def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
            import io

            return {"Body": io.BytesIO(self.objects[Key])}

        def get_paginator(self, name: str):
            return self

        def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803
            keys = [k for k in self.objects if k.startswith(Prefix)]
            return [{"Contents": [{"Key": k} for k in keys]}]

    from umrs.data_catalog.legacy_s3_store import list_legacy_meta_keys, read_meta

    s3 = _S3()
    meta = _ready_meta("proj-1/uploads/legacy.csv", "proj-1", "legacy")
    write_meta(s3, "bucket", meta)
    store = FakeCatalogMetaStore()

    for meta_key in list_legacy_meta_keys(s3, "bucket", prefix="proj-1/"):
        from umrs.data_catalog.paths import decode_s3_key

        s3_key = decode_s3_key(
            meta_key.removeprefix("_data_catalog/objects/").removesuffix(".meta.json")
        )
        loaded = read_meta(s3, "bucket", s3_key)
        assert loaded is not None
        store.upsert(loaded, "proj-1")

    assert store.get("proj-1", "proj-1/uploads/legacy.csv") is not None


def test_like_literal_escapes_wildcards():
    """Project ids/prefixes must be literal in LIKE patterns: an ``_`` in a
    project id used to act as a single-char wildcard and over-match foreign
    rows at the SQL layer (review minor: pg_store catalog visibility)."""
    from umrs.data_catalog.pg_store import _like_literal

    assert _like_literal("my_proj") == "my\\_proj"
    assert _like_literal("50%") == "50\\%"
    assert _like_literal("a\\b") == "a\\\\b"
    assert _like_literal("plain") == "plain"
