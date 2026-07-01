"""Tests for data catalog rollup scoping and batch regeneration."""

from __future__ import annotations

import json
from typing import Any

from umrs.data_catalog.constants import MANIFEST_KEY
from umrs.data_catalog.paths import dir_summary_key
from umrs.data_catalog.rollup import (
    _belongs_to_logical_path,
    regenerate_rollups_for_paths,
)
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState


def test_belongs_to_logical_path_matches_project_and_modality():
    meta = _meta("proj-1/uploads/a.csv", "proj-1", "structured")

    assert _belongs_to_logical_path(meta, "proj-1/default/structured/") is True
    assert _belongs_to_logical_path(meta, "proj-1/default/image/") is False
    assert _belongs_to_logical_path(meta, "proj-2/default/structured/") is False


def test_regenerate_rollups_for_paths_single_list_pass():
    """Batch rollup refresh lists metas once and writes manifest once."""
    path_a = "proj-1/default/structured/"
    path_b = "proj-1/default/document/"
    client = _RecordingS3(
        {
            "_data_catalog/objects/proj-1%2Fuploads%2Fa.csv.meta.json": _meta(
                "proj-1/uploads/a.csv", "proj-1", "structured"
            ),
            "_data_catalog/objects/proj-1%2Fuploads%2Fb.pdf.meta.json": _meta(
                "proj-1/uploads/b.pdf", "proj-1", "document"
            ),
        }
    )

    regenerate_rollups_for_paths(client, "bucket", {path_a, path_b})

    assert client.list_calls == 1
    assert client.manifest_writes == 1
    assert dir_summary_key(path_a) in client.put_keys
    assert dir_summary_key(path_b) in client.put_keys


class _RecordingS3:
    def __init__(self, meta_objects: dict[str, CatalogObjectMeta]) -> None:
        self.meta_objects = {
            key: json.dumps(meta.to_dict()).encode("utf-8")
            for key, meta in meta_objects.items()
        }
        self.objects: dict[str, bytes] = dict(self.meta_objects)
        self.list_calls = 0
        self.manifest_writes = 0
        self.put_keys: list[str] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803
        self.list_calls += 1
        contents = [
            {"Key": key}
            for key in self.objects
            if key.startswith(Prefix) and key.endswith(".meta.json")
        ]
        return [{"Contents": contents}]

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        import io

        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs: Any):  # noqa: N803
        self.objects[Key] = Body
        self.put_keys.append(Key)
        if Key == MANIFEST_KEY:
            self.manifest_writes += 1


def _meta(s3_key: str, acl_scope: str, modality: str) -> CatalogObjectMeta:
    return CatalogObjectMeta(
        s3_key=s3_key,
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope=acl_scope,
        updated_at="now",
        detected_modality=modality,
    )
