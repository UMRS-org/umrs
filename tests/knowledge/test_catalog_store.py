"""Tests for CatalogKnowledgeStore."""

from __future__ import annotations

import io
import json
from typing import Any

from botocore.exceptions import ClientError
from catalog_store_fake import FakeCatalogMetaStore
from umrs.data_catalog.adapter import CatalogArtifactAdapter
from umrs.data_catalog.artifacts import artifact_id_for_meta
from umrs.data_catalog.paths import meta_object_key
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState
from umrs.knowledge.catalog_store import CatalogKnowledgeStore


def _adapter(client: Any, project_id: str) -> CatalogArtifactAdapter:
    """Build an adapter with a fake Postgres-SoT meta store seeded from S3 metas."""
    store = FakeCatalogMetaStore()
    for meta in client.metas.values():
        store.upsert(meta, project_id)
    return CatalogArtifactAdapter(client, "bucket", project_id, meta_store=store)


def test_extracted_knowledge_overlay():
    py_key = "proj-1/uploads/code/model.py"
    client = _StoreS3(
        {py_key: b"import torch\n"},
        {py_key: _ready_meta(py_key, "proj-1", "code", summary="torch model")},
    )
    store = CatalogKnowledgeStore(_adapter(client, "proj-1"))

    store.add_extracted_knowledge("art-1", {"summary": "overlay summary"})
    assert store.get_extracted_knowledge("art-1") == {"summary": "overlay summary"}


def test_extracted_knowledge_persisted_to_catalog():
    py_key = "proj-1/uploads/code/model.py"
    meta = _ready_meta(py_key, "proj-1", "code", summary="torch model")
    client = _StoreS3({py_key: b"import torch\n"}, {py_key: meta})
    meta_store = FakeCatalogMetaStore()
    meta_store.upsert(meta, "proj-1")
    adapter = CatalogArtifactAdapter(client, "bucket", "proj-1", meta_store=meta_store)
    store = CatalogKnowledgeStore(adapter)
    artifact_id = artifact_id_for_meta(meta)
    knowledge = {
        "artifact_id": artifact_id,
        "artifact_type": "code",
        "filename": "model.py",
        "objective": "",
        "algorithms": ["train"],
        "extraction_confidence": 0.8,
        "coverage_score": 0.5,
    }

    store.add_extracted_knowledge(artifact_id, knowledge)
    store._extracted_knowledge.clear()

    loaded = store.get_extracted_knowledge(artifact_id)
    assert loaded is not None
    assert loaded["algorithms"] == ["train"]

    # Under the Postgres SoT, the knowledge ref is recorded on the catalog
    # meta store row, not the legacy S3 .meta.json sidecar.
    persisted = meta_store.get("proj-1", py_key)
    assert persisted is not None
    assert persisted.extracted_knowledge_ref


def test_refresh_picks_up_new_catalog_object():
    csv_key = "proj-1/uploads/data/run.csv"
    client = _StoreS3(
        {csv_key: b"x,y\n1,2\n"},
        {csv_key: _ready_meta(csv_key, "proj-1", "structured", summary="csv")},
    )
    meta_store = FakeCatalogMetaStore()
    for meta in client.metas.values():
        meta_store.upsert(meta, "proj-1")
    adapter = CatalogArtifactAdapter(client, "bucket", "proj-1", meta_store=meta_store)
    store = CatalogKnowledgeStore(adapter)
    assert len(store.list_artifacts()) == 1

    py_key = "proj-1/uploads/code/model.py"
    new_meta = _ready_meta(py_key, "proj-1", "code", summary="torch")
    client.objects[py_key] = b"import torch\n"
    client.metas[py_key] = new_meta
    client.objects[meta_object_key(py_key)] = json.dumps(
        new_meta.to_dict(),
        indent=2,
    ).encode("utf-8")
    # Postgres SoT is now authoritative for the catalog listing.
    meta_store.upsert(new_meta, "proj-1")

    assert len(store.list_artifacts()) == 1
    store.refresh()
    assert len(store.list_artifacts()) == 2


class _StoreS3:
    def __init__(
        self,
        objects: dict[str, bytes],
        metas: dict[str, CatalogObjectMeta],
    ) -> None:
        self.objects = dict(objects)
        self.metas = metas
        for s3_key, meta in metas.items():
            self.objects[meta_object_key(s3_key)] = json.dumps(
                meta.to_dict(),
                indent=2,
            ).encode("utf-8")

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self.objects:
            raise _no_such_key(Key)
        return {"ContentLength": len(self.objects[Key]), "ETag": '"abc"'}

    def get_object(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        Range: str | None = None,
    ) -> dict[str, Any]:
        if Key not in self.objects:
            raise _no_such_key(Key)
        content = self.objects[Key]
        if Range is not None:
            start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            content = content[start : end + 1]
        return {"Body": io.BytesIO(content)}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803
        contents = []
        for key in sorted(self.objects):
            if not key.startswith(Prefix) or not key.endswith(".meta.json"):
                continue
            contents.append({"Key": key})
        return [{"Contents": contents}]

    def put_object(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        Body: bytes | str,
        ContentType: str | None = None,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.objects[Key] = Body.encode("utf-8") if isinstance(Body, str) else Body
        return {"ETag": '"new"'}


def _ready_meta(
    s3_key: str,
    project_id: str,
    modality: str,
    *,
    summary: str,
) -> CatalogObjectMeta:
    return CatalogObjectMeta(
        s3_key=s3_key,
        object_etag="etag-1",
        state=CatalogState.READY,
        owner="user-1",
        acl_scope=project_id,
        updated_at="2026-01-01T00:00:00",
        summary=summary,
        deterministic_summary=summary,
        detected_modality=modality,
    )


def _no_such_key(key: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": f"missing {key}"}},
        "GetObject",
    )
