"""Tests for CatalogArtifactAdapter."""

from __future__ import annotations

import io
import json
from typing import Any

from umrs.data_catalog.adapter import CatalogArtifactAdapter
from umrs.data_catalog.artifacts import artifact_id_for_meta
from umrs.data_catalog.constants import EXTRACTED_TEXT_PREFIX
from umrs.data_catalog.paths import meta_object_key, meta_objects_list_prefix_for_s3_key_prefix
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState
from umrs.knowledge.verify import KnowledgeVerifier
from botocore.exceptions import ClientError


def test_list_and_get_meta_filters_by_project():
    csv_key = "proj-1/uploads/data/run.csv"
    other_key = "proj-2/uploads/data/run.csv"
    client = _AdapterS3(
        {
            csv_key: b"smiles,prop\nCCO,1.5\n",
            other_key: b"x,y\n1,2\n",
        },
        {
            csv_key: _ready_meta(csv_key, "proj-1", "structured", summary="qm9 dataset"),
            other_key: _ready_meta(other_key, "proj-2", "structured", summary="other"),
        },
    )
    adapter = CatalogArtifactAdapter(client, "bucket", "proj-1")

    metas = adapter.list_metas()
    assert [m.s3_key for m in metas] == [csv_key]
    assert adapter.get_meta(csv_key) is not None
    assert adapter.get_meta(other_key) is None


def test_search_and_read_text():
    csv_key = "proj-1/uploads/data/batch-3-yields.csv"
    extracted_key = f"{EXTRACTED_TEXT_PREFIX}proj-1%2Fuploads%2Fdata%2Fbatch-3-yields.csv.txt"
    client = _AdapterS3(
        {csv_key: b"raw,bytes\n"},
        {
            csv_key: _ready_meta(
                csv_key,
                "proj-1",
                "structured",
                summary="batch-3 yields csv",
                tags=["csv"],
                extracted_text_ref=extracted_key,
            ),
        },
        extra_objects={extracted_key: b"col1,col2\n1,2\n"},
    )
    adapter = CatalogArtifactAdapter(client, "bucket", "proj-1")

    hits = adapter.search("batch-3 yields")
    assert len(hits) == 1
    assert hits[0].s3_key == csv_key

    text = adapter.read_text(csv_key)
    assert "col1,col2" in text


def test_list_metas_uses_project_scoped_list_prefix():
    """LIST should be scoped to proj-1 meta keys, not the full catalog."""
    proj1_key = "proj-1/uploads/data/run.csv"
    proj2_key = "proj-2/uploads/data/run.csv"
    client = _AdapterS3(
        {
            proj1_key: b"a\n",
            proj2_key: b"b\n",
        },
        {
            proj1_key: _ready_meta(proj1_key, "proj-1", "structured", summary="one"),
            proj2_key: _ready_meta(proj2_key, "proj-2", "structured", summary="two"),
        },
    )
    adapter = CatalogArtifactAdapter(client, "bucket", "proj-1")
    adapter.list_metas()

    expected_prefix = meta_objects_list_prefix_for_s3_key_prefix("proj-1/")
    assert client.list_prefixes == [expected_prefix]


def test_get_artifact_uses_index_not_repeated_list():
    py_key = "proj-1/uploads/code/model.py"
    client = _AdapterS3(
        {py_key: b"import torch\n"},
        {py_key: _ready_meta(py_key, "proj-1", "code", summary="torch model")},
    )
    adapter = CatalogArtifactAdapter(client, "bucket", "proj-1")
    artifact_id = artifact_id_for_meta(client.metas[py_key])

    adapter.get_artifact(artifact_id)
    adapter.get_artifact(artifact_id)

    assert client.list_calls == 1


def test_knowledge_store_protocol_and_verifier():
    py_key = "proj-1/uploads/code/model.py"
    client = _AdapterS3(
        {py_key: b"import torch\nclass Model: pass\n"},
        {
            py_key: _ready_meta(
                py_key,
                "proj-1",
                "code",
                summary="torch model",
                tags=["python"],
            ),
        },
    )
    adapter = CatalogArtifactAdapter(client, "bucket", "proj-1")

    listed = adapter.list_artifacts(artifact_type="code")
    assert len(listed) == 1
    assert listed[0]["metadata"]["s3_key"] == py_key
    assert listed[0]["content"] == b""

    artifact_id = artifact_id_for_meta(client.metas[py_key])
    loaded = adapter.get_artifact(artifact_id)
    assert loaded is not None
    assert b"import torch" in loaded["content"]

    knowledge = adapter.get_extracted_knowledge(artifact_id)
    assert knowledge is not None
    assert "import torch" in knowledge["extracted_text"]

    verifier = KnowledgeVerifier(adapter)
    answer = verifier.verify("What code files do we have?")
    assert answer.answered
    assert answer.confidence > 0


class _AdapterS3:
    def __init__(
        self,
        objects: dict[str, bytes],
        metas: dict[str, CatalogObjectMeta],
        *,
        extra_objects: dict[str, bytes] | None = None,
    ) -> None:
        self.objects = dict(objects)
        if extra_objects:
            self.objects.update(extra_objects)
        self.metas = metas
        self.list_prefixes: list[str] = []
        self.list_calls = 0
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
        self.list_prefixes.append(Prefix)
        self.list_calls += 1
        contents = []
        for key in sorted(self.objects):
            if not key.startswith(Prefix) or not key.endswith(".meta.json"):
                continue
            contents.append({"Key": key})
        return [{"Contents": contents}]


def _ready_meta(
    s3_key: str,
    project_id: str,
    modality: str,
    *,
    summary: str,
    tags: list[str] | None = None,
    extracted_text_ref: str | None = None,
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
        tags=tags or [],
        extracted_text_ref=extracted_text_ref,
    )


def _no_such_key(key: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": f"missing {key}"}},
        "GetObject",
    )
