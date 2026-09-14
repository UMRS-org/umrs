"""Tests for best-effort generative perception in catalog ingestion."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from catalog_store_fake import FakeCatalogMetaStore
from umrs.data_catalog import perception
from umrs.data_catalog.ingest import ingest_object


@dataclass
class _FakeEndpoint:
    model: str = "Kimi-K2.7"
    metadata: dict[str, Any] = field(default_factory=lambda: {"model_type": "vision"})

    @property
    def litellm_url(self) -> str:
        return "http://fake:8000"


def test_run_perception_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "false")
    result = perception.run_perception(
        project_id="proj-1",
        filename="a.txt",
        modality="text",
        extracted_text="hello world",
    )
    assert result is None


def test_caption_image_layers_into_result(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_IMAGE_PERCEPTION", "true")
    monkeypatch.setattr(perception, "_find_vision_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception, "_chat_completion", lambda _ep, _payload, **_kwargs: "A bar chart of yields."
    )

    result = perception.run_perception(
        project_id="proj-1",
        filename="plot.png",
        modality="image",
        extracted_text="",
        image_bytes=b"\x89PNG\r\n\x1a\n fake",
    )
    assert result is not None
    assert result.kind == "vision_caption"
    assert result.text == "A bar chart of yields."


def test_image_perception_off_by_default(monkeypatch):
    monkeypatch.delenv("UMRS_DATA_CATALOG_ENABLE_IMAGE_PERCEPTION", raising=False)
    monkeypatch.setattr(perception, "_find_vision_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception, "_chat_completion", lambda _ep, _payload, **_kwargs: "should not be called"
    )

    result = perception.run_perception(
        project_id="proj-1",
        filename="plot.png",
        modality="image",
        extracted_text="",
        image_bytes=b"\x89PNG\r\n\x1a\n fake",
    )
    assert result is None


def test_image_perception_can_be_disabled_independently(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_IMAGE_PERCEPTION", "false")
    monkeypatch.setattr(perception, "_find_vision_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception, "_chat_completion", lambda _ep, _payload, **_kwargs: "should not be called"
    )

    result = perception.run_perception(
        project_id="proj-1",
        filename="plot.png",
        modality="image",
        extracted_text="",
        image_bytes=b"\x89PNG\r\n\x1a\n fake",
    )
    assert result is None


def test_summarize_text_on_by_default(monkeypatch):
    monkeypatch.delenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", raising=False)
    monkeypatch.setattr(perception, "_find_text_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception, "_chat_completion", lambda _ep, _payload, **_kwargs: "Run notes for batch 3."
    )

    result = perception.run_perception(
        project_id="proj-1",
        filename="notes.txt",
        modality="text",
        extracted_text="batch 3 yield 42%",
    )
    assert result is not None
    assert result.kind == "llm_summary"


def test_summarize_text_uses_text_endpoint(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setattr(perception, "_find_text_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception, "_chat_completion", lambda _ep, _payload, **_kwargs: "Run notes for batch 3."
    )

    result = perception.run_perception(
        project_id="proj-1",
        filename="notes.txt",
        modality="text",
        extracted_text="batch 3 yield 42%",
    )
    assert result is not None
    assert result.kind == "llm_summary"
    assert result.text == "Run notes for batch 3."


def test_run_perception_summarizes_code(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setattr(perception, "_find_text_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception,
        "_chat_completion",
        lambda _ep, _payload, **_kwargs: "PyTorch neural network for MOF yield prediction.",
    )

    source = "import torch\nclass YieldPredictor(torch.nn.Module):\n    pass\n"
    result = perception.run_perception(
        project_id="proj-1",
        filename="model.py",
        modality="code",
        extracted_text=source,
    )
    assert result is not None
    assert result.kind == "llm_summary"
    assert "pytorch" in result.text.lower() or "torch" in result.text.lower()


def test_run_perception_failure_returns_none(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setattr(perception, "_find_text_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception,
        "_chat_completion",
        lambda _ep, _payload, **_kwargs: None,
    )

    result = perception.run_perception(
        project_id="proj-1",
        filename="notes.txt",
        modality="text",
        extracted_text="batch 3 yield 42%",
    )
    assert result is None


def test_ingest_tags_perception_skipped_when_unavailable(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setattr(perception, "_usable_endpoints", lambda _pid: [])

    s3_key = "proj-1/uploads/notes.txt"
    client = _CatalogS3Client({s3_key: b"batch 3 yield 42%"})
    store = FakeCatalogMetaStore()

    ingest_object(
        client,
        "bucket",
        s3_key=s3_key,
        object_etag="abc",
        project_id="proj-1",
        owner="user-1",
        meta_store=store,
    )

    meta = store.get("proj-1", s3_key)
    assert meta.state.value == "ready"
    assert "perception_skipped" in meta.tags


def test_ingest_applies_vision_caption(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_IMAGE_PERCEPTION", "true")
    monkeypatch.setattr(perception, "_find_vision_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception,
        "_chat_completion",
        lambda _ep, _payload, **_kwargs: "Microscopy image of stained cells.",
    )

    s3_key = "proj-1/uploads/cells.png"
    client = _CatalogS3Client({s3_key: b"\x89PNG\r\n\x1a\n fake-image-bytes"})
    store = FakeCatalogMetaStore()

    ingest_object(
        client,
        "bucket",
        s3_key=s3_key,
        object_etag="abc",
        project_id="proj-1",
        owner="user-1",
        meta_store=store,
    )

    meta = store.get("proj-1", s3_key)
    assert meta.state.value == "ready"
    assert meta.detected_modality == "image"
    assert "vision_captioned" in meta.tags
    assert meta.perception_summary == "Microscopy image of stained cells."
    assert meta.summary == meta.perception_summary


def test_ingest_structured_keeps_deterministic_summary_when_llm_runs(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true")
    monkeypatch.setattr(perception, "_find_text_endpoint", lambda _pid: _FakeEndpoint())
    monkeypatch.setattr(
        perception,
        "_chat_completion",
        lambda _ep, _payload, **_kwargs: "Narrative about batch yields.",
    )

    csv_body = b"col_a,col_b\n1,2\n"
    s3_key = "proj-1/uploads/batch.csv"
    client = _CatalogS3Client({s3_key: csv_body})
    store = FakeCatalogMetaStore()

    ingest_object(
        client,
        "bucket",
        s3_key=s3_key,
        object_etag="abc",
        project_id="proj-1",
        owner="user-1",
        meta_store=store,
    )

    meta = store.get("proj-1", s3_key)
    assert meta.detected_modality == "structured"
    assert "llm_summarized" in meta.tags
    assert meta.perception_summary == "Narrative about batch yields."
    assert meta.summary == meta.deterministic_summary
    assert "col_a" in meta.deterministic_summary or "col_a" in meta.tags


class _CatalogS3Client:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)

    def head_object(self, *, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.objects:
            raise _missing_key(Key)
        return {"ContentLength": len(self.objects[Key]), "ETag": '"abc"'}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):  # noqa: N803
        if Key not in self.objects:
            raise _missing_key(Key)
        content = self.objects[Key]
        if Range is not None:
            start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
            content = content[int(start_raw) : int(end_raw) + 1]
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

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803
        contents = [{"Key": key} for key in self.objects if key.startswith(Prefix)]
        return [{"Contents": contents}]


def _missing_key(key: str):
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": f"missing {key}"}},
        "GetObject",
    )


def test_parse_tag_array_accepts_json_list():
    from umrs.data_catalog.perception import _parse_tag_array

    tags = _parse_tag_array('["batch-3", "gcmc", "mof"]')
    assert tags == ["batch-3", "gcmc", "mof"]
