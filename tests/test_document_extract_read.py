"""Tests for document PDF extraction and extracted-text catalog reads."""

from __future__ import annotations

import io

from catalog_store_fake import FakeCatalogMetaStore
from umrs.data_catalog.constants import EXTRACTED_TEXT_PREFIX
from umrs.data_catalog.extract import extract_content
from umrs.data_catalog.ingest import ingest_object
from umrs.data_catalog.query import read_catalog_object, resolve_read_source
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState


def test_extract_pdf_tags_when_empty(monkeypatch):
    monkeypatch.setattr(
        "umrs.data_catalog.extract._extract_pdf_detailed",
        lambda _content: ("", ["extract_empty"]),
    )
    _summary, text, tags = extract_content("paper.pdf", b"%PDF-1.4\n")
    assert text == ""
    assert "extract_empty" in tags


def test_extract_pdf_success_has_no_error_tags(monkeypatch):
    monkeypatch.setattr(
        "umrs.data_catalog.extract._extract_pdf_detailed",
        lambda _content: ("GCMC simulation at 298 K and 1 bar.", []),
    )
    summary, text, tags = extract_content("paper.pdf", b"%PDF-1.4\n")
    assert "GCMC" in text
    assert "extract_empty" not in tags
    assert "GCMC" in summary


def test_resolve_read_source_auto_prefers_extracted_for_documents():
    meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/paper.pdf",
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        detected_modality="document",
        extracted_text_ref=f"{EXTRACTED_TEXT_PREFIX}paper.txt",
    )
    assert resolve_read_source(meta, "auto") == "extracted"
    assert resolve_read_source(meta, "raw") == "raw"


def test_read_catalog_object_can_read_extracted_text_sidecar():
    s3_key = "proj-1/uploads/paper.pdf"
    extracted_ref = f"{EXTRACTED_TEXT_PREFIX}encoded.txt"
    meta = CatalogObjectMeta(
        s3_key=s3_key,
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        detected_modality="document",
        extracted_text_ref=extracted_ref,
    )
    store = FakeCatalogMetaStore()
    store.upsert(meta, "proj-1")
    client = _SidecarS3Client(
        {
            s3_key: b"%PDF-1.4 binary",
            extracted_ref: b"GCMC simulation at 298 K and 1 bar.",
        }
    )

    payload = read_catalog_object(
        client,
        "bucket",
        store,
        s3_key=s3_key,
        project_id="proj-1",
        source="auto",
    )

    assert payload["read_source"] == "extracted"
    assert "GCMC simulation" in (payload["text"] or "")
    assert payload["extracted_text_ref"] == extracted_ref


def test_ingest_document_downloads_beyond_default_sample(monkeypatch):
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "false")
    monkeypatch.setenv("UMRS_DATA_CATALOG_ENABLE_LLM_TAGS", "false")

    s3_key = "proj-1/uploads/paper.pdf"
    pdf_body = b"%PDF-1.4\n" + (b"x" * 2_000_000)
    client = _TrackingS3Client({s3_key: pdf_body})
    store = FakeCatalogMetaStore()

    ingest_object(
        client,
        "bucket",
        s3_key=s3_key,
        object_etag="abc",
        project_id="proj-1",
        owner="user-1",
        max_extract_bytes=1024,
        meta_store=store,
    )

    assert client.max_read_length >= len(pdf_body)
    meta = store.get("proj-1", s3_key)
    assert meta is not None
    assert meta.detected_modality == "document"


class _SidecarS3Client:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def head_object(self, *, Bucket: str, Key: str):  # noqa: N803
        return {"ContentLength": len(self.objects[Key]), "ETag": '"abc"'}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):  # noqa: N803
        content = self.objects[Key]
        if Range is not None:
            start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
            content = content[int(start_raw) : int(end_raw) + 1]
        return {"Body": io.BytesIO(content)}


class _TrackingS3Client(_SidecarS3Client):
    def __init__(self, objects: dict[str, bytes]):
        super().__init__(objects)
        self.max_read_length = 0

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):  # noqa: N803
        if Range is not None:
            _start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
            self.max_read_length = max(self.max_read_length, int(end_raw) + 1)
        return super().get_object(Bucket=Bucket, Key=Key, Range=Range)

    def put_object(self, *, Bucket: str, Key: str, Body: bytes | str, **kwargs):  # noqa: N803
        data = Body.encode("utf-8") if isinstance(Body, str) else Body
        self.objects[Key] = data
        return {"ETag": '"abc"'}
