"""Tests for catalog scoring and structured search filters."""

from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState
from umrs.data_catalog.scoring import score_meta

from catalog_store_fake import FakeCatalogMetaStore


def test_score_meta_boosts_tag_and_ready_hits():
    meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/batch-3-yields.csv",
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        summary="csv data",
        detected_modality="structured",
        tags=["yield_pct", "batch_id"],
        filename_signals=["batch-3-yields.csv", "batch", "3", "yields"],
    )
    score = score_meta(meta, ["yield"], query="csv yield columns")
    assert score >= 2.0


def test_fake_store_filter_by_modality_and_tag():
    store = FakeCatalogMetaStore()
    csv_meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/batch.csv",
        object_etag="abc",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        detected_modality="structured",
        tags=["batch_id"],
    )
    pdf_meta = CatalogObjectMeta(
        s3_key="proj-1/uploads/paper.pdf",
        object_etag="def",
        state=CatalogState.READY,
        owner="user",
        acl_scope="proj-1",
        updated_at="now",
        detected_modality="document",
        tags=["paper"],
    )
    store.upsert(csv_meta, "proj-1")
    store.upsert(pdf_meta, "proj-1")

    hits = store.search("proj-1", "", modality="structured", tag="batch_id")
    assert len(hits) == 1
    assert hits[0].meta.s3_key.endswith("batch.csv")
