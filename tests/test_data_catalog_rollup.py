"""Tests for data catalog rollup scoping."""

from umrs.data_catalog.rollup import _belongs_to_logical_path
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState


def test_belongs_to_logical_path_matches_project_and_modality():
    meta = _meta("proj-1/uploads/a.csv", "proj-1", "structured")

    assert _belongs_to_logical_path(meta, "proj-1/default/structured/") is True
    assert _belongs_to_logical_path(meta, "proj-1/default/image/") is False
    assert _belongs_to_logical_path(meta, "proj-2/default/structured/") is False


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
