"""Agent-navigable data catalog stored under ``_data_catalog/`` in S3."""

from .adapter import CatalogArtifactAdapter
from .artifacts import (
    artifact_id_for_meta,
    load_artifact_content,
    meta_to_artifact,
    modality_to_artifact_type,
    read_artifact_text,
)
from .ingest import ingest_object, write_pending_stub
from .paths import encode_s3_key, logical_path_for_object
from .schema import CatalogObjectMeta, CatalogState

__all__ = [
    "CatalogArtifactAdapter",
    "CatalogObjectMeta",
    "CatalogState",
    "artifact_id_for_meta",
    "encode_s3_key",
    "ingest_object",
    "load_artifact_content",
    "logical_path_for_object",
    "meta_to_artifact",
    "modality_to_artifact_type",
    "read_artifact_text",
    "write_pending_stub",
]
