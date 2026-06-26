"""S3 key helpers for the data catalog."""

from __future__ import annotations

from urllib.parse import quote, unquote

from .constants import OBJECTS_PREFIX


def encode_s3_key(s3_key: str) -> str:
    """URL-encode an S3 object key for use in catalog paths."""
    return quote(s3_key, safe="")


def decode_s3_key(encoded: str) -> str:
    """Decode a catalog-encoded S3 key."""
    return unquote(encoded)


def meta_object_key(s3_key: str) -> str:
    """S3 key for the per-object ``.meta.json`` sidecar."""
    return f"{OBJECTS_PREFIX}{encode_s3_key(s3_key)}.meta.json"


def encode_path_component(component: str) -> str:
    """Encode one logical path component (project/experiment/modality)."""
    return quote(component, safe="")


def logical_path_for_object(
    project_id: str,
    s3_key: str,
    modality: str,
    *,
    experiment: str = "default",
) -> str:
    """
    Build logical catalog path ``{project}/{experiment}/{modality}/``.

    Gateway uploads use ``{project_id}/...`` keys; experiment defaults to
    ``default`` until experience-aware routing is added.
    """
    return (
        f"{encode_path_component(project_id)}/"
        f"{encode_path_component(experiment)}/"
        f"{encode_path_component(modality)}/"
    )


def dir_summary_key(logical_path: str) -> str:
    """S3 key for a directory rollup ``summary.json``."""
    normalized = logical_path if logical_path.endswith("/") else f"{logical_path}/"
    return f"_data_catalog/dirs/{normalized}summary.json"
