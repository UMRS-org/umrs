"""Legacy S3 .meta.json sidecar I/O — backfill only."""

from __future__ import annotations

import json
import logging
from typing import Any

from botocore.exceptions import ClientError

from .paths import meta_object_key
from .schema import CatalogObjectMeta

logger = logging.getLogger(__name__)


def read_meta(s3_client: Any, bucket: str, s3_key: str) -> CatalogObjectMeta | None:
    """Load a legacy catalog metadata sidecar, or None if missing."""
    key = meta_object_key(s3_key)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        payload = json.loads(response["Body"].read().decode("utf-8"))
        return CatalogObjectMeta.from_dict(payload)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise


def write_meta(s3_client: Any, bucket: str, meta: CatalogObjectMeta) -> None:
    """Persist a legacy catalog metadata sidecar (backfill / migration tooling only)."""
    body = json.dumps(meta.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=meta_object_key(meta.s3_key),
        Body=body,
        ContentType="application/json",
    )


def list_legacy_meta_keys(
    s3_client: Any, bucket: str, *, prefix: str | None = None
) -> list[str]:
    """List legacy ``.meta.json`` sidecar keys under ``_data_catalog/objects/``."""
    from .constants import OBJECTS_PREFIX

    list_prefix = OBJECTS_PREFIX
    if prefix:
        from .paths import meta_objects_list_prefix_for_s3_key_prefix

        list_prefix = meta_objects_list_prefix_for_s3_key_prefix(prefix)

    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if key and key.endswith(".meta.json"):
                keys.append(key)
    return keys
