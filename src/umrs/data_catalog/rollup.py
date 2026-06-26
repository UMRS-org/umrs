"""Directory rollups and manifest regeneration (MVP: end-of-job, no sharding)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any
from urllib.parse import unquote

from botocore.exceptions import ClientError

from .constants import CATALOG_PREFIX, MANIFEST_KEY, OBJECTS_PREFIX, SCHEMA_VERSION
from .paths import dir_summary_key
from .schema import CatalogObjectMeta, CatalogState

logger = logging.getLogger(__name__)


def regenerate_rollups(
    s3_client: Any,
    bucket: str,
    *,
    logical_path: str,
) -> None:
    """Rebuild ``summary.json`` for one logical path and refresh ``manifest.json``."""
    metas = _list_object_metas(s3_client, bucket)
    path_metas = [m for m in metas if _belongs_to_logical_path(m, logical_path)]

    by_modality = Counter(m.detected_modality for m in path_metas)
    by_state = Counter(m.state.value for m in path_metas)
    summary = {
        "logical_path": logical_path,
        "object_count": len(path_metas),
        "modalities": dict(by_modality),
        "states": dict(by_state),
        "sample_keys": [m.s3_key for m in path_metas[:20]],
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=dir_summary_key(logical_path),
        Body=json.dumps(summary, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    _write_manifest(s3_client, bucket, metas)


def _write_manifest(s3_client: Any, bucket: str, metas: list[CatalogObjectMeta]) -> None:
    by_state = Counter(m.state.value for m in metas)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "object_count": len(metas),
        "states": dict(by_state),
        "catalog_prefix": CATALOG_PREFIX,
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=MANIFEST_KEY,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _list_object_metas(s3_client: Any, bucket: str) -> list[CatalogObjectMeta]:
    metas: list[CatalogObjectMeta] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=OBJECTS_PREFIX):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if not key or not key.endswith(".meta.json"):
                continue
            try:
                response = s3_client.get_object(Bucket=bucket, Key=key)
                payload = json.loads(response["Body"].read().decode("utf-8"))
                metas.append(CatalogObjectMeta.from_dict(payload))
            except (ClientError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping corrupt catalog meta %s: %s", key, exc)
    return metas


def _belongs_to_logical_path(meta: CatalogObjectMeta, logical_path: str) -> bool:
    """Match ``{project}/{experiment}/{modality}/`` rollups without leaking projects."""
    parts = [unquote(part) for part in logical_path.strip("/").split("/") if part]
    if not parts:
        return True

    project_id = parts[0]
    if meta.acl_scope != project_id and not meta.s3_key.startswith(f"{project_id}/"):
        return False

    if len(parts) >= 3 and meta.detected_modality != parts[2]:
        return False

    return True
