"""S3 read/write helpers for catalog blobs (extracted text, knowledge, raw bytes)."""

from __future__ import annotations

import json
import logging
from typing import Any

from botocore.exceptions import ClientError

from .constants import EXTRACTED_KNOWLEDGE_PREFIX, EXTRACTED_TEXT_PREFIX
from .paths import encode_s3_key

logger = logging.getLogger(__name__)


def write_extracted_text(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    text: str,
) -> str:
    """Store extracted text blob; return its S3 key ref."""
    ref = f"{EXTRACTED_TEXT_PREFIX}{encode_s3_key(s3_key)}.txt"
    s3_client.put_object(
        Bucket=bucket,
        Key=ref,
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )
    return ref


def write_structured_knowledge(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    knowledge: dict[str, Any],
) -> str:
    """Store Phase 4 ``ExtractedKnowledge`` JSON; return its S3 key ref."""
    ref = f"{EXTRACTED_KNOWLEDGE_PREFIX}{encode_s3_key(s3_key)}.json"
    body = json.dumps(knowledge, indent=2, sort_keys=True).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=ref,
        Body=body,
        ContentType="application/json",
    )
    return ref


def read_structured_knowledge(
    s3_client: Any,
    bucket: str,
    ref: str,
) -> dict[str, Any] | None:
    """Load persisted Phase 4 structured knowledge from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=ref)
        payload = json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Failed to decode structured knowledge at %s: %s", ref, exc)
        return None
    return payload if isinstance(payload, dict) else None


def get_object_size(s3_client: Any, bucket: str, s3_key: str) -> int:
    """Return object size in bytes."""
    response = s3_client.head_object(Bucket=bucket, Key=s3_key)
    return int(response.get("ContentLength", 0) or 0)


def get_object_range(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    *,
    offset: int,
    length: int,
) -> tuple[bytes, int]:
    """
    Read a bounded byte range and return ``(content, total_size)``.

    The head request keeps empty/out-of-range reads cheap and avoids asking S3
    for invalid ranges.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if length < 0:
        raise ValueError("length must be non-negative")

    total_size = get_object_size(s3_client, bucket, s3_key)
    if length == 0 or total_size == 0 or offset >= total_size:
        return b"", total_size

    end = min(total_size - 1, offset + length - 1)
    response = s3_client.get_object(
        Bucket=bucket,
        Key=s3_key,
        Range=f"bytes={offset}-{end}",
    )
    return response["Body"].read(), total_size


def get_object_etag(s3_client: Any, bucket: str, s3_key: str) -> str:
    """Head object and return normalized etag."""
    response = s3_client.head_object(Bucket=bucket, Key=s3_key)
    return _normalize_etag(response.get("ETag", ""))


def _normalize_etag(etag: str) -> str:
    return etag.strip('"')
