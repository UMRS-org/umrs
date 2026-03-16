"""S3 connector — thin wrapper around boto3 for pipeline data I/O."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Supported image extensions for upload scanning
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".jsonl"}


class S3Connector:
    """Upload, download, and list objects in an S3-compatible store.

    Args:
        endpoint_url: MinIO / S3 endpoint (e.g. ``http://minio:9000``).
        access_key: S3 access key.
        secret_key: S3 secret key.
        bucket: Default bucket name.
    """

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str,
                 bucket: str):
        import boto3
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._ensure_bucket(bucket)

    # ── public API ───────────────────────────────────────────────────────

    def upload_file(self, local_path: str, key: Optional[str] = None,
                    bucket: Optional[str] = None) -> str:
        """Upload a single file.  Returns the S3 key."""
        bucket = bucket or self.bucket
        key = key or os.path.basename(local_path)
        self._client.upload_file(local_path, bucket, key)
        logger.info("Uploaded %s → s3://%s/%s", local_path, bucket, key)
        return key

    def upload_dir(self, local_dir: str, extensions: Optional[set] = None,
                   bucket: Optional[str] = None) -> List[str]:
        """Upload all matching files from a directory.  Returns list of keys."""
        bucket = bucket or self.bucket
        keys = []
        for path in sorted(Path(local_dir).rglob("*")):
            if not path.is_file():
                continue
            if extensions and path.suffix.lower() not in extensions:
                continue
            key = path.name
            self._client.upload_file(str(path), bucket, key)
            keys.append(key)
        logger.info("Uploaded %d file(s) from %s", len(keys), local_dir)
        return keys

    def download_file(self, key: str, local_path: str,
                      bucket: Optional[str] = None) -> str:
        """Download an object to a local path.  Returns the local path."""
        bucket = bucket or self.bucket
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        self._client.download_file(bucket, key, local_path)
        logger.info("Downloaded s3://%s/%s → %s", bucket, key, local_path)
        return local_path

    def list_objects(self, prefix: str = "",
                     bucket: Optional[str] = None) -> List[str]:
        """List object keys under a prefix."""
        bucket = bucket or self.bucket
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def make_s3_url(self, key: str, bucket: Optional[str] = None) -> str:
        """Return an ``s3://bucket/key`` URL."""
        bucket = bucket or self.bucket
        return f"s3://{bucket}/{key}"

    # ── internal ─────────────────────────────────────────────────────────

    def _ensure_bucket(self, bucket: str):
        try:
            self._client.head_bucket(Bucket=bucket)
        except Exception:
            try:
                self._client.create_bucket(Bucket=bucket)
                logger.info("Created bucket: %s", bucket)
            except Exception:
                pass  # May already exist or we lack permission
