"""S3/MinIO environment configuration (foundation layer).

Lives in shared/storage so foundation modules (jobs/*) can consume S3
settings without importing from the api/ layer. Moved out of
umrs.api.files (import-boundary repair, WP-09 EXEC wave tail).
"""

from __future__ import annotations

import os

import boto3
from botocore.config import Config as BotoConfig


class S3Config:
    """S3/MinIO configuration from environment."""

    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        self.access_key = os.getenv("S3_ACCESS_KEY", "")
        self.secret_key = os.getenv("S3_SECRET_KEY", "")
        self.region = os.getenv("S3_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET", "umrs-files")
        self.default_ttl = int(os.getenv("UMRS_PRESIGNED_URL_TTL_SEC", "3600"))
        self.max_ttl = int(os.getenv("UMRS_PRESIGNED_URL_MAX_TTL_SEC", "86400"))
        self.use_ssl = os.getenv("S3_USE_SSL", "false").lower() == "true"

    def get_client(self) -> boto3.client:
        """Create boto3 S3 client."""
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
            use_ssl=self.use_ssl,
            verify=self.use_ssl,
        )


_s3_config: S3Config | None = None


def get_s3_config() -> S3Config:
    """Get S3 configuration singleton."""
    global _s3_config
    if _s3_config is None:
        _s3_config = S3Config()
    return _s3_config
