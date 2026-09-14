"""S3 artifact storage helper for knowledge artifacts.

Put/get raw bytes at projects/{project_id}/knowledge/artifacts/{content_hash}.
Presigned URLs are internal-only — never exposed in API responses.

Key operations:
  - put_artifact: upload raw bytes, return content_hash
  - get_artifact: download by content_hash, return bytes
  - get_presigned_url: generate internal-only presigned URL (TTL 5 min)
  - artifact_exists: check if content exists by hash
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)

# Default TTL for presigned URLs (seconds)
_DEFAULT_PRESIGNED_TTL = 300  # 5 minutes


class ArtifactStorage:
    """S3 artifact storage for knowledge content-addressed files.

    Keys follow: projects/{project_id}/knowledge/artifacts/{content_hash}
    """

    def __init__(
        self,
        s3_client: BaseClient,  # boto3 client (sync)
        bucket: str,
    ) -> None:
        """Initialize storage with an S3 client and bucket name."""
        self._s3 = s3_client
        self._bucket = bucket

    @staticmethod
    def key(project_id: str, content_hash: str) -> str:
        """Build the S3 key for an artifact.

        Key pattern: projects/{project_id}/knowledge/artifacts/{content_hash}
        """
        return f"projects/{project_id}/knowledge/artifacts/{content_hash}"

    def put_artifact(
        self,
        project_id: str,
        content: bytes,
        mime_type: str = "",
        ttl_seconds: int = 0,  # 0 = no presigned URL returned
    ) -> tuple[str, int, str | None]:
        """Upload content to S3.

        Returns (content_hash, size_bytes, presigned_url_or_None).
        Uses SHA-256 for content-addressed storage (hash-based dedup).
        """
        content_hash = hashlib.sha256(content).hexdigest()
        key = self.key(project_id, content_hash)

        # Check if already exists (idempotency via content-hash dedup)
        head = self._head_object(key)
        if head:
            logger.debug(
                "Artifact already exists: %s (size=%d)",
                content_hash,
                head.get("content_length", 0),
            )
            return content_hash, head["content_length"], None

        # Upload
        extra_args: dict = {}
        if mime_type:
            extra_args["ContentType"] = mime_type

        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content,
            **extra_args,
        )
        logger.info("Artifact uploaded: %s (size=%d)", content_hash, len(content))

        # Generate presigned URL if requested
        presigned_url: str | None = None
        if ttl_seconds > 0:
            presigned_url = self._generate_presigned_url(key, ttl_seconds)

        return content_hash, len(content), presigned_url

    def get_artifact(self, project_id: str, content_hash: str) -> bytes | None:
        """Download artifact by content hash.

        Returns None if artifact not found.
        """
        key = self.key(project_id, content_hash)
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            body = response["Body"].read()
            logger.debug("Artifact retrieved: %s (size=%d)", content_hash, len(body))
            return body
        except self._s3.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                logger.debug("Artifact not found: %s", content_hash)
                return None
            raise

    def artifact_exists(self, project_id: str, content_hash: str) -> bool:
        """Check if an artifact exists by content hash."""
        key = self.key(project_id, content_hash)
        return self._head_object(key) is not None

    def delete_artifact(self, project_id: str, content_hash: str) -> bool:
        """Delete an artifact. Returns True if deleted, False if not found."""
        key = self.key(project_id, content_hash)
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
            logger.info("Artifact deleted: %s", content_hash)
            return True
        except self._s3.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                return False
            raise

    def get_presigned_url(
        self,
        project_id: str,
        content_hash: str,
        ttl_seconds: int = _DEFAULT_PRESIGNED_TTL,
    ) -> str | None:
        """Generate an internal-only presigned URL.

        Presigned URLs are NEVER issued to external clients — they exist
        solely so compute nodes can reach S3 without holding MinIO credentials.
        Default TTL: 300 seconds (5 minutes).
        """
        key = self.key(project_id, content_hash)
        return self._generate_presigned_url(key, ttl_seconds)

    def get_all_artifacts(self, project_id: str) -> list[tuple[str, int]]:
        """List all artifacts for a project.

        Returns list of (content_hash, size_bytes).
        """
        prefix = f"projects/{project_id}/knowledge/artifacts/"
        try:
            response = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
            artifacts = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                size = obj["Size"]
                # Extract content_hash from key tail
                content_hash = key.rsplit("/", 1)[-1]
                artifacts.append((content_hash, size))
            return artifacts
        except self._s3.exceptions.ClientError:
            return []

    def _head_object(self, key: str) -> dict | None:
        """HEAD request to check object existence."""
        try:
            response = self._s3.head_object(Bucket=self._bucket, Key=key)
            return {
                "content_length": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", ""),
            }
        except self._s3.exceptions.ClientError:
            return None

    def _generate_presigned_url(self, key: str, ttl_seconds: int) -> str:
        """Generate a presigned URL for an S3 GET object.

        Returns a URL valid for ttl_seconds (default 300 = 5 min).
        """
        return self._s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self._bucket,
                "Key": key,
            },
            ExpiresIn=ttl_seconds,
        )
