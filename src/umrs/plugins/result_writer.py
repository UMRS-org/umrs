"""
Plugin result writer (X-105).

Handles writing plugin outputs to approved destinations:
- S3/MinIO storage
- pgvector for embeddings
- UMRS internal storage
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class WriteResult:
    """Result of a write operation."""

    destination: str
    success: bool
    bytes_written: int = 0
    error: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OutputWriter(ABC):
    """Abstract base class for output writers."""

    @abstractmethod
    async def write(
        self,
        plugin_id: str,
        data: bytes | str | dict[str, Any],
        path: str,
        content_type: str | None = None,
    ) -> WriteResult:
        """Write data to destination."""
        ...

    @abstractmethod
    async def write_file(
        self,
        plugin_id: str,
        file_path: Path,
        dest_path: str,
    ) -> WriteResult:
        """Write file to destination."""
        ...


class S3OutputWriter(OutputWriter):
    """Write plugin outputs to S3/MinIO."""

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
    ):
        """Initialize S3 writer."""
        self.endpoint = endpoint or os.environ.get("S3_ENDPOINT", "http://minio:9000")
        self.access_key = access_key or os.environ.get("S3_ACCESS_KEY", "")
        self.secret_key = secret_key or os.environ.get("S3_SECRET_KEY", "")
        self.bucket = bucket or os.environ.get("PLUGIN_OUTPUT_BUCKET", "plugin-outputs")
        self._client = None

    def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config

                self._client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=Config(signature_version="s3v4"),
                )
            except ImportError:
                raise RuntimeError("boto3 required for S3 output writer") from None
        return self._client

    async def write(
        self,
        plugin_id: str,
        data: bytes | str | dict[str, Any],
        path: str,
        content_type: str | None = None,
    ) -> WriteResult:
        """Write data to S3."""
        client = self._get_client()

        # Prepare data
        if isinstance(data, dict):
            body = json.dumps(data).encode("utf-8")
            content_type = content_type or "application/json"
        elif isinstance(data, str):
            body = data.encode("utf-8")
            content_type = content_type or "text/plain"
        else:
            body = data
            content_type = content_type or "application/octet-stream"

        # Build key with plugin prefix
        key = f"plugins/{plugin_id}/{path}"

        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )

            return WriteResult(
                destination=f"s3://{self.bucket}/{key}",
                success=True,
                bytes_written=len(body),
            )
        except Exception as e:
            logger.exception(f"S3 write failed: {key}")
            return WriteResult(
                destination=f"s3://{self.bucket}/{key}",
                success=False,
                error=str(e),
            )

    async def write_file(
        self,
        plugin_id: str,
        file_path: Path,
        dest_path: str,
    ) -> WriteResult:
        """Write file to S3."""
        client = self._get_client()
        key = f"plugins/{plugin_id}/{dest_path}"

        try:
            with open(file_path, "rb") as f:
                client.upload_fileobj(f, self.bucket, key)

            return WriteResult(
                destination=f"s3://{self.bucket}/{key}",
                success=True,
                bytes_written=file_path.stat().st_size,
            )
        except Exception as e:
            logger.exception(f"S3 file upload failed: {key}")
            return WriteResult(
                destination=f"s3://{self.bucket}/{key}",
                success=False,
                error=str(e),
            )


class PgVectorOutputWriter(OutputWriter):
    """Write plugin embeddings to pgvector."""

    def __init__(
        self,
        connection_string: str | None = None,
    ):
        """Initialize pgvector writer."""
        self.connection_string = connection_string or os.environ.get(
            "PGVECTOR_URL", "postgresql://localhost:5432/vectors"
        )
        self._pool = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg

                from umrs.shared.pg_settings import asyncpg_pool_kwargs

                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    **asyncpg_pool_kwargs("plugin-result-writer"),
                )
            except ImportError:
                raise RuntimeError(
                    "asyncpg required for pgvector output writer"
                ) from None
        return self._pool

    async def write(
        self,
        plugin_id: str,
        data: bytes | str | dict[str, Any],
        path: str,
        content_type: str | None = None,
    ) -> WriteResult:
        """Write embeddings to pgvector."""
        pool = await self._get_pool()

        # Expect data to be a dict with embeddings
        if not isinstance(data, dict) or "embeddings" not in data:
            return WriteResult(
                destination=f"pgvector://{path}",
                success=False,
                error="Data must be dict with 'embeddings' key",
            )

        embeddings = data["embeddings"]
        texts = data.get("texts", [])
        metadata = data.get("metadata", [{}] * len(embeddings))

        try:
            async with pool.acquire() as conn:
                # Ensure table exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS plugin_embeddings (
                        id SERIAL PRIMARY KEY,
                        plugin_id TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        embedding vector(1536),
                        text TEXT,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Insert embeddings
                for i, emb in enumerate(embeddings):
                    await conn.execute(
                        """
                        INSERT INTO plugin_embeddings
                        (plugin_id, collection, embedding, text, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        plugin_id,
                        path,
                        emb,
                        texts[i] if i < len(texts) else None,
                        json.dumps(metadata[i]) if i < len(metadata) else "{}",
                    )

            return WriteResult(
                destination=f"pgvector://{path}",
                success=True,
                bytes_written=len(embeddings),
                metadata={"count": len(embeddings)},
            )
        except Exception as e:
            logger.exception(f"pgvector write failed: {path}")
            return WriteResult(
                destination=f"pgvector://{path}",
                success=False,
                error=str(e),
            )

    async def write_file(
        self,
        plugin_id: str,
        file_path: Path,
        dest_path: str,
    ) -> WriteResult:
        """Write file contents as embeddings."""
        # Read JSON file with embeddings
        try:
            data = json.loads(file_path.read_text())
            return await self.write(plugin_id, data, dest_path)
        except Exception as e:
            return WriteResult(
                destination=f"pgvector://{dest_path}",
                success=False,
                error=str(e),
            )


class UMRSOutputWriter(OutputWriter):
    """Write plugin outputs to UMRS internal storage."""

    def __init__(
        self,
        storage_path: str | None = None,
    ):
        """Initialize UMRS storage writer."""
        self.storage_path = Path(
            storage_path or os.environ.get("UMRS_STORAGE_PATH", "/data/umrs")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def write(
        self,
        plugin_id: str,
        data: bytes | str | dict[str, Any],
        path: str,
        content_type: str | None = None,
    ) -> WriteResult:
        """Write data to UMRS storage."""
        dest = self.storage_path / "plugins" / plugin_id / path
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(data, dict):
                body = json.dumps(data, indent=2)
                dest.write_text(body)
                size = len(body)
            elif isinstance(data, str):
                dest.write_text(data)
                size = len(data)
            else:
                dest.write_bytes(data)
                size = len(data)

            return WriteResult(
                destination=f"umrs://plugins/{plugin_id}/{path}",
                success=True,
                bytes_written=size,
            )
        except Exception as e:
            logger.exception(f"UMRS write failed: {path}")
            return WriteResult(
                destination=f"umrs://plugins/{plugin_id}/{path}",
                success=False,
                error=str(e),
            )

    async def write_file(
        self,
        plugin_id: str,
        file_path: Path,
        dest_path: str,
    ) -> WriteResult:
        """Copy file to UMRS storage."""
        import shutil

        dest = self.storage_path / "plugins" / plugin_id / dest_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(file_path, dest)
            return WriteResult(
                destination=f"umrs://plugins/{plugin_id}/{dest_path}",
                success=True,
                bytes_written=file_path.stat().st_size,
            )
        except Exception as e:
            logger.exception(f"UMRS file copy failed: {dest_path}")
            return WriteResult(
                destination=f"umrs://plugins/{plugin_id}/{dest_path}",
                success=False,
                error=str(e),
            )


class PluginResultWriter:
    """
    Unified result writer for plugins (X-105).

    Routes writes to appropriate backend based on destination URI.
    """

    def __init__(self):
        """Initialize result writer with all backends."""
        self._writers: dict[str, OutputWriter] = {
            "s3": S3OutputWriter(),
            "pgvector": PgVectorOutputWriter(),
            "umrs": UMRSOutputWriter(),
        }

    def _get_writer(self, destination: str) -> tuple[OutputWriter, str]:
        """Get appropriate writer for destination."""
        parsed = urlparse(destination)
        scheme = parsed.scheme

        if scheme in ("s3", "s3a"):
            # Extract path from s3://bucket/path
            path = f"{parsed.netloc}/{parsed.path.lstrip('/')}"
            return self._writers["s3"], path
        elif scheme == "pgvector":
            return self._writers["pgvector"], parsed.path.lstrip("/")
        elif scheme == "umrs":
            return self._writers["umrs"], parsed.path.lstrip("/")
        else:
            raise ValueError(f"Unsupported destination scheme: {scheme}")

    async def write(
        self,
        plugin_id: str,
        data: bytes | str | dict[str, Any],
        destination: str,
        content_type: str | None = None,
    ) -> WriteResult:
        """
        Write data to destination.

        Args:
            plugin_id: Plugin execution ID.
            data: Data to write.
            destination: Destination URI (s3://, pgvector://, umrs://).
            content_type: Optional content type.

        Returns:
            WriteResult with outcome.
        """
        writer, path = self._get_writer(destination)
        return await writer.write(plugin_id, data, path, content_type)

    async def write_file(
        self,
        plugin_id: str,
        file_path: Path,
        destination: str,
    ) -> WriteResult:
        """
        Write file to destination.

        Args:
            plugin_id: Plugin execution ID.
            file_path: Local file path.
            destination: Destination URI.

        Returns:
            WriteResult with outcome.
        """
        writer, path = self._get_writer(destination)
        return await writer.write_file(plugin_id, file_path, path)

    async def write_all_outputs(
        self,
        plugin_id: str,
        output_dir: Path,
        destinations: list[str],
    ) -> list[WriteResult]:
        """
        Write all files from output directory to destinations.

        Args:
            plugin_id: Plugin execution ID.
            output_dir: Directory containing output files.
            destinations: List of destination URIs.

        Returns:
            List of WriteResults.
        """
        results = []

        if not output_dir.exists():
            return results

        for file_path in output_dir.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = file_path.relative_to(output_dir)

            for dest in destinations:
                full_dest = f"{dest.rstrip('/')}/{rel_path}"
                result = await self.write_file(plugin_id, file_path, full_dest)
                results.append(result)

        return results
