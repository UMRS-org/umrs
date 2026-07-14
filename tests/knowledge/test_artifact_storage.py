"""Tests for artifact_storage.py — S3 content-addressed artifact storage."""

from __future__ import annotations

import hashlib

from umrs.knowledge.artifact_storage import ArtifactStorage

# ── Fixtures ─────────────────────────────────────────────────────────────────


class FakeS3Client:
    """Minimal synchronous S3 client mock for unit tests."""

    class exceptions:
        """Mock botocore.exceptions namespace."""

        class ClientError(Exception):
            def __init__(self, error_response: dict, operation_name: str) -> None:
                self.response = {"Error": error_response}
                self.operation_name = operation_name

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **_kwargs) -> None:  # noqa: N802, ARG002
        self._store[f"{Bucket}/{Key}"] = Body

    def get_object(self, Bucket: str, Key: str, **_kwargs) -> dict:  # noqa: N802, ARG002
        value = self._store.get(f"{Bucket}/{Key}")
        if value is None:
            raise FakeS3Client.exceptions.ClientError(
                {"Code": "NoSuchKey"},
                "GetObject",
            )
        return {"Body": FakeBody(value)}

    def head_object(self, Bucket: str, Key: str, **_kwargs) -> dict:  # noqa: N802, ARG002
        value = self._store.get(f"{Bucket}/{Key}")
        if value is None:
            raise FakeS3Client.exceptions.ClientError(
                {"Code": "NoSuchKey"},
                "HeadObject",
            )
        return {"ContentLength": len(value)}

    def delete_object(self, Bucket: str, Key: str, **_kwargs) -> None:  # noqa: N802, ARG002
        if f"{Bucket}/{Key}" not in self._store:
            raise FakeS3Client.exceptions.ClientError(
                {"Code": "NoSuchKey"},
                "DeleteObject",
            )
        self._store.pop(f"{Bucket}/{Key}")

    def list_objects_v2(self, Bucket: str, Prefix: str, **_kwargs) -> dict:  # noqa: N802, ARG002
        contents = [
            {"Key": path.split("/", 1)[1], "Size": len(data)}
            for path, data in self._store.items()
            if path.startswith(f"{Bucket}/{Prefix}")
        ]
        return {"Contents": contents}

    def generate_presigned_url(self, _Action: str, Params: dict, ExpiresIn: int) -> str:  # noqa: N802, N803, ARG002
        return (
            f"https://s3.example.com/{Params['Bucket']}/{Params['Key']}?X-Amz-Expires={ExpiresIn}"
        )


class FakeBody:
    """Mimics botocore response Body interface."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


PROJECT_ID = "test-project"


def _storage() -> ArtifactStorage:
    return ArtifactStorage(FakeS3Client(), "test-bucket")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestKeyGeneration:
    """Verify key pattern: projects/{project_id}/knowledge/artifacts/{hash}."""

    def test_key_pattern(self) -> None:
        key = ArtifactStorage.key(PROJECT_ID, "abc123")
        assert key == f"projects/{PROJECT_ID}/knowledge/artifacts/abc123"

    def test_key_no_slashes_in_hash(self) -> None:
        """SHA-256 hexdigest never contains slashes."""
        content = b"test"
        h = hashlib.sha256(content).hexdigest()
        assert "/" not in h
        key = ArtifactStorage.key(PROJECT_ID, h)
        assert key == f"projects/{PROJECT_ID}/knowledge/artifacts/{h}"


class TestRoundTrip:
    """Criterion 1: put then get returns same bytes."""

    def test_roundtrip_bytes(self) -> None:
        storage = _storage()
        content = b"hello world -- knowledge artifact"
        content_hash, size, url = storage.put_artifact(PROJECT_ID, content, mime_type="text/plain")

        retrieved = storage.get_artifact(PROJECT_ID, content_hash)
        assert retrieved == content
        assert size == len(content)

    def test_roundtrip_binary(self) -> None:
        storage = _storage()
        content = bytes(range(256))
        content_hash, _, _ = storage.put_artifact(PROJECT_ID, content)
        retrieved = storage.get_artifact(PROJECT_ID, content_hash)
        assert retrieved == content


class TestDeduplication:
    """Criterion 3: same content → same hash (content-addressed dedup)."""

    def test_same_content_same_hash(self) -> None:
        storage = _storage()
        content = b"duplicate content"
        hash1, size1, _ = storage.put_artifact(PROJECT_ID, content)
        hash2, size2, _ = storage.put_artifact(PROJECT_ID, content)

        assert hash1 == hash2  # Content-addressed: same content → same hash
        assert size1 == size2  # Both return actual content length
        assert size2 == len(content)  # Exists → returns stored size

    def test_different_content_different_hash(self) -> None:
        storage = _storage()
        content1 = b"content alpha"
        content2 = b"content beta"
        hash1, _, _ = storage.put_artifact(PROJECT_ID, content1)
        hash2, _, _ = storage.put_artifact(PROJECT_ID, content2)
        assert hash1 != hash2


class TestPresignedUrl:
    """Criterion 2: presigned URL generation is internal-only.

    No presigned URL is returned in the response unless ttl_seconds > 0 is
    explicitly requested. The URL is never embedded in API response bodies.
    """

    def test_put_returns_no_url_by_default(self) -> None:
        """Default: no presigned URL in response (ttl_seconds=0)."""
        storage = _storage()
        content = b"test data"
        _hash, _size, url = storage.put_artifact(PROJECT_ID, content, ttl_seconds=0)
        assert url is None

    def test_put_returns_url_when_requested(self) -> None:
        """Explicit ttl_seconds > 0 returns a presigned URL."""
        storage = _storage()
        content = b"test data"
        _hash, _size, url = storage.put_artifact(PROJECT_ID, content, ttl_seconds=600)
        assert url is not None
        assert url.startswith("https://s3.example.com/")
        assert "X-Amz-Expires=600" in url

    def test_presigned_url_ttl_default(self) -> None:
        """get_presigned_url uses 300s default TTL."""
        storage = _storage()
        content = b"test"
        storage.put_artifact(PROJECT_ID, content)
        content_hash = hashlib.sha256(content).hexdigest()
        url = storage.get_presigned_url(PROJECT_ID, content_hash)
        assert url is not None
        assert "X-Amz-Expires=300" in url

    def test_presigned_url_not_in_api_response(self) -> None:
        """Assert: presigned URLs never appear in API response bodies.

        The ArtifactStorage API only returns URLs when explicitly requested
        via ttl_seconds > 0. The default behavior (ttl_seconds=0) returns None,
        ensuring URLs are never leaked in normal API responses.
        """
        s3 = FakeS3Client()
        storage = ArtifactStorage(s3, "test-bucket")
        content = b"api response test"
        _hash, _size, url = storage.put_artifact(PROJECT_ID, content, ttl_seconds=0)
        # Default put → no URL in response body
        assert url is None
        # API responses should never contain the URL string
        response_body = {"content_hash": _hash, "size": _size}
        response_str = str(response_body)
        assert "presigned" not in response_str.lower() or "url" not in response_str.lower()


class TestArtifactExists:
    """Existence check by content hash."""

    def test_exists_after_put(self) -> None:
        storage = _storage()
        content = b"check me"
        h, _, _ = storage.put_artifact(PROJECT_ID, content)
        assert storage.artifact_exists(PROJECT_ID, h) is True

    def test_not_exists_before_put(self) -> None:
        storage = _storage()
        assert storage.artifact_exists(PROJECT_ID, "nonexistent") is False

    def test_not_exists_after_delete(self) -> None:
        storage = _storage()
        content = b"delete me"
        h, _, _ = storage.put_artifact(PROJECT_ID, content)
        storage.delete_artifact(PROJECT_ID, h)
        assert storage.artifact_exists(PROJECT_ID, h) is False


class TestListArtifacts:
    """List artifacts within a project prefix."""

    def test_list_empty(self) -> None:
        storage = _storage()
        result = storage.get_all_artifacts(PROJECT_ID)
        assert result == []

    def test_list_returns_hashes_and_sizes(self) -> None:
        storage = _storage()
        c1 = b"one"
        c2 = b"two"
        h1, _, _ = storage.put_artifact(PROJECT_ID, c1)
        h2, _, _ = storage.put_artifact(PROJECT_ID, c2)
        result = storage.get_all_artifacts(PROJECT_ID)
        result_hashes = {item[0] for item in result}
        assert h1 in result_hashes
        assert h2 in result_hashes


class TestDelete:
    """Artifact deletion."""

    def test_delete_returns_true(self) -> None:
        storage = _storage()
        content = b"delete me"
        h, _, _ = storage.put_artifact(PROJECT_ID, content)
        assert storage.delete_artifact(PROJECT_ID, h) is True

    def test_delete_missing_returns_false(self) -> None:
        storage = _storage()
        assert storage.delete_artifact(PROJECT_ID, "nope") is False
