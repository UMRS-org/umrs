"""Tests for the unified knowledge ingestion pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from umrs.data_catalog.ingest import IngestionError
from umrs.knowledge.ingest import (
    BatchIngestionResult,
    IngestionResult,
    IngesterConfig,
    KnowledgeIngester,
    ingest_object,
)


# ── Test Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "test/example.py", "ETag": '"abc123"', "Size": 100},
                {"Key": "test/example.ipynb", "ETag": '"def456"', "Size": 200},
            ]
        }
    ]
    client.get_paginator.return_value = paginator
    return client


@pytest.fixture
def mock_s3_client_with_content():
    """Create a mock S3 client that returns content for list_objects."""
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "test/example.py", "ETag": '"abc123"', "Size": 100},
            ]
        }
    ]
    client.get_paginator.return_value = paginator
    return client


@pytest.fixture
def ready_meta():
    from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState

    return CatalogObjectMeta(
        s3_key="test/example.py",
        object_etag="abc123",
        state=CatalogState.READY,
        owner="test-user",
        acl_scope="test-project",
        updated_at="2026-01-01T00:00:00",
        detected_modality="code",
        summary="Test summary",
        tags=["python"],
    )


# ── Configuration Tests ─────────────────────────────────────────────────────


def test_ingester_config_defaults():
    """Test that IngesterConfig has sensible defaults."""
    config = IngesterConfig()

    assert config.max_workers == 4
    assert config.batch_size == 10
    assert config.max_extract_bytes == 10 * 1024 * 1024
    assert config.skip_if_unchanged is True
    assert config.incremental is True
    assert config.enable_rag_indexing is True


def test_ingester_config_custom():
    """Test that IngesterConfig can be customized."""
    config = IngesterConfig(
        max_workers=8,
        batch_size=20,
        max_extract_bytes=5 * 1024 * 1024,
        skip_if_unchanged=False,
        incremental=False,
        enable_rag_indexing=False,
    )

    assert config.max_workers == 8
    assert config.batch_size == 20
    assert config.max_extract_bytes == 5 * 1024 * 1024
    assert config.skip_if_unchanged is False
    assert config.incremental is False
    assert config.enable_rag_indexing is False


# ── KnowledgeIngester Tests ──────────────────────────────────────────────────


def test_knowledge_ingester_init_defaults():
    """Test KnowledgeIngester initialization with default config."""
    s3_client = MagicMock()
    ingester = KnowledgeIngester(s3_client, "test-bucket")

    assert ingester.s3_client is s3_client
    assert ingester.bucket == "test-bucket"
    assert isinstance(ingester.config, IngesterConfig)
    assert ingester.config.max_workers == 4


def test_knowledge_ingester_init_custom_config():
    """Test KnowledgeIngester initialization with custom config."""
    s3_client = MagicMock()
    config = IngesterConfig(max_workers=8)
    ingester = KnowledgeIngester(s3_client, "test-bucket", config)

    assert ingester.config.max_workers == 8


def test_knowledge_ingester_discover_objects(mock_s3_client):
    """Test that _discover_objects correctly lists objects."""
    ingester = KnowledgeIngester(mock_s3_client, "test-bucket")
    objects = ingester._discover_objects(prefix="")

    assert len(objects) == 2
    assert objects[0]["key"] == "test/example.py"
    assert objects[0]["etag"] == "abc123"
    assert objects[1]["key"] == "test/example.ipynb"


def test_knowledge_ingester_filter_unchanged(mock_s3_client):
    """Test that _filter_unchanged correctly filters unchanged objects."""
    from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState

    ingester = KnowledgeIngester(mock_s3_client, "test-bucket")

    objects = [
        {"key": "test/example.py", "etag": "abc123", "size": 100},
        {"key": "test/example2.py", "etag": "xyz789", "size": 200},
    ]

    with patch("umrs.knowledge.ingest.read_meta") as mock_read_meta:
        ready_meta = CatalogObjectMeta(
            s3_key="test/example.py",
            object_etag="abc123",
            state=CatalogState.READY,
            owner="test-user",
            acl_scope="test-project",
            updated_at="2026-01-01T00:00:00",
        )

        def side_effect(s3_client, bucket, key):
            if key == "test/example.py":
                return ready_meta
            return None

        mock_read_meta.side_effect = side_effect
        filtered = ingester._filter_unchanged(objects)

    assert len(filtered) == 1
    assert filtered[0]["key"] == "test/example2.py"


def test_knowledge_ingester_ingest_object_skipped(
    mock_s3_client_with_content,
    ready_meta,
):
    """Test that ingest_object skips unchanged objects via catalog."""
    ingester = KnowledgeIngester(mock_s3_client_with_content, "test-bucket")

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            return_value={"action": "skipped", "s3_key": "test/example.py"},
        ),
        patch("umrs.knowledge.ingest.read_meta", return_value=ready_meta),
    ):
        result = ingester.ingest_object(
            project_id="test-project",
            owner="test-user",
            s3_key="test/example.py",
            object_etag="abc123",
        )

    assert result.status == "skipped"
    assert result.s3_key == "test/example.py"
    assert result.summary == "Test summary"


def test_knowledge_ingester_delegates_to_catalog_with_10mb_limit(
    mock_s3_client_with_content,
    ready_meta,
):
    """Batch ingest delegates enrichment to data_catalog with 10 MB sample limit."""
    ingester = KnowledgeIngester(mock_s3_client_with_content, "test-bucket")

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            return_value={
                "action": "ingested",
                "s3_key": "test/example.py",
                "modality": "code",
                "has_extracted_text": True,
                "sampled_bytes": 35,
                "total_size": 35,
            },
        ) as mock_catalog,
        patch("umrs.knowledge.ingest.read_meta", return_value=ready_meta),
    ):
        result = ingester.ingest_object(
            project_id="test-project",
            owner="test-user",
            s3_key="test/example.py",
            object_etag="abc123",
        )

    assert result.status == "ingested"
    assert result.modality == "code"
    assert result.has_extracted_text is True
    mock_catalog.assert_called_once()
    call_kwargs = mock_catalog.call_args.kwargs
    assert call_kwargs["max_extract_bytes"] == 10 * 1024 * 1024
    assert call_kwargs["force"] is False


def test_knowledge_ingester_force_reingest_when_skip_disabled(
    mock_s3_client_with_content,
    ready_meta,
):
    """When skip_if_unchanged is False, catalog ingest is called with force=True."""
    config = IngesterConfig(skip_if_unchanged=False)
    ingester = KnowledgeIngester(mock_s3_client_with_content, "test-bucket", config)

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            return_value={
                "action": "ingested",
                "s3_key": "test/example.py",
                "modality": "code",
                "has_extracted_text": True,
            },
        ) as mock_catalog,
        patch("umrs.knowledge.ingest.read_meta", return_value=ready_meta),
    ):
        ingester.ingest_object(
            project_id="test-project",
            owner="test-user",
            s3_key="test/example.py",
            object_etag="abc123",
        )

    assert mock_catalog.call_args.kwargs["force"] is True


# ── Batch Ingestion Tests ──────────────────────────────────────────────────


def test_ingest_project_empty(mock_s3_client):
    """Test ingest_project with no objects."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]
    mock_s3_client.get_paginator.return_value = paginator

    ingester = KnowledgeIngester(mock_s3_client, "test-bucket")
    result = ingester.ingest_project("test-project", "test-user")

    assert isinstance(result, BatchIngestionResult)
    assert result.total == 0
    assert result.ingested == 0
    assert result.skipped == 0
    assert result.errors == 0


def test_ingest_project_with_objects(mock_s3_client_with_content, ready_meta):
    """Test ingest_project with multiple objects."""
    config = IngesterConfig(incremental=False)
    ingester = KnowledgeIngester(mock_s3_client_with_content, "test-bucket", config)

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            return_value={
                "action": "ingested",
                "s3_key": "test/example.py",
                "modality": "code",
                "logical_path": "test-project/default/code/",
                "has_extracted_text": True,
            },
        ) as mock_catalog_ingest,
        patch("umrs.knowledge.ingest.read_meta", return_value=ready_meta),
        patch("umrs.knowledge.ingest.regenerate_rollups_for_paths") as mock_rollups,
    ):
        result = ingester.ingest_project("test-project", "test-user")

    assert result.total == 1
    assert result.ingested == 1
    assert result.errors == 0
    mock_catalog_ingest.assert_called_once()
    assert mock_catalog_ingest.call_args.kwargs["regenerate_rollups"] is False
    mock_rollups.assert_called_once_with(
        mock_s3_client_with_content,
        "test-bucket",
        {"test-project/default/code/"},
    )


def test_ingest_object_refreshes_rollups_per_object(mock_s3_client_with_content, ready_meta):
    """Single-object ingest keeps per-object rollup refresh."""
    ingester = KnowledgeIngester(mock_s3_client_with_content, "test-bucket")

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            return_value={
                "action": "ingested",
                "s3_key": "test/example.py",
                "modality": "code",
                "logical_path": "test-project/default/code/",
                "has_extracted_text": True,
            },
        ) as mock_catalog_ingest,
        patch("umrs.knowledge.ingest.read_meta", return_value=ready_meta),
        patch("umrs.knowledge.ingest.regenerate_rollups_for_paths") as mock_rollups,
    ):
        result = ingester.ingest_object(
            "test-project",
            "test-user",
            s3_key="test/example.py",
            object_etag="abc123",
        )

    assert result.status == "ingested"
    assert mock_catalog_ingest.call_args.kwargs["regenerate_rollups"] is True
    mock_rollups.assert_not_called()


# ── Progress Callback Tests ────────────────────────────────────────────────


def test_progress_callback_called(mock_s3_client_with_content, ready_meta):
    """Test that the progress callback is called."""
    config = IngesterConfig(incremental=False)
    ingester = KnowledgeIngester(mock_s3_client_with_content, "test-bucket", config)

    progress_calls = []

    def callback(current, total, s3_key, status):
        progress_calls.append((current, total, s3_key, status))

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            return_value={
                "action": "ingested",
                "s3_key": "test/example.py",
                "modality": "code",
                "has_extracted_text": True,
            },
        ),
        patch("umrs.knowledge.ingest.read_meta", return_value=ready_meta),
    ):
        ingester.ingest_project(
            "test-project",
            "test-user",
            progress_callback=callback,
        )

    assert len(progress_calls) == 1
    assert progress_calls[0][0] == 1
    assert progress_calls[0][1] == 1
    assert progress_calls[0][3] == "ingested"


# ── Convenience Function Tests ────────────────────────────────────────────


def test_ingest_object_convenience_function(ready_meta):
    """Test the ingest_object convenience function."""
    s3_client = MagicMock()

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            return_value={
                "action": "ingested",
                "s3_key": "test/example.py",
                "modality": "code",
                "has_extracted_text": True,
            },
        ),
        patch("umrs.knowledge.ingest.read_meta", return_value=ready_meta),
    ):
        result = ingest_object(
            s3_client,
            "test-bucket",
            s3_key="test/example.py",
            object_etag="abc123",
            project_id="test-project",
            owner="test-user",
        )

    assert result["action"] == "ingested"
    assert result["modality"] == "code"
    assert result["has_extracted_text"] is True


# ── Result Type Tests ────────────────────────────────────────────────────


def test_ingestion_result_defaults():
    """Test IngestionResult defaults."""
    result = IngestionResult(
        s3_key="test/example.py",
        modality="code",
        status="ingested",
    )

    assert result.s3_key == "test/example.py"
    assert result.modality == "code"
    assert result.status == "ingested"
    assert result.summary == ""
    assert result.tags == []
    assert result.extracted_text == ""
    assert result.has_extracted_text is False
    assert result.sampled_bytes == 0
    assert result.total_size == 0
    assert result.error is None


def test_batch_ingestion_result_defaults():
    """Test BatchIngestionResult defaults."""
    result = BatchIngestionResult()

    assert result.total == 0
    assert result.ingested == 0
    assert result.skipped == 0
    assert result.errors == 0
    assert result.results == []
    assert result.duration_seconds == 0.0


# ── Error Handling Tests ──────────────────────────────────────────────────


def test_ingest_object_error_handling(mock_s3_client_with_content):
    """Catalog failures become error results; catalog already wrote ERROR meta."""
    ingester = KnowledgeIngester(mock_s3_client_with_content, "test-bucket")

    from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState

    error_meta = CatalogObjectMeta(
        s3_key="test/example.py",
        object_etag="abc123",
        state=CatalogState.ERROR,
        owner="test-user",
        acl_scope="test-project",
        updated_at="2026-01-01T00:00:00",
        detected_modality="code",
        error_note="object missing: test/example.py",
    )

    with (
        patch(
            "umrs.knowledge.ingest.catalog_ingest_object",
            side_effect=IngestionError("object missing: test/example.py"),
        ),
        patch("umrs.knowledge.ingest.read_meta", return_value=error_meta),
    ):
        result = ingester.ingest_object(
            project_id="test-project",
            owner="test-user",
            s3_key="test/example.py",
            object_etag="abc123",
        )

    assert result.status == "error"
    assert "object missing" in result.error
    assert result.modality == "code"
