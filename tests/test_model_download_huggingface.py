"""Tests for HuggingFace model download helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from umrs.jobs.model_download import download_from_huggingface


@pytest.fixture
def temp_dir():
    """Create a temporary directory for download tests."""
    with tempfile.TemporaryDirectory() as directory:
        yield directory


@pytest.mark.asyncio
async def test_download_from_huggingface_success(temp_dir):
    """Successful HuggingFace downloads report completed progress."""
    mock_api = MagicMock()
    mock_api.repo_info.return_value = MagicMock(
        siblings=[
            MagicMock(size=100),
            MagicMock(size=200),
        ]
    )
    test_file = Path(temp_dir) / "model.bin"
    test_file.write_bytes(b"test content")

    class GatedRepoError(Exception):
        """Mock gated repository exception."""

    class RepositoryNotFoundError(Exception):
        """Mock missing repository exception."""

    class RevisionNotFoundError(Exception):
        """Mock missing revision exception."""

    hub_module = SimpleNamespace(
        HfApi=lambda token=None: mock_api,
        snapshot_download=lambda **kwargs: temp_dir,
    )
    utils_module = SimpleNamespace(
        GatedRepoError=GatedRepoError,
        RepositoryNotFoundError=RepositoryNotFoundError,
        RevisionNotFoundError=RevisionNotFoundError,
    )

    with patch.dict(
        "sys.modules",
        {
            "huggingface_hub": hub_module,
            "huggingface_hub.utils": utils_module,
        },
    ):
        progress = await download_from_huggingface(
            repo_id="test/model",
            target_path=temp_dir,
        )

    assert progress.status == "completed"


@pytest.mark.asyncio
async def test_download_from_huggingface_missing_hub():
    """Missing huggingface_hub returns a failed progress object."""
    with patch.dict("sys.modules", {"huggingface_hub": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            progress = await download_from_huggingface(
                repo_id="test/model",
                target_path="/tmp/model",
            )

    assert progress.status == "failed"
    assert "huggingface_hub not installed" in progress.error_message
