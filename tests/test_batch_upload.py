"""Tests for B2 — Batch Upload with Resume."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from umrs.client.batch_upload import (
    BatchUploadManager,
    FileEntry,
    FileState,
    UploadManifest,
    UploadReport,
    _compute_md5,
)


class TestFileEntry:
    def test_roundtrip(self):
        e = FileEntry(relative_path="a.txt", size_bytes=100, md5="abc", state=FileState.UPLOADED)
        d = e.to_dict()
        e2 = FileEntry.from_dict(d)
        assert e2.relative_path == "a.txt"
        assert e2.state == FileState.UPLOADED

    def test_default_state(self):
        e = FileEntry(relative_path="b.txt", size_bytes=50)
        assert e.state == FileState.PENDING
        assert e.attempts == 0


class TestUploadManifest:
    def test_save_and_load(self, tmp_path):
        m = UploadManifest(source_dir=str(tmp_path), bucket_alias="test-bucket")
        m.files["a.txt"] = FileEntry(relative_path="a.txt", size_bytes=10)
        m.files["b.txt"] = FileEntry(relative_path="b.txt", size_bytes=20, state=FileState.UPLOADED)
        m.save()

        loaded = UploadManifest.load(m.manifest_path)
        assert loaded.bucket_alias == "test-bucket"
        assert len(loaded.files) == 2
        assert loaded.files["b.txt"].state == FileState.UPLOADED

    def test_load_or_create_new(self, tmp_path):
        m = UploadManifest.load_or_create(str(tmp_path), "bucket1")
        assert m.bucket_alias == "bucket1"
        assert len(m.files) == 0

    def test_load_or_create_existing(self, tmp_path):
        m = UploadManifest(source_dir=str(tmp_path), bucket_alias="bucket1")
        m.files["x.txt"] = FileEntry(relative_path="x.txt", size_bytes=5)
        m.save()

        m2 = UploadManifest.load_or_create(str(tmp_path), "bucket1")
        assert len(m2.files) == 1

    def test_pending_files(self, tmp_path):
        m = UploadManifest(source_dir=str(tmp_path), bucket_alias="b")
        m.files["a"] = FileEntry(relative_path="a", size_bytes=1, state=FileState.PENDING)
        m.files["b"] = FileEntry(relative_path="b", size_bytes=1, state=FileState.UPLOADED)
        m.files["c"] = FileEntry(relative_path="c", size_bytes=1, state=FileState.FAILED)
        assert len(m.pending_files) == 2

    def test_is_complete(self, tmp_path):
        m = UploadManifest(source_dir=str(tmp_path), bucket_alias="b")
        m.files["a"] = FileEntry(relative_path="a", size_bytes=1, state=FileState.UPLOADED)
        assert m.is_complete
        m.files["b"] = FileEntry(relative_path="b", size_bytes=1, state=FileState.PENDING)
        assert not m.is_complete


class TestComputeMd5:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        md5_1 = _compute_md5(f)
        md5_2 = _compute_md5(f)
        assert md5_1 == md5_2
        assert len(md5_1) == 32


class TestBatchUploadManager:
    def test_upload_directory(self, tmp_path):
        # Create test files
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")

        uploaded = []

        def mock_upload(path, bucket):
            uploaded.append(str(path.name))

        mgr = BatchUploadManager(mock_upload, max_workers=1, max_retries=1)
        report = mgr.upload_directory(str(tmp_path), "test-bucket")

        assert report.uploaded == 2
        assert report.failed == 0
        assert report.total == 2
        assert set(uploaded) == {"a.txt", "b.txt"}

    def test_resume_skips_uploaded(self, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")

        uploaded = []

        def mock_upload(path, bucket):
            uploaded.append(str(path.name))

        mgr = BatchUploadManager(mock_upload, max_workers=1, max_retries=1)
        mgr.upload_directory(str(tmp_path), "test-bucket")
        assert len(uploaded) == 2

        # Second run should skip already uploaded
        uploaded.clear()
        mgr.upload_directory(str(tmp_path), "test-bucket")
        assert len(uploaded) == 0

    def test_retry_on_failure(self, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")

        call_count = 0

        def flaky_upload(path, bucket):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("connection reset")

        mgr = BatchUploadManager(flaky_upload, max_workers=1, max_retries=3)
        report = mgr.upload_directory(str(tmp_path), "test-bucket")
        assert report.uploaded == 1

    def test_failure_after_max_retries(self, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")

        def always_fail(path, bucket):
            raise OSError("permanent failure")

        mgr = BatchUploadManager(always_fail, max_workers=1, max_retries=2)
        report = mgr.upload_directory(str(tmp_path), "test-bucket")
        assert report.failed == 1
        assert report.uploaded == 0

    def test_excludes_hidden_files(self, tmp_path):
        (tmp_path / "visible.txt").write_text("ok")
        (tmp_path / ".hidden").write_text("hidden")

        uploaded = []

        def mock_upload(path, bucket):
            uploaded.append(path.name)

        mgr = BatchUploadManager(mock_upload, max_workers=1, max_retries=1)
        mgr.upload_directory(str(tmp_path), "b", exclude_hidden=True)
        assert ".hidden" not in uploaded

    def test_nonexistent_directory(self):
        mgr = BatchUploadManager(lambda p, b: None)
        with pytest.raises(FileNotFoundError):
            mgr.upload_directory("/nonexistent/dir", "bucket")
