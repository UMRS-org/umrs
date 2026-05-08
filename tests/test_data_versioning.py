"""Tests for B6 — Data Versioning."""

import json
import time

import pytest

from umrs.shared.data_versioning import (
    DataVersion,
    DataVersionManager,
    ManifestDiff,
    ObjectRecord,
    VersionManifest,
)


class TestObjectRecord:
    def test_creation(self):
        r = ObjectRecord(key="data/file.csv", etag="abc123", size_bytes=1024)
        assert r.key == "data/file.csv"
        assert r.size_bytes == 1024


class TestVersionManifest:
    def test_fingerprint_deterministic(self):
        m = VersionManifest(objects={
            "a.txt": ObjectRecord(key="a.txt", etag="e1", size_bytes=10),
            "b.txt": ObjectRecord(key="b.txt", etag="e2", size_bytes=20),
        })
        f1 = m.fingerprint()
        f2 = m.fingerprint()
        assert f1 == f2
        assert len(f1) > 0

    def test_fingerprint_changes_with_content(self):
        m1 = VersionManifest(objects={
            "a.txt": ObjectRecord(key="a.txt", etag="e1", size_bytes=10),
        })
        m2 = VersionManifest(objects={
            "a.txt": ObjectRecord(key="a.txt", etag="e2", size_bytes=10),
        })
        assert m1.fingerprint() != m2.fingerprint()

    def test_diff_no_changes(self):
        objects = {"a.txt": ObjectRecord(key="a.txt", etag="e1", size_bytes=10)}
        m1 = VersionManifest(objects=dict(objects))
        m2 = VersionManifest(objects=dict(objects))
        diff = m1.diff(m2)
        assert not diff.has_changes
        assert len(diff.unchanged) == 1

    def test_diff_added_removed_modified(self):
        m1 = VersionManifest(objects={
            "a.txt": ObjectRecord(key="a.txt", etag="e1", size_bytes=10),
            "b.txt": ObjectRecord(key="b.txt", etag="e2", size_bytes=20),
        })
        m2 = VersionManifest(objects={
            "a.txt": ObjectRecord(key="a.txt", etag="e1_changed", size_bytes=15),
            "c.txt": ObjectRecord(key="c.txt", etag="e3", size_bytes=30),
        })
        diff = m1.diff(m2)
        assert diff.has_changes
        assert "c.txt" in diff.added
        assert "b.txt" in diff.removed
        assert "a.txt" in diff.modified


class TestManifestDiff:
    def test_summary(self):
        diff = ManifestDiff(
            added=["new.txt"],
            removed=["old.txt"],
            modified=["changed.txt"],
            unchanged=["same.txt"],
        )
        s = diff.summary()
        assert "1 added" in s
        assert "1 removed" in s


class TestDataVersionManager:
    def test_create_and_get_version(self, tmp_path):
        mgr = DataVersionManager(store_dir=str(tmp_path))
        objs = [ObjectRecord(key="a.txt", etag="e1", size_bytes=10)]
        v = mgr.create_version(bucket="test-bucket", objects=objs, tag="v1")
        assert v.version_id
        assert v.tag == "v1"

        retrieved = mgr.get_version(v.version_id)
        assert retrieved is not None
        assert retrieved.tag == "v1"

    def test_list_versions(self, tmp_path):
        mgr = DataVersionManager(store_dir=str(tmp_path))
        objs: list = []
        mgr.create_version(bucket="b1", objects=objs, tag="v1")
        mgr.create_version(bucket="b1", objects=objs, tag="v2")

        versions = mgr.list_versions()
        assert len(versions) == 2

    def test_get_by_tag(self, tmp_path):
        mgr = DataVersionManager(store_dir=str(tmp_path))
        objs = [ObjectRecord(key="x.txt", etag="e1", size_bytes=5)]
        mgr.create_version(bucket="b1", objects=objs, tag="release-1")

        v = mgr.get_by_tag("b1", "release-1")
        assert v is not None
        assert v.tag == "release-1"

        assert mgr.get_by_tag("b1", "nonexistent") is None

    def test_diff_versions(self, tmp_path):
        mgr = DataVersionManager(store_dir=str(tmp_path))

        objs1 = [ObjectRecord(key="a.txt", etag="e1", size_bytes=10)]
        objs2 = [
            ObjectRecord(key="a.txt", etag="e2", size_bytes=15),
            ObjectRecord(key="b.txt", etag="e3", size_bytes=20),
        ]
        v1 = mgr.create_version(bucket="b1", objects=objs1, tag="v1")
        v2 = mgr.create_version(bucket="b1", objects=objs2, tag="v2")

        diff = mgr.diff_versions(v1.version_id, v2.version_id)
        assert diff is not None
        assert diff.has_changes
        assert "b.txt" in diff.added

    def test_lineage(self, tmp_path):
        mgr = DataVersionManager(store_dir=str(tmp_path))
        objs: list = []
        v1 = mgr.create_version(bucket="b1", objects=objs, tag="v1")
        v2 = mgr.create_version(bucket="b1", objects=objs, tag="v2")
        v3 = mgr.create_version(bucket="b1", objects=objs, tag="v3")

        chain = mgr.lineage(v3.version_id)
        assert len(chain) == 3

    def test_persistence(self, tmp_path):
        mgr1 = DataVersionManager(store_dir=str(tmp_path))
        objs = [ObjectRecord(key="f.txt", etag="e1", size_bytes=5)]
        mgr1.create_version(bucket="b1", objects=objs, tag="persistent")

        # New manager reads persisted data
        mgr2 = DataVersionManager(store_dir=str(tmp_path))
        v = mgr2.get_by_tag("b1", "persistent")
        assert v is not None
