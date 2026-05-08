"""
B6 — Data Versioning

Version-track datasets in S3 buckets.  Each version captures a snapshot
of bucket contents (object keys + checksums).  Supports tagging, rollback,
and lineage queries.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ObjectRecord:
    """Record of a single S3 object in a version snapshot."""
    key: str
    etag: str = ""
    size_bytes: int = 0
    content_type: str = ""
    last_modified: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObjectRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class VersionManifest:
    """Mapping of object keys to their ETags/checksums at a point in time."""
    objects: dict[str, ObjectRecord] = field(default_factory=dict)

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def total_size(self) -> int:
        return sum(o.size_bytes for o in self.objects.values())

    def fingerprint(self) -> str:
        """Deterministic hash of manifest contents."""
        parts = sorted(f"{k}:{v.etag}:{v.size_bytes}" for k, v in self.objects.items())
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]

    def diff(self, other: VersionManifest) -> ManifestDiff:
        """Compare two manifests."""
        keys_a = set(self.objects.keys())
        keys_b = set(other.objects.keys())
        added = keys_b - keys_a
        removed = keys_a - keys_b
        common = keys_a & keys_b
        modified = {k for k in common if self.objects[k].etag != other.objects[k].etag}
        unchanged = common - modified
        return ManifestDiff(
            added=sorted(added),
            removed=sorted(removed),
            modified=sorted(modified),
            unchanged=sorted(unchanged),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"objects": {k: v.to_dict() for k, v in self.objects.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VersionManifest:
        objects = {k: ObjectRecord.from_dict(v) for k, v in d.get("objects", {}).items()}
        return cls(objects=objects)


@dataclass
class ManifestDiff:
    """Diff between two version manifests."""
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified)

    def summary(self) -> str:
        return (
            f"+{len(self.added)} added, -{len(self.removed)} removed, "
            f"~{len(self.modified)} modified, ={len(self.unchanged)} unchanged"
        )


@dataclass
class DataVersion:
    """Immutable snapshot of a dataset version."""
    version_id: str
    bucket: str
    manifest: VersionManifest
    tag: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    parent_version_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return self.manifest.fingerprint()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "bucket": self.bucket,
            "manifest": self.manifest.to_dict(),
            "tag": self.tag,
            "description": self.description,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "parent_version_id": self.parent_version_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataVersion:
        return cls(
            version_id=d["version_id"],
            bucket=d["bucket"],
            manifest=VersionManifest.from_dict(d["manifest"]),
            tag=d.get("tag", ""),
            description=d.get("description", ""),
            created_at=d.get("created_at", 0),
            created_by=d.get("created_by", ""),
            parent_version_id=d.get("parent_version_id"),
            metadata=d.get("metadata", {}),
        )


class DataVersionManager:
    """Create, list, compare, and restore dataset versions.

    Versions are persisted as JSON files under a configurable store directory.

    Usage::

        manager = DataVersionManager(store_dir="/data/.versions")

        # Create a version from current bucket state
        version = manager.create_version(
            bucket="user1-images",
            objects=list_s3_objects("user1-images"),
            tag="v1.0",
            description="Initial dataset",
            created_by="user1",
        )

        # Compare versions
        diff = manager.diff_versions(version_id_a, version_id_b)
    """

    def __init__(self, store_dir: str = ".umrs_versions") -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        bucket: str,
        objects: list[ObjectRecord],
        tag: str = "",
        description: str = "",
        created_by: str = "",
        parent_version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DataVersion:
        """Create a new version from a list of object records."""
        version_id = f"v-{uuid.uuid4().hex[:12]}"
        manifest = VersionManifest(objects={o.key: o for o in objects})

        # Auto-detect parent if not specified
        if parent_version_id is None:
            versions = self.list_versions(bucket)
            if versions:
                parent_version_id = versions[-1].version_id

        version = DataVersion(
            version_id=version_id,
            bucket=bucket,
            manifest=manifest,
            tag=tag,
            description=description,
            created_at=time.time(),
            created_by=created_by,
            parent_version_id=parent_version_id,
            metadata=metadata or {},
        )

        self._save_version(version)
        return version

    def get_version(self, version_id: str) -> DataVersion | None:
        path = self._version_path(version_id)
        if not path.exists():
            return None
        return DataVersion.from_dict(json.loads(path.read_text()))

    def list_versions(
        self,
        bucket: str | None = None,
        tag: str | None = None,
    ) -> list[DataVersion]:
        versions = []
        for p in sorted(self._store_dir.glob("v-*.json")):
            try:
                v = DataVersion.from_dict(json.loads(p.read_text()))
                if bucket and v.bucket != bucket:
                    continue
                if tag and v.tag != tag:
                    continue
                versions.append(v)
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(versions, key=lambda v: v.created_at)

    def get_by_tag(self, bucket: str, tag: str) -> DataVersion | None:
        for v in self.list_versions(bucket=bucket, tag=tag):
            return v
        return None

    def diff_versions(
        self,
        version_id_a: str,
        version_id_b: str,
    ) -> ManifestDiff | None:
        a = self.get_version(version_id_a)
        b = self.get_version(version_id_b)
        if not a or not b:
            return None
        return a.manifest.diff(b.manifest)

    def lineage(self, version_id: str) -> list[DataVersion]:
        """Trace the lineage of a version back to the root."""
        chain: list[DataVersion] = []
        seen: set = set()
        current_id: str | None = version_id
        while current_id and current_id not in seen:
            seen.add(current_id)
            v = self.get_version(current_id)
            if not v:
                break
            chain.append(v)
            current_id = v.parent_version_id
        chain.reverse()
        return chain

    def _save_version(self, version: DataVersion) -> None:
        path = self._version_path(version.version_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(version.to_dict(), indent=2, default=str))
        tmp.rename(path)

    def _version_path(self, version_id: str) -> Path:
        return self._store_dir / f"{version_id}.json"
