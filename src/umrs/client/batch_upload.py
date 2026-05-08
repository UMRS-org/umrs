"""
B2 — Batch Upload with Resume

Resume-capable batch file uploads using a local manifest that tracks
which files have been successfully uploaded.  Supports parallel uploads
and automatic retry on transient failures.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from umrs.shared.progress import ProgressGroup


class FileState(str, Enum):
    """Upload state for an individual file."""
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FileEntry:
    """Tracks upload state for a single file."""
    relative_path: str
    size_bytes: int
    md5: str = ""
    state: FileState = FileState.PENDING
    attempts: int = 0
    last_error: str = ""
    uploaded_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FileEntry:
        d["state"] = FileState(d["state"])
        return cls(**d)


@dataclass
class UploadManifest:
    """JSON-based tracking of batch upload state.

    Persisted to a `.umrs_upload_manifest.json` file alongside the source
    directory so uploads survive client restarts.
    """
    source_dir: str
    bucket_alias: str
    created_at: float = field(default_factory=time.time)
    files: dict[str, FileEntry] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return Path(self.source_dir) / ".umrs_upload_manifest.json"

    def save(self) -> None:
        data = {
            "source_dir": self.source_dir,
            "bucket_alias": self.bucket_alias,
            "created_at": self.created_at,
            "files": {k: v.to_dict() for k, v in self.files.items()},
        }
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self.manifest_path)

    @classmethod
    def load(cls, path: Path) -> UploadManifest:
        data = json.loads(path.read_text())
        files = {k: FileEntry.from_dict(v) for k, v in data.get("files", {}).items()}
        return cls(
            source_dir=data["source_dir"],
            bucket_alias=data["bucket_alias"],
            created_at=data.get("created_at", 0),
            files=files,
        )

    @classmethod
    def load_or_create(cls, source_dir: str, bucket_alias: str) -> UploadManifest:
        p = Path(source_dir) / ".umrs_upload_manifest.json"
        if p.exists():
            manifest = cls.load(p)
            if manifest.bucket_alias == bucket_alias:
                return manifest
        return cls(source_dir=source_dir, bucket_alias=bucket_alias)

    @property
    def pending_files(self) -> list[FileEntry]:
        return [f for f in self.files.values() if f.state in (FileState.PENDING, FileState.FAILED)]

    @property
    def uploaded_count(self) -> int:
        return sum(1 for f in self.files.values() if f.state == FileState.UPLOADED)

    @property
    def total_count(self) -> int:
        return len(self.files)

    @property
    def is_complete(self) -> bool:
        return all(f.state == FileState.UPLOADED for f in self.files.values())


def _compute_md5(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Type alias for the upload function provided by the caller
UploadFunction = Callable[[Path, str], None]  # (local_path, bucket_alias) -> None


class BatchUploadManager:
    """Orchestrates multi-file uploads with resume capability.

    Usage::

        def my_upload(local_path, bucket):
            # actual S3/UMRS upload
            ...

        manager = BatchUploadManager(my_upload, max_workers=4, max_retries=3)
        report = manager.upload_directory("/data/images", "user1-images")
        print(report.summary())
    """

    def __init__(
        self,
        upload_fn: UploadFunction,
        max_workers: int = 4,
        max_retries: int = 3,
        compute_checksums: bool = True,
    ) -> None:
        self.upload_fn = upload_fn
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.compute_checksums = compute_checksums
        self._progress_group: ProgressGroup | None = None

    def upload_directory(
        self,
        source_dir: str,
        bucket_alias: str,
        pattern: str = "*",
        exclude_hidden: bool = True,
    ) -> UploadReport:
        """Upload all files in a directory with resume support."""
        src = Path(source_dir)
        if not src.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        manifest = UploadManifest.load_or_create(source_dir, bucket_alias)

        # Discover files
        for p in sorted(src.rglob(pattern)):
            if not p.is_file():
                continue
            if p.name.startswith(".") and exclude_hidden:
                continue
            rel = str(p.relative_to(src))
            if rel not in manifest.files:
                md5 = _compute_md5(p) if self.compute_checksums else ""
                manifest.files[rel] = FileEntry(
                    relative_path=rel,
                    size_bytes=p.stat().st_size,
                    md5=md5,
                )

        manifest.save()

        # Setup progress tracking
        self._progress_group = ProgressGroup(f"upload-{bucket_alias}")
        tracker = self._progress_group.create_tracker("overall", total=manifest.total_count)
        tracker.set_progress(manifest.uploaded_count, "Resuming...")
        tracker.start()

        # Upload pending files
        pending = manifest.pending_files
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._upload_one, src, entry, bucket_alias, manifest): entry
                for entry in pending
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    future.result()
                    tracker.advance(1, f"Uploaded {entry.relative_path}")
                except Exception as exc:
                    errors.append(f"{entry.relative_path}: {exc}")
                    tracker.advance(1, f"Failed {entry.relative_path}")

        if manifest.is_complete:
            tracker.complete("All files uploaded")
        else:
            tracker.fail(f"{len(errors)} file(s) failed")

        manifest.save()
        return UploadReport(manifest=manifest, errors=errors)

    def _upload_one(
        self,
        src: Path,
        entry: FileEntry,
        bucket_alias: str,
        manifest: UploadManifest,
    ) -> None:
        """Upload a single file with retry."""
        local_path = src / entry.relative_path
        for attempt in range(1, self.max_retries + 1):
            entry.state = FileState.UPLOADING
            entry.attempts = attempt
            try:
                self.upload_fn(local_path, bucket_alias)
                entry.state = FileState.UPLOADED
                entry.uploaded_at = time.time()
                entry.last_error = ""
                manifest.save()
                return
            except Exception as exc:
                entry.last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))

        entry.state = FileState.FAILED
        manifest.save()
        raise RuntimeError(f"Upload failed after {self.max_retries} retries: {entry.last_error}")


@dataclass
class UploadReport:
    """Summary of a batch upload operation."""
    manifest: UploadManifest
    errors: list[str] = field(default_factory=list)

    @property
    def uploaded(self) -> int:
        return self.manifest.uploaded_count

    @property
    def failed(self) -> int:
        return sum(1 for f in self.manifest.files.values() if f.state == FileState.FAILED)

    @property
    def total(self) -> int:
        return self.manifest.total_count

    @property
    def success(self) -> bool:
        return self.manifest.is_complete

    def summary(self) -> str:
        lines = [
            f"Upload Report: {self.manifest.source_dir} → {self.manifest.bucket_alias}",
            f"  Total files: {self.total}",
            f"  Uploaded:    {self.uploaded}",
            f"  Failed:      {self.failed}",
        ]
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors[:10]:
                lines.append(f"    - {e}")
            if len(self.errors) > 10:
                lines.append(f"    ... and {len(self.errors) - 10} more")
        return "\n".join(lines)
