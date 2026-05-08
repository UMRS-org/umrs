"""
B9 — Experiment Export/Import

Export full experiment state (config, data manifest, results, embeddings)
to a portable ZIP archive.  Import archives to reproduce experiments on
different UMRS instances.
"""

from __future__ import annotations

import json
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass
class ExperimentManifest:
    """Metadata for an exported experiment."""
    experiment_id: str
    name: str
    user_id: str = ""
    project_id: str = ""
    created_at: float = field(default_factory=time.time)
    umrs_version: str = "1.0"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    config_hash: str = ""
    data_object_count: int = 0
    result_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentManifest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DataObjectRef:
    """Reference to a data object in an experiment."""
    key: str
    bucket: str
    etag: str = ""
    size_bytes: int = 0
    content_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataObjectRef:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentArchive:
    """Represents a complete experiment archive."""
    manifest: ExperimentManifest
    config: dict[str, Any] = field(default_factory=dict)
    data_manifest: list[DataObjectRef] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    pipeline_log: list[dict[str, Any]] = field(default_factory=list)
    extra_files: dict[str, bytes] = field(default_factory=dict)


class ExperimentExporter:
    """Serialize experiment to ZIP archive.

    Archive Structure::

        experiment_<name>_<timestamp>.zip
        ├── manifest.json
        ├── config/
        │   └── user_config.json
        ├── data/
        │   └── data_manifest.json
        ├── results/
        │   └── results.json
        ├── metadata/
        │   └── pipeline_log.json
        └── extra/
            └── <user files>

    Usage::

        exporter = ExperimentExporter()
        archive_path = exporter.export(
            name="hmae-run-001",
            user_id="user1",
            config=user_config,
            data_objects=[...],
            results={"embeddings": [...], "scores": [...]},
            output_dir="/exports",
        )
    """

    def export(
        self,
        name: str,
        user_id: str = "",
        project_id: str = "",
        description: str = "",
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        data_objects: list[DataObjectRef] | None = None,
        results: dict[str, Any] | None = None,
        pipeline_log: list[dict[str, Any]] | None = None,
        extra_files: dict[str, bytes] | None = None,
        output_dir: str = ".",
    ) -> str:
        """Export experiment to a ZIP file. Returns archive path."""
        experiment_id = f"exp-{uuid.uuid4().hex[:12]}"
        timestamp = int(time.time())

        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            name=name,
            user_id=user_id,
            project_id=project_id,
            created_at=time.time(),
            description=description,
            tags=tags or [],
            data_object_count=len(data_objects or []),
            result_count=len(results or {}),
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        archive_name = f"experiment_{safe_name}_{timestamp}.zip"
        archive_path = out_dir / archive_name

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # manifest.json
            zf.writestr("manifest.json", json.dumps(manifest.to_dict(), indent=2, default=str))

            # config/
            if config:
                zf.writestr("config/user_config.json", json.dumps(config, indent=2, default=str))

            # data/
            if data_objects:
                data_manifest = [o.to_dict() for o in data_objects]
                zf.writestr("data/data_manifest.json", json.dumps(data_manifest, indent=2, default=str))

            # results/
            if results:
                zf.writestr("results/results.json", json.dumps(results, indent=2, default=str))

            # metadata/
            if pipeline_log:
                zf.writestr("metadata/pipeline_log.json", json.dumps(pipeline_log, indent=2, default=str))

            # extra/
            if extra_files:
                for filename, data in extra_files.items():
                    zf.writestr(f"extra/{filename}", data)

        return str(archive_path)

    def export_to_bytes(
        self,
        name: str,
        **kwargs: Any,
    ) -> bytes:
        """Export experiment to in-memory bytes (for streaming)."""
        buf = BytesIO()
        experiment_id = f"exp-{uuid.uuid4().hex[:12]}"

        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            name=name,
            user_id=kwargs.get("user_id", ""),
            project_id=kwargs.get("project_id", ""),
            created_at=time.time(),
            description=kwargs.get("description", ""),
            tags=kwargs.get("tags", []),
        )

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest.to_dict(), indent=2, default=str))

            config = kwargs.get("config")
            if config:
                zf.writestr("config/user_config.json", json.dumps(config, indent=2, default=str))

            results = kwargs.get("results")
            if results:
                zf.writestr("results/results.json", json.dumps(results, indent=2, default=str))

        return buf.getvalue()


class ExperimentImporter:
    """Reconstitute experiment from a ZIP archive.

    Usage::

        importer = ExperimentImporter()
        archive = importer.import_archive("/exports/experiment_hmae_1712345678.zip")
        print(archive.manifest.name)
        print(archive.config)
    """

    def import_archive(self, archive_path: str) -> ExperimentArchive:
        """Import experiment from a ZIP file."""
        path = Path(archive_path)
        if not path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        with zipfile.ZipFile(path, "r") as zf:
            return self._read_archive(zf)

    def import_from_bytes(self, data: bytes) -> ExperimentArchive:
        """Import experiment from in-memory bytes."""
        buf = BytesIO(data)
        with zipfile.ZipFile(buf, "r") as zf:
            return self._read_archive(zf)

    def _read_archive(self, zf: zipfile.ZipFile) -> ExperimentArchive:
        manifest_data = json.loads(zf.read("manifest.json"))
        manifest = ExperimentManifest.from_dict(manifest_data)

        config: dict[str, Any] = {}
        data_manifest: list[DataObjectRef] = []
        results: dict[str, Any] = {}
        pipeline_log: list[dict[str, Any]] = []
        extra_files: dict[str, bytes] = {}

        if "config/user_config.json" in zf.namelist():
            config = json.loads(zf.read("config/user_config.json"))

        if "data/data_manifest.json" in zf.namelist():
            objs = json.loads(zf.read("data/data_manifest.json"))
            data_manifest = [DataObjectRef.from_dict(o) for o in objs]

        if "results/results.json" in zf.namelist():
            results = json.loads(zf.read("results/results.json"))

        if "metadata/pipeline_log.json" in zf.namelist():
            pipeline_log = json.loads(zf.read("metadata/pipeline_log.json"))

        for name in zf.namelist():
            if name.startswith("extra/") and not name.endswith("/"):
                extra_files[name[6:]] = zf.read(name)

        return ExperimentArchive(
            manifest=manifest,
            config=config,
            data_manifest=data_manifest,
            results=results,
            pipeline_log=pipeline_log,
            extra_files=extra_files,
        )

    def validate_archive(self, archive_path: str) -> list[str]:
        """Validate archive structure. Returns list of errors."""
        errors: list[str] = []
        path = Path(archive_path)

        if not path.exists():
            return [f"Archive not found: {archive_path}"]

        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "manifest.json" not in zf.namelist():
                    errors.append("Missing manifest.json")
                else:
                    try:
                        data = json.loads(zf.read("manifest.json"))
                        if "experiment_id" not in data:
                            errors.append("manifest.json missing 'experiment_id'")
                        if "name" not in data:
                            errors.append("manifest.json missing 'name'")
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid manifest.json: {e}")

                # Check for bad zip entries
                bad = zf.testzip()
                if bad:
                    errors.append(f"Corrupted file in archive: {bad}")

        except zipfile.BadZipFile:
            errors.append("File is not a valid ZIP archive")

        return errors
