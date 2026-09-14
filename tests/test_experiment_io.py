"""Tests for B9 — Experiment Export/Import."""

from pathlib import Path

import pytest
from umrs.shared.experiment_io import (
    DataObjectRef,
    ExperimentExporter,
    ExperimentImporter,
    ExperimentManifest,
)


class TestExperimentManifest:
    def test_roundtrip(self):
        m = ExperimentManifest(
            experiment_id="exp-123",
            name="test-exp",
            user_id="u1",
            tags=["test", "demo"],
        )
        d = m.to_dict()
        m2 = ExperimentManifest.from_dict(d)
        assert m2.experiment_id == "exp-123"
        assert m2.name == "test-exp"
        assert m2.tags == ["test", "demo"]


class TestDataObjectRef:
    def test_roundtrip(self):
        r = DataObjectRef(key="data/file.csv", bucket="b1", etag="abc", size_bytes=1024)
        d = r.to_dict()
        r2 = DataObjectRef.from_dict(d)
        assert r2.key == "data/file.csv"
        assert r2.size_bytes == 1024


class TestExperimentExporter:
    def test_export_minimal(self, tmp_path):
        exporter = ExperimentExporter()
        archive_path = exporter.export(
            name="minimal-exp",
            output_dir=str(tmp_path),
        )
        assert Path(archive_path).exists()
        assert archive_path.endswith(".zip")

    def test_export_with_all_fields(self, tmp_path):
        exporter = ExperimentExporter()
        archive_path = exporter.export(
            name="full-exp",
            user_id="u1",
            project_id="p1",
            description="A full experiment",
            tags=["tag1"],
            config={"model": "e5-base", "batch_size": 32},
            data_objects=[
                DataObjectRef(key="a.txt", bucket="b1", etag="e1", size_bytes=100),
                DataObjectRef(key="b.txt", bucket="b1", etag="e2", size_bytes=200),
            ],
            results={"accuracy": 0.95, "loss": 0.05},
            pipeline_log=[{"step": "embed", "duration": 1.5}],
            extra_files={"notes.txt": b"some notes"},
            output_dir=str(tmp_path),
        )
        assert Path(archive_path).exists()

    def test_export_to_bytes(self):
        exporter = ExperimentExporter()
        data = exporter.export_to_bytes(
            name="bytes-exp",
            config={"key": "value"},
            results={"score": 0.9},
        )
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_safe_filename(self, tmp_path):
        exporter = ExperimentExporter()
        archive_path = exporter.export(
            name="test/with spaces!",
            output_dir=str(tmp_path),
        )
        path = Path(archive_path)
        assert "/" not in path.name or path.name.count("/") == 0
        assert path.exists()


class TestExperimentImporter:
    def test_import_archive(self, tmp_path):
        # Export first
        exporter = ExperimentExporter()
        archive_path = exporter.export(
            name="import-test",
            user_id="u1",
            config={"model": "clip"},
            data_objects=[
                DataObjectRef(key="x.txt", bucket="b1", etag="e1", size_bytes=50),
            ],
            results={"score": 0.8},
            pipeline_log=[{"step": "s1", "ok": True}],
            extra_files={"extra.txt": b"extra data"},
            output_dir=str(tmp_path),
        )

        # Import
        importer = ExperimentImporter()
        archive = importer.import_archive(archive_path)

        assert archive.manifest.name == "import-test"
        assert archive.manifest.user_id == "u1"
        assert archive.config["model"] == "clip"
        assert len(archive.data_manifest) == 1
        assert archive.results["score"] == 0.8
        assert len(archive.pipeline_log) == 1
        assert "extra.txt" in archive.extra_files

    def test_import_from_bytes(self):
        exporter = ExperimentExporter()
        data = exporter.export_to_bytes(
            name="bytes-test",
            config={"a": 1},
        )

        importer = ExperimentImporter()
        archive = importer.import_from_bytes(data)
        assert archive.manifest.name == "bytes-test"
        assert archive.config["a"] == 1

    def test_import_nonexistent(self):
        importer = ExperimentImporter()
        with pytest.raises(FileNotFoundError):
            importer.import_archive("/nonexistent/archive.zip")

    def test_validate_valid_archive(self, tmp_path):
        exporter = ExperimentExporter()
        path = exporter.export(name="valid", output_dir=str(tmp_path))

        importer = ExperimentImporter()
        errors = importer.validate_archive(path)
        assert len(errors) == 0

    def test_validate_nonexistent(self):
        importer = ExperimentImporter()
        errors = importer.validate_archive("/no/such/file.zip")
        assert len(errors) > 0

    def test_validate_not_zip(self, tmp_path):
        bad_file = tmp_path / "not_a_zip.zip"
        bad_file.write_text("this is not a zip file")

        importer = ExperimentImporter()
        errors = importer.validate_archive(str(bad_file))
        assert len(errors) > 0

    def test_roundtrip_preserves_data(self, tmp_path):
        """Full round-trip: export → import → verify all fields."""
        exporter = ExperimentExporter()
        config = {"model": "e5", "dim": 768}
        results = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        tags = ["experiment", "test"]

        path = exporter.export(
            name="roundtrip",
            user_id="user1",
            project_id="proj1",
            description="Round trip test",
            tags=tags,
            config=config,
            results=results,
            output_dir=str(tmp_path),
        )

        importer = ExperimentImporter()
        archive = importer.import_archive(path)

        assert archive.manifest.name == "roundtrip"
        assert archive.manifest.user_id == "user1"
        assert archive.manifest.description == "Round trip test"
        assert archive.manifest.tags == tags
        assert archive.config == config
        assert archive.results == results
