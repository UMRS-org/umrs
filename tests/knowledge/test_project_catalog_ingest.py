"""Tests for ResearchProject S3 + catalog ingestion path."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from umrs.data_catalog.constants import EXTRACTED_TEXT_PREFIX
from umrs.data_catalog.paths import meta_object_key
from umrs.data_catalog.schema import CatalogObjectMeta
from umrs.knowledge.ingest import IngesterConfig
from umrs.knowledge.local_ingest import (
    build_s3_key,
    classify_local_file,
    should_skip_relative_path,
    upload_local_files,
)
from umrs.knowledge.project import ProjectPhase, ResearchProject
from umrs.knowledge.workflow import run_project_workflow


class _FakeProjectS3:
    """Minimal in-memory S3 client for upload → catalog → project ingest tests."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self._etags: dict[str, str] = {}
        self._counter = 0

    def put_object(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        Body: bytes | str,
        ContentType: str | None = None,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._counter += 1
        etag = f"etag-{self._counter}"
        self.objects[Key] = Body.encode("utf-8") if isinstance(Body, str) else Body
        self._etags[Key] = etag
        return {"ETag": f'"{etag}"'}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self.objects:
            raise _no_such_key(Key)
        return {
            "ContentLength": len(self.objects[Key]),
            "ETag": f'"{self._etags[Key]}"',
        }

    def get_object(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        Range: str | None = None,
    ) -> dict[str, Any]:
        if Key not in self.objects:
            raise _no_such_key(Key)
        content = self.objects[Key]
        if Range is not None:
            start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            content = content[start : end + 1]
        return {"Body": io.BytesIO(content)}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _FakePaginator(self.objects, self._etags)


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes], etags: dict[str, str]) -> None:
        self.objects = objects
        self.etags = etags

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803
        catalog_list = Prefix.startswith("_data_catalog/")
        contents = [
            {"Key": key, "ETag": f'"{self.etags[key]}"', "Size": len(data)}
            for key, data in sorted(self.objects.items())
            if key.startswith(Prefix)
            and (catalog_list or not key.startswith("_data_catalog/"))
        ]
        return [{"Contents": contents}]


def _no_such_key(key: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": f"missing {key}"}},
        "GetObject",
    )


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
  (tmp_path / "README.md").write_text("# Demo\n")
  papers = tmp_path / "papers"
  papers.mkdir()
  (papers / "paper1.pdf").write_bytes(b"%PDF-1.4\nobjective: test paper\n")
  code = tmp_path / "code"
  code.mkdir()
  (code / "model.py").write_text("import torch\n")
  data = tmp_path / "data"
  data.mkdir()
  (data / "run.csv").write_text("x,y\n1,2\n")
  git_dir = tmp_path / ".git"
  git_dir.mkdir()
  (git_dir / "config").write_text("should be skipped\n")
  return tmp_path


def test_classify_and_skip_helpers():
    assert classify_local_file(Path("a.ipynb")) == "notebook"
    assert classify_local_file(Path("a.exe")) is None
    assert should_skip_relative_path(Path(".git/config")) is True
    assert should_skip_relative_path(Path("papers/a.pdf")) is False


def test_build_s3_key_preserves_relative_layout(tmp_path: Path):
    base = tmp_path / "root"
    papers = base / "papers"
    papers.mkdir(parents=True)
    pdf = papers / "paper1.pdf"
    pdf.write_bytes(b"pdf")
    assert build_s3_key("proj-1", base, pdf) == "proj-1/uploads/papers/paper1.pdf"


def test_upload_local_files_skips_git(project_dir: Path):
    client = _FakeProjectS3()
    uploaded = upload_local_files(client, "bucket", "proj-1", project_dir)
    keys = {item.s3_key for item in uploaded}
    assert "proj-1/uploads/README.md" in keys
    assert "proj-1/uploads/papers/paper1.pdf" in keys
    assert "proj-1/uploads/code/model.py" in keys
    assert "proj-1/uploads/data/run.csv" in keys
    assert not any(".git" in key for key in keys)


def test_ingest_directory_via_catalog(project_dir: Path):
    client = _FakeProjectS3()
    project = ResearchProject(
        project_id="proj-1",
        s3_client=client,
        s3_bucket="bucket",
        owner_id="user-1",
        ingester_config=IngesterConfig(max_extract_bytes=1024 * 1024),
    )

    count = project.ingest_directory(project_dir)

    assert count == 4
    assert project.phase == ProjectPhase.INGESTING
    artifacts = project.knowledge_store.list_artifacts()
    assert len(artifacts) == count
    for artifact in artifacts:
        assert artifact["metadata"]["s3_key"].startswith("proj-1/uploads/")
        assert artifact["metadata"]["catalog_ref"] == artifact["metadata"]["s3_key"]
        loaded = project.knowledge_store.get_artifact(artifact["artifact_id"])
        assert loaded is not None
        assert loaded["content"]

    meta_keys = [key for key in client.objects if key.endswith(".meta.json")]
    assert len(meta_keys) == count
    for meta_key in meta_keys:
        meta = CatalogObjectMeta.from_dict(
            json.loads(client.objects[meta_key].decode("utf-8"))
        )
        assert meta.state.value == "ready"


def test_ingest_directory_local_fallback_without_s3(project_dir: Path):
    project = ResearchProject(project_id="proj-1")
    count = project.ingest_directory(project_dir)
    assert count == 4
    assert all(
        "s3_key" not in artifact.get("metadata", {})
        for artifact in project.knowledge_store.list_artifacts()
    )


def test_understand_after_catalog_ingest(project_dir: Path):
    client = _FakeProjectS3()
    project = ResearchProject(
        project_id="proj-1",
        s3_client=client,
        s3_bucket="bucket",
        owner_id="user-1",
    )
    project.ingest_directory(project_dir)
    project.understand()

    assert project.phase == ProjectPhase.READY
    assert len(project.knowledge_graph.nodes) > 0
    assert project.coverage_dashboard is not None


def test_verify_after_catalog_ingest(project_dir: Path):
    client = _FakeProjectS3()
    project = ResearchProject(
        project_id="proj-1",
        s3_client=client,
        s3_bucket="bucket",
        owner_id="user-1",
    )
    project.ingest_directory(project_dir)
    project.understand()

    answer = project.verify("What code files do we have?")
    assert answer.answered
    assert answer.confidence > 0


def test_run_project_workflow_uses_catalog_when_s3_configured(project_dir: Path):
    """run_project_workflow passes S3 config through to catalog ingest."""
    client = _FakeProjectS3()
    result = run_project_workflow(
        "proj-1",
        project_dir,
        auto_approve=True,
        max_iterations=1,
        s3_client=client,
        s3_bucket="bucket",
        user_id="user-1",
    )

    assert result.success
    assert result.artifacts_ingested > 0
    for artifact in result.project.knowledge_store.list_artifacts():
        assert artifact["metadata"]["s3_key"].startswith("proj-1/uploads/")
    meta_keys = [k for k in client.objects if k.endswith(".meta.json")]
    assert len(meta_keys) == result.artifacts_ingested
