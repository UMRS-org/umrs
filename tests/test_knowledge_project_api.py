"""HTTP tests for RC1 knowledge workflow endpoints."""

from __future__ import annotations

import pytest

# Splice/stub machinery removed (ADR-053): umrs.api.__init__ is a lazy
# __getattr__ shim since Phase 1.3, so plain module imports are cycle-safe.


def _ensure_umrs_namespace() -> None:
    """No-op retained for call sites; plain imports happen at module top."""


def _load_api_module(qualname: str, filename: str):
    # Back-compat shim: plain import via the lazy umrs.api package.
    return getattr(__import__("umrs.api", fromlist=[filename.removesuffix(".py")]), filename.removesuffix(".py"))


def _knowledge_api_deps():
    pytest.importorskip("fastapi")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except ModuleNotFoundError as exc:
        pytest.skip(f"fastapi is not usable in this environment: {exc}")

    _ensure_umrs_namespace()
    files_module = _load_api_module("umrs.api.files", "files.py")
    knowledge_module = _load_api_module("umrs.api.knowledge", "knowledge.py")
    # ADR-026 wave C: the sat facade module no longer exists; knowledge/
    # canonical graph surface is the only graph face.
    return FastAPI, TestClient, files_module, knowledge_module


class _LocalS3Config:
    bucket = ""

    def get_client(self):
        return None


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "sample.txt").write_text(
        "The goal of this paper is to test HTTP understand.\n"
        "Keywords: http, understand, api\n"
    )
    return papers


@pytest.fixture
def knowledge_client(project_dir: Path, monkeypatch):
    FastAPI, TestClient, files_module, knowledge_module = _knowledge_api_deps()
    get_s3_config = files_module.get_s3_config
    get_project_id = files_module.get_project_id

    from umrs.knowledge.project import ResearchProject

    def _build(project_id: str, config: _LocalS3Config) -> ResearchProject:
        project = ResearchProject(project_id=project_id)
        project.ingest_directory(project_dir)
        return project

    monkeypatch.setattr(knowledge_module, "build_research_project", _build)

    app = FastAPI()
    app.include_router(knowledge_module.router)
    app.dependency_overrides[get_s3_config] = lambda: _LocalS3Config()
    app.dependency_overrides[get_project_id] = lambda: "proj-http"
    return TestClient(app)


def test_http_sidecar_exposes_knowledge_routes():
    _ensure_umrs_namespace()
    knowledge_module = _load_api_module("umrs.api.knowledge", "knowledge.py")

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(knowledge_module.router)
    paths = app.openapi()["paths"]
    assert "post" in paths["/umrs/knowledge/understand"]
    assert "post" in paths["/umrs/knowledge/verify"]
    assert "/umrs/knowledge-graph/query" not in paths


def test_http_understand_returns_graph_stats(knowledge_client):
    response = knowledge_client.post(
        "/umrs/knowledge/understand",
        headers={"X-Project-ID": "proj-http"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "proj-http"
    assert body["phase"] == "ready"
    assert body["artifact_count"] >= 1
    assert body["extracted_count"] >= 1
    assert body["graph_nodes"] >= 1


def test_http_verify_returns_answer(knowledge_client):
    knowledge_client.post(
        "/umrs/knowledge/understand",
        headers={"X-Project-ID": "proj-http"},
    )

    response = knowledge_client.post(
        "/umrs/knowledge/verify",
        headers={"X-Project-ID": "proj-http"},
        json={"question": "Explain my project"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Explain my project"
    assert isinstance(body["answer"], str)


def test_http_canonical_graph_still_serves(knowledge_client):
    """sat facade retired entirely (ADR-026 wave C): canonical graph intact."""
    knowledge_client.post(
        "/umrs/knowledge/understand",
        headers={"X-Project-ID": "proj-http"},
    )

    canonical = knowledge_client.get(
        "/umrs/knowledge/graph",
        headers={"X-Project-ID": "proj-http"},
    )
    assert canonical.status_code != 404
