"""Cycle-3 hard lane, N6 (2026-10-21): a catalogued object that vanished out
of band is an honest typed error, and the remedy it names works.

Found live (lane stack, repro ``.stab-cycle3/hard/repro/n6_vanished.py``):
upload a text file, let the catalog ingest it, delete the object and its
extracted-text copy straight from the object store (an operator's ``rm -r``),
then POST /umrs/knowledge/understand and /verify: both answered 500
``knowledge.failure`` "An error occurred (404) when calling the HeadObject
operation". The catalog row had outlived its bytes; nothing named the object
or what to do. And the remedy, DELETE /umrs/files/{key}, could not be reached
for any uploaded file: every upload key has a slash (``uploads/<id>_name``)
and the route was ``/{file_id}``, so the router answered 404.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import files, knowledge
from umrs.data_catalog import CatalogObjectMissing, load_artifact_content
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState
from umrs.knowledge.project import ResearchProject

PROJECT = "pv"
KEY = f"{PROJECT}/uploads/0b74_notes.txt"


def _not_found(op: str) -> ClientError:
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, op)


class _GoneS3:
    """An object store the object was deleted from."""

    def head_object(self, **_kw: Any) -> dict[str, Any]:
        raise _not_found("HeadObject")

    def get_object(self, **_kw: Any) -> dict[str, Any]:
        raise _not_found("GetObject")


class _BrokenS3(_GoneS3):
    def head_object(self, **_kw: Any) -> dict[str, Any]:
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")


class _Metas:
    def __init__(self, meta: CatalogObjectMeta) -> None:
        self._meta = meta

    def list_by_project(self, *_a: Any, **_kw: Any) -> list[CatalogObjectMeta]:
        return [self._meta]

    def get(self, _project: str, s3_key: str) -> CatalogObjectMeta | None:
        return self._meta if s3_key == self._meta.s3_key else None


def _meta() -> CatalogObjectMeta:
    return CatalogObjectMeta(
        s3_key=KEY,
        object_etag="e",
        state=CatalogState.READY,
        owner="alice",
        acl_scope="project",
        updated_at="2026-10-21T00:00:00Z",
        detected_modality="text",
        extracted_text_ref="_data_catalog/extracted/gone.txt",
    )


def test_vanished_object_is_catalog_object_missing() -> None:
    with pytest.raises(CatalogObjectMissing) as caught:
        load_artifact_content(_GoneS3(), "b", KEY, _meta())
    assert caught.value.s3_key == KEY
    assert KEY in str(caught.value)


def test_other_store_faults_stay_what_they_are() -> None:
    with pytest.raises(ClientError):
        load_artifact_content(_BrokenS3(), "b", KEY, _meta())


@pytest.fixture
def knowledge_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def _build(project_id: str, _config: Any) -> ResearchProject:
        return ResearchProject(
            project_id=project_id,
            s3_client=_GoneS3(),
            s3_bucket="b",
            meta_store=_Metas(_meta()),
        )

    monkeypatch.setattr(knowledge, "build_research_project", _build)
    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[files.get_s3_config] = lambda: object()
    app.dependency_overrides[files.get_project_id] = lambda: PROJECT
    return TestClient(app)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/umrs/knowledge/understand", {}),
        ("/umrs/knowledge/verify", {"question": "What is the objective?"}),
    ],
)
def test_understand_and_verify_name_the_vanished_object(
    knowledge_client: TestClient, path: str, body: dict[str, str]
) -> None:
    r = knowledge_client.post(path, json=body)
    assert r.status_code == 404, r.text
    assert r.headers["x-platform-error-code"] == "knowledge.not_found"
    assert KEY in r.text
    assert "DELETE /umrs/files/" in r.text


class _DeletingS3:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_object(self, **kw: str) -> None:
        self.deleted.append(kw["Key"])


class _Config:
    bucket = "b"

    def __init__(self, client: _DeletingS3) -> None:
        self._client = client

    def get_client(self) -> _DeletingS3:
        return self._client


def test_an_uploaded_file_with_a_nested_key_can_be_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _DeletingS3()
    dropped: list[tuple[str, str]] = []

    class _Catalog:
        def delete(self, project_id: str, key: str) -> bool:
            dropped.append((project_id, key))
            return True

    monkeypatch.setattr(
        "umrs.api.catalog_store.get_optional_catalog_meta_store", lambda: _Catalog()
    )
    app = FastAPI()
    app.include_router(files.router)
    app.dependency_overrides[files.get_project_id] = lambda: PROJECT
    app.dependency_overrides[files.get_project_s3_config] = lambda: _Config(s3)
    r = TestClient(app).delete("/umrs/files/uploads/0b74_notes.txt")
    assert r.status_code == 204, r.text
    assert s3.deleted == [KEY]
    assert dropped == [(PROJECT, KEY)]
