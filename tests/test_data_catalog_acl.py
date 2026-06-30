"""Cross-project ACL tests for the data catalog (search + read + API)."""

from __future__ import annotations

import io
import json

import pytest

from umrs.data_catalog.paths import meta_object_key
from umrs.data_catalog.query import read_catalog_object, search_catalog
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState


def _meta(
    s3_key: str,
    *,
    acl_scope: str,
    summary: str = "",
) -> CatalogObjectMeta:
    return CatalogObjectMeta(
        s3_key=s3_key,
        object_etag="etag",
        state=CatalogState.READY,
        owner="user-a",
        acl_scope=acl_scope,
        updated_at="2026-06-30T00:00:00Z",
        summary=summary or s3_key.rsplit("/", 1)[-1],
    )


class _AclS3Client:
    """Minimal S3 stub for catalog list/search/read ACL tests."""

    def __init__(
        self,
        metas: list[CatalogObjectMeta],
        *,
        object_bodies: dict[str, bytes] | None = None,
    ):
        self._metas = {m.s3_key: m for m in metas}
        self._object_bodies = object_bodies or {
            key: b"payload-bytes" for key in self._metas
        }
        meta_keys = [meta_object_key(key) for key in self._metas]
        self._list_pages = [{"Contents": [{"Key": key} for key in meta_keys]}]

    def get_paginator(self, operation_name: str):
        assert operation_name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803
        assert Prefix.endswith("objects/")
        return iter(self._list_pages)

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):  # noqa: N803
        from botocore.exceptions import ClientError

        for s3_key, meta in self._metas.items():
            if Key == meta_object_key(s3_key):
                return {"Body": io.BytesIO(json.dumps(meta.to_dict()).encode("utf-8"))}

        if Key.endswith(".meta.json"):
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                "GetObject",
            )

        body = self._object_bodies.get(Key, b"")
        if Range is None:
            return {"Body": io.BytesIO(body)}

        start_raw, end_raw = Range.removeprefix("bytes=").split("-", 1)
        start = int(start_raw)
        end = int(end_raw)
        return {"Body": io.BytesIO(body[start : end + 1])}

    def head_object(self, *, Bucket: str, Key: str):  # noqa: N803
        body = self._object_bodies.get(Key, b"")
        return {"ContentLength": len(body), "ETag": '"etag"'}


# ── search_catalog ACL ───────────────────────────────────────────────


def test_search_catalog_returns_only_callers_project():
    proj1 = _meta("proj-1/uploads/protein.csv", acl_scope="proj-1", summary="protein yield")
    proj2 = _meta("proj-2/uploads/secret.csv", acl_scope="proj-2", summary="protein secret")
    client = _AclS3Client([proj1, proj2])

    hits = search_catalog(client, "bucket", query="protein", project_id="proj-1")

    assert len(hits) == 1
    assert hits[0].meta.s3_key == "proj-1/uploads/protein.csv"


def test_search_catalog_empty_for_unrelated_project():
    other = _meta("proj-2/uploads/data.csv", acl_scope="proj-2", summary="batch results")
    client = _AclS3Client([other])

    hits = search_catalog(client, "bucket", query="batch", project_id="proj-1")

    assert hits == []


def test_search_catalog_allows_prefix_match_when_acl_scope_differs():
    """Objects under proj-1/ prefix are visible to proj-1 even if acl_scope is stale."""
    meta = _meta("proj-1/uploads/stale-acl.csv", acl_scope="legacy-scope", summary="stale acl")
    client = _AclS3Client([meta])

    hits = search_catalog(client, "bucket", query="stale", project_id="proj-1")

    assert len(hits) == 1
    assert hits[0].meta.s3_key == "proj-1/uploads/stale-acl.csv"


# ── read_catalog_object ACL ──────────────────────────────────────────


def test_read_catalog_object_denied_for_other_project():
    meta = _meta("proj-2/private.csv", acl_scope="proj-2")
    client = _AclS3Client([meta])

    with pytest.raises(PermissionError, match="access denied"):
        read_catalog_object(
            client,
            "bucket",
            s3_key="proj-2/private.csv",
            project_id="proj-1",
        )


def test_read_catalog_object_denied_when_acl_scope_matches_but_key_prefix_does_not():
    """acl_scope alone is not enough — object key must live under project prefix."""
    meta = _meta("proj-2/mis-tagged.csv", acl_scope="proj-1")
    client = _AclS3Client([meta])

    with pytest.raises(PermissionError, match="access denied"):
        read_catalog_object(
            client,
            "bucket",
            s3_key="proj-2/mis-tagged.csv",
            project_id="proj-1",
        )


def test_read_catalog_object_allowed_for_own_project():
    meta = _meta("proj-1/uploads/readable.csv", acl_scope="proj-1")
    client = _AclS3Client([meta], object_bodies={"proj-1/uploads/readable.csv": b"hello"})

    payload = read_catalog_object(
        client,
        "bucket",
        s3_key="proj-1/uploads/readable.csv",
        project_id="proj-1",
    )

    assert payload["text"] == "hello"
    assert payload["meta"]["acl_scope"] == "proj-1"


def test_read_catalog_object_denied_when_meta_missing():
    client = _AclS3Client([])

    with pytest.raises(PermissionError, match="access denied"):
        read_catalog_object(
            client,
            "bucket",
            s3_key="proj-1/uploads/missing.csv",
            project_id="proj-1",
        )


# ── HTTP API ACL ─────────────────────────────────────────────────────


def _catalog_api_deps():
    pytest.importorskip("fastapi")
    import importlib.util
    import sys
    import types
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    api_dir = Path(__file__).resolve().parents[1] / "src" / "umrs" / "api"

    def _load_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # Isolate catalog + files routers from umrs.api.__init__ (heavy imports).
    sys.modules.setdefault("umrs", types.ModuleType("umrs"))
    sys.modules.setdefault("umrs.api", types.ModuleType("umrs.api"))
    files_module = _load_module("umrs.api.files", api_dir / "files.py")
    catalog_module = _load_module("umrs.api.catalog", api_dir / "catalog.py")
    catalog_module.CatalogReadResponse.model_rebuild()
    catalog_router = catalog_module.router
    S3Config = files_module.S3Config
    get_s3_config = files_module.get_s3_config

    class _StubS3Config(S3Config):
        def __init__(self, client: _AclS3Client):
            super().__init__()
            self._client = client
            self.bucket = "test-bucket"

        def get_client(self):
            return self._client

    def _catalog_test_app(s3_client: _AclS3Client) -> FastAPI:
        app = FastAPI()
        app.include_router(catalog_router)
        app.dependency_overrides[get_s3_config] = lambda: _StubS3Config(s3_client)
        return app

    return FastAPI, TestClient, _catalog_test_app


def test_api_catalog_search_scoped_by_x_project_id_header():
    FastAPI, TestClient, _catalog_test_app = _catalog_api_deps()
    proj1 = _meta("proj-1/uploads/a.csv", acl_scope="proj-1", summary="alpha dataset")
    proj2 = _meta("proj-2/uploads/b.csv", acl_scope="proj-2", summary="alpha secret")
    app = _catalog_test_app(_AclS3Client([proj1, proj2]))

    with TestClient(app) as client:
        response = client.get(
            "/umrs/catalog/search",
            params={"q": "alpha"},
            headers={"X-Project-ID": "proj-1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "proj-1"
    assert len(body["results"]) == 1
    assert body["results"][0]["s3_key"] == "proj-1/uploads/a.csv"


def test_api_catalog_read_returns_403_for_cross_project_key():
    _, TestClient, _catalog_test_app = _catalog_api_deps()
    meta = _meta("proj-2/private.csv", acl_scope="proj-2")
    app = _catalog_test_app(_AclS3Client([meta]))

    with TestClient(app) as client:
        response = client.get(
            "/umrs/catalog/read",
            params={"s3_key": "proj-2/private.csv"},
            headers={"X-Project-ID": "proj-1"},
        )

    assert response.status_code == 403
    assert "access denied" in response.json()["detail"].lower()


def test_api_catalog_read_returns_200_for_own_project_key():
    _, TestClient, _catalog_test_app = _catalog_api_deps()
    meta = _meta("proj-1/uploads/ok.csv", acl_scope="proj-1")
    app = _catalog_test_app(
        _AclS3Client([meta], object_bodies={"proj-1/uploads/ok.csv": b"ok-data"})
    )

    with TestClient(app) as client:
        response = client.get(
            "/umrs/catalog/read",
            params={"s3_key": "proj-1/uploads/ok.csv"},
            headers={"X-Project-ID": "proj-1"},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "ok-data"
