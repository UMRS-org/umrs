"""Cycle-3 mem lane (2026-10-22): GET /umrs/files/{key}/metadata reaches the
keys /upload writes.

Every key /upload writes has a slash (``<project>/uploads/<id>_<name>``) and
the route was ``/{file_id}/metadata``, which never matches one: the router
answered 404 for every uploaded file, encoded or not. DELETE had the same
bug (fixed in fa99c60a). A download of a key ending in ``/metadata`` stays a
download.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import files

PROJECT = "pv"
KEY = f"{PROJECT}/uploads/0b74_notes.txt"


class _S3:
    def __init__(self) -> None:
        self.heads: list[str] = []
        self.gets: list[str] = []

    def head_object(self, **kw: str) -> dict[str, Any]:
        self.heads.append(kw["Key"])
        return {
            "ContentLength": 5,
            "ContentType": "text/plain",
            "LastModified": dt.datetime(2026, 10, 22, tzinfo=dt.timezone.utc),
            "ETag": '"e"',
            "Metadata": {},
        }

    def get_object(self, **kw: str) -> dict[str, Any]:
        self.gets.append(kw["Key"])
        return {"Body": _Body(), "ContentType": "text/plain", "ContentLength": 5}


class _Body:
    def __init__(self) -> None:
        self._left = b"hello"

    def read(self, *_a: Any) -> bytes:
        out, self._left = self._left, b""
        return out

    def close(self) -> None:
        return None


class _Config:
    bucket = "b"

    def __init__(self, client: _S3) -> None:
        self._client = client

    def get_client(self) -> _S3:
        return self._client


def _client(s3: _S3) -> TestClient:
    app = FastAPI()
    app.include_router(files.router)
    app.dependency_overrides[files.get_project_id] = lambda: PROJECT
    app.dependency_overrides[files.get_project_s3_config] = lambda: _Config(s3)
    return TestClient(app)


def test_an_uploaded_file_with_a_nested_key_has_metadata() -> None:
    s3 = _S3()
    r = _client(s3).get("/umrs/files/uploads/0b74_notes.txt/metadata")
    assert r.status_code == 200, r.text
    assert r.json()["key"] == KEY
    assert r.json()["size"] == 5
    assert s3.heads == [KEY]


def test_an_encoded_nested_key_has_metadata() -> None:
    s3 = _S3()
    r = _client(s3).get("/umrs/files/uploads%2F0b74_notes.txt/metadata")
    assert r.status_code == 200, r.text
    assert s3.heads == [KEY]


def test_a_flat_key_still_has_metadata() -> None:
    s3 = _S3()
    r = _client(s3).get("/umrs/files/notes.txt/metadata")
    assert r.status_code == 200, r.text
    assert s3.heads == [f"{PROJECT}/notes.txt"]


def test_a_download_of_a_key_ending_in_metadata_stays_a_download() -> None:
    s3 = _S3()
    r = _client(s3).get("/umrs/files/download/uploads/metadata")
    assert r.status_code == 200, r.text
    assert r.content == b"hello"
    assert s3.gets == [f"{PROJECT}/uploads/metadata"]
