"""files/upload lane metering pins (ADR-051 D-3, remaining-lanes leg 2026-09-16).

Direct server-mediated uploads carry known bytes — the lane charges once per
unique content hash per uploader, and a refusal un-stages the object it just
wrote (cleanup failure never masks the typed 503).
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.files import router

TEST_BUCKET = "test-bucket"


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def put_object(self, *, Bucket, Key, Body, **_kw):
        assert Bucket == TEST_BUCKET
        self.objects[Key] = Body
        return {"ETag": '"etag-x"'}

    def delete_object(self, *, Bucket, Key):
        assert Bucket == TEST_BUCKET
        self.deleted.append(Key)
        self.objects.pop(Key, None)


class _FakeConfig:
    bucket = TEST_BUCKET

    def __init__(self, client) -> None:
        self._client = client

    def get_client(self):
        return self._client


class _LedgerCursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = []

    def execute(self, sql, params=None):
        if "SELECT 1 FROM user_disk_objects" in sql:
            self._result = [(1,)] if params[1] in self._conn.charged_hashes else []
        elif "bytes_used FROM user_disk_usage" in sql:
            self._result = [(self._conn.usage_total,)] if self._conn.usage_total else []
        elif "INSERT INTO user_disk_usage" in sql:
            self._conn.usage_total += params[1]
            self._result = []
        elif "INSERT INTO user_disk_objects" in sql:
            self._conn.charges.append(params)
            self._conn.charged_hashes.add(params[1])
            self._result = []
        else:
            self._result = []

    def fetchone(self):
        return self._result.pop(0) if self._result else None


class _LedgerConn:
    def __init__(self) -> None:
        self.charges: list[tuple] = []
        self.charged_hashes: set[str] = set()
        self.usage_total = 0

    def cursor(self):
        return _LedgerCursor(self)

    def commit(self):
        pass


class _LedgerPool:
    def __init__(self, conn: _LedgerConn) -> None:
        self._conn = conn
        self.returned = 0

    def getconn(self):
        return self._conn

    def putconn(self, _c):
        self.returned += 1


_HEADERS = {"X-User-ID": "alice", "X-Project-ID": "proj-a"}


@pytest.fixture
def rig(monkeypatch):
    import umrs.api.files as mod

    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    s3 = _FakeS3Client()
    monkeypatch.setattr(mod, "_fire_post_upload_ingest", lambda *_a, **_kw: None)
    conn = _LedgerConn()
    pool = _LedgerPool(conn)
    app = FastAPI()

    # Depends() binds the original callable at decoration time — override via
    # the app-level hook, not a module attribute swap.
    from umrs.shared.storage.s3_config import get_s3_config

    app.dependency_overrides[get_s3_config] = lambda: _FakeConfig(s3)
    app.include_router(router)
    app.state.umrs_db_pool = pool
    return TestClient(app), s3, conn, pool


def _post(client: TestClient, content: bytes, filename: str = "big.bin"):
    return client.post(
        "/umrs/files/upload",
        files={"file": (filename, content, "application/octet-stream")},
        headers=_HEADERS,
    )


class TestFilesUploadMetering:
    def test_upload_charges_md5_and_size(self, rig):
        client, s3, conn, _pool = rig
        content = b"payload-bytes" * 32
        resp = _post(client, content)
        assert resp.status_code == 200, resp.text
        assert conn.charges == [("alice", hashlib.md5(content).hexdigest(), len(content))]

    def test_identical_bytes_cost_nothing_twice(self, rig):
        client, s3, conn, _pool = rig
        content = b"same-thing" * 16
        assert _post(client, content).status_code == 200
        assert _post(client, content, filename="big-again.bin").status_code == 200
        assert len(conn.charges) == 1  # hash-dedup contract

    def test_quota_refusal_unstages_and_stays_typed(self, rig, monkeypatch):
        client, s3, _conn, pool = rig
        content = b"x" * 64
        monkeypatch.setenv("UMRS_USER_DISK_BYTES_MAX", "8")
        resp = _post(client, content)
        assert resp.status_code == 503
        assert resp.headers.get("x-platform-error-code") == "quota.disk_bytes_exceeded"
        # the just-staged key must be gone (no attacker-inflation of shared S3)
        assert not s3.objects
        assert s3.deleted
        assert pool.returned == 1

    def test_no_pool_keeps_legacy_silent_path(self, rig):
        # Harness adds the pool by default; remove it to mirror unit-mode.
        client, s3, conn, _pool = rig
        delattr(client.app.state, "umrs_db_pool")
        resp = _post(client, b"quiet")
        assert resp.status_code == 200
        assert conn.charges == []
