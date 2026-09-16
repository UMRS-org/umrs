"""I3 tests — resumable/batch intake upload routes.

Route-level (TestClient) with an in-memory manifest store and a fake
S3 client recording parts; the post-ingestion trigger is stubbed out.
"""

from __future__ import annotations

import io

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.intake_uploads import router
from umrs.data_catalog.intake_uploads import InMemoryIntakeUploadStore

TEST_BUCKET = "test-bucket"


class FakeS3Client:
    """Minimal S3 stand-in: single-bucket object dict."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.put_calls: list[tuple[str, bytes]] = []
        self.buckets_seen: set[str] = set()

    def _check_bucket(self, bucket: str) -> None:
        self.buckets_seen.add(bucket)
        assert bucket == TEST_BUCKET, f"unexpected bucket {bucket}"

    def head_object(self, *, Bucket: str, Key: str):
        self._check_bucket(Bucket)
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        body = self.objects[Key]
        return {
            "ContentLength": len(body),
            "ETag": self.etags.get(Key, '"etag-default"'),
        }

    def get_object(self, *, Bucket: str, Key: str):
        self._check_bucket(Bucket)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body, **_kwargs):
        self._check_bucket(Bucket)
        data = Body.read() if hasattr(Body, "read") else Body
        self.objects[Key] = data
        self.etags[Key] = '"final-etag"'
        self.put_calls.append((Key, data))
        return {"ETag": '"final-etag"'}

    def delete_object(self, *, Bucket: str, Key: str):
        self._check_bucket(Bucket)
        self.objects.pop(Key, None)
        self.etags.pop(Key, None)

    def generate_presigned_url(self, *, ClientMethod: str, Params: dict, ExpiresIn: int) -> str:
        return (
            f"http://fake-s3/{Params['Bucket']}/{Params['Key']}"
            f"?method={ClientMethod}&sig=x&expires={ExpiresIn}"
        )


class _FakeConfig:
    bucket = TEST_BUCKET

    def __init__(self, client: FakeS3Client) -> None:
        self._client = client

    def get_client(self) -> FakeS3Client:
        return self._client


def _headers() -> dict[str, str]:
    return {
        "X-User-ID": "alice",
        "X-Project-ID": "proj-a",
        "X-Verified-Project-ID": "proj-a",
    }


@pytest.fixture
def s3() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture
def store() -> InMemoryIntakeUploadStore:
    return InMemoryIntakeUploadStore()


@pytest.fixture
def client(store, s3, monkeypatch) -> TestClient:
    import umrs.api.intake_uploads as mod
    import umrs.data_catalog.triggers as triggers_mod

    # mvp is an explicit dev opt-in since B12 (strict is the default); these
    # tests exercise the mvp verified-header path without HMAC signing.
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setattr(mod, "_store", store)
    monkeypatch.setattr(
        mod,
        "get_s3_config",
        lambda: _FakeConfig(s3),
    )
    triggered: list[dict] = []
    monkeypatch.setattr(
        triggers_mod,
        "trigger_post_upload_ingestion",
        lambda *_a, **kw: triggered.append(kw) or "job-1",
    )
    app = FastAPI()
    app.include_router(router, dependencies=[])
    app.state.triggered = triggered
    return TestClient(app)


def _create(client: TestClient, parts: int = 2) -> str:
    resp = client.post(
        "/umrs/uploads/manifests",
        json={
            "target_key": "proj-a/inputs/big-dataset.tiff",
            "expected_parts": parts,
        },
        headers=_headers(),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["manifest"]["manifest_id"]


def _seed_parts(s3: FakeS3Client, target_key: str, manifest_id: str, numbers=(1, 2)):
    for n in numbers:
        s3.objects[f"{target_key}.__parts/{manifest_id}/part-{n:05d}"] = f"chunk-{n}\n".encode()


class TestManifestLifecycle:
    def test_create_and_fetch(self, client):
        manifest_id = _create(client, parts=3)
        resp = client.get(f"/umrs/uploads/manifests/{manifest_id}", headers=_headers())
        payload = resp.json()["manifest"]
        assert payload["expected_parts"] == 3
        assert payload["missing_parts"] == [1, 2, 3]
        assert payload["status"] == "uploading"

    def test_get_unknown_is_404(self, client):
        resp = client.get("/umrs/uploads/manifests/ul-nope", headers=_headers())
        assert resp.status_code == 404

    def test_presign_part_yields_signed_url(self, client):
        manifest_id = _create(client)
        resp = client.post(
            f"/umrs/uploads/manifests/{manifest_id}/parts/1/presign",
            headers=_headers(),
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["part"] == 1
        assert payload["url"]
        assert "part-00001" in payload["part_key"]

    def test_presign_part_beyond_expected_is_400(self, client):
        manifest_id = _create(client, parts=1)
        resp = client.post(
            f"/umrs/uploads/manifests/{manifest_id}/parts/2/presign",
            headers=_headers(),
        )
        assert resp.status_code == 400

    def test_record_part_dedups_and_sorts(self, client):
        manifest_id = _create(client)
        for body in (
            {"etag": "e2", "size_bytes": 10, "_n": 2},
            {"etag": "e1", "size_bytes": 8, "_n": 1},
            {"etag": "e1b", "size_bytes": 9, "_n": 1},
        ):
            n = body.pop("_n")
            resp = client.post(
                f"/umrs/uploads/manifests/{manifest_id}/parts/{n}",
                json=body,
                headers=_headers(),
            )
            assert resp.status_code == 200
        final = client.get(f"/umrs/uploads/manifests/{manifest_id}", headers=_headers())
        manifest = final.json()["manifest"]
        assert manifest["received_parts"] == [1, 2]
        assert manifest["parts_detail"]["1"] == {"etag": "e1b", "size_bytes": 9}

    def test_complete_missing_parts_409(self, client):
        manifest_id = _create(client)
        resp = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["missing_parts"] == [1, 2]


class TestCompleteFlow:
    def test_complete_assembles_and_ingests(self, client, s3):
        manifest_id = _create(client)
        target = "proj-a/inputs/big-dataset.tiff"
        _seed_parts(s3, target, manifest_id)

        resp = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "assembled"
        assert payload["final_etag"] == "final-etag"
        assert payload["job_id"] == "job-1"
        assert s3.objects[target] == b"chunk-1\nchunk-2\n"
        manifest = payload["manifest"]
        assert manifest["assembled_at"] is not None
        assert manifest["received_parts"] == [1, 2]
        assert manifest["size_bytes"] == len(b"chunk-1\nchunk-2\n")

    def test_complete_is_idempotent(self, client, s3):
        manifest_id = _create(client)
        target = "proj-a/inputs/big-dataset.tiff"
        _seed_parts(s3, target, manifest_id)

        first = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert first.status_code == 200
        puts_after_first = len(s3.put_calls)

        second = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert second.status_code == 200
        assert second.json()["idempotent"] is True
        assert len(s3.put_calls) == puts_after_first

    def test_reserved_manifest_blocks_new_parts(self, client, s3):
        manifest_id = _create(client, parts=2)
        target = "proj-a/inputs/big-dataset.tiff"
        _seed_parts(s3, target, manifest_id)
        client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        resp = client.post(
            f"/umrs/uploads/manifests/{manifest_id}/parts/1",
            json={"etag": "x", "size_bytes": 1},
            headers=_headers(),
        )
        assert resp.status_code == 409
        presign = client.post(
            f"/umrs/uploads/manifests/{manifest_id}/parts/1/presign",
            headers=_headers(),
        )
        assert presign.status_code == 409


# ── ADR-051 D-3 remaining-lane metering (2026-09-16) ─────────────────────────
# The complete edge is the catalog write point of the multipart presign lane:
# charge happens after assembly, before the status flip + ingest trigger, so a
# refusal deletes the assembled object but keeps parts + manifest re-completable.


class _LedgerCursor:
    def __init__(self):
        self._result = []

    def execute(self, sql, params=None):
        if "SELECT 1 FROM user_disk_objects" in sql:
            self._result = []
        elif "bytes_used FROM user_disk_usage" in sql:
            self._result = [(self._conn.usage_total,)] if self._conn.usage_total else []
        elif "INSERT INTO user_disk_usage" in sql:
            _uid, add1, add2 = params
            self._conn.usage_total += add1
            self._result = []
        elif "INSERT INTO user_disk_objects" in sql:
            self._conn.charges.append(params)
            self._result = []
        else:
            self._result = []

    def fetchone(self):
        return self._result.pop(0) if self._result else None


class _LedgerConn:
    def __init__(self):
        self.charges: list[tuple] = []
        self.usage_total = 0

    def cursor(self):
        cur = _LedgerCursor()
        cur._conn = self
        return cur

    def commit(self):
        pass


class _LedgerPool:
    def __init__(self, conn: _LedgerConn):
        self._conn = conn
        self.returned = 0

    def getconn(self):
        return self._conn

    def putconn(self, _conn):
        self.returned += 1


def _client_with_pool(store, s3, pool, monkeypatch) -> TestClient:
    import umrs.api.intake_uploads as mod
    import umrs.data_catalog.triggers as triggers_mod

    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setattr(mod, "_store", store)
    monkeypatch.setattr(mod, "get_s3_config", lambda: _FakeConfig(s3))
    monkeypatch.setattr(triggers_mod, "trigger_post_upload_ingestion", lambda *_a, **_kw: "job-1")
    app = FastAPI()
    app.include_router(router, dependencies=[])
    app.state.umrs_db_pool = pool
    app.state.triggered = []
    return TestClient(app)


class TestCompleteLaneMetering:
    def test_complete_charges_uploader_once_per_hash(self, store, s3, monkeypatch):
        conn = _LedgerConn()
        client = _client_with_pool(store, s3, _LedgerPool(conn), monkeypatch)
        manifest_id = _create(client)
        target = "proj-a/inputs/big-dataset.tiff"
        _seed_parts(s3, target, manifest_id)

        resp = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert resp.status_code == 200, resp.text
        assert conn.charges == [("alice", "final-etag", len(b"chunk-1\nchunk-2\n"))]

        # idempotent completes never double-charge (hash-dedup on top)
        resp2 = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert resp2.status_code == 200
        assert len(conn.charges) == 1

    def _refused_complete(self, store, s3, monkeypatch):
        monkeypatch.setenv("UMRS_USER_DISK_BYTES_MAX", "8")  # parts total 16 bytes
        conn = _LedgerConn()
        client = _client_with_pool(store, s3, _LedgerPool(conn), monkeypatch)
        manifest_id = _create(client)
        target = "proj-a/inputs/big-dataset.tiff"
        _seed_parts(s3, target, manifest_id)
        resp = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        return client, s3, manifest_id, target, resp

    def _assert_refusal_state(self, client, s3, manifest_id, target, resp):
        assert resp.status_code == 503
        assert resp.headers.get("x-platform-error-code") == "quota.disk_bytes_exceeded"
        # assembled object deleted, PARTS KEPT, manifest still completable
        assert target not in s3.objects
        assert any(k.startswith(f"{target}.__parts/") for k in s3.objects)
        state = client.get(f"/umrs/uploads/manifests/{manifest_id}", headers=_headers())
        assert state.json()["manifest"]["status"] == "uploading"
        assert client.app.state.triggered == []

    def test_refusal_keeps_parts_and_manifest_completable_state(self, store, s3, monkeypatch):
        args = self._refused_complete(store, s3, monkeypatch)
        self._assert_refusal_state(*args)

    def test_free_cap_lets_same_manifest_complete_without_reupload(self, store, s3, monkeypatch):
        client, s3_ctx, manifest_id, target, resp = self._refused_complete(store, s3, monkeypatch)
        self._assert_refusal_state(client, s3_ctx, manifest_id, target, resp)
        monkeypatch.setenv("UMRS_USER_DISK_BYTES_MAX", str(1024 * 1024))
        retry = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert retry.status_code == 200, retry.text
        assert retry.json()["status"] == "assembled"
        assert s3_ctx.objects[target] == b"chunk-1\nchunk-2\n"

    def test_pool_conn_returned_on_refusal(self, store, s3, monkeypatch):
        monkeypatch.setenv("UMRS_USER_DISK_BYTES_MAX", "1")
        conn = _LedgerConn()
        pool = _LedgerPool(conn)
        client = _client_with_pool(store, s3, pool, monkeypatch)
        manifest_id = _create(client)
        target = "proj-a/inputs/big-dataset.tiff"
        _seed_parts(s3, target, manifest_id)
        resp = client.post(f"/umrs/uploads/manifests/{manifest_id}/complete", headers=_headers())
        assert resp.status_code == 503
        assert pool.returned == 1
