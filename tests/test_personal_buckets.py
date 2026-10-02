"""ADR-079: self-service personal buckets and the owned-bucket files routes.

A verified user creates ``<user>.<project>.<alias>`` (+ ``-output``) through
UMRS, moves data with the files API and the HPC lane stages it — ownership
comes from the ``user_buckets`` row, and the server performs every S3 call.
"""

from __future__ import annotations

import io

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.server import step_staging
from umrs.server.step_executor import StepExecutionError
from umrs.shared.storage import user_buckets as ub


class _FakeStore:
    def __init__(self) -> None:
        self.rows: dict[str, ub.UserBucket] = {}

    def add(self, entry: ub.UserBucket) -> bool:
        if entry.bucket in self.rows:
            return False
        self.rows[entry.bucket] = entry
        return True

    def get(self, bucket: str):
        return self.rows.get(bucket)

    def list_for(self, owner: str):
        return [r for r in self.rows.values() if r.owner_id == owner]

    def owner_of(self, bucket: str):
        row = self.rows.get(ub.input_bucket_of(bucket))
        return row.owner_id if row else None

    def set_access(self, bucket: str, access: list) -> bool:
        row = self.rows.get(ub.input_bucket_of(bucket))
        if row is None:
            return False
        self.rows[row.bucket] = ub.UserBucket(
            row.bucket, row.owner_id, row.project, row.alias, tuple(access)
        )
        return True

    def can_access(self, bucket: str, user_id: str) -> bool:
        row = self.rows.get(ub.input_bucket_of(bucket))
        if row is None:
            return False
        return row.owner_id == user_id

    def list_shared_with(self, user_id: str):
        return [
            r
            for r in self.rows.values()
            if r.owner_id != user_id and "system" in r.access
        ]


class _FakeS3:
    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], bytes] = {}

    def head_bucket(self, Bucket):
        if Bucket not in self.buckets:
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    def create_bucket(self, Bucket):
        self.buckets.add(Bucket)

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):  # noqa: ARG002 — boto3 shape
        data = b""
        while chunk := fileobj.read(3):  # small reads: the body is streamed
            data += chunk
        self.objects[(bucket, key)] = data

    def list_objects_v2(self, **kw):
        items = [
            {"Key": k, "Size": len(v)}
            for (b, k), v in sorted(self.objects.items())
            if b == kw["Bucket"] and k.startswith(kw.get("Prefix", ""))
        ]
        return {"Contents": items, "IsTruncated": False}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


class _FakeS3Config:
    def __init__(self, s3: _FakeS3) -> None:
        self.bucket = "r4-data"
        self._s3 = s3

    def get_client(self):
        return self._s3


@pytest.fixture
def store(monkeypatch):
    fake = _FakeStore()
    ub.set_user_bucket_store(fake)  # type: ignore[arg-type]
    monkeypatch.setattr(step_staging, "_bucket_registry", None)
    yield fake
    ub.set_user_bucket_store(None)


@pytest.fixture
def s3(monkeypatch):
    fake = _FakeS3()
    import umrs.api.buckets as buckets_api
    import umrs.api.files as files_api

    monkeypatch.setattr(buckets_api, "get_s3_config", lambda: _FakeS3Config(fake))
    monkeypatch.setattr(files_api, "get_s3_config", lambda: _FakeS3Config(fake))
    return fake


# -- names -------------------------------------------------------------------


def test_bucket_name_follows_the_lane_shape():
    assert ub.bucket_name("Alice_B", "Opossum Study", "videos") == "alice-b.opossum-study.videos"
    assert ub.bucket_name("00123456", "p", "a") == "00123456.p.a"  # numeric OIDC subs fit


@pytest.mark.parametrize(
    "owner,project,alias",
    [("a/b", "p", "a"), ("a", "p/x", "a"), ("a", "p", "-x"), ("a", "", "v")],
)
def test_bucket_name_refuses_what_s3_or_the_lane_cannot_hold(owner, project, alias):
    with pytest.raises(ub.UserBucketError):
        ub.bucket_name(owner, project, alias)


def test_an_email_shaped_identity_maps_to_one_segment():
    # OIDC access tokens can carry the login (an e-mail) as `sub`.
    assert ub.bucket_name("Masaru.Nagaso@riken.jp", "opossum", "videos") == (
        "masaru-nagaso-riken-jp.opossum.videos"
    )


def test_bucket_name_keeps_room_for_the_output_suffix():
    with pytest.raises(ub.UserBucketError):
        ub.bucket_name("a" * 20, "b" * 20, "c" * 20)


# -- ownership: the lane and the files API share one rule ---------------------


def test_a_personal_bucket_and_its_output_belong_to_their_owner(store):
    store.add(ub.UserBucket("alice.opossum.videos", "alice", "opossum", "videos"))
    assert step_staging.bucket_owned_by("alice.opossum.videos", "alice")
    assert step_staging.bucket_owned_by("alice.opossum.videos-output", "alice")
    assert not step_staging.bucket_owned_by("alice.opossum.videos", "bob")
    assert not step_staging.bucket_owned_by("alice.opossum.other", "alice")


def test_the_lane_stages_a_personal_bucket_for_its_owner_only(store, monkeypatch):
    class _Registry:
        def get(self, _name):
            return None

        def find_by_output_path(self, _name):
            return None, None

    monkeypatch.setattr(step_staging, "_bucket_registry", _Registry())
    store.add(ub.UserBucket("alice.opossum.videos", "alice", "opossum", "videos"))
    step_staging.check_bucket_scope("alice.opossum.videos", "alice", [])
    step_staging.check_bucket_scope("alice.opossum.videos-output", "alice", [])
    with pytest.raises(StepExecutionError):
        step_staging.check_bucket_scope("alice.opossum.unknown", "alice", [])


def test_an_email_owner_is_staged_through_the_ownership_row(store, monkeypatch):
    class _Registry:
        def get(self, _name):
            return None

        def find_by_output_path(self, _name):
            return None, None

    monkeypatch.setattr(step_staging, "_bucket_registry", _Registry())
    owner = "Masaru.Nagaso@riken.jp"
    name = ub.bucket_name(owner, "opossum", "videos")
    store.add(ub.UserBucket(name, owner, "opossum", "videos"))
    step_staging.check_bucket_scope(name, owner, [])
    step_staging.check_bucket_scope(name + "-output", owner, [])
    assert step_staging.bucket_owned_by(name, owner)
    assert not step_staging.bucket_owned_by(name, "someone@riken.jp")
    with pytest.raises(StepExecutionError):
        step_staging.check_bucket_scope(name, "someone@riken.jp", [])


def test_a_lookup_failure_never_grants_ownership(monkeypatch):
    class _Broken:
        def owner_of(self, _b):
            raise RuntimeError("db down")

    ub.set_user_bucket_store(_Broken())  # type: ignore[arg-type]
    try:
        assert ub.owner_of("alice.p.a") is None
    finally:
        ub.set_user_bucket_store(None)


# -- buckets API ---------------------------------------------------------------


def _client(user: str) -> TestClient:
    from umrs.api import buckets as buckets_api
    from umrs.api._verified_headers import CallerIdentity, get_verified_caller

    app = FastAPI()
    app.include_router(buckets_api.router)
    app.dependency_overrides[get_verified_caller] = lambda: CallerIdentity(
        user_id=user, project_id="default", is_admin=False
    )
    return TestClient(app)


def test_create_makes_both_buckets_and_records_the_owner(store, s3):
    resp = _client("alice").post("/umrs/buckets", json={"project": "opossum", "alias": "videos"})
    assert resp.status_code == 201, resp.text
    assert resp.json()["bucket"] == "alice.opossum.videos"
    assert s3.buckets == {"alice.opossum.videos", "alice.opossum.videos-output"}
    assert store.owner_of("alice.opossum.videos-output") == "alice"
    # idempotent for the same owner
    assert (
        _client("alice")
        .post("/umrs/buckets", json={"project": "opossum", "alias": "videos"})
        .status_code
        == 201
    )
    listed = _client("alice").get("/umrs/buckets").json()["buckets"]
    assert [b["bucket"] for b in listed] == ["alice.opossum.videos"]
    assert _client("bob").get("/umrs/buckets").json()["buckets"] == []


def test_create_refuses_a_name_another_owner_holds(store, s3):
    store.add(ub.UserBucket("alice.opossum.videos", "mallory", "opossum", "videos"))
    resp = _client("alice").post("/umrs/buckets", json={"project": "opossum", "alias": "videos"})
    assert resp.status_code == 409


def test_two_identities_mapping_to_one_name_cannot_share_it(store, s3):
    assert _client("a.b").post("/umrs/buckets", json={"project": "p", "alias": "v"}).status_code == 201
    resp = _client("a-b").post("/umrs/buckets", json={"project": "p", "alias": "v"})
    assert resp.status_code == 409


def test_create_refuses_an_invalid_name(store, s3):
    resp = _client("alice").post("/umrs/buckets", json={"project": "op/x", "alias": "v"})
    assert resp.status_code == 400


# -- files API on owned buckets -------------------------------------------------


def _files_client(user: str) -> TestClient:
    import umrs.api.files as files_api

    app = FastAPI()
    app.include_router(files_api.router)
    app.dependency_overrides[files_api.get_project_id] = lambda: "default"
    app.dependency_overrides[files_api.get_project_s3_config] = lambda: _FakeS3Config(_FakeS3())
    client = TestClient(app)
    client.headers.update({"X-User-ID": user})
    return client


def test_owner_uploads_lists_and_downloads_with_raw_keys(store, s3, monkeypatch):
    import umrs.api.files as files_api

    monkeypatch.setattr(files_api, "_charge_upload_quota", lambda *_a, **_k: None)
    store.add(ub.UserBucket("alice.opossum.videos", "alice", "opossum", "videos"))
    c = _files_client("alice")
    up = c.post(
        "/umrs/files/upload",
        data={"bucket": "alice.opossum.videos", "key": "opossum_215_A/clip.mp4"},
        files={"file": ("clip.mp4", b"0123456789", "video/mp4")},
    )
    assert up.status_code == 200, up.text
    assert up.json()["key"] == "opossum_215_A/clip.mp4" and up.json()["size"] == 10
    assert s3.objects[("alice.opossum.videos", "opossum_215_A/clip.mp4")] == b"0123456789"
    listed = c.get(
        "/umrs/files/list", params={"bucket": "alice.opossum.videos", "prefix": "opossum_"}
    )
    assert [o["key"] for o in listed.json()["objects"]] == ["opossum_215_A/clip.mp4"]
    s3.objects[("alice.opossum.videos-output", "runs/wf-1/agg/results.tsv")] = b"a\tb\n"
    got = c.get(
        "/umrs/files/download/runs/wf-1/agg/results.tsv",
        params={"bucket": "alice.opossum.videos-output"},
    )
    assert got.status_code == 200 and got.content == b"a\tb\n"


def test_quota_is_charged_with_the_streamed_hash_and_size(store, s3, monkeypatch):
    import hashlib

    import umrs.api.files as files_api

    seen = {}
    monkeypatch.setattr(
        files_api,
        "_charge_upload_quota",
        lambda _r, _s3, _b, _k, content, _u: seen.update(c=content),
    )
    store.add(ub.UserBucket("alice.p.v", "alice", "p", "v"))
    _files_client("alice").post(
        "/umrs/files/upload",
        data={"bucket": "alice.p.v"},
        files={"file": ("w.pt", b"weights", "application/octet-stream")},
    )
    assert seen["c"] == (hashlib.md5(b"weights").hexdigest(), 7)


def test_another_users_bucket_is_refused_everywhere(store, s3):
    store.add(ub.UserBucket("alice.opossum.videos", "alice", "opossum", "videos"))
    c = _files_client("bob")
    assert (
        c.post(
            "/umrs/files/upload",
            data={"bucket": "alice.opossum.videos"},
            files={"file": ("x", b"x", "text/plain")},
        ).status_code
        == 403
    )
    assert c.get("/umrs/files/list", params={"bucket": "alice.opossum.videos"}).status_code == 403
    assert (
        c.get(
            "/umrs/files/download/x", params={"bucket": "alice.opossum.videos-output"}
        ).status_code
        == 403
    )


def test_traversal_in_an_owned_bucket_is_refused(store, s3):
    store.add(ub.UserBucket("alice.p.v", "alice", "p", "v"))
    resp = _files_client("alice").post(
        "/umrs/files/upload",
        data={"bucket": "alice.p.v", "key": "../etc/x"},
        files={"file": ("x", b"x", "text/plain")},
    )
    assert resp.status_code == 400


def test_store_round_trip_against_postgres():
    # Env-gated like test_job_venue_scoping: runs where CI provides Postgres.
    pytest.importorskip("psycopg2")
    from pathlib import Path

    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")
    sql = (
        Path(ub.__file__).resolve().parents[2] / "migrations" / "0115_user_buckets.sql"
    ).read_text()
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql)
            cur.execute("DELETE FROM user_buckets WHERE owner_id = 'pytest-owner'")
    finally:
        pool.putconn(conn)
    try:
        store = ub.UserBucketStore(lambda: pool)
        entry = ub.UserBucket("pytest-owner.p.v", "pytest-owner", "p", "v")
        assert store.add(entry) is True
        assert store.add(entry) is False
        assert store.get("pytest-owner.p.v") == entry
        assert store.list_for("pytest-owner") == [entry]
        assert store.owner_of("pytest-owner.p.v-output") == "pytest-owner"
        assert store.owner_of("nobody.p.v") is None
    finally:
        conn = pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("DELETE FROM user_buckets WHERE owner_id = 'pytest-owner'")
        finally:
            pool.putconn(conn)
        pool.closeall()
