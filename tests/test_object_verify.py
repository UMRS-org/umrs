"""ADR-115 S2c (slice MB-S2c): ``OBJECT_VERIFY``, server-only ``runs/`` and
create-only ``reviews/``.

Before this slice nothing re-read an object to check its bytes, any client
could write (or presign a PUT, or delete) under ``runs/`` of an ``-output``
bucket where step collection keeps a run's outputs, and two review saves to
one key both "won" (the second silently overwrote the first). Now:

* ``OBJECT_VERIFY`` (a durable job type, and ``provenance.verify_object``
  for in-process callers) re-reads the object inside UMRS and records
  ``verified`` or ``mismatch`` (with the reason) on its provenance row; a
  legacy object (no row) is attested on demand;
* a client write under ``runs/`` of an ``-output`` bucket is 403
  ``files.prefix_write_denied`` before any store call;
* a save under ``reviews/`` (or any save with ``create_only=true``) claims
  the key with a provenance reservation row before its PUT: the first save
  is 201, a later or concurrent one 409 ``files.create_only_exists``; with no
  database it is 503 and nothing is written.

The ``integration`` tests run the reservation race and the verdict upsert on
real Postgres (``UMRS_TEST_DB_URL``).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import audit_events, files
from umrs.data_catalog import provenance
from umrs.jobs import adapters
from umrs.jobs import data_ingestion as ingest_jobs
from umrs.jobs.models import DEFAULT_BACKOFFS, JobType
from umrs.jobs.object_verify import (
    ObjectVerifyBackend,
    ObjectVerifyStatusChecker,
    parse_ref,
    validate_spec,
)
from umrs.server import job_tracking_handler

OUT = "alice.p.a-output"
BODY = b"frame-bytes-" * 5000
SHA = hashlib.sha256(BODY).hexdigest()
ERROR_CODE = "x-platform-error-code"
_MIGRATION = (
    Path(provenance.__file__).resolve().parents[1]
    / "migrations"
    / "0142_object_provenance.sql"
)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── doubles ─────────────────────────────────────────────────────────────────


class _Db:
    """The ``object_provenance`` statements this slice runs, on a dict keyed
    by the table's primary key ``(bucket, key, etag)``. Thread-safe, so two
    concurrent saves race on it the way they race on the real table."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.lock = threading.Lock()

    def run(self, sql: str, params: tuple[Any, ...]) -> tuple[int, Any]:
        if "object_provenance" not in sql:
            return 0, None  # the disk ledger's statements: not this slice's
        with self.lock:
            for marker, handler in self._handlers():
                if marker in sql:
                    return handler(params)
        raise AssertionError(f"unexpected SQL: {sql}")

    def _handlers(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("sha256_state, source_kind, uploader, project_id) VALUES", self._reserve),
            ("DO UPDATE SET", self._upsert),
            ("DO NOTHING", self._insert),
            ("DELETE FROM object_provenance", self._delete),
            ("LIMIT 1", self._lookup),
        )

    def _reserve(self, params: tuple[Any, ...]) -> tuple[int, Any]:
        bucket, key, etag, source_kind, uploader, project_id = params
        if (bucket, key, etag) in self.rows:
            return 0, None
        self.rows[(bucket, key, etag)] = {
            **dict.fromkeys(provenance._COLUMNS),
            "bucket": bucket,
            "key": key,
            "etag": etag,
            "size": 0,
            "sha256_state": "pending",
            "source_kind": source_kind,
            "source_ref": {},
            "uploader": uploader,
            "project_id": project_id,
        }
        return 1, None

    def _insert(self, params: tuple[Any, ...]) -> tuple[int, Any]:
        row = dict(zip(provenance._COLUMNS, params, strict=False))
        pk = (row["bucket"], row["key"], row["etag"])
        if pk in self.rows:
            return 0, None
        self.rows[pk] = row
        return 1, None

    def _upsert(self, params: tuple[Any, ...]) -> tuple[int, Any]:
        names = ("bucket", "key", "etag", "size", "sha256", "sha256_state",
                 "content_type", "source_kind", "source_ref")
        new = dict(zip(names, params, strict=False))
        pk = (new["bucket"], new["key"], new["etag"])
        old = self.rows.get(pk)
        if old is None:
            self.rows[pk] = {**dict.fromkeys(provenance._COLUMNS), **new}
            return 1, None
        old["sha256_state"] = new["sha256_state"]
        old["sha256"] = old["sha256"] or new["sha256"]
        return 1, None

    def _delete(self, params: tuple[Any, ...]) -> tuple[int, Any]:
        return int(self.rows.pop(tuple(params), None) is not None), None

    def _lookup(self, params: tuple[Any, ...]) -> tuple[int, Any]:
        bucket, key, etag, _same = params
        found = [
            row for (b, k, e), row in self.rows.items()
            if (b, k) == (bucket, key)
            and e != provenance.RESERVED_ETAG
            and etag in ("", e)
        ]
        if not found:
            return 0, None
        return 1, tuple(found[-1][column] for column in provenance._COLUMNS)


class _Cursor:
    def __init__(self, db: _Db) -> None:
        self._db = db
        self.rowcount = 0
        self._found: Any = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.rowcount, self._found = self._db.run(sql, params)

    def fetchone(self) -> Any:
        return self._found


class _Conn:
    def __init__(self, db: _Db) -> None:
        self._db = db
        self.commits = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self._db)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        return None


class _Pool:
    def __init__(self, db: _Db | None = None) -> None:
        self.db = db or _Db()

    def getconn(self) -> _Conn:
        return _Conn(self.db)

    def putconn(self, _conn: _Conn) -> None:
        return None


class _S3:
    """An object store whose ETag is the body MD5, or a pinned value (a
    store-level tamper keeps the ETag and changes the bytes)."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.etags: dict[tuple[str, str], str] = {}
        self.calls: list[str] = []
        self.on_head: Any = None

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):  # noqa: N803, ARG002 — boto3
        self.calls.append("upload")
        chunks = []
        while chunk := fileobj.read(4096):
            chunks.append(chunk)
        self.objects[(bucket, key)] = b"".join(chunks)
        self.etags.pop((bucket, key), None)

    def head_object(self, *, Bucket, Key):  # noqa: N803 — boto3
        self.calls.append("head")
        if self.on_head is not None:
            self.on_head(Bucket, Key)
        if (Bucket, Key) not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        body = self.objects[(Bucket, Key)]
        etag = self.etags.get((Bucket, Key), _md5(body))
        return {"ETag": f'"{etag}"', "ContentLength": len(body), "ContentType": "video/mp4"}

    def get_object(self, *, Bucket, Key):  # noqa: N803 — boto3
        self.calls.append("get")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, *, Bucket, Key):  # noqa: N803 — boto3
        self.calls.append("delete")
        self.objects.pop((Bucket, Key), None)

    def generate_presigned_url(self, **_kw: Any) -> str:
        self.calls.append("presign")
        return "http://store.invalid/signed"


class _Config:
    bucket = "test-bucket"
    default_ttl = 900
    max_ttl = 3600
    presign_endpoint = None

    def __init__(self, client: _S3) -> None:
        self._client = client

    def get_client(self) -> _S3:
        return self._client


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(audit_events, "_audit_store", None)
    monkeypatch.setattr(audit_events, "_audit_buffer", [])
    provenance.set_provenance_pool(None)
    yield
    provenance.set_provenance_pool(None)


def _app(monkeypatch: pytest.MonkeyPatch, s3: _S3, pool: Any, project: str = "pa") -> FastAPI:
    monkeypatch.setattr(files, "_fire_post_upload_ingest", lambda *_a: None)
    monkeypatch.setattr(files, "_require_owned_bucket", lambda *_a: None)
    monkeypatch.setattr(files, "bucket_store", lambda *_a: _Config(s3))
    app = FastAPI()
    app.include_router(files.router)
    app.state.umrs_db_pool = pool
    app.dependency_overrides[files.get_project_id] = lambda: project
    app.dependency_overrides[files.get_project_s3_config] = lambda: _Config(s3)
    return app


def _save(app: FastAPI, key: str, body: bytes = BODY, **data: str) -> Any:
    return TestClient(app).post(
        "/umrs/files/upload",
        files={"file": ("r.jsonl", body, "application/json")},
        data={"bucket": OUT, "key": key, **data},
        headers={"X-User-ID": "alice"},
    )


# ── the prefix rule ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("bucket", "key", "rule"),
    [
        (OUT, "runs/wf-1/s/a.mp4", provenance.SERVER_ONLY),
        (OUT, "/runs/wf-1/s/a.mp4", provenance.SERVER_ONLY),
        (OUT, "reviews/wf-1/alice/t.jsonl", provenance.CREATE_ONLY),
        (OUT, "intake/a.mp4", None),
        (OUT, "runsheet.csv", None),
        ("alice.p.a", "runs/wf-1/a.mp4", None),
    ],
)
def test_the_prefix_rule_holds_only_in_output_buckets(bucket, key, rule) -> None:
    assert provenance.output_prefix_rule(bucket, key) == rule


# ── server-only runs/ ───────────────────────────────────────────────────────


def test_a_client_upload_under_runs_is_403_before_any_store_call(monkeypatch) -> None:
    s3 = _S3()
    r = _save(_app(monkeypatch, s3, _Pool()), "runs/wf-1/s/a.mp4")
    assert r.status_code == 403, r.text
    assert r.headers[ERROR_CODE] == "files.prefix_write_denied"
    assert s3.calls == [] and s3.objects == {}


def test_runs_in_an_input_bucket_is_an_ordinary_key(monkeypatch) -> None:
    s3 = _S3()
    r = TestClient(_app(monkeypatch, s3, _Pool())).post(
        "/umrs/files/upload",
        files={"file": ("a.mp4", BODY, "video/mp4")},
        data={"bucket": "alice.p.a", "key": "runs/a.mp4"},
        headers={"X-User-ID": "alice"},
    )
    assert r.status_code == 200, r.text


def test_a_presigned_put_under_runs_or_reviews_is_refused(monkeypatch) -> None:
    s3 = _S3()
    client = TestClient(_app(monkeypatch, s3, _Pool(), project="runs"))
    for project in ("runs", "reviews"):
        client.app.dependency_overrides[files.get_project_id] = lambda p=project: p
        r = client.post(
            "/umrs/files/presign",
            json={"operation": "upload", "key": "x.mp4", "bucket": "test-bucket-output"},
        )
        assert r.status_code == 403, r.text
        assert r.headers[ERROR_CODE] == "files.prefix_write_denied"
    assert "presign" not in s3.calls


def test_a_presigned_get_under_runs_still_works(monkeypatch) -> None:
    client = TestClient(_app(monkeypatch, _S3(), _Pool(), project="runs"))
    r = client.post(
        "/umrs/files/presign",
        json={"operation": "download", "key": "x.mp4", "bucket": "test-bucket-output"},
    )
    assert r.status_code == 200, r.text


def test_a_client_delete_under_runs_or_reviews_is_refused(monkeypatch) -> None:
    s3 = _S3()
    s3.objects[("test-bucket-output", "reviews/x.jsonl")] = b"{}"
    client = TestClient(_app(monkeypatch, s3, _Pool(), project="reviews"))
    r = client.delete("/umrs/files/x.jsonl", params={"bucket": "test-bucket-output"})
    assert r.status_code == 403, r.text
    assert ("test-bucket-output", "reviews/x.jsonl") in s3.objects


# ── create-only reviews/ ────────────────────────────────────────────────────


def test_the_first_review_save_is_201_with_a_review_record_row(monkeypatch) -> None:
    pool = _Pool()
    r = _save(_app(monkeypatch, _S3(), pool), "reviews/wf-1/alice/t1.jsonl")
    assert r.status_code == 201, r.text
    assert r.json()["sha256"] == SHA
    row = provenance.lookup(pool.getconn(), OUT, "reviews/wf-1/alice/t1.jsonl")
    assert (row.source_kind, row.sha256_state, row.sha256) == ("review_record", "verified", SHA)
    assert (OUT, "reviews/wf-1/alice/t1.jsonl", provenance.RESERVED_ETAG) in pool.db.rows


def test_a_second_save_to_the_same_review_key_is_409_and_keeps_the_first(monkeypatch) -> None:
    s3 = _S3()
    app = _app(monkeypatch, s3, _Pool())
    assert _save(app, "reviews/k.jsonl").status_code == 201
    r = _save(app, "reviews/k.jsonl", body=b"overwrite")
    assert r.status_code == 409, r.text
    assert r.headers[ERROR_CODE] == "files.create_only_exists"
    assert s3.objects[(OUT, "reviews/k.jsonl")] == BODY


def test_a_review_key_written_before_create_only_existed_is_409(monkeypatch) -> None:
    s3 = _S3()
    s3.objects[(OUT, "reviews/old.jsonl")] = b"legacy"
    r = _save(_app(monkeypatch, s3, _Pool()), "reviews/old.jsonl")
    assert r.status_code == 409
    assert s3.objects[(OUT, "reviews/old.jsonl")] == b"legacy"


def test_two_concurrent_saves_to_one_review_key_one_201_one_409(monkeypatch) -> None:
    s3 = _S3()
    both_checked = threading.Barrier(2, timeout=10)
    first_heads = iter(range(2))

    def _wait_for_the_other(_bucket: str, _key: str) -> None:
        if next(first_heads, None) is not None:
            both_checked.wait()  # both saves see "absent" before either claims

    s3.on_head = _wait_for_the_other
    app = _app(monkeypatch, s3, _Pool())
    codes: list[int] = []

    def _one(body: bytes) -> None:
        codes.append(_save(app, "reviews/race.jsonl", body=body).status_code)

    threads = [threading.Thread(target=_one, args=(body,)) for body in (b"first", b"second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(codes) == [201, 409]


def test_create_only_without_a_database_is_503_and_writes_nothing(monkeypatch) -> None:
    s3 = _S3()
    r = _save(_app(monkeypatch, s3, None), "reviews/k.jsonl")
    assert r.status_code == 503
    assert r.headers[ERROR_CODE] == "files.create_only_unavailable"
    assert s3.objects == {}


class _FailingS3(_S3):
    def upload_fileobj(self, *_a: Any, **_kw: Any) -> None:
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "500", "Message": "down"}}, "PutObject")


def test_a_failed_put_gives_the_key_back(monkeypatch) -> None:
    pool = _Pool()
    r = _save(_app(monkeypatch, _FailingS3(), pool), "reviews/k.jsonl")
    assert r.status_code == 500
    assert pool.db.rows == {}
    assert _save(_app(monkeypatch, _S3(), pool), "reviews/k.jsonl").status_code == 201


def test_create_only_true_makes_any_key_write_once(monkeypatch) -> None:
    app = _app(monkeypatch, _S3(), _Pool())
    first = _save(app, "intake/a.mp4", create_only="true")
    assert first.status_code == 201, first.text
    assert _save(app, "intake/a.mp4", create_only="true").status_code == 409
    assert _save(app, "intake/b.mp4").status_code == 200


def test_reads_never_see_a_reservation_row() -> None:
    db = _Db()
    conn = _Conn(db)
    assert provenance.reserve_create_only(conn, OUT, "reviews/k.jsonl")
    assert not provenance.reserve_create_only(conn, OUT, "reviews/k.jsonl")
    assert provenance.lookup(conn, OUT, "reviews/k.jsonl").legacy
    rows = {(OUT, "reviews/k.jsonl"): [provenance._row_from(
        tuple(db.rows[(OUT, "reviews/k.jsonl", provenance.RESERVED_ETAG)][c]
              for c in provenance._COLUMNS)
    )]}
    assert provenance._pick(rows, (OUT, "reviews/k.jsonl", "")).legacy


# ── OBJECT_VERIFY ───────────────────────────────────────────────────────────


def _stored(s3: _S3, key: str = "intake/a.mp4", body: bytes = BODY) -> None:
    s3.objects[(OUT, key)] = body


def _record(db: _Db, key: str, etag: str, sha: str = SHA, kind: str = "local_upload") -> None:
    provenance.record(_Conn(db), provenance.ObjectProvenance(
        bucket=OUT, key=key, etag=etag, size=len(BODY), sha256=sha,
        sha256_state="verified", content_type="video/mp4", source_kind=kind,
    ))


def test_an_untouched_object_verifies() -> None:
    s3, db = _S3(), _Db()
    _stored(s3)
    _record(db, "intake/a.mp4", _md5(BODY))
    outcome = provenance.verify_object(s3, _Conn(db), OUT, "intake/a.mp4")
    assert (outcome.sha256_state, outcome.reason, outcome.sha256) == ("verified", None, SHA)
    assert outcome.source_kind == "local_upload"


def test_a_tampered_object_under_its_old_etag_is_a_mismatch() -> None:
    """Bytes changed in the store with the ETag kept (a store-level write):
    the version's ETag is no longer the MD5 of what it serves."""
    s3, db = _S3(), _Db()
    _stored(s3, body=b"tampered" + BODY[8:])
    s3.etags[(OUT, "intake/a.mp4")] = _md5(BODY)
    _record(db, "intake/a.mp4", _md5(BODY))
    outcome = provenance.verify_object(s3, _Conn(db), OUT, "intake/a.mp4")
    assert outcome.sha256_state == "mismatch"
    assert outcome.reason == "etag_not_body_md5"
    assert db.rows[(OUT, "intake/a.mp4", _md5(BODY))]["sha256_state"] == "mismatch"


def test_a_tampered_multipart_object_is_a_mismatch_by_its_recorded_hash() -> None:
    s3, db = _S3(), _Db()
    tampered = b"tampered" + BODY[8:]
    _stored(s3, body=tampered)
    s3.etags[(OUT, "intake/a.mp4")] = "0123abcd-2"
    _record(db, "intake/a.mp4", "0123abcd-2")
    outcome = provenance.verify_object(s3, _Conn(db), OUT, "intake/a.mp4")
    assert (outcome.sha256_state, outcome.reason) == ("mismatch", "sha256_differs")
    assert outcome.sha256 == SHA  # what it should be
    assert outcome.observed_sha256 == hashlib.sha256(tampered).hexdigest()


def test_a_legacy_object_is_verified_on_demand_and_gets_a_row() -> None:
    s3, db = _S3(), _Db()
    _stored(s3, key="old/clip.mp4")
    conn = _Conn(db)
    assert provenance.lookup(conn, OUT, "old/clip.mp4").legacy
    outcome = provenance.verify_object(s3, conn, OUT, "old/clip.mp4")
    assert (outcome.sha256_state, outcome.source_kind) == ("verified", "legacy")
    row = provenance.lookup(conn, OUT, "old/clip.mp4")
    assert (row.sha256, row.sha256_state, row.etag) == (SHA, "verified", _md5(BODY))


def test_a_declared_hash_that_differs_is_a_mismatch() -> None:
    s3 = _S3()
    _stored(s3)
    outcome = provenance.verify_object(
        s3, _Conn(_Db()), OUT, "intake/a.mp4", declared_sha256="0" * 64
    )
    assert (outcome.sha256_state, outcome.reason) == ("mismatch", "declared_sha256_differs")


def test_an_object_that_changes_while_read_has_no_verdict() -> None:
    s3, db = _S3(), _Db()
    _stored(s3)
    heads = iter(range(10))

    def _rewrite_after_first_head(bucket: str, key: str) -> None:
        if next(heads) == 1:
            s3.objects[(bucket, key)] = b"new version"

    s3.on_head = _rewrite_after_first_head
    with pytest.raises(provenance.ObjectUnreadable):
        provenance.verify_object(s3, _Conn(db), OUT, "intake/a.mp4")
    assert db.rows == {}


def test_a_missing_object_or_database_has_no_verdict() -> None:
    with pytest.raises(provenance.ObjectUnreadable):
        provenance.verify_object(_S3(), _Conn(_Db()), OUT, "gone.mp4")
    with pytest.raises(provenance.ObjectUnreadable):
        provenance.verify_object_pooled(None, _S3(), OUT, "gone.mp4")


# ── the job lane ────────────────────────────────────────────────────────────


def _job(**spec: Any) -> Any:
    return SimpleNamespace(job_id="job-1", project_id="pa", user_id="alice", spec=spec)


def test_object_verify_is_a_job_type_with_a_lane_and_a_backoff() -> None:
    assert JobType("object_verify") is JobType.OBJECT_VERIFY
    assert JobType.OBJECT_VERIFY in DEFAULT_BACKOFFS
    assert isinstance(adapters._LAZY_BACKENDS[JobType.OBJECT_VERIFY](), ObjectVerifyBackend)


def test_the_job_completes_with_the_verdict_in_its_metrics(monkeypatch) -> None:
    s3, pool = _S3(), _Pool()
    _stored(s3, body=b"tampered" + BODY[8:])
    s3.etags[(OUT, "intake/a.mp4")] = _md5(BODY)
    _record(pool.db, "intake/a.mp4", _md5(BODY))
    provenance.set_provenance_pool(pool)
    monkeypatch.setattr(ingest_jobs, "_ingest_s3_client", lambda *_a: s3)
    job = _job(bucket=OUT, key="intake/a.mp4")

    async def scenario() -> Any:
        ref = await ObjectVerifyBackend().submit_job(job, _attempt("att-m"))
        assert parse_ref(ref) == {"attempt": "att-m"}
        return await _settle(ObjectVerifyStatusChecker(), ref, job)

    result = asyncio.run(scenario())
    assert result.status == "succeeded"
    assert result.metrics["sha256_state"] == "mismatch"
    assert result.metrics["reason"] == "etag_not_body_md5"


def test_the_job_fails_without_a_provenance_database(monkeypatch) -> None:
    s3 = _S3()
    _stored(s3)
    monkeypatch.setattr(ingest_jobs, "_ingest_s3_client", lambda *_a: s3)
    with pytest.raises(provenance.ObjectUnreadable):
        asyncio.run(ObjectVerifyBackend().submit_job(_job(bucket=OUT, key="intake/a.mp4"), None))


@pytest.mark.parametrize(
    "spec",
    [{}, {"bucket": OUT}, {"bucket": OUT, "key": "k", "sha256": "ABC"}],
)
def test_a_malformed_spec_is_refused(spec) -> None:
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_a_caller_may_verify_only_a_bucket_they_may_use(monkeypatch) -> None:
    from umrs.server import step_staging

    monkeypatch.setattr(step_staging, "bucket_owned_by", lambda bucket, _u: bucket == OUT)
    refuse = job_tracking_handler.spec_resource_refusal
    assert refuse(JobType.OBJECT_VERIFY, {"bucket": OUT, "key": "k"}, "alice", False) is None
    refusal = refuse(JobType.OBJECT_VERIFY, {"bucket": "bob.p-output", "key": "k"}, "alice", False)
    assert refusal and "object_verify" in refusal


# ── Postgres ────────────────────────────────────────────────────────────────


@pytest.fixture
def scratch(db_connection):
    cur = db_connection.cursor()
    schema = f"mbs2c_{uuid.uuid4().hex[:12]}"
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(f"SET LOCAL search_path TO {schema}")
    cur.execute(_MIGRATION.read_text())
    return db_connection


@pytest.fixture
def committed_schema(db_url):
    """A committed scratch schema (two connections must both see it),
    dropped afterwards."""
    import psycopg2

    schema = f"mbs2c_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(db_url)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE SCHEMA {schema}")
    admin.cursor().execute(f"SET search_path TO {schema}")
    admin.cursor().execute(_MIGRATION.read_text())
    try:
        yield schema
    finally:
        admin.cursor().execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


def _connect(db_url: str, schema: str) -> Any:
    import psycopg2

    conn = psycopg2.connect(db_url)
    conn.cursor().execute(f"SET search_path TO {schema}")
    conn.commit()
    return conn


@pytest.mark.integration
def test_two_connections_racing_for_one_key_one_wins(db_url, committed_schema) -> None:
    first, second = _connect(db_url, committed_schema), _connect(db_url, committed_schema)
    try:
        won = provenance.reserve_create_only(first, OUT, "reviews/k.jsonl")
        # The second blocks on the first's uncommitted row; resolve it in a thread.
        result: list[bool] = []
        racer = threading.Thread(
            target=lambda: result.append(
                provenance.reserve_create_only(second, OUT, "reviews/k.jsonl")
            )
        )
        racer.start()
        first.commit()
        racer.join(10)
        second.commit()
        assert (won, result) == (True, [False])
    finally:
        first.close()
        second.close()


@pytest.mark.integration
def test_the_verdict_upserts_on_postgres(scratch) -> None:
    s3 = _S3()
    _stored(s3, key="old/clip.mp4")
    outcome = provenance.verify_object(s3, scratch, OUT, "old/clip.mp4")
    assert outcome.sha256_state == "verified"
    s3.objects[(OUT, "old/clip.mp4")] = b"tampered" + BODY[8:]
    s3.etags[(OUT, "old/clip.mp4")] = _md5(BODY)
    again = provenance.verify_object(s3, scratch, OUT, "old/clip.mp4")
    assert again.sha256_state == "mismatch"
    row = provenance.lookup(scratch, OUT, "old/clip.mp4", _md5(BODY))
    assert (row.sha256, row.sha256_state, row.source_kind) == (SHA, "mismatch", "legacy")


@pytest.mark.integration
def test_a_reservation_is_invisible_to_reads_on_postgres(scratch) -> None:
    assert provenance.reserve_create_only(scratch, OUT, "reviews/k.jsonl")
    assert not provenance.reserve_create_only(scratch, OUT, "reviews/k.jsonl")
    assert provenance.lookup(scratch, OUT, "reviews/k.jsonl").legacy
    found = provenance.lookup_many(scratch, [(OUT, "reviews/k.jsonl", "")])
    assert found[(OUT, "reviews/k.jsonl")].legacy


# ── review fixes (MB-S2c adversarial review, 2026-11-27) ────────────────────


class _HeadDownS3(_S3):
    """The store answers HEAD with an error that is not "no such key"."""

    def head_object(self, *, Bucket, Key):  # noqa: N803, ARG002 — boto3
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "503", "Message": "SlowDown"}}, "HeadObject")


def test_a_create_only_save_when_the_store_cannot_say_is_503_not_an_overwrite(
    monkeypatch,
) -> None:
    """A HEAD that fails for another reason than "no such key" (store busy,
    denied) read as "absent": the claim won and the PUT overwrote a review
    written before create-only existed. It fails closed now."""
    s3 = _HeadDownS3()
    s3.objects[(OUT, "reviews/k.jsonl")] = b"the first review"
    pool = _Pool()
    r = _save(_app(monkeypatch, s3, pool), "reviews/k.jsonl")
    assert r.status_code == 503, r.text
    assert r.headers[ERROR_CODE] == "files.create_only_unavailable"
    assert s3.objects[(OUT, "reviews/k.jsonl")] == b"the first review"
    assert pool.db.rows == {}


def test_a_create_only_save_of_a_missing_key_saves_on_a_real_404(monkeypatch) -> None:
    app = _app(monkeypatch, _S3(), _Pool())
    assert _save(app, "reviews/k.jsonl").status_code == 201
    assert _save(app, "reviews/k.jsonl").status_code == 409


class _SlowS3(_S3):
    """A re-read that blocks until the test lets it go (an 8 GiB object)."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def get_object(self, *, Bucket, Key):  # noqa: N803 — boto3
        self.release.wait(30)
        return super().get_object(Bucket=Bucket, Key=Key)


def _attempt(attempt_id: str = "att-1", ref: str | None = None) -> Any:
    return SimpleNamespace(attempt_id=attempt_id, backend_ref=ref)


async def _settle(checker: Any, ref: str, job: Any) -> Any:
    for _ in range(300):
        result = await checker.check_status(_attempt(ref=ref), job)
        if result.status != "running":
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("the verify never settled")


def test_submit_returns_before_the_re_read_so_the_scheduler_is_not_held(monkeypatch) -> None:
    """The scheduler awaits every submit in turn: a submit that awaited the
    whole re-read held every other job of the front for its duration."""
    s3, pool = _SlowS3(), _Pool()
    _stored(s3)
    provenance.set_provenance_pool(pool)
    monkeypatch.setattr(ingest_jobs, "_ingest_s3_client", lambda *_a: s3)
    job = _job(bucket=OUT, key="intake/a.mp4")

    async def scenario() -> Any:
        ref = await asyncio.wait_for(
            ObjectVerifyBackend().submit_job(job, _attempt()), timeout=2
        )
        checker = ObjectVerifyStatusChecker()
        assert (await checker.check_status(_attempt(ref=ref), job)).status == "running"
        s3.release.set()
        return await _settle(checker, ref, job)

    try:
        result = asyncio.run(scenario())
    finally:
        s3.release.set()
    assert result.status == "succeeded"
    assert result.metrics["sha256_state"] == "verified"


def test_a_verify_lost_to_a_restart_fails_its_attempt_for_a_retry() -> None:
    job = _job(bucket=OUT, key="intake/a.mp4")
    result = asyncio.run(
        ObjectVerifyStatusChecker().check_status(
            _attempt(ref="object_verify|attempt=gone-after-restart"), job
        )
    )
    assert result.status == "failed"
    assert "restart" in (result.exit_reason or "")


def test_a_gone_object_fails_the_attempt_in_the_checker(monkeypatch) -> None:
    s3 = _S3()
    provenance.set_provenance_pool(_Pool())
    monkeypatch.setattr(ingest_jobs, "_ingest_s3_client", lambda *_a: s3)
    job = _job(bucket=OUT, key="intake/gone.mp4")

    async def scenario() -> Any:
        ref = await ObjectVerifyBackend().submit_job(job, _attempt("att-g"))
        return await _settle(ObjectVerifyStatusChecker(), ref, job)

    result = asyncio.run(scenario())
    assert result.status == "failed"
    assert "no verdict" in (result.exit_reason or "")


def test_a_malformed_spec_fails_at_submit_not_in_the_background() -> None:
    with pytest.raises(ValueError):
        asyncio.run(ObjectVerifyBackend().submit_job(_job(bucket=OUT), _attempt()))


def test_the_http_submit_seam_refuses_a_malformed_spec_with_400() -> None:
    from fastapi import HTTPException
    from umrs.api import jobs as jobs_api

    with pytest.raises(HTTPException) as caught:
        jobs_api._enforce_spec_shape(JobType.OBJECT_VERIFY, {"bucket": OUT}, None)
    assert caught.value.status_code == 400
    with pytest.raises(HTTPException):
        jobs_api._enforce_spec_shape(
            JobType.DATA_INGESTION,
            {},
            [{"job_type": "object_verify", "spec": {"bucket": OUT, "sha256": "x"}}],
        )
    jobs_api._enforce_spec_shape(JobType.OBJECT_VERIFY, {"bucket": OUT, "key": "k"}, None)
