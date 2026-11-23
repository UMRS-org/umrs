"""ADR-115 S2a (slice MB-S2a): object provenance and the ``data.*`` audit family.

Before this slice the files upload route hashed MD5 only (the quota dedup
key), step-lane outputs were stored with no ContentType and no hash, and no
object had an origin record. Now:

* the upload route returns the server-computed SHA-256, writes a
  ``local_upload`` provenance row (``verified``: the server hashed every
  byte it stored) and one ``data.upload`` audit row;
* ``collect_out`` uploads each output with its ContentType, writes a
  ``step_output`` row per object and one ``data.step_output`` audit row;
* an object with no row reads back as legacy;
* none of it fails the write it describes (no database, no ETag, a broken
  pool: the object is simply legacy, verified on demand later).

The ``integration`` tests apply the migration twice in a scratch schema
(idempotent on a stand-alone stack that re-runs migrations at boot) and
round-trip rows on real Postgres; they roll back.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs import migrations
from umrs.api import audit_events, files
from umrs.audit import security_events
from umrs.data_catalog import provenance
from umrs.server.step_staging import collect_out

BUCKET = "test-bucket"
OUT = "alice.p.a-output"
BODY = b"frame-bytes-" * 5000
SHA = hashlib.sha256(BODY).hexdigest()


# ── doubles ─────────────────────────────────────────────────────────────────


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._conn.executed.append((sql, params))

    def fetchone(self) -> None:
        return None


class _Conn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.commits = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        return None


class _Pool:
    def __init__(self) -> None:
        self.conn = _Conn()

    def getconn(self) -> _Conn:
        return self.conn

    def putconn(self, _conn: _Conn) -> None:
        return None

    def rows(self) -> list[dict[str, Any]]:
        """Each committed INSERT as ``{column: value}``."""
        return [
            dict(zip(provenance._COLUMNS, params, strict=False))
            for sql, params in self.conn.executed
            if sql.startswith("INSERT INTO object_provenance")
        ]


class _BrokenPool:
    def getconn(self) -> None:
        raise ConnectionError("pool exhausted")


class _S3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.extra: dict[tuple[str, str], dict[str, Any]] = {}
        self.heads: list[tuple[str, str]] = []

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):  # noqa: N803 — boto3
        chunks = []
        while chunk := fileobj.read(4096):
            chunks.append(chunk)
        self.objects[(bucket, key)] = b"".join(chunks)
        self.extra[(bucket, key)] = dict(ExtraArgs or {})

    def head_object(self, *, Bucket, Key):  # noqa: N803 — boto3
        self.heads.append((Bucket, Key))
        body = self.objects[(Bucket, Key)]
        return {"ETag": '"' + hashlib.md5(body).hexdigest() + '"', "ContentLength": len(body)}

    def delete_object(self, **_kw: Any) -> None:
        return None


class _Config:
    bucket = BUCKET

    def __init__(self, client: _S3) -> None:
        self._client = client

    def get_client(self) -> _S3:
        return self._client


def _audit(event: str) -> list[dict[str, Any]]:
    return audit_events.query_audit_events(event=event, limit=50)


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(audit_events, "_audit_store", None)
    monkeypatch.setattr(audit_events, "_audit_buffer", [])
    provenance.set_provenance_pool(None)
    yield
    provenance.set_provenance_pool(None)


def _client(monkeypatch: pytest.MonkeyPatch, s3: _S3, pool: Any) -> TestClient:
    monkeypatch.setattr(files, "_fire_post_upload_ingest", lambda *_a: None)
    app = FastAPI()
    app.include_router(files.router)
    app.state.umrs_db_pool = pool
    app.dependency_overrides[files.get_project_id] = lambda: "pa"
    app.dependency_overrides[files.get_project_s3_config] = lambda: _Config(s3)
    return TestClient(app)


def _upload(client: TestClient, name: str = "clip.mp4") -> dict[str, Any]:
    r = client.post(
        "/umrs/files/upload",
        files={"file": (name, BODY, "video/mp4")},
        data={"purpose": "uploads"},
    )
    assert r.status_code == 200, r.text
    return r.json()


# ── upload route ────────────────────────────────────────────────────────────


def test_upload_returns_the_server_sha256(monkeypatch) -> None:
    body = _upload(_client(monkeypatch, _S3(), _Pool()))
    assert body["sha256"] == SHA


def test_upload_writes_a_verified_local_upload_row(monkeypatch) -> None:
    s3, pool = _S3(), _Pool()
    body = _upload(_client(monkeypatch, s3, pool))
    [row] = pool.rows()
    want = {
        "bucket": BUCKET,
        "key": body["key"],
        "etag": hashlib.md5(BODY).hexdigest(),  # the store's, unquoted
        "sha256": SHA,
        "size": len(BODY),
        "sha256_state": "verified",
        "source_kind": "local_upload",
        "content_type": "video/mp4",
        "project_id": "pa",
    }
    assert {name: row[name] for name in want} == want
    assert pool.conn.commits == 1


def test_upload_emits_one_data_upload_audit_row(monkeypatch) -> None:
    body = _upload(_client(monkeypatch, _S3(), _Pool()))
    [event] = _audit("data.upload")
    details = event["details"]
    assert event["scope"] == "project" and event["project_id"] == "pa"
    assert details["key"] == body["key"] and details["sha256"] == SHA
    assert details["provenance"] is True and details["data_schema"] == "1"


def test_upload_without_a_database_still_succeeds_and_reads_no_etag(monkeypatch) -> None:
    s3 = _S3()
    body = _upload(_client(monkeypatch, s3, None))
    assert body["sha256"] == SHA
    assert s3.heads == []
    assert _audit("data.upload")[0]["details"]["provenance"] is False


def test_a_broken_pool_never_fails_the_upload(monkeypatch) -> None:
    body = _upload(_client(monkeypatch, _S3(), _BrokenPool()))
    assert body["sha256"] == SHA
    assert _audit("data.upload")[0]["details"]["provenance"] is False


def test_owned_bucket_upload_records_provenance_too(monkeypatch) -> None:
    s3, pool = _S3(), _Pool()
    monkeypatch.setattr(files, "_require_owned_bucket", lambda *_a: None)
    monkeypatch.setattr(files, "bucket_store", lambda *_a: _Config(s3))
    client = _client(monkeypatch, s3, pool)
    r = client.post(
        "/umrs/files/upload",
        files={"file": ("a.mp4", BODY, "video/mp4")},
        data={"bucket": "alice.p.a", "key": "intake/a.mp4"},
        headers={"X-User-ID": "alice"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["sha256"] == SHA
    [row] = pool.rows()
    assert (row["bucket"], row["key"], row["uploader"]) == ("alice.p.a", "intake/a.mp4", "alice")


# ── step-lane collection ────────────────────────────────────────────────────


def _tar(*members: tuple[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def _collect(s3: _S3) -> dict[str, str]:
    stream = _tar((f"{OUT}/det/events.json", b"{}"), (f"{OUT}/clip.mp4", BODY))
    return collect_out(stream, s3, OUT, "wf-0123456789ab", "detect")


def test_step_outputs_get_a_content_type_from_their_extension() -> None:
    s3 = _S3()
    mapping = _collect(s3)
    assert s3.extra[(OUT, mapping["clip.mp4"])]["ContentType"] == "video/mp4"
    assert s3.extra[(OUT, mapping["det/events.json"])]["ContentType"] == "application/json"


def test_step_outputs_get_step_output_rows() -> None:
    s3, pool = _S3(), _Pool()
    provenance.set_provenance_pool(pool)
    mapping = _collect(s3)
    rows = {row["key"]: row for row in pool.rows()}
    assert set(rows) == {mapping["clip.mp4"], mapping["det/events.json"]}
    clip = rows[mapping["clip.mp4"]]
    assert clip["sha256"] == SHA and clip["sha256_state"] == "verified"
    assert clip["source_kind"] == "step_output"
    assert (clip["execution_id"], clip["step"]) == ("wf-0123456789ab", "detect")
    [event] = _audit("data.step_output")
    assert event["details"]["files"] == 2 and event["details"]["provenance_rows"] == 2


def test_step_collection_without_a_pool_reads_no_etag_and_still_audits() -> None:
    s3 = _S3()
    _collect(s3)
    assert s3.heads == []
    assert _audit("data.step_output")[0]["details"]["provenance_rows"] == 0


# ── module contract ─────────────────────────────────────────────────────────


def test_an_object_without_a_row_reads_as_legacy() -> None:
    found = provenance.lookup(_Conn(), BUCKET, "old/file.bin", "abc")
    assert found.legacy and found.sha256 is None and found.sha256_state == "pending"


def test_unknown_source_kind_is_refused() -> None:
    row = provenance.legacy_record(BUCKET, "k", "e")
    bad = provenance.ObjectProvenance(**{**row.as_dict(), "source_kind": "client_said_so"})
    with pytest.raises(ValueError, match="source_kind"):
        provenance.record(_Conn(), bad)


def test_no_etag_means_no_row() -> None:
    obj = provenance.HashedObject(BUCKET, "missing", 1, SHA, "text/plain")
    assert provenance.provenance_row(_S3(), obj, "local_upload") is None


class _RacedS3(_S3):
    """Another write lands on the key between our PUT and our HEAD."""

    def __init__(self, etag: str, length: int) -> None:
        super().__init__()
        self._etag, self._length = etag, length

    def head_object(self, *, Bucket, Key):  # noqa: N803 — boto3
        self.heads.append((Bucket, Key))
        return {"ETag": f'"{self._etag}"', "ContentLength": self._length}


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _hashed_body(key: str = "k") -> provenance.HashedObject:
    return provenance.HashedObject(BUCKET, key, len(BODY), SHA, "video/mp4", _md5(BODY))


def test_a_write_racing_on_the_key_never_gets_our_hash_as_verified(monkeypatch) -> None:
    s3, pool = _RacedS3(_md5(b"OTHER BYTES"), len(b"OTHER BYTES")), _Pool()
    body = _upload(_client(monkeypatch, s3, pool))
    assert body["sha256"] == SHA  # the upload itself still succeeds
    assert pool.rows() == []  # no row: the object stays legacy
    assert _audit("data.upload")[0]["details"]["provenance"] is False


def test_a_multipart_etag_of_our_size_is_pending_not_verified() -> None:
    row = provenance.provenance_row(
        _RacedS3(_md5(b"x") + "-3", len(BODY)), _hashed_body(), "local_upload"
    )
    assert row is not None and row.sha256_state == "pending" and row.sha256 == SHA


def test_a_multipart_etag_of_another_size_gets_no_row() -> None:
    row = provenance.provenance_row(
        _RacedS3(_md5(b"x") + "-3", len(BODY) + 1), _hashed_body(), "local_upload"
    )
    assert row is None


def test_a_single_part_etag_of_our_bytes_is_verified() -> None:
    row = provenance.provenance_row(
        _RacedS3(_md5(BODY).upper(), len(BODY)), _hashed_body(), "local_upload"
    )
    assert row is not None and row.sha256_state == "verified"


@pytest.mark.parametrize(
    "key",
    [
        "proj/eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhIn0.sig/a.txt",
        "proj/ghp_ABCDEF0123456789/x.csv",
        "proj/fe80::1/x",
        "proj/x/10.0.0.1.log",
    ],
)
def test_data_audit_key_redaction_checks_every_segment(key: str) -> None:
    clean = security_events.redact_details("data.upload", {"key": key})
    assert clean["key"] == "<redacted>"


def test_an_ordinary_dotted_key_is_kept() -> None:
    key = "pa/uploads/2026-11-24/clip_v1.2.mp4"
    assert security_events.redact_details("data.upload", {"key": key})["key"] == key


def test_data_audit_redacts_by_kind() -> None:
    clean = security_events.emit_data(
        "data.upload",
        key="bad\x01name",
        sha256="not-a-hash",
        bucket="eyJhbGciOi.x.y",
        prompt="free text",
    )
    assert clean is not None
    assert clean["key"] == clean["sha256"] == clean["bucket"] == "<redacted>"
    assert clean["dropped_keys"] == 1 and "prompt" not in clean


def test_data_and_security_names_share_one_block_and_do_not_cross() -> None:
    assert set(security_events.DATA_FAMILIES) <= set(security_events.AUDIT_FAMILIES)
    assert all(name.startswith("data.") for name in security_events.DATA_FAMILIES)
    assert not set(security_events.DATA_FAMILIES) & set(security_events.SECURITY_FAMILIES)
    assert security_events.emit_security("data.upload") is None
    assert security_events.emit_data("auth.failed") is None


# ── migration ───────────────────────────────────────────────────────────────

_MIGRATION = next(migrations.MIGRATIONS_DIR.glob("0142_object_provenance.sql"))


def test_the_migration_has_a_unique_number_after_0140() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    mine = _MIGRATION.stem.split("_")[0]
    assert versions.count(mine) == 1 and mine > "0140"


def test_the_migration_is_additive_and_idempotent_by_text() -> None:
    sql = " ".join(_MIGRATION.read_text().split()).upper()
    assert "CREATE TABLE IF NOT EXISTS OBJECT_PROVENANCE" in sql
    assert sql.count("CREATE INDEX IF NOT EXISTS") == sql.count("CREATE INDEX")
    assert "DROP " not in sql and "ALTER TABLE" not in sql


@pytest.fixture
def scratch(db_connection):
    """A scratch schema holding the migrated table, applied twice; the
    connection rolls back, so the schema never outlives the test."""
    cur = db_connection.cursor()
    schema = f"mbs2a_{uuid.uuid4().hex[:12]}"
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(f"SET LOCAL search_path TO {schema}")
    sql = _MIGRATION.read_text()
    cur.execute(sql)
    cur.execute(sql)  # a stand-alone stack re-runs it: no error, no change
    return db_connection


def _row(**over: Any) -> provenance.ObjectProvenance:
    base = provenance.ObjectProvenance(
        bucket=BUCKET,
        key="runs/wf-1/s/a.mp4",
        etag="e1",
        size=len(BODY),
        sha256=SHA,
        sha256_state="verified",
        content_type="video/mp4",
        source_kind="step_output",
        source_ref={"step": "s"},
        execution_id="wf-1",
        step="s",
    )
    return provenance.ObjectProvenance(**{**base.as_dict(), **over})


@pytest.mark.integration
def test_round_trip_on_postgres(scratch) -> None:
    provenance.record(scratch, _row())
    found = provenance.lookup(scratch, BUCKET, "runs/wf-1/s/a.mp4")
    assert found == _row()
    assert provenance.lookup(scratch, BUCKET, "never/written").legacy
    cur = scratch.cursor()
    cur.execute("SELECT verified_at IS NOT NULL FROM object_provenance")
    assert cur.fetchone() == (True,)


@pytest.mark.integration
def test_the_first_provenance_of_a_version_wins(scratch) -> None:
    provenance.record(scratch, _row())
    provenance.record(scratch, _row(source_kind="local_upload", uploader="mallory"))
    found = provenance.lookup(scratch, BUCKET, "runs/wf-1/s/a.mp4", "e1")
    assert found.source_kind == "step_output" and found.uploader is None


@pytest.mark.integration
def test_the_table_refuses_a_malformed_hash(scratch) -> None:
    import psycopg2

    with pytest.raises(psycopg2.errors.CheckViolation):
        provenance.record(scratch, _row(sha256="NOT-HEX"))
