"""ADR-115 D-4 (slice MB-S2b): folder-shaped listing, raw-key metadata and
the per-execution outputs route.

Before this slice ``/umrs/files/list`` was flat (no ``Delimiter``, no ETag,
no provenance), the metadata route re-namespaced every key into the
caller's project (so ``runs/<id>/x.mp4`` in an owned ``-output`` bucket was
not addressable), and nothing listed what a run wrote except its in-memory
blobs. Now:

* ``delimiter=/`` answers one level: ``prefixes`` beside ``objects``;
  ``next_token`` pages both; each object carries its ETag;
* ``fields=provenance`` joins the page to ``object_provenance`` in one
  SELECT (never a ``head_object`` per key);
* ``/files/metadata/{key}?bucket=`` (and the old route with an owned
  ``bucket``) read the raw key with its provenance at the live ETag;
* ``/workflows/executions/{id}/outputs`` is the union of the steps'
  ``output_keys`` joined to provenance, with the execution's visibility;
* an ADR-080 grantee lists and reads a shared bucket; anyone else is 403.

The ``integration`` test runs the batched lookup on real Postgres.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs import migrations
from umrs.api import files
from umrs.api.workflows import router as workflows_router
from umrs.data_catalog import provenance
from umrs.server import step_staging
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.shared.storage import user_buckets as ub

from umrs_protocol.errors import ERROR_CODE_HEADER

OWNED = "alice.p.v"
OUT = "alice.p.v-output"
SHA = hashlib.sha256(b"clip").hexdigest()


# ── doubles ─────────────────────────────────────────────────────────────────


def _etag(body: bytes) -> str:
    return hashlib.md5(body).hexdigest()


class _S3:
    """``list_objects_v2`` with ``Delimiter``/``MaxKeys``/``ContinuationToken``
    the way S3 answers them (keys and common prefixes share one budget)."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.heads: list[tuple[str, str]] = []

    def put(self, bucket: str, key: str, body: bytes = b"x") -> None:
        self.objects[(bucket, key)] = body

    def _entries(self, bucket: str, prefix: str, delimiter: str) -> list[tuple[str, str]]:
        entries: dict[str, str] = {}
        for (b, key) in sorted(self.objects):
            if b != bucket or not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if delimiter and delimiter in rest:
                folder = prefix + rest.split(delimiter, 1)[0] + delimiter
                entries.setdefault(folder, "prefix")
            else:
                entries[key] = "object"
        return sorted(entries.items())

    def _item(self, bucket: str, key: str) -> dict[str, Any]:
        body = self.objects[(bucket, key)]
        return {
            "Key": key,
            "Size": len(body),
            "ETag": f'"{_etag(body)}"',
            "LastModified": datetime(2026, 11, 27, tzinfo=timezone.utc),
        }

    def list_objects_v2(self, **kw: Any) -> dict[str, Any]:
        entries = self._entries(kw["Bucket"], kw.get("Prefix", ""), kw.get("Delimiter", ""))
        start = int(kw.get("ContinuationToken") or 0)
        page = entries[start : start + kw["MaxKeys"]]
        out: dict[str, Any] = {
            "Contents": [self._item(kw["Bucket"], k) for k, kind in page if kind == "object"],
            "CommonPrefixes": [{"Prefix": k} for k, kind in page if kind == "prefix"],
            "IsTruncated": start + len(page) < len(entries),
        }
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + len(page))
        return out

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 — boto3
        self.heads.append((Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        body = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(body),
            "ContentType": "video/mp4",
            "ETag": f'"{_etag(body)}"',
            "LastModified": datetime(2026, 11, 27, tzinfo=timezone.utc),
            "Metadata": {"filename": Key.rsplit("/", 1)[-1]},
        }


class _Config:
    bucket = "r4-data"

    def __init__(self, s3: _S3) -> None:
        self._s3 = s3

    def get_client(self) -> _S3:
        return self._s3


class _Cursor:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._pool.selects.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._pool.rows)


class _Pool:
    """Answers every SELECT with ``rows`` and counts the SELECTs."""

    def __init__(self, rows: list[provenance.ObjectProvenance] | None = None) -> None:
        self.rows = [
            tuple(row.as_dict()[c] for c in provenance._COLUMNS) for row in rows or []
        ]
        self.selects: list[tuple[str, tuple[Any, ...]]] = []

    def getconn(self) -> _Pool:
        return self

    def putconn(self, _conn: Any) -> None:
        return None

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def rollback(self) -> None:
        return None


def _row(bucket: str, key: str, etag: str, **over: Any) -> provenance.ObjectProvenance:
    base = {
        "bucket": bucket,
        "key": key,
        "etag": etag,
        "size": 4,
        "sha256": SHA,
        "sha256_state": "verified",
        "content_type": "video/mp4",
        "source_kind": "step_output",
        "execution_id": "wf-1",
        "step": "detect",
    }
    return provenance.ObjectProvenance(**{**base, **over})


class _Buckets:
    """``user_buckets`` store: alice owns ``alice.p.v``; ``access`` admits."""

    def __init__(self, access: tuple[str, ...] = ()) -> None:
        self.access = access

    def get(self, bucket: str) -> ub.UserBucket | None:
        if ub.input_bucket_of(bucket) != OWNED:
            return None
        return ub.UserBucket(OWNED, "alice", "p", "v", self.access)

    def owner_of(self, bucket: str) -> str | None:
        return "alice" if ub.input_bucket_of(bucket) == OWNED else None

    def can_access(self, bucket: str, user_id: str) -> bool:
        if ub.input_bucket_of(bucket) != OWNED:
            return False
        return user_id == "alice" or f"user:{user_id}" in self.access


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> _S3:
    fake = _S3()
    monkeypatch.setattr(files, "get_s3_config", lambda: _Config(fake))
    monkeypatch.setattr(files, "bucket_store", lambda *_a: _Config(fake))
    monkeypatch.setattr(step_staging, "_bucket_registry", None)
    ub.set_user_bucket_store(_Buckets(access=("user:bob",)))  # type: ignore[arg-type]
    yield fake
    ub.set_user_bucket_store(None)


def _files_client(s3: _S3, user: str = "alice", pool: Any = None) -> TestClient:
    app = FastAPI()
    app.include_router(files.router)
    app.state.umrs_db_pool = pool
    app.dependency_overrides[files.get_project_id] = lambda: "pa"
    app.dependency_overrides[files.get_project_s3_config] = lambda: _Config(s3)
    client = TestClient(app)
    client.headers.update({"X-User-ID": user})
    return client


def _list(client: TestClient, **params: Any) -> dict[str, Any]:
    r = client.get("/umrs/files/list", params={"bucket": OWNED, **params})
    assert r.status_code == 200, r.text
    return r.json()


# ── tree listing ────────────────────────────────────────────────────────────


def test_delimiter_lists_one_level_with_prefixes(s3) -> None:
    for key in ("a/1", "a/b/2", "c"):
        s3.put(OWNED, key)
    body = _list(_files_client(s3), delimiter="/")
    assert body["prefixes"] == ["a/"]
    assert [o["key"] for o in body["objects"]] == ["c"]
    assert body["delimiter"] == "/" and body["next_token"] is None


def test_delimiter_under_a_prefix_lists_its_children(s3) -> None:
    for key in ("a/1", "a/b/2", "c"):
        s3.put(OWNED, key)
    body = _list(_files_client(s3), delimiter="/", prefix="a/")
    assert body["prefixes"] == ["a/b/"]
    assert [o["key"] for o in body["objects"]] == ["a/1"]


def test_without_a_delimiter_the_listing_stays_flat(s3) -> None:
    for key in ("a/1", "a/b/2", "c"):
        s3.put(OWNED, key)
    body = _list(_files_client(s3))
    assert body["prefixes"] == []
    assert [o["key"] for o in body["objects"]] == ["a/1", "a/b/2", "c"]


def test_1001_keys_page_with_next_token(s3) -> None:
    for i in range(1001):
        s3.put(OWNED, f"k/{i:04d}")
    client = _files_client(s3)
    first = _list(client, prefix="k/", delimiter="/")
    assert len(first["objects"]) == 1000 and first["next_token"]
    second = _list(client, prefix="k/", delimiter="/", token=first["next_token"])
    assert [o["key"] for o in second["objects"]] == ["k/1000"]
    assert second["next_token"] is None


def test_each_listed_object_carries_its_unquoted_etag(s3) -> None:
    s3.put(OWNED, "c", b"clip")
    [obj] = _list(_files_client(s3))["objects"]
    assert obj["etag"] == _etag(b"clip")


@pytest.mark.parametrize("params", [{"delimiter": ","}, {"fields": "owner"}])
def test_unknown_list_options_are_refused_by_name(s3, params) -> None:
    r = _files_client(s3).get("/umrs/files/list", params={"bucket": OWNED, **params})
    assert r.status_code == 400
    assert r.headers[ERROR_CODE_HEADER] == "files.list_option_invalid"


def test_fields_provenance_joins_the_page_in_one_select(s3) -> None:
    s3.put(OWNED, "a.mp4", b"clip")
    s3.put(OWNED, "b.mp4", b"other")
    pool = _Pool([_row(OWNED, "a.mp4", _etag(b"clip"))])
    body = _list(_files_client(s3, pool=pool), fields="provenance")
    by_key = {o["key"]: o["provenance"] for o in body["objects"]}
    assert by_key["a.mp4"]["sha256"] == SHA and by_key["a.mp4"]["source_kind"] == "step_output"
    assert by_key["b.mp4"]["source_kind"] == "legacy" and by_key["b.mp4"]["sha256"] is None
    assert len(pool.selects) == 1
    assert s3.heads == []  # never a head_object per key


def test_a_row_of_another_version_does_not_describe_the_object(s3) -> None:
    s3.put(OWNED, "a.mp4", b"rewritten")
    pool = _Pool([_row(OWNED, "a.mp4", _etag(b"clip"))])
    [obj] = _list(_files_client(s3, pool=pool), fields="provenance")["objects"]
    assert obj["provenance"]["source_kind"] == "legacy"


def test_fields_provenance_without_a_database_reads_legacy(s3) -> None:
    s3.put(OWNED, "a.mp4", b"clip")
    [obj] = _list(_files_client(s3), fields="provenance")["objects"]
    assert obj["provenance"]["source_kind"] == "legacy"


def test_an_adr080_grantee_lists_the_shared_bucket(s3) -> None:
    s3.put(OWNED, "a/1")
    body = _list(_files_client(s3, user="bob"), delimiter="/")
    assert body["prefixes"] == ["a/"]


def test_a_non_member_gets_403_on_list_and_metadata(s3) -> None:
    s3.put(OUT, "runs/wf-1/x.mp4")
    client = _files_client(s3, user="carol")
    assert client.get("/umrs/files/list", params={"bucket": OWNED}).status_code == 403
    meta = client.get("/umrs/files/metadata/runs/wf-1/x.mp4", params={"bucket": OUT})
    assert meta.status_code == 403
    assert s3.heads == []  # refused before any store call


# ── raw-key metadata ────────────────────────────────────────────────────────


def test_raw_key_metadata_returns_that_keys_etag_and_provenance(s3) -> None:
    s3.put(OUT, "runs/wf-1/x.mp4", b"clip")
    pool = _Pool([_row(OUT, "runs/wf-1/x.mp4", _etag(b"clip"))])
    r = _files_client(s3, pool=pool).get(
        "/umrs/files/metadata/runs/wf-1/x.mp4", params={"bucket": OUT}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["bucket"], body["key"]) == (OUT, "runs/wf-1/x.mp4")
    assert body["etag"] == _etag(b"clip") and body["content_type"] == "video/mp4"
    assert body["provenance"]["sha256"] == SHA and body["provenance"]["etag"] == body["etag"]


def test_old_metadata_route_with_an_owned_bucket_reads_the_raw_key(s3) -> None:
    s3.put(OUT, "runs/wf-1/x.mp4", b"clip")
    r = _files_client(s3).get(
        "/umrs/files/runs/wf-1/x.mp4/metadata", params={"bucket": OUT}
    )
    assert r.status_code == 200, r.text
    assert r.json()["key"] == "runs/wf-1/x.mp4"  # was "pa/runs/wf-1/x.mp4"
    assert r.json()["etag"] == _etag(b"clip")


def test_old_metadata_route_without_a_bucket_stays_project_namespaced(s3) -> None:
    s3.put("r4-data", "pa/uploads/x.mp4", b"clip")
    r = _files_client(s3).get("/umrs/files/uploads/x.mp4/metadata")
    assert r.status_code == 200, r.text
    assert r.json()["key"] == "pa/uploads/x.mp4"


def test_raw_key_metadata_needs_a_bucket_and_refuses_traversal(s3) -> None:
    client = _files_client(s3)
    assert client.get("/umrs/files/metadata/runs/x.mp4").status_code == 422
    bad = client.get("/umrs/files/metadata/runs/../x.mp4", params={"bucket": OUT})
    assert bad.status_code in (400, 404)


def test_raw_key_metadata_of_a_missing_key_is_404(s3) -> None:
    r = _files_client(s3).get("/umrs/files/metadata/nope.mp4", params={"bucket": OUT})
    assert r.status_code == 404
    assert r.headers[ERROR_CODE_HEADER] == "files.not_found"


# ── executions outputs ──────────────────────────────────────────────────────


@pytest.fixture
def registry():
    reset_execution_registry_for_tests()
    yield get_execution_registry()
    reset_execution_registry_for_tests()


def _wf_client(pool: Any = None) -> TestClient:
    app = FastAPI()
    app.include_router(workflows_router)
    app.state.umrs_db_pool = pool
    return TestClient(app)


def _caller(user: str = "alice", project: str = "project-a") -> dict[str, str]:
    return {"X-User-ID": user, "X-Verified-Project-ID": project}


def _run(registry: Any) -> Any:
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a", "bucket": OWNED},
        identity_provenance="gateway_verified",
    )
    record.results["steps"] = {
        "detect": {"output_keys": [f"runs/{record.execution_id}/detect/x.mp4"], "extra": {}},
        "aggregate": {
            "output_keys": [
                f"s3://{OUT}/runs/{record.execution_id}/agg/events.json",
                f"runs/{record.execution_id}/detect/x.mp4",  # declared twice
            ],
            "extra": {},
        },
        "leak": {"output_keys": ["s3://someone-else/secret.bin"], "extra": {}},
    }
    record.update(status="succeeded", progress=1.0)
    return record


def _outputs(registry: Any) -> tuple[Any, _Pool, dict[str, Any]]:
    record = _run(registry)
    clip = f"runs/{record.execution_id}/detect/x.mp4"
    pool = _Pool([_row(OUT, clip, "e1", execution_id=record.execution_id)])
    r = _wf_client(pool).get(
        f"/umrs/workflows/executions/{record.execution_id}/outputs", headers=_caller()
    )
    assert r.status_code == 200, r.text
    return record, pool, r.json()


def test_outputs_are_the_union_of_declared_keys(registry) -> None:
    record, _pool, body = _outputs(registry)
    assert body["execution_status"] == "succeeded"
    got = [(o["step"], o["bucket"], o["key"]) for o in body["outputs"]]
    assert got == [
        ("detect", OUT, f"runs/{record.execution_id}/detect/x.mp4"),
        ("aggregate", OUT, f"runs/{record.execution_id}/agg/events.json"),
        ("leak", "someone-else", "secret.bin"),
    ]


def test_outputs_join_provenance_in_one_select(registry) -> None:
    record, pool, body = _outputs(registry)
    prov = {o["key"]: o["provenance"] for o in body["outputs"]}
    assert prov[f"runs/{record.execution_id}/detect/x.mp4"]["sha256"] == SHA
    assert prov[f"runs/{record.execution_id}/agg/events.json"]["source_kind"] == "legacy"
    assert prov["secret.bin"] is None  # never a row for a bucket the run does not own
    assert len(pool.selects) == 1


def test_outputs_of_another_projects_run_are_403(registry) -> None:
    record = _run(registry)
    r = _wf_client().get(
        f"/umrs/workflows/executions/{record.execution_id}/outputs",
        headers=_caller(user="bob", project="project-b"),
    )
    assert r.status_code == 403


def test_outputs_of_an_unknown_run_are_404(registry) -> None:
    r = _wf_client().get("/umrs/workflows/executions/wf-nope/outputs", headers=_caller())
    assert r.status_code == 404


def test_a_run_with_no_steps_yet_has_no_outputs(registry) -> None:
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    r = _wf_client().get(
        f"/umrs/workflows/executions/{record.execution_id}/outputs", headers=_caller()
    )
    assert r.status_code == 200 and r.json()["outputs"] == []


# ── the batched lookup on Postgres ──────────────────────────────────────────

_MIGRATION = next(migrations.MIGRATIONS_DIR.glob("0142_object_provenance.sql"))


@pytest.mark.integration
def test_lookup_many_on_postgres(db_connection) -> None:
    cur = db_connection.cursor()
    schema = f"mbs2b_{uuid.uuid4().hex[:12]}"
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(f"SET LOCAL search_path TO {schema}")
    cur.execute(_MIGRATION.read_text())
    provenance.record(db_connection, _row(OUT, "a.mp4", "e1"))
    provenance.record(db_connection, _row(OUT, "a.mp4", "e2", sha256_state="pending"))
    found = provenance.lookup_many(
        db_connection, [(OUT, "a.mp4", "e1"), (OUT, "b.mp4", ""), (OWNED, "a.mp4", "")]
    )
    assert found[(OUT, "a.mp4")].etag == "e1" and found[(OUT, "a.mp4")].sha256 == SHA
    assert found[(OUT, "b.mp4")].legacy and found[(OWNED, "a.mp4")].legacy


# ── review MB-S2b (2026-11-27) ──────────────────────────────────────────────


class _S3NoEtag(_S3):
    """A store whose listing carries no ETag (some S3-compatible gateways)."""

    def _item(self, bucket: str, key: str) -> dict[str, Any]:
        item = super()._item(bucket, key)
        item.pop("ETag")
        return item


def test_an_object_listed_without_an_etag_reads_legacy(monkeypatch) -> None:
    """Without the listed ETag no row can prove it describes this version:
    the newest row of *any* version must not stand in for it."""
    fake = _S3NoEtag()
    monkeypatch.setattr(files, "get_s3_config", lambda: _Config(fake))
    monkeypatch.setattr(files, "bucket_store", lambda *_a: _Config(fake))
    monkeypatch.setattr(step_staging, "_bucket_registry", None)
    ub.set_user_bucket_store(_Buckets())  # type: ignore[arg-type]
    try:
        fake.put(OWNED, "a.mp4", b"rewritten")
        pool = _Pool([_row(OWNED, "a.mp4", _etag(b"clip"))])
        [obj] = _list(_files_client(fake, pool=pool), fields="provenance")["objects"]
    finally:
        ub.set_user_bucket_store(None)
    assert obj["etag"] is None
    assert obj["provenance"]["source_kind"] == "legacy"


class _Jobs:
    def __init__(self, job: Any) -> None:
        self.job = job

    def list_jobs(self, **_kw: Any) -> list[Any]:
        return [self.job]


def _durable_only_run(monkeypatch: pytest.MonkeyPatch, execution_id: str) -> None:
    """A finished run the registry no longer holds (a server restart): only
    its durable job is left."""
    from umrs.api import workflows as wf_api
    from umrs.jobs.models import Job, JobStatus, JobType, RestartPolicy

    now = datetime.now(timezone.utc)
    job = Job(
        job_id="job-restart",
        project_id="project-a",
        user_id="alice",
        job_type=JobType.WORKFLOW_RUN,
        spec={
            "workflow_name": "wf",
            "execution_id": execution_id,
            "params": {"project_id": "project-a", "bucket": OWNED},
        },
        idempotency_key="idem-restart",
        status=JobStatus.COMPLETED,
        restart_policy=RestartPolicy.ON_CRASH,
        created_at=now,
        updated_at=now,
        last_state_change_at=now,
    )
    monkeypatch.setattr(wf_api, "get_optional_job_store", lambda: _Jobs(job))


def test_outputs_after_a_restart_come_from_the_runs_step_output_rows(
    registry, monkeypatch
) -> None:
    """Every converge restarts the front and empties the in-memory registry;
    the run's ``step_output`` provenance rows (execution id, step) still say
    what it wrote, so its outputs do not vanish."""
    _durable_only_run(monkeypatch, "wf-restart")
    clip = "runs/wf-restart/detect/x.mp4"
    pool = _Pool([_row(OUT, clip, "e1", execution_id="wf-restart")])
    r = _wf_client(pool).get("/umrs/workflows/executions/wf-restart/outputs", headers=_caller())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["execution_status"] == "succeeded"
    assert body["outputs_source"] == "provenance"
    [out] = body["outputs"]
    assert (out["step"], out["bucket"], out["key"]) == ("detect", OUT, clip)
    assert out["provenance"]["sha256"] == SHA
    [(sql, params)] = pool.selects
    assert "execution_id" in sql
    assert params[0] == "wf-restart" and sorted(params[1]) == [OWNED, OUT]


def test_outputs_of_a_live_run_say_they_come_from_its_record(registry) -> None:
    _record, _pool, body = _outputs(registry)
    assert body["outputs_source"] == "run_record"


@pytest.mark.integration
def test_rows_of_execution_on_postgres(db_connection) -> None:
    cur = db_connection.cursor()
    schema = f"mbs2b_{uuid.uuid4().hex[:12]}"
    cur.execute(f"CREATE SCHEMA {schema}")
    cur.execute(f"SET LOCAL search_path TO {schema}")
    cur.execute(_MIGRATION.read_text())
    for key, etag, run in (
        ("runs/r/a.mp4", "e1", "r"),
        ("runs/r/a.mp4", "e2", "r"),
        ("runs/r/b.json", "e3", "r"),
        ("runs/q/c.mp4", "e4", "q"),
    ):
        provenance.record(db_connection, _row(OUT, key, etag, execution_id=run))
    provenance.record(db_connection, _row("elsewhere", "runs/r/d", "e5", execution_id="r"))
    # one transaction stamps one now(): age the first version explicitly
    cur.execute(
        "UPDATE object_provenance SET created_at = now() - interval '1 hour' "
        "WHERE etag = 'e1'"
    )
    rows =provenance.rows_of_execution(db_connection, "r", [OWNED, OUT])
    assert sorted((row.key, row.etag) for row in rows) == [
        ("runs/r/a.mp4", "e2"),
        ("runs/r/b.json", "e3"),
    ]
