"""Regression tests: auto-ingest wiring + ArtifactStore JSONB adaptation.

Covers the two live failures found after the PG blob round-trip landed:

1. ``register_auto_ingest_callbacks`` never assigned ``_auto_ingest_callback``
   (post-success hook crashed with ``'NoneType' object is not callable``
   forever — KNOWLEDGE_INGEST never fired).
2. ``ArtifactStore.create`` passed a raw dict into a jsonb column
   (psycopg2: ``can't adapt type 'dict'``).

The critical property under test is the *indirection through the module
globals*: registration must make the store-invoked hook reach the real
enqueue implementation.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
import umrs.jobs.adapters as adapters
from psycopg2.extras import Json
from umrs.artifact_registry.store import ArtifactStore
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs.models import JobType

# ── Fakes ────────────────────────────────────────────────────────────────────


async def _no_pool():
    return None


class _CapturingStore:
    """JobStore test double: records post-completion callback registrations."""

    def __init__(self) -> None:
        self.callbacks: dict[str, object] = {}

    def set_post_completion_callbacks(
        self, on_succeeded: object = None, on_failed: object = None
    ) -> None:
        self.callbacks["on_succeeded"] = on_succeeded
        self.callbacks["on_failed"] = on_failed


def _make_job() -> SimpleNamespace:
    return SimpleNamespace(
        job_id="job-2b",
        user_id="alice",
        project_id="proj-2b",
        priority=0,
        job_type=JobType.WORKFLOW_RUN,
        spec={"ingest_results": True, "workflow": "demo08_yield_eval", "step": "*"},
    )


_DESCRIPTOR = {
    "artifact_id": "wfart-aa11bb22",
    "content_hash": "aa11bb22cc33",
    "kind": "s3_uri",
    "source_ref": "s3://r4-data/jobs/job-2b/attempt_1/out.json",
}


# ── Test A: module-global wiring reaches the real enqueue path ───────────────


def test_register_assigns_module_globals() -> None:
    store = _CapturingStore()
    adapters.register_auto_ingest_callbacks(store)

    assert adapters._auto_ingest_callback is adapters.enqueue_auto_ingest
    assert adapters._auto_ingest_failure_callback is adapters.enqueue_auto_ingest_failure
    assert callable(store.callbacks["on_succeeded"])
    assert callable(store.callbacks["on_failed"])


def test_success_hook_reaches_enqueue_through_indirection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple] = []
    monkeypatch.setattr(
        ka,
        "enqueue_knowledge_ingest",
        lambda job, attempt, workflow="default", step="main": (
            captured.append(
                (job.job_id, workflow, step, attempt.attempt_id)
            )
            or ["ingest-job"]
        ),
    )
    store = _CapturingStore()
    adapters.register_auto_ingest_callbacks(store)

    on_succeeded = store.callbacks["on_succeeded"]
    on_succeeded(_make_job(), SimpleNamespace(attempt_id="att-2b"))  # type: ignore[operator]

    assert captured == [("job-2b", "demo08_yield_eval", "*", "att-2b")]


def test_success_hook_produces_real_knowledge_ingest_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full chain: global → enqueue_auto_ingest → KNOWLEDGE_INGEST JobSubmission."""
    monkeypatch.setattr(ka, "_collect_artifact_ids", lambda _job: [dict(_DESCRIPTOR)])
    monkeypatch.setattr(
        ka, "_dedup_artifacts", lambda _refs, _pid: ([dict(_DESCRIPTOR)], "r4-data")
    )
    monkeypatch.setattr(ka, "_created_pool", _no_pool)

    submissions: list[object] = []

    class _Svc:
        def submit_job(self, submission: object) -> str:
            submissions.append(submission)
            return "ingest-job-1"

    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())

    store = _CapturingStore()
    adapters.register_auto_ingest_callbacks(store)
    on_succeeded = store.callbacks["on_succeeded"]
    on_succeeded(_make_job(), SimpleNamespace(attempt_id="att-2b"))  # type: ignore[operator]

    from umrs.jobs.models import JobType as _JT

    by_type = {s.job_type: s for s in submissions}  # type: ignore[attr-defined]
    assert _JT.KNOWLEDGE_INGEST in by_type
    spec = by_type[_JT.KNOWLEDGE_INGEST].spec  # type: ignore[attr-defined]
    assert spec["bucket"] == "r4-data"
    assert spec["project_id"] == "proj-2b"
    src = spec["sources"][0]
    assert src["content_hash"] == _DESCRIPTOR["content_hash"]
    assert src["uri"] == _DESCRIPTOR["source_ref"]
    # The descriptor-based UNDERSTAND submission rides along (Phase 2H).
    us = by_type.get(_JT.KNOWLEDGE_UNDERSTAND)
    assert us is not None
    assert us.spec["project_id"] == "proj-2b"  # type: ignore[attr-defined]
    assert us.spec["sources"][0]["content_hash"] == _DESCRIPTOR["content_hash"]  # type: ignore[attr-defined]


def test_failure_hook_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple] = []
    monkeypatch.setattr(
        ka,
        "enqueue_ingest_failure",
        lambda job_id, attempt_no, exit_reason, user_id, project_id: captured.append(
            (job_id, attempt_no, exit_reason, user_id, project_id)
        ),
    )
    store = _CapturingStore()
    adapters.register_auto_ingest_callbacks(store)

    on_failed = store.callbacks["on_failed"]
    on_failed(_make_job(), 2, "boom")  # type: ignore[operator]

    assert captured == [("job-2b", 2, "boom", "alice", "proj-2b")]


# ── Test B: ArtifactStore JSONB adaptation ───────────────────────────────────


class _FakeCursor:
    def __init__(self, row: tuple) -> None:
        self._row = row
        self.params: tuple | None = None
        self.sql: str | None = None

    def execute(self, sql: str, params: tuple) -> None:
        self.sql = sql
        self.params = params

    def fetchone(self) -> tuple:
        return self._row


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False

    def cursor(self) -> _FakeCursorCtx:
        return _FakeCursorCtx(self._cursor)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        pass


class _FakeCursorCtx:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> _FakeCursor:
        return self._cursor

    def __exit__(self, *_exc: object) -> None:
        pass


def _returning_row(metadata: dict) -> tuple:
    return (
        uuid.uuid4(),
        "result",
        "out.json",
        "alice",
        "proj-2b",
        "s3://r4-data/k",
        "r4-data",
        "jobs/job-2b/attempt_1/out.json",
        127,
        "aa11bb22",
        metadata,
        "wf-1",
        "demo08_yield_eval",
        "job-2b",
        1,
        None,
        None,
    )


def test_artifact_store_create_wraps_metadata_in_json_adapter() -> None:
    from umrs.artifact_registry.models import Artifact

    metadata = {"workflow": "demo08_yield_eval", "step": "*", "source": "workflow_run"}
    cursor = _FakeCursor(_returning_row(metadata))
    conn = _FakeConn(cursor)
    store = ArtifactStore(lambda: conn)

    artifact = Artifact(
        artifact_type="result",
        name="out.json",
        owner_id="alice",
        project_id="proj-2b",
        uri="s3://r4-data/jobs/job-2b/attempt_1/out.json",
        bucket="r4-data",
        s3_key="jobs/job-2b/attempt_1/out.json",
        size_bytes=127,
        checksum="aa11bb22",
        metadata=metadata,
        execution_id="wf-1",
        workflow_id="demo08_yield_eval",
        job_id="job-2b",
    )
    created = store.create(artifact)

    assert cursor.sql is not None and "INSERT INTO artifacts" in cursor.sql
    params = cursor.params
    assert params is not None
    # The exact regression guard: metadata must be adapted, never a raw dict.
    assert isinstance(params[10], Json)
    assert params[10].adapted == metadata
    # Other fields unchanged.
    assert params[1] == "result"
    assert params[2] == "out.json"
    assert params[3] == "alice"
    assert params[4] == "proj-2b"
    assert params[7] == "jobs/job-2b/attempt_1/out.json"
    assert params[12] == "demo08_yield_eval"
    assert params[13] == "job-2b"
    assert created.id is not None
    assert conn.committed


# ── Test C: combined artifact → registry → auto-ingest path ──────────────────


def test_combined_persist_registry_and_auto_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from umrs.jobs import workflow_run as wr

    monkeypatch.setenv("S3_BUCKET", "r4-data")
    blob = b'{"metrics": {"yield_pct": 85.0}}'
    row = _returning_row({"workflow": "w"})
    conn = _FakeConn(_FakeCursor(row))

    class _S3:
        def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> dict:
            return {"ETag": '"x"'}

    # 1) Persist blobs through the REAL ArtifactStore (Json adaptation live).
    record = SimpleNamespace(
        results={"blobs": {"b/demo08_yield_eval.json": blob}},
        execution_id="wf-2b",
        workflow_name="demo08_yield_eval",
        user_id="alice",
    )
    artifacts = wr._persist_run_artifacts(
        record=record,
        job=_make_job(),
        attempt=SimpleNamespace(
            attempt_id="att-2b",
            job_id="job-2b",
            attempt_no=1,
            output_prefix="jobs/job-2b/attempt_1",
        ),
        s3_client=_S3(),
        artifact_store=ArtifactStore(lambda: conn),
    )
    assert artifacts and artifacts[0]["registry_id"] not in (None, "")
    descriptor = artifacts[0]

    # 2) Auto-ingest through the REAL module-global wiring with REAL metrics.
    monkeypatch.setattr(ka, "_collect_artifact_ids", lambda _job: [descriptor])
    monkeypatch.setattr(ka, "_dedup_artifacts", lambda _refs, _pid: ([descriptor], "r4-data"))
    monkeypatch.setattr(ka, "_created_pool", _no_pool)

    submissions: list[object] = []

    class _Svc:
        def submit_job(self, submission: object) -> str:
            submissions.append(submission)
            return "ingest-job-2"

    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())

    store = _CapturingStore()
    adapters.register_auto_ingest_callbacks(store)
    on_succeeded = store.callbacks["on_succeeded"]
    on_succeeded(_make_job(), SimpleNamespace(attempt_id="att-2b"))  # type: ignore[operator]

    assert any(
        s.job_type is JobType.KNOWLEDGE_INGEST
        for s in submissions  # type: ignore[attr-defined]
    )
