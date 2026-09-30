"""Regression tests: workflow_run -> persistent artifacts -> knowledge ingest.

Covers the poster golden-path link that was broken:
    workflow blobs -> attempt-scoped S3 object -> ArtifactStore row
    -> metrics["artifacts"] -> auto-ingest gate (spec["ingest_results"]).

Test A: blob persistence (fake S3 + fake ArtifactStore, no real infra).
Test B: asyncio loop safety of the knowledge-artifact collectors.
Test C: durable workflow_run spec carries the ingest opt-in metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs import workflow_run as wr
from umrs.server.workflow_registry_handler import _submit_durable_workflow_run

# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> dict:
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body})
        return {"ETag": '"fake"'}


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.created: list[object] = []

    def create(self, artifact: object) -> SimpleNamespace:
        self.created.append(artifact)
        return SimpleNamespace(id=uuid4())


def _fake_job() -> SimpleNamespace:
    return SimpleNamespace(
        job_id="job-abc123",
        user_id="alice",
        project_id="proj-1",
        priority=0,
        spec={},
    )


def _fake_attempt() -> SimpleNamespace:
    return SimpleNamespace(
        attempt_id="att-xyz",
        job_id="job-abc123",
        attempt_no=2,
        output_prefix="jobs/job-abc123/attempt_2",
    )


def _fake_record(blobs: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        results={"blobs": blobs or {}},
        execution_id="wf-exec-1",
        workflow_name="demo08_yield_eval",
        user_id="alice",
    )


# ── Test A: blob persistence ─────────────────────────────────────────────────


def _run_persist(
    monkeypatch: pytest.MonkeyPatch, blob: bytes
) -> tuple[list, _FakeS3, _FakeArtifactStore]:
    monkeypatch.setenv("S3_BUCKET", "r4-data")
    record = _fake_record({"umrs-data/demo08_yield_eval.json": blob})
    s3 = _FakeS3()
    store = _FakeArtifactStore()
    artifacts = wr._persist_run_artifacts(
        record=record,
        job=_fake_job(),
        attempt=_fake_attempt(),
        s3_client=s3,
        artifact_store=store,
    )
    return artifacts, s3, store


_BLOB = b'{"metrics": {"yield_pct": 85.0}, "status": "succeeded"}'


def test_persist_uploads_attempt_scoped_s3_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, s3, _ = _run_persist(monkeypatch, _BLOB)
    assert len(s3.puts) == 1
    put = s3.puts[0]
    assert put["Key"] == "jobs/job-abc123/attempt_2/demo08_yield_eval.json"
    assert put["Body"] == _BLOB


def test_persist_emits_ingest_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts, _, _ = _run_persist(monkeypatch, _BLOB)
    assert artifacts, "metrics['artifacts'] must be non-empty"
    desc = artifacts[0]
    expected_sha = hashlib.sha256(_BLOB).hexdigest()
    assert desc["content_hash"] == expected_sha
    assert desc["artifact_id"] == f"wfart-{expected_sha[:16]}"
    assert desc["kind"] == "s3_uri"
    assert desc["source_ref"] == ("s3://r4-data/jobs/job-abc123/attempt_2/demo08_yield_eval.json")
    assert desc["size_bytes"] == len(_BLOB)


def test_persist_registers_artifact_with_lineage_linkage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, store = _run_persist(monkeypatch, _BLOB)
    assert len(store.created) == 1
    art = store.created[0]
    assert art.job_id == "job-abc123"
    assert art.workflow_id == "demo08_yield_eval"
    assert art.execution_id == "wf-exec-1"
    assert art.owner_id == "alice"
    assert art.project_id == "proj-1"


def test_persist_registers_artifact_checksum_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, store = _run_persist(monkeypatch, _BLOB)
    art = store.created[0]
    assert art.checksum == hashlib.sha256(_BLOB).hexdigest()
    assert art.s3_key == "jobs/job-abc123/attempt_2/demo08_yield_eval.json"
    assert art.artifact_type == "result"
    assert art.metadata["attempt_no"] == 2
    assert art.metadata["attempt_id"] == "att-xyz"


def test_persist_run_artifacts_metrics_via_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The success branch of WorkflowRunStatusChecker surfaces metrics['artifacts']."""
    record = _fake_record({"b/k.json": b"{}"})
    record.to_dict = lambda: {"status": "succeeded", "progress": 1.0, "steps": []}

    class _Registry:
        def get(self, _eid: str) -> SimpleNamespace:
            return record

    monkeypatch.setattr(
        "umrs.server.workflow_execution_registry.get_execution_registry",
        lambda: _Registry(),
    )
    captured: list[tuple] = []

    def _fake_persist(*, record, job, attempt):
        captured.append((job, attempt))
        return [
            {
                "artifact_id": "wfart-x",
                "content_hash": "deadbeef",
                "kind": "s3_uri",
                "source_ref": "s3://r4-data/jobs/j/attempt_1/k.json",
            }
        ]

    monkeypatch.setattr(wr, "_persist_run_artifacts", _fake_persist)

    checker = wr.WorkflowRunStatusChecker()
    job = _fake_job()
    job.spec = {"execution_id": "wf-exec-1", "workflow_name": "demo08_yield_eval"}
    attempt = _fake_attempt()
    result = asyncio.run(checker.check_status(attempt, job))

    assert captured, "persistence hook must fire on success"
    assert result.metrics["artifacts"][0]["artifact_id"] == "wfart-x"
    for key in ("artifact_id", "content_hash", "kind", "source_ref"):
        assert key in result.metrics["artifacts"][0]


def test_persist_run_artifacts_degraded_no_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 unavailable -> empty result, no exception (non-fatal degradation)."""
    monkeypatch.setattr(wr, "_open_artifact_store", lambda: (None, None))

    class _ExplodingModule:
        @staticmethod
        def _build_s3_client(_spec: dict) -> tuple[None, str]:
            return None, "r4-data"

    monkeypatch.setattr(ka, "_build_s3_client", _ExplodingModule._build_s3_client)
    record = _fake_record({"b/k.json": b"{}"})

    artifacts = wr._persist_run_artifacts(record=record, job=_fake_job(), attempt=_fake_attempt())
    assert artifacts == []


def test_persist_run_artifacts_degraded_no_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """PG registry unavailable -> S3 object still persisted, store skipped."""
    monkeypatch.setenv("S3_BUCKET", "r4-data")
    s3 = _FakeS3()

    class _FailingStore:
        def create(self, _artifact: object) -> None:
            raise RuntimeError("pg down")

    record = _fake_record({"b/out.json": b"data"})
    artifacts = wr._persist_run_artifacts(
        record=record,
        job=_fake_job(),
        attempt=_fake_attempt(),
        s3_client=s3,
        artifact_store=_FailingStore(),
    )
    assert len(s3.puts) == 1  # S3 write succeeded
    assert artifacts[0]["registry_id"] is None  # registry failure tolerated
    assert artifacts[0]["content_hash"] == hashlib.sha256(b"data").hexdigest()


def test_persist_run_artifacts_no_blobs() -> None:
    record = _fake_record({})
    artifacts = wr._persist_run_artifacts(record=record, job=_fake_job(), attempt=_fake_attempt())
    assert artifacts == []


# ── Test B: asyncio loop safety ──────────────────────────────────────────────


def test_run_coroutine_sync_without_running_loop() -> None:
    async def _inner() -> int:
        return 42

    assert ka._run_coroutine_sync(_inner(), "t-sync") == 42


def test_run_coroutine_sync_inside_running_loop() -> None:
    async def _inner() -> int:
        return 7

    async def _outer() -> int:
        # Inside a running loop: must NOT raise
        # RuntimeError("asyncio.run() cannot be called from a running event loop")
        return ka._run_coroutine_sync(_inner(), "t-async")

    assert asyncio.run(_outer()) == 7


def test_run_coroutine_sync_propagates_errors_inside_loop() -> None:
    async def _boom() -> None:
        raise ValueError("boom")

    async def _outer() -> None:
        with pytest.raises(ValueError, match="boom"):
            ka._run_coroutine_sync(_boom(), "t-err")

    asyncio.run(_outer())


class _FakeAsyncpgPool:
    """Minimal asyncpg-shaped pool: fetch/fetchval/close/wait_closed."""

    def __init__(self, metrics_row: dict | None) -> None:
        self._metrics_row = metrics_row
        self.closed = False

    async def fetch(self, _query: str, *_args: object) -> list[dict]:
        return [{"attempt_id": "att-1"}]

    async def fetchval(self, _query: str, *_args: object) -> dict | None:
        return self._metrics_row

    async def fetchrow(self, _query: str, *_args: object) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


def _open_pool_ctx(pool):
    """Return a context manager mimicking ka._open_pool yielding ``pool``."""

    @asynccontextmanager
    async def _ctx():
        yield pool

    return _ctx


def test_collect_artifact_ids_loop_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")  # satisfy the collector DSN gate
    expected = [
        {
            "artifact_id": "wfart-x",
            "content_hash": "deadbeef",
            "kind": "s3_uri",
            "source_ref": "s3://r4-data/jobs/j/attempt_1/k.json",
        }
    ]
    monkeypatch.setattr(ka, "_open_pool", _open_pool_ctx(_FakeAsyncpgPool({"artifacts": expected})))
    # knowledge_artifacts table read: make it return nothing extra.
    monkeypatch.setattr(ka, "_artifacts_from_store", _noop_store_rows)

    job = SimpleNamespace(job_id="j-1", project_id="proj-1", commit_attempt_id="att-1")

    # 1) Plain sync path (no running loop).
    sync_result = ka._collect_artifact_ids(job)
    assert sync_result and sync_result[0]["artifact_id"] == "wfart-x"

    # 2) Inside an already-running event loop: previously RuntimeError.
    async def _outer() -> list[dict]:
        return ka._collect_artifact_ids(job)

    async_result = asyncio.run(_outer())
    assert async_result and async_result[0]["source_ref"].startswith(
        "s3://r4-data/jobs/j/attempt_1/"
    )


async def _noop_store_rows(_pool: object, _project_id: str) -> list[dict]:
    return []


def test_collect_artifact_ids_accepts_jsonb_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ad-hoc asyncpg pools return JSONB as str — must parse."""
    monkeypatch.setenv("DB_HOST", "127.0.0.1")  # satisfy the collector DSN gate
    expected = [
        {
            "artifact_id": "wfart-jsonb",
            "content_hash": "aa11bb22cc33",
            "kind": "s3_uri",
            "source_ref": "s3://r4-data/jobs/j-1/attempt_1/out.json",
        }
    ]
    monkeypatch.setattr(
        ka,
        "_open_pool",
        _open_pool_ctx(_FakeAsyncpgPool(json.dumps({"artifacts": expected}))),
    )
    monkeypatch.setattr(ka, "_artifacts_from_store", _noop_store_rows)
    job = SimpleNamespace(job_id="j-1", project_id="proj-1", commit_attempt_id="att-1")

    sync_result = ka._collect_artifact_ids(job)
    assert sync_result and sync_result[0]["artifact_id"] == "wfart-jsonb"

    async def _outer() -> list[dict]:
        return ka._collect_artifact_ids(job)

    async_result = asyncio.run(_outer())
    assert async_result[0]["content_hash"] == "aa11bb22cc33"


def test_parse_metrics_jsonb_malformed_safe_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Malformed JSON: returns {} (no crash) and emits an observable warning."""
    assert ka._parse_metrics_jsonb("{not valid json") == {}
    assert ka._parse_metrics_jsonb(None) == {}
    assert ka._parse_metrics_jsonb(3.5) == {}
    with caplog.at_level(logging.WARNING, logger="umrs.jobs.knowledge_adapters"):
        ka._parse_metrics_jsonb('{"broken": true')
    assert any("Malformed metrics JSON" in r.message for r in caplog.records)


def test_artifacts_from_attempt_metrics_malformed_row_no_crash() -> None:
    pool = _FakeAsyncpgPool("{malformed")
    out = asyncio.run(ka._artifacts_from_attempt_metrics(pool, ["att-1"]))
    assert out == []


def test_list_project_artifact_sources_loop_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")  # satisfy the collector DSN gate
    row = SimpleNamespace(artifact_id="a1", content_hash="h", kind="s3_uri", source_ref="s3://b/k")

    class _Store:
        def __init__(self, _pool: object) -> None:
            pass

        async def list_artifacts(self, _project_id: str) -> tuple[list, int]:
            return [row], 1

    monkeypatch.setattr(ka, "_open_pool", _open_pool_ctx(_FakeAsyncpgPool(None)))
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", _Store)

    async def _outer() -> list[dict]:
        return ka._list_project_artifact_sources("proj-1")

    result = asyncio.run(_outer())
    assert result == [
        {"artifact_id": "a1", "content_hash": "h", "kind": "s3_uri", "source_ref": "s3://b/k"}
    ]


# ── Test C: ingest opt-in on the durable workflow_run spec ───────────────────


class _FakeJobBackend:
    def __init__(self) -> None:
        self.submitted_payload: dict | None = None

    def submit_workflow_run(self, _client_id: str, payload: str) -> str:
        self.submitted_payload = json.loads(payload)
        return json.dumps({"status": "ok", "job": {"job_id": "job-new-1"}})


def _submit_demo08() -> tuple[dict, dict]:
    backend = _FakeJobBackend()
    raw = _submit_durable_workflow_run(
        client_id="alice",
        workflow_name="demo08_yield_eval",
        params={"project_id": "proj-1"},
        execution_id="wf-new-1",
        source_path="/tmp/x",
        step_count=1,
        job_backend=backend,
    )
    return json.loads(raw), backend.submitted_payload or {}


def test_durable_workflow_run_spec_enables_ingest() -> None:
    resp, payload = _submit_demo08()
    assert resp["status"] == "ok"
    assert resp["durable_job_id"] == "job-new-1"
    assert payload["job_type"] == "workflow_run"
    spec = payload["spec"]
    # New ingest opt-in metadata.
    assert spec["ingest_results"] is True
    assert spec["workflow"] == "demo08_yield_eval"
    assert spec["step"] == "*"


def test_durable_workflow_run_spec_preserves_fields() -> None:
    _, payload = _submit_demo08()
    spec = payload["spec"]
    assert spec["workflow_name"] == "demo08_yield_eval"
    assert spec["execution_id"] == "wf-new-1"
    assert spec["params"] == {"project_id": "proj-1"}
    assert spec["source_path"] == "/tmp/x"


def test_ingest_gate_accepts_workflow_run_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """adapters.enqueue_auto_ingest must not early-return for this spec/type."""
    import umrs.jobs.adapters as adapters
    from umrs.jobs.models import JobType

    calls: list[tuple] = []

    def _capture(*, job, attempt, workflow, step) -> list[str]:
        calls.append((workflow, step))
        return ["ingest-job-1"]

    monkeypatch.setattr(ka, "enqueue_knowledge_ingest", _capture)

    job = SimpleNamespace(
        spec={"ingest_results": True, "workflow": "demo08_yield_eval", "step": "*"},
        job_type=JobType.WORKFLOW_RUN,
        job_id="job-1",
    )
    adapters.enqueue_auto_ingest(job, SimpleNamespace())

    assert calls == [("demo08_yield_eval", "*")]


# ── Ingest enqueue passthrough (descriptor -> KNOWLEDGE_INGEST spec) ─────────


def test_enqueue_ingest_job_descriptor_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[object] = []

    class _Svc:
        def submit_job(self, submission: object) -> str:
            submitted.append(submission)
            return "ingest-job-1"

    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())
    monkeypatch.setattr(
        ka,
        "_store_artifact_provenance",
        lambda _d, _p, _j, _a, _w, _s: None,
    )

    desc = {
        "artifact_id": "wfart-deadbeefcafe",
        "content_hash": "deadbeefcafe",
        "kind": "s3_uri",
        "source_ref": "s3://r4-data/jobs/job-1/attempt_1/out.json",
    }
    job_id = ka._enqueue_ingest_job(
        [desc], "r4-data", "proj-1", "alice", _fake_job(), _fake_attempt(), "demo08_yield_eval", "*"
    )
    assert job_id == "ingest-job-1"
    assert len(submitted) == 1
    spec = submitted[0].spec  # type: ignore[attr-defined]
    assert spec["bucket"] == "r4-data"
    assert spec["project_id"] == "proj-1"
    src = spec["sources"][0]
    assert src["kind"] == "s3_uri"
    assert src["uri"] == "s3://r4-data/jobs/job-1/attempt_1/out.json"
    assert src["artifact_id"] == "wfart-deadbeefcafe"
    assert src["content_hash"] == "deadbeefcafe"
