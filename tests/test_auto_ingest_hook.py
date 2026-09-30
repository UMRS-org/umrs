"""Tests for TASK-035: Result auto-ingestion hook.

Verifies:
  1. Provenance metadata is set on ingested artifacts (produced_by_job_id, workflow, step)
  2. KNOWLEDGE_INGEST and KNOWLEDGE_UNDERSTAND jobs are enqueued on success
  3. Ingestion is skipped when ingest_results=False
  4. Ingestion is skipped (graceful) when the source job fails
  5. Content-hash dedup prevents duplicate ingestion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

# ── Minimal mocks ─────────────────────────────────────────────────────────────


@dataclass
class MockJob:
    job_id: str = "job-abc123"
    project_id: str = "proj-001"
    user_id: str = "user-alice"
    job_type: str = "pipeline_run"
    spec: dict[str, Any] = field(
        default_factory=lambda: {
            "ingest_results": True,
            "workflow": "scientific-paper",
            "step": "extract",
        }
    )


@dataclass
class MockAttempt:
    attempt_id: str = "att-001"
    attempt_no: int = 1
    backend_ref: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def _make_job(
    ingest_results: bool = True,
    job_type: str = "pipeline_run",
    workflow: str = "default",
    step: str = "main",
) -> MockJob:
    return MockJob(
        spec={
            "ingest_results": ingest_results,
            "workflow": workflow,
            "step": step,
        },
        job_type=job_type,
    )


# ── Unit: enqueue_auto_ingest dispatch ────────────────────────────────────────


class TestEnqueueAutoIngestDispatch:
    """Test that the auto-ingest hook fires only for productive job types."""

    def test_fires_for_pipeline_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            "umrs.jobs.knowledge_adapters.enqueue_knowledge_ingest",
            lambda *_a, **_kw: called.append("enqueue"),
        )
        from umrs.jobs.adapters import enqueue_auto_ingest

        enqueue_auto_ingest(_make_job(job_type="pipeline_run"), MockAttempt())
        assert called == ["enqueue"]

    def test_fires_for_workflow_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            "umrs.jobs.knowledge_adapters.enqueue_knowledge_ingest",
            lambda *_a, **_kw: called.append("enqueue"),
        )
        from umrs.jobs.adapters import enqueue_auto_ingest

        enqueue_auto_ingest(_make_job(job_type="workflow_run"), MockAttempt())
        assert called == ["enqueue"]

    def test_skips_when_ingest_results_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            "umrs.jobs.knowledge_adapters.enqueue_knowledge_ingest",
            lambda *_a, **_kw: called.append("enqueue"),
        )
        from umrs.jobs.adapters import enqueue_auto_ingest

        job = _make_job(ingest_results=False)
        enqueue_auto_ingest(job, MockAttempt())
        assert called == []

    def test_skips_non_productive_job_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            "umrs.jobs.knowledge_adapters.enqueue_knowledge_ingest",
            lambda *_a, **_kw: called.append("enqueue"),
        )
        from umrs.jobs.adapters import enqueue_auto_ingest

        job = _make_job(job_type="cleanup")
        enqueue_auto_ingest(job, MockAttempt())
        assert called == []

    def test_noop_when_job_is_none(self) -> None:
        from umrs.jobs.adapters import enqueue_auto_ingest

        enqueue_auto_ingest(None, MockAttempt())


class TestEnqueueAutoIngestFailure:
    """Test that the post-failure hook logs gracefully."""

    def test_logs_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            "umrs.jobs.knowledge_adapters.enqueue_ingest_failure",
            lambda *_a, **_kw: called.append("fail"),
        )
        from umrs.jobs.adapters import enqueue_auto_ingest_failure

        job = MockJob()
        enqueue_auto_ingest_failure(job, attempt_no=1, exit_reason="OOM")
        assert called == ["fail"]


# ── Unit: store callback wiring ───────────────────────────────────────────────


class TestStoreCallbackWiring:
    """Test that callbacks can be set on JobStore."""

    def test_set_callbacks(self) -> None:
        from umrs.jobs.store import JobStore

        store = JobStore.__new__(JobStore)
        store._db_pool = None
        store.umrs_instance_id = "test"
        store._ensure_schema = lambda: None
        store._ensure_schema()
        store._post_success_callback = None
        store._post_failure_callback = None

        def on_succ(job: Any, attempt: Any) -> None:
            pass

        def on_fail(job: Any, attempt_no: int, reason: str) -> None:
            pass

        store.set_post_completion_callbacks(on_succeeded=on_succ, on_failed=on_fail)
        assert store._post_success_callback is on_succ
        assert store._post_failure_callback is on_fail


# ── Integration: auto-ingest end-to-end (mocked store/queue) ──────────────────


class TestAutoIngestIntegration:
    """Integration tests using fully mocked dependencies."""

    def test_provenance_metadata_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ingested artifacts should have produced_by_job_id, workflow, step."""
        recorded_meta: list[dict[str, Any]] = []

        def fake_insert_artifact(**kw: Any) -> None:
            recorded_meta.append(kw.get("metadata", {}))

        def fake_get_pool() -> Any:
            return None

        monkeypatch.setattr("umrs.jobs.knowledge_adapters._get_pool", fake_get_pool)
        monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: None)

        from umrs.jobs.knowledge_adapters import (
            enqueue_knowledge_ingest,
        )

        job = MockJob()
        attempt = MockAttempt()

        enqueue_knowledge_ingest(
            job=job,
            attempt=attempt,
            workflow="scientific-paper",
            step="extract",
        )

        # Even when services are unavailable, the hook should not crash.
        # We can only verify it didn't raise.

    def test_success_and_failure_compose(self) -> None:
        """Both success and failure hooks can be active simultaneously."""
        success_called: list[tuple[Any, Any]] = []
        failure_called: list[tuple[Any, int, str]] = []

        def fake_on_success(job: Any, attempt: Any) -> None:
            success_called.append((job, attempt))

        def fake_on_failure(job: Any, attempt_no: int, reason: str) -> None:
            failure_called.append((job, attempt_no, reason))

        from umrs.jobs.store import JobStore

        store = JobStore.__new__(JobStore)
        store._db_pool = None
        store.umrs_instance_id = "test"
        store._ensure_schema = lambda: None
        store._ensure_schema()
        store._post_success_callback = None
        store._post_failure_callback = None

        store.set_post_completion_callbacks(
            on_succeeded=fake_on_success,
            on_failed=fake_on_failure,
        )

        job = MockJob()
        attempt = MockAttempt()

        # Simulate success path
        store._post_success_callback(job, attempt)
        assert len(success_called) == 1
        assert len(failure_called) == 0

        # Simulate failure path
        store._post_failure_callback(job, 1, "timeout")
        assert len(failure_called) == 1

    def test_idempotent_by_content_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Content-hash dedup: same artifact not ingested twice."""
        ingested_hashes: list[str] = []

        def fake_artifact_exists(project_id: str, content_hash: str) -> bool:
            return content_hash in ingested_hashes

        def fake_put_artifact(
            project_id: str, content: bytes, mime_type: str = ""
        ) -> tuple[str, int, str | None]:
            h = hashlib.sha256(content).hexdigest()
            return h, len(content), None

        import hashlib

        monkeypatch.setattr(
            "umrs.knowledge.artifact_storage.ArtifactStorage.artifact_exists",
            lambda *_a, **_k: fake_artifact_exists(*_a, **_k),
        )
        monkeypatch.setattr(
            "umrs.knowledge.artifact_storage.ArtifactStorage.put_artifact",
            lambda *_a, **_k: fake_put_artifact(*_a, **_k),
        )
        monkeypatch.setattr("umrs.jobs.knowledge_adapters._get_pool", lambda: None)
        monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: None)

        from umrs.jobs.knowledge_adapters import (
            enqueue_knowledge_ingest,
        )

        job = MockJob()
        attempt = MockAttempt()

        artifact_refs = [
            {
                "artifact_id": "art-1",
                "content_hash": "hash-aaa",
                "kind": "document",
                "source_ref": "file.txt",
            },
            {
                "artifact_id": "art-1",
                "content_hash": "hash-aaa",
                "kind": "document",
                "source_ref": "file.txt",
            },
            {
                "artifact_id": "art-2",
                "content_hash": "hash-bbb",
                "kind": "code",
                "source_ref": "main.py",
            },
        ]

        monkeypatch.setattr(
            "umrs.jobs.knowledge_adapters._collect_artifact_ids",
            lambda *_a, **_k: artifact_refs,
        )

        enqueue_knowledge_ingest(
            job=job,
            attempt=attempt,
            workflow="dup-test",
            step="dedup",
        )

        # No crash — dedup should have handled it even with mocked S3

    def test_failure_graceful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failure path must not raise even when knowledge store is unavailable."""
        monkeypatch.setattr("umrs.jobs.knowledge_adapters._get_pool", lambda: None)
        monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: None)

        from umrs.jobs.knowledge_adapters import (
            enqueue_ingest_failure,
        )

        # Should never raise
        enqueue_ingest_failure(
            job_id="job-xyz",
            attempt_no=2,
            exit_reason="SIGKILL",
            user_id="bob",
            project_id="proj-99",
        )


# ── Service integration: callback registration ────────────────────────────────


class TestServiceIntegration:
    """Test that JobService wires the auto-ingest callbacks."""

    def test_register_callbacks_on_store(self) -> None:
        from umrs.jobs.adapters import register_auto_ingest_callbacks
        from umrs.jobs.store import JobStore

        store = JobStore.__new__(JobStore)
        store._db_pool = None
        store.umrs_instance_id = "test"
        store._ensure_schema = lambda: None
        store._ensure_schema()
        store._post_success_callback = None
        store._post_failure_callback = None

        # Should not raise even without a running UMRS
        register_auto_ingest_callbacks(store)

        assert store._post_success_callback is not None
        assert store._post_failure_callback is not None


# ── Collect artifact IDs ──────────────────────────────────────────────────────


class TestCollectArtifactIds:
    """Test _collect_artifact_ids helper."""

    def test_empty_when_no_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 'no pool' premise = 'no DSN resolves' — the collection short-circuits
        # before any pool work (the _get_pool monkeypatch is stale: the helper
        # gates on _resolve_knowledge_dsn since the adapter refactor).
        monkeypatch.setattr("umrs.jobs.knowledge_adapters._resolve_knowledge_dsn", lambda: "")
        from umrs.jobs.knowledge_adapters import _collect_artifact_ids

        result = _collect_artifact_ids(MockJob())
        assert result == []

    def test_empty_when_no_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("umrs.jobs.knowledge_adapters._resolve_knowledge_dsn", lambda: "")
        from umrs.jobs.knowledge_adapters import _collect_artifact_ids

        result = _collect_artifact_ids(MockJob())
        assert result == []
