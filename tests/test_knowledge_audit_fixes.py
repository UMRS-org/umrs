"""Audit 2026-08-25 fix pins (B1/B2/B3/B7/B8/B10/B12).

- B1  ingest route rejects body project ≠ verified project (409, fail loud)
- B2  ingest job spec always carries the resolved graph_id
- B3  KNOWLEDGE_EMBED backend: degraded no-op without embedder; chunk-node +
      vector wiring when embedder present
- B7  ingest/understand status checkers fail honestly on all-failed attempts
- B8  artifact dedup probe is per (project_id, content_hash)
- B10 /knowledge/search binds caller scope + project or fails closed
- B12 UMRS_AUTH_MODE defaults are strict everywhere; mvp is explicit opt-in
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from umrs.jobs.knowledge_adapters import (
    KnowledgeEmbedBackend,
    KnowledgeEmbedResult,
    KnowledgeEmbedStatusChecker,
    KnowledgeIngestStatusChecker,
    KnowledgeUnderstandStatusChecker,
    _knowledge_ref_counts,
)
from umrs.jobs.models import (
    DEFAULT_BACKOFFS,
    AttemptStatus,
    Job,
    JobAttempt,
    JobType,
    generate_attempt_id,
    generate_job_id,
)
from umrs.knowledge.embed_sandbox import InProcessChunkEmbedder

# ── small fakes ──────────────────────────────────────────────────────────────


def _attempt(ref: str) -> JobAttempt:
    return JobAttempt(
        attempt_id=generate_attempt_id(),
        job_id="job-x",
        attempt_no=1,
        backend="compose",
        status=AttemptStatus.RUNNING,
        backend_ref=ref,
        metrics={},
    )


def _job(job_type: JobType, spec: dict[str, Any]) -> Job:
    return Job(
        job_id=generate_job_id(),
        project_id="proj-a",
        user_id="alice",
        job_type=job_type,
        spec=spec,
        idempotency_key="",
    )


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, fetchval_result=None):
        self.fetchval_result = fetchval_result
        self.fetchval_calls: list[tuple] = []
        self.executes: list[tuple] = []

    async def fetchval(self, *args):
        self.fetchval_calls.append(args)
        return self.fetchval_result

    async def execute(self, *args):
        self.executes.append(args)
        return "INSERT 0 1"


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


class _TxConn:
    """One connection of the embed writer's transaction: the artifact row and
    the writer's own file node exist; deletes and the stamp remove nothing."""

    async def fetchval(self, *_args):
        return 1

    async def execute(self, *_args):
        return "DELETE 0"

    @asynccontextmanager
    async def transaction(self):
        yield self


class _TxPool:
    @asynccontextmanager
    async def acquire(self):
        yield _TxConn()


class _CapturingStore:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.embeddings = []
        self._pool = _TxPool()

    async def upsert_node(self, node, **_k):
        self.nodes.append(node)

    async def upsert_edge(self, edge, **_k):
        self.edges.append(edge)

    async def upsert_node_embedding(self, node_id, model_name, embedding, *, content_hash="", **_k):
        self.embeddings.append(
            {
                "node_id": node_id,
                "model_name": model_name,
                "embedding": embedding,
                "content_hash": content_hash,
            }
        )


class _FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


# ── B7: status-checker honesty ──────────────────────────────────────────────


class TestRefCounts:
    def test_parses_counts(self):
        assert (
            _knowledge_ref_counts("knowledge_ingest|project=p|artifacts=3|failed=1", "artifacts")
            == 3
        )
        assert (
            _knowledge_ref_counts("knowledge_ingest|project=p|artifacts=3|failed=1", "failed") == 1
        )
        assert _knowledge_ref_counts("knowledge_ingest|mode=degraded|project=p", "failed") == 0

    def test_non_numeric_is_zero(self):
        assert _knowledge_ref_counts("knowledge_ingest|mode=degraded|reason=s3_down", "reason") == 0


class TestIngestStatusHonesty:
    @pytest.mark.asyncio
    async def test_all_failed_sources_is_failure(self):
        checker = KnowledgeIngestStatusChecker()
        job = _job(JobType.KNOWLEDGE_INGEST, {})
        ref = "knowledge_ingest|project=p|artifacts=0|failed=2"
        result = await checker.check_status(_attempt(ref), job)
        assert result.status == "failed"
        assert "2 source(s) failed" in (result.exit_reason or "")

    @pytest.mark.asyncio
    async def test_partial_success_stays_success(self):
        checker = KnowledgeIngestStatusChecker()
        job = _job(JobType.KNOWLEDGE_INGEST, {})
        ref = "knowledge_ingest|project=p|artifacts=1|failed=1"
        result = await checker.check_status(_attempt(ref), job)
        assert result.status == "succeeded"


class TestUnderstandStatusHonesty:
    @pytest.mark.asyncio
    async def test_all_failed_is_failure(self):
        checker = KnowledgeUnderstandStatusChecker()
        job = _job(JobType.KNOWLEDGE_UNDERSTAND, {})
        ref = "knowledge_understand|project=p|ir_ids=0|already=0|fail=3"
        result = await checker.check_status(_attempt(ref), job)
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_idempotent_noop_is_success(self):
        checker = KnowledgeUnderstandStatusChecker()
        job = _job(JobType.KNOWLEDGE_UNDERSTAND, {})
        ref = "knowledge_understand|project=p|ir_ids=0|already=2|fail=0"
        result = await checker.check_status(_attempt(ref), job)
        assert result.status == "succeeded"


class TestEmbedStatusChecker:
    @pytest.mark.asyncio
    async def test_no_embedder_is_a_failure(self):
        """Maintainer decision 2026-10-09: the image bundles a text embedder, so
        lacking one fails (it was a documented no-op before)."""
        checker = KnowledgeEmbedStatusChecker()
        job = _job(JobType.KNOWLEDGE_EMBED, {})
        ref = "knowledge_embed|mode=degraded|reason=no_embedder|project=p"
        result = await checker.check_status(_attempt(ref), job)
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_all_failed_is_failure(self):
        checker = KnowledgeEmbedStatusChecker()
        job = _job(JobType.KNOWLEDGE_EMBED, {})
        ref = "knowledge_embed|project=p|embedded=0|skipped=0|failed=1"
        result = await checker.check_status(_attempt(ref), job)
        assert result.status == "failed"


# ── B3: KnowledgeEmbedBackend ────────────────────────────────────────────────


class TestEmbedJobType:
    def test_job_type_registered_with_backoff(self):
        assert JobType.KNOWLEDGE_EMBED.value == "knowledge_embed"
        assert DEFAULT_BACKOFFS[JobType.KNOWLEDGE_EMBED]["max_attempts"] == 3

    def test_backend_name(self):
        assert KnowledgeEmbedBackend().backend_name == "knowledge_embed"


class TestEmbedDegradedWhenNoEmbedder:
    @pytest.mark.asyncio
    async def test_degraded_no_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("UMRS_KNOWLEDGE_EMBED_MODEL", raising=False)
        from umrs.jobs import knowledge_adapters as ka

        ka._reset_node_embedder_for_tests()
        job = _job(
            JobType.KNOWLEDGE_EMBED,
            {
                "project_id": "proj-a",
                "sources": [{"artifact_id": "art-1", "content_hash": "h", "source_ref": "k"}],
            },
        )
        attempt = _attempt("")
        ref = await KnowledgeEmbedBackend().submit_job(job, attempt)
        assert ref.startswith("knowledge_embed|mode=degraded")
        result = json.loads(attempt.metrics["result"])
        assert result["degraded"] is True
        assert "UMRS_KNOWLEDGE_EMBED_MODEL" in result["degraded_reason"]

    @pytest.mark.asyncio
    async def test_non_staged_model_path_fails_closed(self, monkeypatch):
        monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "intfloat/multilingual-e5-large")
        from umrs.jobs import knowledge_adapters as ka

        ka._reset_node_embedder_for_tests()
        assert ka._get_node_embedder() is None
        job = _job(
            JobType.KNOWLEDGE_EMBED,
            {"project_id": "p", "sources": [{"artifact_id": "a", "content_hash": "h"}]},
        )
        attempt = _attempt("")
        ref = await KnowledgeEmbedBackend().submit_job(job, attempt)
        assert "mode=degraded" in ref


class TestEmbedOneArtifact:
    @pytest.mark.asyncio
    async def test_chunk_nodes_edges_vectors_written(self, monkeypatch, tmp_path):
        from umrs.jobs import knowledge_adapters as ka

        text = "The catalogue is the physical-existence authority. " * 40
        monkeypatch.setattr(ka, "_fetch_artifact_bytes", lambda *_a, **_k: text.encode("utf-8"))
        model_dir = str(tmp_path / "fake-model")
        monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", model_dir)
        store = _CapturingStore()
        embedder = _FakeEmbedder()
        result = KnowledgeEmbedResult()
        await ka._embed_one_artifact(
            store,
            {"artifact_id": "art-x", "content_hash": "hh", "source_ref": "s3key"},
            "proj-a",
            "g-default-x",
            InProcessChunkEmbedder(embedder, prefix_model_name=model_dir, max_tokens=256),
            MagicMock(),
            "bucket",
            result,
        )
        assert result.nodes_embedded == len(store.nodes) >= 1
        assert result.nodes_embedded == len(store.embeddings)
        # chunk nodes link part_of the file node (named-graph scope prefix)
        file_node = "node-file-g-default-x-art-x"
        assert store.edges and all(e.relation_type == "part_of" for e in store.edges)
        assert all(e.target_id == file_node for e in store.edges)
        assert all(n.node_id.startswith("node-chunk-g-default-x-art-x-") for n in store.nodes)
        # vectors carry the model + chunk content hash
        assert all(e["model_name"] == model_dir for e in store.embeddings)
        assert all(e["content_hash"] for e in store.embeddings)
        # chunk content excerpt for retrieval display
        assert all(n.content.get("excerpt") for n in store.nodes)

    @pytest.mark.asyncio
    async def test_missing_content_fails_artifact(self, monkeypatch):
        from umrs.jobs import knowledge_adapters as ka

        monkeypatch.setattr(ka, "_fetch_artifact_bytes", lambda *_a, **_k: None)
        store = _CapturingStore()
        result = KnowledgeEmbedResult()
        await ka._embed_one_artifact(
            store,
            {"artifact_id": "art-y", "content_hash": "hh", "source_ref": "s3key"},
            "proj-a",
            "",
            InProcessChunkEmbedder(_FakeEmbedder(), prefix_model_name="", max_tokens=256),
            MagicMock(),
            "bucket",
            result,
        )
        assert result.nodes_embedded == 0
        assert result.failed_items and result.failed_items[0]["artifact_id"] == "art-y"


class TestEmbedChainEnqueue:
    def test_enqueue_embed_job_builds_spec(self):
        from umrs.jobs import knowledge_adapters as ka

        parent = _job(
            JobType.KNOWLEDGE_UNDERSTAND,
            {"project_id": "proj-a", "graph_id": "g-1"},
        )
        captured: dict[str, Any] = {}

        class _Svc:
            def submit_job(self, submission):
                captured["submission"] = submission
                return "job-embed-1"

        with (
            patch.object(ka, "_enqueue_embed_job", wraps=ka._enqueue_embed_job),
            patch("umrs.jobs.service.get_job_service", return_value=_Svc()),
        ):
            out = ka._enqueue_embed_job("proj-a", "alice", parent, [{"artifact_id": "a1"}])
        assert out == "job-embed-1"
        sub = captured["submission"]
        assert sub.job_type is JobType.KNOWLEDGE_EMBED
        assert sub.spec["graph_id"] == "g-1"
        assert sub.spec["parent_job_id"] == parent.job_id


# ── B8: per-project dedup probe ──────────────────────────────────────────────


class TestProjectScopedDedup:
    @pytest.mark.asyncio
    async def test_probe_filters_by_project(self):
        from umrs.knowledge.store import KnowledgeStore

        conn = _FakeConn(fetchval_result="art-existing")
        store = KnowledgeStore(_FakePool(conn))
        inserted = await store.insert_artifact(
            artifact_id="art-new",
            content_hash="deadbeef",
            kind="document",
            project_id="proj-b",
        )
        assert inserted == "art-existing"  # the stored id, never a rowless fresh one
        (sql, *params) = conn.fetchval_calls[0]
        assert "project_id = $2" in sql
        assert params == ["deadbeef", "proj-b"]

    @pytest.mark.asyncio
    async def test_insert_new_hash_other_project(self):
        from umrs.knowledge.store import KnowledgeStore

        conn = _FakeConn(fetchval_result=None)
        store = KnowledgeStore(_FakePool(conn))
        inserted = await store.insert_artifact(
            artifact_id="art-new",
            content_hash="deadbeef",
            kind="document",
            project_id="proj-b",
        )
        assert inserted == "art-new"

    def test_migration_0076_swaps_unique_scope(self):
        mig = (
            Path(__file__).parent.parent
            / "src/umrs/migrations/0076_knowledge_artifacts_project_dedup.sql"
        )
        text = mig.read_text()
        assert "DROP CONSTRAINT IF EXISTS knowledge_artifacts_content_hash_key" in text
        assert "(project_id, content_hash)" in text


# ── B1/B2: ingest route guard + spec graph_id ────────────────────────────────


class _FakeRuntime:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def submit_job_sync(self, **kwargs):
        self.calls.append(kwargs)
        return "job-captured-1"


def _build_ingest_app(verified_project: str = "verified-proj"):
    from fastapi import FastAPI
    from umrs.api.files import get_project_id
    from umrs.api.knowledge_v2 import router

    app = FastAPI()
    app.include_router(router)  # router already carries prefix="/umrs/k2"
    app.dependency_overrides[get_project_id] = lambda: verified_project
    return app


class TestIngestRouteProjectGuard:
    def test_mismatch_rejected_409(self):
        app = _build_ingest_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post(
            "/umrs/k2/ingest",
            json={
                "project_id": "someone-else",
                "sources": [{"kind": "inline_text", "uri": "a.md", "content": "x"}],
            },
        )
        assert resp.status_code == 409, f"unexpected: {resp.status_code} {resp.text[:300]}"
        detail = resp.json()["detail"]
        assert detail["error"] == "project_mismatch"
        assert detail["verified_project_id"] == "verified-proj"

    def test_spec_carries_resolved_graph_id(self):
        app = _build_ingest_app()
        from fastapi.testclient import TestClient

        fake_runtime = _FakeRuntime()
        with (
            patch("umrs.server.durable_job_runtime.get_runtime", return_value=fake_runtime),
            patch("umrs.api.governance.guard_ingest", return_value=None),
        ):
            client = TestClient(app)
            resp = client.post(
                "/umrs/k2/ingest",
                json={
                    "sources": [{"kind": "inline_text", "uri": "a.md", "content": "hello graph"}],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-captured-1"
        (call,) = fake_runtime.calls
        assert call["spec"]["project_id"] == "verified-proj"
        graph_id = call["spec"]["graph_id"]
        assert isinstance(graph_id, str) and graph_id, "spec must always carry a graph id"
        assert graph_id != ""

    def test_matching_override_accepted(self):
        app = _build_ingest_app()
        from fastapi.testclient import TestClient

        fake_runtime = _FakeRuntime()
        with (
            patch("umrs.server.durable_job_runtime.get_runtime", return_value=fake_runtime),
            patch("umrs.api.governance.guard_ingest", return_value=None),
        ):
            client = TestClient(app)
            resp = client.post(
                "/umrs/k2/ingest",
                json={
                    "project_id": "verified-proj",
                    "sources": [{"kind": "inline_text", "uri": "a.md", "content": "x"}],
                },
            )
        assert resp.status_code == 200


# ── B12: strict defaults everywhere ─────────────────────────────────────────


class TestAuthModeDefaults:
    def test_no_mvp_defaults_left_in_api(self):
        api_dir = Path(__file__).parent.parent / "src/umrs/api"
        simple = re.compile(r'get\("UMRS_AUTH_MODE",\s*"mvp"\)')
        offenders = []
        for path in api_dir.rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if simple.search(line):
                    offenders.append(f"{path}:{i}")
        assert not offenders, f"mvp defaults remain: {offenders}"

    def test_scope_identity_default_is_strict(self, monkeypatch):
        monkeypatch.delenv("UMRS_AUTH_MODE", raising=False)
        from umrs.api import _scope_identity

        assert _scope_identity._strict_mode() is True

    def test_identity_default_is_strict(self, monkeypatch):
        monkeypatch.delenv("UMRS_AUTH_MODE", raising=False)
        from umrs.api import _identity

        assert _identity._strict_mode() is True


@pytest.fixture(autouse=True)
def _restore_embedder():
    from umrs.jobs import knowledge_adapters as ka

    ka._reset_node_embedder_for_tests()
    yield
    ka._reset_node_embedder_for_tests()
