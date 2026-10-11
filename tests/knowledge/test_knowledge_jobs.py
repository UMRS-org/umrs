"""Tests for KNOWLEDGE_INGEST / KNOWLEDGE_UNDERSTAND job backends.

Covers:
  - Crash-resume and idempotency on compose harness
  - Degraded mode when S3 unavailable
  - INGEST: partial failures tracked
  - UNDERSTAND: artifacts without IRs processed, already-IR skipped
"""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import MagicMock, patch

import pytest

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = None  # type: ignore[misc,assignment]

from umrs.jobs.knowledge_adapters import (
    KnowledgeEvent,
    KnowledgeIngestBackend,
    KnowledgeIngestResult,
    KnowledgeUnderstandBackend,
    KnowledgeUnderstandResult,
    _build_summary,
    _parse_for_understand,
    _with_excerpt,
)
from umrs.jobs.models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobType,
    compute_idempotency_key,
    generate_attempt_id,
    generate_job_id,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_job() -> Job:
    return Job(
        job_id=generate_job_id(),
        project_id="test-project",
        user_id="test-user",
        job_type=JobType.KNOWLEDGE_INGEST,
        spec={
            "sources": [
                {"kind": "local_file", "uri": "/tmp/test.py"},
            ],
        },
        idempotency_key=compute_idempotency_key(
            "test-project", "test-user", JobType.KNOWLEDGE_INGEST, {}
        ),
    )


@pytest.fixture
def sample_attempt() -> JobAttempt:
    return JobAttempt(
        attempt_id=generate_attempt_id(),
        job_id="job-001",
        attempt_no=1,
        backend="compose",
        status=AttemptStatus.CREATED,
        backend_ref="",
        metrics={},
    )


@pytest.fixture
def tmp_py_file(tmp_path) -> str:
    path = tmp_path / "test_example.py"
    path.write_text(
        '"""Example module."""\n\nclass MyClass:\n'
        '    """A test class."""\n\n'
        "    def my_method(self):\n"
        '        """A method."""\n        pass\n\n'
        "def main():\n    pass\n\n"
        "if __name__ == '__main__':\n    main()\n",
    )
    return str(path)


@pytest.fixture
def tmp_dir_with_files(tmp_path) -> str:
    """Create a temp dir with multiple files."""
    for name, content in [
        ("file1.py", "def func1(): pass\n"),
        ("file2.py", "class Class2:\n    pass\n"),
        ("file3.cpp", "int main() { return 0; }\n"),
    ]:
        (tmp_path / name).write_text(content)
    return str(tmp_path)


class _FakeClientError(Exception):
    """Stand-in for botocore ClientError (works without botocore)."""

    def __init__(self, response: dict, operation_name: str):
        super().__init__(f"{operation_name}: {response}")
        self.response = response


def _make_fake_s3() -> MagicMock:
    """Fake S3 client matching real boto3 semantics.

    ``head_object`` raises ``ClientError`` (404) for missing keys —
    real boto3 never returns ``None``.
    """
    client = MagicMock()
    client.list_buckets.return_value = {"Buckets": []}
    client.put_object.return_value = {"ETag": '"abc123"'}
    client.head_object.side_effect = _FakeClientError({"Error": {"Code": "404"}}, "HeadObject")
    client.exceptions = MagicMock(ClientError=_FakeClientError)
    return client


@pytest.fixture
def fake_s3_client() -> MagicMock:
    """Mock S3 client that accepts puts."""
    return _make_fake_s3()


@pytest.fixture
def understand_job() -> Job:
    return Job(
        job_id=generate_job_id(),
        project_id="test-project",
        user_id="test-user",
        job_type=JobType.KNOWLEDGE_UNDERSTAND,
        spec={
            "project_id": "test-project",
            "sources": [
                {
                    "artifact_id": "art-existing",
                    "content_hash": "abc123hash",
                    "kind": "code",
                    "source_ref": "/tmp/test.py",
                },
            ],
        },
        idempotency_key=compute_idempotency_key(
            "test-project", "test-user", JobType.KNOWLEDGE_UNDERSTAND, {}
        ),
    )


# ── KnowledgeEvent Tests ────────────────────────────────────────────────────


class TestKnowledgeEvent:
    """Test KnowledgeEvent emission."""

    def test_emit_logs(self):
        event = KnowledgeEvent(
            event_type="create",
            job_id="job-001",
            attempt_id="att-001",
            details="test",
        )
        result = event.emit()
        assert result.event_type == "create"
        assert result.timestamp != ""

    def test_emit_with_artifact(self):
        event = KnowledgeEvent(
            event_type="ingest",
            job_id="job-001",
            attempt_id="att-001",
            artifact_id="art-001",
            ir_id="ir-001",
            source_ref="/tmp/test.py",
            details="ingested",
        )
        result = event.emit()
        assert result.artifact_id == "art-001"
        assert result.ir_id == "ir-001"

    def test_chaining(self):
        e1 = KnowledgeEvent(event_type="create", job_id="j", attempt_id="a").emit()
        e2 = e1  # emit returns self
        assert e2.job_id == "j"


# ── KnowledgeIngestBackend Tests ─────────────────────────────────────────────


class TestKnowledgeIngestBackend:
    """Tests for knowledge ingestion backend."""

    def test_backend_name(self):
        backend = KnowledgeIngestBackend()
        assert backend.backend_name == "knowledge_ingest"

    def test_init_no_s3(self):
        """Backend initializes without S3 client."""
        backend = KnowledgeIngestBackend()
        assert backend._s3_client is None

    def test_init_s3_success(self, fake_s3_client):
        """Backend initializes S3 when available."""
        with patch(
            "boto3.client",
            return_value=fake_s3_client,
        ):
            backend = KnowledgeIngestBackend()
            backend._init_s3({"bucket": "test-bucket"})
            assert backend._s3_client is fake_s3_client

    def test_init_s3_fails(self):
        """Backend degrades gracefully when S3 unavailable."""
        with patch(
            "boto3.client",
            side_effect=RuntimeError("no network"),
        ):
            backend = KnowledgeIngestBackend()
            backend._init_s3({"bucket": "test-bucket"})
            assert backend._s3_client is None

    def test_init_s3_bucket_env_fallback(self, monkeypatch):
        """No spec bucket must fall back to UMRS_DATA_S3_BUCKET/S3_BUCKET env.

        (Regression: the old hardcoded 'umrs-data' default failed with
        NoSuchBucket on live deployments where only r4-data exists.)
        """
        fake = _make_fake_s3()
        monkeypatch.setenv("S3_BUCKET", "r4-data")
        monkeypatch.delenv("UMRS_DATA_S3_BUCKET", raising=False)
        with patch("boto3.client", return_value=fake):
            backend = KnowledgeIngestBackend()
            backend._init_s3({})
            assert backend._bucket == "r4-data"

    def test_init_s3_explicit_spec_bucket_wins(self, monkeypatch):
        """Explicit spec bucket overrides env."""
        fake = _make_fake_s3()
        monkeypatch.setenv("S3_BUCKET", "r4-data")
        with patch("boto3.client", return_value=fake):
            backend = KnowledgeIngestBackend()
            backend._init_s3({"bucket": "explicit-bucket"})
            assert backend._bucket == "explicit-bucket"

    @pytest.mark.asyncio
    async def test_submit_requires_sources(self, sample_job, sample_attempt):
        """Jobs with empty sources list must fail."""
        sample_job.spec["sources"] = []
        backend = KnowledgeIngestBackend()
        with pytest.raises(ValueError, match="requires spec.sources"):
            await backend.submit_job(sample_job, sample_attempt)

    @pytest.mark.asyncio
    async def test_degraded_no_s3(self, sample_job, monkeypatch):
        """When S3 unavailable, job records failed items."""
        # Premise pin (configured-env wave, 2026-09-16): "unavailable" must be
        # explicit — the legacy premise read ambient env, so a live object
        # store on localhost (CI's SeaweedFS lane) silently flipped the
        # degraded branch off. Point at a guaranteed-closed port.
        monkeypatch.setenv("S3_ENDPOINT_URL", "http://127.0.0.1:59999")
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        sample_job.spec["bucket"] = "test-bucket"
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )

        backend = KnowledgeIngestBackend()
        result = await backend.submit_job(sample_job, sample_attempt)

        assert result.startswith("knowledge_ingest|")
        assert "degraded" in result
        metadata = json.loads(sample_attempt.metrics["result"])
        assert metadata["degraded"] is True
        assert len(metadata["failed_items"]) == 1

    @pytest.mark.asyncio
    async def test_ingest_local_file(self, sample_job, tmp_py_file):
        """Ingest a single local Python file."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()

        with patch(
            "boto3.client",
            return_value=fake_s3,
        ):
            backend = KnowledgeIngestBackend()
            result = await backend.submit_job(sample_job, sample_attempt)

        assert "artifacts=1" in result
        metadata = json.loads(sample_attempt.metrics["result"])
        assert len(metadata["artifact_ids"]) == 1
        assert len(metadata["failed_items"]) == 0

    @pytest.mark.asyncio
    async def test_submit_chains_understand(self, sample_job, tmp_py_file):
        """API-triggered ingest enqueues KNOWLEDGE_UNDERSTAND (graph_id propagate)."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"
        sample_job.spec["graph_id"] = "g-named-1"

        fake_s3 = _make_fake_s3()

        with (
            patch("boto3.client", return_value=fake_s3),
            patch(
                "umrs.jobs.knowledge_adapters._enqueue_understand_job",
                return_value="job-understand-1",
            ) as chain,
        ):
            backend = KnowledgeIngestBackend()
            await backend.submit_job(sample_job, sample_attempt)

        chain.assert_called_once()
        project_id, user_id, parent_job = chain.call_args.args
        assert project_id == sample_job.project_id
        assert user_id == sample_job.user_id
        assert parent_job is sample_job
        assert parent_job.spec["graph_id"] == "g-named-1"

    @pytest.mark.asyncio
    async def test_chain_gets_acquisition_descriptors_without_db(self, sample_job, tmp_py_file):
        """Regression: chained understand receives desc inline (DB listing race).

        Production evidence: the artifact insert is fire-and-forget, so the
        enqueue-time re-listing raced ahead of the commit and understand jobs
        ran with stale/empty source lists (ir_ids=0 on every chain). The chain
        must receive sources=... inline, with no dependency on the DB listing.
        """
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()

        with (
            patch("boto3.client", return_value=fake_s3),
            patch(
                "umrs.jobs.knowledge_adapters._list_project_artifact_sources",
                return_value=[],  # simulate the racing-empty DB listing
            ),
            patch(
                "umrs.jobs.knowledge_adapters._enqueue_understand_job",
                return_value="job-understand-1",
            ) as chain,
        ):
            backend = KnowledgeIngestBackend()
            await backend.submit_job(sample_job, sample_attempt)

        chain.assert_called_once()
        sources = chain.call_args.kwargs["sources"]
        assert len(sources) == 1
        desc = sources[0]
        assert desc["content_hash"]
        assert desc["kind"] == "code"
        assert desc["source_ref"].startswith("projects/test-project/knowledge/artifacts/")

    @pytest.mark.asyncio
    async def test_submit_chain_failure_is_best_effort(self, sample_job, tmp_py_file):
        """A broken understand-chain enqueue must not fail the acquisition attempt."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()

        with (
            patch("boto3.client", return_value=fake_s3),
            patch(
                "umrs.jobs.knowledge_adapters._enqueue_understand_job",
                side_effect=RuntimeError("db down"),
            ),
        ):
            backend = KnowledgeIngestBackend()
            result = await backend.submit_job(sample_job, sample_attempt)

        assert "artifacts=1" in result

    @pytest.mark.asyncio
    async def test_artifact_persist_runs_sync_before_chain(self, sample_job, tmp_py_file):
        """FK guard: artifact rows commit before the understand job is enqueued.

        knowledge_ir.artifact_id has an FK to knowledge_artifacts. The old
        fire-and-forget persist raced the chain enqueue — FK violations on
        every live chain, masked as duplicate IR inserts.
        """
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()
        events: list[str] = []

        def _mark_persist(coro, label: str = ""):
            events.append(label or "krc")
            return None

        def _mark_enqueue(*_args, **_kwargs):
            events.append("enqueue-understand")
            return "job-understand-1"

        with (
            patch("boto3.client", return_value=fake_s3),
            patch(
                "umrs.jobs.knowledge_adapters._resolve_knowledge_dsn",
                return_value="postgresql://unit:test@127.0.0.1/umrs_test",
            ),
            patch(
                "umrs.jobs.knowledge_adapters._run_coroutine_sync",
                side_effect=_mark_persist,
            ),
            patch(
                "umrs.jobs.knowledge_adapters._enqueue_understand_job",
                side_effect=_mark_enqueue,
            ),
        ):
            backend = KnowledgeIngestBackend()
            await backend.submit_job(sample_job, sample_attempt)

        assert events.index("persist_artifact_meta") < events.index("enqueue-understand")

    @pytest.mark.asyncio
    async def test_ingest_registers_full_artifact_metadata(self, sample_job, tmp_py_file):
        """Knowledge_artifacts rows must carry hash + S3-key source_ref + kind.

        The chained KNOWLEDGE_UNDERSTAND job fetches bytes via
        ArtifactStorage.key-style source_ref — ids alone get skipped.
        """
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()
        captured: list[tuple[str, list]] = []

        def _capture_store(project_id: str, descs: list[dict]) -> object:
            from umrs.jobs.knowledge_adapters import _StoredArtifacts

            captured.append((project_id, descs))
            return _StoredArtifacts()

        with (
            patch("boto3.client", return_value=fake_s3),
            patch(
                "umrs.jobs.knowledge_adapters._store_artifacts_in_db_sync",
                side_effect=_capture_store,
            ),
        ):
            backend = KnowledgeIngestBackend()
            result = await backend.submit_job(sample_job, sample_attempt)

        assert "artifacts=1" in result
        project_id, descs = captured[0]
        assert project_id == sample_job.project_id
        assert len(descs) == 1
        desc = descs[0]
        assert desc["content_hash"]
        assert desc["kind"] == "code"  # .py suffix heuristic
        expected_key = (
            f"projects/{sample_job.project_id}/knowledge/artifacts/{desc['content_hash']}"
        )
        assert desc["source_ref"] == expected_key
        assert desc["artifact_id"].startswith("art-")

    @pytest.mark.asyncio
    async def test_a_reupload_continues_on_the_stored_artifact(self, sample_job, tmp_py_file):
        """Bytes the project already holds: the result, the provenance and the
        chained understand all name the stored artifact, not a fresh id with no
        row (SL-CONF T2, 2026-10-09: understand "completed" with every node
        failing its FK, and embed failed)."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"
        chained: list[list[dict]] = []

        def _known_bytes(_project_id: str, descs: list[dict]) -> object:
            from umrs.jobs.knowledge_adapters import _StoredArtifacts

            fresh = descs[0]["artifact_id"]
            descs[0]["artifact_id"] = "art-stored"
            return _StoredArtifacts(deduped={fresh: "art-stored"})

        with (
            patch("boto3.client", return_value=_make_fake_s3()),
            patch(
                "umrs.jobs.knowledge_adapters._store_artifacts_in_db_sync",
                side_effect=_known_bytes,
            ),
            patch(
                "umrs.jobs.knowledge_adapters._enqueue_understand_job",
                side_effect=lambda *_a, sources, **_k: chained.append(sources),
            ),
        ):
            await KnowledgeIngestBackend().submit_job(sample_job, sample_attempt)

        result = json.loads(sample_attempt.metrics["result"])
        assert result["artifact_ids"] == ["art-stored"]
        assert [a["artifact_id"] for a in sample_attempt.metrics["artifacts"]] == ["art-stored"]
        assert [d["artifact_id"] for d in chained[0]] == ["art-stored"]

    @pytest.mark.asyncio
    async def test_an_artifact_whose_row_was_not_written_leaves_the_chain(
        self, sample_job, tmp_py_file
    ):
        """Wave F (ADR-093 D-11): a descriptor whose row insert failed is named
        in failed_items and never reaches the chained understand; before, the
        failure was swallowed and the rowless id went down the chain."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_file", "uri": tmp_py_file}]
        sample_job.spec["bucket"] = "test-bucket"
        chained: list[list[dict]] = []

        def _row_lost(_project_id: str, descs: list[dict]) -> object:
            from umrs.jobs.knowledge_adapters import _StoredArtifacts

            lost = {"artifact_id": descs[0]["artifact_id"], "error": "artifact row not written: x"}
            return _StoredArtifacts(failed=[lost])

        with (
            patch("boto3.client", return_value=_make_fake_s3()),
            patch(
                "umrs.jobs.knowledge_adapters._store_artifacts_in_db_sync",
                side_effect=_row_lost,
            ),
            patch(
                "umrs.jobs.knowledge_adapters._enqueue_understand_job",
                side_effect=lambda *_a, sources, **_k: chained.append(sources),
            ),
        ):
            ref = await KnowledgeIngestBackend().submit_job(sample_job, sample_attempt)

        result = json.loads(sample_attempt.metrics["result"])
        assert result["artifact_ids"] == []
        assert "artifact row not written" in result["failed_items"][0]["error"]
        assert chained == [[]]  # _enqueue_understand_job enqueues nothing for []
        assert "artifacts=0" in ref and "failed=1" in ref

    @pytest.mark.asyncio
    async def test_ingest_local_dir(self, sample_job, tmp_dir_with_files):
        """Ingest a directory of files."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [{"kind": "local_dir", "uri": tmp_dir_with_files}]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()

        with patch(
            "boto3.client",
            return_value=fake_s3,
        ):
            backend = KnowledgeIngestBackend()
            result = await backend.submit_job(sample_job, sample_attempt)

        assert "artifacts=3" in result

    @pytest.mark.asyncio
    async def test_partial_failures(self, sample_job, tmp_py_file):
        """Some sources fail, others succeed — tracked separately."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [
            {"kind": "local_file", "uri": tmp_py_file},  # valid
            {"kind": "local_file", "uri": "/nonexistent/file.py"},  # invalid
            {"kind": "s3_uri", "uri": "s3://bucket/key"},  # valid (readable object)
        ]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()
        # s3_uri sources are read once and copied content-addressed (2026-10-09),
        # so the referenced object must exist to count as a success.
        fake_s3.get_object.return_value = {"Body": io.BytesIO(b"staged upload")}

        with patch(
            "boto3.client",
            return_value=fake_s3,
        ):
            backend = KnowledgeIngestBackend()
            await backend.submit_job(sample_job, sample_attempt)

        metadata = json.loads(sample_attempt.metrics["result"])
        # 2 artifacts (local_file succeeded + s3 object copied)
        assert len(metadata["artifact_ids"]) == 2
        # 1 failure (nonexistent file)
        assert len(metadata["failed_items"]) == 1

    @pytest.mark.asyncio
    async def test_egress_blocked(self, sample_job):
        """Remote sources (http_url, git_repo) without egress are blocked."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=sample_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        sample_job.spec["sources"] = [
            {"kind": "http_url", "uri": "https://example.com/file.py"},
        ]
        sample_job.spec["bucket"] = "test-bucket"

        fake_s3 = _make_fake_s3()

        with patch(
            "boto3.client",
            return_value=fake_s3,
        ):
            backend = KnowledgeIngestBackend()
            await backend.submit_job(sample_job, sample_attempt)

        metadata = json.loads(sample_attempt.metrics["result"])
        assert len(metadata["failed_items"]) == 1
        assert "egress" in metadata["failed_items"][0]["error"].lower()


# ── KnowledgeUnderstandBackend Tests ─────────────────────────────────────────


class TestKnowledgeUnderstandBackend:
    """Tests for knowledge understanding backend."""

    def test_backend_name(self):
        backend = KnowledgeUnderstandBackend()
        assert backend.backend_name == "knowledge_understand"

    @pytest.mark.asyncio
    async def test_parse_for_understand_python(self):
        """Python source parsed correctly from real content bytes."""
        real_content = b'"""Example."""\nclass Foo:\n    pass\ndef bar(): pass\n'

        result = _parse_for_understand("test-001", real_content, "code")
        assert result is not None
        assert "modules" in result
        assert "classes" in result
        assert "functions" in result

    @pytest.mark.asyncio
    async def test_parse_for_understand_cpp(self):
        """C++ source parsed with regex fallback from real content bytes."""
        real_content = b"int main() { return 0; }\nvoid foo() {}\n"

        result = _parse_for_understand("test-cpp", real_content, "cpp")
        assert result is not None
        assert result["language"] == "cpp"

    @pytest.mark.asyncio
    async def test_parse_invalid(self):
        """Truly invalid bytes are parse-rejected (returns None)."""
        # Unparseable binary content must return None, never a fake IR.
        result = _parse_for_understand("bad", b"\x00\x01\xff\xfe\xfb", "code")
        assert result is None

    def test_build_summary_empty(self):
        """Empty parsed dict produces minimal summary."""
        summary = _build_summary({})
        assert summary == "no structure extracted"

    def test_build_summary_with_classes(self):
        """Summary includes class/function counts."""
        parsed = {
            "classes": ["A", "B"],
            "functions": ["f1", "f2", "f3"],
            "algorithms": ["algo1"],
        }
        summary = _build_summary(parsed)
        assert "2 class(es)" in summary
        assert "3 function(s)" in summary
        assert "1 algorithm(s)" in summary

    @pytest.mark.asyncio
    async def test_understand_no_pool(self, understand_job):
        """Without Postgres pool, job returns degraded result."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=understand_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        # Mock the real no-DSN control point: _open_pool yields None when
        # _resolve_knowledge_dsn() is empty. Patching the sync _get_pool had
        # no effect — async consumers resolve through _open_pool directly.
        with (
            patch(
                "umrs.jobs.knowledge_adapters._resolve_knowledge_dsn",
                return_value="",
            ),
            patch(
                "umrs.jobs.knowledge_adapters._build_s3_client",
                return_value=(None, "umrs-data"),
            ),
        ):
            backend = KnowledgeUnderstandBackend()
            await backend.submit_job(understand_job, sample_attempt)

        metadata = json.loads(sample_attempt.metrics["result"])
        assert metadata["degraded"] is True

    @pytest.mark.asyncio
    async def test_understand_with_pool_mock(self, understand_job):
        """With a mock pool, artifacts are processed or skipped."""
        mock_conn = MagicMock()
        mock_conn.fetchval.return_value = None  # No existing IR
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = MagicMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = MagicMock(return_value=False)
        mock_pool.fetchval.return_value = None

        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=understand_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )

        # Make wait_closed async-friendly
        mock_pool.wait_closed = MagicMock(return_value=asyncio.sleep(0))

        with (
            patch(
                "umrs.jobs.knowledge_adapters._build_s3_client",
                return_value=(None, "umrs-data"),
            ),
            patch("umrs.knowledge.store.KnowledgeStore") as MockStore,
        ):
            store_instance = MagicMock()
            MockStore.return_value = store_instance
            store_instance.list_artifacts.return_value = ([], 0)
            store_instance.insert_ir.return_value = True
            store_instance._pool = mock_pool

            backend = KnowledgeUnderstandBackend()
            result = await backend.submit_job(understand_job, sample_attempt)

        # At least should not crash
        assert "project=" in result


# ── Crash-Resume & Idempotency Tests ────────────────────────────────────────


class TestCrashResumeIdempotency:
    """Tests for crash-resume and write-once idempotency."""

    def test_ingest_result_serializable(self):
        """Ingest result can be JSON-serialized for metadata."""
        result = KnowledgeIngestResult(
            artifact_ids=["art-001", "art-002"],
            failed_items=[{"file": "/bad.py", "error": "not found"}],
            degraded=False,
        )
        data = json.dumps(
            {
                "artifact_ids": result.artifact_ids,
                "failed_items": result.failed_items,
                "degraded": result.degraded,
            }
        )
        parsed = json.loads(data)
        assert parsed["artifact_ids"] == ["art-001", "art-002"]

    def test_understand_result_serializable(self):
        """Understand result can be JSON-serialized."""
        result = KnowledgeUnderstandResult(
            ir_ids=["ir-001", "ir-002"],
            nodes_created=5,
            edges_created=4,
            already_ir=2,
            failed_items=[],
            degraded=False,
        )
        data = json.dumps(
            {
                "ir_ids": result.ir_ids,
                "nodes_created": result.nodes_created,
                "edges_created": result.edges_created,
                "already_ir": result.already_ir,
                "failed_items": result.failed_items,
                "degraded": result.degraded,
            }
        )
        parsed = json.loads(data)
        assert parsed["already_ir"] == 2

    @pytest.mark.asyncio
    async def test_event_emit_does_not_crash(self):
        """Event emit must always succeed even without log handlers."""
        event = KnowledgeEvent(
            event_type="create",
            job_id="j",
            attempt_id="a",
            artifact_id="art-1",
            ir_id="ir-1",
            source_ref="/t.py",
            details="r",
        )
        result = event.emit()
        assert result.event_type == "create"

    @pytest.mark.asyncio
    async def test_idempotent_ir_store(self):
        """Multiple calls with same artifact produce same ir_id."""
        from umrs.knowledge.ir import build_knowledge_ir

        artifact_id = "test-art-001"
        content = b"class Foo:\n    pass\n"

        ir1 = build_knowledge_ir(
            project_id="proj",
            artifact_id=artifact_id,
            artifact_type="code",
            content=content,
            extracted={"modules": [], "classes": [], "functions": []},
            extractor_version="1.0.0",
        )
        ir2 = build_knowledge_ir(
            project_id="proj",
            artifact_id=artifact_id,
            artifact_type="code",
            content=content,
            extracted={"modules": [], "classes": [], "functions": []},
            extractor_version="1.0.0",
        )
        assert ir1.ir_id == ir2.ir_id


# ── Degraded Mode Tests ─────────────────────────────────────────────────────


class TestDegradedMode:
    """Tests for degraded mode behavior."""

    def test_ingest_degraded_records_reason(self):
        result = KnowledgeIngestResult(
            degraded=True,
            degraded_reason="S3 unavailable",
            failed_items=[{"source": "local_file", "error": "S3 unavailable"}],
        )
        assert result.degraded is True
        assert result.degraded_reason == "S3 unavailable"

    def test_understand_degraded_records_reason(self):
        result = KnowledgeUnderstandResult(
            degraded=True,
            degraded_reason="Postgres pool unavailable",
            already_ir=0,
        )
        assert result.degraded is True
        assert result.already_ir == 0

    @pytest.mark.asyncio
    async def test_understand_degraded_empty_result(self, understand_job):
        """Degraded understand returns empty IR ids."""
        sample_attempt = JobAttempt(
            attempt_id=generate_attempt_id(),
            job_id=understand_job.job_id,
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            backend_ref="",
            metrics={},
        )
        with patch(
            "umrs.jobs.knowledge_adapters._resolve_knowledge_dsn",
            return_value="",
        ):
            backend = KnowledgeUnderstandBackend()
            await backend.submit_job(understand_job, sample_attempt)

        metadata = json.loads(sample_attempt.metrics["result"])
        assert metadata["degraded"] is True
        assert metadata["ir_ids"] == []


class TestWithExcerpt:
    """_with_excerpt: bounded prose text lands on the file node only."""

    def _file_node(self):
        from umrs.knowledge.store import KnowledgeNode

        return KnowledgeNode(
            node_id="node-file-g-x-art-1",
            project_id="p",
            label="art-1",
            node_type="file",
            content={"artifact_id": "art-1", "ir_id": "ir-1"},
            content_hash="x",
            source_ir="ir-1",
            graph_id="g-x",
        )

    def test_prose_kind_attaches_excerpt(self):
        node = self._file_node()
        nodes = _with_excerpt([node], "document", b"# hello\n\nFACT: kgaccept-zz body.\n")
        assert "excerpt" in nodes[0].content
        assert "kgaccept-zz" in nodes[0].content["excerpt"]
        assert nodes[0].content["artifact_id"] == "art-1"

    def test_code_kind_keeps_metadata_only(self):
        node = self._file_node()
        nodes = _with_excerpt([node], "code", b"def f(): pass\n")
        assert "excerpt" not in nodes[0].content

    def test_excerpt_is_bounded(self):
        node = self._file_node()
        big = ("x" * 5000).encode()
        nodes = _with_excerpt([node], "document", big)
        assert len(nodes[0].content["excerpt"]) <= 2000

    def test_empty_nodes_list_safe(self):
        assert _with_excerpt([], "document", b"abc") == []

    def test_binary_content_decodes_lossy(self):
        node = self._file_node()
        nodes = _with_excerpt([node], "document", b"\xff\xfeabc")
        assert "excerpt" in nodes[0].content
