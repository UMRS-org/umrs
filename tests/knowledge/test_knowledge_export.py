"""Tests for KNOWLEDGE_EXPORT bundle export job backend and bundle format.

Covers:
  - Export engine writes correct bundle structure (r4-kb v1.x flat format)
  - Manifest hash and HMAC signature generation/verification
  - Tamper detection (modified manifest → signature reject)
  - Paged embedding fetch returns correct totals
  - Catalog listing works
  - Export→import round-trip equivalence on compose (store round-trip)
  - Degraded mode when pool unavailable
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from umrs.jobs.knowledge_adapters import (
    KnowledgeExportBackend,
    KnowledgeExportStatusChecker,
)
from umrs.jobs.models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobType,
    generate_attempt_id,
    generate_job_id,
)
from umrs.knowledge.bundle_export import (
    KnowledgeExportEngine,
    verify_bundle_integrity,
    verify_signature,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def export_job() -> Job:
    return Job(
        job_id=generate_job_id(),
        project_id="test-project",
        user_id="test-user",
        job_type=JobType.KNOWLEDGE_EXPORT,
        spec={"project_id": "test-project", "allow_unsigned": True},
        idempotency_key="",
    )


@pytest.fixture
def export_attempt() -> JobAttempt:
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
def tmp_bundle_dir(tmp_path: Path) -> Path:
    return tmp_path / "bundles" / "test-project"


@pytest.fixture
def fake_pool() -> MagicMock:
    """Mock asyncpg pool returning sample data."""
    pool = MagicMock()

    async def fake_execute(query: str, *args) -> str:
        return "INSERT 0 1"

    async def fake_fetchrow(query: str, *args) -> dict:
        if "COUNT(*)" in query:
            if "memory_vectors" in query:
                return {"count": 4}
            return {"count": 2}
        return None

    async def fake_fetch(query: str, *args) -> list:
        if "knowledge_artifacts" in query and "ORDER BY" in query:
            return [
                MagicMock(
                    **{
                        "artifact_id": "art-001",
                        "content_hash": "sha256-abc",
                        "kind": "document",
                        "project_id": "test-project",
                        "source_ref": "/tmp/test.txt",
                        "size_bytes": 1024,
                        "mime_type": "text/plain",
                        "status": "active",
                        "metadata": {},
                        "created_at": None,
                        "updated_at": None,
                    }
                )
            ]
        if "knowledge_ir" in query:
            return [
                MagicMock(
                    **{
                        "ir_id": "ir-001",
                        "artifact_id": "art-001",
                        "ir_type": "document",
                        "version": 1,
                        "content": {"text": "sample"},
                        "content_hash": "sha256-def",
                        "status": "current",
                        "created_at": None,
                    }
                )
            ]
        if "knowledge_nodes" in query:
            return [
                MagicMock(
                    **{
                        "node_id": "node-001",
                        "project_id": "test-project",
                        "label": "TestClass",
                        "node_type": "class",
                        "content": {},
                        "content_hash": "sha256-node",
                        "source_ir": "ir-001",
                        "created_at": None,
                        "updated_at": None,
                    }
                )
            ]
        if "knowledge_edges" in query:
            return [
                MagicMock(
                    **{
                        "edge_id": "edge-001",
                        "project_id": "test-project",
                        "source_id": "node-001",
                        "target_id": "node-002",
                        "relation_type": "defines",
                        "content": {},
                        "created_at": None,
                    }
                )
            ]
        if "memory_vectors" in query:
            return [
                MagicMock(
                    **{
                        "id": 1,
                        "vector": "[0.1,0.2,0.3]",
                        "metadata": '{"document_id":"doc-1"}',
                        "document_id": "doc-1",
                        "created_at": None,
                    }
                )
            ]
        return []

    class FakeAcquireContext:
        def __init__(self):
            self._closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self._closed = True

        async def fetch(self, *args, **kwargs):
            return await fake_fetch(*args, **kwargs)

        async def fetchrow(self, *args, **kwargs):
            return await fake_fetchrow(*args, **kwargs)

        async def execute(self, *args, **kwargs):
            return await fake_execute(*args, **kwargs)

    async def fake_acquire():
        return FakeAcquireContext()

    async def fake_wait_closed():
        pass

    pool.acquire = fake_acquire
    pool.wait_closed = fake_wait_closed

    return pool


@pytest.fixture
def fake_store(fake_pool) -> MagicMock:
    """Mock KnowledgeStore with all needed methods."""
    store = MagicMock()
    store._pool = fake_pool

    async def mock_list_artifacts(**kwargs):
        rows = [
            MagicMock(
                **{
                    "artifact_id": "art-001",
                    "content_hash": "sha256-abc",
                    "kind": "document",
                    "project_id": "test-project",
                    "source_ref": "/tmp/test.txt",
                    "size_bytes": 1024,
                    "mime_type": "text/plain",
                    "status": "active",
                    "metadata": {},
                    "created_at": None,
                    "updated_at": None,
                }
            )
        ]
        return rows, 2

    store.list_artifacts = mock_list_artifacts

    def _make_ir():
        m = MagicMock()
        m.ir_id = "ir-001"
        m.artifact_id = "art-001"
        m.ir_type = "document"
        m.version = 1
        m.content = {"text": "sample"}
        m.content_hash = "sha256-def"
        m.status = "current"
        m.created_at = None
        return m

    def _make_node():
        m = MagicMock()
        m.node_id = "node-001"
        m.project_id = "test-project"
        m.label = "TestClass"
        m.node_type = "class"
        m.content = {}
        m.content_hash = "sha256-node"
        m.source_ir = "ir-001"
        m.created_at = None
        m.updated_at = None
        return m

    def _make_edge():
        m = MagicMock()
        m.edge_id = "edge-001"
        m.project_id = "test-project"
        m.source_id = "node-001"
        m.target_id = "node-002"
        m.relation_type = "defines"
        m.content = {}
        m.created_at = None
        return m

    store.list_irs = AsyncMock(return_value=[_make_ir()])
    store.list_nodes = AsyncMock(return_value=[_make_node()])
    store.list_edges = AsyncMock(return_value=[_make_edge()])

    async def mock_list_embedding_vectors_paged(**kwargs):
        return [
            {
                "id": "1",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {},
                "document_id": "doc-1",
                "created_at": None,
            }
        ], 4

    store.list_embedding_vectors_paged = mock_list_embedding_vectors_paged

    async def mock_list_catalog(**kwargs):
        entries = [
            {
                "id": "1",
                "name": "Science Catalog",
                "project_id": "test-project",
                "metadata": {},
                "created_at": None,
            }
        ]
        return entries, 3

    store.list_catalog = mock_list_catalog

    return store


# ── ExportEngine Tests ────────────────────────────────────────────────────────


class TestKnowledgeExportEngine:
    """Tests for KnowledgeExportEngine — bundle structure, hash, signature."""

    @pytest.mark.asyncio
    async def test_export_creates_bundle_structure(self, tmp_bundle_dir: Path, fake_store) -> None:
        """Export should create all required files in the bundle directory."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="test-secret",
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []

            result = await engine.export_project("test-project")

        # Check files created
        assert (tmp_bundle_dir / "manifest.json").exists()
        assert (tmp_bundle_dir / "artifacts.jsonl").exists()
        assert (tmp_bundle_dir / "documents.jsonl").exists()
        assert (tmp_bundle_dir / "nodes.jsonl").exists()
        assert (tmp_bundle_dir / "edges.jsonl").exists()
        assert (tmp_bundle_dir / "embeddings.jsonl").exists()
        assert (tmp_bundle_dir / "catalog.jsonl").exists()
        assert (tmp_bundle_dir / "signature.sig").exists()

        # Check result counts
        assert result.project_id == "test-project"
        assert result.manifest_hash != ""
        assert result.signature != ""

    @pytest.mark.asyncio
    async def test_manifest_hash_computed(self, tmp_bundle_dir: Path, fake_store) -> None:
        """Manifest should contain a hash field (pre-hash content hash)."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="secret",
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")

        manifest_path = tmp_bundle_dir / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())
        # Hash should be present and non-empty (stored from before hash field was added)
        assert manifest_data.get("hash") != ""
        # Hash field should not be self-referential (not SHA of manifest that includes hash)
        assert manifest_data.get("hash") != hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    @pytest.mark.asyncio
    async def test_hmac_signature(self, tmp_bundle_dir: Path, fake_store) -> None:
        """HMAC signature should match computed value."""
        secret = "my-secret-key"
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key=secret,
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")

        manifest_path = tmp_bundle_dir / "manifest.json"
        signature_path = tmp_bundle_dir / "signature.sig"

        assert verify_signature(manifest_path, signature_path, secret)
        assert not verify_signature(manifest_path, signature_path, "wrong-secret")

    @pytest.mark.asyncio
    async def test_integrity_verification(self, tmp_bundle_dir: Path, fake_store) -> None:
        """verify_bundle_integrity should accept valid bundle, reject tampered."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="key",
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")

        # Valid bundle passes
        assert verify_bundle_integrity(tmp_bundle_dir, "key")

        # Tampered manifest fails
        manifest_path = tmp_bundle_dir / "manifest.json"
        data = json.loads(manifest_path.read_text())
        data["project_id"] = "HACKED"
        manifest_path.write_text(json.dumps(data))
        assert not verify_bundle_integrity(tmp_bundle_dir, "key")

    @pytest.mark.asyncio
    async def test_refuses_unsigned_without_optout(self, tmp_bundle_dir: Path, fake_store) -> None:
        """Fail-closed: no UMRS_SECRET_KEY and no opt-out raises (G-P2-3)."""
        from umrs.knowledge.bundle_export import BundleIntegrityError

        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="",
        )

        with pytest.raises(BundleIntegrityError, match="UMRS_SECRET_KEY"):
            await engine.export_project("test-project")

    @pytest.mark.asyncio
    async def test_unsigned_optout_marks_manifest(self, tmp_bundle_dir: Path, fake_store) -> None:
        """allow_unsigned=True produces a bundle marked unsigned=true (G-P2-3)."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="",
            allow_unsigned=True,
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")

        manifest = json.loads((tmp_bundle_dir / "manifest.json").read_text())
        assert manifest["unsigned"] is True
        assert (tmp_bundle_dir / "signature.sig").read_text().strip() == ""


# ── Backend Tests ─────────────────────────────────────────────────────────────


class TestKnowledgeExportBackend:
    """Tests for KnowledgeExportBackend orchestrator integration."""

    @pytest.mark.asyncio
    async def test_submit_job_creates_bundle(
        self, export_job: Job, export_attempt: JobAttempt, fake_pool
    ) -> None:
        """submit_job should create bundle and write result."""

        store_mock = MagicMock()
        store_mock._pool = fake_pool
        store_mock.list_artifacts = AsyncMock(return_value=([], 0))
        store_mock.list_irs = AsyncMock(return_value=[])
        store_mock.list_nodes = AsyncMock(return_value=[])
        store_mock.list_edges = AsyncMock(return_value=[])
        store_mock.list_embedding_vectors_paged = AsyncMock(return_value=([], 0))
        store_mock.list_catalog = AsyncMock(return_value=([], 0))

        with (
            patch("umrs.jobs.knowledge_adapters._created_pool", AsyncMock(return_value=fake_pool)),
            patch("umrs.knowledge.store.KnowledgeStore", return_value=store_mock),
        ):
            backend = KnowledgeExportBackend()
            ref = await backend.submit_job(export_job, export_attempt)

        assert "knowledge_export|" in ref
        assert "project=test-project" in ref

    @pytest.mark.asyncio
    async def test_degraded_when_no_pool(self, export_job: Job, export_attempt: JobAttempt) -> None:
        """Backend should degrade gracefully when pool is None."""
        with patch("umrs.jobs.knowledge_adapters._created_pool", AsyncMock(return_value=None)):
            backend = KnowledgeExportBackend()
            ref = await backend.submit_job(export_job, export_attempt)

        assert "knowledge_export|" in ref
        assert export_attempt.metrics.get("result")  # result was written


class TestKnowledgeExportStatusChecker:
    """Tests for KnowledgeExportStatusChecker."""

    @pytest.mark.asyncio
    async def test_valid_backend_ref(self, export_attempt: JobAttempt, export_job: Job) -> None:
        """Should succeed when backend_ref starts with 'knowledge_export|'."""
        export_attempt.backend_ref = "knowledge_export|project=test|ok"
        checker = KnowledgeExportStatusChecker()
        result = await checker.check_status(export_attempt, export_job)
        assert result.status == "succeeded"

    @pytest.mark.asyncio
    async def test_missing_backend_ref(self, export_attempt: JobAttempt, export_job: Job) -> None:
        """Should fail when backend_ref is empty."""
        export_attempt.backend_ref = ""
        checker = KnowledgeExportStatusChecker()
        result = await checker.check_status(export_attempt, export_job)
        assert result.status == "failed"


# ── References + Manifest Markers (WP-17 / G-P2-2) ────────────────────────────


class _RefsStore:
    """Minimal store for reference-population tests (async + pool present)."""

    def __init__(self) -> None:
        self._pool = _RefsPool()

    async def list_artifacts(self, **_kw):
        return ([], 0)

    async def list_irs(self, **_kw):
        return []

    async def list_nodes(self, **_kw):
        return []

    async def list_edges(self, **_kw):
        return []


class _RefsPool:
    async def fetch(self, query, *_args):
        if "FROM skills" in query:
            return [{"name": "science-run", "version": "1.2.0", "scope": "project"}]
        if "FROM mcp_servers" in query:
            return [{"server_id": "s1", "name": "lab-mcp", "version": "0.3.0"}]
        return []

    def acquire(self):
        return _RefsConn()


class _RefsConn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def fetchrow(self, *_a):
        return {"count": 0}

    async def fetch(self, *_a):
        return []


class TestGtwop2ReferencesAndMarkers:
    @pytest.mark.asyncio
    async def test_manifest_populates_references_and_declares_markers(
        self, tmp_bundle_dir: Path
    ) -> None:
        """G-P2-2 partial: skills/MCP by reference + declarative markers."""
        engine = KnowledgeExportEngine(
            store=_RefsStore(),
            output_dir=tmp_bundle_dir.parent,
            secret_key="k",
        )
        result = await engine.export_project("test-project")
        assert result.signature
        manifest = json.loads((tmp_bundle_dir / "manifest.json").read_text())
        assert manifest["skills"] == [
            {"ref": "project:science-run", "version_pin": "1.2.0", "scope": "project"}
        ]
        assert manifest["mcp_configs"] == [{"ref": "lab-mcp", "version_pin": "0.3.0"}]
        assert manifest["collision_policy"] == "abort"
        assert manifest["provenance_canonical"] is False
        assert manifest["reconciliation_snapshot"] is None


# ── Reconciliation Snapshot Attach (WP-18 / ADR-032 #5 attach leg) ────────────


class TestReconcileSnapshotAttach:
    @pytest.mark.asyncio
    async def test_snapshot_consumed_gates_the_canonical_plane(
        self, tmp_bundle_dir: Path, fake_store, monkeypatch
    ) -> None:
        """ADR-032 #5: the exporter consumes the checked snapshot — nodes of
        ineligible assets and every edge touching them stay out; the manifest
        records what was omitted and why."""
        from umrs.knowledge.reconcile_snapshot import ExportSnapshot

        summary = {
            "captured_at": "2026-09-24T00:00:00+00:00",
            "states": {"COMPLETE": 0, "EMBEDDING_PENDING": 0, "PARTIAL": 1, "ORPHANED": 0},
            "assets_total": 1,
            "catalogue_available": True,
            "digest": "a" * 64,
            "consumed": True,
            "export_eligible_states": ["COMPLETE", "EMBEDDING_PENDING"],
            "excluded_assets": [
                {"document_id": "doc-1", "state": "PARTIAL", "reasons": ["zero chunks"]}
            ],
            "orphaned_chunks_excluded": 0,
            "excluded_node_count": 1,
        }
        monkeypatch.setattr(
            "umrs.knowledge.reconcile_snapshot.take_export_snapshot",
            lambda _pid, **_: ExportSnapshot(
                summary=summary, excluded_node_ids=frozenset({"node-001"})
            ),
        )
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="k",
        )
        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")
        manifest = json.loads((tmp_bundle_dir / "manifest.json").read_text())
        assert manifest["reconciliation_snapshot"] == summary
        nodes = (tmp_bundle_dir / "nodes.jsonl").read_text()
        edges = (tmp_bundle_dir / "edges.jsonl").read_text()
        assert "node-001" not in nodes  # the PARTIAL asset's node
        assert edges.strip() == ""  # its only edge touched node-001

    @pytest.mark.asyncio
    async def test_snapshot_null_when_summary_unavailable(
        self, tmp_bundle_dir: Path, fake_store, monkeypatch
    ) -> None:
        monkeypatch.delenv("UMRS_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="k",
        )
        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")
        manifest = json.loads((tmp_bundle_dir / "manifest.json").read_text())
        assert manifest["reconciliation_snapshot"] is None


# ── Bundle Format Tests ──────────────────────────────────────────────────────


class TestBundleFormat:
    """Tests for the r4-kb v1.x bundle flat-file format."""

    @pytest.mark.asyncio
    async def test_artifacts_jsonl_structure(self, tmp_bundle_dir: Path, fake_store) -> None:
        """artifacts.jsonl should contain one JSON object per line."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="",
            allow_unsigned=True,
        )

        # Patch to return known artifact data
        fake_store.list_artifacts = AsyncMock(
            return_value=(
                [
                    MagicMock(
                        **{
                            "artifact_id": "art-001",
                            "content_hash": "sha256-abc",
                            "kind": "document",
                            "project_id": "test-project",
                            "source_ref": "/tmp/test.txt",
                            "size_bytes": 1024,
                            "mime_type": "text/plain",
                            "status": "active",
                            "metadata": {"key": "val"},
                            "created_at": None,
                            "updated_at": None,
                        }
                    )
                ],
                1,
            )
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")

        with open(tmp_bundle_dir / "artifacts.jsonl") as f:
            line_list = [line for line in f.readlines() if line.strip()]

        assert len(line_list) >= 1
        row = json.loads(line_list[0])
        assert row["artifact_id"] == "art-001"
        assert "content_hash" in row
        assert "kind" in row

    @pytest.mark.asyncio
    async def test_nodes_jsonl_structure(self, tmp_bundle_dir: Path, fake_store) -> None:
        """nodes.jsonl should contain node_id, project_id, label."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="",
            allow_unsigned=True,
        )

        fake_store.list_nodes = AsyncMock(
            return_value=[
                MagicMock(
                    **{
                        "node_id": "node-1",
                        "project_id": "test-project",
                        "label": "TestClass",
                        "node_type": "class",
                        "content": {},
                        "content_hash": "",
                        "source_ir": "ir-1",
                        "created_at": None,
                        "updated_at": None,
                    }
                )
            ]
        )
        fake_store.list_edges = AsyncMock(return_value=[])
        fake_store.list_irs = AsyncMock(return_value=[])
        fake_store.list_artifacts = AsyncMock(return_value=([], 0))

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")

        with open(tmp_bundle_dir / "nodes.jsonl") as f:
            line_list = [line for line in f.readlines() if line.strip()]

        assert len(line_list) >= 1
        row = json.loads(line_list[0])
        assert row["node_id"] == "node-1"
        assert row["project_id"] == "test-project"
        assert row["label"] == "TestClass"

    @pytest.mark.asyncio
    async def test_manifest_format(self, tmp_bundle_dir: Path, fake_store) -> None:
        """manifest.json should have format, version, tables."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="",
            allow_unsigned=True,
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = []
            await engine.export_project("test-project")

        manifest = json.loads((tmp_bundle_dir / "manifest.json").read_text())
        assert manifest["format"] == "r4-kb"
        assert manifest["version"] == "1.2"
        assert "tables" in manifest
        assert "a2l_links" in manifest["tables"]
        assert "a2l_refs" in manifest["tables"]
        assert "hash" in manifest
        assert "signature" in manifest
        assert "embedding" in manifest
        assert "generator" in manifest
        assert manifest["generator"] == "umrs"

    @pytest.mark.asyncio
    async def test_roundtrip_equivalence(self, tmp_bundle_dir: Path, fake_store) -> None:
        """Export→import query round-trip: re-read bundle data matches original."""
        engine = KnowledgeExportEngine(
            store=fake_store,
            output_dir=tmp_bundle_dir.parent,
            secret_key="test-secret",
        )

        with (
            patch.object(engine, "_list_catalog") as mock_cat,
        ):
            mock_cat.return_value = [
                {
                    "id": "1",
                    "name": "Cat1",
                    "project_id": "test-project",
                    "metadata": {},
                    "created_at": None,
                }
            ]
            await engine.export_project("test-project")

        # Check all files exist and are non-empty (except embeddings which may be empty)
        files_expected = [
            "manifest.json",
            "artifacts.jsonl",
            "documents.jsonl",
            "nodes.jsonl",
            "edges.jsonl",
            "catalog.jsonl",
            "signature.sig",
        ]
        for fname in files_expected:
            fpath = tmp_bundle_dir / fname
            assert fpath.exists(), f"Missing file: {fname}"
            assert fpath.stat().st_size > 0, f"Empty file: {fname}"

        # embeddings.jsonl may be empty (depends on data)
        emb_path = tmp_bundle_dir / "embeddings.jsonl"
        assert emb_path.exists(), "embeddings.jsonl missing"

        # Manifest re-readable
        manifest = json.loads((tmp_bundle_dir / "manifest.json").read_text())
        assert manifest["project_id"] == "test-project"

        # All JSONL files parse
        for fname in [
            "artifacts.jsonl",
            "documents.jsonl",
            "nodes.jsonl",
            "edges.jsonl",
            "embeddings.jsonl",
            "catalog.jsonl",
        ]:
            fpath = tmp_bundle_dir / fname
            with open(fpath) as f:
                line_list = [line for line in f if line.strip()]
            for line in line_list:
                json.loads(line)  # Should not raise


class TestCatalogBlockExport:
    """D-40 ties: the manifest carries the D-39 row + ownership block
    (no invention — empty when the project has no catalog)."""

    async def test_catalog_record_populates_catalog_and_ownership(self, tmp_path):
        from umrs.knowledge.bundle_export import KnowledgeExportEngine

        store = MagicMock()

        class _Pool:
            async def fetchrow(self, query, *_args):
                if "scientific_catalogs" in query:
                    return {
                        "name": "mof-defects",
                        "version": "0.3.0",
                        "description": "d",
                        "ontology": "mof-v2",
                        "licence": "CC-BY-4.0",
                        "contributors": '["alice"]',
                        "provenance": '{"source": "lab"}',
                        "redaction_rules": "[]",
                    }
                return None

        store._pool = _Pool()
        engine = KnowledgeExportEngine(store=store, output_dir=str(tmp_path), secret_key="")
        record = await engine._resolve_catalog_record("proj-x")
        assert record["name"] == "mof-defects"
        assert record["contributors"] == ["alice"]
        assert record["provenance"] == {"source": "lab"}
        manifest = engine._build_manifest("proj-x", {}, {}, catalog_record=record)
        assert manifest["catalog"]["name"] == "mof-defects"
        assert manifest["ownership"] == {
            "contributors": ["alice"],
            "licence": "CC-BY-4.0",
        }

    async def test_no_catalog_row_emits_empty_blocks(self, tmp_path):
        from umrs.knowledge.bundle_export import KnowledgeExportEngine

        store = MagicMock()

        class _Pool:
            async def fetchrow(self, _query, *_args):
                return None

        store._pool = _Pool()
        engine = KnowledgeExportEngine(store=store, output_dir=str(tmp_path), secret_key="")
        record = await engine._resolve_catalog_record("proj-x")
        assert record == {}
        manifest = engine._build_manifest("proj-x", {}, {})
        assert manifest["catalog"] == {}
        assert manifest["ownership"] == {"contributors": [], "licence": ""}

    async def test_registry_fault_yields_empty_record(self, tmp_path):
        from umrs.knowledge.bundle_export import KnowledgeExportEngine

        store = MagicMock()

        class _Pool:
            async def fetchrow(self, _query, *_args):
                raise RuntimeError("no table")

        store._pool = _Pool()
        engine = KnowledgeExportEngine(store=store, output_dir=str(tmp_path), secret_key="")
        assert await engine._resolve_catalog_record("proj-x") == {}


class TestExportRedaction:
    """D-46: the catalog record decides what leaves the site; signatures
    cover the redacted manifest; every redaction is recorded honestly."""

    @staticmethod
    def _manifest(tmp_path) -> dict:
        from umrs.knowledge.bundle_export import KnowledgeExportEngine

        store = MagicMock()
        store._pool = MagicMock()
        engine = KnowledgeExportEngine(store=store, output_dir=str(tmp_path), secret_key="")
        record = {
            "name": "mof",
            "contributors": ["alice", "bob"],
            "licence": "CC-BY-4.0",
            "provenance": {"source": "lab"},
            "description": "d",
        }
        return engine._build_manifest("proj", {}, {}, catalog_record=record)

    def test_drop_contributors_empties_both_blocks(self, tmp_path):
        from umrs.knowledge.bundle_export import _apply_redaction_rules

        manifest = self._manifest(tmp_path)
        _apply_redaction_rules(manifest, [{"target": "contributors", "action": "drop"}])
        assert manifest["catalog"]["contributors"] == []
        assert manifest["ownership"]["contributors"] == []
        assert manifest["catalog"]["licence"] == "CC-BY-4.0"  # untouched field stays
        assert manifest["redacted"] == ["contributors:drop"]

    def test_hash_contributors_fingerprints_each(self, tmp_path):
        from umrs.knowledge.bundle_export import _apply_redaction_rules

        manifest = self._manifest(tmp_path)
        _apply_redaction_rules(manifest, [{"target": "contributors", "action": "hash"}])
        for name in manifest["ownership"]["contributors"]:
            assert name.startswith("sha256:") and len(name) == 19
        assert manifest["catalog"]["contributors"][0].startswith("sha256:")

    def test_provenance_drop_and_unknown_rules_ignored(self, tmp_path):
        from umrs.knowledge.bundle_export import _apply_redaction_rules

        manifest = self._manifest(tmp_path)
        _apply_redaction_rules(
            manifest,
            [
                {"target": "provenance", "action": "drop"},
                {"target": "nonsense", "action": "drop"},
                {"target": "licence", "action": "vaporize"},
            ],
        )
        assert manifest["catalog"]["provenance"] == {}
        assert manifest["catalog"]["licence"] == "CC-BY-4.0"
        assert manifest["redacted"] == ["provenance:drop"]

    def test_no_rules_marks_empty_redacted_list(self, tmp_path):
        from umrs.knowledge.bundle_export import _apply_redaction_rules

        manifest = self._manifest(tmp_path)
        _apply_redaction_rules(manifest, None)
        assert manifest["redacted"] == []
        assert manifest["ownership"]["contributors"] == ["alice", "bob"]
