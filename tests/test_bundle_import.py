"""K4 tests — bundle import/restore engine and route.

Engine tests use a recording fake store; route tests use TestClient
with a fake app-state pool. Integrity checks run against real bundles
written by the export writer when a secret is involved.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.bundle_restore import router as restore_router
from umrs.knowledge.bundle_import import (
    BundleFormatError,
    KnowledgeImportEngine,
)

_SECRET = "test-secret"


# ── Fakes ─────────────────────────────────────────────────────────────────


class _FakeConn:
    def __init__(self, parent):
        self._parent = parent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def execute(self, sql, *params):
        self._parent.executed.append((sql, params))
        return "INSERT 0 1"

    async def fetchval(self, _sql, *_params):
        return None

    async def fetchrow(self, _sql, *_params):
        return {"id": 1}


class _FakePool:
    def __init__(self) -> None:
        self.executed: list = []

    def acquire(self):
        return _FakeConn(self)


class _FakeStore:
    """Recording stand-in for KnowledgeStore write methods."""

    def __init__(self) -> None:
        self._pool = _FakePool()
        self.artifacts: list[dict] = []
        self.irs: list[dict] = []
        self.nodes: list = []
        self.edges: list = []
        self.a2l_links: list[dict] = []
        self.a2l_refs: list[dict] = []
        self.existing_hashes: set[str] = set()

    async def insert_artifact(self, **kwargs) -> bool:
        if kwargs["content_hash"] in self.existing_hashes:
            return False
        self.existing_hashes.add(kwargs["content_hash"])
        self.artifacts.append(kwargs)
        return True

    async def insert_ir(self, **kwargs) -> bool:
        self.irs.append(kwargs)
        return True

    async def upsert_node(self, node) -> None:
        self.nodes.append(node)

    async def upsert_edge(self, edge) -> None:
        self.edges.append(edge)

    async def insert_knowledge_a2l_link(self, **kwargs) -> bool:
        self.a2l_links.append(kwargs)
        return True

    async def insert_a2l_ref(self, **kwargs) -> bool:
        self.a2l_refs.append(kwargs)
        return True


# ── Bundle writer helper ──────────────────────────────────────────────────


def _write_bundle(
    root: Path,
    *,
    project_id: str = "proj-a",
    with_signature: bool = False,
    corrupt: bool = False,
    version: str = "1.1",
    fmt: str = "r4-kb",
) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()

    manifest = {
        "format": fmt,
        "version": version,
        "project_id": project_id,
        "exported_at": "2026-08-02T00:00:00+00:00",
        "tables": {
            "artifacts": {"count": 2, "file": "artifacts.jsonl"},
            "documents": {"count": 2, "file": "documents.jsonl"},
            "nodes": {"count": 1, "file": "nodes.jsonl"},
            "edges": {"count": 1, "file": "edges.jsonl"},
            "embeddings": {"count": 1, "file": "embeddings.jsonl"},
            "catalog": {"count": 1, "file": "catalog.jsonl"},
            "a2l_links": {"count": 1, "file": "a2l_links.jsonl"},
            "a2l_refs": {"count": 1, "file": "a2l_refs.jsonl"},
        },
    }

    artifacts = [
        {
            "artifact_id": f"art-{i}",
            "content_hash": f"hash-{i}",
            "kind": "document",
            "size_bytes": 100,
            "metadata": {"n": i},
        }
        for i in range(2)
    ]
    documents = [
        {
            "ir_id": f"ir-{i}",
            "artifact_id": f"art-{i}",
            "ir_type": "document",
            "version": 1,
            "content": {"text": f"doc-{i}"},
            "content_hash": f"ir-hash-{i}",
        }
        for i in range(2)
    ]
    nodes = [
        {
            "node_id": "node-1",
            "project_id": project_id,
            "label": "Concept",
            "node_type": "concept",
            "content": {},
            "content_hash": "node-hash",
            "source_ir": "ir-0",
        }
    ]
    edges = [
        {
            "edge_id": "edge-1",
            "project_id": project_id,
            "source_id": "node-1",
            "target_id": "node-1",
            "relation_type": "related_to",
            "content": {},
        }
    ]
    embeddings = [{"vector": [0.1, 0.2], "metadata": {}, "document_id": "art-0"}]
    catalog = [{"name": "row-1", "project_id": project_id, "metadata": {}}]
    a2l_links = [
        {
            "knowledge_artifact_id": "art-0",
            "a2l_artifact_id": "a2l-1",
            "job_id": "j-1",
            "attempt_id": "att-1",
            "edge_kind": "derived",
        }
    ]
    a2l_refs = [{"artifact_id": "art-0", "a2l_hash": "h1", "a2l_id": "a2l-1", "source": "auto"}]

    rows_map = {
        "artifacts.jsonl": artifacts,
        "documents.jsonl": documents,
        "nodes.jsonl": nodes,
        "edges.jsonl": edges,
        "embeddings.jsonl": embeddings,
        "catalog.jsonl": catalog,
        "a2l_links.jsonl": a2l_links,
        "a2l_refs.jsonl": a2l_refs,
    }
    for name, rows in rows_map.items():
        (bundle / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["hash"] = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["signature"] = hmac.new(
        _SECRET.encode("utf-8"), manifest_bytes, hashlib.sha256
    ).hexdigest()
    if corrupt:
        manifest["signature"] = "bad-signature"
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if with_signature:
        (bundle / "signature.sig").write_text(manifest["signature"], encoding="utf-8")
    return bundle


# ── Engine tests ──────────────────────────────────────────────────────────


class TestManifestChecks:
    async def test_missing_manifest_raises(self, tmp_path):
        engine = KnowledgeImportEngine(store=_FakeStore())
        with pytest.raises(BundleFormatError, match="manifest.json"):
            await engine.import_project(tmp_path / "absent")

    async def test_wrong_format_raises(self, tmp_path):
        bundle = _write_bundle(tmp_path, fmt="other")
        engine = KnowledgeImportEngine(store=_FakeStore())
        with pytest.raises(BundleFormatError, match="format"):
            await engine.import_project(bundle)

    async def test_wrong_version_raises(self, tmp_path):
        bundle = _write_bundle(tmp_path, version="0.9")
        engine = KnowledgeImportEngine(store=_FakeStore())
        with pytest.raises(BundleFormatError, match="version"):
            await engine.import_project(bundle)

    async def test_corrupt_signature_detected(self, tmp_path):
        bundle = _write_bundle(tmp_path, with_signature=True, corrupt=True)
        engine = KnowledgeImportEngine(store=_FakeStore(), secret_key=_SECRET)
        with pytest.raises(BundleFormatError, match="integrity"):
            await engine.import_project(bundle)


class TestRestoreFlow:
    async def test_restore_core_tables(self, tmp_path):
        bundle = _write_bundle(tmp_path)
        store = _FakeStore()
        engine = KnowledgeImportEngine(store=store)

        result = await engine.import_project(bundle, verify=False)

        assert result.project_id == "proj-a"
        assert result.artifact_count == 2
        assert result.document_count == 2
        assert result.node_count == 1
        assert result.edge_count == 1
        assert result.skipped_count == 0

    async def test_restore_aux_tables(self, tmp_path):
        bundle = _write_bundle(tmp_path)
        store = _FakeStore()
        engine = KnowledgeImportEngine(store=store)

        result = await engine.import_project(bundle, verify=False)

        assert result.embedding_count == 1
        assert result.catalog_count == 1
        assert result.a2l_link_count == 1
        assert result.a2l_ref_count == 1

    async def test_retarget_rewrites_project_scope(self, tmp_path):
        bundle = _write_bundle(tmp_path, project_id="old-project")
        store = _FakeStore()
        engine = KnowledgeImportEngine(store=store)

        result = await engine.import_project(bundle, target_project_id="new-project", verify=False)

        assert result.project_id == "new-project"
        assert all(a["project_id"] == "new-project" for a in store.artifacts)
        assert all(n.project_id == "new-project" for n in store.nodes)
        # SQL tables carry the new scope in their bound params.
        # embeddings: (project, vector, metadata, document_id)
        # catalog:    (name, project, metadata)
        for _sql, params in store._pool.executed:
            assert "new-project" in params

    async def test_rerun_is_idempotently_skipped(self, tmp_path):
        bundle = _write_bundle(tmp_path)
        store = _FakeStore()
        engine = KnowledgeImportEngine(store=store)

        await engine.import_project(bundle, verify=False)
        second = await engine.import_project(bundle, verify=False)

        # Artifacts dedup by content hash on the second run.
        assert second.artifact_count == 0
        assert second.skipped_count == 2
        assert second.document_count == 2  # write-once API allows re-insert here

    async def test_integrity_verification_with_secret(self, tmp_path):
        bundle = _write_bundle(tmp_path, with_signature=False, corrupt=False)
        engine = KnowledgeImportEngine(store=_FakeStore(), secret_key=_SECRET)
        # SIGNATURE file missing → integrity check fails.
        with pytest.raises(BundleFormatError, match="integrity"):
            await engine.import_project(bundle)

    async def test_verified_bundle_passes_and_marks_flag(self, tmp_path):
        bundle = _write_bundle(tmp_path, with_signature=True)
        engine = KnowledgeImportEngine(store=_FakeStore(), secret_key=_SECRET)
        result = await engine.import_project(bundle)
        assert result.verified is True


# ── Route tests ───────────────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch) -> FastAPI:
    app = FastAPI()
    app.include_router(restore_router)
    app.state.umrs_db_pool = _FakePool()
    app.state.umrs_knowledge_asyncpg_pool = _FakePool()
    monkeypatch.delenv("UMRS_SECRET_KEY", raising=False)
    return app


class TestRestoreRoute:
    def test_503_without_pool(self):
        bare = FastAPI()
        bare.include_router(restore_router)
        client = TestClient(bare)
        resp = client.post("/umrs/knowledge/restore", json={"bundle_dir": "/tmp/x"})
        assert resp.status_code == 503

    def test_route_happy_path(self, app, tmp_path):
        bundle = _write_bundle(tmp_path)
        client = TestClient(app)
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": str(bundle)},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["restore"]
        assert payload["artifact_count"] == 2
        assert payload["node_count"] == 1

    def test_route_maps_format_error_to_400(self, app, tmp_path):
        bad = _write_bundle(tmp_path, version="0.1")
        client = TestClient(app)
        resp = client.post(
            "/umrs/knowledge/restore",
            json={"bundle_dir": str(bad)},
        )
        assert resp.status_code == 400
