"""Cycle-2 umrs-follow lane (N1, 2026-10-21): read the vectors ingest writes.

Found by the umrs-api lane: ``POST /umrs/knowledge/corpus-attrition-report``
and the bundle export/import read ``memory_vectors(id, vector, metadata,
document_id, project_id)``. No migration creates those columns (0010's
``memory_vectors`` is context memory keyed by ``memory_id``), so the report
was a 500 on every deployment, the export's ``embeddings.jsonl`` was always
empty behind a "degraded" log line (its ``embedding_count`` always 0), and
an import of a bundle carrying such rows failed.

The per-project vectors knowledge ingest writes are ``knowledge_node_embeddings``
(0075; KNOWLEDGE_EMBED, one row per chunk node), joined to ``knowledge_nodes``
for the project. The report reads them; the export counts them as its
embeddings (they already ride ``node_embeddings.jsonl``); a deployment
without the table answers a typed ``knowledge.unavailable``, never a 500,
and an export that cannot read them says "embeddings not included".
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from asyncpg.exceptions import UndefinedColumnError, UndefinedTableError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.files import get_project_id
from umrs.api.knowledge_attrition import _report_for_project, router
from umrs.knowledge.bundle_export import KnowledgeExportEngine

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"


class _RaisingConn:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def __aenter__(self) -> _RaisingConn:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def fetch(self, *_args: Any) -> list[Any]:
        raise self.exc


def _client(monkeypatch: pytest.MonkeyPatch, pool: Any) -> TestClient:
    async def _pool(_request: Any) -> Any:
        return pool

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_project_id] = lambda: "pa"
    monkeypatch.setattr("umrs.api.knowledge_attrition.knowledge_asyncpg_pool", _pool)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "exc",
    [
        UndefinedTableError('relation "knowledge_node_embeddings" does not exist'),
        UndefinedColumnError('column "project_id" does not exist'),
    ],
)
def test_report_without_the_vector_table_is_a_typed_503(monkeypatch, exc) -> None:
    pool = SimpleNamespace(acquire=lambda: _RaisingConn(exc))
    resp = _client(monkeypatch, pool).post("/umrs/knowledge/corpus-attrition-report")
    assert resp.status_code == 503, resp.text
    assert resp.headers["x-platform-error-code"] == "knowledge.unavailable"
    assert "not available on this deployment" in resp.json()["detail"]


def test_report_reads_the_ingest_vector_table() -> None:
    seen: list[str] = []

    class _Conn(_RaisingConn):
        async def fetch(self, sql: str, *_args: Any) -> list[Any]:
            seen.append(" ".join(sql.split()))
            return []

    pool = SimpleNamespace(acquire=lambda: _Conn(RuntimeError()))
    report = asyncio.run(_report_for_project(pool, "pa"))
    assert report["source"] == "knowledge_node_embeddings"
    assert "FROM knowledge_node_embeddings kne JOIN knowledge_nodes kn" in seen[0]
    assert "memory_vectors" not in seen[0]


# ── export: the vectors are the node vectors; never silently empty ─────────


class _ExportPool:
    """Answers like a real deployment: memory_vectors has no project_id."""

    def __init__(self, node_vectors_error: Exception | None = None) -> None:
        self.sql: list[str] = []
        self.node_vectors_error = node_vectors_error

    def _check(self, sql: str) -> None:
        self.sql.append(sql)
        if "memory_vectors" in sql:
            raise UndefinedColumnError('column "project_id" does not exist')

    async def fetch(self, sql: str, *_args: Any) -> list[Any]:
        self._check(sql)
        if "FROM knowledge_node_embeddings" in sql and "kne.embedding::text" in sql:
            if self.node_vectors_error is not None:
                raise self.node_vectors_error
            return [
                {"node_id": f"n{i}", "model_name": "e5", "dimension": 2,
                 "embedding": "[0.1,0.2]", "content_hash": "h"}
                for i in range(3)
            ]
        return []

    async def fetchrow(self, sql: str, *_args: Any) -> Any:
        self._check(sql)
        return None

    def acquire(self) -> Any:
        pool = self

        class _Ctx:
            async def __aenter__(self) -> Any:
                return pool

            async def __aexit__(self, *exc: object) -> bool:
                return False

        return _Ctx()


def _export_store(pool: _ExportPool) -> SimpleNamespace:
    async def _empty_page(**_kw: Any) -> tuple[list[Any], int]:
        return [], 0

    async def _none(**_kw: Any) -> list[Any]:
        return []

    return SimpleNamespace(
        _pool=pool,
        list_artifacts=_empty_page,
        list_irs=_none,
        list_nodes=_none,
        list_edges=_none,
    )


def _export(tmp_path: Path, pool: _ExportPool) -> tuple[Any, dict[str, Any]]:
    engine = KnowledgeExportEngine(
        store=_export_store(pool), output_dir=tmp_path, allow_unsigned=True
    )
    result = asyncio.run(engine.export_project("pa"))
    manifest = json.loads((tmp_path / "pa" / "manifest.json").read_text())
    return result, manifest


def test_export_counts_the_node_vectors(tmp_path) -> None:
    result, manifest = _export(tmp_path, _ExportPool())
    assert result.embedding_count == 3
    assert manifest["tables"]["node_embeddings"]["count"] == 3
    assert manifest["tables"]["embeddings"]["count"] == 0
    assert (tmp_path / "pa" / "embeddings.jsonl").exists()  # v1.x layout kept


def test_export_never_reads_memory_vectors_and_is_not_degraded(tmp_path) -> None:
    pool = _ExportPool()
    result, manifest = _export(tmp_path, pool)
    assert not result.degraded
    assert "embeddings_note" not in manifest
    assert not [q for q in pool.sql if "memory_vectors" in q]


def test_export_that_cannot_read_the_vectors_says_so(tmp_path) -> None:
    pool = _ExportPool(UndefinedTableError('relation "knowledge_node_embeddings" does not exist'))
    result, manifest = _export(tmp_path, pool)
    assert result.embedding_count == 0
    assert result.degraded
    assert result.degraded_reason.startswith("embeddings not included")
    assert manifest["embeddings_note"].startswith("embeddings not included")


# ── live Postgres: the report's SQL against the migrated shape ────────────


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


_NODES_DDL = """
CREATE TABLE knowledge_nodes (
    node_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    content JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""


async def _live_report(schema: str) -> dict[str, Any]:
    import asyncpg

    admin = await asyncpg.connect(_db_url())
    try:
        try:
            await admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except asyncpg.PostgresError as exc:
            pytest.skip(f"pgvector unavailable: {exc}")
        await admin.execute(f"CREATE SCHEMA {schema}")
        await admin.execute(f"SET search_path = {schema}, public")
        await admin.execute(_NODES_DDL)
        await admin.execute((_MIGRATIONS / "0075_knowledge_node_embeddings.sql").read_text())
        old = datetime.now(timezone.utc) - timedelta(days=120)
        nodes = [
            ("c1", "pa", {"kind": "chunk", "artifact_id": "a1"}),
            ("c2", "pa", {"kind": "chunk", "artifact_id": "a1"}),
            ("c3", "pa", {"kind": "chunk", "artifact_id": "a2"}),
            ("x1", "pb", {"kind": "chunk", "artifact_id": "b1"}),
        ]
        for node_id, project, content in nodes:
            await admin.execute(
                "INSERT INTO knowledge_nodes VALUES ($1, $2, $3::jsonb)",
                node_id, project, json.dumps(content),
            )
            await admin.execute(
                "INSERT INTO knowledge_node_embeddings (node_id, model_name, dimension,"
                " embedding, created_at) VALUES ($1, 'e5', 2, '[0.1,0.2]'::vector, $2)",
                node_id, old if node_id == "c3" else datetime.now(timezone.utc),
            )
        pool = await asyncpg.create_pool(
            _db_url(), min_size=1, max_size=2,
            server_settings={"search_path": f"{schema},public"},
        )
        try:
            return await _report_for_project(pool, "pa")
        finally:
            await pool.close()
    finally:
        await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin.close()


@pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")
def test_live_report_buckets_the_projects_ingested_vectors() -> None:
    report = asyncio.run(_live_report(f"c2n1_{uuid.uuid4().hex[:8]}"))
    assert report["rows_total"] == 3  # pb's vector is not pa's
    assert report["documents"] == 2  # a1 (two chunks) and a2
    assert report["buckets"]["lt_30d"]["rows"] == 2
    assert report["buckets"]["gt_90d"]["documents"] == 1
    assert report["venues"] == {"unset": 3}  # never inferred
