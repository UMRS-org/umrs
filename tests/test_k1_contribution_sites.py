"""K1.2: every artifact insert site records a contribution (migration 0129).

The ingest chain's site runs on Postgres in test_k1_contributions_pg.py;
these pin the other sites with fakes: the rag proxy mint, the rag executor
mint, legacy canonization, workflow outputs, loop findings and bundle import
(including a deduplicated hit, which records the stored id).
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.knowledge.graph_registry import default_graph_id


@pytest.fixture()
def recorded(monkeypatch) -> list[Any]:
    """Capture every best-effort contribution write."""
    rows: list[Any] = []

    async def _record(_pool: Any, contribution: Any) -> bool:
        rows.append(contribution)
        return True

    monkeypatch.setattr("umrs.knowledge.contributions.record_with_pool", _record)
    return rows


class _FakeStore:
    """Async store double: ``insert_artifact`` answers *held_by*."""

    def __init__(self, _pool: Any = None, *, held_by: str = "") -> None:
        self.held_by = held_by

    async def insert_artifact(self, *args: Any, **kwargs: Any) -> str:
        return self.held_by or kwargs.get("artifact_id") or args[0]

    async def upsert_node(self, *_a: Any, **_k: Any) -> None:
        return None

    async def upsert_edge(self, *_a: Any, **_k: Any) -> None:
        return None


def _who(row: Any) -> tuple[str, str, str, str]:
    return row.artifact_id, row.graph_id, row.user_id, row.origin_actor


def test_rag_proxy_mint_records_the_uploader(monkeypatch, recorded) -> None:
    from umrs.api import rag

    async def _pool(_request: Any) -> object:
        return object()

    monkeypatch.setattr("umrs.api._asyncpg_pool.knowledge_asyncpg_pool", _pool)
    monkeypatch.setattr(
        "umrs.knowledge.store.KnowledgeStore",
        lambda pool: _FakeStore(pool, held_by="artifact-rag-held"),
    )
    asyncio.run(
        rag._mint_knowledge_nodes(
            SimpleNamespace(),
            user_id="alice",
            project_id="p1",
            document_id="doc-1",
            filename="notes.txt",
            mime_type="text/plain",
            chunk_details=[],
        )
    )
    # The deduplicated (stored) id is the one attributed.
    assert [_who(r) for r in recorded] == [
        ("artifact-rag-held", default_graph_id("p1"), "alice", "user")
    ]


class _Cursor:
    def __init__(self, log: list[tuple[str, tuple]]) -> None:
        self.log = log

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.log.append((" ".join(sql.split()), tuple(params)))

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _Conn:
    def __init__(self) -> None:
        self.log: list[tuple[str, tuple]] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.log)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def _contribution_writes(log: list[tuple[str, tuple]]) -> list[tuple]:
    return [p for sql, p in log if sql.startswith("INSERT INTO knowledge_artifact_contributions")]


def test_rag_executor_mint_writes_an_untraced_placeholder() -> None:
    from umrs.knowledge.rag_upload_mint import mint_rag_upload

    conn = _Conn()
    pool = SimpleNamespace(getconn=lambda: conn, putconn=lambda _c: None)
    mint_rag_upload(
        pool,
        project_id="p1",
        document_id="doc-9",
        filename="a.txt",
        mime_type="text/plain",
        chunks=[],
        embeddings=[],
        model_name="m",
    )
    assert _contribution_writes(conn.log) == [
        ("artifact-rag-doc-9", default_graph_id("p1"), "p1", "", "unknown", "a.txt")
    ]


class _No0129Cursor(_Cursor):
    """A database the executor reaches before migration 0129 ran."""

    def execute(self, sql: str, params: tuple = ()) -> None:
        super().execute(sql, params)
        if "INTO knowledge_artifact_contributions" in sql:
            raise RuntimeError('relation "knowledge_artifact_contributions" does not exist')


class _No0129Conn(_Conn):
    def cursor(self) -> _Cursor:
        return _No0129Cursor(self.log)


def test_a_database_without_0129_keeps_the_mint() -> None:
    """Review of K1 part A: the placeholder insert ran in the mint's own
    transaction with no guard, so a missing table rolled the mint back."""
    from umrs.knowledge.rag_upload_mint import mint_rag_upload

    conn = _No0129Conn()
    pool = SimpleNamespace(getconn=lambda: conn, putconn=lambda _c: None)
    mint_rag_upload(
        pool,
        project_id="p1",
        document_id="doc-9",
        filename="a.txt",
        mime_type="text/plain",
        chunks=[],
        embeddings=[],
        model_name="m",
    )
    statements = [sql.split(" (")[0] for sql, _p in conn.log]
    assert "ROLLBACK TO SAVEPOINT k1_contribution" in statements
    assert any(sql.startswith("INSERT INTO knowledge_nodes") for sql, _p in conn.log)


def test_legacy_canonization_survives_a_database_without_0129() -> None:
    from umrs.knowledge.legacy_corpus_canonize import canonize_document

    conn = _No0129Conn()
    canonize_document(
        conn, doc_key="k", doc_uri="bucket/legacy.pdf", chunk_rows=[], project_id="p1",
        max_per_doc=0,
    )
    assert any(sql.startswith("INSERT INTO knowledge_nodes") for sql, _p in conn.log)


def test_legacy_canonization_writes_an_untraced_placeholder() -> None:
    from umrs.knowledge.legacy_corpus_canonize import canonize_document

    conn = _Conn()
    canonize_document(
        conn,
        doc_key="k",
        doc_uri="bucket/legacy.pdf",
        chunk_rows=[],
        project_id="p1",
        max_per_doc=0,
    )
    writes = _contribution_writes(conn.log)
    assert len(writes) == 1
    assert writes[0][1:5] == (default_graph_id("p1"), "p1", "", "unknown")


def test_workflow_outputs_record_the_run_owner(monkeypatch, recorded) -> None:
    from umrs.jobs import knowledge_adapters as ka

    @contextlib.asynccontextmanager
    async def _pool():
        yield object()

    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "postgresql://x")
    monkeypatch.setattr(ka, "_open_pool", _pool)
    monkeypatch.setattr(
        "umrs.knowledge.store.KnowledgeStore", lambda pool: _FakeStore(pool, held_by="art-held")
    )
    monkeypatch.setattr(ka, "_fire_coroutine", lambda coro, _label="": asyncio.run(coro))
    ka._store_artifact_provenance(
        [{"artifact_id": "art-new", "content_hash": "h1"}],
        "p1",
        SimpleNamespace(job_id="job-wf", user_id="bob"),
        SimpleNamespace(attempt_id="att-1"),
        "wf",
        "step",
    )
    assert [_who(r) for r in recorded] == [("art-held", default_graph_id("p1"), "bob", "workflow")]


def test_loop_finding_records_the_loop_owner(monkeypatch, recorded) -> None:
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge.finding_ingest import KnowledgeFindingSink

    @contextlib.asynccontextmanager
    async def _pool():
        yield object()

    monkeypatch.setattr(ka, "_open_pool", _pool)
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", lambda pool: _FakeStore(pool))
    fields = {
        "artifact_id": "finding-1",
        "content_hash": "h",
        "project_id": "p1",
        "source_ref": "loop:l#iteration-1",
        "size_bytes": 3,
        "metadata": {"finding": {"statement": "Yield rose"}},
        "graph_id": "g-loop",
        "owner_user_id": "carol",
    }
    assert asyncio.run(KnowledgeFindingSink()._mint(fields)) is True
    assert [_who(r) for r in recorded] == [("finding-1", "g-loop", "carol", "loop")]
    assert recorded[0].title == "Yield rose"


def test_mint_iteration_hands_the_sink_graph_and_owner() -> None:
    from umrs.knowledge.finding_ingest import InMemoryFindingSink, mint_iteration
    from umrs.knowledge.loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop

    loop = ResearchLoop(
        loop_id="l1",
        project_id="p1",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.PUBLISHING,
        metadata={"owner_user_id": "carol"},
    )
    iteration = LoopIteration(
        iteration_id="it",
        loop_id="l1",
        iteration_number=1,
        findings=[{"finding_id": "f1", "statement": "s"}],
        status="completed",
    )
    sink = InMemoryFindingSink()
    mint_iteration(loop, iteration, sink=sink)
    assert sink.artifacts["f1"]["graph_id"] == default_graph_id("p1")
    assert sink.artifacts["f1"]["owner_user_id"] == "carol"


@pytest.mark.parametrize(("held_by", "expected"), [("", "art-b1"), ("art-local", "art-local")])
def test_bundle_import_records_the_importer(monkeypatch, recorded, held_by, expected) -> None:
    from umrs.knowledge.bundle_import import KnowledgeImportEngine

    store = _FakeStore(held_by=held_by)
    store._pool = object()
    engine = KnowledgeImportEngine(store=store, imported_by="dave")

    async def _absent(*_a: Any) -> bool:
        return False

    monkeypatch.setattr(engine, "_artifact_present", _absent)
    row = {
        "artifact_id": "art-b1",
        "content_hash": "h",
        "kind": "document",
        "metadata": {"filename": "b.md"},
    }
    asyncio.run(engine._restore_artifact(row, "p2"))
    assert [_who(r) for r in recorded] == [(expected, default_graph_id("p2"), "dave", "import")]
    assert recorded[0].title == "b.md"


def test_titles_come_from_the_name_the_scientist_gave() -> None:
    from umrs.knowledge.contributions import title_for

    assert title_for({"source_name": "run3.pdf"}, "knowledge/p/abc123") == "run3.pdf"
    assert title_for({}, "knowledge/p/abc123") == "abc123"


def test_backfill_finds_the_database_from_the_db_component_variables(monkeypatch) -> None:
    """The umrs-front containers set DB_HOST / UMRS_JOBS_DB / DB_SUPERUSER*
    only; the backfill read UMRS_DATA_CATALOG_DB / DATABASE_URL alone and
    refused to run there (2026-10-25)."""
    from umrs.knowledge.contribution_backfill import _db_url

    for var in ("UMRS_DATA_CATALOG_DB", "DATABASE_URL", "DB_PORT", "POSTGRES_PORT", "POSTGRES_DB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DB_HOST", "pg.example")
    monkeypatch.setenv("UMRS_JOBS_DB", "umrs_db")
    monkeypatch.setenv("DB_SUPERUSER", "u")
    monkeypatch.setenv("DB_SUPERUSER_PASSWORD", "p")

    assert _db_url(None) == "postgresql://u:p@pg.example:5432/umrs_db"
    assert _db_url("postgresql://x/y") == "postgresql://x/y"
