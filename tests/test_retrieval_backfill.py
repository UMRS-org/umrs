"""ADR-093 C6: ``knowledge_nodes.search_text`` — the writers and the backfill.

New chunks get their full body from the embed job's chunk writer
(``_write_embedded_chunks`` through ``KnowledgeStore.upsert_node``); existing
chunks get it from ``python -m umrs.knowledge.retrieval.backfill``, which
writes only md5-verified text and never re-embeds.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any

import pytest
from umrs.knowledge import store as store_mod
from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.retrieval import backfill
from umrs.knowledge.store import KnowledgeNode, KnowledgeStore

_TAG = re.compile(r"/\*\s*([\w:]+)\s*\*/")
DOC = "Alpha beta gamma. Delta epsilon zeta. Eta theta."


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── a routing psycopg2-style fake ────────────────────────────────────────────


class _Cursor:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool
        self._rows: list[tuple] = []
        self.rowcount = -1

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        match = _TAG.search(sql)
        tag = match.group(1) if match else "?"
        self._pool.calls.append((tag, sql, list(params or [])))
        if tag == "backfill:write":
            self._pool.writes.append(tuple(params))
            self.rowcount = 1
            return
        if tag == "backfill:reset":
            self._pool.resets.append(tuple(params))
            self.rowcount = 1
            return
        self._rows = list(self._pool.answers.get(tag, []))

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple]:
        return list(self._rows)


class _Conn:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._pool.commits += 1
        return False

    def cursor(self) -> _Cursor:
        return _Cursor(self._pool)


class _Pool:
    def __init__(
        self, rows: list[tuple], *, graph: str | None = None, filled: list[tuple] = ()
    ) -> None:
        self.answers: dict[str, Any] = {
            "backfill:graph": [(graph,)] if graph else [],
            "backfill:counts": [(len(rows) + 1, 1)],
            "backfill:rows": rows,
            "backfill:filled": list(filled),
        }
        self.calls: list[tuple[str, str, list[Any]]] = []
        self.writes: list[tuple] = []
        self.resets: list[tuple] = []
        self.commits = 0

    def getconn(self) -> _Conn:
        return _Conn(self)

    def putconn(self, _conn: _Conn) -> None:
        return None

    def call(self, tag: str) -> tuple[str, list[Any]]:
        for got, sql, params in self.calls:
            if got == tag:
                return sql, params
        raise AssertionError(f"no {tag} statement")


def _row(
    node_id: str,
    start: int,
    end: int,
    *,
    artifact: str = "art-1",
    md5: str | None = None,
    text: str | None = None,
) -> tuple:
    content = {"kind": "chunk", "artifact_id": artifact, "chunk_start": start, "chunk_end": end}
    content["chunk_md5"] = md5 if md5 is not None else _md5(DOC[start:end])
    if text is not None:
        content["text"] = text
    ref = f"projects/p/knowledge/artifacts/{artifact}" if artifact else None
    return (
        node_id,
        json.dumps(content),
        artifact or None,
        ref,
        "document" if artifact else None,
        {},
    )


class _Loader:
    def __init__(self, text: str | None = DOC) -> None:
        self.text = text
        self.calls: list[str] = []

    def __call__(self, artifact: dict[str, Any]) -> str | None:
        self.calls.append(artifact["artifact_id"])
        return self.text


# ── backfill ─────────────────────────────────────────────────────────────────


def test_backfill_writes_md5_verified_artifact_slices_reading_each_artifact_once() -> None:
    pool, loader = _Pool([_row("c0", 0, 17), _row("c1", 18, 36)]), _Loader()
    report = backfill.run_backfill(pool, "p", load_document=loader)
    assert loader.calls == ["art-1"]
    assert pool.writes == [
        (DOC[0:17], "c0", _md5(DOC[0:17])),
        (DOC[18:36], "c1", _md5(DOC[18:36])),
    ]
    counts = report.to_dict()
    assert counts["from_artifact"] == 2 and counts["written"] == 2
    assert counts["chunks"] == 3 and counts["already"] == 1 and counts["artifacts_read"] == 1


def test_backfill_never_writes_a_mismatching_text() -> None:
    rows = [_row("c0", 0, 17, md5="0" * 32), _row("c1", 18, 36, text="stale", md5="1" * 32)]
    pool = _Pool(rows)
    report = backfill.run_backfill(pool, "p", load_document=_Loader())
    assert pool.writes == []
    assert report.outcomes == {"from_node": 0, "from_artifact": 0, "mismatch": 2, "unavailable": 0}


def test_backfill_prefers_the_nodes_own_text_and_skips_the_fetch() -> None:
    body = "a legacy canonized chunk"
    pool, loader = _Pool([_row("c0", 0, 5, artifact="", md5=_md5(body), text=body)]), _Loader()
    report = backfill.run_backfill(pool, "p", load_document=loader)
    assert pool.writes == [(body, "c0", _md5(body))]
    assert report.outcomes["from_node"] == 1 and loader.calls == []


def test_backfill_counts_chunks_with_nothing_to_verify_as_unavailable() -> None:
    rows = [
        _row("c0", 0, 17, artifact=""),  # rag.upload chunk: no artifact to slice
        _row("c1", 0, 17, md5=""),  # no md5 to verify against
        _row("c2", 18, 36, artifact="art-gone"),
    ]
    pool = _Pool(rows)
    report = backfill.run_backfill(pool, "p", load_document=_Loader(text=None))
    assert pool.writes == []
    assert report.outcomes["unavailable"] == 3


def test_backfill_dry_run_counts_and_writes_nothing() -> None:
    pool = _Pool([_row("c0", 0, 17), _row("c1", 18, 36, md5="f" * 32)])
    report = backfill.run_backfill(pool, "p", load_document=_Loader(), dry_run=True)
    assert pool.writes == [] and "backfill:write" not in [t for t, _, _ in pool.calls]
    counts = report.to_dict()
    assert counts["dry_run"] is True and counts["from_artifact"] == 1 and counts["mismatch"] == 1
    assert counts["written"] == 0


def test_backfill_update_rechecks_null_and_md5_and_commits_per_batch() -> None:
    pool = _Pool([_row("c0", 0, 17), _row("c1", 18, 36), _row("c2", 37, 47)])
    backfill.run_backfill(pool, "p", load_document=_Loader(), batch_size=2)
    sql, _ = pool.call("backfill:write")
    assert "search_text IS NULL" in sql and "content->>'chunk_md5' = %s" in sql
    assert len(pool.writes) == 3
    assert pool.commits == 1 + 2  # the read, then two write batches


def test_backfill_reads_only_unfilled_chunks_of_the_project() -> None:
    pool = _Pool([])
    backfill.run_backfill(pool, "p", load_document=_Loader())
    sql, params = pool.call("backfill:rows")
    assert "n.search_text IS NULL" in sql and "n.content->>'kind' = 'chunk'" in sql
    assert params == ["p"]
    assert "backfill:graph" not in [t for t, _, _ in pool.calls]


def test_backfill_graph_scope_uses_the_legacy_default_graph_rule() -> None:
    pool = _Pool([], graph=default_graph_id("p"))
    report = backfill.run_backfill(pool, "p", load_document=_Loader(), graph="p")
    assert report.graph_id == default_graph_id("p")
    _, params = pool.call("backfill:graph")
    assert params == ["p", "p", "p", "p", "p"]
    sql, params = pool.call("backfill:rows")
    assert "(n.graph_id = %s OR (n.graph_id = '' AND %s))" in sql
    assert params == ["p", default_graph_id("p"), True]


def test_backfill_cli_prints_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = _Pool([_row("c0", 0, 17)])
    pool.closeall = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(backfill, "_build_pool", lambda: pool)
    monkeypatch.setattr(backfill, "_s3_document_loader", lambda: _Loader())
    assert backfill.main(["--project", "p", "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["from_artifact"] == 1 and printed["dry_run"] is True and printed["written"] == 0


# ── headed chunks (sent-v1): header, newline, verified body ──────────────────

HEADER = "Title \u203a Intro"


def _headed_row(node_id: str, start: int, end: int, header: str, header_chars: int) -> tuple:
    row = list(_row(node_id, start, end))
    content = json.loads(row[1])
    content.update(header=header, header_chars=header_chars, chunk_scheme="sent-v1")
    row[1] = json.dumps(content)
    return tuple(row)


def test_backfill_writes_a_headed_chunk_as_header_plus_body() -> None:
    """hydrate strips ``header_chars`` before its md5 check: a body-only text
    would fail it on every read (review finding data-deploy-7)."""
    from umrs.knowledge.retrieval import hydrate

    pool = _Pool([_headed_row("c0", 0, 17, HEADER, len(HEADER) + 1)])
    backfill.run_backfill(pool, "p", load_document=_Loader())
    [(text, node_id, md5)] = pool.writes
    content = json.loads(pool.answers["backfill:rows"][0][1])
    assert (text, node_id, md5) == (f"{HEADER}\n{DOC[0:17]}", "c0", _md5(DOC[0:17]))
    assert hydrate.stored_text(content, text) == DOC[0:17]


def test_backfill_never_writes_a_header_its_header_chars_disown() -> None:
    pool = _Pool([_headed_row("c0", 0, 17, HEADER, len(HEADER) + 5)])
    report = backfill.run_backfill(pool, "p", load_document=_Loader())
    assert pool.writes == [] and report.outcomes["mismatch"] == 1


# ── --verify: a filled row its md5 disowns is repaired or cleared ────────────


def _filled(node_id: str, search_text: str, *, header: str | None = None) -> tuple:
    """A chunk over DOC[0:17] that already has ``search_text``."""
    row = _headed_row(node_id, 0, 17, header, len(header) + 1) if header else _row(node_id, 0, 17)
    return (*row, search_text)


def test_verify_repairs_a_stale_text_from_the_verified_slice() -> None:
    """Content re-chunked under the same node id by a writer that kept the old
    column (review finding data-deploy-5): the lexical lanes would rank the
    chunk for words it no longer holds."""
    pool = _Pool([], filled=[_filled("c0", "perovskite humidity")])
    report = backfill.run_backfill(pool, "p", load_document=_Loader(), verify=True)
    assert pool.resets == [(DOC[0:17], "c0", "perovskite humidity", _md5(DOC[0:17]))]
    counts = report.to_dict()
    assert (counts["verify"], counts["stale"], counts["repaired"], counts["cleared"]) == (
        True,
        1,
        1,
        0,
    )


def test_verify_clears_a_stale_text_nothing_verified_can_replace() -> None:
    pool = _Pool([], filled=[_filled("c0", "perovskite humidity")])
    report = backfill.run_backfill(pool, "p", load_document=_Loader(None), verify=True)
    assert pool.resets == [(None, "c0", "perovskite humidity", _md5(DOC[0:17]))]
    assert (report.stale, report.repaired, report.cleared) == (1, 0, 1)


def test_verify_keeps_texts_that_honour_the_column_contract() -> None:
    rows = [_filled("c0", DOC[0:17]), _filled("c1", f"{HEADER}\n{DOC[0:17]}", header=HEADER)]
    pool = _Pool([], filled=rows)
    report = backfill.run_backfill(pool, "p", load_document=_Loader(), verify=True)
    assert pool.resets == [] and report.stale == 0


def test_verify_rewrites_a_header_the_node_no_longer_records() -> None:
    old = "Title \u203a Outr"  # same length: the body still verifies
    stale = f"{old}\n{DOC[0:17]}"
    pool = _Pool([], filled=[_filled("c1", stale, header=HEADER)])
    backfill.run_backfill(pool, "p", load_document=_Loader(), verify=True)
    assert pool.resets == [(f"{HEADER}\n{DOC[0:17]}", "c1", stale, _md5(DOC[0:17]))]


def test_verify_reset_rechecks_the_text_it_read_and_the_md5() -> None:
    pool = _Pool([], filled=[_filled("c0", "perovskite humidity")])
    backfill.run_backfill(pool, "p", load_document=_Loader(), verify=True)
    sql, _ = pool.call("backfill:reset")
    assert "search_text = %s" in sql.split("WHERE", 1)[1]
    assert "content->>'chunk_md5' IS NOT DISTINCT FROM %s" in sql
    read, params = pool.call("backfill:filled")
    assert "n.search_text IS NOT NULL" in read and params == ["p"]


def test_verify_dry_run_counts_and_writes_nothing() -> None:
    pool = _Pool([], filled=[_filled("c0", "perovskite humidity")])
    report = backfill.run_backfill(pool, "p", load_document=_Loader(), verify=True, dry_run=True)
    assert pool.resets == [] and (report.stale, report.repaired) == (1, 1)


def test_without_verify_filled_rows_are_never_read() -> None:
    pool = _Pool([], filled=[_filled("c0", "perovskite humidity")])
    report = backfill.run_backfill(pool, "p", load_document=_Loader())
    assert "backfill:filled" not in [t for t, _, _ in pool.calls]
    assert pool.resets == [] and report.to_dict()["verify"] is False


def test_backfill_cli_verify_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pool = _Pool([], filled=[_filled("c0", "perovskite humidity")])
    pool.closeall = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(backfill, "_build_pool", lambda: pool)
    monkeypatch.setattr(backfill, "_s3_document_loader", lambda: _Loader())
    assert backfill.main(["--project", "p", "--verify"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["stale"] == 1 and printed["repaired"] == 1 and len(pool.resets) == 1


# ── the writer: KnowledgeStore.upsert_node ───────────────────────────────────


class _AsyncConn:
    def __init__(self, column: Any) -> None:
        self.column = column
        self.probes = 0
        self.executed: list[tuple[str, tuple]] = []

    async def fetchval(self, _sql: str, *_args: Any) -> Any:
        self.probes += 1
        return self.column

    async def execute(self, sql: str, *params: Any) -> str:
        self.executed.append((sql, params))
        return "INSERT 0 1"


def _acquire(conn: _AsyncConn) -> Any:
    class _Ctx:
        async def __aenter__(self) -> _AsyncConn:
            return conn

        async def __aexit__(self, *exc: Any) -> None:
            return None

    return _Ctx()


class _AsyncPool:
    def __init__(self, conn: _AsyncConn) -> None:
        self._conn = conn

    def acquire(self) -> Any:
        return _acquire(self._conn)


class _SlotPool:
    """asyncpg's ``Pool``: ``__slots__`` without ``__weakref__``."""

    __slots__ = ("_closing", "_conn")

    def __init__(self, conn: _AsyncConn) -> None:
        self._conn = conn
        self._closing = False

    def acquire(self) -> Any:
        return _acquire(self._conn)

    def is_closing(self) -> bool:
        return self._closing


def _chunk_node(text: str | None, body: str = "body") -> KnowledgeNode:
    return KnowledgeNode(
        node_id="node-chunk-x-0",
        project_id="p",
        label="chunk 0",
        node_type="entity",
        content={"kind": "chunk", "chunk_md5": _md5(body)},
        content_hash=_md5(body),
        search_text=text,
    )


def _concept_node() -> KnowledgeNode:
    return KnowledgeNode(
        node_id="node-concept-x",
        project_id="p",
        label="perovskite",
        node_type="concept",
        content={"kind": "concept"},
    )


def _upsert(pool: Any, node: KnowledgeNode) -> None:
    asyncio.run(KnowledgeStore(pool).upsert_node(node))


@pytest.fixture(autouse=True)
def _fresh_probe_memo() -> Any:
    store_mod._search_text_probes.clear()
    yield
    store_mod._search_text_probes.clear()


def test_upsert_node_writes_a_given_text_whole() -> None:
    conn = _AsyncConn(column=True)
    _upsert(_AsyncPool(conn), _chunk_node("body"))
    sql, params = conn.executed[-1]
    assert sql == store_mod._UPSERT_NODE_TEXT_SQL
    assert len(params) == 13 and params[12] == "body"
    assert "WHEN $13::text IS NOT NULL THEN $13::text" in sql


def test_a_non_chunk_upsert_never_touches_the_column() -> None:
    conn = _AsyncConn(column=True)
    _upsert(_AsyncPool(conn), _concept_node())
    sql, params = conn.executed[-1]
    assert "search_text" not in sql and len(params) == 12
    assert conn.probes == 0  # non-chunk writers pay nothing


def test_a_chunk_upsert_without_text_keeps_only_text_its_content_still_owns() -> None:
    """A writer that replaces a chunk's content without text (bundle import
    in degraded mode, the canonize/upload mints' re-chunk) used to keep the
    old text for the new chunk_md5 (review finding data-deploy-5)."""
    conn = _AsyncConn(column=True)
    _upsert(_AsyncPool(conn), _chunk_node(None))
    sql, params = conn.executed[-1]
    assert sql == store_mod._UPSERT_NODE_TEXT_SQL and params[12] is None
    keep = sql[sql.index("search_text = CASE") :]
    for key in ("chunk_md5", "header", "header_chars"):
        assert (
            f"knowledge_nodes.content->'{key}' IS NOT DISTINCT FROM EXCLUDED.content->'{key}'"
            in keep
        )
    assert keep.rstrip().endswith("THEN knowledge_nodes.search_text\n       END")


def test_upsert_node_on_a_database_without_0120_omits_the_column() -> None:
    conn = _AsyncConn(column=False)
    _upsert(_AsyncPool(conn), _chunk_node("body"))
    _upsert(_AsyncPool(conn), _chunk_node(None))
    for sql, params in conn.executed:
        assert "search_text" not in sql and len(params) == 12


def test_search_text_probe_is_remembered_per_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    present, absent = _AsyncConn(column=True), _AsyncConn(column=False)
    present_pool, absent_pool = _AsyncPool(present), _AsyncPool(absent)
    for _ in range(3):
        _upsert(present_pool, _chunk_node("body"))
        _upsert(absent_pool, _chunk_node("body"))
    assert present.probes == 1 and absent.probes == 1
    clock = [store_mod.time.monotonic() + store_mod._SEARCH_TEXT_RECHECK_S + 1]
    monkeypatch.setattr(store_mod.time, "monotonic", lambda: clock[0])
    absent.column = True  # migration 0120 applied while the process runs
    _upsert(absent_pool, _chunk_node("body"))
    assert absent.probes == 2 and "search_text" in absent.executed[-1][0]


def test_the_probe_is_remembered_for_an_asyncpg_pool_without_weak_references(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """asyncpg's Pool cannot be weakly referenced: the memo used to fail and
    re-probe (and log) before every chunk write (review finding
    data-deploy-8)."""
    present, absent = _AsyncConn(column=True), _AsyncConn(column=False)
    present_pool, absent_pool = _SlotPool(present), _SlotPool(absent)
    with caplog.at_level("INFO", logger=store_mod.logger.name):
        for _ in range(3):
            _upsert(present_pool, _chunk_node("body"))
            _upsert(absent_pool, _chunk_node("body"))
    assert present.probes == 1 and absent.probes == 1
    assert caplog.text.count("search_text absent") == 1


def test_a_closed_pool_leaves_the_memo() -> None:
    first, second = _SlotPool(_AsyncConn(column=True)), _SlotPool(_AsyncConn(column=True))
    _upsert(first, _chunk_node("body"))
    first._closing = True  # the job's `async with _open_pool()` closed it
    _upsert(second, _chunk_node("body"))
    held = [entry[0] for entry in store_mod._search_text_probes.values()]
    assert held == [second]


def test_the_memo_is_bounded() -> None:
    pools = [_SlotPool(_AsyncConn(column=True)) for _ in range(store_mod._SEARCH_TEXT_MEMO_MAX + 5)]
    for pool in pools:
        _upsert(pool, _chunk_node("body"))
    assert len(store_mod._search_text_probes) == store_mod._SEARCH_TEXT_MEMO_MAX
    assert [entry[0] for entry in store_mod._search_text_probes.values()][-1] is pools[-1]


# ── the guard on a real Postgres (skipped without a database URL) ───────────


class _OneConn:
    """``KnowledgeStore``'s pool surface over one connection in a transaction."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> Any:
        return _acquire(self._conn)


async def _stored_text(conn: Any) -> Any:
    return await conn.fetchval(
        "SELECT search_text FROM knowledge_nodes WHERE node_id = $1", "node-chunk-x-0"
    )


async def _guard_steps(conn: Any) -> list[Any]:
    """Insert with text, then upserts without text: same md5, new md5, new text."""
    from umrs import migrations

    await conn.execute(
        (migrations.MIGRATIONS_DIR / "0120_knowledge_chunk_search_text.sql").read_text()
    )
    store, seen = KnowledgeStore(_OneConn(conn)), []
    for node in (
        _chunk_node("old body", body="old body"),
        _chunk_node(None, body="old body"),
        _chunk_node(None, body="new body"),
        _chunk_node("new body", body="new body"),
    ):
        await store.upsert_node(node)
        seen.append(await _stored_text(conn))
    return seen


async def _on_postgres(url: str) -> list[Any]:
    asyncpg = pytest.importorskip("asyncpg")
    conn = await asyncpg.connect(url)
    try:
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )
        transaction = conn.transaction()
        await transaction.start()
        try:
            return await _guard_steps(conn)
        finally:
            await transaction.rollback()
    finally:
        await conn.close()


@pytest.mark.integration
def test_the_upsert_guard_on_postgres(request: pytest.FixtureRequest) -> None:
    url = request.config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")
    seen = asyncio.run(_on_postgres(url))
    assert seen == ["old body", "old body", None, "new body"]


# ── the writer: the embed job's chunk nodes ──────────────────────────────────


class _ChunkStore:
    def __init__(self) -> None:
        self.nodes: list[KnowledgeNode] = []

    async def upsert_node(self, node: KnowledgeNode) -> None:
        self.nodes.append(node)

    async def upsert_edge(self, *_a: Any, **_k: Any) -> None:
        return None

    async def upsert_node_embedding(self, *_a: Any, **_k: Any) -> None:
        return None


def test_embed_job_stores_each_chunks_full_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/srv/models/e5")
    long_body = "x" * 900  # longer than the 500-character excerpt
    embedded = {
        "chunks": [
            {"text": long_body, "start": 0, "end": 900},
            {"text": "tail", "start": 901, "end": 905},
        ],
        "vectors": [[0.1, 0.2]] * 2,
    }
    store = _ChunkStore()
    asyncio.run(
        ka._write_embedded_chunks(store, "art-1", "p", "", embedded, ka.KnowledgeEmbedResult())
    )
    assert [n.search_text for n in store.nodes] == [long_body, "tail"]
    for node in store.nodes:
        assert _md5(node.search_text or "") == node.content["chunk_md5"]
