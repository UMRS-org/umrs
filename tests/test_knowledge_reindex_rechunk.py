"""ADR-093 D-11 (C9): honest chunk writes and the same-model reindex repair.

- The embed writer records the scheme the worker cut with; a ``sent-v1``
  chunk stores ``header + newline + body`` in ``search_text`` with
  ``header_chars``, so hydrate's md5 rule still verifies the body, and its
  node ids keep the scheme the neighbour merge derives.
- After writing n chunks the writer deletes the artifact's chunks n.. in
  that scope (their edges and vectors cascade); the 128 cap stays and the
  stamp records ``chunks_total`` / ``chunks_indexed``.
- ``KNOWLEDGE_REINDEX`` sweeps chunk-scheme mismatches in the job's own
  project, re-embeds each artifact into the graphs it lives in, and no
  longer calls the index maintenance seam on the dimension-less 0075 column.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs import reindex
from umrs.jobs.knowledge_adapters import KnowledgeEmbedResult
from umrs.jobs.reindex import KnowledgeReindexBackend
from umrs.knowledge import embed_worker as ew
from umrs.knowledge.embedding_metadata import (
    EmbeddingMetadataError,
    select_stale_artifacts,
)
from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.retrieval.expand import neighbour_id
from umrs.knowledge.retrieval.hydrate import stored_text
from umrs.knowledge.retrieval.types import Candidate

MODEL = "/srv/models/e5-base-v2"
CAP_ENV = "UMRS_KNOWLEDGE_EMBED_MAX_CHUNKS"
SCHEME_ENV = "UMRS_KNOWLEDGE_CHUNK_SCHEME"


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", MODEL)
    for name in (CAP_ENV, SCHEME_ENV, "UMRS_EMBEDDING_SANDBOX"):
        monkeypatch.delenv(name, raising=False)


# ── fakes ───────────────────────────────────────────────────────────────────


class _Pool:
    """An asyncpg pool stand-in: records every statement; a DELETE answers
    ``delete_status``; ``fetch`` answers the artifact label rows."""

    def __init__(self, delete_status: str = "DELETE 0", rows: list | None = None):
        self.executed: list[tuple[str, tuple]] = []
        self.delete_status = delete_status
        self.rows = rows or []

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self

    @contextlib.asynccontextmanager
    async def transaction(self):
        yield self

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return self.delete_status if sql.startswith("DELETE") else "UPDATE 1"

    async def fetch(self, _sql, ids):
        return [row for row in self.rows if row[0] in ids]

    async def fetchval(self, _sql, *_args):
        return 1  # the writer's own file node exists

    def statements(self, verb: str) -> list[tuple[str, tuple]]:
        return [(sql, args) for sql, args in self.executed if sql.startswith(verb)]


class _Store:
    def __init__(self, pool: _Pool | None = None):
        self._pool = pool or _Pool()
        self.nodes: list[Any] = []
        self.edges: list[Any] = []
        self.vectors: list[str] = []

    async def upsert_node(self, node, **_k):
        self.nodes.append(node)

    async def upsert_edge(self, edge, **_k):
        self.edges.append(edge)

    async def upsert_node_embedding(self, node_id, *_a, **_k):
        self.vectors.append(node_id)


class _Embedder:
    client = SimpleNamespace(tokenizer=None)

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


def _prose(count: int, first: int = 0) -> str:
    return " ".join(f"W{first + i} a b c d e f." for i in range(count))


PAPER = f"# Paper\n\n{_prose(40)}\n\n## Methods\n\n{_prose(40, first=50)}"


def _sent_v1(text: str = PAPER, title: str = "Paper") -> dict:
    (result,) = ew.chunk_and_embed(
        _Embedder(),
        [text],
        prefix_model_name=MODEL,
        max_tokens=256,
        scheme="sent-v1",
        titles=[title],
    )
    return result


def _write(store, embedded, graph_id="", result=None):
    result = result or KnowledgeEmbedResult()
    asyncio.run(ka._write_embedded_chunks(store, "art-1", "p1", graph_id, embedded, result))
    return result


# ── the writer: sent-v1 content and the wave A seams ────────────────────────


def test_sent_v1_chunks_record_their_scheme_header_and_counts():
    store, embedded = _Store(), _sent_v1()
    _write(store, embedded)
    content = store.nodes[-1].content
    assert len(store.nodes) == len(embedded["chunks"]) > 1
    assert content["chunk_scheme"] == "sent-v1"
    assert (content["section"], content["header"]) == ("Methods", "Paper › Methods")
    assert content["header_chars"] == len("Paper › Methods\n")
    assert content["num_tokens"] == embedded["chunks"][-1]["num_tokens"]


def test_sent_v1_search_text_hydrates_to_the_body():
    """Wave A's md5 rule: ``search_text[header_chars:]`` is the body whose md5
    the node records, and that body is the artifact's own slice."""
    store = _Store()
    _write(store, _sent_v1())
    for node in store.nodes:
        body = stored_text(node.content, node.search_text)
        start, end = node.content["chunk_start"], node.content["chunk_end"]
        assert body == PAPER[start:end]
        assert node.search_text == f"{node.content['header']}\n{body}"


def test_sent_v1_node_ids_keep_the_neighbour_scheme():
    store = _Store()
    _write(store, _sent_v1(), graph_id="g-named")
    ids = [node.node_id for node in store.nodes]
    assert ids == [f"node-chunk-g-named-art-1-{i}" for i in range(len(ids))]
    middle = Candidate(node_id=ids[1], chunk_index=1)
    assert (neighbour_id(middle, 0), neighbour_id(middle, 2)) == (ids[0], ids[2])


def test_a_result_without_a_scheme_is_written_as_fixed_v0():
    store = _Store()
    _write(store, {"chunks": [{"text": "body", "start": 0, "end": 4}], "vectors": [[0.1, 0.2]]})
    (node,) = store.nodes
    assert node.content["chunk_scheme"] == "fixed-v0"
    assert "header_chars" not in node.content and node.search_text == "body"
    (_, args) = store._pool.statements("UPDATE")[-1]
    assert args[3:] == ("fixed-v0", 1, 1)


# ── the writer: orphans, the cap, the stamp ─────────────────────────────────


def test_a_rechunk_deletes_the_chunks_from_n_on(monkeypatch):
    """Three chunks written: the artifact's chunks 3.. in this scope go (by
    its part_of edges and by exact id), and the count is reported."""
    monkeypatch.setenv(CAP_ENV, "6")
    store = _Store(_Pool(delete_status="DELETE 2"))
    chunks = [{"text": t, "start": 0, "end": 1} for t in "abc"]
    result = _write(store, {"chunks": chunks, "vectors": [[0.1]] * 3}, graph_id="g-x")
    (sql, args), _vectors = store._pool.statements("DELETE")
    prefix = "node-chunk-g-x-art-1-"
    assert args == (
        "node-file-g-x-art-1",
        [f"{prefix}{i}" for i in (3, 4, 5)],
        "p1",
        "art-1",
        prefix,
        3,
    )
    assert "n.node_id = $5 || (n.content->>'chunk_index')" in sql  # never another scope
    assert result.chunks_removed == 2


def test_the_delete_runs_before_the_stamp():
    store = _Store()
    _write(store, _sent_v1())
    verbs = [" ".join(sql.split()[:3]) for sql, _ in store._pool.executed]
    assert verbs == [
        "DELETE FROM knowledge_nodes",
        "DELETE FROM knowledge_node_embeddings",
        "UPDATE knowledge_artifacts SET",
    ]


def test_the_cap_stays_and_the_stamp_records_total_and_indexed(monkeypatch):
    monkeypatch.setenv(CAP_ENV, "2")
    store = _Store()
    embedded = _sent_v1(_prose(150))  # six chunks
    result = _write(store, embedded)
    (_, stamp_args) = store._pool.statements("UPDATE")[-1]
    (_, delete_args) = store._pool.statements("DELETE FROM knowledge_nodes")[-1]
    assert len(store.vectors) == 2 and result.nodes_embedded == 2
    assert stamp_args == (MODEL, 2, "art-1", "sent-v1", len(embedded["chunks"]), 2)
    assert delete_args[-1] == 2 and result.chunks_truncated == 1


def test_nothing_is_deleted_or_stamped_without_chunks():
    store = _Store()
    result = _write(store, {"chunks": [], "vectors": []})
    assert store._pool.executed == [] and result.skipped == 1


@pytest.mark.parametrize(("status", "removed"), [("DELETE 7", 7), ("DELETE 0", 0), (None, 0)])
def test_deleted_rows_reads_the_command_status(status, removed):
    assert ka._deleted_rows(status) == removed


# ── the embed lane: the site scheme and the header titles ───────────────────


def test_the_site_scheme_reaches_the_in_process_lane(monkeypatch):
    monkeypatch.setenv(SCHEME_ENV, "sent-v1")
    monkeypatch.setattr(ka, "_get_node_embedder", lambda: _Embedder())
    lane, reason = ka._chunk_embedder_for_site()
    assert (lane.lane, lane.scheme, reason) == ("in_process", "sent-v1", "")


def test_the_site_scheme_reaches_the_sandbox_lane(monkeypatch):
    monkeypatch.setenv(SCHEME_ENV, "sent-v1")
    monkeypatch.setenv("UMRS_EMBEDDING_SANDBOX", "podman")
    monkeypatch.setenv("UMRS_EMBEDDING_SANDBOX_IMAGE", "localhost/umrs-server:t")
    lane, _ = ka._chunk_embedder_for_site()
    assert (lane.lane, lane.scheme) == ("sandbox", "sent-v1")


def test_an_unknown_site_scheme_fails_the_embed_by_name(monkeypatch):
    monkeypatch.setenv(SCHEME_ENV, "sentences")
    with pytest.raises(ValueError, match=SCHEME_ENV):
        ka._chunk_embedder_for_site()


class _Lane:
    def __init__(self, scheme: str):
        self.scheme = scheme
        self.calls: list[tuple] = []

    def chunk_and_embed(self, texts, *titles):
        self.calls.append((texts, *titles))
        return [{"chunks": [], "vectors": []} for _ in texts]


LABEL_ROWS = [
    ("art-1", "Meta title", "", "a.md", None),
    ("art-2", None, "", "b.md", None),
    ("art-3", None, "", None, "/stage/in/c.md"),
    ("art-4", None, "", "d.md", None),
]


def test_sent_v1_titles_are_metadata_then_leading_h1_then_source_name():
    """A later heading names a section, not the document (review
    chunk-data-8): ``b.md`` opens with prose, so its name is the title."""
    lane, store = _Lane("sent-v1"), _Store(_Pool(rows=LABEL_ROWS))
    prepared = [
        ({"artifact_id": "art-1"}, "# Heading one\nbody"),
        ({"artifact_id": "art-2"}, "intro\n## Heading two\nbody"),
        ({"artifact_id": "art-3"}, "no heading"),
        ({"artifact_id": "art-4"}, "# Heading four\nbody"),
    ]
    asyncio.run(ka._chunk_and_embed_batch(store, lane, prepared))
    ((texts, titles, _headings),) = lane.calls
    assert titles == ["Meta title", "b.md", "c.md", "Heading four"]


CODE_ROWS = [
    ("art-md", None, "", "notes.md", None),
    ("art-py", None, "", "plot.py", None),
    ("art-yaml", None, "", None, "/stage/in/env.yaml"),
    ("art-bare", None, "", "README", None),
]


def test_sent_v1_tells_the_lane_which_texts_have_markdown_headings():
    """A code, config or plain-text file's '# comment' never names a section
    (review chunk-data-8): the lane is told per text, by the file type."""
    lane, store = _Lane("sent-v1"), _Store(_Pool(rows=CODE_ROWS))
    prepared = [
        ({"artifact_id": "art-md"}, "# Notes\nbody"),
        ({"artifact_id": "art-py"}, "# helpers\nx = 1"),
        ({"artifact_id": "art-yaml"}, "# settings\na: 1"),
        ({"artifact_id": "art-bare"}, "# Readme\nbody"),
        ({"artifact_id": "art-gone", "metadata": {"source_name": "run.sh"}}, "# x\ny"),
    ]
    asyncio.run(ka._chunk_and_embed_batch(store, lane, prepared))
    ((_texts, titles, headings),) = lane.calls
    assert headings == [True, False, False, True, False]
    assert titles[1:3] == ["plot.py", "env.yaml"]  # a comment never titles them


def test_a_fixed_v0_lane_gets_texts_only():
    lane, store = _Lane("fixed-v0"), _Store(_Pool(rows=LABEL_ROWS))
    asyncio.run(ka._chunk_and_embed_batch(store, lane, [({"artifact_id": "art-1"}, "t")]))
    assert lane.calls == [(["t"],)]


# ── the sweep ───────────────────────────────────────────────────────────────


class _Cur:
    """``batches`` answer successive fetchall calls (then empty)."""

    def __init__(self, batches=()):
        self.statements: list[tuple[str, Any]] = []
        self._batches = [list(b) for b in batches]

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchall(self):
        return self._batches.pop(0) if self._batches else []

    def fetchone(self):
        return (True,)  # the project's pass lock is free (wave F)


def _conn(cur):
    return SimpleNamespace(cursor=lambda: cur)


def test_the_sweep_selects_scheme_mismatches_in_one_project():
    cur = _Cur([[("art-1",)]])
    ids = select_stale_artifacts(_conn(cur), "m", limit=9, chunk_scheme="sent-v1", project_id="p1")
    ((sql, params),) = cur.statements
    assert ids == ["art-1"]
    assert "OR COALESCE(metadata->>'chunk_scheme', %s) <> %s OR" in sql
    assert sql.endswith("AND project_id = %s ORDER BY artifact_id LIMIT %s")
    assert params == ("m", "fixed-v0", "sent-v1", "fixed-v0", "sent-v1", "p1", "p1", 9)


def test_the_sweep_without_scope_is_the_model_sweep():
    cur = _Cur()
    select_stale_artifacts(_conn(cur), "m")
    ((sql, params),) = cur.statements
    assert "chunk_scheme" not in sql and "project_id" not in sql
    assert params == ("m", 500)


# ── the reindex job ─────────────────────────────────────────────────────────

SPEC = {
    "target_model": "e5-base-v2",
    "index_name": "unused",
    "database": "umrs_db",
    "table": "knowledge_node_embeddings",
    "batch_limit": 50,
}


def _submit(monkeypatch, cur, spec):
    @contextlib.contextmanager
    def scope():
        yield _conn(cur)

    submitted: list = []

    class _Svc:
        def submit_job(self, submission):
            submitted.append(submission)
            return f"job-{len(submitted)}"

    monkeypatch.setattr(reindex, "_artifact_conn", scope)
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())
    job = SimpleNamespace(job_id="j-1", spec=spec, user_id="u", priority=0, project_id="p1")
    ref = asyncio.run(KnowledgeReindexBackend().submit_job(job, SimpleNamespace(attempt_id="a")))
    return ref, submitted


def _no_maintenance(monkeypatch):
    import umrs.shared.pgvector_index as pgvector_index

    def boom(*_a, **_k):
        raise AssertionError("no ANN index can exist on the 0075 column")

    monkeypatch.setattr(pgvector_index, "maintain_pgvector_index", boom)


def test_reindex_never_maintains_an_index_on_the_untyped_node_vectors(monkeypatch):
    """Until now the call failed every reindex after the re-embed jobs were
    already enqueued (and a retry enqueued them again)."""
    _no_maintenance(monkeypatch)
    monkeypatch.setenv(SCHEME_ENV, "sent-v1")
    cur = _Cur([[("art-1",)], [("art-1", "p1", "k1", "document", {})]])
    ref, submitted = _submit(monkeypatch, cur, {**SPEC, "project_id": "p1"})
    assert "swept=1|reembed_jobs=1" in ref and "scheme=sent-v1" in ref
    held = (list(reindex.HELD_BY.job_types), list(reindex.HELD_BY.statuses))
    sweep = (MODEL, "fixed-v0", "sent-v1", "fixed-v0", "sent-v1", "p1", "p1", *held, 50)
    assert cur.statements[0][0].startswith("SELECT pg_try_advisory_xact_lock")
    assert cur.statements[1][1] == sweep
    assert submitted[0].spec["graph_id"] == default_graph_id("p1")


def test_reindex_reembeds_each_artifact_into_the_graphs_it_lives_in(monkeypatch):
    _no_maintenance(monkeypatch)
    cur = _Cur(
        [
            [("art-1",), ("art-2",)],
            [("art-1", "p1", "k1", "document", {}), ("art-2", "p1", "k2", "code", None)],
            [("art-1", "g-named"), ("art-1", ""), ("art-2", "g-named")],
        ]
    )
    _, submitted = _submit(monkeypatch, cur, dict(SPEC))
    batches = {s.spec["graph_id"]: [x["artifact_id"] for x in s.spec["sources"]] for s in submitted}
    assert batches == {default_graph_id("p1"): ["art-1"], "g-named": ["art-1", "art-2"]}
    assert "node_type = 'file' OR content->>'kind' = 'chunk'" in cur.statements[3][0]


def test_a_spec_scheme_other_than_the_site_is_refused(monkeypatch):
    monkeypatch.setenv(SCHEME_ENV, "sent-v1")
    cur = _Cur()
    with pytest.raises(EmbeddingMetadataError, match="not this site's"):
        _submit(monkeypatch, cur, {**SPEC, "chunk_scheme": "fixed-v0"})
    assert cur.statements == []  # refused before any sweep


def test_an_unknown_spec_scheme_is_refused_at_submit():
    with pytest.raises(EmbeddingMetadataError, match="unknown chunk scheme"):
        reindex.validate_knowledge_reindex_spec({**SPEC, "chunk_scheme": "v2"})


# ── the orphan delete against Postgres (``--db-url`` / UMRS_TEST_DB_URL) ────

_IT_PROJECT = "c9-it"
_IT_ART = "c9-it-art"


class _ConnPool:
    def __init__(self, conn):
        self._conn = conn

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self._conn


async def _seed(conn, scope: str, count: int, artifact: str = _IT_ART) -> None:
    file_id = f"node-file-{scope}{artifact}"
    await conn.execute(
        "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
        "content_hash) VALUES ($1, $2, $1, 'file', jsonb_build_object('artifact_id', $3::text), "
        "'h') ON CONFLICT (node_id) DO NOTHING",
        file_id,
        _IT_PROJECT,
        artifact,
    )
    for i in range(count):
        chunk_id = f"node-chunk-{scope}{artifact}-{i}"
        await conn.execute(
            "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
            "content_hash) VALUES ($1, $2, $1, 'entity', jsonb_build_object('kind', 'chunk', "
            "'artifact_id', $3::text, 'chunk_index', $4::int), 'h')",
            chunk_id,
            _IT_PROJECT,
            artifact,
            i,
        )
        await conn.execute(
            "INSERT INTO knowledge_edges (edge_id, project_id, source_id, target_id, "
            "relation_type) VALUES ($1, $2, $3, $4, 'part_of')",
            f"edge-{chunk_id}",
            _IT_PROJECT,
            chunk_id,
            file_id,
        )
        await conn.execute(
            "INSERT INTO knowledge_node_embeddings (node_id, model_name, dimension, "
            "embedding, content_hash) VALUES ($1, 'm', 2, '[1,0]'::vector, 'h')",
            chunk_id,
        )


async def _left(conn) -> tuple[list[str], int, int]:
    ids = await conn.fetch(
        "SELECT node_id FROM knowledge_nodes WHERE project_id = $1 "
        "AND content->>'kind' = 'chunk' ORDER BY node_id",
        _IT_PROJECT,
    )
    edges = await conn.fetchval(
        "SELECT count(*) FROM knowledge_edges WHERE project_id = $1", _IT_PROJECT
    )
    vectors = await conn.fetchval(
        "SELECT count(*) FROM knowledge_node_embeddings WHERE node_id LIKE 'node-chunk-%c9-it-%'"
    )
    return [row[0] for row in ids], int(edges), int(vectors)


async def _orphan_delete_round(url: str):
    asyncpg = pytest.importorskip("asyncpg")
    conn = await asyncpg.connect(url)
    transaction = conn.transaction()
    await transaction.start()
    try:
        await _seed(conn, "", 5)
        await _seed(conn, "g-x-", 4)  # the same artifact in a named graph
        await _seed(conn, "", 4, artifact="c9-it-other")
        target = ka._ChunkTarget(_IT_ART, _IT_PROJECT, "", "", "sent-v1")
        removed = await ka._drop_chunks_from(conn, target, 2, 128)
        return removed, await _left(conn)
    finally:
        await transaction.rollback()
        await conn.close()


@pytest.mark.integration
def test_the_orphan_delete_takes_edges_and_vectors_and_nothing_else():
    url = os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("Need UMRS_TEST_DB_URL (a migrated test database)")
    removed, (ids, edges, vectors) = asyncio.run(_orphan_delete_round(url))
    assert removed == 3
    assert [i for i in ids if i.startswith("node-chunk-c9-it-art-")] == [
        "node-chunk-c9-it-art-0",
        "node-chunk-c9-it-art-1",
    ]
    assert len(ids) == 2 + 4 + 4 and edges == len(ids) and vectors == len(ids)
