"""ADR-093 wave B review fixes on the reindex sweep and the chunk writer.

- chunk-data-1: a re-embed descriptor carries the artifact's ``kind`` and
  ``metadata`` (its source name decides PDF/PPTX extraction); a zip
  container the extractor did not open is refused by name, never chunked
  as its bytes; one artifact's failed write fails that artifact alone.
- chunk-data-4: the sweep reads the chunk rows, not only the per-artifact
  stamp (one graph's success hid another graph's stale chunks).
- chunk-data-5: a reindex sweeps the job's own project only.
- chunk-data-6: the reindex finds an artifact's graphs in its own project,
  and the writer re-embeds into the id scope the artifact's file node
  carries (an imported named-graph bundle), never a chunk without its
  part_of target.
- chunk-data-10: a NUL in a chunk body never reaches a TEXT/JSONB column.
- chunk-data-11: a retry goes past a batch that is already enqueued.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest
from tests.s3_ranged import ranged
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs import reindex
from umrs.jobs.knowledge_adapters import KnowledgeEmbedResult
from umrs.jobs.reindex import KnowledgeReindexBackend
from umrs.jobs.store import DuplicateJobError
from umrs.knowledge.embedding_metadata import (
    EmbeddingMetadataError,
    select_stale_artifacts,
)
from umrs.knowledge.graph_registry import default_graph_id

MODEL = "/srv/models/e5-base-v2"
SPEC = {
    "target_model": "e5-base-v2",
    "index_name": "unused",
    "database": "umrs_db",
    "table": "knowledge_node_embeddings",
    "batch_limit": 50,
}


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", MODEL)
    for name in ("UMRS_KNOWLEDGE_CHUNK_SCHEME", "UMRS_EMBEDDING_SANDBOX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("UMRS_KNOWLEDGE_EMBED_MAX_CHUNKS", raising=False)


# ── the reindex job ─────────────────────────────────────────────────────────


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


class _Service:
    """A job service; ``duplicate`` names graph ids already enqueued."""

    def __init__(self, duplicate: frozenset[str] = frozenset()):
        self.duplicate = duplicate
        self.submitted: list[Any] = []

    def submit_job(self, submission):
        if submission.spec["graph_id"] in self.duplicate:
            raise DuplicateJobError("job-already-there")
        self.submitted.append(submission)
        return f"job-{len(self.submitted)}"


def _job(project: str = "A", spec: dict | None = None):
    return SimpleNamespace(
        job_id="j-1", spec=spec or dict(SPEC), user_id="u", priority=0, project_id=project
    )


def _wire(monkeypatch, cur, service: _Service | None = None) -> _Service:
    @contextlib.contextmanager
    def scope():
        yield _conn(cur)

    service = service or _Service()
    monkeypatch.setattr(reindex, "_artifact_conn", scope)
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: service)
    return service


def _submit(monkeypatch, cur, spec=None, project="A", service=None):
    service = _wire(monkeypatch, cur, service)
    attempt = SimpleNamespace(attempt_id="a")
    ref = asyncio.run(KnowledgeReindexBackend().submit_job(_job(project, spec), attempt))
    return ref, service.submitted


ROW_PPTX = ("art-1", "A", "k1", "document", {"source_name": "talk.pptx", "size": 9})


def test_a_reembed_descriptor_carries_kind_and_source_name(monkeypatch):
    """chunk-data-1: without the name a PPTX deck was read as its zip bytes."""
    cur = _Cur([[("art-1",)], [ROW_PPTX]])
    _ref, submitted = _submit(monkeypatch, cur)
    assert submitted[0].spec["sources"] == [
        {
            "artifact_id": "art-1",
            "source_ref": "k1",
            "kind": "document",
            "metadata": {"source_name": "talk.pptx"},
        }
    ]


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [('{"source_name": "a.pdf"}', {"source_name": "a.pdf"}), ("not json", {}), (None, {})],
)
def test_a_text_metadata_column_is_read_as_json(metadata, expected):
    row = ("art-1", "A", "k1", "document", metadata)
    assert reindex._descriptor(row)["metadata"] == expected


def test_a_reindex_sweeps_the_jobs_project_when_its_spec_names_none(monkeypatch):
    """chunk-data-5: a spec without ``project_id`` swept every project."""
    cur = _Cur([[("art-1",)], [ROW_PPTX]])
    _ref, submitted = _submit(monkeypatch, cur)
    sweep_sql, sweep_params = cur.statements[1]  # [0] is the pass lock (wave F)
    assert "AND project_id = %s AND NOT EXISTS (SELECT 1 FROM jobs j" in sweep_sql
    assert sweep_sql.endswith("ORDER BY artifact_id LIMIT %s")
    assert sweep_params[-1] == 50 and "A" in sweep_params
    assert cur.statements[2][1] == (["art-1"], "A")  # the rows: the job's project
    assert {s.project_id for s in submitted} == {"A"}


def test_a_spec_naming_another_project_is_refused_before_any_sweep(monkeypatch):
    cur = _Cur([[("art-b",)]])
    with pytest.raises(EmbeddingMetadataError, match="not the job's project"):
        _submit(monkeypatch, cur, {**SPEC, "project_id": "B"})
    assert cur.statements == []


def test_a_job_without_a_project_is_refused(monkeypatch):
    cur = _Cur([[("art-b",)]])
    with pytest.raises(EmbeddingMetadataError, match="has none"):
        _submit(monkeypatch, cur, project="")
    assert cur.statements == []


def test_the_graphs_of_an_artifact_are_read_in_its_own_project(monkeypatch):
    """chunk-data-6 (b): ``content.artifact_id`` is node data; another
    project's node naming the artifact must not add a target."""
    cur = _Cur([[("art-1",)], [ROW_PPTX], [("art-1", "g-x")]])
    _ref, submitted = _submit(monkeypatch, cur)
    graphs_sql, graphs_params = cur.statements[3]
    assert "AND project_id = %s AND content->>'artifact_id' = ANY(%s)" in graphs_sql
    assert graphs_params == ("A", ["art-1"])
    assert [s.spec["graph_id"] for s in submitted] == ["g-x"]


def test_a_retry_goes_past_a_batch_that_is_already_enqueued(monkeypatch):
    """chunk-data-11: the first duplicate ended every retry."""
    service = _Service(duplicate=frozenset({"g-one"}))
    _wire(monkeypatch, _Cur(), service)
    batches = {("A", "g-one"): [{"artifact_id": "a1"}], ("A", "g-two"): [{"artifact_id": "a2"}]}
    ids = reindex._enqueue_reembed(_job(), batches)
    assert ids == ["job-already-there", "job-1"]
    assert [s.spec["graph_id"] for s in service.submitted] == ["g-two"]


# ── the sweep (chunk-data-4) ────────────────────────────────────────────────

_CHUNK_ROWS_CLAUSE = (
    "(project_id, artifact_id) IN (SELECT n.project_id, n.content->>'artifact_id' "
    "FROM knowledge_nodes n WHERE n.content->>'kind' = 'chunk' "
    "AND COALESCE(n.content->>'chunk_scheme', %s) <> %s AND n.project_id = %s)"
)


def test_the_sweep_reads_the_chunk_rows_not_only_the_artifact_stamp():
    """One graph's job stamping the artifact must not hide another graph's
    chunks of the old scheme."""
    cur = _Cur([[("art-1",)]])
    select_stale_artifacts(_conn(cur), "m", limit=9, chunk_scheme="sent-v1", project_id="p1")
    ((sql, params),) = cur.statements
    assert f"<> %s OR {_CHUNK_ROWS_CLAUSE})" in sql
    assert params == ("m", "fixed-v0", "sent-v1", "fixed-v0", "sent-v1", "p1", "p1", 9)


def test_the_unscoped_sweep_reads_every_projects_chunk_rows():
    cur = _Cur()
    select_stale_artifacts(_conn(cur), "m", chunk_scheme="sent-v1")
    ((sql, params),) = cur.statements
    assert _CHUNK_ROWS_CLAUSE.replace(" AND n.project_id = %s", "") in sql
    assert params == ("m", "fixed-v0", "sent-v1", "fixed-v0", "sent-v1", 500)


_IT = "c-d4-it"


def _seed_stamped_artifact(cur) -> None:
    """Artifact stamped current under sent-v1 (graph 1's job), while graph
    2 still holds a fixed-v0 chunk of it."""
    cur.execute(
        "INSERT INTO knowledge_artifacts (artifact_id, project_id, content_hash, kind, "
        "source_ref, embedding_model, embedding_status, metadata) VALUES "
        "(%s, %s, 'h', 'document', 'k', 'm', 'current', '{\"chunk_scheme\": \"sent-v1\"}')",
        (_IT, _IT),
    )
    cur.execute(
        "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
        "content_hash, graph_id) VALUES (%s, %s, 'c', 'entity', "
        "jsonb_build_object('kind', 'chunk', 'artifact_id', %s::text), 'h', 'g-two')",
        (f"node-chunk-g-two-{_IT}-0", _IT, _IT),
    )


@pytest.mark.integration
def test_a_stamped_artifact_with_an_old_chunk_in_another_graph_is_swept():
    url = os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("Need UMRS_TEST_DB_URL (a migrated test database)")
    psycopg2 = pytest.importorskip("psycopg2")
    conn = psycopg2.connect(url)
    try:
        _seed_stamped_artifact(conn.cursor())
        found = select_stale_artifacts(conn, "m", chunk_scheme="sent-v1", project_id=_IT)
    finally:
        conn.rollback()
        conn.close()
    assert found == [_IT]


# ── the chunk writer ────────────────────────────────────────────────────────


class _Conn:
    """asyncpg-shaped reads of the writer: the artifact rows (every artifact
    with a file node has one), the file-node lookups (``files``: node_id ->
    (graph_id, artifact_id)); writes are recorded. A transaction that raises
    rolls back the store's rows written inside it."""

    def __init__(self, files: dict[str, tuple[str, str]]):
        self.files = files
        self.executed: list[str] = []
        self.store: Any = None

    @contextlib.asynccontextmanager
    async def transaction(self):
        marks = (len(self.store.nodes), len(self.store.edges))
        try:
            yield self
        except BaseException:
            del self.store.nodes[marks[0] :], self.store.edges[marks[1] :]
            raise

    async def fetchval(self, sql, *args):
        if sql == ka._ARTIFACT_ROW_LOCK_SQL:
            return 1  # the artifact row exists
        assert sql == ka._FILE_NODE_SQL, sql
        return 1 if args[0] in self.files else None

    async def fetch(self, sql, *args):
        if sql != ka._FILE_NODES_SQL:
            return []  # the header labels
        _project, artifact, graphs = args
        return [(nid,) for nid, (g, a) in self.files.items() if a == artifact and g in graphs]

    async def execute(self, sql, *_args):
        self.executed.append(sql)
        return "DELETE 0" if sql.startswith("DELETE") else "UPDATE 1"


class _Store:
    def __init__(self, files: dict[str, tuple[str, str]], fail_for: str = ""):
        conn = _Conn(files)

        @contextlib.asynccontextmanager
        async def acquire():
            yield conn

        self._pool = SimpleNamespace(acquire=acquire)
        self.conn, self.fail_for = conn, fail_for
        conn.store = self
        self.nodes: list[Any] = []
        self.edges: list[Any] = []

    async def upsert_node(self, node, **_k):
        if self.fail_for and self.fail_for in node.node_id:
            raise ValueError(f"row refused: {node.node_id}")
        self.nodes.append(node)

    async def upsert_edge(self, edge, **_k):
        self.edges.append(edge)

    async def upsert_node_embedding(self, *_a, **_k):
        return None


def _embedded(texts=("first body", "second body")) -> dict:
    chunks = [{"text": t, "start": 0, "end": len(t)} for t in texts]
    return {"chunks": chunks, "vectors": [[1.0, 0.0]] * len(chunks)}


def _write(store, graph_id: str) -> KnowledgeEmbedResult:
    result = KnowledgeEmbedResult()
    asyncio.run(ka._write_embedded_chunks(store, "art-1", "P", graph_id, _embedded(), result))
    return result


def test_an_imported_named_graph_artifact_is_rechunked_in_its_own_id_scope():
    """chunk-data-6 (a): the bundle restored ``node-file-g-src-art-1`` into
    graph ''; the reindex targets the default graph, whose writer scope is
    ''. The chunks must replace the imported ones and hang on that node."""
    store = _Store({"node-file-g-src-art-1": ("", "art-1")})
    _write(store, default_graph_id("P"))
    assert [n.node_id for n in store.nodes] == [
        "node-chunk-g-src-art-1-0",
        "node-chunk-g-src-art-1-1",
    ]
    assert {e.target_id for e in store.edges} == {"node-file-g-src-art-1"}


def test_the_writers_own_file_node_keeps_its_scope():
    store = _Store({"node-file-g-x-art-1": ("g-x", "art-1")})
    _write(store, "g-x")
    assert [n.node_id for n in store.nodes][:1] == ["node-chunk-g-x-art-1-0"]


class _Lane:
    scheme = "fixed-v0"

    def chunk_and_embed(self, texts):
        return [_embedded((f"body of {t}",)) for t in texts]


class _S3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def get_object(self, Bucket, Key, Range=""):  # noqa: N803 — boto3's names
        return ranged(self.objects[f"{Bucket}/{Key}"], Range)


def _sources(*ids: str) -> list[dict]:
    return [
        {"artifact_id": a, "source_ref": f"k/{a}", "kind": "document",
         "metadata": {"source_name": f"{a}.md"}}
        for a in ids
    ]  # fmt: skip


def _embed_all(store, ids=("art-1", "art-2", "art-3")) -> KnowledgeEmbedResult:
    s3 = _S3({f"b/k/{a}": f"Text of {a}.".encode() for a in ids})
    result = KnowledgeEmbedResult()
    asyncio.run(ka._embed_sources(store, _sources(*ids), "P", "", _Lane(), (s3, "b"), result))
    return result


def _files(*ids: str) -> dict[str, tuple[str, str]]:
    return {f"node-file-{a}": ("", a) for a in ids}


def test_a_failed_write_fails_that_artifact_alone():
    """chunk-data-1: one bad row failed the whole 16-artifact batch, and the
    artifacts after it were never written or stamped."""
    store = _Store(_files("art-1", "art-2", "art-3"), fail_for="art-2")
    result = _embed_all(store)
    assert [f["artifact_id"] for f in result.failed_items] == ["art-2"]
    assert sorted({n.content["artifact_id"] for n in store.nodes}) == ["art-1", "art-3"]
    stamps = [sql for sql in store.conn.executed if sql.startswith("UPDATE")]
    assert len(stamps) == 2


def test_an_artifact_without_its_file_node_writes_nothing_and_is_named():
    """chunk-data-6: chunk 0 was written, then its part_of edge broke the
    0055 FK, leaving a stray chunk on every sweep."""
    store = _Store(_files("art-1"))
    result = _embed_all(store, ids=("art-1", "art-2"))
    (failed,) = result.failed_items
    assert failed["artifact_id"] == "art-2" and "no single file node" in failed["error"]
    assert {n.content["artifact_id"] for n in store.nodes} == {"art-1"}


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", "<p:sld/>")
    return buffer.getvalue()


def test_a_zip_container_without_its_name_is_refused_not_chunked():
    """chunk-data-1: a PPTX descriptor without its source name came back
    as its zip bytes, and their NULs failed the batch's DB writes."""
    result = KnowledgeEmbedResult()
    src = {"artifact_id": "art-1", "source_ref": "k/deck"}
    text = ka._artifact_text(src, _S3({"b/k/deck": _zip_bytes()}), "b", result)
    assert text is None
    assert "zip container" in result.failed_items[0]["error"]


# ── NUL (chunk-data-10) ─────────────────────────────────────────────────────


def test_a_nul_in_a_text_artifact_is_dropped_before_chunking():
    result = KnowledgeEmbedResult()
    src = {"artifact_id": "a", "source_ref": "k/log", "metadata": {"source_name": "run.log"}}
    raw = b"Run started.\x00\x00 Step one done. " + b"x" * 600 + b"\x00 end."
    text = ka._artifact_text(src, _S3({"b/k/log": raw}), "b", result)
    assert text is not None and "\x00" not in text
    assert text.startswith("Run started. Step one done.")


@pytest.mark.parametrize("scheme", ["fixed-v0", "sent-v1"])
def test_a_body_holding_a_nul_stores_no_search_text_and_a_clean_excerpt(scheme):
    target = ka._ChunkTarget("a", "p", "g", "", scheme)
    body = "x" * 600 + "\x00y"
    chunk = {"text": body, "start": 0, "end": len(body), "header": "T"}
    content, search_text = ka._chunk_content(target, 0, chunk)
    assert search_text is None
    assert "\x00" not in content["excerpt"] and content["nul_in_body"] is True
    assert content["chunk_end"] == len(body)  # offsets and md5 keep the body
