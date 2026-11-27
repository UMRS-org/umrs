"""KN-S6: one partial HNSW per embedding space, and the vector-read fence.

ADR-083 D-2 / decision KN-3: every vector read carries a model filter, the
ANN reads carry exactly their space's index predicate, run on
``hnsw_<space_id>`` with an iterative scan, over-fetch ``max(k, 100)`` and
re-sort by exact distance. Acceptance (decisions-2026-11-24 §5, KN-S6):

- the fence fails a vector read without a model filter (unit, recording pool);
- ``EXPLAIN`` of the dense read uses ``hnsw_<space>`` (live Postgres);
- the dense lane's top-k equals the exact top-k (live Postgres);
- the calibrated null is recomputed after a reindex (live Postgres).

The live half runs against a CI-shaped database (``UMRS_TEST_DB_URL`` or
``POSTGRES_*``) with pgvector >= 0.8; it skips cleanly otherwise.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kn_s6_pending import pending
from space_registry_fake import REGISTRY_MARKER, registry_row
from umrs.api import knowledge_query as kq
from umrs.knowledge import embedding_space_index as esi
from umrs.knowledge import embedding_spaces as es
from umrs.knowledge.embedding_spaces import EmbeddingSpace, EmbeddingSpaceError
from umrs.knowledge.retrieval import ann, calibrate, dense
from umrs.knowledge.retrieval.types import RetrieveSpec

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs"
MODEL = "kns6/test-model"
SPACE = EmbeddingSpace(
    space_id="kns6_test",
    model_name=MODEL,
    dimension=2,
    legacy_names=("kns6/legacy-model",),
)


# ── the SQL builders (designer-implemented) ─────────────────────────────────


def test_the_index_ddl_is_a_partial_expression_hnsw_on_the_space_names():
    assert es.hnsw_index_ddl(SPACE) == (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS hnsw_kns6_test "
        "ON knowledge_node_embeddings USING hnsw "
        "(((embedding)::vector(2)) vector_cosine_ops) "
        "WHERE model_name IN ('kns6/test-model', 'kns6/legacy-model')"
    )


def test_the_read_filter_is_the_index_predicate_over_an_alias():
    assert es.space_filter_sql(SPACE) == "e." + es.index_predicate_sql(SPACE)
    assert es.space_filter_sql(SPACE, "x") == (
        "x.model_name IN ('kns6/test-model', 'kns6/legacy-model')"
    )


def test_drop_and_plain_forms():
    assert es.drop_index_ddl(SPACE) == "DROP INDEX CONCURRENTLY IF EXISTS hnsw_kns6_test"
    assert es.drop_index_ddl(SPACE, concurrently=False) == "DROP INDEX IF EXISTS hnsw_kns6_test"
    assert es.hnsw_index_ddl(SPACE, concurrently=False).startswith(
        "CREATE INDEX IF NOT EXISTS hnsw_kns6_test "
    )


@pytest.mark.parametrize("name", ["a'b", "a b", "a\\b", "", "a;b", "x" * 256])
def test_a_literal_refuses_anything_outside_the_name_pattern(name):
    with pytest.raises(EmbeddingSpaceError):
        es.sql_literal(name)


@pytest.mark.parametrize("alias", ["e.x", "E", "1e", "e; drop", ""])
def test_the_filter_refuses_an_alias_that_is_not_an_identifier(alias):
    with pytest.raises(EmbeddingSpaceError):
        es.space_filter_sql(SPACE, alias)


def test_an_invalid_space_never_reaches_ddl():
    from dataclasses import replace

    with pytest.raises(EmbeddingSpaceError):
        es.hnsw_index_ddl(replace(SPACE, space_id="x; DROP TABLE t"))
    with pytest.raises(EmbeddingSpaceError):
        es.space_filter_sql(replace(SPACE, legacy_names=("it's",)))


def test_the_overfetch_and_its_settings():
    assert [ann.overfetch(k) for k in (1, 100, 250)] == [100, 100, 250]
    assert ann.settings_sql(5000) == (
        "SET LOCAL hnsw.iterative_scan = relaxed_order; SET LOCAL hnsw.ef_search = 1000; "
    )


def test_an_ann_query_of_another_dimension_is_refused_before_sql():
    with pytest.raises(ValueError, match=ann.CODE_DIMENSION):
        ann.ann_query(SPACE, columns=(("e.node_id", "node_id"),), where="true",
                      where_params=(), vector=[1.0, 2.0, 3.0], k=5)


# ── the fence (recording pool) ──────────────────────────────────────────────

_MODEL_FILTER = re.compile(r"\be\.model_name\s*(=|IN\s*\()")


def has_model_filter(sql: str) -> bool:
    """A vector read over the 0075 table names the model it ranks."""
    if "<=>" not in sql or "knowledge_node_embeddings" not in sql:
        return True
    return bool(_MODEL_FILTER.search(sql))


def test_the_fence_rejects_a_vector_read_without_a_model_filter():
    bad = "SELECT e.node_id, e.embedding <=> %s FROM knowledge_node_embeddings e"
    assert not has_model_filter(bad)
    assert has_model_filter(bad + " WHERE e.model_name = %s")
    assert has_model_filter(bad + " WHERE " + es.space_filter_sql(SPACE))


class _Recorder:
    """A psycopg2-shaped pool and cursor that records every statement."""

    def __init__(self) -> None:
        self.sql: list[str] = []

    def getconn(self) -> _Recorder:
        return self

    def putconn(self, _conn: Any) -> None:
        return None

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def cursor(self) -> _Recorder:
        return self

    def execute(self, sql: str, _params: Any = None) -> None:
        self.sql.append(sql)
        self._rows: list[tuple] = []
        if REGISTRY_MARKER in sql:
            self._rows = [registry_row(SPACE.space_id, MODEL, 2, legacy_names=SPACE.legacy_names)]
        elif sql.startswith("SELECT count(*)"):
            self._rows = [(40, None)]

    def fetchall(self) -> list[tuple]:
        return list(self._rows)

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


def _spec(**kw: Any) -> RetrieveSpec:
    return RetrieveSpec(queries=("q",), project_id="p1", **kw)


def _every_vector_read() -> list[str]:
    """Run every knowledge read lane that ranks vectors; return their SQL."""
    rec = _Recorder()
    dense.dense_lane(rec, [[1.0, 0.0]], MODEL, _spec(graph_ids=("g",)), limit=5)
    dense.dense_lane(rec, [[1.0, 0.0]], MODEL, _spec(threshold=0.5), limit=5)
    dense.cosines_for(rec, ["n1"], [[1.0, 0.0]], MODEL)
    calibrate.probe_cosines(rec, _spec(), MODEL, [[1.0, 0.0]])
    calibrate.scope_signature(rec, _spec(), MODEL)
    from umrs.knowledge.retrieval import graph_lane

    graph_lane._hop(rec, _spec(), [1.0, 0.0], MODEL, ["a1"])
    backend = kq._EmbeddingsSearchBackend(rec)
    backend.search_embeddings("q", project_id="p1", query_embedding=[1.0, 0.0], model_name=MODEL)
    return [sql for sql in rec.sql if "<=>" in sql]


def test_every_vector_read_carries_a_model_filter():
    reads = _every_vector_read()
    assert len(reads) == 6
    assert [sql[:60] for sql in reads if not has_model_filter(sql)] == []


def test_the_ann_reads_carry_exactly_the_space_index_predicate():
    ann_reads = [sql for sql in _every_vector_read() if "AS ann_vec" in sql]
    assert len(ann_reads) == 2  # the dense lane and the K3 similarity lane
    for sql in ann_reads:
        assert es.space_filter_sql(SPACE) in sql
        assert "ORDER BY (e.embedding)::vector(2) <=> (%s::vector)::vector(2)" in sql


#: Modules whose source holds a ``<=>`` vector read of the knowledge read
#: lanes. A new one must join the fence above (and this list).
FENCED = {
    "knowledge/retrieval/ann.py",
    "knowledge/retrieval/calibrate.py",
    "knowledge/retrieval/dense.py",
    "knowledge/retrieval/graph_lane.py",
}
# api/knowledge_query.py builds its vector read through ann.ann_query (no
# ``<=>`` of its own); the recording-pool fence above runs it.


def test_the_fence_inventory_covers_every_read_lane_module():
    scanned = [*sorted((SRC / "knowledge" / "retrieval").glob("*.py")),
               SRC / "api" / "knowledge_query.py"]
    found = {
        str(path.relative_to(SRC))
        for path in scanned
        if "<=>" in path.read_text(encoding="utf-8")
    }
    assert found == FENCED


def test_the_expression_statistics_migration_exists():
    """Without them the planner never takes the index (see 0150)."""
    sql = (SRC / "migrations" / "0150_knowledge_nodes_expression_stats.sql").read_text()
    assert "ON (content->>'kind') FROM knowledge_nodes" in sql
    assert "ON (content->>'origin') FROM knowledge_nodes" in sql


# ── live Postgres ───────────────────────────────────────────────────────────


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "test")
    password = os.environ.get("POSTGRES_PASSWORD", "test")
    db = os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


live = pytest.mark.skipif(_db_url() is None, reason="CI-shaped Postgres absent")

LIVE_DIM = 8
LIVE = EmbeddingSpace(
    space_id="kns6_live",
    model_name="kns6/live-model",
    dimension=LIVE_DIM,
    legacy_names=("kns6/live-legacy",),
)
SMALL, BIG = "kns6-small", "kns6-big"
SIZES = {SMALL: 60, BIG: 1500}


def _vector(rng: random.Random, dim: int = LIVE_DIM) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def _literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(v) for v in vector) + "]"


def _cleanup(cur: Any) -> None:
    cur.execute("DELETE FROM knowledge_node_embeddings WHERE node_id LIKE 'kns6-%%'")
    cur.execute("DELETE FROM knowledge_nodes WHERE node_id LIKE 'kns6-%%'")
    cur.execute("DELETE FROM knowledge_embedding_spaces WHERE space_id = %s", (LIVE.space_id,))
    cur.execute(es.drop_index_ddl(LIVE, concurrently=False))


_NODE_SQL = (
    "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
    "content_hash, graph_id, visibility) VALUES (%s, %s, %s, 'concept', %s::jsonb, '', '', 'project')"
)
_VEC_SQL = (
    "INSERT INTO knowledge_node_embeddings (node_id, model_name, dimension, embedding) "
    "VALUES (%s, %s, %s, %s::vector)"
)


def _seed_project(cur: Any, rng: random.Random, project: str) -> dict[str, np.ndarray]:
    """Chunks of ``project`` under the space's two names (and a 4-d foreign
    model the cast must never see); returns the in-space vectors by node."""
    seeded: dict[str, np.ndarray] = {}
    for i in range(SIZES[project]):
        node = f"kns6-{project}-{i}"
        content = json.dumps({"kind": "chunk", "chunk_index": 0, "artifact_id": f"{project}-{i}"})
        cur.execute(_NODE_SQL, (node, project, node, content))
        model = LIVE.model_name if i % 4 else LIVE.legacy_names[0]
        vector = _vector(rng)
        cur.execute(_VEC_SQL, (node, model, LIVE_DIM, _literal(vector)))
        cur.execute(_VEC_SQL, (node, "kns6/foreign-model", 4, _literal(_vector(rng, 4))))
        seeded[node] = np.asarray(vector) / np.linalg.norm(vector)
    return seeded


@pytest.fixture(scope="module")
def corpus() -> Iterator[tuple[Any, dict[str, dict[str, np.ndarray]]]]:
    psycopg2 = pytest.importorskip("psycopg2")
    from psycopg2.pool import ThreadedConnectionPool
    from umrs.migrations import apply_migrations

    url = _db_url()
    if url is None:
        pytest.skip("CI-shaped Postgres absent")
    pool = ThreadedConnectionPool(1, 4, url)
    conn = pool.getconn()
    try:
        apply_migrations(conn)
        conn.commit()
        conn.autocommit = True
        cur = conn.cursor()
        _cleanup(cur)
        cur.execute(
            f"INSERT INTO {es.TABLE} ({', '.join(es.COLUMNS)}) VALUES "
            f"({', '.join(['%s'] * len(es.COLUMNS))})",
            es.space_to_row(LIVE),
        )
        rng = random.Random(20261127)
        vectors = {p: _seed_project(cur, rng, p) for p in (SMALL, BIG)}
        cur.execute(es.hnsw_index_ddl(LIVE))
        cur.execute("ANALYZE knowledge_node_embeddings")
        cur.execute("ANALYZE knowledge_nodes")
    except psycopg2.Error as exc:  # pgvector missing or too old: not this test's subject
        pool.putconn(conn)
        pool.closeall()
        pytest.skip(f"live corpus unavailable: {exc}")
    pool.putconn(conn)
    yield pool, vectors
    conn = pool.getconn()
    try:
        conn.autocommit = True
        _cleanup(conn.cursor())
    finally:
        pool.putconn(conn)
        pool.closeall()


def _explain(pool: Any, project: str, vector: list[float]) -> str:
    where, params = dense.chunk_scope_clause(RetrieveSpec(queries=("q",), project_id=project))
    sql, args = ann.ann_query(LIVE, columns=dense._DENSE_COLUMNS, where=where,
                              where_params=params, vector=vector, k=10)
    settings, select = sql.rsplit("; ", 1)
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(settings)
            cur.execute("EXPLAIN " + select, args)
            plan = "\n".join(row[0] for row in cur.fetchall())
        conn.rollback()
    finally:
        pool.putconn(conn)
    return plan


@live
def test_explain_of_the_dense_read_uses_the_space_index(corpus):
    pool, _vectors = corpus
    plan = _explain(pool, BIG, _vector(random.Random(1)))
    assert f"Index Scan using {LIVE.index_name}" in plan, plan


def _exact_top(vectors: dict[str, np.ndarray], query: list[float], k: int) -> list[str]:
    q = np.asarray(query) / np.linalg.norm(query)
    ranked = sorted(vectors, key=lambda node: (1.0 - float(vectors[node] @ q), node))
    return ranked[:k]


@live
@pytest.mark.parametrize("project", [SMALL, BIG])
def test_the_dense_top_k_equals_the_exact_top_k(corpus, project):
    pool, vectors = corpus
    rng = random.Random(hash(project) % 1000)
    for _ in range(5):
        query = _vector(rng)
        spec = RetrieveSpec(queries=("q",), project_id=project)
        lane = dense.dense_lane(pool, [query], LIVE.model_name, spec, limit=10)
        assert lane.status == "ok", lane.reason
        assert [c.node_id for c in lane.candidates] == _exact_top(vectors[project], query, 10)
        assert lane.info["in_scope"] == SIZES[project]


@live
def test_a_legacy_name_resolves_to_the_same_space(corpus):
    pool, _vectors = corpus
    spec = RetrieveSpec(queries=("q",), project_id=SMALL)
    lane = dense.dense_lane(pool, [_vector(random.Random(2))], "kns6/live-legacy", spec, limit=5)
    assert (lane.status, lane.info["space_id"]) == ("ok", LIVE.space_id)


@live
def test_an_unregistered_model_reads_nothing(corpus):
    pool, _vectors = corpus
    spec = RetrieveSpec(queries=("q",), project_id=SMALL)
    lane = dense.dense_lane(pool, [[1.0] * 4], "kns6/foreign-model", spec, limit=5)
    assert lane.status == "unavailable"
    assert "knowledge.embedding_space_unregistered" in lane.reason


class _CountingEncoder:
    """A deterministic 8-d encoder that counts full probe batches."""

    def __init__(self) -> None:
        self.batches = 0

    def __call__(self, texts: list[str]) -> tuple[str, list[list[float]], str]:
        if len(texts) == len(calibrate.NULL_PROBES_V1):
            self.batches += 1
        return LIVE.model_name, [_vector(random.Random(t)) for t in texts], ""


@live
def test_the_null_is_recomputed_after_a_reindex(corpus):
    pool, _vectors = corpus
    calibrate.clear_cache()
    encode = _CountingEncoder()
    spec = RetrieveSpec(queries=("q",), project_id=SMALL)
    first = calibrate.null_for(pool, spec, encode, LIVE.model_name)
    again = calibrate.null_for(pool, spec, encode, LIVE.model_name)
    assert first is not None and again == first and encode.batches == 1
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:  # a reindex re-upserts: updated_at moves
            cur.execute(
                "UPDATE knowledge_node_embeddings SET updated_at = now() + interval '1 second' "
                "WHERE node_id = %s AND model_name = %s",
                (f"kns6-{SMALL}-1", LIVE.model_name),
            )
        conn.commit()
    finally:
        pool.putconn(conn)
    calibrate.null_for(pool, spec, encode, LIVE.model_name)
    assert encode.batches == 2


@live
@pending("kn-s6-01", es.index_actions, es.index_signature, es.comment_index_sql)
@pending("kn-s6-02", esi.read_index_states, esi.plan_index_work, esi.apply_index_work)
def test_the_index_runner_converges_on_a_live_database(corpus):
    pool, _vectors = corpus
    conn = pool.getconn()
    try:
        conn.autocommit = True
        states = esi.read_index_states(conn, [LIVE])
        assert states[LIVE.space_id].exists  # built by the fixture, no comment yet
        esi.apply_index_work(conn, esi.plan_index_work([LIVE], states))
        settled = esi.read_index_states(conn, [LIVE])
        assert settled[LIVE.space_id].comment == es.index_signature(LIVE)
        assert esi.plan_index_work([LIVE], settled) == []
    finally:
        conn.autocommit = False
        pool.putconn(conn)
