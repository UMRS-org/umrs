"""KN-S6 review: the read filter covers every name the writer admits.

The embed writer (``embedding_space_guard.require_writer_space``) admits a
model whose path only BASENAME-matches a bare registered name (the
bundled space answers to ``e5-base-v2``, so ``/srv/models/e5-base-v2`` is
admitted) and stores its vectors under that path. Before KN-S6 the vector
reads ranked such rows (``model_name = m OR model_name LIKE '%/<base>'``);
the KN-S6 ANN read filters on the registry names only, so a self-hosted
install whose encoder lives at another path ranked nothing: the lane said
``ok`` with zero candidates. The read must also cover the query model's own
name (an exact scan until the operator registers it as a legacy name and
rebuilds the index, which ``admin embedding-space index`` warns about).
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from typing import Any

import pytest
from umrs.knowledge import embedding_spaces as es
from umrs.knowledge.embedding_spaces import EmbeddingSpace
from umrs.knowledge.retrieval import ann, dense
from umrs.knowledge.retrieval.types import RetrieveSpec

DIM = 4
BARE = EmbeddingSpace(space_id="kns6_bare", model_name="kns6-bare-model", dimension=DIM)
STAGED = "/srv/models/kns6-bare-model"
PROJECT = "kns6-bare-project"
ROWS = 12


class _RegistryPool:
    """A psycopg2-shaped pool whose only read is the registry."""

    def getconn(self) -> _RegistryPool:
        return self

    def putconn(self, _conn: Any) -> None:
        return None

    def __enter__(self) -> _RegistryPool:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def cursor(self) -> _RegistryPool:
        return self

    def execute(self, _sql: str, _params: Any = None) -> None:
        return None

    def fetchall(self) -> list[tuple]:
        return [es.space_to_row(BARE)]


def test_the_writer_admits_a_basename_match_of_a_bare_name():
    assert es.require_space(STAGED, [BARE]) == BARE


def test_the_query_space_filter_covers_the_query_models_own_path():
    space = ann.query_space(_RegistryPool(), STAGED)
    assert space.space_id == BARE.space_id
    assert es.sql_literal(STAGED) in es.space_filter_sql(space)
    assert es.sql_literal(BARE.model_name) in es.space_filter_sql(space)


def test_a_covered_query_model_keeps_the_index_predicate_exactly():
    space = ann.query_space(_RegistryPool(), BARE.model_name)
    assert es.space_filter_sql(space) == es.space_filter_sql(BARE)


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


_NODE_SQL = (
    "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
    "content_hash, graph_id, visibility) VALUES (%s, %s, %s, 'concept', %s::jsonb, '', '', 'project')"
)
_VEC_SQL = (
    "INSERT INTO knowledge_node_embeddings (node_id, model_name, dimension, embedding) "
    "VALUES (%s, %s, %s, %s::vector)"
)


def _cleanup(cur: Any) -> None:
    cur.execute("DELETE FROM knowledge_node_embeddings WHERE node_id LIKE 'kns6bare-%%'")
    cur.execute("DELETE FROM knowledge_nodes WHERE node_id LIKE 'kns6bare-%%'")
    cur.execute(f"DELETE FROM {es.TABLE} WHERE space_id = %s", (BARE.space_id,))


def _seed(cur: Any) -> None:
    rng = random.Random(20261127)
    cur.execute(
        f"INSERT INTO {es.TABLE} ({', '.join(es.COLUMNS)}) VALUES "
        f"({', '.join(['%s'] * len(es.COLUMNS))})",
        es.space_to_row(BARE),
    )
    for i in range(ROWS):
        node = f"kns6bare-{i}"
        content = json.dumps({"kind": "chunk", "chunk_index": 0, "artifact_id": node})
        cur.execute(_NODE_SQL, (node, PROJECT, node, content))
        vector = "[" + ",".join(repr(rng.gauss(0.0, 1.0)) for _ in range(DIM)) + "]"
        cur.execute(_VEC_SQL, (node, STAGED, DIM, vector))


@pytest.fixture(scope="module")
def staged_pool() -> Iterator[Any]:
    psycopg2 = pytest.importorskip("psycopg2")
    from psycopg2.pool import ThreadedConnectionPool
    from umrs.migrations import apply_migrations

    url = _db_url()
    if url is None:
        pytest.skip("CI-shaped Postgres absent")
    pool = ThreadedConnectionPool(1, 2, url)
    conn = pool.getconn()
    try:
        apply_migrations(conn)
        conn.commit()
        conn.autocommit = True
        _cleanup(conn.cursor())
        _seed(conn.cursor())
    except psycopg2.Error as exc:
        pool.putconn(conn)
        pool.closeall()
        pytest.skip(f"live corpus unavailable: {exc}")
    pool.putconn(conn)
    yield pool
    conn = pool.getconn()
    try:
        _cleanup(conn.cursor())
    finally:
        pool.putconn(conn)
        pool.closeall()


@pytest.mark.skipif(_db_url() is None, reason="CI-shaped Postgres absent")
def test_vectors_stored_under_a_staged_path_are_ranked(staged_pool):
    spec = RetrieveSpec(queries=("q",), project_id=PROJECT)
    lane = dense.dense_lane(staged_pool, [[1.0, 0.5, -0.5, 0.25]], STAGED, spec, limit=5)
    assert lane.status == "ok", lane.reason
    assert lane.info["in_scope"] == ROWS
    assert len(lane.candidates) == 5
